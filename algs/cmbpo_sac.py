import torch
import torch.nn.functional as F
import numpy as np

from algs.sac import SAC, ReplayBuffer
from dynamics.causal_models import StructureLearning, set_p_matrix
from dynamics.utils import compute_jsd, compute_path_ce, linear_scheduler
from dynamics.causal_dynamics_models import FactorizedEnsembleModel
import matplotlib.pyplot as plt

import gymnasium as gym

import wandb
import time
import random

torch.set_default_dtype(torch.float32)

# Check if MPS is available and set the device to 'mps' if on MacOS, 'cuda' if on GPU, or 'cpu' otherwise
def set_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


class CMBPO_SAC:
    def __init__(self,
                 env: gym.Env,
                 seed: int = 0,
                 dev: torch.device = None,
                 log_wandb: bool = False,
                 model_based: bool = True,
                 sl_method: str = 'PC',
                 bootstrap: str = 'standard',
                 uncer_cgm_model: str = 'ensemble_sampling',
                 n_bootstrap: int = 10,
                 cgm_train_freq: int = 5_000,
                 warmup_steps: int = 1_000,
                 eval_freq: int = 1_000,
                 causal_bonus: bool = True,
                 causal_eta: float = 0.01,
                 var_causal_bonus: bool = False,
                 var_causal_eta: float = 0.001,
                 jsd_bonus: float = False,
                 jsd_eta: float = 0.01,
                 jsd_thres: float = 1.0,
                 lr_model: float = 1e-3,
                 lr_sac: float = 0.0003,
                 agent_steps: int = 20,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 max_rollout_len: int = 1,
                 rollout_schedule: list = None,  # Schedule rollout length - for Half-Cheetah = [20_000, 100_000, 1, 15]
                 rollout_per_step: int = 400,  # Maybe put 100_000 as it is batched anyway
                 update_size: int = 256,
                 sac_train_freq: int = 1,
                 model_train_freq: int = 250,
                 batch_size: int = 256):

        self.env = env
        self.seed = seed
        self.device = dev
        self.log_wandb = log_wandb
        self.model_based = model_based
        if model_based:
            self.alg_name = f"CMBPO_SAC_{sl_method}_boot{str(bootstrap)}_ce{str(causal_bonus)}_varce{str(var_causal_bonus)}"
        else:
            self.alg_name = f"C_SAC_{sl_method}_boot{str(bootstrap)}_ce{str(causal_bonus)}_varce{str(var_causal_bonus)}"

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.max_rollout_len = max_rollout_len
        self.num_model_rollouts = int(rollout_per_step * model_train_freq)  # Number of rollouts per training step 400 * 250 = 100_000

        self.update_size = update_size  # Size of the final buffer to train the SAC agent made of %5-95% real-imaginary

        # The agent steps are 1 for SAC and agent_steps for MBPO
        self.total_steps, self.max_steps = 0, 0
        self.warmup_steps, self.eval_freq = warmup_steps, eval_freq
        self.agent_steps = agent_steps if self.model_based else 1
        self.sac_train_freq = sac_train_freq
        self.model_train_freq = model_train_freq
        self.batch_size = batch_size

        self.sac_agent = SAC(self.state_dim, self.action_dim, self.max_action, lr=lr_sac, gamma=gamma,
                             tau=tau, alpha=alpha, device=self.device)

        self.ensemble_model = FactorizedEnsembleModel(state_dim=self.state_dim, action_dim=self.action_dim,
                                                      device=self.device, ensemble_size=7, lr=lr_model).to(self.device)
        self.rollout_schedule = rollout_schedule
        self.real_buffer = ReplayBuffer(int(1_000_000))
        self.imaginary_buffer = ReplayBuffer(int(1_000_000))

        self.jsd_thres, self.jsd_bonus, self.jsd_eta = jsd_thres, jsd_bonus, jsd_eta

        # Causal MBPO specific
        self.sl_method = sl_method
        self.uncer_cgm_model = uncer_cgm_model
        self.bootstrap = bootstrap
        self.n_bootstrap = n_bootstrap
        self.cgm_train_freq = cgm_train_freq
        self.local_cgm = StructureLearning(n_nodes=self.state_dim + self.action_dim + self.state_dim + 1,
                                           sl_method=sl_method, bootstrap=bootstrap)
        self.p_matrix = set_p_matrix(self.state_dim, self.action_dim)

        self.pk = self.local_cgm.set_prior_knowledge(p_matrix=self.p_matrix)

        # Initialize the estimated CGM as a fully connected graph
        self.est_cgm = np.ones((self.state_dim + self.action_dim + self.state_dim + 1,
                                self.state_dim + self.action_dim + self.state_dim + 1))
        self.est_cgm = torch.FloatTensor(self.est_cgm).to(self.device)

        self.true_cgm = self.env.get_adj_matrix() if hasattr(self.env, 'get_adj_matrix') else None

        self.causal_bonus, self.causal_eta = causal_bonus, causal_eta
        self.var_causal_bonus, self.var_causal_eta = var_causal_bonus, var_causal_eta

    def update_model(self, batch_size=256, epochs=100):
        """
        Updates the ensemble dynamics model using a batch of data from the real buffer.
        The model is trained to predict the next state and reward given the current state and action.
        """
        model_loss = self.ensemble_model.train_factorized_ensemble(buffer=self.real_buffer,
                                                             batch_size=batch_size, epochs=epochs)

        return model_loss


    def counterfact_rollout(self):
        """
        Rolls out from real states using the learned model. The length of each rollout
        is dynamically adjusted based on ensemble disagreement/uncertainty.

        Idea:
        - We keep rolling out up to 'self.max_rollout_len' steps.
        - At each step, compute the standard deviation (or variance) across
          the ensemble for the *next state*. If it exceeds some threshold,
          we stop rolling out that particular state.
        """

        # Augment max_length_traj by 1 every 10_000 steps
        max_length_traj = linear_scheduler(self.rollout_schedule, self.total_steps)
        num_samples = int(self.num_model_rollouts // max_length_traj)  # keeps the seed fixed to the number of rollouts
        initial_states, _, _, _, _ = self.real_buffer.sample(num_samples, replace=True)
        initial_states = torch.FloatTensor(initial_states).to(self.device)

        # "active_mask[i] = False" => stop rolling out sample i
        active_mask = torch.ones(num_samples, dtype=torch.bool, device=self.device)
        jsd_threshold = 1.0  # JSD threshold for uncertainty (0.5 is too low, 1.0 is more reasonable)

        for t in range(max_length_traj):

            # If everything is "inactive", exit early
            if not active_mask.any():
                break

            actions = self.sac_agent.select_action(initial_states)
            actions = torch.FloatTensor(actions).to(self.device)

            # TODO: Change for Factored Dynamics Model
            # Ensemble predictions: shape [ensemble_size, batch_size, next_state_dim+1]
            model_input = torch.cat([initial_states, actions], dim=1)
            model_input = self.ensemble_model.input_normalizer.normalize(model_input)
            with torch.no_grad():
                mean_preds, logvar_preds = self.ensemble_model(model_input)
            all_preds_mean, all_preds_logvar = torch.stack(mean_preds, dim=0), torch.stack(logvar_preds, dim=0)

            # Next state is sampled from the ensemble
            head_idx = torch.randint(self.ensemble_model.ensemble_size, (num_samples,), device=self.device)
            mean_pred = all_preds_mean[head_idx, torch.arange(num_samples)]
            logvar_pred = all_preds_logvar[head_idx, torch.arange(num_samples)]

            # Add model noise
            std_pred = torch.exp(0.5 * logvar_pred).clamp(max=0.25)  # σ ≤ 0.25 (normalised)
            noise = torch.randn_like(mean_pred) * std_pred
            mean_pred = mean_pred + noise

            # Pick normalized rewards
            reward_pred = mean_pred[:, -1].clone().unsqueeze(1)
            reward_original = reward_pred.clone()  # For logging

            # Compute uncertainty as disagreement across ensemble (JSD) - can use it also as intrinsic reward
            ns_jsd = compute_jsd(all_preds_mean, torch.exp(all_preds_logvar))

            # REWARDS AUGMENTATION EXPLORATION AND CAUSAL EMPOWERMENT
            if self.jsd_bonus:

                # Scale JSD by running std of the reward channel for numerical stability
                reward_std = self.ensemble_model.output_normalizer.var[-1].detach().clamp_min(1e-6).sqrt()
                de_meaned_jsd = ns_jsd - ns_jsd.mean(dim=0)
                norm_jsd = (de_meaned_jsd.sum(dim=1, keepdim=True) / (reward_std + 1e-6))
                reward_pred += self.jsd_eta * norm_jsd

            if self.causal_bonus:

                # TODO: Check if causal empowerment for only parents of reward is correctly implemented
                # causal_empow = compute_causal_emp(self.ensemble_model, current_states, self.sac_agent)
                causal_empow = compute_path_ce(self.est_cgm, self.ensemble_model, initial_states, self.sac_agent)
                causal_empow_bonus = causal_empow.mean(dim=0)  # shape: (n_batch, sts_dim)
                std_causal_empow_bonus = causal_empow.std(dim=0)  # shape: (n_batch, sts_dim)

                # Scale CE by running std of the reward channel for numerical stability
                reward_std = self.ensemble_model.output_normalizer.var[-1].detach().clamp_min(1e-6).sqrt()
                de_meaned_ce = causal_empow_bonus - causal_empow_bonus.mean(dim=0)

                norm_ce = (de_meaned_ce.sum(dim=1, keepdim=True) / (reward_std + 1e-6))

                causal_bonus_tot = self.causal_eta * norm_ce
                reward_pred += causal_bonus_tot

                if self.var_causal_bonus:
                    std_bonus = (std_causal_empow_bonus.sum(dim=1, keepdim=True) / (reward_std + 1e-6))
                    reward_pred += self.var_causal_eta * std_bonus

            # Replace the reward prediction with the augmented one
            mean_pred[:, -1] = reward_pred.squeeze(1)

            # Denormalize the outputs
            denorm_mean_pred = self.ensemble_model.output_normalizer.denormalize(mean_pred)
            delta_next_state = denorm_mean_pred[:, :-1]
            next_states = initial_states + delta_next_state

            rewards = denorm_mean_pred[:, -1].unsqueeze(1)
            dones = torch.zeros_like(rewards)

            # Compute the mask and update for samples that are still active but exceed the threshold
            push_mask = active_mask & (ns_jsd <= jsd_threshold)
            active_mask[active_mask & (ns_jsd > jsd_threshold)] = False

            # Check if any sample should be pushed in this rollout step
            if push_mask.any():
                # Get the indices of the samples to push
                indices_to_push = push_mask.nonzero(as_tuple=True)[0]

                states_to_push = initial_states[indices_to_push].cpu().numpy()
                actions_to_push = actions[indices_to_push].cpu().numpy()
                rewards_to_push = rewards[indices_to_push].detach().cpu().numpy()
                next_states_to_push = next_states[indices_to_push].detach().cpu().numpy()
                dones_to_push = dones[indices_to_push].cpu().numpy()

                # Push the samples to the imaginary buffer
                self.imaginary_buffer.push_batch(
                    states_to_push,
                    actions_to_push,
                    rewards_to_push,
                    next_states_to_push,
                    dones_to_push
                )

            initial_states = next_states.detach()
        # End of imaginary rollouts

        # If wandb is enabled, log the normal rewards and causal rewards
        if self.log_wandb:
            wandb.log({
                "Train/Normal Rewards": reward_original.mean().item(),
                "Train/Causal Rewards": norm_ce.mean().item() if self.causal_bonus else 0,
                "Train/Uncertainty": ns_jsd.mean().item()
            })

    def update_cgm(self):

        # Learn the CGM with the last self.cgm_train_freq samples
        last_cgm_data = self.real_buffer.buffer[-self.cgm_train_freq:]
        state, action, reward, next_state, _ = map(np.stack, zip(*last_cgm_data))
        reward = reward.reshape(-1, 1)

        data_cgm = np.concatenate([state, action, next_state, reward], axis=1)
        self.est_cgm = self.local_cgm.learn_dag(data_cgm,
                                                prior_knowledge=self.pk,
                                                n_bootstrap=self.n_bootstrap)

        # Update the dynamics model with structural uncertainty
        self.ensemble_model.set_causal_structure_with_uncertainty(
            adjacency_probs=torch.FloatTensor(self.est_cgm).to(self.device),
            uncertainty_method=self.uncer_cgm_model
        )

        # Plot DAG at the end of the training
        if self.log_wandb and len(self.real_buffer) % (self.cgm_train_freq + 1) == 0:
            fig = self.local_cgm.plot_dag(self.est_cgm, self.true_cgm)
            wandb.log({"Train/Estimated CGM": wandb.Image(fig)})
            plt.close(fig)

    def get_final_buffer(self, proportion_real=0.05):

        # Function that creates a new ReplayBuffer with the data from the real buffer and imaginary buffer.
        if self.model_based:

            # Sample 5% of the real buffer
            real_batch = random.sample(self.real_buffer.buffer, int(proportion_real * self.update_size))

            # Sample 95% of the imaginary buffer
            imaginary_size = min(int((1 - proportion_real) * self.update_size), len(self.imaginary_buffer))
            imaginary_batch = random.sample(self.imaginary_buffer.buffer, imaginary_size)

            # Concatenate the two batches
            final_batch = real_batch + imaginary_batch
            s, a, r, ns, d = map(np.stack, zip(*final_batch))

        else:

            # Sample all the real buffer
            real_batch = random.sample(self.real_buffer.buffer, self.update_size)
            s, a, r, ns, d = map(np.stack, zip(*real_batch))

        return torch.as_tensor(s, device=self.device, dtype=torch.float32), \
            torch.as_tensor(a, device=self.device, dtype=torch.float32), \
            torch.as_tensor(r, device=self.device, dtype=torch.float32).unsqueeze(-1), \
            torch.as_tensor(ns, device=self.device, dtype=torch.float32), \
            torch.as_tensor(d, device=self.device, dtype=torch.float32).unsqueeze(-1)

    def train(self, num_episodes=200, max_steps=1_000):
        if self.log_wandb:
            project_name = self.env.unwrapped.spec.id if self.env.unwrapped.spec != None else 'SimpleCausal_Multi'
            wandb.init(project=project_name, sync_tensorboard=False,
                       name=f"{self.alg_name}_SAC_seed_{self.seed}_time_{time.time()}",
                       config=self.__dict__, group=self.alg_name, dir='/tmp')

        episode = 0
        self.max_steps = max_steps
        target_steps = num_episodes * max_steps

        while self.total_steps < target_steps:
            state, _ = self.env.reset()
            episode_reward, episode_steps = 0, 0

            # 1) First chunk: roll an episode within the real environment and populate the real buffer
            for step in range(max_steps):

                if self.total_steps > self.warmup_steps:
                    action = self.sac_agent.select_action(state).flatten()
                else:
                    action = self.env.action_space.sample()

                next_state, reward, done, truncated, _ = self.env.step(action)
                terminal = done or truncated

                self.real_buffer.push(state, action, reward, next_state, terminal)

                # Set state to next_state and increment the episode reward and steps
                state = next_state
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1

                # 2) Every self.eval_freq steps, evaluate the policy on deterministic actions
                if self.total_steps % self.eval_freq == 0 and self.total_steps > 0:
                    self._evaluate_policy()

                # 3) Update normalizers every self.model_train_freq steps in model-based training
                if self.total_steps % self.model_train_freq == 0 and self.model_based:

                    batch = self.real_buffer.buffer[-self.model_train_freq:]
                    states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))

                    states = torch.FloatTensor(states).to(self.device)
                    actions = torch.FloatTensor(actions).to(self.device)
                    next_states = torch.FloatTensor(next_states).to(self.device)
                    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)

                    # Compute delta_state
                    delta_states = next_states - states

                    # Concatenate inputs and outputs
                    model_inputs = torch.cat([states, actions], dim=-1)
                    targets = torch.cat([delta_states, rewards], dim=-1)

                    # Normalize training data
                    self.ensemble_model.input_normalizer.update(model_inputs)
                    self.ensemble_model.output_normalizer.update(targets)

                    # Train the model
                    if len(self.real_buffer) > (self.batch_size):
                        # Train the model
                        model_loss = self.update_model(self.batch_size)

                # All this after the warm-up period of 5_000 steps used to populate the real buffer
                if self.total_steps >= self.warmup_steps:

                    # 3) Train the dynamics model and populate the imaginary buffer if self.model_based
                    if self.total_steps % self.model_train_freq == 0 and self.model_based:
                        self.counterfact_rollout()  # Generate imaginary rollouts

                    # 4) Learn Local Causal Graphical Model from the real buffer
                    if self.total_steps % (self.cgm_train_freq + 1) == 0 and self.model_based:
                    # if total_steps == (self.cgm_train_freq + 1) and self.model_based and self.causal_bonus:
                        self.update_cgm()

                    # 5) Train the SAC agent
                    if self.total_steps % self.sac_train_freq == 0 and len(self.real_buffer) > self.batch_size:
                        for _ in range(self.agent_steps):
                            s, a, r, ns, d = self.get_final_buffer()
                            critic_loss, actor_loss, alpha_loss = self.sac_agent.update(s, a, r, ns, d)

            # 6) Logging and Printing
            if self.log_wandb:
                wandb.log({
                    "Train/Episode Reward": episode_reward,
                    "Train/Episode Length": episode_steps,
                    "Train/Global Step": self.total_steps,
                    "Train/Model Loss": model_loss if 'model_loss' in locals() else 0,
                    "Train/Critic Loss": critic_loss if 'critic_loss' in locals() else 0,
                    "Train/Actor Loss": actor_loss if 'actor_loss' in locals() else 0,
                    "Train/Alpha Loss": alpha_loss if 'alpha_loss' in locals() else 0
                })

            if episode % 1 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Steps: {episode_steps}")
                print("Model Loss: ", model_loss if 'model_loss' in locals() else 0)

        wandb.finish()

    def save_agent(self, base_dir='trained_agents/'):

        # Save the entire SAC agent for later use
        filename = base_dir + f"{self.alg_name}_seed_{self.seed}"
        torch.save(self.sac_agent, filename)

    @torch.no_grad()
    def _evaluate_policy(self, eval_episodes: int = 10) -> float:

        self.sac_agent.actor.eval()

        avg_return = 0.0
        for _ in range(10):  # Eval episodes

            state, _ = self.env.reset()
            ep_ret = 0.0
            for _ in range(self.max_steps):  # Max steps per episode
                if self.total_steps > self.warmup_steps:
                    action = self.sac_agent.select_action(state, deterministic=True).flatten()
                else:
                    action = self.env.action_space.sample()

                state, r, d, t, _ = self.env.step(action)

                ep_ret += r
                if d or t:
                    break

            avg_return += ep_ret
        avg_return /= eval_episodes

        # restore the actor to training mode
        self.sac_agent.actor.train()

        # Log the average return
        if self.log_wandb:
            wandb.log({"Eval/Average Return": avg_return,
                       "Eval/Global Step": self.total_steps})
        print(f"Eval Average Return: {avg_return:.2f}")

        return avg_return
