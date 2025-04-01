import torch
import torch.nn.functional as F
import numpy as np

from algs.sac import SAC, ReplayBuffer
from dynamics.causal_models import StructureLearning, set_p_matrix
from dynamics.utils import compute_jsd, compute_causal_emp
from dynamics.causal_dynamics_models import FactorizedEnsembleModel
import matplotlib.pyplot as plt

import wandb
import time
import random

torch.set_default_dtype(torch.float32)


class CMBPO_SAC:
    def __init__(self, env, seed, dev, log_wandb=True, model_based=False, pure_imaginary=True,
                 sl_method="PC", bootstrap=None, n_bootstrap=10, cgm_train_freq=1_000,
                 causal_bonus=False, causal_eta=0.01, var_causal_bonus=False, var_causal_eta=0.01,
                 jsd_bonus=False, jsd_eta=0.1, jsd_thres=1.0,
                 lr_model=1e-3,
                 lr_sac=0.0003, agent_steps=1, gamma=0.99, tau=0.005, alpha=0.2, max_rollout_len=15,
                 num_model_rollouts=400,  # Maybe put 100_000 as it is batched anyway
                 update_size=250, sac_train_freq=1, model_train_freq=100, batch_size=250):

        self.env = env
        self.seed = seed
        self.log_wandb = log_wandb
        self.model_based = model_based
        self.pure_imaginary = pure_imaginary
        self.alg_name = f"CMBPO_SAC_{sl_method}_boot{str(bootstrap)}_ce{str(causal_bonus)}_varce{str(var_causal_bonus)}"

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.max_rollout_len = max_rollout_len
        self.num_model_rollouts = num_model_rollouts

        self.update_size = update_size  # Size of the final buffer to train the SAC agent made of %5-95% real-imaginary

        self.agent_steps = agent_steps
        self.sac_train_freq = sac_train_freq
        self.model_train_freq = model_train_freq

        self.batch_size = batch_size

        self.device = dev

        self.sac_agent = SAC(self.state_dim, self.action_dim, self.max_action, lr=lr_sac, gamma=gamma,
                             tau=tau, alpha=alpha, device=self.device)

        self.ensemble_model = FactorizedEnsembleModel(state_dim=self.state_dim, action_dim=self.action_dim, lr=lr_model,
                                                      ensemble_size=10).to(self.device)

        self.real_buffer = ReplayBuffer(int(10_000))
        self.imaginary_buffer = ReplayBuffer(int(10_000))

        self.jsd_thres, self.jsd_bonus, self.jsd_eta = jsd_thres, jsd_bonus, jsd_eta

        # Causal MBPO specific
        self.sl_method = sl_method
        self.bootstrap = bootstrap
        self.n_bootstrap = n_bootstrap
        self.cgm_train_freq = cgm_train_freq
        self.local_cgm = StructureLearning(n_nodes=self.state_dim + self.action_dim + self.state_dim + 1,
                                           sl_method=sl_method, bootstrap=bootstrap)
        self.p_matrix = set_p_matrix(self.state_dim, self.action_dim)

        # # In SimpleCausalEnv, allow to learn same time state depencencies
        # if self.env.unwrapped.spec is None:
        #     self.p_matrix[:self.state_dim, :self.state_dim] = -1
        #     # self.p_matrix[self.state_dim + self.action_dim:, self.state_dim + self.action_dim:] = -1  # Also for S'?

        self.pk = self.local_cgm.set_prior_knowledge(p_matrix=self.p_matrix)

        # Initialize the estimated CGM as a fully connected graph
        self.est_cgm = np.ones((self.state_dim + self.action_dim + self.state_dim + 1,
                                self.state_dim + self.action_dim + self.state_dim + 1))
        self.true_cgm = self.env.get_adj_matrix() if hasattr(self.env, 'get_adj_matrix') else None

        self.causal_bonus, self.causal_eta = causal_bonus, causal_eta
        self.var_causal_bonus, self.var_causal_eta = var_causal_bonus, var_causal_eta

    def update_model(self, batch_size=256, epochs=50):

        model_loss = self.ensemble_model.train_factorized_ensemble(self.real_buffer, batch_size, epochs)

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

        if len(self.real_buffer) < 2_000:
            max_length_traj = 1
        else:
            max_length_traj = self.max_rollout_len

        # Number of rollouts to generate from the real buffer
        num_samples = min(self.num_model_rollouts, len(self.real_buffer))
        if num_samples == 0:
            return

        initial_states, _, _, _, _ = self.real_buffer.sample(num_samples)
        initial_states = torch.FloatTensor(initial_states).to(self.device)

        # Whether we should continue for each sample in the batch
        # "active_mask[i] = False" => stop rolling out sample i
        active_mask = torch.ones(num_samples, dtype=torch.bool, device=self.device)

        # Let us define a hyperparameter or heuristic threshold for "too high" variance
        # In practice, you can tune this threshold or make it adapt over time.
        jsd_threshold = 1.0  # or any other measure you want

        current_states = initial_states.clone().numpy()

        for t in range(max_length_traj):

            # If everything is "inactive", exit early
            if not active_mask.any():
                break

            actions = self.sac_agent.select_action(current_states)
            actions = torch.FloatTensor(actions).to(self.device)

            model_input = torch.cat([initial_states, actions], dim=1)

            # Ensemble predictions: shape [ensemble_size, batch_size, next_state_dim+1]
            with torch.no_grad():
                mean_preds, logvar_preds = self.ensemble_model(model_input)
            all_preds_mean = torch.stack([torch.stack(group, dim=0).squeeze(-1) for group in mean_preds], dim=0)
            all_preds_var = torch.stack([torch.stack(group, dim=0).squeeze(-1) for group in logvar_preds], dim=0)

            # Swap axes to [n_ens, n_batch, n_dim]
            all_preds_mean = all_preds_mean.permute(1, 2, 0)
            all_preds_var = all_preds_var.permute(1, 2, 0)

            # Next state is sampled from the ensemble
            # TODO:  NB, Here we can also pick a single ensemble member - graph-based
            ensemble_idx = torch.randint(self.ensemble_model.ensemble_size, (1,)).item()
            mean_pred, log_var_pred = all_preds_mean[ensemble_idx], all_preds_var[ensemble_idx]
            next_states = mean_pred[:, :-1]
            rewards = mean_pred[:, -1].unsqueeze(1)
            dones = torch.zeros_like(rewards)

            # Compute uncertainty as disagreement across ensemble (JSD) - can use it also as intrinsic reward
            ns_jsd = compute_jsd(all_preds_mean, torch.exp(all_preds_var))

            if self.jsd_bonus:
                rewards += self.jsd_eta * ns_jsd

            if self.causal_bonus:

                # TODO: Here the mixture still creates a bit of computational bottleneck
                causal_empow = compute_causal_emp(self.ensemble_model, current_states, self.sac_agent)
                causal_empow_bonus = causal_empow.mean(dim=0)  # shape: (n_batch)
                std_causal_empow_bonus = causal_empow.std(dim=0)  # shape: (n_batch)

                rewards += self.causal_eta * causal_empow_bonus.unsqueeze(1)

                if self.var_causal_bonus:
                    rewards += self.var_causal_eta * std_causal_empow_bonus.unsqueeze(1)

            # We only continue rolling out for samples with active_mask == True
            # If the uncertainty is above threshold, we turn off that sample
            for i in range(num_samples):
                if active_mask[i]:
                    if ns_jsd[i].item() > jsd_threshold:
                        active_mask[i] = False
                    else:
                        self.imaginary_buffer.push(
                            initial_states[i].cpu().numpy(),
                            actions[i].cpu().numpy(),
                            rewards[i].item(),
                            next_states[i].cpu().numpy(),
                            dones[i].item()
                        )

            current_states = next_states.detach()

        # End of imaginary rollouts

    def update_cgm(self):

        # Learn the CGM with the last self.cgm_train_freq samples
        last_cgm_data = self.real_buffer.buffer[-self.cgm_train_freq:]
        state, action, reward, next_state, _ = map(np.stack, zip(*last_cgm_data))
        reward = reward.reshape(-1, 1)

        data_cgm = np.concatenate([state, action, next_state, reward], axis=1)
        self.est_cgm = self.local_cgm.learn_dag(data_cgm,
                                                # prior_knowledge=self.pk,
                                                n_bootstrap=self.n_bootstrap)

        # Set parents of (s, a) to (ns, r) to -1
        self.ensemble_model.set_parents_from_prob_matrix(self.est_cgm)

        # Plot DAG at the end of the training
        if self.log_wandb and len(self.real_buffer) % 1_000 == 0:
            fig = self.local_cgm.plot_dag(self.est_cgm, self.true_cgm)
            wandb.log({"Train/Estimated CGM": wandb.Image(fig)})
            plt.close(fig)

    def get_final_buffer(self):

        if len(self.imaginary_buffer) == 0:
            proportion_real = 0.05
        else:
            proportion_real = 0.00

        # Function that creates a new ReplayBuffer with the data from the real buffer and imaginary buffer.
        if self.model_based:

            # Sample 5% of the real buffer
            real_batch = random.sample(self.real_buffer.buffer, int(proportion_real * self.update_size))

            # Sample 95% of the imaginary buffer
            imaginary_size = min(int((1 - proportion_real) * self.update_size), len(self.imaginary_buffer))
            imaginary_batch = random.sample(self.imaginary_buffer.buffer, imaginary_size)

            # Concatenate the two batches
            final_batch = real_batch + imaginary_batch

            # Put in the ReplayBuffer format
            final_buffer = ReplayBuffer(self.update_size + 1)
            final_buffer.buffer = final_batch
            final_buffer.position = len(final_buffer.buffer)

        else:

            # Sample 5% of the real buffer
            real_batch = random.sample(self.real_buffer.buffer, self.update_size)

            # Put in the ReplayBuffer format
            final_buffer = ReplayBuffer(self.update_size + 1)
            final_buffer.buffer = real_batch
            final_buffer.position = len(final_buffer.buffer)

        return final_buffer

    def train(self, num_episodes=100, max_steps=200):
        if self.log_wandb:
            project_name = self.env.unwrapped.spec.id if self.env.unwrapped.spec != None else 'SimpleCausalEnv'
            wandb.init(project=project_name, sync_tensorboard=False,
                       name=f"{self.alg_name}_SAC_seed_{self.seed}_time_{time.time()}",
                       config=self.__dict__, group=self.alg_name, dir='/tmp')

        total_steps = 0
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0

            # 1) First chunk: roll an episode within the real environment and populate the real buffer
            for step in range(max_steps):

                if total_steps > 1_000:
                    action = self.sac_agent.select_action(state).flatten()
                else:
                    action = self.env.action_space.sample()

                next_state, reward, done, _ = self.env.step(action)

                self.real_buffer.push(state, action, reward, next_state, done)

                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                state = next_state

                if done:
                    break

                # 2) Second chunk: train the dynamics model and populate the imaginary buffer if self.model_based
                if total_steps % self.model_train_freq == 0 and len(self.real_buffer) > self.batch_size and self.model_based:

                    model_loss = self.update_model(self.batch_size)

                    # if len(self.real_buffer) > 5_000:
                    if total_steps > 1_000:

                        self.counterfact_rollout()

                # 3) Learn Local Causal Graphical Model from the real buffer
                if total_steps % self.cgm_train_freq == 0:

                    # Learn the CGM with the last self.cgm_train_freq samples
                    self.update_cgm()

                # 4) Third chunk: train the SAC agent
                if total_steps % self.sac_train_freq == 0 and len(self.real_buffer) > self.batch_size:

                    for _ in range(self.agent_steps):

                        final_buffer = self.get_final_buffer()
                        critic_loss, actor_loss, alpha_loss = self.sac_agent.update(final_buffer, self.batch_size)

            # 4) Logging and Printing
            if self.log_wandb:
                wandb.log({
                    "Train/Episode Reward": episode_reward,
                    "Train/Episode Length": episode_steps,
                    "Train/Global Step": total_steps,
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

        filename = base_dir + f"{self.alg_name}_seed_{self.seed}"

        # Save the entire SAC agent
        torch.save(self.sac_agent, filename)
