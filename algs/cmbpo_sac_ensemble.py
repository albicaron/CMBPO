import torch
import torch.nn.functional as F
import numpy as np

from algs.sac import SAC, ReplayBuffer
from dynamics.causal_models import StructureLearning, set_p_matrix
from dynamics.utils import compute_jsd, compute_path_ce
from dynamics.dynamics_models import EnsembleModel
import matplotlib.pyplot as plt

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
    def __init__(self, env, seed, dev, log_wandb=True, model_based=False, pure_imaginary=True,
                 sl_method="PC", bootstrap=None, n_bootstrap=100, cgm_train_freq=5_000,
                 causal_bonus=True, causal_eta=0.01, var_causal_bonus=False, var_causal_eta=0.001,
                 jsd_bonus=False, jsd_eta=0.01, jsd_thres=1.0,
                 lr_model=1e-3,
                 lr_sac=0.0003, agent_steps=10, gamma=0.99, tau=0.005, alpha=0.2, max_rollout_len=1,
                 rollout_per_step=400,  # Maybe put 100_000 as it is batched anyway
                 update_size=256, sac_train_freq=1, model_train_freq=250, batch_size=256):

        steps_per_epoch = 1_000
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
        self.num_model_rollouts = rollout_per_step

        self.update_size = update_size  # Size of the final buffer to train the SAC agent made of %5-95% real-imaginary

        # The agent steps are 1 for SAC and agent_steps for MBPO
        self.agent_steps = agent_steps if self.model_based else 1
        self.sac_train_freq = sac_train_freq
        self.model_train_freq = model_train_freq

        self.batch_size = batch_size

        self.device = dev

        self.sac_agent = SAC(self.state_dim, self.action_dim, self.max_action, lr=lr_sac, gamma=gamma,
                             tau=tau, alpha=alpha, device=self.device)

        self.ensemble_model = EnsembleModel(state_dim=self.state_dim, action_dim=self.action_dim, lr=lr_model,
                                            device=self.device, ensemble_size=5).to(self.device)

        self.real_buffer = ReplayBuffer(int(1_000_000))
        self.imaginary_buffer = ReplayBuffer(int(rollout_per_step * steps_per_epoch))  # 1_000 * 400 = 400_000

        self.jsd_thres, self.jsd_bonus, self.jsd_eta = jsd_thres, jsd_bonus, jsd_eta

        # Causal MBPO specific
        self.sl_method = sl_method
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

        if len(self.real_buffer) < 50_000:
            epochs = epochs * 10
        elif len(self.real_buffer) < 150_000:
            epochs = epochs * 5
        else:
            epochs = epochs

        for _ in range(epochs):

            state, action, reward, next_state, done = self.real_buffer.sample(batch_size)

            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)

            # Compute delta state
            delta_state = next_state - state

            model_input = torch.cat([state, action], dim=1)
            next_s_rew = torch.cat([delta_state, reward], dim=1)

            # Normalize the inputs and outputs
            model_input = self.ensemble_model.input_normalizer.normalize(model_input)
            next_s_rew = self.ensemble_model.output_normalizer.normalize(next_s_rew)

            mean_preds, logvar_preds = self.ensemble_model(model_input)

            model_loss = 0
            self.ensemble_model.model_optimizer.zero_grad()
            for mean_pred, logvar_pred in zip(mean_preds, logvar_preds):
                var_pred = torch.exp(logvar_pred)
                model_loss += F.gaussian_nll_loss(mean_pred, next_s_rew, var_pred, reduction='mean')

            # Compute the total loss
            model_loss /= self.ensemble_model.ensemble_size

            model_loss.backward()
            self.ensemble_model.model_optimizer.step()

        return model_loss.item()

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
        add_on = 0
        if len(self.real_buffer) > 200_000:
            add_on = int(len(self.real_buffer) / 200_000)

        max_length_traj = self.max_rollout_len + add_on

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

        # Sample an ensemble index for each sample in the batch, to use consistent ensemble members for each rollout
        ensemble_idx = torch.randint(self.ensemble_model.ensemble_size, (num_samples,))

        for t in range(max_length_traj):

            # If everything is "inactive", exit early
            if not active_mask.any():
                break

            actions = self.sac_agent.select_action(initial_states)
            actions = torch.FloatTensor(actions).to(self.device)

            model_input = torch.cat([initial_states, actions], dim=1)

            # Ensemble predictions: shape [ensemble_size, batch_size, next_state_dim+1]
            # Normalize the inputs
            model_input = self.ensemble_model.input_normalizer.normalize(model_input)
            with torch.no_grad():
                mean_preds, logvar_preds = self.ensemble_model(model_input)
            all_preds_mean, all_preds_var = torch.stack(mean_preds, dim=0), torch.stack(logvar_preds, dim=0)

            # Next state is sampled from the ensemble according to the ensemble_idx previously sampled outside the loop
            # TODO:  NB, Here we pick a single ensemble member for each batch sample to keep consistent graphs
            # ensemble_idx = torch.randint(self.ensemble_model.ensemble_size, (1,)).item()
            # mean_pred, log_var_pred = all_preds_mean[ensemble_idx], all_preds_var[ensemble_idx]
            batch_idx = torch.arange(num_samples, device=self.device)
            mean_pred = all_preds_mean[ensemble_idx, batch_idx]

            # Pick normalized rewards
            reward_pred = mean_pred[:, -1].clone().unsqueeze(1)

            # Compute uncertainty as disagreement across ensemble (JSD) - can use it also as intrinsic reward
            ns_jsd = compute_jsd(all_preds_mean, torch.exp(all_preds_var))

            # REWARDS AUGMENTATION
            if self.jsd_bonus:
                reward_pred += self.jsd_eta * ns_jsd

            if self.causal_bonus:

                # TODO: Here the mixture still creates a bit of computational bottleneck
                # causal_empow = compute_causal_emp(self.ensemble_model, current_states, self.sac_agent)
                causal_empow = compute_path_ce(self.est_cgm, self.ensemble_model, initial_states, self.sac_agent)
                causal_empow_bonus = causal_empow.mean(dim=0)  # shape: (n_batch, sts_dim)
                std_causal_empow_bonus = causal_empow.std(dim=0)  # shape: (n_batch, sts_dim)

                # Scale CE by running std of the reward channel for numerical stability
                reward_std = self.ensemble_model.output_normalizer.var[-1].detach().clamp_min(1e-6).sqrt()
                de_meaned_ce = causal_empow_bonus - causal_empow_bonus.mean(dim=0)

                causal_bonus_tot = self.causal_eta * (de_meaned_ce.sum(dim=1, keepdim=True) / (reward_std + 1e-6))
                reward_pred += causal_bonus_tot

                if self.var_causal_bonus:
                    std_bonus = (std_causal_empow_bonus.sum(dim=1, keepdim=True) / (reward_std + 1e-6))
                    reward_pred += self.var_causal_eta * std_bonus

                # # Normalize the causal empowerment before adding it to the reward
                # causal_empow_bonus = (causal_empow_bonus - causal_empow_bonus.mean(dim=0)) / (causal_empow_bonus.std(dim=0) + 1e-8)
                #
                # causal_bonus_tot = self.causal_eta * causal_empow_bonus.sum(dim=1, keepdim=True)
                # reward_pred += causal_bonus_tot
                #
                # if self.var_causal_bonus:
                #     reward_pred += self.var_causal_eta * std_causal_empow_bonus.sum(dim=1, keepdim=True)

            # Replace the reward prediction with the augmented one
            mean_pred[:, -1] = reward_pred.squeeze(1)

            # Denormalize the outputs
            denorm_mean_pred = self.ensemble_model.output_normalizer.denormalize(mean_pred)

            next_states = denorm_mean_pred[:, :-1]
            next_states = initial_states + next_states  # Add the delta state to the initial state

            rewards = denorm_mean_pred[:, -1].unsqueeze(1)
            dones = torch.zeros_like(rewards)

            # Compute the mask
            push_mask = active_mask & (ns_jsd <= jsd_threshold)

            # Update active_mask for samples that are still active but exceed the threshold
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

    def update_cgm(self):

        # Learn the CGM with the last self.cgm_train_freq samples
        last_cgm_data = self.real_buffer.buffer[-self.cgm_train_freq:]
        state, action, reward, next_state, _ = map(np.stack, zip(*last_cgm_data))
        reward = reward.reshape(-1, 1)

        data_cgm = np.concatenate([state, action, next_state, reward], axis=1)
        self.est_cgm = self.local_cgm.learn_dag(data_cgm,
                                                prior_knowledge=self.pk,
                                                n_bootstrap=self.n_bootstrap)

        # # Set parents of (s, a) to (ns, r) to -1
        # self.ensemble_model.set_parents_from_prob_matrix(self.est_cgm)

        # Plot DAG at the end of the training
        if self.log_wandb and len(self.real_buffer) % (self.cgm_train_freq + 1) == 0:
            fig = self.local_cgm.plot_dag(self.est_cgm, self.true_cgm)
            wandb.log({"Train/Estimated CGM": wandb.Image(fig)})
            plt.close(fig)

    def get_final_buffer(self, proportion_real=0.05):

        if len(self.real_buffer) < 10_000:
            proportion_real = 0.5

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
            project_name = self.env.unwrapped.spec.id if self.env.unwrapped.spec != None else 'SimpleCausal_Multi'
            wandb.init(project=project_name, sync_tensorboard=False,
                       name=f"{self.alg_name}_SAC_seed_{self.seed}_time_{time.time()}",
                       config=self.__dict__, group=self.alg_name, dir='/tmp')

        total_steps = 0
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0

            # 1) First chunk: roll an episode within the real environment and populate the real buffer
            for step in range(max_steps):

                if total_steps > 5_000:
                    action = self.sac_agent.select_action(state).flatten()
                else:
                    action = self.env.action_space.sample()

                next_state, reward, done, truncated, _ = self.env.step(action)
                terminal = done or truncated

                self.real_buffer.push(state, action, reward, next_state, terminal)

                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                state = next_state

                if terminal:
                    break

                # 2) Update normalizers every self.model_train_freq steps in model-based training
                if total_steps % self.model_train_freq == 0 and self.model_based:

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

                # All this after the warm-up period of 5_000 steps used to populate the real buffer
                if total_steps >= 5_000:

                    # 3) Third chunk: train the dynamics model and populate the imaginary buffer if self.model_based
                    if total_steps % self.model_train_freq == 0 and len(self.real_buffer) > self.batch_size and self.model_based:

                        model_loss = self.update_model(self.batch_size * 4)
                        self.counterfact_rollout()

                    # 4) Learn Local Causal Graphical Model from the real buffer
                    # if total_steps % self.cgm_train_freq == 0:  # Re-Learn every self.cgm_train_freq steps
                    if total_steps == (self.cgm_train_freq + 1):

                        # Learn the CGM with the last self.cgm_train_freq samples
                        self.update_cgm()

                    # 5) Fifth chunk: train the SAC agent
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
