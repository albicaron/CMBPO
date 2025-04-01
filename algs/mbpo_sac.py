import torch
import torch.nn as nn
import torch.nn.functional as F

from algs.sac import SAC, ReplayBuffer
from dynamics.utils import compute_jsd
from dynamics.dynamics_models import EnsembleModel

import wandb
import time
import random


class MBPO_SAC:
    def __init__(self, env, seed, dev, log_wandb=True, model_based=False, pure_imaginary=True, lr_model=1e-3,
                 lr_sac=0.0003, agent_steps=1, gamma=0.99, tau=0.005, alpha=0.2, max_rollout_len=15,
                 num_model_rollouts=400,  # Maybe put 100_000 as it is batched anyway
                 update_size=250, sac_train_freq=1, model_train_freq=100, batch_size=250):

        self.env = env
        self.seed = seed
        self.log_wandb = log_wandb
        self.model_based = model_based
        self.pure_imaginary = pure_imaginary
        self.alg_name = 'MBPO_SAC' if self.model_based else 'SAC'

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

        self.ensemble_model = EnsembleModel(state_dim=self.state_dim, action_dim=self.action_dim, lr=lr_model,
                                            ensemble_size=5).to(self.device)

        self.real_buffer = ReplayBuffer(int(10_000))
        self.imaginary_buffer = ReplayBuffer(int(10_000))

    def update_model(self, batch_size=256, epochs=50):

        for _ in range(epochs):

            state, action, reward, next_state, done = self.real_buffer.sample(batch_size)

            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)

            model_input = torch.cat([state, action], dim=1)
            next_s_rew = torch.cat([next_state, reward], dim=1)

            # Normalize the inputs
            mean_preds, logvar_preds = self.ensemble_model(model_input)

            model_loss = 0
            self.ensemble_model.model_optimizer.zero_grad()
            for mean_pred, logvar_pred in zip(mean_preds, logvar_preds):

                var_pred = torch.exp(logvar_pred)
                model_loss += F.gaussian_nll_loss(mean_pred, next_s_rew, var_pred)

            model_loss.backward()
            self.ensemble_model.model_optimizer.step()

        return model_loss.item()

    def imaginary_rollout(self):
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
            all_preds_mean, all_preds_var = torch.stack(mean_preds, dim=0), torch.stack(logvar_preds, dim=0)

            # Next state is sampled from the ensemble
            ensemble_idx = torch.randint(self.ensemble_model.ensemble_size, (1,)).item()
            mean_pred, log_var_pred = all_preds_mean[ensemble_idx], all_preds_var[ensemble_idx]
            next_states = mean_pred[:, :-1]
            rewards = mean_pred[:, -1].unsqueeze(1)
            dones = torch.zeros_like(rewards)

            # Compute uncertainty as disagreement across ensemble (JSD)
            ns_jsd = compute_jsd(all_preds_mean, torch.exp(all_preds_var))

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

            # 1) First chunk: roll an episode with the real environment and populate the real buffer
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
                        self.imaginary_rollout()

                # 3) Third chunk: train the SAC agent
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
