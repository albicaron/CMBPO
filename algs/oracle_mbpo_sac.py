import numpy as np
import torch
import torch.nn.functional as F

from algs.sac import SAC, ReplayBuffer
from dynamics.causal_models import EnsembleModel
import pandas as pd

import wandb
import time
import random

from pgmpy.estimators import PC


class ORACLE_MBPO_SAC:
    def __init__(self, env, seed, dev, log_wandb=True, model_based=True, pure_imaginary=True, lr_model=1e-3,
                 lr_sac=0.0003, gamma=0.99, tau=0.005, alpha=0.2, model_rollout_length=5, num_model_rollouts=400,  # Maybe put 100_000 as it is batched anyway
                 update_size=250, sac_train_freq=1, model_train_freq=100, batch_size=200):

        self.env = env
        self.seed = seed
        self.log_wandb = log_wandb
        self.model_based = model_based
        self.pure_imaginary = pure_imaginary
        self.alg_name = 'C_MBPO_SAC'

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.model_rollout_length = model_rollout_length
        self.num_model_rollouts = num_model_rollouts

        self.update_size = update_size  # Size of the final buffer to train the SAC agent made of %5-95% real-imaginary

        self.sac_train_freq = sac_train_freq
        self.model_train_freq = model_train_freq

        self.batch_size = batch_size

        self.device = dev

        # The agent is fed only the true causal parent of action (state 1), as this is the only relevant feature for the
        # agent to learn the optimal policy.
        self.sac_agent = SAC(self.state_dim - 1, self.action_dim, self.max_action, lr=lr_sac, gamma=gamma,
                             tau=tau, alpha=alpha, device=self.device)

        # Here in the causal version, we first use the "cdt" kernel conditional independence test to select the
        # relevant features for the reward and dynamics models. We then use only the discovered parents as inputs to
        # the models. The EnsembleModel is the same as before, but we declare it after parents are discovered.
        # self.reward_parents = None
        # self.next_states_parents = None

        # self.reward_model = None
        # self.next_states_models = None

        # Declare reward model: this takes only S2 as input
        self.reward_model = EnsembleModel(1, 1, lr=5e-4).to(self.device)  # S1
        self.state_1_model = EnsembleModel(2, 1, lr=5e-4).to(self.device)  # S1, A
        self.state_2_model = EnsembleModel(1, 1, lr=5e-4).to(self.device)  # S1_next

        self.real_buffer = ReplayBuffer(int(10_000))
        self.imaginary_buffer = ReplayBuffer(int(10_000))

    def update_causal_models(self, batch_size=256, epochs=100):

        if len(self.real_buffer) < 5_000:
            epochs = 50  # Overriding the epochs to train the model more if first 10 episodes
        elif len(self.real_buffer) < 10_000:
            epochs = 20
        else:
            epochs = 20

        for _ in range(epochs):

            state, action, reward, next_state, done = self.real_buffer.sample(batch_size)

            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)

            # 1) First train the reward model
            reward_input = state[:, 0].unsqueeze(1)  # S1
            reward_output = reward

            reward_model_loss = 0
            self.reward_model.model_optimizer.zero_grad()
            mean_r_preds, logvar_r_preds = self.reward_model(reward_input)
            for mean_pred, logvar_pred in zip(mean_r_preds, logvar_r_preds):
                var_pred = torch.exp(logvar_pred)
                reward_model_loss += F.gaussian_nll_loss(mean_pred, reward_output, var_pred)
            reward_model_loss.backward()
            self.reward_model.model_optimizer.step()

            # 2) Then train the next states models: train S1 first
            state_1_input = torch.cat([state[:, 0].unsqueeze(1), action], dim=1)  # S1, A
            state_1_output = next_state[:, 0].unsqueeze(1)  # S1_next

            state_1_model_loss = 0
            self.state_1_model.model_optimizer.zero_grad()
            mean_s1_preds, logvar_s1_preds = self.state_1_model(state_1_input)
            for mean_pred, logvar_pred in zip(mean_s1_preds, logvar_s1_preds):
                var_pred = torch.exp(logvar_pred)
                state_1_model_loss += F.gaussian_nll_loss(mean_pred, state_1_output, var_pred)
            state_1_model_loss.backward()
            self.state_1_model.model_optimizer.step()

            # 3) Train S2 model. S1_next, S2
            state_2_input = next_state[:, 0].unsqueeze(1)  # S1_next
            state_2_output = next_state[:, 1].unsqueeze(1)  # S2_next

            state_2_model_loss = 0
            self.state_2_model.model_optimizer.zero_grad()
            mean_s2_preds, logvar_s2_preds = self.state_2_model(state_2_input)
            for mean_pred, logvar_pred in zip(mean_s2_preds, logvar_s2_preds):
                var_pred = torch.exp(logvar_pred)
                state_2_model_loss += F.gaussian_nll_loss(mean_pred, state_2_output, var_pred)
            state_2_model_loss.backward()
            self.state_2_model.model_optimizer.step()

        return reward_model_loss.item(), state_1_model_loss.item(), state_2_model_loss.item()

    def imaginary_rollout(self):

        ############################################################
        # TODO: Implement adjustment of imaginary horizon based on the model uncertainty! "When to trust the model?"
        ############################################################

        if len(self.real_buffer) < 2_000:
            length_traj = 1
        else:
            length_traj = self.model_rollout_length

        # Populate the self.imaginary_buffer with the new model rollout
        num_samples = min(self.num_model_rollouts, len(self.real_buffer))

        if num_samples == 0:
            return  # Skip rollout if the buffer is empty

        initial_states, _, _, _, _ = self.real_buffer.sample(num_samples)
        initial_states = torch.FloatTensor(initial_states).to(self.device)

        for _ in range(length_traj):

            # First select actions from the SAC agent
            actions = self.sac_agent.select_action(initial_states[:, :1])  # Only S1, as causal parent
            actions = torch.FloatTensor(actions).to(self.device)

            # Second, predict the rewards using initial_states S2 only
            reward_input = initial_states[:, 0].unsqueeze(1)  # S1
            mean_r_preds, logvar_r_preds = self.reward_model(reward_input)
            rewards = torch.stack(mean_r_preds).mean(dim=0)

            # Third, predict next S1 using S1 model
            state_1_input = torch.cat([initial_states[:, 0].unsqueeze(1), actions], dim=1)  # S1, A
            mean_s1_preds, logvar_s1_preds = self.state_1_model(state_1_input)
            next_states_1 = torch.stack(mean_s1_preds).mean(dim=0)

            # Fourth, predict next S2 using S2 model
            state_2_input = next_states_1  # S1_next
            mean_s2_preds, logvar_s2_preds = self.state_2_model(state_2_input)
            next_states_2 = torch.stack(mean_s2_preds).mean(dim=0)

            next_states = torch.cat([next_states_1, next_states_2], dim=1)

            dones = torch.zeros_like(rewards)

            for i in range(num_samples):
                self.imaginary_buffer.push(
                    initial_states[i].cpu().numpy(),
                    actions[i].cpu().numpy(),
                    rewards[i].item(),
                    next_states[i].detach().cpu().numpy(),
                    dones[i].item()
                )

            initial_states = next_states.detach()

    def get_final_buffer(self):

        if len(self.imaginary_buffer) == 0:
            proportion_real = 0.05
        else:
            proportion_real = 0.0

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

        # Create a copy of the final buffer, but with the new states in the buffer consisting only of the first state
        # (the causal parent of the action)
        final_buffer_states = []
        for state, action, reward, next_state, done in final_buffer.buffer:
            final_buffer_states.append((state[:1], action, reward, next_state[:1], done))

        final_buffer.buffer = final_buffer_states

        return final_buffer

    def select_reward_parents(self):

        # Run only if self.reward_parents and self.next_states_parents are None
        if self.reward_parents is not None and self.next_states_parents is not None:
            return

        # Unpack the real buffer
        state, action, reward, next_state, done = self.real_buffer.sample(len(self.real_buffer))

        parent_set_rewards = np.c_[reward, state, action]
        parent_set_next_states = np.c_[next_state, state, action]

        data_reward = pd.DataFrame(parent_set_rewards, columns=['R', 'S1', 'S2', 'A'])
        data_next_states = pd.DataFrame(parent_set_next_states, columns=['next_S1', 'next_S2', 'S1', 'S2', 'A'])

        # Initialize the PC algorithm
        pc_reward = PC(data_reward)
        pc_next_states = PC(data_next_states)

        # Estimate the directed acyclic graph
        reward_graph = pc_reward.estimate(ci_test='pearsonr', show_progress=False, significance_level=0.05)
        next_states_graph = pc_next_states.estimate(ci_test='pearsonr', show_progress=False, significance_level=0.05)

        # Print the graph edges
        print("Reward Parents: ", reward_graph.edges())
        print("Next States Parents: ", next_states_graph.edges())

        self.reward_parents = list(reward_graph.edges())
        self.next_states_parents = list(next_states_graph.edges())

    def declare_causal_models(self):

        # Do this only if self.reward_model and self.next_states_models are None
        if self.reward_model is not None and self.next_states_models is not None:
            return

        # 1) Reward Model first
        # Extract from self.reward_parents list only the tuples that features "R" in it
        reward_parents = [x for x in self.reward_parents if 'R' in x]

        # Extract the parents of the reward
        reward_parents = [x for reward_parent in reward_parents for x in reward_parent if x != 'R']

        # Declare input and output dimensions
        reward_input_dim = len(reward_parents)
        reward_output_dim = 1

        # Declare the reward model
        self.reward_model = EnsembleModel(reward_input_dim, reward_output_dim, lr=1e-3)

        # 2) Next States Models
        # For each of the next states, extract the parents and declare the model. Then put these models in a list.
        self.next_states_models = []

        # Here for loop over anything that contains "next_S" in the tuple
        for i in range(self.state_dim):
            reg_ex = f'next_S{i + 1}'
            filtered_tuples = [x for x in self.next_states_parents if reg_ex in x]

            state_i_parents = [x for filtered_tuple in filtered_tuples for x in filtered_tuple if x != f'next_S{i + 1}']
            state_i_input_dim = len(state_i_parents)
            state_i_output_dim = 1

            state_i_model = EnsembleModel(state_i_input_dim, state_i_output_dim, lr=1e-3)
            self.next_states_models.append(state_i_model)

        # Replace the self.reward_parents and self.next_states_parents with the filtered versions
        self.reward_parents = reward_parents

    def train(self, num_episodes=100, max_steps=200):
        if self.log_wandb:
            project_name = "SimpleCausalEnv_v1"
            wandb.init(project=project_name, sync_tensorboard=True,
                       name=f"{self.alg_name}_SAC_seed_{self.seed}_time_{time.time()}",
                       config=self.__dict__, group=self.alg_name, dir='/tmp')

        total_steps = 0
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0

            # 1) First chunk: roll an episode with the real environment and populate the real buffer
            for step in range(max_steps):

                if episode > 5:
                    # Pass only the first state to the SAC agent
                    action = self.sac_agent.select_action(state.reshape(1, -1)[:, :1]).flatten()  # Only S1
                else:
                    action = self.env.action_space.sample()

                next_state, reward, done, truncated, _ = self.env.step(action)

                self.real_buffer.push(state, action, reward, next_state, done)

                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                state = next_state

                if done or truncated:
                    break

                # 2) Second chunk: train the dynamics model and populate the imaginary buffer if self.model_based
                if total_steps % self.model_train_freq == 0 and len(self.real_buffer) >= self.batch_size and self.model_based:

                    # self.select_reward_parents()
                    # self.declare_causal_models()
                    # if self.reward_model is not None and self.next_states_models is not None:
                    model_loss = self.update_causal_models(self.batch_size)

                    # if len(self.real_buffer) > 5_000:
                    # if episode >= 3:
                    self.imaginary_rollout()

                # 3) Third chunk: train the SAC agent
                if total_steps % self.sac_train_freq == 0 and len(self.real_buffer) >= self.batch_size:

                    for _ in range(1):

                        final_buffer = self.get_final_buffer()

                        # Update the SAC agent for 20 epochs
                        # for _ in range(5):
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
