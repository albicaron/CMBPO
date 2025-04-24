import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import copy
import random


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# Replay Buffer for storing transitions
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_and_return_buffer_format(self, batch_size):

        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        # Put in the ReplayBuffer format
        batch = []
        for i in range(batch_size):
            batch.append((state[i], action[i], reward[i], next_state[i], done[i]))

        return batch

    def push_batch(self, states, actions, rewards, next_states, dones):
        """
        Insert a batch of transitions into the replay buffer. It takes a numpy array or list of transitions for
        each component.
        :param states:
        :param actions:
        :param rewards:
        :param next_states:
        :param dones:
        :return:
        """
        batch_size = len(states)
        # If the buffer is not full, extend it
        for i in range(batch_size):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (states[i],
                                          actions[i],
                                          rewards[i].item(),
                                          next_states[i],
                                          dones[i].item())
            self.position = (self.position + 1) % self.capacity

    def clear(self):
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)


# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 2 * action_dim), std=0.01)
        )

        self.log_std_min = -5.0
        self.log_std_max = 2.0
        self.max_action = max_action

    def forward(self, state):

        out = self.net(state)

        # Split mean and log_std
        mean, log_std = out.chunk(2, dim=-1)

        # Clamp log_std to be within the specified range
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # Replace any NaN / ±Inf that survived clamp (can happen if `out` had NaN)
        bad_mask = torch.isnan(log_std) | torch.isinf(log_std)
        if bad_mask.any():
            log_std[bad_mask] = self.log_std_min

        # Convert log_std to std
        std = log_std.exp().clamp_min(1e-6)

        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)

        # Sometimes the mean is not a Real number, so we need to handle this
        if torch.isnan(mean).any() or torch.isinf(mean).any():

            # Replace the non-real numbers with 0
            mean[torch.isnan(mean)] = 0
            mean[torch.isinf(mean)] = 0

        normal = Normal(mean, std)
        x_t = normal.rsample()  # re-parameterization trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action * self.max_action, log_prob


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        # Q1 architecture
        self.q1 = nn.Sequential(
            layer_init(nn.Linear(state_dim + action_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1))
        )

        # Q2 architecture
        self.q2 = nn.Sequential(
            layer_init(nn.Linear(state_dim + action_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1))
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2


# SAC Agent
class SAC:
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, device='cpu'):

        self.device = device

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state):

        # Ensure the state is a 2D tensor
        if state.ndim == 1:
            state = state.reshape(1, -1)

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action, _ = self.actor.sample(state)
        return action.cpu().data.numpy()

    def update(self, replay_buffer, batch_size):

        # Define the batch size as the minimum of the batch size and the replay buffer length
        batch_size = min(batch_size, len(replay_buffer))

        # Sample a batch from memory
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(np.float32(done)).to(self.device).unsqueeze(1)

        # Update Critic
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            min_qf_next_target = torch.min(target_q1, target_q2) - self.alpha * next_log_pi
            next_q_value = reward + (1 - done) * self.gamma * min_qf_next_target

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, next_q_value) + F.mse_loss(current_q2, next_q_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actions_pred, log_pis = self.actor.sample(state)
        q1, q2 = self.critic(state, actions_pred)
        min_qf_pi = torch.min(q1, q2)
        actor_loss = ((self.alpha * log_pis) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update temperature parameter (alpha)
        alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item(), alpha_loss.item()

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
