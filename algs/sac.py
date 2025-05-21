import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import copy
import random
from collections import deque


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

    def sample(self, batch_size, replace=False):
        if not replace or batch_size <= len(self.buffer):
            # standard behaviour: without replacement
            batch = random.sample(self.buffer, batch_size)
        else:
            # with replacement: draw indices independently
            idxs = np.random.randint(0, len(self.buffer), size=batch_size)
            batch = [self.buffer[i] for i in idxs]

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


class HERReplayBuffer(ReplayBuffer):
    """
    ReplayBuffer + future-based HER (see Andrychowicz et al. 2017).
    Stores one episode at a time; when the episode ends we relabel
    each transition with K goals sampled from the *future* of that
    episode.
    """
    def __init__(self, capacity, env, her_k: int = 4):
        super().__init__(capacity)
        self.env   = env                    # to call env.compute_reward
        self.her_k = her_k
        self.episode: deque = deque(maxlen=env._max_episode_steps)

        # goal dimension (needed to overwrite the tail of the vector)
        self.gdim = env.observation_space["desired_goal"].shape[0]

    # ------------------------------------------------------------------ #
    # During interaction ------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def push_transition(
        self,
        state, action, reward, next_state, done,
        ag, next_ag, dg
    ):
        """Store transition and remember episode context for HER."""
        self.episode.append((
            state, action, reward, next_state, done, ag, next_ag, dg
        ))
        super().push(state, action, reward, next_state, done)

        if done or len(self.episode) == self.episode.maxlen:
            self._relabel_and_store()
            self.episode.clear()

    # ------------------------------------------------------------------ #
    def _relabel_and_store(self):
        """
        Implements the ‘future’ strategy: for every transition t
        sample K future time-steps τ ≥ t and replace the desired goal
        with the achieved goal at τ.
        """
        ep = list(self.episode)
        T  = len(ep)
        for t in range(T):
            # choose K future indices including t itself
            future_idx = np.random.choice(
                np.arange(t, T, dtype=int), self.her_k, replace=True
            )
            for τ in future_idx:
                (
                    s, a, _r, s_next, d,
                    ag, ag_next, _dg
                ) = ep[t]

                # the *new* desired goal is the achieved goal at τ
                new_dg = ep[τ][5]                         # ag at τ
                # recompute reward with env-provided function (vectorised false)
                reward_env = self.env.unwrapped  # always has compute_reward
                new_r = reward_env.compute_reward(ag_next, new_dg, {}).astype(np.float32)

                # overwrite the tail (desired_goal) of the state vectors
                s_her      = s.copy()
                s_next_her = s_next.copy()
                s_her[     -self.gdim:] = new_dg
                s_next_her[-self.gdim:] = new_dg

                super().push(s_her, a, new_r, s_next_her, d)



# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()

        self.action_dim = action_dim

        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 2 * action_dim), std=0.01)
        )

        self.log_std_min, self.log_std_max = -20.0, 2.0
        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        mean, log_std = x.split(self.action_dim, dim=-1)

        # --------- NUMERICAL GUARDS ----------------------------------
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        mean    = torch.nan_to_num(mean, nan=0.0)                  # ❹ no in-place write!
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # re-parameterised Gaussian noise
        action = torch.tanh(z)  # squash to (-1,1)

        # log π(a|s) with stable tanh-Jacobian
        log_prob = normal.log_prob(z)  # N(mean,σ)
        log_prob -= torch.log1p(-action.pow(2) + 1e-6)  # ❺ log(1-tanh²)+ϵ
        log_prob = log_prob.sum(dim=-1, keepdim=True)

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
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, weight_decay=1e-4)

        self.max_action = max_action
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device)).item()
        self.target_entropy = -float(action_dim)  # for continuous action spaces
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr, weight_decay=1e-4)

    # ------------------------------------------------------------------ #
    # Select an action.
    # deterministic=False  →  stochastic  (default, training time)
    # deterministic=True   →  tanh(mean)  (used for evaluation)
    # ------------------------------------------------------------------ #
    def select_action(self, state, deterministic: bool = False):
        # Ensure the state is a 2D tensor
        if state.ndim == 1:
            state = state.reshape(1, -1)

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        if deterministic:
            with torch.no_grad():
                mean, _ = self.actor(state)  # μ(s)
                action = torch.tanh(mean)  # squash to (‑1,1)
        else:
            action, _ = self.actor.sample(state)  # re‑parameterised sample

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

            # Clamp the target Q-values to avoid overestimation
            next_q_value = torch.clamp(next_q_value, -1e6, 1e6)

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
