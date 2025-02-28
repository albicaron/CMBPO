import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Replay buffer
class ReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.next_states = []

    def add(self, state, action, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.states), batch_size)
        states = torch.FloatTensor([self.states[i] for i in indices])
        actions = torch.FloatTensor([self.actions[i] for i in indices])
        next_states = torch.FloatTensor([self.next_states[i] for i in indices])
        return states, actions, next_states


# Define the model for dynamics
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(DynamicsModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        next_state = self.model(x)
        return next_state


# Define the policy model
class PolicyModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(PolicyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()  # Because action space is between -1 and 1
        )

    def forward(self, state):
        action = self.model(state)
        return action



# Train dynamics model
def train_dynamics_model(replay_buffer, dynamics_model, optimizer, epochs=100, batch_size=64):
    for epoch in range(epochs):
        states, actions, next_states = replay_buffer.sample(batch_size)
        optimizer.zero_grad()
        predicted_next_states = dynamics_model(states, actions)
        loss = nn.MSELoss()(predicted_next_states, next_states)
        loss.backward()
        optimizer.step()