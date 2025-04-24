import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RunningNormalizer:
    def __init__(self, size, device, eps=1e-6):
        self.mean = torch.zeros(size, device=device)
        self.var = torch.ones(size, device=device)
        self.count = torch.tensor(eps, device=device)

    def update(self, x: torch.Tensor):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.size(0)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (self.var.sqrt() + 1e-6)

    def denormalize(self, x):
        return x * (self.var.sqrt() + 1e-6) + self.mean


class EnsembleModel(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_units=200, ensemble_size=10, lr=0.001):
        super(EnsembleModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.device = device

        self.min_logvar, self.max_logvar = torch.tensor(-10.0, device=device), torch.tensor(5.0, device=device)

        self.models = nn.ModuleList([nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, (state_dim + 1) * 2)  # +1 for reward prediction, *2 for mean and variance
        ) for _ in range(ensemble_size)])

        self.model_optimizer = optim.Adam(self.parameters(), lr=lr)

        # Initialize running normalizers for inputs and outputs
        self.input_normalizer = RunningNormalizer(state_dim + action_dim, device)
        self.output_normalizer = RunningNormalizer(state_dim + 1, device)

    def forward(self, x):

        # # Normalize the input
        # if self.input_normalizer is not None:
        #     x = self.input_normalizer.normalize(x)

        # Return the mean and variance of the state and reward predictions
        means = []
        logvars = []
        for model in self.models:
            pred = model(x)
            mean = pred[:, :(self.state_dim + 1)]  # State and reward mean are the first (state_dim + 1) elements
            logvar = pred[:, (self.state_dim + 1):]  # logvar is the last (state_dim + 1) elements

            # Smoothly constrain logvar to be within [min_logvar, max_logvar]
            constr_logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            constr_logvar = self.min_logvar + F.softplus(constr_logvar - self.min_logvar)

            means.append(mean)
            logvars.append(constr_logvar)

        return means, logvars

