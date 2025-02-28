import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class EnsembleModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=64, ensemble_size=10, lr=0.001):
        super(EnsembleModel, self).__init__()
        self.state_dim = state_dim
        self.ensemble_size = ensemble_size

        self.models = nn.ModuleList([nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, (state_dim + 1) * 2)  # +1 for reward prediction, *2 for mean and variance
        ) for _ in range(ensemble_size)])

        self.model_optimizer = optim.Adam(self.parameters(), lr=lr)

        self.mean = None
        self.std = None

    def forward(self, x):

        # # First normalize the inputs
        # x = self.normalize_inputs(x)

        # Return the mean and variance of the state and reward predictions
        means = []
        logvars = []
        for model in self.models:
            pred = model(x)
            mean = pred[:, :(self.state_dim + 1)]  # State and reward mean are the first (state_dim + 1) elements
            logvar = pred[:, (self.state_dim + 1):]  # logvar is the last (state_dim + 1) elements

            means.append(mean)
            logvars.append(logvar)

        # Denormalize the outputs
        # means = [self.denormalize_inputs(mean) for mean in means]

        return means, logvars

    def normalize_inputs(self, x):

        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0) + 1e-6

        return (x - self.mean) / self.std

    def denormalize_outputs(self, x):

        return x * self.std + self.mean

