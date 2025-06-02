import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RunningNormalizer:
    def __init__(self, size, device, eps=1e-8):
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
        return (x - self.mean) / (self.var.sqrt() + 1e-8)

    def denormalize(self, x):
        return x * (self.var.sqrt() + 1e-8) + self.mean


class FactorizedEnsembleModel(nn.Module):
    """
    A collection of (state_dim + 1) deep ensemble models. Each ensemble predicts one scalar
    (state_i' or reward), outputting mean and logvar. So each dimension has
    'ensemble_size' small MLPs. At forward time, we produce separate predictions
    for each dimension. This factorization ensures each dimension can see only
    the inputs we decide, if we like.
    """
    def __init__(self, state_dim, action_dim, device, hidden_units=128, ensemble_size=10, lr=0.001):
        super(FactorizedEnsembleModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dimensions = state_dim + 1  # +1 for reward prediction
        self.ensemble_size = ensemble_size
        self.hidden_units = hidden_units
        self.device = device

        self.min_logvar, self.max_logvar = torch.tensor(-10.0, device=device), torch.tensor(5.0, device=device)

        # We will store a separate "ModuleList of MLPs" for each dimension:
        #   dimension_models[d] is a ModuleList of length 'ensemble_size'
        #   dimension_models[d][k] is an MLP that outputs mean+logvar for dimension d
        self.dimensions_models = nn.ModuleList()
        for _ in range(self.dimensions):
            self.dimensions_models.append(nn.ModuleList([nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_units),
                nn.SiLU(),
                nn.Linear(hidden_units, hidden_units),
                nn.SiLU(),
                nn.Linear(hidden_units, 2)  # mean and logvar
            ) for _ in range(ensemble_size)]))

        # Create a single optimizer for all the models parameters
        self.model_optimizer = optim.Adam(self.parameters(), lr=lr)

        # Initialize running normalizers for inputs and outputs
        self.input_normalizer = RunningNormalizer(state_dim + action_dim, device)
        self.output_normalizer = RunningNormalizer(state_dim + 1, device)

        # Put everything on the device
        self.to(device)

    def forward(self, x):
        """
        x: shape (batch, state_dim + action_dim)
        Returns:
          means_all, logvars_all: each is a list of length (state_dim+1),
              where means_all[d] is a list of 'ensemble_size' predictions
              for dimension d, each shape (batch, 1).

        So for dimension d:
          means_all[d][k], logvars_all[d][k] => the (mean, logvar) from
          ensemble member k for that dimension d.
        """

        means_all = []
        logvars_all = []

        # dimension_models[d] => the ensemble for dimension d
        for d in range(self.dimensions):
            means_dim = []
            logvars_dim = []
            for model_k in self.dimensions_models[d]:
                pred = model_k(x)  # shape (batch, 2)
                mean_k = pred[:, 0:1]  # shape (batch,1)
                logvar_k = pred[:, 1:2]  # shape (batch,1)

                # Constrain logvar to be within [min_logvar, max_logvar]
                constr_logvar_k = self.max_logvar - F.softplus(self.max_logvar - logvar_k)
                constr_logvar_k = self.min_logvar + F.softplus(constr_logvar_k - self.min_logvar)

                means_dim.append(mean_k)
                logvars_dim.append(constr_logvar_k)

            means_all.append(means_dim)  # length ensemble_size
            logvars_all.append(logvars_dim)  # length ensemble_size

        return means_all, logvars_all

    def set_parents_from_prob_matrix(self, adjancency):
        """
        adjacency: a tensor of shape [ (state_dim + action_dim), (state_dim + 1) ],
          where adjacency[i, d] = 1 (or True) means "Input i is a parent of output dim d."
          - i in [0..(state_dim+action_dim-1)]
          - d in [0..(state_dim)]

        This method forcibly zeros out columns in the first Linear layer
        for any input 'i' that is NOT a parent of dimension 'd'.

        That is, if adjacency[i,d] = 0, then dimension d is not allowed
        to see input i, so we set the entire column i in the first layer's weight to 0.

        NOTE: This won't permanently freeze them at zero unless you also
        clamp gradients or incorporate a forward-time mask. But it does
        forcibly set them to zero *now*, so the network won't be using them
        unless you train further with unmasked gradients.
        :param adjancency:
        :return:
        """

        # Sample a causal graph mask for each sample in the batch, for each in parents (ns, r)
        sub_cgm_matrix = adjancency[:(self.state_dim + self.action_dim), (self.state_dim + self.action_dim):]
        sub_cgm_matrix = torch.FloatTensor(sub_cgm_matrix).to(self.device)

        for d in range(self.dimensions):
            parents_d = sub_cgm_matrix[:, d]
            for k in range(self.ensemble_size):

                # Sample a causal mask specific for the k-th ensemble member
                parents_d_k = torch.bernoulli(parents_d).to(self.device)

                # Assure that if no parent is selected, the one with the highest weight is selected
                if parents_d_k.sum() == 0:
                    parents_d_k[torch.argmax(parents_d).item()] = 1

                mlp_k = self.dimensions_models[d][k]
                first_layer = mlp_k[0]
                for i_in in range(self.state_dim + self.action_dim):
                    if parents_d_k[i_in] == 0:
                        # Zero out the entire column i_in
                        first_layer.weight.data[:, i_in].zero_()

    def train_factorized_ensemble(self, buffer, batch_size, epochs=100):
        """
        real_batch: a tuple (states, actions, next_states, rewards)
          - states.shape = (N, state_dim)
          - actions.shape= (N, action_dim)
          - next_states.shape= (N, state_dim)
          - rewards.shape= (N,)
        """

        if len(buffer) < 50_000:
            epochs = epochs * 10
        elif len(buffer) < 150_000:
            epochs = epochs * 5
        else:
            epochs = epochs

        for epoch in range(epochs):

            states, actions, rewards, next_states, done = buffer.sample(batch_size)

            # Convert to tensors on device
            states     = torch.FloatTensor(states).to(self.device)
            actions    = torch.FloatTensor(actions).to(self.device)
            next_states= torch.FloatTensor(next_states).to(self.device)
            rewards    = torch.FloatTensor(rewards).to(self.device).unsqueeze(-1)  # shape (N,1)

            # Compute delta state
            delta_state = next_states - states

            # We'll merge next_states & rewards into a single target => shape (N, state_dim+1)
            inputs = torch.cat([states, actions], dim=1)  # shape (N, state_dim+action_dim)
            targets = torch.cat([delta_state, rewards], dim=1)  # shape (N, state_dim+1)

            # Normalize inputs and targets
            inputs = self.input_normalizer.normalize(inputs)
            targets = self.output_normalizer.normalize(targets)

            means_all, logvars_all = self(inputs)

            # means_all[d][k] => shape (N,1)
            # logvars_all[d][k] => shape (N,1)

            # 2) Compute total loss as sum of Gaussian NLL across all dims & ensemble members
            #    For dimension d, the target is targets[:, d], shape (N,)
            #    We'll do a small sum
            total_loss = 0
            self.model_optimizer.zero_grad()
            for d in range(self.dimensions):
                y = targets[:, d:d+1]  # shape (N,1)
                for k in range(self.ensemble_size):
                    mu = means_all[d][k]
                    logvar = logvars_all[d][k]
                    var = torch.exp(logvar)
                    # Gaussian NLL for dimension d, ensemble k:
                    # we can do e.g. MSE * 1/(2*var) + 0.5*logvar
                    nll_k = 0.5*(logvar + (y - mu)**2 / var)
                    total_loss += nll_k.mean()

                # Average over ensemble members
                total_loss /= self.ensemble_size

            # 3) Backprop
            total_loss.backward()
            self.model_optimizer.step()

            return total_loss.item()

