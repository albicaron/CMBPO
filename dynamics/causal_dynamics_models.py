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


class AdaptiveCausalNormalizer:
    """
    A sophisticated normalizer that adapts to changing causal structures
    by normalizing per-dimension empowerment before summing.
    """

    def __init__(self, state_dim, device, eps=1e-8):
        self.state_dim = state_dim
        self.device = device
        self.eps = eps

        # Maintain normalizers for individual dimension empowerment
        self.per_dim_normalizer = RunningNormalizer(state_dim, device, eps)

        # Also track the sum statistics
        self.sum_normalizer = RunningNormalizer(1, device, eps)

        # Track which dimensions are typically active
        self.dim_activity = torch.zeros(state_dim, device=device)
        self.activity_momentum = 0.99

    def normalize_empowerment(self, empowerment_per_dim, active_dims):
        """
        Normalize empowerment values accounting for which dimensions are active.

        Args:
            empowerment_per_dim: Empowerment values for each dimension (K, B, n_dims)
            active_dims: Indices of dimensions that are causally relevant

        Returns:
            Normalized total empowerment (K, B, 1)
        """
        K, B, n_dims = empowerment_per_dim.shape

        # Update activity tracking
        activity_mask = torch.zeros(self.state_dim, device=self.device)
        activity_mask[active_dims] = 1.0
        self.dim_activity = (self.activity_momentum * self.dim_activity +
                             (1 - self.activity_momentum) * activity_mask)

        # Create a full-sized tensor with zeros for inactive dimensions
        full_empowerment = torch.zeros(K, B, self.state_dim, device=self.device)
        full_empowerment[:, :, active_dims] = empowerment_per_dim

        # Update per-dimension statistics using the full tensor
        # This maintains consistent statistics even as dimensions activate/deactivate
        for k in range(K):
            self.per_dim_normalizer.update(full_empowerment[k])

        # Normalize each dimension independently
        normalized_per_dim = torch.zeros_like(full_empowerment)
        for i in range(self.state_dim):
            if i in active_dims:
                # Find the index in the compressed empowerment_per_dim
                dim_idx = (active_dims == i).nonzero(as_tuple=True)[0].item()

                # Normalize using dimension-specific statistics
                dim_mean = self.per_dim_normalizer.mean[i]
                dim_std = torch.sqrt(self.per_dim_normalizer.var[i] + self.eps)

                normalized_per_dim[:, :, i] = (empowerment_per_dim[:, :, dim_idx] - dim_mean) / dim_std

        # Sum only the active dimensions
        empowerment_sum = normalized_per_dim[:, :, active_dims].sum(dim=2, keepdim=True)

        # Update sum statistics
        self.sum_normalizer.update(empowerment_sum.reshape(-1, 1))

        # Final normalization of the sum
        final_normalized = self.sum_normalizer.normalize(empowerment_sum)

        return final_normalized


class FactorizedEnsembleModel(nn.Module):
    """
    A collection of (state_dim + 1) deep ensemble models. Each ensemble predicts one scalar
    (state_i' or reward), outputting mean and logvar. This factorization allows each dimension
    to see only the inputs specified by the causal graph.
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

        # Initialize running normalizers
        self.input_normalizer = RunningNormalizer(state_dim + action_dim, device)
        self.output_normalizer = RunningNormalizer(state_dim + 1, device)

        # Put everything on the device
        self.to(device)

        # Store individual causal masks for each ensemble member
        # Shape: (ensemble_size, dimensions, state_dim + action_dim)
        self.register_buffer('ensemble_causal_masks',
                             torch.ones(ensemble_size, self.dimensions, state_dim + action_dim))

        # Store the probabilistic adjacency matrix
        self.register_buffer('prob_adjacency',
                             torch.ones(self.dimensions, state_dim + action_dim))

    def set_causal_structure_with_uncertainty(self, adjacency_probs, uncertainty_method='ensemble_sampling'):
        """
        Set causal structure handling uncertainty from bootstrapped PC.

        Args:
            adjacency_probs: Probabilistic adjacency matrix (dimensions, inputs)
            uncertainty_method: How to handle uncertainty
            min_prob_threshold: Minimum probability to consider an edge
        """
        # Extract relevant submatrix: (states+actions) -> (next_states+reward)
        sub_cgm = adjacency_probs[:(self.state_dim + self.action_dim), (self.state_dim + self.action_dim):]
        sub_cgm = torch.tensor(sub_cgm.clone().detach(), dtype=torch.float32, device=self.device)
        self.prob_adjacency = sub_cgm.T  # Shape: (dimensions, inputs)

        if uncertainty_method == 'ensemble_sampling':
            self._set_ensemble_sampling_masks()
        elif uncertainty_method == 'probabilistic_masking':
            self._set_probabilistic_masks()
        elif uncertainty_method == 'threshold_deterministic':
            self._set_threshold_masks(threshold=0.5)
        else:
            raise ValueError(f"Unknown uncertainty method: {uncertainty_method}")

    # 1) First method: Set masks based on ensemble sampling
    def _set_ensemble_sampling_masks(self):
        """
        Set masks based on ensemble sampling from the probabilistic adjacency matrix.
        """
        # Each ensemble member will sample its own mask for each dimension
        for d in range(self.dimensions):
            for k in range(self.ensemble_size):
                # Sample a mask for ensemble member k and dimension d
                mask = torch.bernoulli(self.prob_adjacency[d]).to(self.device)

                # Ensure at least one parent is selected
                if mask.sum() == 0:
                    mask[torch.argmax(self.prob_adjacency[d]).item()] = 1

                self.ensemble_causal_masks[k, d] = mask

    # 2) Second method: Set masks based on probabilistic adjacency matrix
    def _set_probabilistic_masks(self):
        """
        Set masks based on the probabilistic adjacency matrix. Each ensemble member uses the same mask.
        """
        # All ensemble members use the same probabilistic mask, which means sample only once for each dimension d
        for d in range(self.dimensions):
            mask = torch.bernoulli(self.prob_adjacency[d]).to(self.device)

            # Ensure at least one parent is selected
            if mask.sum() == 0:
                mask[torch.argmax(self.prob_adjacency[d]).item()] = 1

            for k in range(self.ensemble_size):
                self.ensemble_causal_masks[k, d] = mask

    # 3) Third method: Set masks based on a threshold
    def _set_threshold_masks(self, threshold=0.5):
        """
        Set masks based on a threshold applied to the probabilistic adjacency matrix.
        Each ensemble member uses the same mask.
        """
        for d in range(self.dimensions):
            mask = (self.prob_adjacency[d] >= threshold).to(self.device)

            # Ensure at least one parent is selected
            if mask.sum() == 0:
                mask[torch.argmax(self.prob_adjacency[d]).item()] = 1

            for k in range(self.ensemble_size):
                self.ensemble_causal_masks[k, d] = mask

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

        means_all, logvars_all = [], []

        # dimension_models[d] => the ensemble for dimension d
        for d in range(self.dimensions):
            means_dim = []
            logvars_dim = []

            for k, model_k in enumerate(self.dimensions_models[d]):

                # CRITICAL FIX: Apply ensemble-specific causal mask
                causal_mask = self.ensemble_causal_masks[k, d]  # Shape: (state_dim + action_dim,)
                x_masked = x * causal_mask.unsqueeze(0)  # Apply mask to inputs

                pred = model_k(x_masked)  # shape (batch, 2)
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

    # def set_parents_from_prob_matrix(self, adjancency):
    #     """
    #     adjacency: a tensor of shape [ (state_dim + action_dim), (state_dim + 1) ],
    #       where adjacency[i, d] = 1 (or True) means "Input i is a parent of output dim d."
    #       - i in [0..(state_dim+action_dim-1)]
    #       - d in [0..(state_dim)]
    #
    #     This method forcibly zeros out columns in the first Linear layer
    #     for any input 'i' that is NOT a parent of dimension 'd'.
    #
    #     That is, if adjacency[i,d] = 0, then dimension d is not allowed
    #     to see input i, so we set the entire column i in the first layer's weight to 0.
    #
    #     NOTE: This won't permanently freeze them at zero unless you also
    #     clamp gradients or incorporate a forward-time mask. But it does
    #     forcibly set them to zero *now*, so the network won't be using them
    #     unless you train further with unmasked gradients.
    #     :param adjancency:
    #     :return:
    #     """
    #
    #     # Sample a causal graph mask for each sample in the batch, for each in parents (ns, r)
    #     sub_cgm_matrix = adjancency[:(self.state_dim + self.action_dim), (self.state_dim + self.action_dim):]
    #     sub_cgm_matrix = torch.FloatTensor(sub_cgm_matrix).to(self.device)
    #
    #     for d in range(self.dimensions):
    #         parents_d = sub_cgm_matrix[:, d]
    #         for k in range(self.ensemble_size):
    #
    #             # Sample a causal mask specific for the k-th ensemble member
    #             parents_d_k = torch.bernoulli(parents_d).to(self.device)
    #
    #             # Assure that if no parent is selected, the one with the highest weight is selected
    #             if parents_d_k.sum() == 0:
    #                 parents_d_k[torch.argmax(parents_d).item()] = 1
    #
    #             mlp_k = self.dimensions_models[d][k]
    #             first_layer = mlp_k[0]
    #             for i_in in range(self.state_dim + self.action_dim):
    #                 if parents_d_k[i_in] == 0:
    #                     # Zero out the entire column i_in
    #                     first_layer.weight.data[:, i_in].zero_()

    def train_factorized_ensemble(self, buffer, batch_size, epochs=100):
        """
        real_batch: a tuple (states, actions, next_states, rewards)
          - states.shape = (N, state_dim)
          - actions.shape= (N, action_dim)
          - next_states.shape= (N, state_dim)
          - rewards.shape= (N,)
        """
        total_loss = 0.0
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

            means_all, logvars_all = self(inputs)  # shape (state_dim+1, ensemble_size, N, 1)
            # means_all[d][k] => shape (N,1)
            # logvars_all[d][k] => shape (N,1)

            # 2) Compute total loss as sum of Gaussian NLL across all dims & ensemble members
            #    For dimension d, the target is targets[:, d], shape (N,)
            #    We'll do a small sum
            total_loss = 0
            self.model_optimizer.zero_grad()
            for d in range(self.dimensions):
                y_d = targets[:, d:d+1]  # shape (N,1)
                for k in range(self.ensemble_size):
                    mu = means_all[d][k]
                    logvar = logvars_all[d][k]
                    var = torch.exp(logvar)
                    # Gaussian NLL for dimension d, ensemble k:
                    # we can do e.g. MSE * 1/(2*var) + 0.5*logvar
                    nll_k = 0.5*(logvar + (y_d - mu)**2 / var)
                    total_loss += nll_k.mean()

                # Average over ensemble members
                total_loss /= self.ensemble_size

            # 3) Backprop
            total_loss.backward()
            self.model_optimizer.step()

        return total_loss.item()
