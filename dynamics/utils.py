import torch
import numpy as np
from collections import deque


def compute_jsd(means, var_s):

    # Data are in (ens_size, n_actors, d_state). Need to transpose to (n_actors, ens_size, d_state)
    state_delta_means = means.transpose(0, 1)
    next_state_vars = var_s.transpose(0, 1)

    mu, var = state_delta_means, next_state_vars                         # shape: both (n_actors, ensemble_size, d_state)
    n_act, es, d_s = mu.size()                                            # shape: (n_actors, ensemble_size, d_state)

    # entropy of the mean
    mu_diff = mu.unsqueeze(1) - mu.unsqueeze(2)                           # shape: (n_actors, ensemble_size, ensemble_size, d_state)
    var_sum = var.unsqueeze(1) + var.unsqueeze(2)                         # shape: (n_actors, ensemble_size, ensemble_size, d_state)

    err = (mu_diff * 1 / var_sum * mu_diff)                               # shape: (n_actors, ensemble_size, ensemble_size, d_state)
    err = torch.sum(err, dim=-1)                                          # shape: (n_actors, ensemble_size, ensemble_size)
    det = torch.sum(torch.log(var_sum), dim=-1)                           # shape: (n_actors, ensemble_size, ensemble_size)

    log_z = -0.5 * (err + det)                                            # shape: (n_actors, ensemble_size, ensemble_size)
    log_z = log_z.reshape(n_act, es * es)                                 # shape: (n_actors, ensemble_size * ensemble_size)
    mx, _ = log_z.max(dim=1, keepdim=True)                                # shape: (n_actors, 1)
    log_z = log_z - mx                                                    # shape: (n_actors, ensemble_size * ensemble_size)
    exp = torch.exp(log_z).mean(dim=1, keepdim=True)                      # shape: (n_actors, 1)
    entropy_mean = -mx - torch.log(exp)                                   # shape: (n_actors, 1)
    entropy_mean = entropy_mean[:, 0]                                     # shape: (n_actors)

    # mean of entropies
    total_entropy = torch.sum(torch.log(var), dim=-1)                     # shape: (n_actors, ensemble_size)
    mean_entropy = total_entropy.mean(dim=1) / 2 + d_s * np.log(2.) / 2    # shape: (n_actors)

    # jensen-shannon divergence
    jsd = entropy_mean - mean_entropy                                 # shape: (n_actors)

    return jsd
