import torch
import numpy as np


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


def compute_causal_emp(deep_ensemble,
                       causal_masks,
                       current_states,
                       policy,
                       n_action_samples=50,
                       n_mixture_samples=100):
    """
    For each ensemble member k, compute:
       E_k = H(s'| s) - E_{a}[ H(s'| s,a) ],
    where H(s'| s) is approximated by a mixture-of-Gaussians over sampled actions,
    and the expectation is wrt a-samples from the given policy.

    Returns: Tensor [K], empowerment for each ensemble model.
    """

    # Sample n_action_samples actions
    n_batch = current_states.shape[0]
    n_ens = deep_ensemble.ensemble_size
    n_sts = deep_ensemble.state_dim

    actions_sample = [policy.select_action(current_states) for _ in range(n_action_samples)]
    actions_sample = np.stack(actions_sample, axis=0)  # shape: (n_action_samples, n_batch, d_action)
    actions_sample = torch.tensor(actions_sample, dtype=torch.float32)

    current_states = torch.tensor(current_states, dtype=torch.float32)

    # 1) For each k, compute p(r, s' | s, a^(i)) for all a in actions_sample
    means_all_acts, logvars_all_acts = [], []
    for k in range(n_action_samples):

        action_i = actions_sample[k]
        model_input = torch.cat([current_states, action_i], dim=1)

        with torch.no_grad():

            # Here for each target variable in (ns, r), we multiply the parent causal mask
            all_preds_mean = torch.zeros(n_ens, n_batch, n_sts + 1)
            all_preds_var = torch.zeros(n_ens, n_batch, n_sts + 1)

            # Loop over the target variables
            for i in range(deep_ensemble.state_dim + 1):
                masked_inputs = model_input * causal_masks[:, :, i]
                mean_pred_i, logvar_pred_i = deep_ensemble(masked_inputs)
                mean_pred_i, logvar_pred_i = torch.stack(mean_pred_i, dim=0), torch.stack(logvar_pred_i, dim=0)
                all_preds_mean[:, :, i], all_preds_var[:, :, i] = mean_pred_i[:, :, i], logvar_pred_i[:, :, i]

        means_all_acts.append(all_preds_mean)
        logvars_all_acts.append(all_preds_var)

    means_all_acts = torch.stack(means_all_acts, dim=0)  # shape: (n_ens, n_action_samples, n_batch, d_state)
    logvars_all_acts = torch.stack(logvars_all_acts, dim=0)  # shape: (n_ens, n_action_samples, n_batch, d_state)
    vars_all_acts = torch.exp(logvars_all_acts)

    # 2) Compute conditional entropy E_{a}[ H(s'| s,a) ]
    entr_per_dim = gaussian_1d_entropy(vars_all_acts)
    cond_entr = entr_per_dim.sum(dim=-1)  # shape: (n_action_samples, n_ens, n_batch)

    # Average over actions
    cond_entr_mean = cond_entr.mean(dim=0)  # shape: (n_ens, n_batch)

    # 3) Approximate H(s'| s) by a mixture of Gaussians: p(r, s'| s) ~ (1/n_action) sum_i=1..n_action p(r,s'| s,a_i)
    # Compute this for each ensemble member k
    # We'll store an estimate for H_mix per (n_ens, n_batch)
    marg_entr = torch.zeros(n_ens, n_batch)

    for k in range(n_ens):
        # means_k => shape (n_action_samples, B, D)
        # vars_k  => shape (n_action_samples, B, D)
        means_k = means_all_acts[:, k, :, :]
        vars_k = vars_all_acts[:, k, :, :]

        for b in range(n_batch):
            # We do n_mixture_samples Monte Carlo draws from the mixture
            log_p_mix_accum = 0.0
            for _ in range(n_mixture_samples):
                # 1) pick a mixture index i ~ Uniform
                i_choice = np.random.randint(0, n_action_samples)

                # 2) sample x from that chosen Gaussian
                m_ = means_k[i_choice, b, :]  # shape (D,)
                v_ = vars_k[i_choice, b, :]
                # sample_x => shape (D,)
                sample_x = torch.normal(mean=m_, std=torch.sqrt(v_))

                # 3) compute p_mix(sample_x) = 1/n_actions * sum_j Normal_j(sample_x)
                #    We'll do it dimension-by-dimension (diagonal):
                #      Normal_j(sample_x) = exp( gaussian_1d_logpdf(sample_x, means_k[j,b,:], vars_k[j,b,:]) )
                # We'll accumulate them in a loop or vectorized.
                # shape (n_action_samples,)
                # all_log_p_j => log of each Normal_j(sample_x)
                # p_j => we need to exponentiate
                # then sum up, multiply by 1/n_action_samples
                # then log again.

                # Let's do it in a vectorized form:
                all_log_p_j = gaussian_1d_logpdf(
                    sample_x.unsqueeze(0),  # shape (1, D)
                    means_k[:, b, :],  # shape (n_action_samples, D)
                    vars_k[:, b, :]  # shape (n_action_samples, D)
                )  # => shape (n_action_samples,)
                # exponentiate
                p_j = torch.exp(all_log_p_j)  # shape (n_action_samples,)
                p_mix_x = (1.0 / n_action_samples) * p_j.sum(dim=0)  # scalar
                log_p_mix_x = torch.log(p_mix_x + 1e-45)  # add tiny for numerical safety

                log_p_mix_accum += log_p_mix_x.item()

            # Average log p_mixture
            avg_log_p_mix = log_p_mix_accum / n_mixture_samples
            # negative is the differential entropy
            marg_entr[k, b] = -avg_log_p_mix

    # 4) Compute E_k = H(s'| s) - E_{a}[ H(s'| s,a) ]
    causal_empow = marg_entr - cond_entr_mean

    return causal_empow


def gaussian_1d_entropy(var_1d):
    """
    Differential entropy for a 1D Gaussian N(., var_1d):
      0.5 * log(2 * pi * e * var_1d).
    var_1d: Tensor (any shape)
    Returns the same shape of entropies.
    """
    return 0.5 * torch.log(2 * np.pi * np.e * var_1d)


def gaussian_1d_logpdf(x, mean, var):
    """
    Computes the log of a diagonal Gaussian density:
        log N(x | mean, diag(var))
    summing across all dimensions in the last axis.

    Args:
        x    : ( ..., D )
        mean : ( ..., D )
        var  : ( ..., D )  diagonal variances

    Returns:
        logprob : ( ... )  (summed across D)
    """
    # 1) log( (2 pi)^(D/2) * det(Sigma)^(1/2) )
    #    = 0.5 * sum_d [ log(2 pi) + log(var_d) ]
    log_det = 0.5 * torch.sum(torch.log(2.0 * np.pi * var), dim=-1)

    # 2) quadratic form:  0.5 * sum_d [ (x_d - m_d)^2 / var_d ]
    quad = 0.5 * torch.sum((x - mean) ** 2 / var, dim=-1)

    # total log likelihood = -( log_det + quad )
    return -(log_det + quad)
