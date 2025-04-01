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

    current_states = current_states.clone().detach() if torch.is_tensor(current_states) else torch.tensor(current_states)

    # 1) For each k, compute p(r, s' | s, a^(i)) for all a in actions_sample
    means_all_acts, logvars_all_acts = [], []
    for k in range(n_action_samples):

        action_i = actions_sample[k]
        model_input = torch.cat([current_states, action_i], dim=1)

        with torch.no_grad():
            mean_preds, logvar_preds = deep_ensemble(model_input)
        all_preds_mean = torch.stack([torch.stack(group, dim=0).squeeze(-1) for group in mean_preds], dim=0)
        all_preds_var = torch.stack([torch.stack(group, dim=0).squeeze(-1) for group in logvar_preds], dim=0)

        # Swap axes to [n_ens, n_batch, n_dim]
        all_preds_mean = all_preds_mean.permute(1, 2, 0)  # shape: (n_batch, d_state, n_ens)
        all_preds_var = all_preds_var.permute(1, 2, 0)  # shape: (n_batch, d_state, n_ens)

        means_all_acts.append(all_preds_mean)
        logvars_all_acts.append(all_preds_var)

    means_all_acts = torch.stack(means_all_acts, dim=1)  # shape: (n_ens, n_action_samples, n_batch, d_state)
    logvars_all_acts = torch.stack(logvars_all_acts, dim=1)  # shape: (n_ens, n_action_samples, n_batch, d_state)
    vars_all_acts = torch.exp(logvars_all_acts)

    # 2) Compute conditional entropy E_{a}[ H(s'| s,a) ]
    entr_per_dim = gaussian_1d_entropy(vars_all_acts)
    cond_entr = entr_per_dim.sum(dim=-1)  # shape: (n_ens, n_action_samples, n_batch)

    # Average over actions
    cond_entr_mean = cond_entr.mean(dim=1)  # shape: (n_ens, n_batch)

    # 3) Approximate H(s'| s) by a mixture of Gaussians: p(r, s'| s) ~ (1/n_action) sum_i=1..n_action p(r,s'| s,a_i)
    # Compute this for each ensemble member k
    # We'll store an estimate for H_mix per (n_ens, n_batch)
    marg_entr = torch.zeros(n_ens, n_batch)

    for k in range(n_ens):

        # means_k => shape (n_action_samples, B, D)
        # vars_k  => shape (n_action_samples, B, D)
        means_k = means_all_acts[k, :, :, :]
        vars_k = vars_all_acts[k, :, :, :]

        # TODO: This is the computational bottleneck
        # We're vectorizing the computation across the batch dimension
        # 1) Draw mixture indices i ~ Uniform(0, n_action_samples)
        i_choice = torch.randint(low=0, high=n_action_samples, size=(n_mixture_samples, n_batch))  # shape (n_mixture_samples, B)

        # 2) Sample x from the chosen Gaussian  # shape (n_mixture_samples, B, D)
        m_ = torch.gather(means_k, dim=0, index=i_choice.unsqueeze(-1).expand(n_mixture_samples, n_batch, n_sts + 1))
        v_ = torch.gather(vars_k, dim=0, index=i_choice.unsqueeze(-1).expand(n_mixture_samples, n_batch, n_sts + 1))
        sample_x = torch.normal(mean=m_, std=torch.sqrt(v_))  # shape (n_mixture_samples, B, D)

        # 3) Compute p_mix(sample_x) = 1/n_actions * sum_j Normal_j(sample_x) (n_action_samples, n_mixture_samples, B)
        # Technicaly I have to do gaussian_1d_logpdf for each mixture sample, but I'll do it in a vectorized form
        x_exp = sample_x.unsqueeze(2)  # (nM, B, 1, D)
        mean_exp = means_k.permute(1, 0, 2).unsqueeze(0)  # (1, B, nA, D)
        var_exp = vars_k.permute(1, 0, 2).unsqueeze(0)  # (1, B, nA, D)

        all_log_p_j = gaussian_1d_logpdf(x_exp, mean_exp, var_exp)  # (nM, B, nA)

        # exponentiate
        p_j = torch.exp(all_log_p_j)  # shape (n_mixture_samples, B, n_action_samples)
        p_mix_x = (1.0 / n_action_samples) * p_j.sum(dim=-1)  # shape (n_mixture_samples, B)

        # Average over mixture samples
        log_p_mix_x = torch.log(p_mix_x + 1e-45)  # add tiny for numerical safety
        avg_log_p_mix = log_p_mix_x.mean(dim=0)  # shape (B,)

        # negative is the differential entropy
        marg_entr[k, :] = -avg_log_p_mix

        # for b in range(n_batch):
        #     # We do n_mixture_samples Monte Carlo draws from the mixture
        #     log_p_mix_accum = 0.0
        #     for _ in range(n_mixture_samples):
        #         # 1) pick a mixture index i ~ Uniform
        #         i_choice = np.random.randint(0, n_action_samples)
        #
        #         # 2) sample x from that chosen Gaussian
        #         m_ = means_k[i_choice, b, :]  # shape (D,)
        #         v_ = vars_k[i_choice, b, :]
        #         # sample_x => shape (D,)
        #         sample_x = torch.normal(mean=m_, std=torch.sqrt(v_))
        #
        #         # 3) compute p_mix(sample_x) = 1/n_actions * sum_j Normal_j(sample_x)
        #         #    We'll do it dimension-by-dimension (diagonal):
        #         #      Normal_j(sample_x) = exp( gaussian_1d_logpdf(sample_x, means_k[j,b,:], vars_k[j,b,:]) )
        #         # We'll accumulate them in a loop or vectorized.
        #         # shape (n_action_samples,)
        #         # all_log_p_j => log of each Normal_j(sample_x)
        #         # p_j => we need to exponentiate
        #         # then sum up, multiply by 1/n_action_samples
        #         # then log again.
        #
        #         # Let's do it in a vectorized form:
        #         all_log_p_j = gaussian_1d_logpdf(
        #             sample_x.unsqueeze(0),  # shape (1, D)
        #             means_k[:, b, :],  # shape (n_action_samples, D)
        #             vars_k[:, b, :]  # shape (n_action_samples, D)
        #         )  # => shape (n_action_samples,)
        #         # exponentiate
        #         p_j = torch.exp(all_log_p_j)  # shape (n_action_samples,)
        #         p_mix_x = (1.0 / n_action_samples) * p_j.sum(dim=0)  # scalar
        #         log_p_mix_x = torch.log(p_mix_x + 1e-45)  # add tiny for numerical safety
        #
        #         log_p_mix_accum += log_p_mix_x.item()
        #
        #     # Average log p_mixture
        #     avg_log_p_mix = log_p_mix_accum / n_mixture_samples
        #     # negative is the differential entropy
        #     marg_entr[k, b] = -avg_log_p_mix

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
    Computes log N(x | mean, diag(var)) for diagonal Gaussians,
    over arbitrary batch dimensions.

    Supports broadcasting between:
        x:     (nM, B, 1, D)
        mean:  (1, B, nA, D)
        var:   (1, B, nA, D)

    Returns:
        logprob: (nM, B, nA)
    """

    assert x.shape[-1] == mean.shape[-1] == var.shape[-1], "Mismatch in last (D) dimension"
    assert x.shape[1] == mean.shape[1] == var.shape[1], "Mismatch in batch dimension"

    # Broadcasted shapes: (..., D)
    log_det = 0.5 * torch.sum(torch.log(2.0 * np.pi * var), dim=-1)  # shape: (1, B, nA)
    quad = 0.5 * torch.sum((x - mean) ** 2 / var, dim=-1)  # shape: (nM, B, nA)

    return -(log_det + quad)
