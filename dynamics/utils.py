import torch
import numpy as np
import math
import copy


def empty_gpu_cache():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def linear_scheduler(schedule_params: list[float], epoch: int) -> int:
    """
    schedule_params = [a, b, x, y]
    where a, b are the start and end of the schedule, measured in epochs
    returns: an integer rollout length = clamp( x + ((epoch-a)/(b-a))*(y-x) , [x,y] )
    """
    a, b, x, y = schedule_params
    if b == a:
        val = x
    else:
        val = x + ((epoch - a) / (b - a)) * (y - x)
    lo, hi = min(x, y), max(x, y)
    # clip and cast to int
    return int(np.clip(val, lo, hi))


def flatten_obs(o: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([o["observation"], o["desired_goal"]], axis=0)


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



def compute_jsd(means, var_s):

    # Data are in (ens_size, n_actors, d_state). Need to transpose to (n_actors, ens_size, d_state)
    state_delta_means = means.transpose(0, 1)
    next_state_vars = var_s.transpose(0, 1)
    next_state_vars = rescale_var(next_state_vars)  # shape: (n_actors, ensemble_size, d_state)

    mu, var = state_delta_means, next_state_vars                         # shape: both (n_actors, ensemble_size, d_state)
    n_act, es, d_s = mu.size()                                            # shape: (n_actors, ensemble_size, d_state)

    # entropy of the mean
    mu_diff = mu.unsqueeze(2) - mu.unsqueeze(1)                          # shape: (n_actors, ensemble_size, ensemble_size, d_state)
    var_sum = var.unsqueeze(2) + var.unsqueeze(1)                        # shape: (n_actors, ensemble_size, ensemble_size, d_state)

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
    mean_entropy = total_entropy.mean(dim=1) / 2 + d_s * np.log(2.) / 2   # shape: (n_actors)

    # jensen-shannon divergence
    jsd = entropy_mean - mean_entropy                                 # shape: (n_actors)

    return jsd


#
# def compute_causal_emp_fast(
#     deep_ensemble,
#     current_states,                   # Tensor (B, D_s)
#     policy,                           # π(a|s)
#     n_action_samples: int = 64,
#     n_mixture_samples: int = 512,
#     reward_idx: int = -1,             # index of reward in the model output
#     chunk_size: int = 2,              # ↓ memory footprint of the mixture loop
#     target_dims=None  # ← NEW  (1-D index tensor / list)
# ):
#     """
#     Memory– and speed-optimised per-state-dimension empowerment
#     I(S'_j; A | S=s) for every ensemble member k, batch element b and
#     state dimension j (reward dim excluded).
#
#     Returns
#     -------
#     Tensor (K, B, D_s)   where K = ensemble size, B = batch, D_s = state dims
#     """
#     device   = deep_ensemble.device
#     B, D_s   = current_states.shape
#     K        = deep_ensemble.ensemble_size
#     TWO_PI_E = 2.0 * math.pi * math.e
#
#     # ------------------------------------------------------------------
#     # 1. Sample actions  a_i ~ π(a|s)            → Tensor (A, B, D_a)
#     # ------------------------------------------------------------------
#     actions_np = np.stack(
#         [policy.select_action(current_states) for _ in range(n_action_samples)],
#         axis=0,
#     )
#     actions_sample = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
#
#     # ------------------------------------------------------------------
#     # 2. Single forward pass through the ensemble  → mu_, sigma2  (K, A, B, D_s+1)
#     # ------------------------------------------------------------------
#     s_rep  = current_states.to(device).unsqueeze(0).expand(n_action_samples, -1, -1)
#     inputs = torch.cat(
#         [s_rep.reshape(-1, D_s), actions_sample.reshape(-1, policy.action_dim)],
#         dim=1,
#     )                                        # (A·B, D_s + D_a)
#     inputs = deep_ensemble.input_normalizer.normalize(inputs)
#
#     with torch.no_grad():
#         mean_preds, logvar_preds = deep_ensemble(inputs)  # lists length K
#
#     mean = torch.stack([m.view(n_action_samples, B, -1) for m in mean_preds], dim=0)
#     var  = torch.exp(
#         torch.stack([v.view(n_action_samples, B, -1) for v in logvar_preds], dim=0)
#     )
#
#     # Drop reward dimension
#     mean = mean[..., :reward_idx]             # (K, A, B, D_s)
#     var  = var [...,  :reward_idx]
#     D_s_eff = mean.size(-1)
#
#     # ------------ NEW: slice early to save memory --------------------- #
#     if target_dims is not None:
#         mean = mean[..., target_dims]       # (K, A, B, |idx|)
#         var  = var [..., target_dims]
#     D_eff = mean.size(-1)
#     # ------------------------------------------------------------------ #
#
#     # ------------------------------------------------------------------
#     # 3. Conditional entropy  H(S'|S,A)              (K, B, D_s)
#     # ------------------------------------------------------------------
#     cond_H = 0.5 * torch.log(TWO_PI_E * var).mean(dim=1)
#
#     # ------------------------------------------------------------------
#     # 4. Marginal entropy  H(S'|S)  via MC mixture   (K, B, D_s)
#     #     Process actions in CHUNKS to keep memory low.
#     # ------------------------------------------------------------------
#     # 4.1 Draw mixture-component indices   i ∈ {0,…,A-1}
#     idx = torch.randint(
#         0, n_action_samples,
#         (n_mixture_samples, B),
#         device=device,
#         dtype=torch.long,
#     )                                         # (M, B)
#
#     idx_exp = (
#         idx.unsqueeze(0)                      # (1, M, B)
#            .unsqueeze(-1)                     # (1, M, B, 1)
#            .expand(K, -1, -1, D_s_eff)        # (K, M, B, D)
#     )
#
#     mean_choice = mean.gather(1, idx_exp)     # (K, M, B, D)
#     var_choice  = var .gather(1, idx_exp)
#     x_samples   = torch.normal(mean_choice, torch.sqrt(var_choice))
#
#     # 4.2 Accumulate log-sum-exp over actions in chunks
#     log_sumexp = None                         # will hold (K, M, B, D)
#
#     for start in range(0, n_action_samples, chunk_size):
#         end   = min(start + chunk_size, n_action_samples)
#         mu_     = mean[:, start:end, :, :].unsqueeze(1)      # (K,1,chunk,B,D)
#         sigma2    = var[:,  start:end, :, :].unsqueeze(1)      # (K,1,chunk,B,D)
#
#         diff  = x_samples.unsqueeze(2) - mu_                 # (K,M,chunk,B,D)
#         log_p = -0.5 * (torch.log(2.0 * math.pi * sigma2) + diff.pow(2) / sigma2)                  # (K,M,chunk,B,D)
#
#         log_p_chunk = torch.logsumexp(log_p, dim=2)        # (K,M,B,D)
#
#         log_sumexp = log_p_chunk if log_sumexp is None     \
#                      else torch.logaddexp(log_sumexp, log_p_chunk)
#
#     log_p_mix = log_sumexp - math.log(n_action_samples)    # uniform weights
#     marg_H    = -log_p_mix.mean(dim=1)                     # average over M
#
#     # ------------------------------------------------------------------
#     # 5. Empowerment   I = H - H_cond
#     # ------------------------------------------------------------------
#     return marg_H - cond_H

# ─────────────────────────────────────────────────────────────────────────────
def _compute_empowerment_core(
        deep_ensemble,
        states_slice,                          # (B_slice, D_s)
        policy,
        *,
        n_action_samples,
        n_mixture_samples,
        reward_idx,
        action_chunk,
        mix_chunk,
        target_dims):
    """
    Empowerment for ONE mini-batch of states.
    This is the body of the previous algorithm, unchanged except that
    it works only on   B_slice ≤ batch_chunk   states.
    Returns: Tensor (K, B_slice, D_eff)
    """
    import math, torch, numpy as np

    device   = deep_ensemble.device
    # wdtype   = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    wdtype   = torch.float32
    K        = deep_ensemble.ensemble_size
    B_sl, D_s = states_slice.shape
    TWO_PI_E = torch.tensor(2.*math.pi*math.e, dtype=wdtype, device=device)

    # 1 ─ sample actions
    a_np  = np.stack([policy.select_action(states_slice) for _ in range(n_action_samples)], 0)
    a_all = torch.as_tensor(a_np, dtype=wdtype, device=device)            # (A,B_sl,D_a)

    # 2 ─ forward pass streamed over action_chunk
    mu_parts, s2_parts = [], []
    for a0 in range(0, n_action_samples, action_chunk):
        a1   = min(a0 + action_chunk, n_action_samples)
        act  = a_all[a0:a1]                                               # (chunk,B_sl,D_a)

        srep = states_slice.to(device, dtype=wdtype).unsqueeze(0).expand(a1-a0, -1, -1)
        inp  = torch.cat([srep.reshape(-1, D_s), act.reshape(-1, policy.action_dim)], 1)
        inp  = deep_ensemble.input_normalizer.normalize(inp)
        with torch.no_grad():
            mu_l, logv_l = deep_ensemble(inp)
        mu_parts.append(torch.stack([m.view(a1-a0, B_sl, -1) for m in mu_l], 0))
        s2_parts.append(torch.exp(torch.stack([v.view(a1-a0, B_sl, -1) for v in logv_l], 0)))

    μ  = torch.cat(mu_parts,  dim=1) ;  del mu_parts
    σ2 = torch.cat(s2_parts, dim=1) ;  del s2_parts
    empty_gpu_cache()

    σ2 = σ2.clamp_min(1e-6)  # ← add this

    μ, σ2 = μ[..., :reward_idx], σ2[..., :reward_idx]
    if target_dims is not None:
        μ, σ2 = μ[..., target_dims], σ2[..., target_dims]
    D_eff = μ.size(-1)

    # 3 ─ conditional entropy
    SAFE_LOG = -40.0  # exp(-40) ≈ 4.2e-18

    H_cond = 0.5 * torch.log(TWO_PI_E * σ2).mean(dim=1)
    H_cond = torch.where(torch.isfinite(H_cond),
                         H_cond,
                         torch.full_like(H_cond, SAFE_LOG))

    # 4 ─ marginal entropy streamed over mix_chunk
    acc_H_mix = torch.zeros(K, B_sl, D_eff, dtype=wdtype, device=device)
    done_mix  = 0
    while done_mix < n_mixture_samples:
        m_len = min(mix_chunk, n_mixture_samples - done_mix)
        done_mix += m_len

        idx = torch.randint(0, n_action_samples, (m_len, B_sl),
                            device=device, dtype=torch.long)
        idx_exp = idx.unsqueeze(0).unsqueeze(-1).expand(K, -1, -1, D_eff)
        μ_sel   = μ .gather(1, idx_exp)
        σ2_sel = σ2.gather(1, idx_exp).clamp_min(1e-6)  # ← add clamp here
        x       = torch.normal(μ_sel, torch.sqrt(σ2_sel))

        log_sumexp = None
        for a0 in range(0, n_action_samples, action_chunk):
            a1  = min(a0 + action_chunk, n_action_samples)
            μ_a = μ [:, a0:a1].unsqueeze(1)
            σ2_a = σ2[:, a0:a1].unsqueeze(1).clamp_min(1e-6)  # ← and here
            diff  = x.unsqueeze(2) - μ_a
            log_p = -0.5 * (torch.log(2.*math.pi*σ2_a) + diff.pow(2)/σ2_a)
            lse   = torch.logsumexp(log_p, dim=2)
            log_sumexp = lse if log_sumexp is None else torch.logaddexp(log_sumexp, lse)

        log_p_mix  = log_sumexp - math.log(n_action_samples)
        log_p_mix = torch.where(torch.isfinite(log_p_mix),
                                log_p_mix,
                                torch.full_like(log_p_mix, SAFE_LOG))

        acc_H_mix += (-log_p_mix).mean(dim=1)

        del idx, idx_exp, μ_sel, σ2_sel, x, log_sumexp, lse
        empty_gpu_cache()

    H_mix = acc_H_mix / (n_mixture_samples / mix_chunk)
    return (H_mix - H_cond).to(torch.float32)
# ─────────────────────────────────────────────────────────────────────────────


def compute_causal_emp_fast(
        deep_ensemble,
        current_states,
        policy,
        *,
        n_action_samples : int = 64,
        n_mixture_samples: int = 256,
        reward_idx       : int = -1,
        action_chunk     : int = 16,
        mix_chunk        : int = 128,
        batch_chunk      : int = 5000,          # new
        target_dims          = None):
    """
    Same public API as before, now mini-batches the B dimension.
    """
    B_total = current_states.size(0)
    if B_total <= batch_chunk:
        return _compute_empowerment_core(
            deep_ensemble, current_states, policy,
            n_action_samples=n_action_samples,
            n_mixture_samples=n_mixture_samples,
            reward_idx=reward_idx,
            action_chunk=action_chunk,
            mix_chunk=mix_chunk,
            target_dims=target_dims)

    pieces = []
    for b0 in range(0, B_total, batch_chunk):
        b1 = min(b0 + batch_chunk, B_total)
        slice_out = _compute_empowerment_core(
            deep_ensemble, current_states[b0:b1], policy,
            n_action_samples=n_action_samples,
            n_mixture_samples=n_mixture_samples,
            reward_idx=reward_idx,
            action_chunk=action_chunk,
            mix_chunk=mix_chunk,
            target_dims=target_dims)
        pieces.append(slice_out)
        empty_gpu_cache()

    return torch.cat(pieces, dim=1)         # (K, B_total, D_eff)
# ─────────────────────────────────────────────────────────────────────────────



# def compute_causal_emp(deep_ensemble,
#                        current_states,
#                        policy,
#                        n_action_samples=64,
#                        n_mixture_samples=512):
#     """
#     For each ensemble member k, compute:
#        E_k = H(s'| s) - E_{a}[ H(s'| s,a) ],
#     where H(s'| s) is approximated by a mixture-of-Gaussians over sampled actions,
#     and the expectation is wrt a-samples from the given policy.
#
#     Returns: Tensor [K], empowerment for each ensemble model.
#     """
#
#     # Sample n_action_samples actions
#     n_batch = current_states.shape[0]
#     n_ens = deep_ensemble.ensemble_size
#     n_sts = current_states.shape[1]
#
#     actions_sample = [policy.select_action(current_states) for _ in range(n_action_samples)]
#     actions_sample = np.stack(actions_sample, axis=0)  # shape: (n_action_samples, n_batch, d_action)
#     actions_sample = torch.tensor(actions_sample, dtype=torch.float32, device=deep_ensemble.device)
#
#     current_states = current_states.clone().detach() if torch.is_tensor(current_states) else torch.tensor(current_states)
#
#     # 1) For each k, compute p(r, s' | s, a^(i)) for all a in actions_sample
#     means_all_acts, logvars_all_acts = [], []
#     for k in range(n_action_samples):
#
#         action_i = actions_sample[k]
#         model_input = torch.cat([current_states, action_i], dim=1)
#
#         # # Version with Factorized Ensemble
#         # model_input = deep_ensemble.input_normalizer.normalize(model_input)
#         # with torch.no_grad():
#         #     mean_preds, logvar_preds = deep_ensemble(model_input)
#         # all_preds_mean = torch.stack([torch.stack(group, dim=0).squeeze(-1) for group in mean_preds], dim=0)  # shape: (n_ens, n_batch, d_state)
#         # all_preds_var = torch.stack([torch.stack(group, dim=0).squeeze(-1) for group in logvar_preds], dim=0)  # shape: (n_ens, n_batch, d_state)
#         #
#         # # Swap axes to [n_ens, n_batch, n_dim]
#         # all_preds_mean = all_preds_mean.permute(1, 2, 0)  # shape: (n_batch, d_state, n_ens)
#         # all_preds_var = all_preds_var.permute(1, 2, 0)  # shape: (n_batch, d_state, n_ens)
#
#         # Version with Ensemble
#         model_input = deep_ensemble.input_normalizer.normalize(model_input)
#         with torch.no_grad():
#             mean_preds, logvar_preds = deep_ensemble(model_input)
#         all_preds_mean, all_preds_var = torch.stack(mean_preds, dim=0), torch.stack(logvar_preds, dim=0)  # shape: (n_ens, n_batch, d_state)
#
#         means_all_acts.append(all_preds_mean)
#         logvars_all_acts.append(all_preds_var)
#
#     means_all_acts = torch.stack(means_all_acts, dim=1)  # shape: (n_ens, n_action_samples, n_batch, d_state)
#     logvars_all_acts = torch.stack(logvars_all_acts, dim=1)  # shape: (n_ens, n_action_samples, n_batch, d_state)
#     vars_all_acts = torch.exp(logvars_all_acts)
#
#     # Remove the last reward dimension (d_state) from means_all_acts and vars_all_acts
#     means_all_acts = means_all_acts[:, :, :, :-1]  # shape: (n_ens, n_action_samples, n_batch, d_state)
#     vars_all_acts = vars_all_acts[:, :, :, :-1]  # shape: (n_ens, n_action_samples, n_batch, d_state)
#
#     # 2) Compute conditional entropy E_{a}[ H(s'| s,a) ]
#     entr_per_dim = gaussian_1d_entropy(vars_all_acts)
#     # cond_entr = entr_per_dim.sum(dim=-1)  # shape: (n_ens, n_action_samples, n_batch)
#
#     # Average over actions
#     cond_entr_mean = entr_per_dim.mean(dim=1)  # shape: (n_ens, n_batch, d_state)
#
#     # 3) Approximate H(s'| s) by a mixture of Gaussians: p(r, s'| s) ~ (1/n_action) sum_i=1..n_action p(r,s'| s,a_i)
#     # Compute this for each ensemble member k
#     # We'll store an estimate for H_mix per (n_ens, n_batch, n_sts)
#     marg_entr = torch.zeros(n_ens, n_batch, n_sts, device=deep_ensemble.device)
#
#     for k in range(n_ens):
#
#         # means_k => shape (n_action_samples, B, D)
#         # vars_k  => shape (n_action_samples, B, D)
#         means_k = means_all_acts[k, :, :, :]
#         vars_k = vars_all_acts[k, :, :, :]
#
#         # TODO: This is the computational bottleneck
#         # We're vectorizing the computation across the batch dimension
#         # 1) Draw mixture indices i ~ Uniform(0, n_action_samples)
#         i_choice = torch.randint(low=0, high=n_action_samples, size=(n_mixture_samples, n_batch), device=deep_ensemble.device)  # shape (n_mixture_samples, B)
#
#         # 2) Sample x from the chosen Gaussian  # shape (n_mixture_samples, B, D)
#         m_ = torch.gather(means_k, dim=0, index=i_choice.unsqueeze(-1).expand(n_mixture_samples, n_batch, n_sts))
#         v_ = torch.gather(vars_k, dim=0, index=i_choice.unsqueeze(-1).expand(n_mixture_samples, n_batch, n_sts))
#         sample_x = torch.normal(mean=m_, std=torch.sqrt(v_))  # shape (n_mixture_samples, B, D)
#
#         # 3) Compute p_mix(sample_x) = 1/n_actions * sum_j Normal_j(sample_x) (n_action_samples, n_mixture_samples, B)
#         # Technicaly I have to do gaussian_1d_logpdf for each mixture sample, but I'll do it in a vectorized form
#         x_exp = sample_x.unsqueeze(2)  # (nM, B, 1, D)
#         mean_exp = means_k.permute(1, 0, 2).unsqueeze(0)  # (1, B, nA, D)
#         var_exp = vars_k.permute(1, 0, 2).unsqueeze(0)  # (1, B, nA, D)
#
#         # all_log_p_j = gaussian_1d_logpdf(x_exp, mean_exp, var_exp)  # (nM, B, nA)
#
#         # Compute gaussian_1d_logpdf for each D dimension separately
#         all_log_p_j = torch.zeros(n_mixture_samples, n_batch, n_action_samples, n_sts)  # (nM, B, nA, D)
#         for d in range(n_sts):
#             all_log_p_j[:, :, :, d] = gaussian_1d_logpdf(x_exp[:, :, :, d].unsqueeze(-1),
#                                                          mean_exp[:, :, :, d].unsqueeze(-1),
#                                                          var_exp[:, :, :, d].unsqueeze(-1))  # (nM, B, nA, 1)
#
#         # exponentiate
#         p_j = torch.exp(all_log_p_j)  # shape (n_mixture_samples, B, n_action_samples, D)
#         p_mix_x = (1.0 / n_action_samples) * p_j.sum(dim=2)  # shape (n_mixture_samples, B, D)
#
#         # Average over mixture samples
#         log_p_mix_x = torch.log(p_mix_x + 1e-45)  # add tiny for numerical safety
#         avg_log_p_mix = log_p_mix_x.mean(dim=0)  # shape (B, D)
#
#         # negative is the differential entropy
#         marg_entr[k, :, :] = -avg_log_p_mix  # shape (B, D)
#
#     # 4) Compute E_k = H(s'| s) - E_{a}[ H(s'| s,a) ]
#     causal_empow = marg_entr - cond_entr_mean
#
#     return causal_empow  # shape: (n_ens, n_batch, d_state)


def compute_path_ce(est_cgm,
                    deep_ensemble,
                    current_states,
                    policy,
                    n_action_samples=32,
                    n_mixture_samples=128):
    """
    This function first gathers the indexes of the states S^j such that they satisfy the path A -> S^j -> R, in the
    local_cgm. Then, it computes the empowerment of the states S^j using the compute_causal_emp function.
    """

    if not torch.is_tensor(est_cgm):
        est_cgm = torch.tensor(est_cgm, dtype=torch.float32, device=deep_ensemble.device)
    else:
        est_cgm = est_cgm.to(deep_ensemble.device)

    # Get the indexes of the states S^j such that they satisfy the path A -> S^j -> R from the est_cgm
    state_dim, action_dim = deep_ensemble.state_dim, policy.action_dim

    # edges FROM (actions+states)  TO  (next-states+reward)
    sub_cgm_matrix = est_cgm[:(state_dim + action_dim),
                             (state_dim + action_dim):].detach()

    # ── deterministic parent mask ────────────────────────────────────────
    sub_cgm_matrix = (sub_cgm_matrix > 0.5).float()  # threshold at 0.5

    # states that satisfy  A → Sʲ  **and**  Sʲ → R
    has_a_to_s = sub_cgm_matrix[:action_dim, :state_dim].any(dim=0)      # (D_s,)
    has_s_to_r = sub_cgm_matrix[action_dim:, -1] == 1  # (D_s,)

    r_pa_idx = torch.nonzero(has_a_to_s & has_s_to_r, as_tuple=False).squeeze(-1)  # 1-D tensor

    # r_pa_idx is the indexes of the states S^j such that they satisfy the path A -> S^j -> R.
    # Now we can compute the empowerment of the states S^j using the compute_causal_emp function
    causal_emp = compute_causal_emp_fast(
        deep_ensemble,
        current_states,
        policy,
        n_action_samples=n_action_samples,
        n_mixture_samples=n_mixture_samples,
        action_chunk=16,  # ↓   new explicit arguments
        mix_chunk=64,
        target_dims=r_pa_idx,
    )

    return causal_emp


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


def rescale_var(var, min_log_var=-5., max_log_var=1., decay=0.1):
    min_log_var = torch.tensor(min_log_var, dtype=torch.float32)
    max_log_var = torch.tensor(max_log_var, dtype=torch.float32)
    min_var, max_var = torch.exp(min_log_var), torch.exp(max_log_var)
    return max_var - decay * (max_var - var)
