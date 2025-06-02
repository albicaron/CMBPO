import torch
import numpy as np
import torch.nn.functional as F

from algs.sac import SAC, ReplayBuffer, HERReplayBuffer
from dynamics.utils import compute_jsd, linear_scheduler, flatten_obs
from dynamics.dynamics_models import EnsembleModel, mbpo_nll

import gymnasium as gym
from gymnasium.spaces import Dict

import wandb
import time
import random


# Check if MPS is available and set the device to 'mps' if on MacOS, 'cuda' if on GPU, or 'cpu' otherwise
def set_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


class MBPO_SAC:
    def __init__(self,
                 env: gym.Env,
                 seed: int = 0,
                 dev: torch.device = None,
                 log_wandb: bool = False,
                 model_based: bool = True,
                 lr_model: float = 1e-3,
                 lr_sac: float = 3e-4,
                 agent_steps: int = 40,  # Number of MBPO agent steps per training step (if SAC agent_steps=1)
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 max_rollout_len: int = 1,  # Max length of the rollout in steps
                 rollout_schedule: list = None,  # Schedule rollout length - for Half-Cheetah = [20_000, 100_000, 1, 15]
                 rollout_per_step: float = 400,  # Number of rollouts per training step
                 update_size: int = 256,  # Size of the final buffer to train the SAC agent made of 5%-95%
                 sac_train_freq: int = 1,  # Frequency of SAC agent training steps
                 model_train_freq: int = 250,  # Frequency of model training steps
                 batch_size: int = 256,
                 eval_freq: int = 1_000,  # Frequency of policy evaluation
                 ):

        self.env = env
        self.seed = seed
        self.device = dev
        self.log_wandb = log_wandb
        self.model_based = model_based
        self.alg_name = 'MBPO_SAC' if self.model_based else 'SAC'
        self.eval_freq = eval_freq

        if isinstance(env.observation_space, Dict):
            self.env_type = "gym_robotics"
            self.state_dim = (env.observation_space["observation"].shape[0] +
                              env.observation_space["desired_goal"].shape[0])
            self.warmup_steps = 500
        else:
            self.env_type = "gym_mujoco"
            self.state_dim = env.observation_space.shape[0]
            self.warmup_steps = 5_000

        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.max_rollout_len = max_rollout_len
        self.num_model_rollouts = int(rollout_per_step * model_train_freq)  # Number of rollouts per training step 400 * 250 = 100_000

        self.update_size = update_size  # Size of the final buffer to train the SAC agent made of %5-95% real-imaginary

        # The agent steps are 1 for SAC and agent_steps for MBPO
        self.total_steps, self.max_steps = 0, 0
        self.agent_steps = agent_steps if self.model_based else 1
        self.sac_train_freq = sac_train_freq
        self.model_train_freq = model_train_freq
        self.batch_size = batch_size

        self.sac_agent = SAC(self.state_dim, self.action_dim, self.max_action, lr=lr_sac, gamma=gamma,
                             tau=tau, alpha=alpha, device=self.device)
        self.ensemble_model = EnsembleModel(state_dim=self.state_dim, action_dim=self.action_dim, lr=lr_model,
                                            device=self.device, ensemble_size=7).to(self.device)
        self.rollout_schedule = rollout_schedule
        self.elite_size = 5
        self.elite_idxs = list(range(self.elite_size))   # fallback before the first update

        # self.real_buffer = ReplayBuffer(int(1_000_000))
        if self.env_type == "gym_robotics":
            self.real_buffer = HERReplayBuffer(int(1_000_000), env, her_k=4)
        else:
            self.real_buffer = ReplayBuffer(int(1_000_000))
        self.imaginary_buffer = ReplayBuffer(int(1_000_000))

    def update_model(
        self,
        batch_size: int = 256,
        max_epochs: int = 20,          # hard cap (MBPO default)
        min_epochs: int = 10,            # early‑stop can only trigger after this
        patience: int = 5,              # “epochs since any net improved”
        val_split: float = 0.2,         # 20% hold‑out
        improvement_threshold: float = 0.01  # 1% relative improvement
    ):
        """
        Train the dynamics ensemble exactly as in the original MBPO implementation:
        * fixed 80/20 train–hold‑out split of *all* real data;
        * each epoch is one complete pass over the train split;
        * per‑network early stopping with best‑snapshot restore.
        Returns the mean final hold‑out loss across the ensemble.
        """

        if len(self.real_buffer) < batch_size:     # not enough data yet
            return 0.0

        # ------------------------------------------------------------------ #
        # 0. Build the full data set (on CPU to save GPU RAM)
        # ------------------------------------------------------------------ #
        s, a, r, ns, _ = map(np.stack, zip(*self.real_buffer.buffer))
        r = r.reshape(-1, 1)
        delta_s = ns - s

        inputs  = torch.from_numpy(np.concatenate([s, a], axis=1)).float()
        targets = torch.from_numpy(np.concatenate([delta_s, r], axis=1)).float()

        N = inputs.size(0)
        n_hold = int(val_split * N)

        perm = torch.randperm(N)
        val_idx, train_idx = perm[:n_hold], perm[n_hold:]

        train_in, train_tg = inputs[train_idx], targets[train_idx]
        val_in,   val_tg   = inputs[val_idx],   targets[val_idx]

        # One‑off normalisation & device transfer for the hold‑out
        val_in = self.ensemble_model.input_normalizer.normalize(val_in.to(self.device))
        val_tg = self.ensemble_model.output_normalizer.normalize(val_tg.to(self.device))

        # ------------------------------------------------------------------ #
        # 1. Initial hold‑out evaluation  ➜  fills best_val with *finite* numbers
        # ------------------------------------------------------------------ #
        E = self.ensemble_model.ensemble_size
        best_val, best_snap = [], []

        with torch.no_grad():
            init_mean, init_logvar = self.ensemble_model(val_in)
            for i, (m, lv) in enumerate(zip(init_mean, init_logvar)):
                # vloss = F.gaussian_nll_loss(m, val_tg, torch.exp(lv), reduction='mean').item()
                vloss = mbpo_nll(pred_mean=m, pred_logvar=lv, target=val_tg).item()
                best_val.append(vloss)
                best_snap.append({k: v.detach().cpu() for k, v in self.ensemble_model.models[i].state_dict().items()})

        # ------------------------------------------------------------------ #
        # 2. Training loop: full‑data sweep per epoch
        # ------------------------------------------------------------------ #
        epochs_since_update = 0
        for epoch in range(max_epochs):

            # shuffle once per epoch
            idx = torch.randperm(train_in.size(0))
            train_in, train_tg = train_in[idx], train_tg[idx]

            for start in range(0, train_in.size(0), batch_size):
                end = start + batch_size
                b_in  = train_in[start:end].to(self.device)
                b_tg  = train_tg[start:end].to(self.device)

                # normalise on‑device
                b_in = self.ensemble_model.input_normalizer.normalize(b_in)
                b_tg = self.ensemble_model.output_normalizer.normalize(b_tg)

                self.ensemble_model.model_optimizer.zero_grad()
                mean, logvar = self.ensemble_model(b_in)

                loss = 0.0
                for m, lv in zip(mean, logvar):
                    # loss += F.gaussian_nll_loss(m, b_tg, torch.exp(lv), reduction='mean')
                    loss += mbpo_nll(pred_mean=m, pred_logvar=lv, target=b_tg)
                (loss / E).backward()
                self.ensemble_model.model_optimizer.step()

            # ---------------- hold‑out evaluation ---------------- #
            with torch.no_grad():
                v_mean, v_logvar = self.ensemble_model(val_in)

                improved = False
                for i, (m, lv) in enumerate(zip(v_mean, v_logvar)):
                    vloss = mbpo_nll(pred_mean=m, pred_logvar=lv, target=val_tg).item()

                    # relative improvement > 1%  (works even at first pass)
                    if (best_val[i] - vloss) / abs(best_val[i]) > improvement_threshold:
                        best_val[i] = vloss
                        best_snap[i] = {k: v.detach().cpu()for k, v in self.ensemble_model.models[i].state_dict().items()}
                        improved = True

            epochs_since_update = 0 if improved else epochs_since_update + 1
            if epoch + 1 >= min_epochs and epochs_since_update > patience:
                break

        # ------------------------------------------------------------------ #
        # 3. Restore the best snapshot for each ensemble member
        # ------------------------------------------------------------------ #
        for i, snap in enumerate(best_snap):
            if snap is not None:
                self.ensemble_model.models[i].load_state_dict(snap)

        # ------------------------------------------------------------------ #
        # 4. Pick the 5 heads with the lowest hold‑out loss  (elite set)
        # ------------------------------------------------------------------ #
        elite_idxs = np.argsort(best_val)[: self.elite_size].tolist()
        self.elite_idxs = elite_idxs  # ←  stored for rollouts

        return float(np.mean(best_val))

    @torch.no_grad()
    def imaginary_rollout(self):
        """
        Rolls out from real states using the learned model. The length of each rollout
        is dynamically adjusted based on ensemble disagreement/uncertainty.

        Idea:
        - We keep rolling out up to 'self.max_rollout_len' steps.
        - At each step, compute the standard deviation (or variance) across
          the ensemble for the *next state*. If it exceeds some threshold,
          we stop rolling out that particular state.
        """

         # Augment max_length_traj by 1 every 10_000 steps
        max_length_traj = linear_scheduler(self.rollout_schedule, self.total_steps)
        num_samples = int(self.num_model_rollouts // max_length_traj)  # keeps the seed fixed to the number of rollouts
        initial_states, _, _, _, _ = self.real_buffer.sample(num_samples, replace=True)
        initial_states = torch.FloatTensor(initial_states).to(self.device)

        # "active_mask[i] = False" => stop rolling out sample i
        active_mask = torch.ones(num_samples, dtype=torch.bool, device=self.device)
        jsd_threshold = 1.0  # JSD threshold for uncertainty (0.5 is too low, 1.0 is more reasonable)

        for t in range(max_length_traj):

            # If everything is "inactive", exit early
            if not active_mask.any():
                break

            actions = self.sac_agent.select_action(initial_states)
            actions = torch.FloatTensor(actions).to(self.device)

            # Normalize the inputs. Ensemble predictions: shape [ensemble_size, batch_size, next_state_dim+1]
            model_input = torch.cat([initial_states, actions], dim=1)
            model_input = self.ensemble_model.input_normalizer.normalize(model_input)
            mean_preds, logvar_preds = self.ensemble_model(model_input)
            all_preds_mean, all_preds_logvar = torch.stack(mean_preds, dim=0), torch.stack(logvar_preds, dim=0)

            # Next state is sampled from the ensemble (only top elite models)
            elite_tensor = torch.tensor(self.elite_idxs, device=self.device)
            if len(elite_tensor) == 0:
                elite_tensor = torch.arange(self.ensemble_model.ensemble_size, device=self.device)
            head_idx = elite_tensor[torch.randint(len(elite_tensor), (num_samples,), device=self.device)]
            mean_pred = all_preds_mean[head_idx, torch.arange(num_samples)]
            logvar_pred = all_preds_logvar[head_idx, torch.arange(num_samples)]

            # Add model noise to the sampled mean predictions
            std_pred = torch.exp(0.5 * logvar_pred).clamp(max=0.25)  # σ ≤ 0.5 (normalised)
            noise = torch.randn_like(mean_pred) * std_pred
            mean_pred = mean_pred + noise

            # De-normalize the outputs
            denorm_mean_pred = self.ensemble_model.output_normalizer.denormalize(mean_pred)
            delta_next_state = denorm_mean_pred[:, :-1]
            next_states = initial_states + delta_next_state  # Add delta to the current state

            rewards = denorm_mean_pred[:, -1].unsqueeze(1)
            dones = torch.zeros_like(rewards)

            # Compute uncertainty as disagreement across ensemble (JSD)
            ns_jsd = compute_jsd(all_preds_mean, torch.exp(all_preds_logvar))

            # Compute the mask and update for samples that are still active but exceed the threshold
            push_mask = active_mask & (ns_jsd <= jsd_threshold)
            active_mask[active_mask & (ns_jsd > jsd_threshold)] = False

            # Check if any sample should be pushed in this rollout step
            if push_mask.any():
                # Get the indices of the samples to push
                indices_to_push = push_mask.nonzero(as_tuple=True)[0]

                states_to_push = initial_states[indices_to_push].cpu().numpy()
                actions_to_push = actions[indices_to_push].cpu().numpy()
                rewards_to_push = rewards[indices_to_push].detach().cpu().numpy()
                next_states_to_push = next_states[indices_to_push].detach().cpu().numpy()
                dones_to_push = dones[indices_to_push].cpu().numpy()

                # Push the samples to the imaginary buffer
                self.imaginary_buffer.push_batch(
                    states_to_push,
                    actions_to_push,
                    rewards_to_push,
                    next_states_to_push,
                    dones_to_push
                )

            initial_states = next_states.detach()
        # End of imaginary rollouts

    def get_final_buffer(self, proportion_real=0.05):

        # # For Mujoco
        # if self.env_type == "gym_mujoco":
        #     if len(self.real_buffer) < (self.warmup_steps + 1_000):
        #         proportion_real = 0.5
        #     else:
        #         proportion_real = 0.05
        # else:
        #     if len(self.real_buffer) < self.warmup_steps:
        #         proportion_real = 0.5
        #     else:
        #         proportion_real = 0.05

        # Function that creates a new ReplayBuffer with the data from the real buffer and imaginary buffer.
        if self.model_based:
            # Sample 5% of the real buffer
            real_batch = random.sample(self.real_buffer.buffer, int(proportion_real * self.update_size))

            # Sample 95% of the imaginary buffer
            imaginary_size = min(int((1 - proportion_real) * self.update_size), len(self.imaginary_buffer))
            imaginary_batch = random.sample(self.imaginary_buffer.buffer, imaginary_size)

            # Concatenate the two batches
            final_batch = real_batch + imaginary_batch
            s, a, r, ns, d = map(np.stack, zip(*final_batch))

        else:
            # Sample all the real buffer
            real_batch = random.sample(self.real_buffer.buffer, self.update_size)
            s, a, r, ns, d = map(np.stack, zip(*real_batch))

        return torch.as_tensor(s, device=self.device, dtype=torch.float32), \
            torch.as_tensor(a, device=self.device, dtype=torch.float32), \
            torch.as_tensor(r, device=self.device, dtype=torch.float32).unsqueeze(-1), \
            torch.as_tensor(ns, device=self.device, dtype=torch.float32), \
            torch.as_tensor(d, device=self.device, dtype=torch.float32).unsqueeze(-1)

    def train(self, num_episodes: int = 200, max_steps: int = 1_000):
        if self.log_wandb:
            project_name = self.env.unwrapped.spec.id if self.env.unwrapped.spec != None else 'SimpleCausal_Multi'
            wandb.init(project=project_name, sync_tensorboard=False,
                       name=f"{self.alg_name}_SAC_seed_{self.seed}_time_{time.time()}",
                       config=self.__dict__, group=self.alg_name, dir='/tmp')

        episode = 0
        self.max_steps = max_steps
        target_steps = num_episodes * max_steps

        while self.total_steps < target_steps:

            if self.env_type == "gym_robotics":
                state_dict, _ = self.env.reset()
                state = flatten_obs(state_dict)
            else:
                state, _ = self.env.reset()

            episode_reward, episode_steps = 0, 0

            # 1) First chunk: roll an episode with the real environment and populate the real buffer
            for step in range(max_steps):

                if self.total_steps > self.warmup_steps:
                    action = self.sac_agent.select_action(state).flatten()
                else:
                    action = self.env.action_space.sample()

                if self.env_type == "gym_robotics":
                    next_state_dict, reward, done, truncated, _ = self.env.step(action)
                    next_state = flatten_obs(next_state_dict)
                    terminal = done or truncated

                    # For HER
                    self.real_buffer.push_transition(state,  action,reward, next_state,  terminal,
                                                     state_dict["achieved_goal"], next_state_dict["achieved_goal"],
                                                     state_dict["desired_goal"])

                    # Shift state_dict to next_state_dict
                    state_dict = next_state_dict
                else:
                    next_state, reward, done, truncated, _ = self.env.step(action)
                    terminal = done or truncated
                    self.real_buffer.push(state, action, reward, next_state, terminal)

                # Set state to next_state and increment the episode reward and steps
                state = next_state
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1

                # 2) Every self.eval_freq steps, evaluate the policy on deterministic actions
                if self.total_steps % self.eval_freq == 0 and self.total_steps > 0:
                    self._evaluate_policy()

                # 3) Update normalizers every self.model_train_freq steps in model-based training
                if self.total_steps % self.model_train_freq == 0 and self.model_based:

                    batch = self.real_buffer.buffer[-self.model_train_freq:]
                    states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))

                    states = torch.FloatTensor(states).to(self.device)
                    actions = torch.FloatTensor(actions).to(self.device)
                    next_states = torch.FloatTensor(next_states).to(self.device)
                    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)

                    # Compute delta_state
                    delta_states = next_states - states

                    # Concatenate inputs and outputs
                    model_inputs = torch.cat([states, actions], dim=-1)
                    targets = torch.cat([delta_states, rewards], dim=-1)

                    # Normalize training data
                    self.ensemble_model.input_normalizer.update(model_inputs)
                    self.ensemble_model.output_normalizer.update(targets)

                    # Train the model
                    if len(self.real_buffer) > (self.batch_size):
                        # Train the model
                        model_loss = self.update_model(self.batch_size)

                # All this after the warm-up period of warmup_steps steps used to populate the real buffer
                if self.total_steps >= self.warmup_steps:

                    # 4) Train the dynamics model
                    if self.total_steps % self.model_train_freq == 0 and self.model_based:
                        self.imaginary_rollout()  # Generate imaginary rollouts

                    # 5) Train the SAC agent
                    if self.total_steps % self.sac_train_freq == 0 and len(self.real_buffer) > self.batch_size:
                        for _ in range(self.agent_steps):
                            s, a, r, ns, d = self.get_final_buffer()
                            critic_loss, actor_loss, alpha_loss = self.sac_agent.update(s, a, r, ns, d)

                # Break if episode is done or if the maximum number of steps is reached
                if terminal or self.total_steps >= target_steps:
                    break

            episode += 1

            # 6) Logging and Printing
            if self.log_wandb:
                wandb.log({
                    "Train/Episode Reward": episode_reward,
                    "Train/Episode Length": episode_steps,
                    "Train/Global Step": self.total_steps,
                    "Train/Model Loss": model_loss if 'model_loss' in locals() else 0,
                    "Train/Critic Loss": critic_loss if 'critic_loss' in locals() else 0,
                    "Train/Actor Loss": actor_loss if 'actor_loss' in locals() else 0,
                    "Train/Alpha Loss": alpha_loss if 'alpha_loss' in locals() else 0
                })

            if episode % 1 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Steps: {episode_steps}")
                print("Model Loss: ", model_loss if 'model_loss' in locals() else 0)

        wandb.finish()  # close the wandb run

    def save_agent(self, base_dir: str = 'trained_agents/'):

        # Save the entire SAC agent
        filename = base_dir + f"{self.alg_name}_seed_{self.seed}"
        torch.save(self.sac_agent, filename)

    @torch.no_grad()
    def _evaluate_policy(self, eval_episodes: int = 10) -> float:

        self.sac_agent.actor.eval()

        avg_return = 0.0
        for _ in range(eval_episodes):
            if self.env_type == "gym_robotics":
                s_dict, _ = self.env.reset()
                state = flatten_obs(s_dict)
            else:
                state, _ = self.env.reset()

            ep_ret = 0.0
            for _ in range(self.max_steps):
                if self.total_steps > self.warmup_steps:
                    action = self.sac_agent.select_action(state, deterministic=True).flatten()
                else:
                    action = self.env.action_space.sample()

                if self.env_type == "gym_robotics":
                    n_dict, r, d, t, _ = self.env.step(action)
                    state = flatten_obs(n_dict)
                else:
                    state, r, d, t, _ = self.env.step(action)

                ep_ret += r
                if d or t:
                    break

            avg_return += ep_ret
        avg_return /= eval_episodes

        # restore the actor to training mode
        self.sac_agent.actor.train()

        # Log the average return
        if self.log_wandb:
            wandb.log({"Eval/Average Return": avg_return,
                       "Eval/Global Step": self.total_steps})
        print(f"Eval Average Return: {avg_return:.2f}")

        return avg_return
