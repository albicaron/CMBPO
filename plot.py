import wandb
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Make sure you are logged into W&B:
#  wandb login

# Initialize W&B API
api = wandb.Api()

# Fetch runs from your project
runs = api.runs("albicaron93/SimpleCausalEnv_v1")

# Dictionary to store rewards for each group
all_rewards = {
    'Random': [],
    'SAC': [],
    'MBPO_SAC': [],
    'C_MBPO_SAC': []
}

# The config key that identifies which group the run belongs to.
# Adjust based on how your runs' configs are structured.
group_key = "alg_name"

# ---------------------
# 1) Download data from W&B and group runs
# ---------------------
for run in runs:

    # Check if this run is in one of our 4 groups
    if group_key in run.config:
        group_value = run.config[group_key]
        # Only consider if the group is one of the 4 we care about
        if group_value in all_rewards:
            # Download the full history for this run
            history = run.history(keys=["Train/Episode Reward"])
            rewards = history["Train/Episode Reward"].dropna().to_numpy()
            all_rewards[group_value].append(rewards)

# ---------------------
# 2) Process, smooth, and plot for each group
# ---------------------
plt.figure(figsize=(5.5, 4.5), dpi=300)

colors = {
    'Random': '#A1A9AD',
    'SAC': '#7D54B2',
    'MBPO_SAC': '#008038',
    'C_MBPO_SAC': '#E57439'
}

sigma = 2.0  # adjust this for more/less smoothing

for group_name, rewards_list in all_rewards.items():
    if not rewards_list:
        print(f"No data found for group: {group_name}")
        continue

    # 2a) Align by clipping to the same min length
    min_len = min(len(r) for r in rewards_list)
    clipped_rewards = [r[:min_len] for r in rewards_list]

    # 2b) Convert to array: shape [num_runs, min_len]
    group_rewards_arr = np.vstack(clipped_rewards)

    # 2c) Apply Gaussian smoothing to each run
    smoothed = np.array([
        gaussian_filter1d(run_rewards, sigma=sigma) for run_rewards in group_rewards_arr
    ])

    # 2d) Compute mean and 95% confidence interval
    mean_reward = np.mean(smoothed, axis=0)
    std_reward = np.std(smoothed, axis=0)
    n = smoothed.shape[0]
    sem = std_reward / np.sqrt(n)
    ci = 1.96 * sem  # ~95% confidence interval

    upper_bound = mean_reward + ci
    lower_bound = mean_reward - ci

    # 2e) Plot
    x_vals = np.arange(min_len) * 200  # 200 steps per episode
    plt.plot(x_vals, mean_reward, label=group_name.replace("_", "-"), color=colors[group_name])
    plt.fill_between(x_vals, lower_bound, upper_bound, alpha=0.2, color=colors[group_name])

# ---------------------
# 3) Final plot styling
# ---------------------
# plt.title("Episodic Reward Comparison (Gaussian Smoothed) — 4 Groups")
plt.xlabel("Step")
plt.ylabel("Episodic Reward")
plt.legend()
plt.grid(True)

# After plotting, define custom tick positions and labels:
max_steps = 200 * min_len  # 200 steps per episode
tick_positions = np.arange(0, max_steps+1, 2000)  # e.g., 0, 2000, 4000, 6000, ...
tick_labels = [f"{int(tp//1000)}k" for tp in tick_positions]
plt.xticks(tick_positions, tick_labels)

plt.tight_layout()
plt.savefig("Reward_Plot.pdf", dpi=300)

