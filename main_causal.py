import torch
import matplotlib.pyplot as plt

from envs.causal_env import SimpleCausalEnv
from algs.causal_mbpo_sac import C_MBPO_SAC

import wandb
import gym

if __name__ == "__main__":

    # Initialize environment
    seed = 2
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_wandb = True
    model_based = True
    env = SimpleCausalEnv(shifted=False)

    agent = C_MBPO_SAC(env, seed, device, log_wandb=log_wandb, model_based=model_based, pure_imaginary=False)
    agent.train(num_episodes=100, max_steps=200)

    # Save the model
    agent.save_agent(base_dir='trained_agents/')

    print("Training done!")
