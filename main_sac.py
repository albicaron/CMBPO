import torch
import numpy as np

# from envs.causal_env import SimpleCausalEnv
# from envs.causal_env_multiaction import SimpleCausal_Multi
from algs.mbpo_sac import set_device, MBPO_SAC

import gym


if __name__ == "__main__":

    # Initialize environment
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = set_device()

    log_wandb = True
    model_based = True
    # env = SimpleCausal_Multi(shifted=False)
    # env = gym.make('Hopper-v4', terminate_when_unhealthy=False)
    env = gym.make('HalfCheetah-v4')

    agent = MBPO_SAC(env, seed, device, log_wandb=log_wandb, model_based=model_based)
    agent.train(num_episodes=200, max_steps=1_000)

    # Save the model
    agent.save_agent(base_dir='trained_agents/')
    # meaning that the model has selected in all the variables regardless of the underlying causal model.

    print("Training done!")
