import torch
import numpy as np

# from envs.causal_env import SimpleCausalEnv
# from algs.cmbpo_sac import set_device, CMBPO_SAC
from algs.cmbpo_sac_ensemble import set_device, CMBPO_SAC

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
    env = gym.make('HalfCheetah-v4')

    agent = CMBPO_SAC(env, seed, device, log_wandb=log_wandb, model_based=model_based, pure_imaginary=False)
    agent.train(num_episodes=200, max_steps=1_000)

    # Save the model
    agent.save_agent(base_dir='trained_agents/')
    # meaning that the model has selected in all the variables regardless of the underlying causal model.

    print("Training done!")
