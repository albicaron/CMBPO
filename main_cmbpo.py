import torch

# from envs.causal_env import SimpleCausalEnv
from envs.causal_env_multiaction import SimpleCausal_Multi
from algs.cmbpo_sac import CMBPO_SAC

import gym


if __name__ == "__main__":

    # Initialize environment
    seed = 0
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_wandb = False
    model_based = True
    env = SimpleCausal_Multi(shifted=False)
    # env = gym.make('HalfCheetah-v4')

    agent = CMBPO_SAC(env, seed, device, log_wandb=log_wandb, model_based=model_based, pure_imaginary=False)
    agent.train(num_episodes=100, max_steps=200)

    # Save the model
    agent.save_agent(base_dir='trained_agents/')
    # meaning that the model has selected in all the variables regardless of the underlying causal model.

    print("Training done!")
