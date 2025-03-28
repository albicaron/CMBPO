import torch

from envs.causal_env import SimpleCausalEnv
from algs.mbpo_sac import MBPO_SAC


if __name__ == "__main__":

    # Initialize environment
    seed = 2
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_wandb = False
    model_based = True
    env = SimpleCausalEnv(shifted=False)

    agent = MBPO_SAC(env, seed, device, log_wandb=log_wandb, model_based=model_based, pure_imaginary=False)
    agent.train(num_episodes=100, max_steps=200)

    # Save the model
    agent.save_agent(base_dir='trained_agents/')
    # meaning that the model has selected in all the variables regardless of the underlying causal model.

    print("Training done!")
