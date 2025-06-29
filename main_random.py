import torch
# from envs.causal_env import SimpleCausalEnv
from envs.causal_env_multiaction import SimpleCausal_Multi
import wandb
import time

# Just run a random policy on the environment
if __name__ == "__main__":

    seed = 1
    num_episodes = 100
    max_steps = 200

    project_name = "SimpleCausalMulti_v2"
    wandb.init(project=project_name, sync_tensorboard=False,
               name=f"RANDOM_seed_{seed}_time_{time.time()}",
               group="Random", dir='/tmp', config={"alg_name": "Random"})

    # env = SimpleCausalEnv(shifted=False)
    env = SimpleCausal_Multi(shifted=False)
    global_step = 0
    for episode in range(num_episodes):

        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0

        for step in range(max_steps):
            action = env.action_space.sample()
            next_state, reward, done, truncated, _ = env.step(action)

            episode_reward += reward
            episode_steps += 1
            global_step += 1

            state = next_state
            if done:
                break

        wandb.log({
            "Train/Episode Reward": episode_reward,
            "Train/Episode Length": episode_steps,
            "Train/Global Step": global_step,
            "Eval/Average Return": episode_reward,  # For consistency, log the same reward in eval
            "Eval/Global Step": global_step,
        })

        if episode % 1 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Steps: {episode_steps}")

    wandb.finish()



