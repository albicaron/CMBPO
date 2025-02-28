import gym
from envs.causal_env import SimpleCausalEnv


# Example usage
if __name__ == "__main__":
    env = SimpleCausalEnv()
    obs = env.reset()
    for _ in range(10):
        action = env.action_space.sample()  # Replace with your policy
        obs, reward, done, info = env.step(action)
        env.render()
        print(f"State: {obs}, Action: {action}, Reward: {reward}")
        if done:
            break

    env.close()
