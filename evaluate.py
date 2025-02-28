# This script should load the saved agents in "trained_agents/" and evaluate them in the environment. The evaluation
# should be done by running the agent for 20 episodes and reporting the average reward plus the 95% confidence
# interval. The evaluation should be done for the following agents:
#
# - SAC
# - MBPO_SAC
# - Causal_MBPO_SAC (to be implemented)

import torch
import numpy as np
import matplotlib.pyplot as plt

from envs.causal_env import SimpleCausalEnv


def evaluate_agent(env, agent, num_episodes=100, max_steps=200):

    episodic_rewards = []
    for episode in range(num_episodes):

        # Set seed
        np.random.seed(episode*110)
        torch.manual_seed(episode*110)

        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0

        for step in range(max_steps):

            # If agent actor input layer has only one neuron, pass only the first state
            if agent.actor.net[0].in_features == 1:
                # Pass only the first state to the SAC agent
                action = agent.select_action(state.reshape(1, -1)[:, :1]).flatten()  # Only S1
            else:
                action = agent.select_action(state).flatten()
            next_state, reward, done, truncated, _ = env.step(action)

            episode_reward += reward
            episode_steps += 1

            state = next_state
            if done or truncated:
                break

        episodic_rewards.append(episode_reward)

    return episodic_rewards


if __name__ == "__main__":

    #### 1) In-Distribution Environment
    env = SimpleCausalEnv()

    # Load the agents
    sac_agent = torch.load("trained_agents/SAC_seed_13")
    mbpo_sac_agent = torch.load("trained_agents/MBPO_SAC_seed_13")
    causal_mbpo_sac_agent = torch.load("trained_agents/C_MBPO_SAC_seed_13")

    # Evaluate the agents
    sac_reward = evaluate_agent(env, sac_agent)
    mbpo_sac_reward = evaluate_agent(env, mbpo_sac_agent)
    causal_mbpo_sac_reward = evaluate_agent(env, causal_mbpo_sac_agent)

    # Save csv file with mean and 95% confidence interval of the episodic rewards
    mean_sac = np.mean(sac_reward)
    mean_mbpo_sac = np.mean(mbpo_sac_reward)
    mean_causal_mbpo_sac = np.mean(causal_mbpo_sac_reward)

    std_sac = np.std(sac_reward)
    std_mbpo_sac = np.std(mbpo_sac_reward)
    std_causal_mbpo_sac = np.std(causal_mbpo_sac_reward)

    # Print rounding until 2 decimal places
    print("\n\n****** In-Distribution Environment:")
    print(f"SAC: {mean_sac:.2f} ± {1.96 * std_sac / np.sqrt(len(sac_reward)):.2f}")
    print(f"MBPO_SAC: {mean_mbpo_sac:.2f} ± {1.96 * std_mbpo_sac / np.sqrt(len(mbpo_sac_reward)):.2f}")
    print(f"Causal_MBPO_SAC: {mean_causal_mbpo_sac:.2f} ± {1.96 * std_causal_mbpo_sac / np.sqrt(len(causal_mbpo_sac_reward)):.2f}")


    # 2) Out-of-Distribution Environment
    shifted_env = SimpleCausalEnv(shifted=True)

    # Evaluate the agents
    sac_reward = evaluate_agent(shifted_env, sac_agent)
    mbpo_sac_reward = evaluate_agent(shifted_env, mbpo_sac_agent)
    causal_mbpo_sac_reward = evaluate_agent(shifted_env, causal_mbpo_sac_agent)

    # Save csv file with mean and 95% confidence interval of the episodic rewards
    mean_sac = np.mean(sac_reward)
    mean_mbpo_sac = np.mean(mbpo_sac_reward)
    mean_causal_mbpo_sac = np.mean(causal_mbpo_sac_reward)

    std_sac = np.std(sac_reward)
    std_mbpo_sac = np.std(mbpo_sac_reward)
    std_causal_mbpo_sac = np.std(causal_mbpo_sac_reward)

    # Print rounding until 2 decimal places
    print("\n\n****** Out-of-Distribution Environment:")
    print(f"SAC: {mean_sac:.2f} ± {1.96 * std_sac / np.sqrt(len(sac_reward)):.2f}")
    print(f"MBPO_SAC: {mean_mbpo_sac:.2f} ± {1.96 * std_mbpo_sac / np.sqrt(len(mbpo_sac_reward)):.2f}")
    print(f"Causal_MBPO_SAC: {mean_causal_mbpo_sac:.2f} ± {1.96 * std_causal_mbpo_sac / np.sqrt(len(causal_mbpo_sac_reward)):.2f}")

