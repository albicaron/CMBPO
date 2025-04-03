import gym
from gym import spaces
import numpy as np


class SimpleCausal_Multi(gym.Env):
    """
    A simple causal RL environment with two state variables S1 and S2, and actions A_1 and A_2
    Dynamics:
        S1_{t+1) = S1_t + 0.8*A_1 + e1_t  # Depends on: S1_t, A_1
        S2_{t+1} = S2_t + 0.8*A_2 + e2_t  # Depends on: S2_t, A_2

        R_t = 0.5 * S2_t + e3_t - 0.01  # Reward depends on: S1_t
    Actions are continuous in [-1, 1]
    """

    def __init__(self, noise_std=.01, shifted=False):
        super(SimpleCausal_Multi, self).__init__()

        self.shifted = shifted

        # Define action and observation space
        # Continuous action space: A1_t, A2_t ∈ [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: S1_t and S2_t. Setting reasonable bounds for the state variables
        # high = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32)
        # low = -high
        high = np.array([1.0, 1.0], dtype=np.float32)
        low = np.array([-1.0, -1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Initialize state
        self.state = None

        # Noise parameters
        self.noise_std = noise_std  # Standard deviation of the noise

        # Causal strength parameters
        self.power = 0.01
        self.friction = 0.005
        self.slope = 0.05

        self.beta = 0.03

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        """
        # Initialize S1 and S2 to random values, e.g., between -1 and 1
        self.state = np.random.uniform(low=-1.0, high=-1.0, size=(2,)).astype(np.float32)
        return self.state

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        S1_t, S2_t = self.state
        A1_t, A2_t = action

        # Ensure action is within the valid range
        A1_t = np.clip(A1_t, self.action_space.low[0], self.action_space.high[0])
        A2_t = np.clip(A2_t, self.action_space.low[1], self.action_space.high[1])

        # Define the non-linear functions
        f_s1 = S1_t + self.power * A1_t - self.friction * np.cos(self.slope * S1_t)
        f_s2 = S2_t + self.power * A2_t - self.friction * np.sin(self.slope * S2_t)

        # Generate noise
        e_s1_t = np.random.normal(0, self.noise_std)
        e_s2_t = np.random.normal(0, self.noise_std)  # Shift in the marginal distribution of Z
        e_r_t = np.random.normal(0, self.noise_std)

        # Compute next state by adding the noise
        next_S1_t = f_s1 + e_s1_t
        next_S2_t = f_s2 + e_s2_t

        # Compute reward
        R_t = 0.1*(1.0 - S2_t**2) + e_r_t

        # Update the state
        self.state = np.array([next_S1_t, next_S2_t], dtype=np.float32)

        # Clip the state to the valid range
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)

        # Define termination condition (optional)
        done = False

        return self.state, R_t, done, False

    def get_adj_matrix(self):
        """
        Return the adjacency matrix of the causal graph underlying the environment dynamics and rewards of the
        (s,a,ns,r) tuples. Which means this is a 7x7 matrix with the following structure:
        :return:
        """
        adj_matrix = np.zeros((7, 7))
        adj_matrix[0, 4] = 1  # S1 -> S1'
        adj_matrix[1, 5] = 1  # S2 -> S2'
        adj_matrix[1, 6] = 1  # S2 -> R
        adj_matrix[2, 4] = 1  # A1 -> S1'
        adj_matrix[3, 5] = 1  # A2 -> S2'

        return adj_matrix

    def render(self, mode='human'):
        """
        Render the environment to the screen (optional).
        """
        pass

    def close(self):
        """
        Perform any necessary cleanup (optional).
        """
        pass


if __name__ == "__main__":
    # Test the non-shifted environment
    env = SimpleCausal_Multi()
    state = env.reset()
    print("\n\nNon-shifted environment:")
    print("Initial state:", state)

    for i in range(200):
        action = np.array([1.0, 1.0], dtype=np.float32)
        next_state, reward, done, _ = env.step(action)
        if i % 10 == 0:
            print("Action:", action, "Next state:", next_state, "Reward:", reward)
        if done:
            break

    # Test the shifted environment
    # TODO
