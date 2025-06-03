import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SimpleCausalEnv(gym.Env):
    """
    A simple RL environment with two state variables S1 and S2.
    Dynamics:
        X_{t+1) = 0.3 * X_t + 0.8 A_t - 0.2 * cos(0.5 * X_t) + e1_t  # Depends on: X_t, A_t
        Z_{t+1} = X_{t+1} + e2_t  # Depends on: X_{t+1}
        R_t = 0.5 * X_t + e3_t - 0.01  # Reward depends on: X_t
    Actions are continuous in [-1, 1]
    """

    def __init__(self, noise_std=.01, shifted=False):
        super(SimpleCausalEnv, self).__init__()

        self.shifted = shifted

        # Define action and observation space
        # Continuous action space: A_t ∈ [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

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
        self.power = 0.05
        self.friction = 0.005
        self.slope = 0.05

        self.beta = 0.03

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        """
        # Initialize S1 and S2 to random values, e.g., between -1 and 1
        self.state = np.random.uniform(low=-1.0, high=-1.0, size=(2,)).astype(np.float32)
        return self.state, {}

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        X_t, Z_t = self.state
        A_t = action[0]

        # Ensure action is within the valid range
        A_t = np.clip(A_t, self.action_space.low[0], self.action_space.high[0])

        # Define the non-linear functions
        f_x = X_t + self.power * A_t - self.friction * np.cos(self.slope * X_t)
        f_z = 0.0*f_x if self.shifted else f_x
        # f_z = -f_x if self.shifted else f_x  # Larger shift

        # Generate noise
        e_x_t = np.random.normal(0, self.noise_std)
        e_z_t = np.random.normal(0, self.noise_std)  # Shift in the marginal distribution of Z
        e_r_t = np.random.normal(0, self.noise_std)

        # Compute next state by adding the noise
        next_X_t = f_x + e_x_t
        next_Z_t = f_z + e_z_t

        # Compute reward
        R_t = 0.1*(1.0 - X_t**2) + e_r_t

        # Update the state
        self.state = np.array([next_X_t, next_Z_t], dtype=np.float32)

        # Clip the state to the valid range
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)

        # Define termination condition (optional)
        done = False

        return self.state, R_t, done, False, {}

    def get_adj_matrix(self):
        """
        Return the adjacency matrix of the causal graph underlying the environment dynamics and rewards of the
        (s,a,ns,r) tuples. Which means this is a 6x6 matrix where the first 2 rows and columns correspond to the
        state variables S1 and S2, the next 2 rows and columns correspond to the action variable A and the next 2 rows
        and columns correspond to the next state variables S1' and S2' and the reward variable R. No self-loops
        :return:
        """
        adj_matrix = np.zeros((6, 6))
        adj_matrix[0, 1] = 1  # S1 -> S2
        adj_matrix[0, 3] = 1  # S1 -> S1'
        adj_matrix[0, 5] = 1  # S1 -> R
        adj_matrix[2, 3] = 1  # A -> S1'
        # adj_matrix[3, 4] = 1  # S1' -> S2'

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
    env = SimpleCausalEnv()
    state = env.reset()
    print("\n\nNon-shifted environment:")
    print("Initial state:", state)

    for i in range(200):
        action = np.array([1.0], dtype=np.float32)
        next_state, reward, done, _ = env.step(action)
        if i % 10 == 0:
            print("Action:", action, "Next state:", next_state, "Reward:", reward)
        if done:
            break

    # Test the shifted environment
    env = SimpleCausalEnv(shifted=True)
    state = env.reset()
    print("\n\nShifted environment:")
    print("Initial state:", state)

    for i in range(200):
        action = np.array([1.0], dtype=np.float32)
        next_state, reward, done, _ = env.step(action)
        if i % 10 == 0:
            print("Action:", action, "Next state:", next_state, "Reward:", reward)
        if done:
            break
