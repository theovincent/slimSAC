import gymnasium as gym
import numpy as np


class Mujoco:
    def __init__(self, task_name, seed, render_mode=None):
        self.rng = np.random.default_rng(seed)
        self.env = gym.make(task_name + "-v5", render_mode=render_mode)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.env.action_space.seed(seed)
        self.action_scaler = float(self.env.env.action_space.high_repr)

    # Called when stored in the replay buffer
    @property
    def observation(self) -> np.ndarray:
        return np.copy(self.state)

    def reset(self):
        self.state, _ = self.env.reset(seed=int(self.rng.integers(0, 1_000_000)))
        self.n_steps = 0

    def step(self, action):
        self.state, reward, absorbing, _, _ = self.env.step(self.action_scaler * action)
        self.n_steps += 1

        return reward, absorbing

    def random_action(self):
        return self.env.action_space.sample()
