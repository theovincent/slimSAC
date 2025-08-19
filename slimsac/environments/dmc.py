import gymnasium as gym
from dm_control import suite
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
from shimmy import DmControlCompatibilityV0 as DmControltoGymnasium


def make_dmc_env(
    env_name: str,
    seed: int,
    flatten: bool = True,
) -> gym.Env:
    domain_name, task_name = env_name.split("-")
    env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs={"random": seed},
    )
    env = DmControltoGymnasium(env, render_mode="rgb_array")
    if flatten and isinstance(env.observation_space, spaces.Dict):
        env = FlattenObservation(env)

    return env


class DMC:
    def __init__(self, task_name, seed):
        self.rng = np.random.default_rng(seed)
        self.env = make_dmc_env(task_name, seed)
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
