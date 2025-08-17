import time
import jax
import numpy as np
from collections import deque
from tqdm import trange

from experiments.base.utils import save_data
from slimsac.algorithms.sac import SAC
from slimsac.sample_collection.replay_buffer import ReplayBuffer
from slimsac.sample_collection.utils import collect_single_sample, evaluate_policy


def train(key: jax.random.PRNGKey, p: dict, agent: SAC, env, eval_env, rb: ReplayBuffer):
    env.reset()
    rolling_returns = deque(maxlen=100)
    rolling_lengths = deque(maxlen=100)

    episode_return = 0
    episode_length = 0
    start_time = time.time()

    for n_training_steps in trange(1, p["n_samples"] + 1, desc="Training", miniters=10000, maxinterval=1000):
        key, update_key, exploration_key = jax.random.split(key, 3)

        reward, has_reset = collect_single_sample(exploration_key, env, agent, rb, p, n_training_steps)
        episode_return += reward
        episode_length += 1

        if has_reset:
            rolling_returns.append(episode_return)
            rolling_lengths.append(episode_length)
            episode_return = 0
            episode_length = 0

        if n_training_steps >= p["n_initial_samples"]:
            agent.update_online_params(rb, update_key)

            if n_training_steps % 10_000 == 0:
                eval_mean_reward, eval_mean_ep_length = evaluate_policy(eval_env, agent, p)
                p["wandb"].log(
                    {
                        "n_training_steps": n_training_steps,
                        "performances/eval_reward": eval_mean_reward,
                        "performances/eval_ep_length": eval_mean_ep_length,
                    }
                )

            if n_training_steps % 2_500 == 0:
                log_dict = {
                    "n_training_steps": n_training_steps,
                    "performances/train_reward": np.mean(rolling_returns),
                    "performances/train_ep_length": np.mean(rolling_lengths),
                    "fps": n_training_steps / (time.time() - start_time),
                }
                log_dict.update(agent.get_logs())

                p["wandb"].log(log_dict)

    save_data(p, list(rolling_returns), list(rolling_lengths), agent.get_model())
