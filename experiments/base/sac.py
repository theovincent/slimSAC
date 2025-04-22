import time
import jax
import numpy as np
from collections import deque
from tqdm import trange

from experiments.base.utils import save_data
from slimsac.networks.sac import SAC
from slimsac.sample_collection.replay_buffer import ReplayBuffer
from slimsac.sample_collection.utils import collect_single_sample, evaluate_policy


def train(
    key: jax.random.PRNGKey,
    p: dict,
    agent: SAC,
    env,
    eval_env,
    rb: ReplayBuffer,
    eval_freq: int = 10000,
    log_interval: int = 2500,
):
    env.reset()
    rolling_returns = deque(maxlen=100)
    rolling_lengths = deque(maxlen=100)

    episode_return = 0
    episode_length = 0
    start_time = time.time()

    for n_training_steps in trange(p["n_samples"], desc="Training"):
        key, update_key, exploration_key = jax.random.split(key, 3)
        log_dict = {"global_step": n_training_steps}
        reward, has_reset = collect_single_sample(exploration_key, env, agent, rb, p, n_training_steps)

        episode_return += reward
        episode_length += 1

        if has_reset:
            rolling_returns.append(episode_return)
            rolling_lengths.append(episode_length)
            episode_return = 0
            episode_length = 0

        log = n_training_steps % log_interval == 0 and n_training_steps > 0

        if n_training_steps > p["learning_starts"]:
            log_dict.update(agent.update_online_params(n_training_steps, rb, update_key))

            if n_training_steps % eval_freq == 0:
                log_dict["eval/mean_reward"], log_dict["eval/mean_ep_length"] = evaluate_policy(eval_env, agent, p)
                log = True

        if log:
            fps = n_training_steps / (time.time() - start_time)
            log_dict["rollout/ep_rew_mean"] = np.mean(rolling_returns)
            log_dict["rollout/ep_len_mean"] = np.mean(rolling_lengths)
            log_dict["time/fps"] = fps
            p["wandb"].log(log_dict)

    save_data(p, list(rolling_returns), list(rolling_lengths), agent.get_model())
