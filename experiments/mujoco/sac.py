import os
import sys

import numpy as np
import random

import jax

from experiments.base.sac import train
from experiments.base.utils import prepare_logs
from slimsac.environments.mujoco import Mujoco
from slimsac.networks.sac import SAC
from slimsac.sample_collection.replay_buffer import ReplayBuffer
from slimsac.sample_collection.samplers import UniformSamplingDistribution


def run(argvs=sys.argv[1:]):
    env_name, algo_name = os.path.abspath(__file__).split("/")[-2], os.path.abspath(__file__).split("/")[-1][:-3]
    p = prepare_logs(env_name, algo_name, argvs)

    random.seed(p["seed"])
    np.random.seed(p["seed"])

    q_key, train_key = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = Mujoco(p["experiment_name"].split("_")[-1], p["seed"])
    eval_env = Mujoco(p["experiment_name"].split("_")[-1], p["seed"])

    rb = ReplayBuffer(
        sampling_distribution=UniformSamplingDistribution(p["seed"]),
        batch_size=p["batch_size"],
        max_capacity=p["replay_buffer_capacity"],
        stack_size=1,
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
        compress=False,
    )
    agent = SAC(
        q_key,
        env.observation_dim,
        env.action_dim,
        learning_rate=p["learning_rate"],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        data_to_update=p["data_to_update"],
        tau=p["tau"],
        features_pi=p["features_pi"],
        features_qf=p["features_qf"],
    )
    train(train_key, p, agent, env, eval_env, rb)


if __name__ == "__main__":
    run()
