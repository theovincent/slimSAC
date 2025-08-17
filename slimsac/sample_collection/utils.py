from slimsac.sample_collection.replay_buffer import ReplayBuffer, TransitionElement


def collect_single_sample(key, env, agent, rb: ReplayBuffer, p, n_training_steps: int):
    if n_training_steps <= p["n_initial_samples"]:
        action = env.random_action()
    else:
        action = agent.sample_action(env.state, agent.actor_params, key)

    obs = env.observation
    reward, absorbing = env.step(action)

    episode_end = absorbing or env.n_steps >= p["horizon"]
    rb.add(
        TransitionElement(
            observation=obs,
            action=action,
            reward=reward if rb._clipping is None else rb._clipping(reward),
            is_terminal=absorbing,
            episode_end=episode_end,
        )
    )

    if episode_end:
        env.reset()

    return reward, episode_end


def evaluate_policy(env, agent, p):
    env.reset()
    episode_end = False
    total_reward = 0.0
    length = 0

    while not episode_end and length < p["horizon"]:
        action = agent.sample_action(env.state, agent.actor_params, None)
        reward, absorbing = env.step(action)
        episode_end = absorbing or env.n_steps >= p["horizon"]
        total_reward += reward
        length += 1

    return total_reward, length
