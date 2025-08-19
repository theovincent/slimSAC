from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp
import jax


class CriticNet(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.squeeze(state)
        x = jnp.concatenate([x, action], -1)
        for n_units in self.features:
            x = nn.Dense(n_units)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class ActorNet(nn.Module):
    features: Sequence[int]
    action_dim: int
    min_log_stds = -10
    max_log_stds = 2

    @nn.compact
    def __call__(self, state: jnp.ndarray, noise_key) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.squeeze(state)
        for n_units in self.features:
            x = nn.Dense(n_units)(x)
            x = nn.relu(x)

        means = nn.Dense(self.action_dim)(x)

        if noise_key is None:  # deterministic
            return means, 1
        else:
            log_stds_unclipped = nn.Dense(self.action_dim)(x)
            log_stds = self.min_log_stds + (self.max_log_stds - self.min_log_stds) / 2 * (
                1 + nn.tanh(log_stds_unclipped)
            )
            stds = jnp.exp(log_stds)

            action_pre_tanh = means + stds * jax.random.normal(noise_key, shape=stds.shape)
            action = jnp.tanh(action_pre_tanh)

            # Gaussian log-prob: -1/2 ((x - mean) / std)^2 -1/2 log(2 pi) -log(sigma)
            log_prob_uncorrected = (
                -0.5 * jnp.square(action_pre_tanh / stds - means / stds) - 0.5 * jnp.log(2 * jnp.pi) - jnp.log(stds)
            )

            # d tanh^{-1}(y) / dy = 1 / (1 - y^2)
            log_prob = log_prob_uncorrected - jnp.log(1 - action**2 + 1e-6)

            return action, jnp.sum(log_prob, axis=-1)
