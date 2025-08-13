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

    @nn.compact
    def __call__(self, state: jnp.ndarray, noise_key) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.squeeze(state)
        for n_units in self.features:
            x = nn.Dense(n_units)(x)
            x = nn.relu(x)

        mean = nn.Dense(self.action_dim)(x)

        if noise_key is None:  # deterministic
            return mean, 1
        else:
            log_std_unclipped = nn.Dense(self.action_dim)(x)
            log_std = jnp.clip(log_std_unclipped, -20, 2)
            std = jnp.exp(log_std)

            noise = jax.random.normal(noise_key, shape=(self.action_dim,))
            action_pre_tanh = mean + std * noise
            action = jnp.tanh(action_pre_tanh)

            # Gaussian log-prob
            log_prob_uncorrected = jax.scipy.stats.norm.logpdf(action_pre_tanh, mean, std * noise)
            # d tanh^{-1}(y) / dy = 1 / (1 - y^2)
            log_prob = log_prob_uncorrected - jnp.log(1 - action**2 + 1e-6)

            return action, jnp.sum(log_prob, axis=-1)
