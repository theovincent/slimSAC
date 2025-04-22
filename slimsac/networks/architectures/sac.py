from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp
import jax


class CriticNet(nn.Module):
    features: Sequence[int]
    n_heads: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.squeeze(x)
        x = jnp.concatenate([x, action], -1)
        for n_units in self.features:
            x = nn.Dense(n_units)(x)
            x = nn.relu(x)
        x = nn.Dense(self.n_heads)(x)
        return x


class ActorNet(nn.Module):
    features: Sequence[int]
    action_dim: int
    log_std_min: float = -20
    log_std_max: float = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray, noise_key=None) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.squeeze(x)
        for n_units in self.features:
            x = nn.Dense(n_units)(x)
            x = nn.relu(x)

        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        std = jnp.exp(log_std)

        if noise_key is None:
            pre_tanh = mean  # deterministic
        else:
            noise = jax.random.normal(noise_key, shape=(self.action_dim,))
            pre_tanh = mean + std * noise

        action = jnp.tanh(pre_tanh)

        # Gaussian log-prob
        log_prob = -0.5 * (((pre_tanh - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
        log_prob = jnp.sum(log_prob, axis=-1)

        # Tanh correction
        log_prob -= jnp.sum(2.0 * (jnp.log(2.0) - pre_tanh - nn.softplus(-2.0 * pre_tanh)), axis=-1)

        return action, log_prob
