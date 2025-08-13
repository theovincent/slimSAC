from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import optax

from slimsac.algorithms.architectures.sac import CriticNet, ActorNet
from slimsac.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class SAC:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        action_dim,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        tau: float,
        features_pi: list,
        features_qf: list,
    ):
        actor_key, critic_key = jax.random.split(key)

        obs = jnp.zeros(observation_dim, dtype=jnp.float32)
        action = jnp.zeros(action_dim, dtype=jnp.float32)

        # Critic (2 Q networks)
        self.critic = CriticNet(features_qf)
        self.critic_params = jax.vmap(self.critic.init, in_axes=(0, None, None))(
            jax.random.split(critic_key, 2), obs, action
        )
        self.critic_target_params = self.critic_params.copy()
        self.critic_optimizer = optax.adam(learning_rate)
        self.critic_optimizer_state = self.critic_optimizer.init(self.critic_params)

        # Actor
        self.actor = ActorNet(features_pi, action_dim)
        self.actor_params = self.actor.init(actor_key, obs, jax.random.PRNGKey(0))
        self.actor_optimizer = optax.adam(learning_rate)
        self.actor_optimizer_state = self.actor_optimizer.init(self.actor_params)

        # Entropy coefficient
        self.log_ent_coef = jnp.log(1.0)
        self.entropy_optimizer = optax.adam(learning_rate)
        self.entropy_optimizer_state = self.entropy_optimizer.init(self.log_ent_coef)
        self.target_entropy = -np.float32(action_dim)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.tau = tau

        self.cumulated_critic_loss = 0
        self.cumulated_actor_loss = 0
        self.cumulated_entropy_loss = 0

    def update_online_params(self, replay_buffer: ReplayBuffer, key):
        batch_samples = replay_buffer.sample()

        (
            self.critic_params,
            self.critic_target_params,
            self.actor_params,
            self.log_ent_coef,
            self.critic_optimizer_state,
            self.actor_optimizer_state,
            self.entropy_optimizer_state,
            self.cumulated_critic_loss,
            self.cumulated_actor_loss,
            self.cumulated_entropy_loss,
        ) = self.learn_on_batch(
            self.critic_params,
            self.critic_target_params,
            self.actor_params,
            self.log_ent_coef,
            self.critic_optimizer_state,
            self.actor_optimizer_state,
            self.entropy_optimizer_state,
            self.cumulated_critic_loss,
            self.cumulated_actor_loss,
            self.cumulated_entropy_loss,
            batch_samples,
            key,
        )

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self,
        critic_params,
        critic_target_params,
        actor_params,
        log_ent_coef,
        critic_optimizer_state,
        actor_optimizer_state,
        entropy_optimizer_state,
        cumulated_critic_loss,
        cumulated_actor_loss,
        cumulated_entropy_loss,
        batch_samples,
        key,
    ):
        critic_key, actor_key = jax.random.split(key, 2)

        # Update critic
        critic_loss, critic_grads = jax.value_and_grad(self.critic_loss_on_batch)(
            critic_params, critic_target_params, actor_params, log_ent_coef, batch_samples, critic_key
        )
        critic_updates, critic_optimizer_state = self.critic_optimizer.update(critic_grads, critic_optimizer_state)
        critic_params = optax.apply_updates(critic_params, critic_updates)

        # Update actor
        (actor_loss, entropy), actor_grads = jax.value_and_grad(self.actor_loss_on_batch, has_aux=True)(
            actor_params, critic_params, log_ent_coef, batch_samples, actor_key
        )
        actor_updates, actor_optimizer_state = self.actor_optimizer.update(actor_grads, actor_optimizer_state)
        actor_params = optax.apply_updates(actor_params, actor_updates)

        # Update entropy coefficient
        entropy_loss, entropy_grads = jax.value_and_grad(self.entropy_loss)(log_ent_coef, entropy)
        entropy_updates, entropy_optimizer_state = self.entropy_optimizer.update(entropy_grads, entropy_optimizer_state)
        log_ent_coef = optax.apply_updates(log_ent_coef, entropy_updates)

        # Update critic target
        critic_target_params = optax.incremental_update(critic_params, critic_target_params, self.tau)

        cumulated_critic_loss = (1 - self.tau) * cumulated_critic_loss + self.tau * critic_loss
        cumulated_actor_loss = (1 - self.tau) * cumulated_actor_loss + self.tau * actor_loss
        cumulated_entropy_loss = (1 - self.tau) * cumulated_entropy_loss + self.tau * entropy_loss

        return (
            critic_params,
            critic_target_params,
            actor_params,
            log_ent_coef,
            critic_optimizer_state,
            actor_optimizer_state,
            entropy_optimizer_state,
            cumulated_critic_loss,
            cumulated_actor_loss,
            cumulated_entropy_loss,
        )

    def critic_loss_on_batch(self, critic_params, critic_target_params, actor_params, log_ent_coef, samples, key):
        next_actions, next_log_probs = self.actor.apply(actor_params, samples.next_state, key)

        # shape (batch_size, 2, 1)
        q_values = jax.vmap(self.critic.apply, in_axes=(0, None, None), out_axes=1)(
            critic_params, samples.state, samples.action
        )
        q_values = q_values.squeeze(axis=-1)

        # shape (batch_size, 2, 1)
        next_q_values_double = jax.vmap(self.critic.apply, in_axes=(0, None, None), out_axes=1)(
            critic_target_params, samples.next_state, next_actions
        )
        # shape (batch_size)
        next_q_values = jnp.min(next_q_values_double.squeeze(axis=-1), axis=1)

        # shape (batch_size)
        targets_ = self.compute_target(samples, next_q_values, jnp.exp(log_ent_coef), next_log_probs)
        # shape (batch_size, 2)
        targets = jnp.repeat(targets_[:, jnp.newaxis], 2, axis=1)

        return jnp.square(q_values - targets).mean()

    def compute_target(
        self, samples: ReplayElement, next_q_values: jax.Array, ent_coef: float, next_log_probs: jax.Array
    ):
        return samples.reward + (1 - samples.is_terminal) * (self.gamma**self.update_horizon) * (
            next_q_values - ent_coef * next_log_probs
        )

    def actor_loss_on_batch(self, actor_params, critic_params, log_ent_coef, samples, key):
        actions, log_probs = self.actor.apply(actor_params, samples.state, key)

        # shape (batch_size, 2, 1)
        q_value_double = jax.vmap(self.critic.apply, in_axes=(0, None, None), out_axes=1)(
            critic_params, samples.state, actions
        )
        # shape (batch_size)
        q_values = jnp.min(q_value_double.squeeze(axis=-1), axis=1)

        losses = jnp.exp(log_ent_coef) * log_probs - q_values
        return losses.mean(), -log_probs.mean()

    def entropy_loss(self, log_ent_coef, entropy):
        return jnp.exp(log_ent_coef) * (entropy - self.target_entropy)

    @partial(jax.jit, static_argnames="self")
    def sample_action(self, state, actor_params, key):
        # only return the action
        return self.actor.apply(actor_params, state, key)[0]

    def get_logs(self):
        logs = {
            "train/critic_loss": self.cumulated_critic_loss,
            "train/actor_loss": self.cumulated_actor_loss,
            "train/entropy_loss": self.cumulated_entropy_loss,
            "train/entropy_coef": np.exp(self.log_ent_coef),
        }
        return logs

    def get_model(self):
        return {"critic": self.critic_params, "actor": self.actor_params, "log_ent_coef": self.log_ent_coef}
