from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import optax

from slimsac.networks.architectures.sac import CriticNet, ActorNet
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
        update_to_data: int,
        tau: float,
        features_pi: list,
        features_qf: list,
    ):
        self.key, actor_key, critic_key, ent_key = jax.random.split(key, 4)

        obs = jnp.zeros(observation_dim, dtype=jnp.float32)
        action = jnp.zeros(action_dim, dtype=jnp.float32)

        # Critic (2 Q networks)
        self.critic = CriticNet(features_qf)
        self.critic_params = jax.vmap(self.critic.init, in_axes=[0, None, None])(
            jax.random.split(critic_key, 2), obs, action
        )
        self.critic_target_params = self.critic_params.copy()
        self.critic_optimizer = optax.adam(learning_rate)
        self.critic_optimizer_state = self.critic_optimizer.init(self.critic_params)

        # Actor
        self.actor = ActorNet(features_pi, action_dim)
        self.actor_params = self.actor.init(actor_key, obs)
        self.actor_optimizer = optax.adam(learning_rate)
        self.actor_optimizer_state = self.actor_optimizer.init(self.actor_params)

        # Entropy coefficient
        self.log_ent_coef = jnp.array(np.log(1.0))
        self.entropy_optimizer = optax.adam(learning_rate)
        self.entropy_optimizer_state = self.entropy_optimizer.init(self.log_ent_coef)
        self.target_entropy = -np.float32(action_dim)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.tau = tau

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer, update_key):
        if step % self.update_to_data != 0:
            return None

        batch_samples = replay_buffer.sample()

        (
            self.critic_params,
            self.critic_target_params,
            self.actor_params,
            self.log_ent_coef,
            self.critic_optimizer_state,
            self.actor_optimizer_state,
            self.entropy_optimizer_state,
            losses,
        ) = self.learn_on_batch(
            self.critic_params,
            self.critic_target_params,
            self.actor_params,
            self.log_ent_coef,
            self.critic_optimizer_state,
            self.actor_optimizer_state,
            self.entropy_optimizer_state,
            batch_samples,
            update_key,
        )

        logs = {
            "train/critic_loss": losses[0],
            "train/actor_loss": losses[1],
            "train/ent_coef": losses[2],
        }
        return logs

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self,
        critic_params,
        critic_target_params,
        actor_params,
        log_ent_coef,
        critic_opt_state,
        actor_opt_state,
        entropy_opt_state,
        batch_samples,
        update_key,
    ):
        critic_key, actor_key = jax.random.split(update_key, 2)

        # Update critic
        critic_keys = jax.random.split(critic_key, batch_samples.action.shape[0])
        critic_loss, critic_grads = jax.value_and_grad(self.critic_loss_on_batch)(
            critic_params, critic_target_params, actor_params, log_ent_coef, batch_samples, critic_keys
        )
        critic_updates, critic_opt_state = self.critic_optimizer.update(critic_grads, critic_opt_state)
        critic_params = optax.apply_updates(critic_params, critic_updates)

        # Update actor
        actor_keys = jax.random.split(actor_key, batch_samples.action.shape[0])
        (actor_loss, entropy), actor_grads = jax.value_and_grad(self.actor_loss_on_batch, has_aux=True)(
            actor_params, critic_params, log_ent_coef, batch_samples, actor_keys
        )
        actor_updates, actor_opt_state = self.actor_optimizer.update(actor_grads, actor_opt_state)
        actor_params = optax.apply_updates(actor_params, actor_updates)

        # Update entropy coefficient
        entropy_loss, entropy_grads = jax.value_and_grad(self.entropy_loss)(log_ent_coef, entropy)
        entropy_updates, entropy_opt_state = self.entropy_optimizer.update(entropy_grads, entropy_opt_state)
        log_ent_coef = optax.apply_updates(log_ent_coef, entropy_updates)

        # Update critic target
        critic_target_params = optax.incremental_update(critic_params, critic_target_params, self.tau)

        return (
            critic_params,
            critic_target_params,
            actor_params,
            log_ent_coef,
            critic_opt_state,
            actor_opt_state,
            entropy_opt_state,
            (critic_loss, actor_loss, entropy_loss),
        )

    def entropy_loss(self, log_ent_coef, entropy):
        return log_ent_coef * (entropy - self.target_entropy)

    def critic_loss_on_batch(self, critic_params, critic_target_params, actor_params, log_ent_coef, samples, keys):
        losses = jax.vmap(self.critic_loss, in_axes=(None, None, None, None, 0, 0))(
            critic_params, critic_target_params, actor_params, log_ent_coef, samples, keys
        )
        return losses.mean()

    def critic_loss(self, critic_params, critic_target_params, actor_params, log_ent_coef, sample: ReplayElement, key):
        next_action, next_log_prob = self.actor.apply(actor_params, sample.next_state, noise_key=key)
        next_q_double = jax.vmap(self.critic.apply, in_axes=(0, None, None))(
            critic_target_params, sample.next_state, next_action
        )
        next_q = jnp.min(next_q_double, axis=0)
        target = sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * (
            next_q - jnp.exp(log_ent_coef) * next_log_prob
        )
        q_value = jax.vmap(self.critic.apply, in_axes=(0, None, None))(critic_params, sample.state, sample.action)

        td_loss = jnp.square(q_value - target)
        return td_loss.mean()

    def actor_loss_on_batch(self, actor_params, critic_params, log_ent_coef, samples, keys):
        losses, entropies = jax.vmap(self.actor_loss, in_axes=(None, None, None, 0, 0))(
            actor_params, critic_params, log_ent_coef, samples, keys
        )
        return losses.mean(), entropies.mean()

    def actor_loss(self, actor_params, critic_params, log_ent_coef, sample: ReplayElement, key):
        action, log_prob = self.actor.apply(actor_params, sample.state, noise_key=key)
        q_value_double = jax.vmap(self.critic.apply, in_axes=(0, None, None))(critic_params, sample.state, action)
        q_value = jnp.min(q_value_double, axis=0)
        return jnp.exp(log_ent_coef) * log_prob - q_value, -log_prob

    @partial(jax.jit, static_argnames="self")
    def sample_action(self, state, actor_params, key=None):
        return self.actor.apply(actor_params, state, noise_key=key)[0]

    def get_model(self):
        return {
            "critic": self.critic_params,
            "critic_target_params": self.critic_target_params,
            "actor": self.actor_params,
            "log_ent_coef": self.log_ent_coef,
        }
