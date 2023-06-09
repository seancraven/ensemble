from typing import Tuple

import gymnasium as gym
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.nn import log_softmax
from jax.random import KeyArray
from jax.typing import ArrayLike


class RLEnvironmentError(Exception):
    """Raised when the environment is not supported."""


class Agent(hk.Module):
    """
    An Actor critic agent, that defines a stochastic policy over a discrete action space.
    The agent
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.spaces.Discrete,
        internal_dim: int,
    ):
        super().__init__()
        match observation_space.shape:
            case tuple(observation_space.shape):
                self.input_dim = int(np.prod(observation_space.shape))
            case None:
                raise RLEnvironmentError("Environment must have shape supported.")

        self.internal_dim = internal_dim

        self.critic = hk.Sequential(
            [
                hk.Linear(self.internal_dim),
                jax.nn.relu,
                hk.Linear(self.internal_dim),
                jax.nn.relu,
                hk.Linear(1),
            ]
        )
        self.actor = hk.Sequential(
            [
                hk.Linear(self.internal_dim),
                jax.nn.relu,
                hk.Linear(self.internal_dim),
                jax.nn.relu,
                hk.Linear(action_space.n),
            ]
        )
        self.transformed_actor = self.get_actor_params()
        self.actor_params = self.transformed_actor.init(jnp.zeros((1, self.input_dim)))
        self.transformed_critic = self.get_critic_params()
        self.critic_params = self.transformed_critic.init(
            jnp.zeros((1, self.input_dim))
        )

    def actor_forward(self, state: ArrayLike) -> Array:
        return self.transformed_actor.apply(self.actor_params, state)

    def critic_forward(self, state: ArrayLike) -> Array:
        return self.transformed_critic.apply(self.critic_params, state)

    def get_actor_params(self) -> hk.Transformed:
        return hk.without_apply_rng(hk.transform(self.actor))

    def get_critic_params(self) -> hk.Transformed:
        return hk.without_apply_rng(hk.transform(self.critic))

    def get_action(
        self, key: KeyArray, state: ArrayLike
    ) -> Tuple[Array, Array, Array, Array, KeyArray]:
        """Selects an action from the agent's policy.

        Args:
            key (KeyArray): A PRNG key.
            state (ArrayLike): The current state of the environment.

        Returns:
            Tuple[Array, Array, Array, Array, KeyArray]: A tuple containing the action, log probability of the action, the state value, the entropy of the policy and the new PRNG key.
        """

        logits = self.actor_forward(state)
        key, subkey = jax.random.split(key)
        actions = jax.random.categorical(subkey, logits)
        log_prob = log_softmax(logits)
        entropy = -jnp.sum(log_prob * jnp.exp(log_prob), axis=-1)
        state_values = self.critic_forward(state)
        return actions, log_probs, state_values, entropy, subkey
