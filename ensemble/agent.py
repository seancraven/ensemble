from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import gymnasium as gym
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array
from jax.nn import log_softmax
from jax.random import KeyArray
from jax.typing import ArrayLike
from optax import GradientTransformation, OptState, adam


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
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
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
        self.transformed_critic = self.get_critic_params()
        self.state = AgentState.new(
            self.transformed_actor.init((1, self.input_dim)),
            self.transformed_critic.init((1, self.input_dim)),
            actor_lr,
            critic_lr,
        )

    def actor_forward(self, params: hk.Params, state: ArrayLike) -> Array:
        return self.transformed_actor.apply(params, state)

    def critic_forward(self, params: hk.Params, state: ArrayLike) -> Array:
        return self.transformed_critic.apply(params, state)

    def get_actor_params(self) -> hk.Transformed:
        return hk.without_apply_rng(hk.transform(self.actor))

    def get_critic_params(self) -> hk.Transformed:
        return hk.without_apply_rng(hk.transform(self.critic))

    @staticmethod
    def get_action_entropy(action_log_probs: ArrayLike) -> Array:
        return -jnp.sum(action_log_probs * jnp.exp(action_log_probs), axis=-1)

    @staticmethod
    def get_action(key: KeyArray, action_log_probs: ArrayLike) -> Array:
        return jax.random.categorical(key, action_log_probs)

    def get_action_log_probs(self, params: hk.Params, state: ArrayLike) -> Array:
        logits = self.actor_forward(params, state)
        return log_softmax(logits)


@dataclass
class AgentState:
    """Container for mutable agent state."""

    actor_opt: GradientTransformation
    critic_opt: GradientTransformation

    actor_params: Union[hk.Params, optax.Params]
    critic_params: Union[hk.Params, optax.Params]
    actor_opt_state: OptState
    critic_opt_state: OptState

    @staticmethod
    def new(
        actor_params: hk.Params,
        critic_params: hk.Params,
        actor_lr: float,
        critic_lr: float,
    ) -> AgentState:
        """Constructs a new AgentState."""
        actor_opt = adam(actor_lr)
        critic_opt = adam(critic_lr)
        actor_opt_init = actor_opt.init(actor_params)
        critic_opt_init = critic_opt.init(critic_params)
        return AgentState(
            actor_opt,
            critic_opt,
            actor_params,
            critic_params,
            actor_opt_init,
            critic_opt_init,
        )

    def update(self, actor_grad: Array, critic_grad: Array):
        """Updates the agent state."""
        actor_updates, self.actor_opt_state = self.actor_opt.update(
            actor_grad, self.actor_opt_state, self.actor_params
        )
        self.actor_params = optax.apply_updates(self.actor_params, actor_updates)

        critic_updates, self.critic_opt_state = self.critic_opt.update(
            critic_grad, self.critic_opt_state, self.critic_params
        )
        self.critic_params = optax.apply_updates(self.critic_params, critic_updates)
