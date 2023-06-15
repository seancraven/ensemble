"""
Single agent for rl training on a discrete action spacce.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

import gymnasium as gym
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.random import KeyArray
from jax.typing import ArrayLike
from jaxtyping import Array, Float
from optax import GradientTransformation, OptState, adam

from ensemble.typing import (
    Actions,
    ActorGrad,
    AgentParams,
    CriticGrad,
    Dones,
    EpisodeActions,
    EpisodeDones,
    EpisodeRewards,
    EpisodeStates,
    Rewards,
    States,
)


class RLEnvironmentError(Exception):
    """Raised when the environment is not supported."""


@dataclass
class ReplayBuffer:
    states: List[States]
    actions: List[Actions]
    rewards: List[Rewards]
    dones: List[Dones]

    def empty(self):
        """Removes all history from the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()

    def to_arrays(
        self,
    ) -> Tuple[EpisodeStates, EpisodeActions, EpisodeRewards, EpisodeDones]:
        """Converts the buffer to arrays."""
        return (
            jnp.stack(self.states),
            jnp.stack(self.actions),
            jnp.stack(self.rewards),
            jnp.stack(self.dones),
        )

    def append(self, states: States, actions: Actions, rewards: Rewards, dones: Dones):
        """Appends the given transition to the buffer."""
        self.states.append(states)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.dones.append(dones)


@dataclass
class AgentState:
    """Container for mutable agent state."""

    actor_opt: GradientTransformation
    critic_opt: GradientTransformation

    actor_params: AgentParams
    critic_params: AgentParams
    actor_opt_state: OptState
    critic_opt_state: OptState

    @staticmethod
    def new(
        actor_params: AgentParams,
        critic_params: AgentParams,
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

    def update(self, actor_grad: ActorGrad, critic_grad: CriticGrad):
        """Updates the agent state."""

        @jax.jit
        def _inner_critic(
            params: AgentParams,
            grad: CriticGrad,
            opt_state: OptState,
        ):
            updates, opt_state = self.critic_opt.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        @jax.jit
        def _inner_actor(
            params: AgentParams,
            grad: ActorGrad,
            opt_state: OptState,
        ):
            updates, opt_state = self.actor_opt.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        self.actor_params, self.actor_opt_state = _inner_actor(
            self.actor_params, actor_grad, self.actor_opt_state
        )
        self.critic_params, self.critic_opt_state = _inner_critic(
            self.critic_params, critic_grad, self.critic_opt_state
        )


class Agent:
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.spaces.Discrete,
        internal_dim: int,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        random_key: KeyArray = jax.random.PRNGKey(0),
    ):
        super().__init__()
        match observation_space.shape:
            case tuple(observation_space.shape):
                self.input_dim = int(np.prod(observation_space.shape))
            case None:
                raise RLEnvironmentError("Environment must have shape supported.")

        self.internal_dim = internal_dim
        self.action_space = action_space

        def actor(states: States) -> Actions:
            mlp = hk.Sequential(
                [
                    hk.Linear(self.internal_dim),
                    jax.nn.relu,
                    hk.Linear(self.internal_dim),
                    jax.nn.relu,
                    hk.Linear(self.action_space.n),
                ]
            )
            return mlp(states)

        def critic(states: States) -> Rewards:
            mlp = hk.Sequential(
                [
                    hk.Linear(self.internal_dim),
                    jax.nn.relu,
                    hk.Linear(self.internal_dim),
                    jax.nn.relu,
                    hk.Linear(1),
                ]
            )
            return mlp(states)

        self.replay_buffer = ReplayBuffer([], [], [], [])

        self.transformed_actor = hk.without_apply_rng(hk.transform(actor))
        self.transformed_critic = hk.without_apply_rng(hk.transform(critic))
        actor_key, critic_key = jax.random.split(random_key)
        self.state = AgentState.new(
            self.transformed_actor.init(actor_key, jnp.ones((1, self.input_dim))),
            self.transformed_critic.init(critic_key, jnp.ones((1, self.input_dim))),
            actor_lr,
            critic_lr,
        )

    def actor_forward(
        self, params: AgentParams, state: Union[EpisodeStates, States]
    ) -> Float[Array, "*envs action_shape"]:
        return jax.jit(self.transformed_actor.apply)(params, state).squeeze()

    def critic_forward(
        self, params: AgentParams, state: Union[States, EpisodeStates]
    ) -> Rewards:
        return jax.jit(self.transformed_critic.apply)(params, state).squeeze()

    def get_action_log_probs(
        self,
        params: AgentParams,
        states: Union[States, EpisodeStates],
        actions: Union[Actions, EpisodeActions],
    ):
        action_logits = self.actor_forward(params, states)
        return action_log_probs(action_logits, actions)


@jax.jit
def sample_action(
    rng_key: KeyArray, action_logits: Float[Array, "*envs action_shape"]
) -> Array:
    """Samples action according to softmax policy defined by input logits."""
    return jax.random.categorical(rng_key, logits=action_logits)


@jax.jit
def action_log_probs(
    action_logits: Float[Array, "*envs action_shape"],
    actions: Union[Actions, EpisodeActions],
) -> Float[Array, "*envs action_shape"]:
    """Computes log probability of actions under softmax policy defined by input logits."""
    action_log_probs = jax.nn.log_softmax(action_logits)
    return action_log_probs.at[actions].get()


@jax.jit
def get_policy_entropy(action_log_probs: Float[Array, "*envs action_shape"]) -> Array:
    """Calculates the entropy of a log probability distribution(policy)."""
    return -jnp.sum(action_log_probs * jnp.exp(action_log_probs), axis=-1)
