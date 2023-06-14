from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

import gymnasium as gym
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from gymnasium.wrappers.record_episode_statistics import \
    RecordEpisodeStatistics
from jax import Array
from jax.random import KeyArray
from jax.typing import ArrayLike
from optax import GradientTransformation, OptState, adam


class RLEnvironmentError(Exception):
    """Raised when the environment is not supported."""


@dataclass
class ReplayBuffer:
    states: List[Array]
    actions: List[Array]
    rewards: List[Array]
    dones: List[Array]

    def empty(self):
        """Removes all history from the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()

    def to_arrays(self) -> Tuple[Array, Array, Array, Array]:
        """Converts the buffer to arrays."""
        return (
            jnp.stack(self.states),
            jnp.stack(self.actions),
            jnp.stack(self.rewards),
            jnp.stack(self.dones),
        )

    def append(self, states: Array, actions: Array, rewards: Array, dones: Array):
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

        def actor(states: ArrayLike) -> Array:
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

        def critic(states: ArrayLike) -> Array:
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

    def actor_forward(self, params: hk.Params, state: ArrayLike) -> Array:
        return jax.jit(self.transformed_actor.apply)(params, state).squeeze()

    def critic_forward(self, params: hk.Params, state: ArrayLike) -> Array:
        return jax.jit(self.transformed_critic.apply)(params, state).squeeze()

    def get_action_log_probs(self, params, states, actions):
        action_logits = self.actor_forward(params, states)
        return action_log_probs(action_logits, actions)


def sample_action(rng_key: KeyArray, action_logits: Array) -> Array:
    """Samples action according to softmax policy defined by input logits."""
    return jax.random.categorical(rng_key, logits=action_logits)


@jax.jit
def action_log_probs(action_logits: Array, actions: Array) -> Array:
    """Computes log probability of actions under softmax policy defined by input logits."""
    action_log_probs = jax.nn.log_softmax(action_logits)
    return action_log_probs.at[actions].get()


@jax.jit
def get_policy_entropy(action_log_probs: ArrayLike) -> Array:
    return -jnp.sum(action_log_probs * jnp.exp(action_log_probs), axis=-1)
