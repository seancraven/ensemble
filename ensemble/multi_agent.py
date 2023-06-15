from typing import List

import gymnasium as gym
import haiku as hk
import jax
import jax.numpy as jnp
from jax.numpy.typing import Array
from jax.random import KeyArray
from jaxtyping import Float, PyTree

from ensemble.single_agent import Agent


class MultiAgent:
    """
    Wrapper for multiple agents.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.spaces.Discrete,
        internal_dim: int,
        num_agents: int,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        random_key: KeyArray = jax.random.PRNGKey(0),
    ):
        _, keys = jax.random.split(random_key, num_agents)
        self.agents = [
            Agent(
                observation_space, action_space, internal_dim, actor_lr, critic_lr, key
            )
            for key in keys
        ]

    def state_update(
        self,
        actor_grads: List[PyTree[Float[Array, "batch params"]]],
        critic_grads: List[PyTree[Float[Array, "batch params"]]],
    ):
        for agent, actor_grad, critic_grad in zip(
            self.agents, actor_grads, critic_grads
        ):
            agent.state.update(actor_grad, critic_grad)

    def buffer_reset(self):
        for agent in self.agents:
            agent.buffer_reset()
