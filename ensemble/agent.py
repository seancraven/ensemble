import gymnasium as gym
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy import ndarray  # pyright: ignore
from jax.random import PRNGKey
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
    def get_action(self,key: PRNGKey,  state: ArrayLike,) -> ndarray:
        self.
