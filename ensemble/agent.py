import gymnasium as gym
import haiku as hk
import jax
import jax.numpy as jnp
from jax.numpy import ndarray  # pyright: ignore


class DiscreteHead(hk.Module):
    pass


class ContinousHead(hk.Module):
    pass


class RLEnvironmentError(Exception):
    """Raised when the environment is not supported."""


class Agent(hk.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        internal_dim: int,
    ):
        super().__init__()
        match observation_space.shape:
            case tuple(observation_space.shape):
                self.input_dim = int(jnp.prod(observation_space.shape))
            case None:
                raise RLEnvironmentError("Environment must have shape supported.")

        self.internal_dim = internal_dim

        self.


def _out_space_to_head(self, space: gym.Space) -> hk.Module:
    """Returns a input layer, that maps the observation_space
    to the internal_dim.

    If the action space is discrete then the head of the model is
    a softmax.

    If the action space is continuous, then the head of the model is a gaussian.


    Args:
        space: The observation space of the environment.
        internal_dim: The dimension of the internal layer.
    Returns:
        head: The input layer.


        NotImplementedError: If the space is not discrete or continuous.
    """
    match space:
        case gym.spaces.Box():
            return ContinousHead(self.internal_dim, int(np.prod(space.shape)))
        case gym.spaces.Discrete():
            return DiscreteHead(self.internal_dim, int(space.n))
        case _:
            print(space)
            raise NotImplementedError("Only Box and Discrete spaces are supported.")
