"""
Module that defines the protocol for agent trainng and how the agent is trained.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Tuple

import gymnasium as gym
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from jax import Array
from jax.nn import log_softmax
from jax.random import KeyArray
from jax.typing import ArrayLike

from ensemble.single_agent import Agent  # pyright: ignore


def train_agent(
    agent: Agent,
    envs: gym.vector.VectorEnv,
    training: AgentTraining,
    dir_name: str = "",
):
    env_wrapper = RecordEpisodeStatistics(
        envs, deque_size=envs.num_envs * training.num_episodes
    )

    actor_losses = []
    critic_losses = []
    entropies = []
    states, _ = env_wrapper.reset(seed=training.seed)

    key = jax.random.PRNGKey(training.seed)
    _, subkey = jax.random.split(key)

    for _ in range(training.num_episodes):
        (
            subkey,
            states,
            actor_loss,
            critic_loss,
            entropy,
        ) = training.episode(subkey, agent, env_wrapper, states)
        _, subkey = jax.random.split(key)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        entropies.append(entropy)
    jnp.stack(actor_losses)
    jnp.stack(critic_losses)
    jnp.stack(entropies)
    np.save(
        f"{dir_name}/{training.seed}_returns.npy",
        np.array(env_wrapper.return_queue),
    )
    np.save(f"{dir_name}/{training.seed}_entropies.npy", entropies)
    np.save(f"{dir_name}/{training.seed}_actor_losses.npy", actor_losses)
    np.save(f"{dir_name}/{training.seed}_critic_losses.npy", critic_losses)


@dataclass
class AgentTraining(Protocol):
    """Container for general training hyperparameters, and training behaviour."""

    update_name: str
    num_episodes: int = 100
    num_timesteps: int = 128
    seed: int = 0
    td_lambda_lambda: float = 0.95
    gamma: float = 0.99
    lrs: Tuple[float, float] = (1e-3, 5e-4)

    def episode(
        self,
        random_key: KeyArray,
        agent: Agent,
        env_wrapper: RecordEpisodeStatistics,
        inital_states: np.ndarray,
    ) -> Tuple[KeyArray, np.ndarray, Array, Array, Array]:
        """Defines how experience from an episode updates the agent.
        Args:
            random_key: The random key for the episode.
            agent: The agent to train.
            env_wrapper: The environment with a wrapper.
        Returns:
            the final random key, The final states, the actor loss, the critic_loss,
            the entropies.
        """
        ...

    def update(self, agent: Agent, update_parameters: Any):
        """Updates the agent's policy using gae actor critic.
        Args:
            agent: The agent to update.
            update_parameters: The hyperparameters unique to the update.

        """
        ...


def calculate_gae(
    agent: Agent,
    params: hk.Params,
    rewards: Array,
    states: Array,
    masks: Array,
    gamma: float,
    lambda_: float,
) -> Array:
    """Calculates the generalized advantage estimate as a function of input parameters.
    Using recursive TD(lambda).
    Args:
        agent: The agent with the experience
        rewards: Tensor of rewards: (batch_size, timestep)
        states: Tensor of entropy values: (batch_size, timestep).
        masks: Tensor of masks: (batch_size, timestep), 1 if the episode is not
        done, 0 otherwise.
        gamma: The discount factor for the mdp.
        lambda_: The lambda parameter for TD(lambda), controls the amount of
        bias/variance.

    Returns:
        advantages: Tensor of advantages: (batch_size, timestep).
    """
    values = agent.critic_forward(params, states)

    @jax.jit
    def _inner(rewards, values, masks):
        max_timestep = rewards.shape[0]
        advantages = [jnp.zeros_like(rewards.at[0].get())]

        for timestep in reversed(range(max_timestep - 1)):
            delta = (
                rewards.at[timestep].get()
                + gamma * values.at[timestep + 1].get() * masks.at[timestep].get()
                - values.at[timestep].get()
            )
            advantages.insert(
                0,
                delta + gamma * lambda_ * masks.at[timestep].get() * advantages[0],
            )
        return jnp.stack(advantages)

    return _inner(rewards, values, masks)
