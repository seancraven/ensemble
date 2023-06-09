from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Tuple

import gymnasium as gym
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.wrappers.record_episode_statistics import \
    RecordEpisodeStatistics
from jax import Array
from jax.nn import log_softmax
from jax.random import KeyArray
from jax.typing import ArrayLike

from ensemble.agent import Agent  # pyright: ignore


def train_agent(
    agent: Agent,
    envs: gym.vector.VectorEnv,
    training: AgentTraining,
    dir_name: str = "",
):
    env_wrapper = RecordEpisodeStatistics(
        envs, deque_size=training.num_envs * training.num_episodes
    )

    advantages = []
    entropies = []
    states, _ = env_wrapper.reset(seed=training.seed)

    for _ in range(training.num_episodes):
        states, advantage, entropy, keys = training.episode(
            states, agent, training, env_wrapper
        )
        keys = jax.random.split(keys, training.num_envs)
        advantages.append(advantage)
        entropies.append(entropy)
    jnp.stack(advantages)
    jnp.stack(entropies)
    np.save(
        f"{dir_name}/{training.seed}_returns.npy",
        np.array(env_wrapper.return_queue),
    )
    np.save(f"{dir_name}/{training.seed}_advantages.npy", advantages)
    np.save(f"{dir_name}/{training.seed}_entropies.npy", entropies)


@dataclass
class AgentTraining(Protocol):
    """Container for general training hyperparameters, and training behaviour."""

    update_name: str
    num_episodes: int = 1000
    num_envs: int = 10
    num_timesteps: int = 128
    seed: int = 0
    td_lambda_lambda: float = 0.95
    gamma: float = 0.99
    lrs: Tuple[float, float] = (1e-3, 5e-4)

    @staticmethod
    def episode(
        keys: KeyArray,
        states: np.ndarray,
        agent: Agent,
        hyp: AgentTraining,
        env_wrapper: RecordEpisodeStatistics,
    ) -> Tuple[np.ndarray, Array, Array, Array, KeyArray]:
        """Defines how experience from an episode updates the agent.
        Args:
            keys: The random keys for the episode.
            states: The initial states for the episode.
            agent: The agent to train.
            hyp: The hyperparameters for training.
            env_wrapper: The environment wrapper.
        Returns:
            The final states, the actor loss, the critic_loss,
            the entropies, the final random keys.
        """
        ...

    @staticmethod
    def update(
        agent: Agent,
        rewards: Array,
        states: Array,
        masks: Array,
        hyperparameters: AgentTraining,
        *args: Any,
    ):
        ...


def calculate_gae(
    params: hk.Params,
    agent: Agent,
    rewards: Array,
    states: Array,
    masks: Array,
    gamma: float,
    lambda_: float,
) -> Array:
    """Calculates the generalized advantage estimate. Using recursive TD(lambda).
    Args:
        rewards: Tensor of rewards: (batch_size, timestep)
        action_log_probs: Tensor of log probabilities of the actions: (batch_size).
        values: Tensor of state values: (batch_size, timestep).
        entropy: Tensor of entropy values: (batch_size, timestep).
        masks: Tensor of masks: (batch_size, timestep), 1 if the episode is not
        done, 0 otherwise.
        gamma: The discount factor for the mdp.
        lambda_: The lambda parameter for TD(lambda), controls the amount of
        bias/variance.
        ent_coef: The entropy coefficient, for exploration encouragement.

    Returns:
        advantages: Tensor of advantages: (batch_size, timestep).
    """
    values = agent.critic_forward(params, states)
    max_timestep = rewards.shape[0]
    advantages = jnp.zeros_like(rewards)
    for timestep in reversed(range(max_timestep - 1)):
        delta = (
            rewards[timestep]
            + gamma * values[timestep + 1] * masks[timestep]
            - values[timestep]
        )
        advantages[timestep] = (
            delta + gamma * lambda_ * masks[timestep] * advantages[timestep + 1]
        )
    return advantages


@dataclass
class A2CTraining(AgentTraining):
    """Hyperparametrs for A2C training, with default values"""

    entropy_coef: float = 0.1
    update_name: str = "a2c"

    @staticmethod
    def episode(
        key: KeyArray,
        states: np.ndarray,
        agent: Agent,
        hyp: A2CTraining,
        env_wrapper: RecordEpisodeStatistics,
    ) -> Tuple[np.ndarray, Array, Array, Array, KeyArray]:
        rewards = jnp.zeros((hyp.num_timesteps, hyp.num_envs))
        state_trajectory = jnp.zeros((hyp.num_timesteps, *states.shape))
        mask = jnp.zeros((hyp.num_timesteps, hyp.num_envs))
        entropy = jnp.zeros((1,))

        key, subkey = jax.random.split(key)
        for timestep in range(hyp.num_timesteps):
            actions: np.ndarray = np.array(agent.get_action(subkey, states))

            states, reward, done, _, _ = env_wrapper.step(
                actions.reshape(env_wrapper.env.action_space.shape)
            )

            rewards[timestep] = reward
            state_trajectory[timestep] = states
            mask[timestep] = 1 - done

            _, subkey = jax.random.split(subkey)

        actor_loss, critic_loss = A2CTraining.update(
            agent, rewards, jnp.array(state_trajectory), mask, hyp, entropy.mean()
        )
        return states, actor_loss, critic_loss, entropy, subkey

    @staticmethod
    def update(  # pylint: Ignore
        agent: Agent,
        rewards: Array,
        states: Array,
        masks: Array,
        hyperparameters: A2CTraining,
        entropy: Array = jnp.array([0]),
    ) -> Tuple[Array, Array]:
        """Updates the agent's policy using gae actor critic.
        Args:
            advantages: Tensor of advantages: (batch_size, timestep).
            action_log_probs: Tensor of log probabilities of the actions:
            (batch_size, timestep).

        """

        def calculate_advantage(params):
            return jnp.mean(
                calculate_gae(
                    params,
                    agent,
                    rewards,
                    states,
                    masks,
                    hyperparameters.gamma,
                    hyperparameters.td_lambda_lambda,
                )
            )

        advantages, advantages_grad = jax.value_and_grad(calculate_advantage)(
            agent.state.critic_params, states
        )
        critic_grad = jnp.mean(-2 * advantages_grad * advantages)

        log_probs, log_prob_grad = jax.value_and_grad(agent.get_action_log_probs)(
            agent.state.actor_params, states
        )

        actor_grad = (
            -advantages.mean() * log_prob_grad
            - hyperparameters.entropy_coef * entropy.mean()
        )
        actor_loss = (
            -advantages.mean() * log_probs.mean()
            - hyperparameters.entropy_coef * entropy.mean()
        )
        critic_loss = advantages.mean() ** 2

        agent.state.update(actor_grad, critic_grad)
        return actor_loss, critic_loss


@dataclass
class PPOTraining(AgentTraining):
    """Hyperparametrs for PPO training, with default values"""

    epsilon: float = 0.1
    update_name: str = "ppo"

    @staticmethod
    def episode(
        states: np.ndarray,
        agent: Agent,
        hyp: PPOTraining,
        env_wrapper: RecordEpisodeStatistics,
    ) -> Tuple[np.ndarray, Array, torch.Tensor]:
        raise NotImplementedError
        rewards = torch.zeros((hyp.num_timesteps, hyp.num_envs)).to(agent.device)
        value_estimates = torch.zeros_like(rewards).to(agent.device)
        action_log_probs = torch.zeros_like(rewards).to(agent.device)
        old_action_log_probs = torch.zeros_like(rewards).to(agent.device)
        mask = torch.zeros_like(rewards).to(agent.device)

        old_actor = deepcopy(agent.actor).to(agent.device)

        entropy = Array([0]).to(agent.device)

        for timestep in range(hyp.num_timesteps):
            actions, log_probs, values, entropy = agent.get_action(states)
            old_log_probs = old_actor(states).log_prob(actions)
            states, reward, done, _, _ = env_wrapper.step(
                actions.cpu().numpy().reshape(env_wrapper.env.action_space.shape)
            )

            rewards[timestep] = Array(reward).squeeze().to(agent.device)
            value_estimates[timestep] = values.squeeze().to(agent.device)
            action_log_probs[timestep] = log_probs.squeeze().to(agent.device)
            old_action_log_probs[timestep] = old_log_probs.squeeze().to(agent.device)
            mask[timestep] = Array(1 - done).squeeze().to(agent.device)

        advantage = calculate_gae(
            rewards,
            value_estimates,
            mask,
            gamma=hyp.gamma,
            lambda_=hyp.td_lambda_lambda,
        )

        PPOTraining.update(
            agent, advantage, action_log_probs, old_action_log_probs, hyp
        )

        return states, advantage, entropy

    @staticmethod
    def update(
        agent: Agent,
        advantages: Array,
        action_log_probs: Array,
        old_log_probs: Array,
        hyperparameters: PPOTraining,
    ):
        """Updates the agent's policy and value estimate using ppo.

        Args:
            advantages: Tensor of advantages: (batch_size, 1).
            action_log_probs: Tensor of log probabilities of the actions: (batch_size).
            old_log_probs: Tensor of log probabilities of the actions on the
            previous episode's policy: (batch_size).
            epsilon: The clipping parameter for the ppo loss.

        """
        raise NotImplementedError
        agent.critic_opt.zero_grad()
        square_td_errors = advantages.pow(2).mean()
        square_td_errors.backward()
        agent.critic_opt.step()
        agent.actor_opt.zero_grad()

        ratio = torch.exp(action_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(
            ratio, 1 - hyperparameters.epsilon, 1 + hyperparameters.epsilon
        )
        actor_loss = -(
            torch.stack(
                (advantages.detach() * clipped_ratio, advantages.detach() * ratio),
                dim=-1,
            )
            .min(
                dim=-1,
            )
            .values.mean()
        )
        actor_loss.backward()
        agent.actor_opt.step()
