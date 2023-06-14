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

    actor_losses = []
    critic_losses = []
    entropies = []
    states, _ = env_wrapper.reset(seed=training.seed)

    key = jax.random.PRNGKey(training.seed)
    _, subkey = jax.random.split(key, env_wrapper.num_envs)

    for _ in range(training.num_episodes):
        states, actor_loss, critic_loss, entropy, subkey = training.episode(
            subkey, states, agent, training, env_wrapper
        )
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
    num_episodes: int = 1000
    num_envs: int = 10
    num_timesteps: int = 128
    seed: int = 0
    td_lambda_lambda: float = 0.95
    gamma: float = 0.99
    lrs: Tuple[float, float] = (1e-3, 5e-4)

    def episode(
        self,
        keys: KeyArray,
        states: np.ndarray,
        agent: Agent,
        training: AgentTraining,
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
        training: AgentTraining,
        *args: Any,
    ):
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


@dataclass
class A2CTraining(AgentTraining):
    """Hyperparametrs for A2C training, with default values"""

    entropy_coef: float = 0.1
    update_name: str = "a2c"

    def episode(
        self,
        key: KeyArray,
        states: np.ndarray,
        agent: Agent,
        training: A2CTraining,
        env_wrapper: RecordEpisodeStatistics,
    ) -> Tuple[np.ndarray, Array, Array, Array, KeyArray]:
        rewards = []
        state_trajectory = []
        masks = []
        entropy = []
        actions = []
        key, subkey = jax.random.split(key, env_wrapper.num_envs)
        for _ in range(training.num_timesteps):
            action_log_probs = agent.get_log_policy(agent.state.actor_params, states)
            policy_entropy = agent.get_policy_entropy(action_log_probs)
            jax_action = agent.get_action(subkey, action_log_probs)
            action = np.array(jax_action).astype(np.int32)

            states, reward, done, _, _ = env_wrapper.step(action)

            rewards.append(reward.squeeze())
            state_trajectory.append(states.squeeze())
            masks.append(1 - done)
            entropy.append(policy_entropy.squeeze())
            actions.append(jax_action.squeeze())
            key, subkey = jax.random.split(subkey)

        rewards = jnp.stack(rewards)
        state_trajectory = jnp.stack(state_trajectory)
        masks = jnp.stack(masks)
        entropy = jnp.stack(entropy)
        actions = jnp.stack(actions)

        actor_loss, critic_loss = a2c_update(
            agent, rewards, state_trajectory, actions, masks, training, entropy.mean()
        )
        return states, actor_loss, critic_loss, entropy, subkey


def a2c_update(
    agent: Agent,
    rewards: Array,
    states: Array,
    actions: Array,
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

    def calculate_advantage(params, states):
        return jnp.mean(
            calculate_gae(
                agent,
                params,
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
    critic_grad = jax.tree_map(lambda x: jnp.mean(-2 * x * advantages), advantages_grad)

    log_probs, log_prob_grad = jax.value_and_grad(agent.get_action_log_probs)(
        agent.state.actor_params, states, actions
    )

    actor_grad = jax.tree_map(
        lambda grad: -advantages.mean() * grad
        - hyperparameters.entropy_coef * entropy.mean(),
        log_prob_grad,
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
