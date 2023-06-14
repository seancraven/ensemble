
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
from ensemble.policy_gradient_algorithms import AgentTraining

from ensemble.agent import Agent  # pyright: ignore
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
def update(
    cls,
    agent: Agent,
    training_params: A2CTraining,
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
                training_params.gamma,
                training_params.td_lambda_lambda,
            )
        )
    def get_mean_log_probs(params, states, actions):
        return jnp.mean(agent.get_action_log_probs(params, states, actions))

    
    @jax.jit
    def _inner(states, actions, entropy):
        advantages, advantages_grad = jax.jit(jax.value_and_grad(calculate_advantage))(
            agent.state.critic_params, states
        )
        critic_grad = jax.tree_map(lambda x: jnp.mean(-2 * x * advantages), advantages_grad)


        log_probs, log_prob_grad = jax.jit(jax.value_and_grad(get_mean_log_probs))(
            agent.state.actor_params, states, actions
        )

        actor_grad = jax.tree_map(
            lambda grad: -advantages.mean() * grad
            - training_params.entropy_coef * entropy.mean(),
            log_prob_grad,
        )
        actor_loss = (
            -advantages.mean() * log_probs.mean()
            - training_params.entropy_coef * entropy.mean()
        )
        critic_loss = advantages.mean() ** 2

        agent.state.update(actor_grad, critic_grad)
        return actor_loss, critic_loss

    states, actions, rewards, masks= agent.replay_buffer.to_arrays()

    return _inner(states, actions,  entropy)

def a2c_episode(random_key: KeyArray, agent:Agent, envs: RecordEpisodeStatistics, training_params: A2CTraining):

    agent.replay_buffer.empty()
    
    states,_ = envs.reset()
    policy_entropy = []
    for _ in range(training_params.num_timesteps):
        action_logits = agent.actor_forward(agent.state.actor_params, states)
        policy_entropy.append(get_policy_entropy(action_logits))
        actions = sample_action(random_key, action_logits)
        _, random_key = jax.random.split(random_key)
        next_states, rewards, dones, _, _= envs.step(np.array(actions).astype(np.int32))
        agent.replay_buffer.append(states, actions, rewards, dones)
        states = next_states
    actor_loss, critic_loss = a2c_update(agent, training_params)
    entropy = jnp.mean(jnp.stack(policy_entropy))
    
    return states, actor_loss, critic_loss, entropy, random_key
