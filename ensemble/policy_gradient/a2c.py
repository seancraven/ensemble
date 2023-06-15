"""
Actor Critic training algorithm implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from jax import Array
from jax.random import KeyArray

from ensemble.policy_gradient.base import AgentTraining, calculate_gae
from ensemble.single_agent import Agent, get_policy_entropy, sample_action
from ensemble.typing import AgentParams, Entropy, EpisodeActions, EpisodeStates, States


@dataclass
class A2CTraining(AgentTraining):
    """Hyperparametrs for A2C training, with default values"""

    entropy_coef: float = 0.1
    update_name: str = "a2c"

    def update(
        self,
        agent: Agent,
        entropy: Entropy = jnp.array([0]),
    ) -> Tuple[Array, Array]:
        """Updates the agent's policy using gae actor critic."""

        def calculate_advantage(params: AgentParams, states: EpisodeStates):
            return jnp.mean(
                calculate_gae(
                    agent,
                    params,
                    rewards,
                    states,
                    masks,
                    self.gamma,
                    self.td_lambda_lambda,
                )
            )

        def get_mean_log_probs(
            params: AgentParams, states: EpisodeStates, actions: EpisodeActions
        ):
            return jnp.mean(agent.get_action_log_probs(params, states, actions))

        @jax.jit
        def _inner(states: EpisodeStates, actions: EpisodeActions, entropy: Entropy):
            advantages, advantages_grad = jax.jit(
                jax.value_and_grad(calculate_advantage)
            )(agent.state.critic_params, states)
            critic_grad = jax.tree_map(
                lambda x: jnp.mean(-2 * x * advantages), advantages_grad
            )

            log_probs, log_prob_grad = jax.jit(jax.value_and_grad(get_mean_log_probs))(
                agent.state.actor_params, states, actions
            )

            actor_grad = jax.tree_map(
                lambda grad: -advantages.mean() * grad
                - self.entropy_coef * entropy.mean(),
                log_prob_grad,
            )
            actor_loss = (
                -advantages.mean() * log_probs.mean()
                - self.entropy_coef * entropy.mean()
            )
            critic_loss = advantages.mean() ** 2

            return (actor_loss, critic_loss), (actor_grad, critic_grad)

        states, actions, rewards, masks = agent.replay_buffer.to_arrays()
        loss_tup, grad_tup = _inner(states, actions, entropy)

        agent.state.update(*grad_tup)
        return loss_tup

    def episode(
        self,
        random_key: KeyArray,
        agent: Agent,
        env_wrapper: RecordEpisodeStatistics,
        inital_states: States,
    ):
        agent.replay_buffer.empty()
        states = inital_states
        policy_entropy = []
        for _ in range(self.num_timesteps):
            action_logits = agent.actor_forward(agent.state.actor_params, states)

            actions = sample_action(random_key, action_logits)

            next_states, rewards, dones, _, _ = env_wrapper.step(
                np.array(actions).astype(np.int32)
            )
            agent.replay_buffer.append(states, actions, rewards, dones)
            policy_entropy.append(get_policy_entropy(action_logits))
            states = next_states
            _, random_key = jax.random.split(random_key)

        actor_loss, critic_loss = self.update(agent)
        entropy = jnp.mean(jnp.stack(policy_entropy))
        final_states = states

        return (
            random_key,
            final_states,
            actor_loss,
            critic_loss,
            entropy,
        )
