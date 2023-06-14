
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from jax import Array
from jax.random import KeyArray
from ensemble.policy_gradient_algorithms import AgentTraining, calculate_gae

from ensemble.single_agent import Agent, sample_action, get_policy_entropy  # pyright: ignore
@dataclass
class A2CTraining(AgentTraining):
    """Hyperparametrs for A2C training, with default values"""

    entropy_coef: float = 0.1
    update_name: str = "a2c"



    def update(
        self,
        agent: Agent,
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
                    self.gamma,
                    self.td_lambda_lambda,
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
                - self.entropy_coef * entropy.mean(),
                log_prob_grad,
            )
            actor_loss = (
                -advantages.mean() * log_probs.mean()
                - self.entropy_coef * entropy.mean()
            )
            critic_loss = advantages.mean() ** 2

            agent.state.update(actor_grad, critic_grad)
            return actor_loss, critic_loss

        states, actions, rewards, masks= agent.replay_buffer.to_arrays()

        return _inner(states, actions,  entropy)

    def episode(self, random_key: KeyArray, agent:Agent, envs: RecordEpisodeStatistics):

        agent.replay_buffer.empty()
        
        states,_ = envs.reset()
        policy_entropy = []
        for _ in range(self.num_timesteps):
            action_logits = agent.actor_forward(agent.state.actor_params, states)
            policy_entropy.append(get_policy_entropy(action_logits))
            actions = sample_action(random_key, action_logits)
            _, random_key = jax.random.split(random_key)
            next_states, rewards, dones, _, _= envs.step(np.array(actions).astype(np.int32))
            agent.replay_buffer.append(states, actions, rewards, dones)
            states = next_states
        actor_loss, critic_loss = self.update(agent)
        entropy = jnp.mean(jnp.stack(policy_entropy))
        
        return states, actor_loss, critic_loss, entropy, random_key
