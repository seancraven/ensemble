from __future__ import annotations
from typing import Tuple
from typing import List
import jax
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
import gymnasium as gym
import jax.numpy as jnp
from dataclasses import dataclass
import haiku as hk
from jax import Array
from jax.random import KeyArray
from jax.typing import ArrayLike
from optax import OptState, GradientTransformation, adam
from ensemble.policy_gradient_algorithms import A2CTraining
import optax
from typing import Union
import numpy as np



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
        return jnp.stack(advantages).mean(axis = 1)
    return _inner(rewards, values, masks)

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




