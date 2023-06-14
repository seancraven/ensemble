import gymnasium as gym
import jax
import jax.numpy as jnp

from ensemble.a2c import A2CTraining
from ensemble.policy_gradient_algorithms import train_agent
from ensemble.single_agent import Agent, action_log_probs, sample_action


def test_init():
    test_envs = gym.vector.make("CartPole-v1", num_envs=2)
    test_agent = Agent(
        test_envs.single_observation_space, test_envs.single_action_space, 32
    )
    assert True
    states, _ = test_envs.reset()
    logits = test_agent.actor_forward(test_agent.state.actor_params, states)
    print(logits)

    assert tuple(logits.shape) == (test_envs.num_envs, test_envs.single_action_space.n)


def test_actions():
    test_envs = gym.vector.make("CartPole-v1", num_envs=2)
    test_agent = Agent(
        test_envs.single_observation_space, test_envs.single_action_space, 32
    )
    states, _ = test_envs.reset()
    logits = test_agent.actor_forward(test_agent.state.actor_params, states)

    key = jax.random.PRNGKey(0)
    actions = sample_action(key, logits)
    assert actions.dtype == jnp.int32
    action_log_prob = action_log_probs(logits, actions)
    assert (jnp.exp(action_log_prob) < 1).all()


def test_episode():
    test_envs = gym.vector.make("CartPole-v1", num_envs=2)
    test_agent = Agent(
        test_envs.single_observation_space, test_envs.single_action_space, 32
    )

    states, _ = test_envs.reset()
    A2CTraining().episode(jax.random.PRNGKey(0), test_agent, test_envs, states)
    assert True


def test_train():
    test_envs = gym.vector.make("CartPole-v1", num_envs=2)
    test_agent = Agent(
        test_envs.single_observation_space, test_envs.single_action_space, 32
    )
    training_params = A2CTraining()
    train_agent(test_agent, test_envs, training_params, "~/Documents/ensemble/plots")
    assert True
