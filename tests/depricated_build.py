import gymnasium as gym
import haiku as hk
import jax
import numpy as np
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from ensemble.agent import Agent
from ensemble.policy_gradient_algorithms import A2CTraining, train_agent


def agent_creation(test_env):
    test_agent = Agent(
        test_env.single_observation_space, test_env.single_action_space, internal_dim=32
    )
    assert test_agent is not None
    return test_agent


def agent_eval(test_agent: Agent, test_env: gym.vector.VectorEnv):
    state, _ = test_env.reset()
    actor_params = test_agent.state.actor_params
    logits = test_agent.actor_forward(actor_params, state)
    assert tuple(logits.shape) == (
        test_env.num_envs,
        test_env.single_action_space.n,
    ), f"""Logits should be (num_envs, action_dim):
    ({test_env.num_envs}, {test_env.single_action_space})
    but are ({tuple(logits.shape)})"""


# def test_full():
#     test_env = gym.vector.make("CartPole-v1", num_envs=2)
#     test_agent = agent_creation(test_env)
#     agent_eval(test_agent, test_env)


def test_action():
    key = jax.random.PRNGKey(0)
    test_env = RecordEpisodeStatistics(
        gym.vector.make("CartPole-v1", num_envs=2, asynchronous=True)
    )
    test_agent = Agent(
        test_env.single_observation_space, test_env.single_action_space, 32
    )
    state, _ = test_env.reset()
    actor_params = test_agent.state.actor_params
    logits = test_agent.actor_forward(actor_params, state)
    action = test_agent.get_action(key, logits)
    # assert tuple(action.shape) == (
    #     test_env.num_envs,
    # ), f"""Action should be (num_envs):
    # ({test_env.num_envs})
    # but are ({tuple(action.shape)})"""
    #
    obs, rew, term, trunc, info = test_env.step(np.array(action).astype(np.int32))
    assert tuple(obs.shape) == (state.shape)
    assert tuple(rew.shape) == (test_env.num_envs,)


def test_train():
    test_env = gym.vector.make("CartPole-v1", num_envs=2, asynchronous=False)
    test_agent = Agent(
        test_env.single_observation_space, test_env.single_action_space, 32
    )
    train_agent(test_agent, test_env, A2CTraining())


