import gymnasium as gym
import haiku as hk

from ensemble.agent import Agent


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


def test_full():
    test_env = gym.vector.make("CartPole-v1", num_envs=2)
    test_agent = agent_creation(test_env)
    agent_eval(test_agent, test_env)
