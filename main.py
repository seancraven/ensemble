import gymnasium as gym
from ensemble.agent import Agent
from ensemble.policy_gradient_algorithms import A2CTraining, train_agent

if __name__ == "__main__":
    envs = gym.vector.make("CartPole-v1", num_envs=2, asynchronous=False)
    agent = Agent(envs.single_observation_space, envs.single_action_space, 32)
    train_agent(agent, envs, A2CTraining())
