import random

import gymnasium as gym

from ensemble.policy_gradient.a2c import A2CTraining
from ensemble.policy_gradient.base import train_agent
from ensemble.single_agent import Agent

if __name__ == "__main__":
    random.seed(0)
    envs = gym.vector.make("CartPole-v1", num_envs=10)
    NUM_SEEDS = 1

    agent = Agent(envs.single_observation_space, envs.single_action_space, 64)
    training_params = [
        A2CTraining(num_episodes=300, seed=random.randint(0, 1000))
        for _ in range(NUM_SEEDS)
    ]
    for training in training_params:
        train_agent(
            agent,
            envs,
            training,
            "/home/sean/Documents/ms_proj/ensemble/experiments/tests",
        )
