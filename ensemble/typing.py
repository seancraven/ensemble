"""Typing Definitions for training ensembles of agents."""
from jaxtyping import Array, Float, Int, Num, PyTree

AgentParams = PyTree[Float[Array, "envs params"]]
ActorGrad = PyTree[Float[Array, "batch params"]]
CriticGrad = PyTree[Float[Array, "batch params"]]

EpisodeStates = Float[Array, "timesteps envs observation_shape"]
States = Float[Array, "envs observation_shape"]
State = Float[Array, "observation_shape"]

EpisodeActions = Int[Array, "timesteps envs action_shape"]
Actions = Int[Array, "envs action_shape"]
Action = Int[Array, "action_shape"]

EpisodeRewards = Float[Array, "timesteps envs"]
Rewards = Float[Array, "envs"]
Reward = Float[Array, ""]

EpisodeDones = Num[Array, "timesteps envs"]
DoneState = Num[Array, "timesteps envs"]
Dones = Num[Array, "envs"]

Loss = Float[Array, ""]
Advantages = Rewards
Entropy = Float[Array, "timesteps"]
