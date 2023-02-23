import random
import gym
import numpy as np

import d3rlpy
from d3rlpy.datasets import MDPDataset

def mix_mdp_mujoco_datasets(env_id, n, dataset_types, ratios):
  assert len(dataset_types) == len(ratios)
  assert sum(ratios) == 1

  print(f"Dataset size: {n}")
  if np.all(np.asarray(ratios) != 1.0):
    episodes = []
    for dataset_type, ratio in zip(dataset_types, ratios):
      dataset_id = f"{env_id}-{dataset_type}"

      print("Env:", env_id)
      if env_id in ["pen", "hammer", "door", "relocate"]:
        dataset, env = d3rlpy.datasets.get_d4rl(f"{env_id}-{dataset_type}")
      else:
        dataset, env = d3rlpy.datasets.get_dataset(f"{env_id}-{dataset_type}")

      num_transitions = 0
      print(f"Mix {ratio} ({n * ratio}) data from {dataset_type}.")
      while num_transitions < int(n * ratio):
        episode = random.choice(dataset.episodes)
        episodes.append(episode)
        num_transitions += len(episode.transitions)

    observations = []
    actions = []
    rewards = []
    terminals = []
    episode_terminals = []

    ep_rets = []
    for episode in episodes:
      ep_rets.append(episode.rewards.sum())
      for idx, transition in enumerate(episode):
        observations.append(transition.observation)
        if isinstance(env.action_space, gym.spaces.Box):
          actions.append(np.reshape(transition.action, env.action_space.shape))
        else:
          actions.append(transition.action)
        rewards.append(transition.reward)
        terminals.append(transition.terminal)
        episode_terminals.append(idx == len(episode) - 1)
    print(f"Max/Mean/Median/Min: {np.max(ep_rets)}/{np.mean(ep_rets)}/{np.median(ep_rets)}/{np.min(ep_rets)}")
    dataset = MDPDataset(
      observations=np.stack(observations),
      actions=np.stack(actions),
      rewards=np.stack(rewards),
      terminals=np.stack(terminals).astype(float),
      episode_terminals=np.stack(episode_terminals).astype(float))
  else:
    for dataset_type, ratio in zip(dataset_types, ratios):
      if ratio == 1.0:
        if env_id in ["pen", "hammer", "door", "relocate"]:
          dataset, env = d3rlpy.datasets.get_d4rl(f"{env_id}-{dataset_type}")
        else:
          dataset, env = d3rlpy.datasets.get_dataset(f"{env_id}-{dataset_type}")
        break
    ep_rets = []
    for episode in dataset.episodes:
      ep_rets.append(episode.rewards.sum())
    print(f"Max/Mean/Median/Min: {np.max(ep_rets)}/{np.mean(ep_rets)}/{np.median(ep_rets)}/{np.min(ep_rets)}")
  return env, dataset