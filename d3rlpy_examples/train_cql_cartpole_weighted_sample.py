from typing import List
from functools import reduce

import gym
import numpy as np
import d3rlpy
from d3rlpy.algos import DiscreteCQLConfig

from weighted_datasets import ReturnWeightedReplayBufferWrapper, AdvantageWeightedReplayBufferWrapper

# get CartPole dataset
dataset, env = d3rlpy.datasets.get_cartpole()

"""
Wrap any D3rlpy dataset with either ReturnWeightedReplayBufferWrapper or AdvantageWeightedReplayBufferWrapper
"""
# dataset = ReturnWeightedReplayBufferWrapper(dataset)
dataset = AdvantageWeightedReplayBufferWrapper(dataset)

cql = d3rlpy.algos.DiscreteCQLConfig().create(device="cpu")

# start training
cql.fit(
    dataset,
    n_steps=10000,
    n_steps_per_epoch=1000,
    evaluators={
        'environment': d3rlpy.metrics.EnvironmentEvaluator(env), # evaluate with CartPole-v1 environment
    },
)

# evaluate
d3rlpy.metrics.evaluate_qlearning_with_environment(cql, env)