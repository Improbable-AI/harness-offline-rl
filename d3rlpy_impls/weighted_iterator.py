import copy
import json
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import gym, os
import numpy as np
from tqdm.auto import tqdm

from d3rlpy.argument_utility import (
    ActionScalerArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_action_scaler,
    check_reward_scaler,
    check_scaler,
)
from d3rlpy.constants import (
    CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR,
    DISCRETE_ACTION_SPACE_MISMATCH_ERROR,
    IMPL_NOT_INITIALIZED_ERROR,
    ActionSpace,
)
from d3rlpy.context import disable_parallel
from d3rlpy.dataset import Episode, MDPDataset, Transition, TransitionMiniBatch
from d3rlpy.decorators import pretty_repr
from d3rlpy.gpu import Device
from d3rlpy.iterators import RandomIterator, RoundIterator, TransitionIterator
from d3rlpy.logger import LOG, D3RLPyLogger
from d3rlpy.models.encoders import EncoderFactory, create_encoder_factory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory, create_q_func_factory
from d3rlpy.online.utility import get_action_size_from_env
from d3rlpy.preprocessing import (
    ActionScaler,
    RewardScaler,
    Scaler,
    create_action_scaler,
    create_reward_scaler,
    create_scaler,
)

import d3rlpy

from typing import List, cast

import numpy as np
import functools

from d3rlpy.dataset import Transition, TransitionMiniBatch
from d3rlpy.iterators.base import TransitionIterator

class WeightedRandomIterator(TransitionIterator):

    _n_steps_per_epoch: int

    def __init__(
        self,
        transitions: List[Transition],
        n_steps_per_epoch: int,
        batch_size: int,
        probs: List[float],
        n_steps: int = 1,
        gamma: float = 0.99,
        n_frames: int = 1,
        real_ratio: float = 1.0,
        generated_maxlen: int = 100000,
    ):
        self.probs = probs
        print(self.probs.shape, len(transitions))
        super().__init__(
            transitions=transitions,
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            n_frames=n_frames,
            real_ratio=real_ratio,
            generated_maxlen=generated_maxlen,
        )
        self._n_steps_per_epoch = n_steps_per_epoch
        self.n_transitions_per_epoch = self._batch_size * n_steps_per_epoch
        self.indices = range(len(self._transitions))
        self.indices_count = 0

    def get_next(self) -> Transition:
        if self._has_finished():
            raise StopIteration
        return self._next()

    def _next(self) -> Transition:
        index = cast(int, self.indices_cache[self.indices_count])
        transition = self._transitions[index]

        self.indices_count = (self.indices_count + 1) % self.n_transitions_per_epoch


        return transition

    def __next__(self) -> TransitionMiniBatch:
        if len(self._generated_transitions) > 0:
            real_batch_size = self._real_batch_size
            fake_batch_size = self._batch_size - self._real_batch_size
            transitions = [self.get_next() for _ in range(real_batch_size)]
            transitions += self._sample_generated_transitions(fake_batch_size)
        else:
            transitions = [self.get_next() for _ in range(self._batch_size)]

        batch = TransitionMiniBatch(
            transitions,
            n_frames=self._n_frames,
            n_steps=self._n_steps,
            gamma=self._gamma,
        )

        self._count += 1

        return batch

    def _has_finished(self) -> bool:
        return self._count >= self._n_steps_per_epoch

    def __len__(self) -> int:
        return self._n_steps_per_epoch

    def set_probs(self, probs):
        self.probs = probs

    def _reset(self) -> None:
        self.indices_cache = np.random.choice(range(len(self._transitions)), self.n_transitions_per_epoch, p=self.probs)
        self.history_indices = []