from typing import List
from functools import reduce

import gym
import numpy as np
from d3rlpy.dataset import Transition, TransitionMiniBatch, Episode
from d3rlpy.dataset import ReplayBuffer

import scipy.special
from sklearn.linear_model import LinearRegression

class Wrapper:
    def  __init__(self, base_obj):
        self.base_obj = base_obj
    
    def __getattr__(self, name):
        return getattr(self.base_obj, name)

class ReturnWeightedReplayBufferWrapper(Wrapper):

    def __init__(self, 
            dataset: ReplayBuffer, 
            alpha: float = 0.1, # Temperature of the softmax. The higher, the closer to uniform distribution.
            cache_size: int = int(1e6),
        ):
        super().__init__(dataset)
        self.alpha = alpha
        self.cache_size = cache_size
        
        self.sample_probs = self._compute_sample_probs(dataset.episodes)
        assert len(dataset._buffer) == len(self.sample_probs)
        
        self._generate_cache(cache_size)

    def _compute_sample_probs(self, episodes: List[Episode]):
        G = np.asarray([ep.rewards.sum() for ep in episodes])
        T = np.asarray([ep.transition_count for ep in episodes])
        G_it = np.asarray(reduce(lambda x, y: x + y, [[G_i] * T_i for G_i, T_i in zip(G, T)]))
        w_it = (G_it - G_it.min()) / (G_it.max() - G_it.min())
        w_it = scipy.special.softmax(G_it / self.alpha)       
        return w_it

    def _generate_cache(self, cache_size: int):
        """
        Sample a large batch of transition indices with weighted sampling.
        This is needed because calling `np.random.choice` with `p` is slow.
        """
        self._cache_indices = np.random.choice(
                range(self.sample_probs.shape[0]),
                cache_size,
                p=self.sample_probs)
        self._cache_pointer = 0

    def _pop_sample_index_from_cache(self):
        idx = self._cache_indices[self._cache_pointer]
        self._cache_pointer += 1

        # Refresh cache when no element in the cache
        if self._cache_pointer == self._cache_indices.shape[0]:
            self._generate_cache(self.cache_size)
            
        return idx

    def sample_transition(self) -> Transition:
        index = self._pop_sample_index_from_cache()
        episode, transition_index = self._buffer[index]
        return self._transition_picker(episode, transition_index)

    def sample_transition_batch(self, batch_size: int) -> TransitionMiniBatch:
        return TransitionMiniBatch.from_transitions(
            [self.sample_transition() for _ in range(batch_size)]
        )

class AdvantageWeightedReplayBufferWrapper(ReturnWeightedReplayBufferWrapper):
       
    def _compute_sample_probs(self, episodes: List[Episode]):
        G = np.asarray([ep.rewards.sum() for ep in episodes])
        T = np.asarray([ep.transition_count for ep in episodes])
        G_it = np.asarray(reduce(lambda x, y: x + y, [[G_i] * T_i for G_i, T_i in zip(G, T)]))
        s0 = np.stack([ep.observations[0] for ep in episodes])
        V = LinearRegression().fit(s0, G).predict(s0)
        V_it = np.asarray(reduce(lambda x, y: x + y, [[V_i] * T_i for V_i, T_i in zip(V, T)]))
        A_it = G_it - V_it
        A_it = (A_it - A_it.min()) / (A_it.max() - A_it.min())
        w_it = scipy.special.softmax(A_it / self.alpha)
        w_it /= w_it.sum() # Avoid numerical errors
        return w_it