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

import gym
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

import scipy.special
from functools import reduce
from weighted_iterator import WeightedRandomIterator
from sklearn.linear_model import LinearRegression

class WeightedSampleAlgoBase(object):

  def fitter(
        self,
        dataset: Union[List[Episode], List[Transition], MDPDataset],
        n_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,
        n_steps_per_epoch: int = 10000,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        eval_episodes: Optional[List[Episode]] = None,
        save_interval: int = 1,
        scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ] = None,
        shuffle: bool = True,
        callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
    ) -> Generator[Tuple[int, Dict[str, float]], None, None]:
        """Iterate over epochs steps to train with the given dataset. At each
             iteration algo methods and properties can be changed or queried.
        .. code-block:: python
            for epoch, metrics in algo.fitter(episodes):
                my_plot(metrics)
                algo.save_model(my_path)
        Args:
            dataset: offline dataset to train.
            n_epochs: the number of epochs to train.
            n_steps: the number of steps to train.
            n_steps_per_epoch: the number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            save_metrics: flag to record metrics in files. If False,
                the log directory is not created and the model parameters are
                not saved during training.
            experiment_name: experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: flag to add timestamp string to the last of
                directory name.
            logdir: root directory name to save logs.
            verbose: flag to show logged information on stdout.
            show_progress: flag to show progress bar for iterations.
            tensorboard_dir: directory to save logged information in
                tensorboard (additional to the csv data).  if ``None``, the
                directory will not be created.
            eval_episodes: list of episodes to test.
            save_interval: interval to save parameters.
            scorers: list of scorer functions used with `eval_episodes`.
            shuffle: flag to shuffle transitions on each epoch.
            callback: callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
        Returns:
            iterator yielding current epoch and metrics dict.
        """

        transitions = []
        if isinstance(dataset, MDPDataset):
            for episode in dataset.episodes:
                transitions += episode.transitions
        elif not dataset:
            raise ValueError("empty dataset is not supported.")
        elif isinstance(dataset[0], Episode):
            for episode in cast(List[Episode], dataset):
                transitions += episode.transitions
        elif isinstance(dataset[0], Transition):
            transitions = list(cast(List[Transition], dataset))
        else:
            raise ValueError(f"invalid dataset type: {type(dataset)}")

        # check action space
        if self.get_action_type() == ActionSpace.BOTH:
            pass
        elif transitions[0].is_discrete:
            assert (
                self.get_action_type() == ActionSpace.DISCRETE
            ), DISCRETE_ACTION_SPACE_MISMATCH_ERROR
        else:
            assert (
                self.get_action_type() == ActionSpace.CONTINUOUS
            ), CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR

        iterator: TransitionIterator
        if n_epochs is None and n_steps is not None:
            assert n_steps >= n_steps_per_epoch
            n_epochs = n_steps // n_steps_per_epoch
            iterator = WeightedRandomIterator(
                transitions,
                n_steps_per_epoch,
                batch_size=self._batch_size,
                probs=self._compute_probs(dataset),
                n_steps=self._n_steps,
                gamma=self._gamma,
                n_frames=self._n_frames,
                real_ratio=self._real_ratio,
                generated_maxlen=self._generated_maxlen,
            )
            LOG.debug("RandomIterator is selected.")
        elif n_epochs is not None and n_steps is None:
            iterator = RoundIterator(
                transitions,
                batch_size=self._batch_size,
                n_steps=self._n_steps,
                gamma=self._gamma,
                n_frames=self._n_frames,
                real_ratio=self._real_ratio,
                generated_maxlen=self._generated_maxlen,
                shuffle=shuffle,
            )
            LOG.debug("RoundIterator is selected.")
        else:
            raise ValueError("Either of n_epochs or n_steps must be given.")

        # setup logger
        logger = self._prepare_logger(
            save_metrics,
            experiment_name,
            with_timestamp,
            logdir,
            verbose,
            tensorboard_dir,
        )

        # add reference to active logger to algo class during fit
        self._active_logger = logger

        # initialize scaler
        if self._scaler:
            LOG.debug("Fitting scaler...", scaler=self._scaler.get_type())
            self._scaler.fit(transitions)

        # initialize action scaler
        if self._action_scaler:
            LOG.debug(
                "Fitting action scaler...",
                action_scaler=self._action_scaler.get_type(),
            )
            self._action_scaler.fit(transitions)

        # initialize reward scaler
        if self._reward_scaler:
            LOG.debug(
                "Fitting reward scaler...",
                reward_scaler=self._reward_scaler.get_type(),
            )
            self._reward_scaler.fit(transitions)

        # instantiate implementation
        if self._impl is None:
            LOG.debug("Building models...")
            transition = iterator.transitions[0]
            action_size = transition.get_action_size()
            observation_shape = tuple(transition.get_observation_shape())
            self.create_impl(
                self._process_observation_shape(observation_shape), action_size
            )
            LOG.debug("Models have been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        # save hyperparameters
        self.save_params(logger)

        # refresh evaluation metrics
        self._eval_results = defaultdict(list)

        # refresh loss history
        self._loss_history = defaultdict(list)

        # training loop
        total_step = 0
        for epoch in range(1, n_epochs + 1):

            # dict to add incremental mean losses to epoch
            epoch_loss = defaultdict(list)

            range_gen = tqdm(
                range(len(iterator)),
                disable=not show_progress,
                desc=f"Epoch {int(epoch)}/{n_epochs}",
            )

            iterator.reset()

            for itr in range_gen:

                # generate new transitions with dynamics models
                new_transitions = self.generate_new_data(
                    transitions=iterator.transitions,
                )
                if new_transitions:
                    iterator.add_generated_transitions(new_transitions)
                    LOG.debug(
                        f"{len(new_transitions)} transitions are generated.",
                        real_transitions=len(iterator.transitions),
                        fake_transitions=len(iterator.generated_transitions),
                    )

                with logger.measure_time("step"):
                    # pick transitions
                    with logger.measure_time("sample_batch"):
                        batch = next(iterator)

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = self.update(batch)

                    # record metrics
                    for name, val in loss.items():
                        logger.add_metric(name, val)
                        epoch_loss[name].append(val)

                    # update progress postfix with losses
                    if itr % 10 == 0:
                        mean_loss = {
                            k: np.mean(v) for k, v in epoch_loss.items()
                        }
                        range_gen.set_postfix(mean_loss)

                total_step += 1

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            # save loss to loss history dict
            self._loss_history["epoch"].append(epoch)
            self._loss_history["step"].append(total_step)
            for name, vals in epoch_loss.items():
                if vals:
                    self._loss_history[name].append(np.mean(vals))

            if scorers and eval_episodes:
                self._evaluate(eval_episodes, scorers, logger)

            # save metrics
            metrics = logger.commit(epoch, total_step)

            # save model parameters
            if epoch % save_interval == 0:
                logger.save_model(total_step, self)

            yield epoch, metrics

        # drop reference to active logger since out of fit there is no active
        # logger
        self._active_logger.close()
        self._active_logger = None


class RW(WeightedSampleAlgoBase):

  alpha = 0.1

  def _compute_probs(self, episodes):
    G = np.asarray([ep.rewards.sum() for ep in episodes])
    T = np.asarray([len(ep.transitions) for ep in episodes])
    G_it = np.asarray(reduce(lambda x, y: x + y, [[G_i] * T_i for G_i, T_i in zip(G, T)]))
    w_it = scipy.special.softmax(G_it / self.alpha)
    w_it = (w_it - w_it.min()) / (w_it.max() - w_it.min())
    return w_it


class AW(WeightedSampleAlgoBase):

  alpha = 0.1

  def _compute_probs(self, episodes):
    G = np.asarray([ep.rewards.sum() for ep in episodes])
    T = np.asarray([len(ep.transitions) for ep in episodes])
    G_it = np.asarray(reduce(lambda x, y: x + y, [[G_i] * T_i for G_i, T_i in zip(G, T)]))
    s0 = np.stack([ep.observations[0] for ep in episodes])
    V = LinearRegression().fit(s0, G).predict(s0)
    V_it = np.asarray(reduce(lambda x, y: x + y, [[V_i] * T_i for V_i, T_i in zip(V, T)]))
    A_it = G_it - V_it
    A_it = (A_it - A_it.min()) / (A_it.max() - A_it.min())
    w_it = scipy.special.softmax(A_it / self.alpha)
    w_it /= w_it.sum() # Numerical errors
    return w_it
