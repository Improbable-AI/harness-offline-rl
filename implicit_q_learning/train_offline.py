import os
from typing import Tuple
import re
import sys
import gym
import numpy as np
import tqdm
import pandas as pd
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from dataset_utils import make_env_and_dataset
from evaluation import evaluate
from learner import Learner

from wandb_osh.hooks import TriggerWandbSyncHook  

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('sampler', 'uniform', 'sampler name')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('track', False, 'Use wandb')
flags.DEFINE_boolean('offline', False, 'Use wandb offline')
flags.DEFINE_string('project', 'offline-subopt-iql', 'wandb project')
config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    if FLAGS.track:
        import wandb

        run = wandb.init(
            project=FLAGS.project,
            entity="improbableai_zwh",
            config=FLAGS,
            dir=FLAGS.save_dir,
            mode=("offline" if FLAGS.offline else "online"),
            settings=wandb.Settings(start_method='fork')
        )

        if FLAGS.offline:
          trigger_sync = TriggerWandbSyncHook()


    # summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb',
    #                                             str(FLAGS.seed)),
    #                                write_to_disk=True)

    # Wrap dataset with weighted sampler
    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed,
                FLAGS.sampler, FLAGS.max_steps, FLAGS.batch_size)
    
    kwargs = dict(FLAGS.config)

    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    **kwargs)

    eval_returns = []
    metrics = {}
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            # for k, v in update_info.items():
            #     if v.ndim == 0:
            #         summary_writer.add_scalar(f'training/{k}', v, i)
            #     else:
            #         summary_writer.add_histogram(f'training/{k}', v, i)
            # summary_writer.flush()
            for k, v in update_info.items():
                if v.ndim == 0:
                    metrics[f'training/{k}'] = v

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            # for k, v in eval_stats.items():
            #     summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            # summary_writer.flush()

            for k, v in eval_stats.items():
                metrics[f'evaluation/average_{k}s'] = v

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])
        
        if (i % FLAGS.log_interval == 0 or i % FLAGS.eval_interval == 0) and FLAGS.track:
            wandb.log(metrics)

        if FLAGS.track and FLAGS.offline:
          trigger_sync()


if __name__ == '__main__':
    app.run(main)
