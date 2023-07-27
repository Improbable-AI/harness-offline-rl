import sys
import uuid
import argparse
import d3rlpy
from sklearn.model_selection import train_test_split

import d3rlpy.preprocessing
from d3rlpy.samplers import generate_sample_weights_from_trajectories
from d3rlpy.logger import WandbLogger, D3RLPyLogger

import re
from common import maybe_setup_wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper-medium-v2')
    parser.add_argument('--sampler', type=str, default='uniform')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--wandb', type=str, default=None)
    parser.add_argument('--project', type=str, default="offline-subopt-td3bc")
    args = parser.parse_args()

    if args.wandb == "online":
        # Setup wandb
        maybe_setup_wandb(args.wandb,
            project=args.project,
            configs=vars(args),
            keys_of_interest=[
                "env",
                "seed",
                "sampler",
            ]
        )

    dataset, env = d3rlpy.datasets.get_dataset(args.env)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)
    
    sample_weights = generate_sample_weights_from_trajectories(args.env,
       args.sampler, dataset.episodes)

    td3 = d3rlpy.algos.TD3PlusBC(actor_learning_rate=3e-4,
                                critic_learning_rate=3e-4,
                                batch_size=256,
                                target_smoothing_sigma=0.2,
                                target_smoothing_clip=0.5,
                                alpha=2.5,
                                update_actor_interval=2,
                                scaler="standard",
                                use_gpu=args.gpu,                                 
                                )

    td3.fit(dataset.episodes,
            sample_weights=sample_weights,
            eval_episodes=test_episodes,            
            n_steps=1000000,
            n_steps_per_epoch=10000,
            save_interval=10,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"TD3PlusBC_{args.env}_{args.seed}_{uuid.uuid4()}",
            logger_cls=(WandbLogger if args.wandb else D3RLPyLogger),
            logger_kwargs={
                "mode": args.wandb
            }
        )


if __name__ == '__main__':
    main()
