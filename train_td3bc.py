import argparse
import d3rlpy
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--sampler', type=str, default="uniform")
    parser.add_argument('--dataset_size', type=int, default=int(1e6))
    parser.add_argument('--dataset_types', type=str, nargs="+", required=True)
    parser.add_argument('--dataset_ratios', type=float, nargs="+", required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()

    from common import wrap_sampler
    from datasets import mix_mdp_mujoco_datasets
    env, dataset = mix_mdp_mujoco_datasets(args.env,
                      n=args.dataset_size,
                      dataset_types=args.dataset_types,
                      ratios=args.dataset_ratios)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    td3 = wrap_sampler(args.sampler, d3rlpy.algos.TD3PlusBC)(actor_learning_rate=3e-4,
                                 critic_learning_rate=3e-4,
                                 batch_size=256,
                                 target_smoothing_sigma=0.2,
                                 target_smoothing_clip=0.5,
                                 alpha=2.5,
                                 update_actor_interval=2,
                                 scaler="standard",
                                 use_gpu=args.gpu)

    td3.fit(dataset.episodes,
            eval_episodes=test_episodes,
            n_steps=500000,
            n_steps_per_epoch=1000,
            save_interval=10,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"TD3PlusBC_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}")


if __name__ == '__main__':
    main()