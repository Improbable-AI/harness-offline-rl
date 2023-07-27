Harnessing Mixed Offline Reinforcement Learning Datasets via Trajectory Weighting
=

This repo is the official implementation of [Harnessing Mixed Offline Reinforcement Learning Datasets via Trajectory Weighting](https://openreview.net/pdf?id=OhUAblg27z).

**Abstract:** Most offline reinforcement learning (RL) algorithms return a target policy maximizing a trade-off between (1) the expected performance gain over the behavior policy that collected the dataset, and (2) the risk stemming from the out-of-distribution-ness of the induced state-action occupancy. It follows that the performance of the target policy is strongly related to the performance of the behavior policy and, thus, the trajectory return distribution of the dataset. We show that in mixed datasets consisting of mostly low-return trajectories and minor high-return trajectories, state-of-the-art offline RL algorithms are overly restrained by low-return trajectories and fail to exploit high-performing trajectories to the fullest. To overcome this issue, we show that, in deterministic MDPs with stochastic initial states, the dataset sampling can be re-weighted to induce an artificial dataset whose behavior policy has a higher return. This re-weighted sampling strategy may be combined with any offline RL algorithm. We further analyze that the opportunity for performance improvement over the behavior policy correlates with the positive-sided variance of the returns of the trajectories in the dataset. We empirically show that while CQL, IQL, and TD3+BC achieve only a part of this potential policy improvement, these same algorithms combined with our reweighted sampling strategy fully exploit the dataset. Furthermore, we empirically demonstrate that, despite its theoretical limitation, the approach may still be efficient in stochastic environments. 

# Updates
- 2023/07/27: Added implementation on official codebase of CQL, IQL, and TD3+BC and their results
- 2023/06/15: Fix max-min normalization in AW and RW

# Original implementation 
See `d3rlpy_impls` in this repo.

# Installation
- Download and install [suboptimal_offline_datasets](https://github.com/williamd4112/suboptimal_offline_datasets/tree/master)
- Please refer to `JaxCQL`, `implicit_q_learning`, and `td3bc` directoriees.

# Benchmark in [`JaxCQL`](https://github.com/young-geng/JaxCQL) implementation
Run `python experiments/d4rl/run_cql.py --mode {screen,sbatch,local} --n_gpus {list of gpu ids} --n_jobs {number of parallel jobs}`
|                                   |      Uniform |        AW |   Top-10\% |
|:----------------------------------|-------------:|----------:|-----------:|
| hopper-random-v2                  |   9.24441    |   7.93173 |   7.84564  |
| hopper-medium-expert-v2           | 104.486      | 104.562   | 107.764    |
| hopper-medium-replay-v2           |  85.9022     |  96.9812  |  91.1704   |
| hopper-full-replay-v2             | 100.51       | 101.231   | 102.101    |
| hopper-medium-v2                  |  62.1446     |  67.6899  |  65.6824   |
| hopper-expert-v2                  | 107.182      | 108.233   | 108.583    |
| halfcheetah-random-v2             |  21.3687     |  16.2642  |   2.96156  |
| halfcheetah-medium-expert-v2      |  71.1836     |  89.4178  |  73.4318   |
| halfcheetah-medium-replay-v2      |  45.2666     |  44.6606  |  42.2252   |
| halfcheetah-full-replay-v2        |  75.069      |  76.7151  |  74.9687   |
| halfcheetah-medium-v2             |  46.5255     |  46.5397  |  45.3898   |
| halfcheetah-expert-v2             |  81.2596     |  87.6424  |  65.1762   |
| ant-random-v2                     |   7.55607    |   7.72678 |   7.11494  |
| ant-medium-expert-v2              | 128.784      | 129.861   | 127.835    |
| ant-medium-replay-v2              |  96.0448     |  88.5638  |  82.6923   |
| ant-full-replay-v2                | 129.303      | 124.512   | 127.528    |
| ant-medium-v2                     |  99.996      |  92.3009  |  93.9649   |
| ant-expert-v2                     | 124.453      | 131.999   | 129.991    |
| walker2d-random-v2                |   6.13915    |   4.78385 |  13.2343   |
| walker2d-medium-expert-v2         | 109.628      | 109.348   | 108.934    |
| walker2d-medium-replay-v2         |  74.4345     |  78.0994  |  71.8967   |
| walker2d-full-replay-v2           |  91.182      |  88.5198  |  90.472    |
| walker2d-medium-v2                |  82.2539     |  81.341   |  78.2447   |
| walker2d-expert-v2                | 108.912      | 108.502   | 108.985    |
| antmaze-umaze-v0                  |  76          |  77.3333  |  69.3333   |
| antmaze-umaze-diverse-v0          |  48          |  36       |  24.7778   |
| antmaze-medium-diverse-v0         |   0          |   6       |   0        |
| antmaze-medium-play-v0            |   2.4        |  10.6667  |   0        |
| antmaze-large-diverse-v0          |   0          |   2       |   2.66667  |
| antmaze-large-play-v0             |   0.4        |   1.33333 |   0        |
| kitchen-complete-v0               |  27.8333     |  30.25    |   9.5      |
| kitchen-partial-v0                |  45          |  36       |  50.7619   |
| kitchen-mixed-v0                  |  41.5        |  50.5     |  52        |
| pen-human-v1                      |   1.60028    |  -2.98219 |   2.63752  |
| pen-cloned-v1                     |  -1.32959    |  -2.47556 |   8.60352  |
| hammer-human-v1                   |  -6.95849    |  -6.98119 |  -6.94442  |
| hammer-cloned-v1                  |  -6.96791    |  -6.96577 |  -6.95725  |
| door-human-v1                     |  -9.40789    |  -9.41327 |  -5.06673  |
| door-cloned-v1                    |  -9.41004    |  -9.39717 |  21.6284   |
| relocate-human-v1                 |   2.13075    |  -2.14781 |  -0.817744 |
| relocate-cloned-v1                |  -2.32878    |  -2.35066 |   0.202246 |
| ant-random-medium-1\%-v2          |  37.084      |  73.5515  |   6.71751  |
| ant-random-medium-5\%-v2          |  53.1406     |  86.0782  |  76.6323   |
| ant-random-medium-10\%-v2         |  82.8248     |  88.078   |  93.117    |
| ant-random-medium-50\%-v2         |  97.3879     |  94.4633  |  95.7075   |
| ant-random-expert-1\%-v2          |  10.0283     |  77.7341  |   4.95247  |
| ant-random-expert-5\%-v2          |  35.1951     | 114.802   |  65.0137   |
| ant-random-expert-10\%-v2         |  48.3196     | 119.969   | 110.044    |
| ant-random-expert-50\%-v2         | 117.347      | 130.603   | 125.519    |
| hopper-random-medium-1\%-v2       |   0.603641   |  55.0556  |  62.1559   |
| hopper-random-medium-5\%-v2       |   1.45364    |  62.0875  |  42.5708   |
| hopper-random-medium-10\%-v2      |   1.55794    |  66.5722  |  66.8342   |
| hopper-random-medium-50\%-v2      |  22.6365     |  46.6942  |  66.9086   |
| hopper-random-expert-1\%-v2       |  17.4588     |  59.5837  |  17.4444   |
| hopper-random-expert-5\%-v2       |  16.2593     |  99.7196  |  40.12     |
| hopper-random-expert-10\%-v2      |  14.8979     | 109.654   |  46.6203   |
| hopper-random-expert-50\%-v2      | 100.574      | 109.588   | 108.523    |
| halfcheetah-random-medium-1\%-v2  |  37.1308     |  39.7639  |  18.8675   |
| halfcheetah-random-medium-5\%-v2  |  41.0571     |  45.4053  |  42.6198   |
| halfcheetah-random-medium-10\%-v2 |  44.5619     |  45.8083  |  45.0028   |
| halfcheetah-random-medium-50\%-v2 |  46.5794     |  46.4837  |  45.1374   |
| halfcheetah-random-expert-1\%-v2  |  21.3933     |  26.4011  |   5.06894  |
| halfcheetah-random-expert-5\%-v2  |  24.7782     |  66.2233  |   7.92592  |
| halfcheetah-random-expert-10\%-v2 |  31.7241     |  72.639   |  75.3741   |
| halfcheetah-random-expert-50\%-v2 |  58.7239     |  80.733   |  61.4816   |
| walker2d-random-medium-1\%-v2     |   2.87035    |  41.9059  |   3.32862  |
| walker2d-random-medium-5\%-v2     |   0.00564381 |  75.0255  |  46.8232   |
| walker2d-random-medium-10\%-v2    |   0.566986   |  74.5646  |  74.0476   |
| walker2d-random-medium-50\%-v2    |  76.8756     |  82.0477  |  82.1249   |
| walker2d-random-expert-1\%-v2     |   3.98307    |  66.3455  |   5.68504  |
| walker2d-random-expert-5\%-v2     |   0.233614   | 107.682   |  32.6821   |
| walker2d-random-expert-10\%-v2    |   3.09605    | 108.105   |  34.3063   |
| walker2d-random-expert-50\%-v2    |   0.77345    | 108.56    | 108.214    |
| average                           |  42.855      |  62.7771  |  51.6177   |
| num                               |  73          |  73       |  73        |


# Benchmark in [`implicit_q_learning`](https://github.com/ikostrikov/implicit_q_learning/) implementation
Run `python experiments/d4rl/run_iql.py --mode {screen,sbatch,local} --n_gpus {list of gpu ids} --n_jobs {number of parallel jobs}`
|                                   |     Uniform |          AW |     Top-10% |
|:----------------------------------|------------:|------------:|------------:|
| hopper-random-v2                  |   7.57487   |   6.82503   |   7.99012   |
| hopper-medium-expert-v2           |  85.3593    | 111.089     | 111.804     |
| hopper-medium-replay-v2           |  86.6692    |  98.1126    |  96.0369    |
| hopper-full-replay-v2             | 108.141     | 102.137     |  88.1928    |
| hopper-medium-v2                  |  65.6634    |  58.2563    |  64.5312    |
| hopper-expert-v2                  | 109.708     | 111.026     | 110.252     |
| halfcheetah-random-v2             |  12.7194    |   7.25561   |   4.18207   |
| halfcheetah-medium-expert-v2      |  90.5977    |  94.6886    |  94.1846    |
| halfcheetah-medium-replay-v2      |  44.0456    |  43.9732    |  29.4009    |
| halfcheetah-full-replay-v2        |  73.4902    |  76.3123    |  72.3025    |
| halfcheetah-medium-v2             |  47.4538    |  47.8104    |  45.429     |
| halfcheetah-expert-v2             |  94.9409    |  95.292     |  73.7911    |
| ant-random-v2                     |  11.8518    |  12.1939    |   8.28141   |
| ant-medium-expert-v2              | 133.311     | 131.858     | 133.246     |
| ant-medium-replay-v2              |  93.8017    |  82.9131    |  71.3909    |
| ant-full-replay-v2                | 130.083     | 129.914     | 128.911     |
| ant-medium-v2                     |  99.985     |  98.8656    |  96.1812    |
| ant-expert-v2                     | 126.227     | 131.373     | 119.55      |
| walker2d-random-v2                |   6.71134   |   2.74685   |  10.4378    |
| walker2d-medium-expert-v2         | 110.088     | 109.745     | 109.752     |
| walker2d-medium-replay-v2         |  61.323     |  47.0284    |  42.2182    |
| walker2d-full-replay-v2           |  86.7943    |  84.5202    |  85.5531    |
| walker2d-medium-v2                |  77.9036    |  70.0193    |  65.3481    |
| walker2d-expert-v2                | 109.911     | 109.886     | 109.6       |
| antmaze-umaze-v0                  |  88         |  90.6667    |   0         |
| antmaze-umaze-diverse-v0          |  67.3333    |  75.3333    |   0         |
| antmaze-medium-diverse-v0         |  76         |  61.3333    |   0         |
| antmaze-medium-play-v0            |  72         |  22         |   0         |
| antmaze-large-diverse-v0          |  36.6667    |  23.3333    |   0         |
| antmaze-large-play-v0             |  43.3333    |   9.33333   |   0         |
| kitchen-complete-v0               |  62.8333    |  26.3333    |  10         |
| kitchen-partial-v0                |  47.6667    |  73.1667    |  72.3333    |
| kitchen-mixed-v0                  |  49.8333    |  47.8333    |  52.1667    |
| pen-human-v1                      |  80.4294    |  83.2065    |  36.2937    |
| pen-cloned-v1                     |  82.8524    |  89.2386    |  53.8445    |
| hammer-human-v1                   |   3.09117   |   0.529091  |   3.23893   |
| hammer-cloned-v1                  |   1.11712   |   1.3948    |   1.0312    |
| door-human-v1                     |   2.46032   |   0.591263  |   0.108776  |
| door-cloned-v1                    |   0.0421634 |   0.622028  |   2.36637   |
| relocate-human-v1                 |   0.45569   |   0.0143342 |  -0.0446024 |
| relocate-cloned-v1                |  -0.0164524 |   0.0961286 |   0.023182  |
| ant-random-medium-1\%-v2          |  17.5089    |  56.0236    |   5.05124   |
| ant-random-medium-5\%-v2          |  68.1371    |  83.3095    |  15.4062    |
| ant-random-medium-10\%-v2         |  82.0196    |  88.8323    |  40.1891    |
| ant-random-medium-50\%-v2         |  93.7056    | 101.437     |  96.8241    |
| ant-random-expert-1\%-v2          |  13.6634    |  28.5398    |   5.51636   |
| ant-random-expert-5\%-v2          |  36.3095    | 100.855     |   5.50125   |
| ant-random-expert-10\%-v2         |  73.7289    | 125.973     |  14.0018    |
| ant-random-expert-50\%-v2         | 122.526     | 128.182     | 127.748     |
| hopper-random-medium-1\%-v2       |  52.2317    |  56.0748    |  42.4189    |
| hopper-random-medium-5\%-v2       |  58.9994    |  57.0564    |  63.3525    |
| hopper-random-medium-10\%-v2      |  63.1737    |  57.0855    |  65.3362    |
| hopper-random-medium-50\%-v2      |  50.6182    |  56.248     |  57.2392    |
| hopper-random-expert-1\%-v2       |  11.1235    |  74.8142    |  16.4055    |
| hopper-random-expert-5\%-v2       |  22.7087    | 111.33      |  24.8823    |
| hopper-random-expert-10\%-v2      |  46.673     | 111.491     |  33.8834    |
| hopper-random-expert-50\%-v2      |  87.9581    | 111.654     |  92.1624    |
| halfcheetah-random-medium-1\%-v2  |  30.9543    |  13.8937    |   3.00345   |
| halfcheetah-random-medium-5\%-v2  |  39.0612    |  41.6837    |  25.9368    |
| halfcheetah-random-medium-10\%-v2 |  40.3211    |  43.0563    |  45.2577    |
| halfcheetah-random-medium-50\%-v2 |  45.337     |  47.2642    |  43.3904    |
| halfcheetah-random-expert-1\%-v2  |   4.19425   |   3.80636   |   2.31604   |
| halfcheetah-random-expert-5\%-v2  |   9.08923   |  74.0195    |   4.43776   |
| halfcheetah-random-expert-10\%-v2 |  16.9594    |  91.2643    |  81.5348    |
| halfcheetah-random-expert-50\%-v2 |  83.6928    |  94.7754    |  31.9586    |
| walker2d-random-medium-1\%-v2     |  54.6278    |  45.4309    |  39.4283    |
| walker2d-random-medium-5\%-v2     |  66.2005    |  62.7774    |  47.3474    |
| walker2d-random-medium-10\%-v2    |  63.3576    |  65.833     |  62.6242    |
| walker2d-random-medium-50\%-v2    |  70.7044    |  70.019     |  69.5619    |
| walker2d-random-expert-1\%-v2     |  20.0016    |   9.62748   |  11.433     |
| walker2d-random-expert-5\%-v2     |  25.2571    | 108.557     |  93.4991    |
| walker2d-random-expert-10\%-v2    |  64.4441    | 109.339     | 107.163     |
| walker2d-random-expert-50\%-v2    | 109.249     | 109.41      | 109.565     |
| average                           |  57.9862    |  65.8703    |  47.8672    |
| num                               |  73         |  73         |  73         |

# Benchmark in [`d3rlpy/td3_plus_bc`](https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/algos/qlearning/td3_plus_bc.py/) implementation
Run `python experiments/d4rl/run_td3bc.py --mode {screen,sbatch,local} --n_gpus {list of gpu ids} --n_jobs {number of parallel jobs}`

|                                   |    Uniform |         AW |   Top-10% |
|:----------------------------------|-----------:|-----------:|-----------:|
| hopper-random-v2                  |   8.47086  |   9.02459  |   7.54435  |
| hopper-medium-expert-v2           |  95.4239   | 105.582    | 106.54     |
| hopper-medium-replay-v2           |  64.1975   |  96.7929   |  83.446    |
| hopper-full-replay-v2             |  70.3673   | 105.576    | 102.886    |
| hopper-medium-v2                  |  59.7683   |  63.7635   |  65.178    |
| hopper-expert-v2                  | 110.527    | 111.455    | 101.469    |
| halfcheetah-random-v2             |  12.273    |  11.2596   |  10.3168   |
| halfcheetah-medium-expert-v2      |  88.8843   |  97.7455   |  88.0424   |
| halfcheetah-medium-replay-v2      |  44.6565   |  45.0612   |  29.5789   |
| halfcheetah-full-replay-v2        |  74.1024   |  77.6774   |  75.3897   |
| halfcheetah-medium-v2             |  48.3612   |  48.6312   |  48.3426   |
| halfcheetah-expert-v2             |  96.3813   |  97.5426   |  78.3743   |
| ant-random-v2                     |  35.1557   |  11.4745   |  -0.1427   |
| ant-medium-expert-v2              | 113.221    | 135.492    | 121.389    |
| ant-medium-replay-v2              | 104.063    | 100.877    |  49.9648   |
| ant-full-replay-v2                | 136.334    | 139.688    | 131.088    |
| ant-medium-v2                     | 122.862    | 120.139    |  95.3376   |
| ant-expert-v2                     | 105.413    | 124.851    |  94.4023   |
| walker2d-random-v2                |   1.17976  |   2.5178   |   2.54119  |
| walker2d-medium-expert-v2         | 109.966    | 110.177    | 110.504    |
| walker2d-medium-replay-v2         |  80.1735   |  80.7586   |  72.3934   |
| walker2d-full-replay-v2           |  93.2661   |  96.173    |  96.0086   |
| walker2d-medium-v2                |  84.3696   |  82.2883   |  76.0017   |
| walker2d-expert-v2                | 110.226    | 110.282    | 110.21     |
| antmaze-umaze-v0                  |  17.3333   |  32.3333   |  53.6667   |
| antmaze-umaze-diverse-v0          |  64.6667   |  67.3333   |  29        |
| antmaze-medium-diverse-v0         |   3.66667  |  10.6667   |   3.33333  |
| antmaze-medium-play-v0            |   0        |   1        |   2        |
| antmaze-large-diverse-v0          |   0        |   0.333333 |   0        |
| antmaze-large-play-v0             |   0        |   0        |   0        |
| kitchen-complete-v0               |   0        |   0        |   0        |
| kitchen-partial-v0                |   0        |   0.833333 |   0        |
| kitchen-mixed-v0                  |   0        |   9.83333  |   1.91667  |
| pen-human-v1                      |   3.0764   |   1.50739  |   8.77934  |
| pen-cloned-v1                     |  11.8952   |   2.2161   |  17.1098   |
| hammer-human-v1                   |   1.09588  |   0.825413 |   0.434885 |
| hammer-cloned-v1                  |   0.241099 |   0.388506 |   0.382582 |
| door-human-v1                     |  -0.33186  |  -0.337163 |  -0.336127 |
| door-cloned-v1                    |  -0.339652 |  -0.338328 |  -0.301146 |
| relocate-human-v1                 |  -0.297545 |  -0.298002 |  -0.300698 |
| relocate-cloned-v1                |  -0.301564 |  -0.188818 |  -0.301626 |
| ant-random-medium-1\%-v2          |  41.2319   |  21.6466   |  -0.515089 |
| ant-random-medium-5\%-v2          |  41.8525   |  84.2181   |   0.872358 |
| ant-random-medium-10\%-v2         |  55.6399   |  84.4948   |  29.2266   |
| ant-random-medium-50\%-v2         |  85.1923   | 114.478    | 111.468    |
| ant-random-expert-1\%-v2          |  18.6702   |  13.9958   |   0.314325 |
| ant-random-expert-5\%-v2          |  27.767    |  39.5692   |   4.80114  |
| ant-random-expert-10\%-v2         |  43.9941   |  65.9493   |  13.5619   |
| ant-random-expert-50\%-v2         |  31.634    |  97.8403   |  96.3247   |
| hopper-random-medium-1\%-v2       |  25.7469   |  49.0665   |  13.3575   |
| hopper-random-medium-5\%-v2       |  41.1588   |  57.9878   |  49.8182   |
| hopper-random-medium-10\%-v2      |  29.4417   |  56.3926   |  50.8551   |
| hopper-random-medium-50\%-v2      |  55.8159   |  64.6845   |  53.6561   |
| hopper-random-expert-1\%-v2       |  21.2567   |  36.3155   |  10.5223   |
| hopper-random-expert-5\%-v2       |  31.0256   |  97.2839   |  37.5945   |
| hopper-random-expert-10\%-v2      |  51.9751   | 107.687    |  84.4173   |
| hopper-random-expert-50\%-v2      |  87.2425   | 106.452    | 107.903    |
| halfcheetah-random-medium-1\%-v2  |  15.8028   |  14.7808   |  27.3719   |
| halfcheetah-random-medium-5\%-v2  |  21.0122   |  46.9444   |  44.0399   |
| halfcheetah-random-medium-10\%-v2 |  36.1586   |  47.8106   |  47.8439   |
| halfcheetah-random-medium-50\%-v2 |  48.4833   |  48.2601   |  47.9683   |
| halfcheetah-random-expert-1\%-v2  |   2.68735  |   3.60436  |   8.24643  |
| halfcheetah-random-expert-5\%-v2  |  20.6033   |  50.2031   |   5.69433  |
| halfcheetah-random-expert-10\%-v2 |  25.8581   |  78.3894   |  82.5897   |
| halfcheetah-random-expert-50\%-v2 |  85.7825   |  96.0325   |  69.9516   |
| walker2d-random-medium-1\%-v2     |   5.95163  |  -0.192207 |   7.81333  |
| walker2d-random-medium-5\%-v2     |  14.5051   |  74.6761   |   1.8669   |
| walker2d-random-medium-10\%-v2    |   9.8683   |  74.2413   |   2.33969  |
| walker2d-random-medium-50\%-v2    |  21.8594   |  78.2168   |  14.6984   |
| walker2d-random-expert-1\%-v2     |   5.30913  |  15.4112   |   0.282913 |
| walker2d-random-expert-5\%-v2     |   6.39543  |  73.1095   |   3.51582  |
| walker2d-random-expert-10\%-v2    |   2.95956  | 110.096    |   1.45098  |
| walker2d-random-expert-50\%-v2    |   8.73495  | 110.265    |   0.777813 |
| average                           |  40.9858   |  56.587    |  39.7646   |
| num                               |  73        |  73        |  73        |