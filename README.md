Harnessing Mixed Offline Reinforcement Learning Datasets via Trajectory Weighting
=

# Updates
- 2023/06/15: Fix max-min normalization in AW and RW

# Installation
Follow the instruction in https://github.com/takuseno/d3rlpy


# Example usage
```
python train_iql.py --env ant --dataset_types medium-v2 random-v2 --dataset_ratios 0.1 0.9
python train_cql.py --env ant --dataset_types medium-v2 random-v2 --dataset_ratios 0.1 0.9
python train_td3bc.py --env ant --dataset_types medium-v2 random-v2 --dataset_ratios 0.1 0.9
python train_bc.py --env ant  --dataset_types medium-v2 random-v2 --dataset_ratios 0.1 0.9
```

# Development
See `AW` and `RW` classes in `weighted.py`
