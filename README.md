# Recursive Skip-Step Planning (RSP)

This repository is the official implementation of Recursive Skip-step Planning in JAX, attached with the paper entitled "Are Expressive Models Truly Necessary for Offline RL?
".

## Getting Started

### Prerequisites

1. [Install JAX](https://github.com/google/jax#installation)

2. [Install W&B](https://github.com/wandb/wandb) and log in to your account to view metrics

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running experiment sets

To reproduce the D4RL benchmark results:

```bash
# Adroit
python exp_launcher.py include=experiment_conf/adroit.yaml
# AntMaze
python exp_launcher.py include=experiment_conf/antmaze.yaml
# MuJoCo
python exp_launcher.py include=experiment_conf/mujoco.yaml
# Franka Kitchen
python exp_launcher.py include=experiment_conf/kitchen.yaml
```

Metrics are uploaded to W&B. The final performance is also stored in the `metrics` folder. To consolidate into a markdown report, run the following command:

```bash
python view_metrics.py
```

This report will be saved in `report.md`.

**Note**: The experiment launcher will automatically allocate all idle GPUs on your machine and run experiments in parallel.

## License

Apache License 2.0

## Citation
```
@inproceedings{wang_niu2024rsp,
          title={Are Expressive Models Truly Necessary for Offline RL?},
          author={Wang, Guan and Niu, Haoyi and Li, Jianxiong and Jiang, Li and Hu, Jianming and Zhan, Xianyuan},
          booktitle={Under Review},
          year={2024}
        }
```
