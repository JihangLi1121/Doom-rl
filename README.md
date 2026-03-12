# VizDoom RL Agent

Reinforcement learning agents for VizDoom using PPO with reward shaping, built on Stable-Baselines3 and a GPU-accelerated Docker environment.

## Project Structure

```
├── Dockerfile              # CUDA 11.8 + Python 3.10 container
├── docker-compose.yml      # GPU-enabled service config
├── Makefile                # Build, run, and management commands
├── requirements.txt        # Python dependencies
├── scripts/                # Training and utility scripts
│   ├── train_ppo_baseline.py
│   ├── train_ppo_improved.py
│   ├── train_ppo_basic_tutorial.py
│   ├── train_ppo_center.py
│   ├── train_ppo_corridor.py
│   ├── train_ppo_corridor_revert.py
│   ├── record_video.py
│   ├── visualize.py
│   └── test_environment.py
├── notebooks/              # Jupyter tutorials
│   ├── VizDoom-Basic-Tutorial.ipynb
│   └── VizDoom-DeadlyCorridor-Tutorial.ipynb
├── configs/                # VizDoom configuration
│   └── _vizdoom.ini
├── tests/                  # Test suite
├── models/                 # Trained models (gitignored)
├── logs/                   # TensorBoard logs (gitignored)
└── videos/                 # Recorded gameplay (gitignored)
```

## Quick Start

### Prerequisites

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA support

### Build and Run

```bash
make build       # Build the Docker image
make run         # Start the container
make shell       # Open a shell inside the container
```

### Train a PPO Agent

```bash
# Inside the container
python scripts/train_ppo_improved.py
```

### Monitor Training

```bash
make tensorboard   # Start TensorBoard at http://localhost:6006
make jupyter       # Start Jupyter Lab at http://localhost:8888
```

## VizDoom Environments

Training scripts target three VizDoom scenarios:

| Scenario | Description | Script |
|----------|-------------|--------|
| **Basic** | Shoot a single enemy in a room | `train_ppo_baseline.py`, `train_ppo_improved.py` |
| **Deadly Corridor** | Navigate a corridor full of enemies | `train_ppo_corridor.py` |
| **Defend the Center** | Survive waves of enemies from all directions | `train_ppo_center.py` |

## PPO Improvements

The baseline PPO agent performed poorly (avg reward: -105, ~38 step survival). Key improvements:

### Reward Shaping
- **Survival bonus**: +0.2 per step alive
- **Death penalty**: -10.0 on death
- **Kill bonus**: +10.0 per enemy killed
- **Reward normalization** via running statistics for stable gradients

### Optimized Hyperparameters

| Parameter | Baseline | Improved | Reason |
|-----------|----------|----------|--------|
| Learning Rate | 2.5e-4 (fixed) | 3e-4 (linear decay) | Better exploration then fine-tuning |
| n_steps | 2048 | 4096 | More diverse experience per update |
| batch_size | 64 | 256 | Stable gradient estimates |
| ent_coef | 0.01 | 0.02 | More exploration |
| Network | Default | 256x256 | More capacity for visual patterns |
| Frame Skip | 4 | 2 | Finer action control |
| Total Steps | 100k | 500k | Sufficient learning time for visual RL |

### Results

| Metric | Baseline | Improved |
|--------|----------|----------|
| Average Reward | -105.21 | +20 to +100 |
| Survival Time | ~38 steps | 100-300+ steps |
| Learning | Slow, unstable | Faster, stable |

## Make Commands

| Command | Description |
|---------|-------------|
| `make build` | Build the Docker image |
| `make run` | Start the container |
| `make stop` | Stop the container |
| `make shell` | Shell into the container |
| `make test` | Run environment tests |
| `make tensorboard` | Start TensorBoard |
| `make jupyter` | Start Jupyter Lab |
| `make logs` | View container logs |
| `make clean` | Remove container and volumes |

## Model Artifacts

After training, models are saved to `models/`:

```
models/<env>_<variant>/
├── final_model.zip
├── vec_normalize.pkl
├── best_model/
│   └── best_model.zip
├── eval_logs/
│   └── evaluations.npz
└── ppo_*_steps.zip          # Periodic checkpoints
```

## Troubleshooting

- **OOM**: Reduce `batch_size` to 128 or `n_steps` to 2048
- **Slow training**: Reduce `total_timesteps` for testing
- **Rewards not improving**: Train longer, adjust reward shaping coefficients, check TensorBoard entropy > 0
- **Agent dies immediately**: Normal in early training (<50k steps)

## References

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [VizDoom](http://vizdoom.cs.put.edu.pl/)
- [Reward Shaping (Ng et al., 1999)](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
