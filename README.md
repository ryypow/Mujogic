# Mujogic

A reinforcement learning framework for robotic hand manipulation using MuJoCo physics simulation. The project focuses on training agents to perform dexterous in-hand object manipulation tasks, specifically rotating a cylindrical object (can) using a simulated robotic hand.

## Features

- **MuJoCo Physics Simulation**: High-fidelity physics engine for realistic robotic hand dynamics
- **Gymnasium Environment**: Custom environment following OpenAI Gym interface for RL training
- **Multiple RL Approaches**:
  - Deep Q-Network (DQN) with PyTorch for continuous action spaces
  - Tabular Q-Learning for discrete state/action spaces
- **UR5e + LEAP Hand**: Simulated robotic arm with dexterous hand for manipulation tasks
- **Configurable Reward Functions**: Progress-based and survival rewards for stable training

## Architecture

```
Mujogic/
├── src_deepq/          # Deep Q-Network implementation
│   ├── DQNagent.py     # Neural network-based DQN agent
│   ├── inhand_env.py   # Gymnasium environment for in-hand manipulation
│   ├── inhand_train.py # Training script for DQN
│   ├── simulation.py   # MuJoCo simulation wrapper
│   └── *.xml           # Robot URDF/MJCF scene files
├── src_tabular/        # Tabular Q-Learning implementation
│   ├── RLagent.py      # Q-table based agent
│   ├── inhand_env.py   # Environment with discrete actions
│   └── inhand_train.py # Training script for tabular RL
├── mvp/                # Minimal viable product experiments
├── archive/            # Archived experiments and iterations
└── versions/           # Version snapshots
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for DQN training)

### Setup

```bash
# Clone the repository
git clone https://github.com/ryypow/Mujogic.git
cd Mujogic

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training with Deep Q-Network

```bash
cd src_deepq
python inhand_train.py
```

### Training with Tabular Q-Learning

```bash
cd src_tabular
python inhand_train.py
```

### Testing a Trained Agent

```bash
cd src_deepq
python inhand_test.py
```

### Visualizing the Environment

```bash
cd src_deepq
python simulation.py  # Opens MuJoCo viewer
```

## Configuration

### Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_EPISODE_STEPS` | 300 | Maximum steps per episode |
| `TARGET_ROTATION` | -90.0 | Target rotation angle in degrees |
| `TARGET_TOLERANCE` | 5.0 | Acceptable error margin in degrees |

### DQN Hyperparameters

- Network: 2 hidden layers (128 units each)
- Activation: ReLU
- Experience replay buffer: 1000 samples

### Q-Learning Hyperparameters

- Learning rate: 0.1
- Discount factor: 0.99
- Epsilon decay: 0.997
- Minimum epsilon: 0.05

## Roadmap

- [ ] Implement PPO and SAC algorithms for comparison
- [ ] Add curriculum learning for progressive difficulty
- [ ] Support for multiple object shapes (sphere, cube)
- [ ] Multi-task learning across manipulation primitives
- [ ] Sim-to-real transfer experiments

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Ryan William Powers
