# CS134 Project: Temporal Difference Learning Algorithms

Implementation and comparison of TD(λ) and Hutter-Legg algorithms for temporal difference learning.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Setup

```bash
git clone <repository-url>
cd cs134_project
uv sync
```

## Running Experiments

```bash
# 21-state Markov chain experiments
uv run ./experiments/run_21_stationary.py
uv run ./experiments/run_21_non_stationary.py

# 51-state Markov chain experiment
uv run ./experiments/run_51_stationary.py

# Windy gridworld experiments
uv run ./experiments/run_windy_gridworld.py
```

## Available Algorithms

- **TDLambda** - Traditional TD(λ) with learning rate
- **HLLambda** - Hutter-Legg TD(λ) without learning rate
- **Sarsa** - SARSA(λ) algorithm
- **HLS** - Hutter-Legg SARSA(λ) variant
- **Q** - Q(λ) algorithm
- **HLQ** - Hutter-Legg Q(λ) variant

## Environments

- **MarkovChain** - Customizable Markov chain environments
- **WindyGridworld** - Classic RL environment with wind effects

## Customization

Modify parameters directly in experiment files or use command line arguments where supported. Common parameters:

- `n_runs` - Number of independent runs
- `n_steps` - Steps per run
- `alpha` - Learning rate (TD methods)
- `lambda_` - TD(λ) parameter
- `gamma` - Discount factor
- `epsilon` - Exploration rate

## Results

Outputs saved to `results/` directory:
- PNG plots
- NumPy arrays with numerical data
- Console progress and metrics
