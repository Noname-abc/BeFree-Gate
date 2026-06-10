# BeFree-Gate: Robotarium Key-Door Experiments

This repository contains Robotarium simulations for a single-robot key-door navigation task with uncertain object identities. The robot must first identify and reach the true key, then identify and reach the true door. The experiments compare two policies:

- **With epistemic term**: the minimization objective includes expected information gain.
- **Without epistemic term**: the minimization objective does not include expected information gain.

The main simulation is implemented in `robotarium_simulations.py`, and the plotting/metrics script is implemented in `plots.py`.

## Repository structure

```text
.
├── robotarium_simulations.py
└──plots.py
```

## Files

### `robotarium_simulations.py`

Runs the Robotarium experiments. It samples randomized key and door candidate locations, runs paired experiments with and without the epistemic term, and saves per-episode data and aggregate metrics.

For each episode, both conditions use the same sampled environment configuration: candidate positions, true key, true door, initial robot pose, and episode seed.

The script saves:

```text
results/robotarium_epi_ep000_data.npz
results/robotarium_no_epi_ep000_data.npz
results/robotarium_epi_metrics.npz
results/robotarium_no_epi_metrics.npz
results/robotarium_epi_all_results.pkl
results/robotarium_no_epi_all_results.pkl
```

It also shows and saves videos of the first episode in both cases if `SHOW_FIGURE=True` and `save_video=True`.

### `plots.py`

Loads the saved `.npz` files from `results/`, prints aggregate metrics, and creates publication-style figures for the first episode of each condition.

It generates:

```text
figures/robotarium_epi_trajectory.pdf
figures/robotarium_epi_weights.pdf
figures/robotarium_epi_omega.pdf
figures/robotarium_no_epi_trajectory.pdf
figures/robotarium_no_epi_weights.pdf
figures/robotarium_no_epi_omega.pdf
```

## Requirements

The code requires Python 3.10.x+ and the [Robotarium Python Simulator](https://github.com/robotarium/robotarium_python_simulator/tree/master) package, which provides the `rps` module.

Python dependencies used by the scripts include:

```text
numpy
matplotlib
cvxpy
pickle
os
```

`pickle` and `os` are part of the Python standard library.

A typical installation is:

```bash
pip install numpy matplotlib cvxpy
```

## Running the simulations

From the repository root, run:

```bash
python robotarium_simulations.py
```

The script will:

1. Create the `results/` and `figures/` folders if they do not exist.
2. Sample randomized key-door task configurations.
3. Run the epistemic condition.
4. Run the non-epistemic condition on the same configurations.
5. Save per-episode data and aggregate metrics.
6. Print a summary to the terminal.

The number of episodes is controlled by:

```python
N_EPISODES = 50
```

## Plotting results

After running the simulations, create figures with:

```bash
python plots.py
```

The script prints metrics for the two conditions and saves PDF plots in `figures/`.

## Important configuration options

### Show Robotarium visualization

```python
SHOW_FIGURE = True
```

Set to `False` for faster batch simulations without visualization.

### Real-time simulation

```python
SIM_IN_REAL_TIME = False
```

For fast offline simulations, keep this as `False`. For hardware-style or real-time execution, set it to `True`.

### Epistemic vs non-epistemic policy

The epistemic term is controlled by the `use_epistemic` argument:

```python
use_epistemic=True
```

or:

```python
use_epistemic=False
```

## Saved data format

Each per-episode `.npz` file contains arrays such as:

```text
trajectory
weights
omega
theta
key_entropy
door_entropy
key_candidates
door_candidates
true_key
true_door
initial_condition
success
```

The aggregate metrics files contain:

```text
success
gate_steps
key_visible_time
door_visible_times
success_rate
mean_gate_steps
std_gate_steps
mean_key_visible
std_key_visible
mean_pickup
std_pickup
mean_door_visible
std_door_visible
```

## Generated files

The `results/` and `figures/` directories are generated outputs.

## Acknowledgment

This project uses the Robotarium Python simulator. If you use this repository for research, cite or acknowledge Robotarium according to the requirements of the Robotarium project and your institution.
