# BeFree-Gate: Robotarium Key-Door Experiments

This repository contains simulations using the Robotarium platform [1] for a single-robot key-door navigation task with uncertain object identities. The robot must first identify and reach the true key, then identify and reach the true door. The experiments compare two policies:

- **With epistemic term**: the minimization objective includes expected information gain.
- **Without epistemic term**: the minimization objective does not include expected information gain.

The main simulation is implemented in `robotarium_simulations.py`, the plotting/metrics script is implemented in `plots.py`, and the Robotarium hardware script is implemented in `Hardware/robotarium_hardware.py`.

## Repository structure

```text
.
├── robotarium_simulations.py
├── plots.py
└── Hardware/
    └── robotarium_hardware.py
````

## Files

### `robotarium_simulations.py`

Runs the Robotarium simulation experiments. It samples randomized key and door candidate locations, runs paired experiments with and without the epistemic term, and saves per-episode data and aggregate metrics.

For each episode, both conditions use the same sampled environment configuration: candidate positions, true key, true door, initial robot pose, and episode seed.

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

To show Robotarium visualization and save video of the first experiment set

```python
SHOW_FIGURE = True
```

Set to `False` for faster batch simulations without visualization.

The script saves:

```text
results/robotarium_epi_ep000_data.npz
results/robotarium_no_epi_ep000_data.npz
results/robotarium_epi_metrics.npz
results/robotarium_no_epi_metrics.npz
```

### `plots.py`

Loads the saved `.npz` files from `results/`, prints aggregate metrics, and creates figures for the first episode of each condition.

It generates:

```text
figures/robotarium_epi_trajectory.pdf
figures/robotarium_epi_weights.pdf
figures/robotarium_epi_omega.pdf
figures/robotarium_no_epi_trajectory.pdf
figures/robotarium_no_epi_weights.pdf
figures/robotarium_no_epi_omega.pdf
```

After running the simulations, run:

```bash
python plots.py
```

The script prints metrics for the two conditions and saves PDF plots in `figures/`.


### `Hardware/robotarium_hardware.py`

This is the script uploaded to the [Robotarium website](https://www.robotarium.gatech.edu) to obtain the two hardware videos corresponding to Fig. 1(a) and Fig. 1(b) of the paper.

To switch between the epistemic and non-epistemic cases, change:

```python
RUN_EPISTEMIC = True
```

or:

```python
RUN_EPISTEMIC = False
```

We report here the two hardware videos corresponding to:
- the experiment in Fig. 1(a) of the paper (using full BeFree-Gate with epistemic term)
  

https://github.com/user-attachments/assets/060504c1-7011-4b72-a24e-640bf5f360a3


- the experiment in Fig. 1(b) of the paper (without epistemic term in the objective)
  

https://github.com/user-attachments/assets/bb27df84-5b7c-473a-9cda-0117932a721e


## Requirements

The code requires Python 3.10.x+ and the [Robotarium Python Simulator](https://github.com/robotarium/robotarium_python_simulator/tree/master) package, which provides the `rps` module.

Python dependencies used by the scripts include:

```text
numpy
matplotlib
cvxpy
```

You can install them with:

```bash
pip install numpy matplotlib cvxpy
```


## References

[1] S. Wilson et al., “The Robotarium: Globally Impactful Opportunities, Challenges, and Lessons Learned in Remote-Access, Distributed Control of Multirobot Systems,” *IEEE Control Systems Magazine*, vol. 40, no. 1, pp. 26–44, Feb. 2020.
