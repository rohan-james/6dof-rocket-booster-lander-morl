# 6DOF Multi-Objective RL Rocket Booster Lander 

![Licence](https://img.shields.io/badge/license-GPLv3-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)

### A 6-DOF PyBullet environment for rocket booster landing, with PPO/SAC baselines and a CAPQL-based multi-objective RL agent that recovers a Pareto front of precision-vs-fuel trade-off policies.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Option A: Conda (recommended)](#option-a-conda-recommended)
  - [Option B: pip](#option-b-pip)
- [Quickstart](#quickstart)
  - [Running the environment](#running-the-environment)
  - [Training an agent](#training-an-agent)
- [Citation](#citation)
- [Licence](#licence)

---

## Repository Structure

```
6dof-rocket-booster-lander-morl/
├── .github/
│   └── workflows/
│       └── black.yml
├── algorithms/
│   ├── __init__.py
│   ├── capql.py                              # CAPQL multi-objective agent
│   ├── ppo.py                                # PPO baseline 
│   └── sac.py                                # SAC baseline
├── controllers/
│   ├── __init__.py
│   ├── gimbal_engine_configuration.py
│   └── quad_engine_configuration.py
├── environments/
│   ├── __init__.py
│   ├── morl_lander.py                        # Multi-objective RL lander env
│   ├── sorl_lander.py                        # Single-objective RL lander env
├── models/
│   ├── __init__.py
│   ├── rocket_booster_visual.mtl
│   ├── rocket_booster_visual.obj
│   ├── rocket_fins.obj
│   ├── rocket_main_body.obj
│   ├── rocket_model.urdf
│   └── rocket_nozzle.obj
├── .gitignore
├── CITATION.cff
├── LICENCE
├── README.md
├── requirements.txt
├── environment.yml
└── __init__.py
```

## Requirements

- Python **>= 3.11**
- [PyBullet](https://pybullet.org/) for 6-DOF physics simulation
- [Gymnasium](https://gymnasium.farama.org/) / [MO-Gymnasium](https://mo-gymnasium.farama.org/) environment APIs
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for PPO/SAC baselines
- [MORL-Baselines](https://github.com/LucasAlegre/morl-baselines) for multi-objective RL baselines
- [PyTorch](https://pytorch.org/)
- [Weights & Biases](https://wandb.ai/) for experiment tracking

Full pinned versions are listed in [`requirements.txt`](#option-b-pip) and [`environment.yml`](#option-a-conda-recommended).

## Installation

### Option A: Conda (recommended)

This project is best run inside a conda environment (I say best because I found compatibility issues between pip and PyBullet). If you don't already have conda installed:

1. **Download and install Miniconda:**

   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
   bash miniconda.sh -b -p $HOME/miniconda3
   source $HOME/miniconda3/bin/activate
   conda init
   ```

   (macOS users can substitute the `MacOSX` installer; Windows users should use the Miniconda installer `.exe` and run the following steps from the Anaconda Prompt.)

2. **Clone the repository:**

   ```bash
   git clone https://github.com/rohan-james/6dof-rocket-booster-lander-morl.git
   cd 6dof-rocket-booster-lander-morl
   ```

3. **Create the environment from `environment.yml`:**

   ```bash
   conda env create -f environment.yml
   ```

4. **Activate the environment:**

   ```bash
   conda activate rocket-morl
   ```

5. **Verify the install:**

   ```bash
   python -c "import pybullet, gymnasium, mo_gymnasium, stable_baselines3, morl_baselines, torch, wandb; print('All dependencies imported successfully')"
   ```

### Option B: pip

If you'd rather not use conda, a standard virtual environment also works (Python >= 3.11 required):

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`:

```
gymnasium>=0.29.0
mo-gymnasium>=1.3.1
stable-baselines3>=2.1.0
morl-baselines
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
tensorboard>=2.13.0
wandb
pybullet
black>=26.5.1
```

## Quickstart

### Running the environment

Once you have your environment running and packages installed, you can choose between a classical proportional-derivative (PD) controller or an RL agent. Controllers only operate the SORL lander.

#### Classical Control

Navigate to `controllers/*.py` and run the configuration of your choice:

```bash
   python3 controllers/gimbal_engine_configuration.py
```

Alternatively, run it as a package:

```bash
   python3 -m controllers.gimbal_engine_configuration
```

Decide whether you wish to merely trace the terminal logs or visualise the booster landing/crashing by setting the flag below:

```python
   USE_GUI = True
```

#### Training an agent

The environments are imported into each file in `algorithms/`. Much like the classical control files, navigate to `algorithms/*.py` and run the algorithm of choice. PPPO and SAC are coded to run only the single-objective environment, while CAPQL runs the multi-objective counterpart.
Below is an example:

```bash
   python3 algorithms/ppo.py
```

Alternatively, run it as a package:

```bash
   python3 -m algorithms.ppo
```

You MUST set the curriculum level prior to a run. Curriculum levels range from 0 to 3; the global variables are self-explanatory.

## Citation

If you use this software or the accompanying dissertation, please cite it. Full machine-readable citation metadata is available in [`CITATION.cff`](./CITATION.cff).

```bibtex
@software{james_morl_6dof_lander_2026,
  author  = {James, Rohan},
  title   = {{Rocket-MO-Gym: A 6-DOF Multi-Objective Reinforcement Learning Environment for Autonomous Rocket Landing}},
  version = {1.0.0},
  year    = {2026},
  url     = {https://github.com/rohan-james/6dof-rocket-booster-lander-morl},
  licence = {GPL-3.0}
}

@mastersthesis{james_morl_astrodynamics_2026,
  author      = {James, Rohan},
  title       = {{On the Reconciliation of Conflicting Astrodynamic Landing Objectives via Learned Policies}},
  school      = {University of Dublin, Trinity College},
  year        = {2026}
}
```

## Licence

This project is licensed under the **GNU General Public Licence v3.0 (GPLv3)**. See [`LICENCE`](./LICENCE) for the full text.

<!-- ## Contact -->

<!-- TODO: add your email here -->

<!-- ## Acknowledgements -->

<!-- TODO -->
