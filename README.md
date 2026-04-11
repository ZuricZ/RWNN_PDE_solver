# RWNN-PDE: Random Neural Networks for Rough Volatility

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Implementation of the method from:

> **Random neural networks for rough volatility**  
> Antoine Jacquier and Zan Zuric, 2023  
> [https://arxiv.org/abs/2305.01035](https://arxiv.org/abs/2305.01035)

## Overview

This repository provides a clean, pip-installable implementation of a reservoir-computing
algorithm for solving path-dependent PDEs arising in rough volatility models.

The method interprets the pricing PDE as a backward stochastic differential equation
(following Bayer, Qiu & Yao) and solves it via a regression-based backward induction
scheme. At each time step, a **random weight neural network (reservoir)** provides fixed,
randomly initialised basis functions; only the output layer is trained, reducing
the optimisation to a **least-squares regression**. Theoretical convergence guarantees
follow from the reservoir-computing framework of Gonon, Grigoryeva & Ortega.

Two model settings are implemented out of the box:

| Model | Class | Parameters |
|-------|-------|------------|
| Multi-dimensional Black-Scholes | `BlackScholes` | `BlackScholesParameters` |
| Rough Bergomi (rBergomi) | `roughBergomi` | `rBergomiParameters` |

## Installation

```bash
git clone <repo-url>
cd RWNN-PDE
pip install -e ".[dev]"
```

Requires Python ≥ 3.9.

## Quick Start

### Black-Scholes (basket option, d=5 assets)

```python
import numpy as np
from scipy.stats import random_correlation
from rwnn_pde import BlackScholes, BlackScholesParameters, BSTrainer

d = 5
sigma = np.linspace(0.05, 0.25, d)
eigs = d * np.logspace(1, -1, d) / np.logspace(1, -1, d).sum()
Corr = random_correlation.rvs(eigs=eigs, random_state=0)
Cov  = np.diag(sigma) @ Corr @ np.diag(sigma)

params = BlackScholesParameters(
    S0=np.ones(d), T=1.0, r=0.05,
    K=np.array([1.0]), Cov=Cov,
    opt_style="basket", opt_type="c",
    n_hidden_nodes=1000,
)

model   = BlackScholes(params, N_samples=50_000, n_timesteps=21)
trainer = BSTrainer(model)
prices  = trainer.fit(alpha=0.5)

print(f"PDE price at t=0: {prices[:, 0, :].mean(0)}")
```

### Rough Bergomi (European call)

```python
from rwnn_pde import roughBergomi, rBergomiParameters, rBergomiTrainer

params = rBergomiParameters(
    S0=1.0, T=1.0, H=0.1, r=0.01,
    rho=-0.7, xi=0.235**2, eta=1.9,
    K=1.0, opt_type="c",
    n_hidden_nodes=100,
)

model   = roughBergomi(params, N_samples=50_000, n_timesteps=21)
trainer = rBergomiTrainer(model)
prices  = trainer.fit(alpha=0.1)

print(f"PDE price at t=0: {prices[:, 0, :].mean(0)}")
```

## Repository Structure

```
rwnn_pde/
├── __init__.py           # public API
├── reservoir.py          # Reservoir class and activation functions
├── trainer.py            # BaseTrainer, BSTrainer, rBergomiTrainer
├── payoffs.py            # option payoff functions
├── utils.py              # timing decorator and utilities
└── models/
    ├── black_scholes.py  # BlackScholes model
    └── rough_bergomi.py  # roughBergomi model + rBergomi simulator
notebooks/
├── 01_black_scholes_pde.ipynb
└── 02_rough_bergomi_pde.ipynb
pyproject.toml
README.md
```

## Extending to a New Model

1. Add a `Parameters` dataclass and a model class (path simulator) in `rwnn_pde/models/`.
2. Subclass `BaseTrainer` in `rwnn_pde/trainer.py`, implementing the six abstract methods
   (`_n_time_points`, `_output_dim`, `_terminal_payoff`, `_make_reservoirs`,
   `_assemble_LS`, `_evaluate_solution`, `_clamp`).
3. Export the new classes from `rwnn_pde/__init__.py`.

## Citation

```bibtex
@misc{jacquier2023random,
  title     = {Random neural networks for rough volatility},
  author    = {Jacquier, Antoine and Zuric, Zan},
  year      = {2023},
  eprint    = {2305.01035},
  archivePrefix = {arXiv},
  primaryClass  = {q-fin.MF},
  url       = {https://arxiv.org/abs/2305.01035}
}
```

## Acknowledgements

The `rBergomi` hybrid-scheme simulator is adapted from
[ryanmccrickerd/rough_bergomi](https://github.com/ryanmccrickerd/rough_bergomi).
