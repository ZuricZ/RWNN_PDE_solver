"""
rwnn_pde
========

Random weight neural networks for solving high-dimensional PDEs in rough volatility.

Reference
---------
Jacquier, A. & Zuric, Z. (2023). Random neural networks for rough volatility.
https://arxiv.org/abs/2305.01035
"""

from rwnn_pde.models.black_scholes import BlackScholes, BlackScholesParameters
from rwnn_pde.models.rough_bergomi import rBergomiParameters, roughBergomi
from rwnn_pde.reservoir import Reservoir
from rwnn_pde.trainer import BSTrainer, rBergomiTrainer

__all__ = [
    "BlackScholes",
    "BlackScholesParameters",
    "roughBergomi",
    "rBergomiParameters",
    "Reservoir",
    "BSTrainer",
    "rBergomiTrainer",
]
