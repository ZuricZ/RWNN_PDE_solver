from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.stats import norm

from rwnn_pde.payoffs import basket_payoff
from rwnn_pde.utils import timing


@dataclass
class BlackScholesParameters:
    """Parameters for the multi-dimensional Black-Scholes model.

    Model and reservoir hyperparameters are collected in a single dataclass
    for convenience; both are forwarded to the model and trainer via
    attribute copying.

    Parameters
    ----------
    T:
        Maturity (years).
    r:
        Risk-free rate.
    S0:
        Initial asset prices, shape ``(d,)``.
    K:
        Strike(s), shape ``(m,)``.
    Cov:
        Covariance matrix of log-returns, shape ``(d, d)``.
    opt_style:
        ``'vanilla'`` (single-asset payoff) or ``'basket'``.
    opt_type:
        ``'c'`` for call, ``'p'`` for put.
    n_hidden_nodes:
        Number of reservoir neurons.
    connectivity:
        Fraction of non-zero reservoir weights.
    input_scaling:
        Stored for API consistency (see :class:`~rwnn_pde.reservoir.Reservoir`).
    weight_compact_radius:
        Reservoir weights drawn from ``[−r, r]``; also used as the default
        Tikhonov regularisation parameter ``alpha`` in :meth:`BSTrainer.fit`.
    """

    T: float
    r: float
    S0: np.ndarray
    K: np.ndarray
    Cov: np.ndarray
    opt_style: Literal["vanilla", "basket"] = "vanilla"
    opt_type: Literal["c", "p"] = "c"

    n_hidden_nodes: int = 100
    connectivity: float = 0.5
    input_scaling: float = 0.1
    weight_compact_radius: float = 0.5


class BlackScholes:
    """Multi-dimensional Black-Scholes model.

    Simulates correlated geometric Brownian motion paths and provides
    analytical option-price benchmarks (single-asset only).

    Parameters
    ----------
    parameters:
        Model and reservoir hyperparameters.
    N_samples:
        Number of Monte Carlo paths.
    n_timesteps:
        Number of time-grid points, including ``t = 0``.
    """

    def __init__(
        self,
        parameters: BlackScholesParameters,
        N_samples: int,
        n_timesteps: int,
    ):
        for key, value in parameters.__dict__.items():
            setattr(self, key, value)

        self.N_samples = N_samples
        self.n_timesteps = n_timesteps
        self.sigma = np.sqrt(np.diagonal(self.Cov))

        # Populated by simulate_paths
        self.time_grid: Optional[np.ndarray] = None
        self.delta: Optional[np.ndarray] = None
        self.diffusion: Optional[np.ndarray] = None
        self.S: Optional[np.ndarray] = None
        self.dW: Optional[np.ndarray] = None

        self.simulate_paths(N_samples=N_samples, n_timesteps=n_timesteps)

    # ------------------------------------------------------------------
    # Analytical pricing  (single-asset, closed-form Black-Scholes)
    # ------------------------------------------------------------------

    def _d1(self, S: float, T: float) -> float:
        return (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * T) / (self.sigma * np.sqrt(T))

    def _d2(self, S: float, T: float) -> float:
        return self._d1(S, T) - self.sigma * np.sqrt(T)

    def call_price(self, S: float, T: float) -> float:
        """Analytical Black-Scholes call price (single asset)."""
        return S * norm.cdf(self._d1(S, T)) - self.K * np.exp(-self.r * T) * norm.cdf(
            self._d2(S, T)
        )

    def put_price(self, S: float, T: float) -> float:
        """Analytical Black-Scholes put price via put-call parity."""
        return self.K * np.exp(-self.r * T) - S + self.call_price(S, T)

    def basket_call_price(self, N_samples: int, n_timesteps: int) -> float:
        """Monte Carlo basket call price (benchmark)."""
        S = self.simulate_paths(N_samples, n_timesteps)
        return float(basket_payoff(S=S[:, -1, :], K=self.K).mean())

    # ------------------------------------------------------------------
    # Path simulation
    # ------------------------------------------------------------------

    @timing
    def simulate_paths(self, N_samples: int, n_timesteps: int) -> np.ndarray:
        """Simulate correlated GBM paths.

        Parameters
        ----------
        N_samples:
            Number of paths.
        n_timesteps:
            Number of time-grid points (including ``t = 0``).

        Returns
        -------
        S:
            Simulated asset prices, shape ``(N_samples, n_timesteps, d)``.
        """
        d = self.S0.shape[0]
        time_grid = np.linspace(0.0, self.T, n_timesteps)
        diffusion = np.linalg.cholesky(self.Cov)
        dt = time_grid[1:] - time_grid[:-1]

        dW = np.tile(np.sqrt(dt[None, :, None]), (N_samples, 1, d)) * np.random.randn(
            N_samples, n_timesteps - 1, d
        )
        drift = np.tile(((self.r - 0.5 * self.sigma[:, None] ** 2) * dt).T, (N_samples, 1, 1))

        S = np.zeros((N_samples, n_timesteps, d))
        S[:, 0] = self.S0
        for i in range(1, n_timesteps):
            S[:, i, :] = S[:, i - 1, :] * np.exp(
                drift[:, i - 1, :] + np.matmul(diffusion, dW[:, i - 1, :].T).T
            )

        if self.S is None:
            self.time_grid = time_grid
            self.delta = dt
            self.diffusion = diffusion
            self.S = S
            self.dW = dW
        return S
