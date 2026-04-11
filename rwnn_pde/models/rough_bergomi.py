from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from rwnn_pde.utils import timing

# ---------------------------------------------------------------------------
# rBergomi hybrid-scheme simulator
# Adapted from https://github.com/ryanmccrickerd/rough_bergomi
# ---------------------------------------------------------------------------


def _g(x: np.ndarray, a: float) -> np.ndarray:
    """TBSS kernel for the rBergomi variance process."""
    return x**a


def _b(k: int, a: float) -> float:
    """Optimal hybrid-scheme discretisation of the TBSS kernel."""
    return ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (1.0 / a)


def _cov(a: float, n: int) -> np.ndarray:
    """2×2 covariance matrix for the correlated Brownian increments (κ = 1)."""
    c = np.zeros((2, 2))
    c[0, 0] = 1.0 / n
    c[0, 1] = 1.0 / ((a + 1) * n ** (a + 1))
    c[1, 1] = 1.0 / ((2.0 * a + 1) * n ** (2.0 * a + 1))
    c[1, 0] = c[0, 1]
    return c


class rBergomi:
    """Hybrid-scheme path simulator for the rough Bergomi model.

    Implements the discretisation of McCrickerd & Pakkanen (2018).

    Parameters
    ----------
    n:
        Time steps per year (granularity).
    N:
        Number of paths.
    T:
        Maturity (years).
    r:
        Risk-free rate.
    a:
        Roughness parameter; related to Hurst index H by ``a = H − 0.5``.
    """

    def __init__(
        self,
        n: int = 100,
        N: int = 1000,
        T: float = 1.0,
        r: float = 0.01,
        a: float = -0.4,
    ):
        self.T = T
        self.n = n
        self.dt = 1.0 / n
        self.s = int(n * T)
        self.t = np.linspace(0, T, 1 + self.s)[np.newaxis, :]
        self.r = r
        self.a = a
        self.N = N

        self.e = np.array([0.0, 0.0])
        self.c = _cov(self.a, self.n)

    def dW1(self) -> np.ndarray:
        """Correlated increments for the variance process, shape (N, s, 2)."""
        return np.random.multivariate_normal(self.e, self.c, (self.N, self.s))

    def Y(self, dW: np.ndarray) -> np.ndarray:
        """Volterra process from correlated 2-d Brownian increments."""
        Y1 = np.zeros((self.N, 1 + self.s))
        for i in np.arange(1, 1 + self.s):
            Y1[:, i] = dW[:, i - 1, 1]

        G = np.zeros(1 + self.s)
        for k in np.arange(2, 1 + self.s):
            G[k] = _g(_b(k, self.a) / self.n, self.a)

        X = dW[:, :, 0]
        GX = np.zeros((self.N, X.shape[1] + len(G) - 1))
        for i in range(self.N):
            GX[i, :] = np.convolve(G, X[i, :])
        Y2 = GX[:, : 1 + self.s]

        return np.sqrt(2 * self.a + 1) * (Y1 + Y2)

    def dW2(self) -> np.ndarray:
        """Independent standard Brownian increments, shape (N, s)."""
        return np.random.randn(self.N, self.s) * np.sqrt(self.dt)

    def dB(self, dW1: np.ndarray, dW2: np.ndarray, rho: float = 0.0) -> np.ndarray:
        """Correlated price-process increments dB = ρ dW1 + √(1−ρ²) dW2."""
        self.rho = rho
        return rho * dW1[:, :, 0] + np.sqrt(1 - rho**2) * dW2

    def V(self, Y: np.ndarray, xi: float = 1.0, eta: float = 1.0) -> np.ndarray:
        """Rough Bergomi variance process."""
        self.xi = xi
        self.eta = eta
        return xi * np.exp(eta * Y - 0.5 * eta**2 * self.t ** (2 * self.a + 1))

    def S(self, V: np.ndarray, dB: np.ndarray, S0: float = 1.0) -> np.ndarray:
        """rBergomi price process."""
        self.S0 = S0
        increments = np.sqrt(V[:, :-1]) * dB + (self.r - 0.5 * V[:, :-1]) * self.dt
        integral = np.cumsum(increments, axis=1)
        price = np.zeros_like(V)
        price[:, 0] = S0
        price[:, 1:] = S0 * np.exp(integral)
        return price


# ---------------------------------------------------------------------------
# roughBergomi model (wraps rBergomi simulator)
# ---------------------------------------------------------------------------


@dataclass
class rBergomiParameters:
    """Parameters for the rough Bergomi model.

    Model and reservoir hyperparameters are collected in a single dataclass
    for convenience.

    Parameters
    ----------
    S0:
        Initial asset price.
    T:
        Maturity (years).
    H:
        Hurst index (roughness); ``H < 0.5`` gives rough volatility.
    r:
        Risk-free rate.
    rho:
        Correlation between price and variance Brownian motions.
    xi:
        Initial forward variance curve (flat, scalar).
    eta:
        Volatility-of-volatility.
    K:
        Strike price.
    opt_type:
        ``'c'`` for call, ``'p'`` for put.
    n_hidden_nodes:
        Number of reservoir neurons.
    connectivity:
        Fraction of non-zero reservoir weights.
    input_scaling:
        Stored for API consistency (see :class:`~rwnn_pde.reservoir.Reservoir`).
    weight_compact_radius:
        Reservoir weights drawn from ``[−r, r]``.
    """

    S0: float
    T: float
    H: float
    r: float
    rho: float
    xi: float
    eta: float
    K: float
    opt_type: Literal["c", "p"]

    n_hidden_nodes: int = 100
    connectivity: float = 0.5
    input_scaling: float = 0.25
    weight_compact_radius: float = 0.5


class roughBergomi:
    """Rough Bergomi model with Monte Carlo path simulation.

    Wraps :class:`rBergomi` and stores the simulated paths for downstream
    use by :class:`~rwnn_pde.trainer.rBergomiTrainer`.

    Parameters
    ----------
    parameters:
        Model and reservoir hyperparameters.
    N_samples:
        Number of Monte Carlo paths.
    n_timesteps:
        Number of time increments; the time grid has ``n_timesteps + 1`` points.
    """

    def __init__(
        self,
        parameters: rBergomiParameters,
        N_samples: int,
        n_timesteps: int,
    ):
        for key, value in parameters.__dict__.items():
            setattr(self, key, value)

        self.N_samples = N_samples
        self.n_timesteps = n_timesteps
        self.delta = self.T / n_timesteps

        # Populated by simulate_paths
        self.time_grid: Optional[np.ndarray] = None
        self.S: Optional[np.ndarray] = None
        self.V: Optional[np.ndarray] = None
        self.dW1: Optional[np.ndarray] = None
        self.dW2: Optional[np.ndarray] = None
        self.dB: Optional[np.ndarray] = None

        self.simulate_paths(N_samples=N_samples, n_timesteps=n_timesteps)

    def call_price(self, N_samples: int = 10**5, n_timesteps: int = 100) -> float:
        """Monte Carlo call price (benchmark)."""
        S, _, _, _ = self.simulate_paths(N_samples=N_samples, n_timesteps=n_timesteps)
        return float(np.maximum(S[:, -1, 0] - self.K, 0).mean())

    def put_price(self, N_samples: int = 10**5, n_timesteps: int = 100) -> float:
        """Monte Carlo put price (benchmark)."""
        S, _, _, _ = self.simulate_paths(N_samples=N_samples, n_timesteps=n_timesteps)
        return float(np.maximum(self.K - S[:, -1, 0], 0).mean())

    @timing
    def simulate_paths(
        self,
        N_samples: int,
        n_timesteps: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate rough Bergomi paths via the hybrid scheme.

        Parameters
        ----------
        N_samples:
            Number of paths.
        n_timesteps:
            Number of time increments; grid has ``n_timesteps + 1`` points.

        Returns
        -------
        S, V, dW1, dW2:
            Price, variance, and Brownian increments, each of shape
            ``(N_samples, n_timesteps + 1, 1)``.
        """
        rb = rBergomi(n=n_timesteps, N=N_samples, T=self.T, r=self.r, a=self.H - 0.5)

        dW1 = rb.dW1()
        dW2 = rb.dW2()
        dB = rb.dB(dW1, dW2, rho=self.rho)
        Y = rb.Y(dW1)
        V = rb.V(Y, xi=self.xi, eta=self.eta)
        S = rb.S(V, dB, S0=self.S0)

        # Keep only the Brownian part of dW1 ([:,:,0]); the second column
        # is used internally for the Volterra convolution.
        dW1 = dW1[:, :, 0]

        # Add trailing dimension for consistency with a multi-asset extension
        S = S[:, :, None]
        V = V[:, :, None]
        dW1 = dW1[:, :, None]
        dW2 = dW2[:, :, None]
        dB = dB[:, :, None]

        if self.S is None:
            self.time_grid = rb.t
            self.S = S
            self.V = V
            self.dW1 = dW1
            self.dW2 = dW2
            self.dB = dB

        return S, V, dW1, dW2
