from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
from scipy.linalg import solve as ls_solve

from rwnn_pde.payoffs import payoff, vanilla_payoff
from rwnn_pde.reservoir import Reservoir
from rwnn_pde.utils import timing


class BaseTrainer(ABC):
    """Abstract base trainer for the RWNN-PDE backward-induction algorithm.

    Implements the regression loop (Algorithm 1 in the paper). Subclasses
    supply the model-specific components via the abstract methods below.

    At each time step *k* (running backward from *T* to *0*), the trainer:

    1. Builds a random reservoir (fixed weights).
    2. Constructs the least-squares system ``(A, B)`` from the reservoir
       output and the Brownian increments.
    3. Solves ``A β = B`` (with optional Tikhonov regularisation).
    4. Evaluates the approximate solution ``u(t_k, ·)`` at all path values.

    Parameters
    ----------
    model:
        A model object (e.g. :class:`~rwnn_pde.models.BlackScholes`) whose
        attributes are copied into the trainer.
    """

    def __init__(self, model):
        for key, value in model.__dict__.items():
            setattr(self, key, value)
        self._einsum_optimize: str = "greedy"

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement all six methods
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def _n_time_points(self) -> int:
        """Total number of time-grid points, including *t = 0*."""

    @property
    @abstractmethod
    def _output_dim(self) -> int:
        """Dimension of the option price (number of strikes)."""

    @abstractmethod
    def _terminal_payoff(self) -> np.ndarray:
        """Terminal condition ``Y(T) = g(S_T)``, shape ``(N, output_dim)``."""

    @abstractmethod
    def _make_reservoirs(self, seed: int, k: int) -> Union[Reservoir, Tuple[Reservoir, Reservoir]]:
        """Construct the reservoir(s) for time step *k*."""

    @abstractmethod
    def _assemble_LS(
        self,
        reservoirs: Union[Reservoir, Tuple[Reservoir, Reservoir]],
        target: np.ndarray,
        i: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build the least-squares matrices ``(A, B)`` for time step *i*."""

    @abstractmethod
    def _evaluate_solution(
        self,
        reservoirs: Union[Reservoir, Tuple[Reservoir, Reservoir]],
        beta: np.ndarray,
        i: int,
    ) -> np.ndarray:
        """Evaluate the fitted solution ``u(t_i, S_{t_i})``."""

    @abstractmethod
    def _clamp(self, values: np.ndarray) -> np.ndarray:
        """Enforce non-negativity of option prices."""

    # ------------------------------------------------------------------
    # Concrete methods
    # ------------------------------------------------------------------

    def _fit_step_exact(
        self,
        reservoirs: Union[Reservoir, Tuple[Reservoir, Reservoir]],
        target: np.ndarray,
        i: int,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Solve the least-squares system, with optional Tikhonov regularisation.

        Parameters
        ----------
        reservoirs:
            Reservoir(s) for this time step.
        target:
            Right-hand side ``Y(t_{i+1})``, shape ``(N, output_dim)``.
        i:
            Time-step index.
        alpha:
            Tikhonov regularisation coefficient added to the diagonal of *A*.

        Returns
        -------
        beta:
            Coefficient matrix, shape ``(output_dim, n_reservoir_features)``.
        """
        A, B = self._assemble_LS(reservoirs, target, i)
        if alpha is not None:
            A = A + alpha * np.eye(A.shape[0])
        return ls_solve(A, B).T

    @timing
    def fit(
        self,
        alpha: Optional[float] = None,
        verbose: int = 0,
        seed: int = 0,
    ) -> np.ndarray:
        """Price the option by backward induction.

        Iterates from the terminal condition ``u(T, ·) = g(S_T)`` backward
        to ``t = 0``, fitting a reservoir approximation at each time step.

        Parameters
        ----------
        alpha:
            Tikhonov regularisation parameter. ``None`` means no regularisation.
        verbose:
            Set to ``1`` to print progress at each time step.
        seed:
            Base seed for reservoir initialisation (each time step uses
            a distinct seed derived from this base).

        Returns
        -------
        Y:
            Approximate option price process,
            shape ``(N_samples, n_time_points, output_dim)``.
            ``Y[:, 0, :]`` gives the *t = 0* prices across all paths.
        """
        n = self._n_time_points
        Y = np.zeros((self.N_samples, n, self._output_dim))
        Y[:, -1, :] = self._terminal_payoff()

        for k in range(n - 2, -1, -1):
            if verbose > 0:
                print(f"Regressing time step: {k + 1}")
            res = self._make_reservoirs(seed, k)
            beta = self._fit_step_exact(res, Y[:, k + 1, :], k, alpha)
            Y[:, k, :] = self._clamp(self._evaluate_solution(res, beta, k))

        return Y


# ---------------------------------------------------------------------------
# Black-Scholes trainer
# ---------------------------------------------------------------------------


class BSTrainer(BaseTrainer):
    """RWNN-PDE trainer for the multi-dimensional Black-Scholes model.

    Solves the backward Kolmogorov PDE for correlated GBM with risk-free
    rate *r* and covariance *Cov*.

    The regression feature vector at time step *i* is:

    .. code-block:: text

        X = φ(S_i) · (1 + r·Δt) + ∇φ(S_i) · Σ(S_i) · ΔW_i

    where ``φ`` denotes the reservoir, ``Σ(S) = diag(S) · L`` and
    ``L = chol(Cov)``.
    """

    @property
    def _n_time_points(self) -> int:
        return self.n_timesteps

    @property
    def _output_dim(self) -> int:
        return self.K.shape[0]

    def _terminal_payoff(self) -> np.ndarray:
        return payoff(self.S[:, -1, :], self.K, opt_style=self.opt_style, opt_type=self.opt_type)

    def _make_reservoirs(self, seed: int, k: int) -> Reservoir:
        return Reservoir(
            n_internal_units=self.n_hidden_nodes,
            connectivity=self.connectivity,
            input_scaling=self.input_scaling,
            weight_compact_radius=self.weight_compact_radius,
            seed=seed + k,
        )

    def _assemble_LS(
        self, res: Reservoir, target: np.ndarray, i: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        dt = self.time_grid[i + 1] - self.time_grid[i]
        S_i = self.S[:, i, :]
        res_out, res_grad = res.get_reservoir_out(S_i)

        # Σ(S_i) · ΔW_i  where Σ(S) = diag(S) @ chol(Cov)
        sigma = self.diffusion * S_i[:, :, None]  # (N, d, d)
        sigma_bm = np.einsum("ijk,ik->ij", sigma, self.dW[:, i, :], optimize=self._einsum_optimize)

        # Feature vector: φ(S_i)(1 + r·Δt) + ∇φ(S_i) · Σ(S_i) ΔW_i
        X = (
            res_out
            - (-self.r * res_out) * dt
            + np.einsum("ijk,ik->ij", res_grad, sigma_bm, optimize=self._einsum_optimize)
        )
        A = np.einsum("ki,kj->ij", X, X, optimize=self._einsum_optimize)
        B = np.einsum("ki,kj->ij", X, target, optimize=self._einsum_optimize)
        return A, B

    def _evaluate_solution(self, res: Reservoir, beta: np.ndarray, i: int) -> np.ndarray:
        res_out, _ = res.get_reservoir_out(self.S[:, i, :])
        return np.tensordot(beta, res_out, axes=(1, 1)).T

    def _clamp(self, values: np.ndarray) -> np.ndarray:
        return np.abs(values)


# ---------------------------------------------------------------------------
# rough Bergomi trainer
# ---------------------------------------------------------------------------


class rBergomiTrainer(BaseTrainer):
    """RWNN-PDE trainer for the rough Bergomi model.

    Solves the associated backward Kolmogorov PDE using two coupled
    reservoirs (ξ and θ) per time step, as derived in Section 3 of the paper.

    The block least-squares system couples:

    * **X1** (ξ-reservoir): ``φ^ξ(S_i) · (ΔW¹_i − b·Δt)``
    * **X2** (θ-reservoir): ``(1 − a·Δt) φ^θ(S_i) + ∇φ^θ(S_i) · √V_i · ΔB_i``

    where ``a = −r``, ``b = 0``, ``c = 0`` for the risk-neutral measure.
    """

    def __init__(self, model):
        super().__init__(model)
        # PDE coefficients (risk-neutral measure)
        self._a: float = -self.r
        self._b: float = 0.0
        self._c: float = 0.0

    @property
    def _n_time_points(self) -> int:
        return self.n_timesteps + 1

    @property
    def _output_dim(self) -> int:
        return 1

    def _terminal_payoff(self) -> np.ndarray:
        return vanilla_payoff(self.S[:, -1, :], self.K, opt_type=self.opt_type)

    def _make_reservoirs(self, seed: int, k: int) -> Tuple[Reservoir, Reservoir]:
        common = dict(
            n_internal_units=self.n_hidden_nodes,
            connectivity=self.connectivity,
            input_scaling=self.input_scaling,
            weight_compact_radius=self.weight_compact_radius,
        )
        return (
            Reservoir(**common, seed=seed + k),
            Reservoir(**common, seed=seed + k + self.n_timesteps),
        )

    def _X1(self, res_out: np.ndarray, i: int) -> np.ndarray:
        """Feature vector for the ξ-reservoir."""
        return np.einsum(
            "ij,ik->ij",
            res_out,
            self.dW1[:, i, :] - self._b * self.delta,
            optimize=self._einsum_optimize,
        )

    def _X2(self, res_out: np.ndarray, res_grad: np.ndarray, i: int) -> np.ndarray:
        """Feature vector for the θ-reservoir."""
        scaled_res_out = (1 - self._a * self.delta) * res_out
        scaled_dB = (
            self.dB[:, i, :]
            - (self._b * self.rho + self._c * np.sqrt(1 - self.rho**2)) * self.delta
        )
        sigma_bm = np.einsum(
            "ijk,ik->ij",
            np.sqrt(self.V[:, i, :])[:, :, None],
            scaled_dB,
            optimize=self._einsum_optimize,
        )
        return scaled_res_out + np.einsum(
            "ijk,ik->ij", res_grad, sigma_bm, optimize=self._einsum_optimize
        )

    def _assemble_LS(
        self,
        res_tuple: Tuple[Reservoir, Reservoir],
        target: np.ndarray,
        i: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        res_xi, res_theta = res_tuple
        res_out_xi, _ = res_xi.get_reservoir_out(self.S[:, i, :])
        res_out_theta, res_grad_theta = res_theta.get_reservoir_out(self.S[:, i, :])

        X1 = self._X1(res_out_xi, i)
        X2 = self._X2(res_out_theta, res_grad_theta, i)

        A11 = np.einsum("ki,kj->ij", X1, X1, optimize=self._einsum_optimize)
        A12 = np.einsum("ki,kj->ij", X1, X2, optimize=self._einsum_optimize)
        A22 = np.einsum("ki,kj->ij", X2, X2, optimize=self._einsum_optimize)
        B1 = np.einsum("ki,kj->ij", X1, target, optimize=self._einsum_optimize)
        B2 = np.einsum("ki,kj->ij", X2, target, optimize=self._einsum_optimize)

        A = np.block([[A11, A12], [A12.T, A22]])
        B = np.concatenate([B1, B2], axis=0)
        return A, B

    def _evaluate_solution(
        self,
        res_tuple: Tuple[Reservoir, Reservoir],
        beta: np.ndarray,
        i: int,
    ) -> np.ndarray:
        _, res_theta = res_tuple
        res_out, _ = res_theta.get_reservoir_out(self.S[:, i, :])
        # beta = [β_ξ | β_θ]; only the θ-block is used for the solution value
        return np.tensordot(beta[:, res_theta._n_internal_units :], res_out, axes=(1, 1)).T

    def _clamp(self, values: np.ndarray) -> np.ndarray:
        return np.maximum(values, 0.0)
