from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from scipy import sparse

# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified linear unit: max(0, x)."""
    return np.maximum(0, x)


def grad_relu(x: np.ndarray) -> np.ndarray:
    """Sub-gradient of ReLU (0 at x=0)."""
    return np.maximum(np.sign(x.reshape(x.shape[0], -1)), 0)


def grad_tanh(x: np.ndarray) -> np.ndarray:
    """Derivative of tanh: 1 − tanh²(x)."""
    x = x.reshape(x.shape[0], -1)
    return 1.0 - x**2


# ---------------------------------------------------------------------------
# Reservoir
# ---------------------------------------------------------------------------


class Reservoir:
    """Random-weight neural network reservoir.

    Internal weights are fixed at construction time; only the linear output
    layer is trained (by the :class:`~rwnn_pde.trainer.BaseTrainer`).

    Parameters
    ----------
    n_internal_units:
        Number of hidden neurons.
    connectivity:
        Fraction of non-zero weights (sparsity of the weight matrix).
    input_scaling:
        Stored for API compatibility; not currently used in weight
        initialisation (weights are scaled by *weight_compact_radius*).
    weight_compact_radius:
        Weights are drawn uniformly from
        ``[−weight_compact_radius, weight_compact_radius]``.
    activation_function:
        Non-linear activation applied after the linear pre-activation.
    activation_derivative:
        Element-wise derivative of *activation_function*.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_internal_units: int = 1000,
        connectivity: float = 0.25,
        input_scaling: float = 0.2,
        weight_compact_radius: float = 0.5,
        activation_function: Callable = relu,
        activation_derivative: Callable = grad_relu,
        seed: int = 0,
    ):
        self._n_internal_units = n_internal_units
        self._input_scaling = input_scaling
        self._connectivity = connectivity
        self._activation_function = activation_function
        self._activation_derivative = activation_derivative
        self.weight_compact_radius = weight_compact_radius

        self._internal_weights: Optional[np.ndarray] = None
        self._internal_bias: Optional[np.ndarray] = None

        self.random_state = np.random.RandomState(seed=seed)

    # ------------------------------------------------------------------
    # Weight initialisation (lazy — deferred until input shape is known)
    # ------------------------------------------------------------------

    def _initialize_internal_weights(
        self, n_internal_units: int, n_data_dimension: int, connectivity: float
    ) -> np.ndarray:
        weights = (
            sparse.rand(
                n_internal_units,
                n_data_dimension,
                density=connectivity,
                random_state=self.random_state,
            ).toarray()
            * 2
            * self.weight_compact_radius
        )
        weights[weights > 0] -= self.weight_compact_radius
        return weights

    def _initialize_internal_bias(self, n_internal_units: int) -> np.ndarray:
        bias = self.random_state.rand(n_internal_units, 1) * 2 * self.weight_compact_radius
        bias -= self.weight_compact_radius
        return bias

    def _compute_state_matrix(self, x: np.ndarray) -> np.ndarray:
        pre_activation = np.tensordot(self._internal_weights, x, axes=(1, 1)) + np.tile(
            self._internal_bias, (1, x.shape[0])
        )
        return self._activation_function(pre_activation)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_reservoir_out(self, input_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate reservoir states and their input Jacobian.

        Parameters
        ----------
        input_array:
            Batch of inputs, shape ``(N, d)``.

        Returns
        -------
        states:
            Reservoir activations, shape ``(N, n_internal_units)``.
        gradient:
            Jacobian of *states* with respect to *input_array*,
            shape ``(N, n_internal_units, d)``.
        """
        n, d = input_array.shape
        if self._internal_weights is None:
            self._internal_weights = self._initialize_internal_weights(
                self._n_internal_units, d, self._connectivity
            )
            self._internal_bias = self._initialize_internal_bias(self._n_internal_units)

        states = self._compute_state_matrix(input_array)
        gradient = self._activation_derivative(states)[:, None, :] * np.tile(
            self._internal_weights[:, :, None], (1, 1, n)
        )
        return states.T, gradient.transpose(2, 0, 1)
