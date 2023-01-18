import numpy as np
from scipy import sparse

def ReLu(X):
    return np.maximum(0, X)


def grad_ReLu(X):
    """returns a vector of derivatives"""
    return np.maximum(np.sign(X.reshape(X.shape[0], -1)), 0)


def grad_tanh(X):
    """returns a vector of derivatives"""
    X = X.reshape(X.shape[0], -1)
    return np.ones_like(X) - (X ** 2)


class Reservoir(object):
    """
    Build a reservoir and evaluate internal states

    Parameters:
        n_internal_units = processing units in the reservoir
        connectivity = percentage of nonzero connection weights
        input_scaling = scaling of the input connection weights
    """

    def __init__(self, n_internal_units=1000, connectivity=0.25, input_scaling=0.2, weight_compact_radius=0.5,
                 activation_function=ReLu, activation_derivative=grad_ReLu, seed=0):
        # Initialize attributes
        self._n_internal_units = n_internal_units
        self._input_scaling = input_scaling
        self._connectivity = connectivity
        self._activation_function = activation_function
        self._activation_derivative = activation_derivative
        self.weight_compact_radius = weight_compact_radius

        # Input weights depend on input size: they are set when data is provided
        self._internal_weights = None
        self._internal_bias = None

        # Set seed
        # np.random.seed(seed)
        self.random_state = np.random.RandomState(seed=seed)  # to keep the seed local

    def _initialize_internal_weights(self, n_internal_units, n_data_dimension, connectivity):
        # Generate sparse, uniformly distributed weights. # TODO: try e.g. student-t distribution
        internal_weights = sparse.rand(n_internal_units,
                                       n_data_dimension,
                                       density=connectivity,
                                       random_state=self.random_state).toarray() * 2 * self.weight_compact_radius

        # Ensure that the nonzero values are uniformly distributed in [-weight_compact_radius, weight_compact_radius]
        internal_weights[np.where(internal_weights > 0)] -= self.weight_compact_radius

        return internal_weights

    def _initialize_internal_bias(self, n_internal_units):
        internal_bias = self.random_state.rand(n_internal_units, 1) * 2 * self.weight_compact_radius
        # Ensure that the values are uniformly distributed in [-weight_compact_radius, weight_compact_radius]
        internal_bias -= self.weight_compact_radius
        return internal_bias

    def _compute_state_matrix(self, X):
        # Calculate state
        state_before_activation = np.tensordot(self._internal_weights, X, axes=(1, 1)) \
                                  + np.tile(self._internal_bias, (1, X.shape[0]))

        # Apply nonlinearity
        state_vec = self._activation_function(state_before_activation)

        return state_vec

    def get_reservoir_out(self, input_array):
        N, d = input_array.shape
        if self._internal_weights is None:
            # Generate internal weights
            self._internal_weights = self._initialize_internal_weights(self._n_internal_units, d,
                                                                       self._connectivity)
            self._internal_bias = self._initialize_internal_bias(self._n_internal_units)

        # compute reservoir states
        states = self._compute_state_matrix(input_array)

        # compute reservoir grad
        gradient = self._activation_derivative(states)[:, None, :] * np.tile(self._internal_weights[:, :, None],
                                                                             (1, 1, N))

        return states.T, gradient.transpose(2, 0, 1)
