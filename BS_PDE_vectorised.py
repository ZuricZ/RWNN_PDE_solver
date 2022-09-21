import matplotlib.pyplot as plt

plt.style.use('dark_background')
from mpl_toolkits.mplot3d import Axes3D

# %matplotlib notebook

import numpy as np
from scipy.stats import norm
import scipy.io
from scipy.linalg import solve as LS_solve
from scipy import sparse
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import accuracy_score, f1_score

from utils import temp_seed, timing


def ReLu(X):
    return np.maximum(0, X)


def grad_ReLu(X):
    """returns a vector of derivatives"""
    return np.maximum(np.sign(X.reshape(X.shape[0], -1)), 0)


def grad_tanh(X):
    """returns a vector of derivatives"""
    X = X.reshape(X.shape[0], -1)
    return np.ones_like(X) - (X ** 2)


class Parameters():
    def __init__(self):
        self.S0 = np.array([1.]*50)  # np.random.uniform(0.5, 2, 5)
        self.T = 1
        self.r = 0.05
        self.K = 1
        self.sigma = np.linspace(0.05, 0.4, 50)
        self.Cov = np.diag(self.sigma ** 2)
        self.opt_type = 'c'


class BlackScholes(Parameters):

    def __init__(self):
        super().__init__()

        # initialised when simulate_paths() is called
        self.time_grid = None
        self.diffusion = None
        self.dW = None
        self.price_path = None
        self.N_samples = None
        self.n_timesteps = None
        self.d_assets = None

    def _get_grid(self, S_array, T_array):
        self.surf_plot_grid = np.meshgrid(S_array, T_array)

    def _d1(self, S, T):
        return (np.log(S / self.K) + (self.r + .5 * self.sigma ** 2) * T) / (self.sigma * np.sqrt(T))

    def _d2(self, S, T):
        return self._d1(S, T) - self.sigma * np.sqrt(T)

    def call_price(self, S, T):
        return S * norm.cdf(self._d1(S, T)) - self.K * np.exp(-self.r * T) * norm.cdf(self._d2(S, T))

    def put_price(self, S, T):
        return self.K * np.exp(-self.r * T) - S + self.call_price(S, T)

    @timing
    def simulate_paths(self, N_samples, n_timesteps):
        """
        :param N_samples:   number of samples
        :param n_timesteps: number of increments
        """
        # initialise for further use
        self.d_assets = self.S0.shape[0]
        self.N_samples = N_samples
        self.n_timesteps = n_timesteps

        self.time_grid = np.linspace(0., self.T, n_timesteps)
        self.diffusion = np.linalg.cholesky(self.Cov)
        dt = self.time_grid[1:] - self.time_grid[:-1]
        self.dW = np.tile(np.sqrt(dt[None, :, None]), (N_samples, 1, self.d_assets)) * np.random.randn(N_samples,
                                                                                                       n_timesteps - 1,
                                                                                                       self.d_assets)
        drift = np.tile(((self.r - 0.5 * self.sigma[:, None] ** 2) * dt).T, (N_samples, 1, 1))

        S = np.zeros((N_samples, n_timesteps, self.d_assets))
        S[:, 0] = self.S0
        for i in range(1, n_timesteps):
            S[:, i, :] = S[:, i - 1, :] * np.exp(
                drift[:, i - 1, :] + np.matmul(self.diffusion, self.dW[:, i - 1, :].T).T)

        # initialise for further use
        self.price_path = S
        return S


class Reservoir(object):
    """
    Build a reservoir and evaluate internal states

    Parameters:
        n_internal_units = processing units in the reservoir
        connectivity = percentage of nonzero connection weights
        input_scaling = scaling of the input connection weights
    """

    def __init__(self, n_internal_units=1000, connectivity=0.3, input_scaling=0.2,
                 activation_function=ReLu, activation_derivative=grad_ReLu, seed=0):
        # Initialize attributes
        self._n_internal_units = n_internal_units
        self._input_scaling = input_scaling
        self._connectivity = connectivity
        self._activation_function = activation_function
        self._activation_derivative = activation_derivative

        # Input weights depend on input size: they are set when data is provided
        self._internal_weights = None
        self._internal_bias = None

        # Set seed
        np.random.seed(seed)

    def _initialize_internal_weights(self, n_internal_units, n_data_dimension, connectivity):
        # Generate sparse, uniformly distributed weights.
        internal_weights = sparse.rand(n_internal_units,
                                       n_data_dimension,
                                       density=connectivity).toarray()

        # Ensure that the nonzero values are uniformly distributed in [-0.5, 0.5]
        internal_weights[np.where(internal_weights > 0)] -= 0.5

        return internal_weights

    def _initialize_internal_bias(self, n_internal_units):
        internal_bias = np.random.rand(n_internal_units, 1)
        # Ensure that the values are uniformly distributed in [-0.5, 0.5]
        internal_bias -= 0.5
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


class Trainer(Parameters):

    def __init__(self, model_class):
        super().__init__()
        self.model = model_class

    def a(self, res_out):
        # In Black Scholes model f(u)=-r*u, therefore f_tilde(x)=-r*reservoir_out(x)
        return - self.r * res_out

    def Sigma_func(self, S):
        # broadcast across columns (same as diag(S).dot(diffusion)
        return self.model.diffusion * S[:, :, None]

    def X(self, res, S, dt, dW):
        # res_out(x) - f_tilde(x) + res_grad(x)*Sigma(x)*dW
        res_out, res_grad = res.get_reservoir_out(S)
        SigmaBM_prod = np.einsum('ijk,ik->ij', self.Sigma_func(S), dW)
        return res_out - self.a(res_out) * dt + np.einsum('ijk,ik->ij', res_grad, SigmaBM_prod)

    def get_LS_problem(self, res, target, i):
        single_regr = self.X(res, S=self.model.price_path[:, i, :],
                             dt=self.model.time_grid[i + 1] - self.model.time_grid[i],
                             dW=self.model.dW[:, i, :])

        # perform the summation of the outer-products https://stackoverflow.com/q/35549082/5285408
        A = np.einsum('ki,kj->ij', single_regr, single_regr)

        B = np.einsum('ki,kj->ij', single_regr, target)
        return A, B

    def get_solution(self, res, beta, i):
        # evaluates the solution of the regression
        res_out, _ = res.get_reservoir_out(self.model.price_path[:, i, :])
        target = np.tensordot(beta, res_out, axes=(1, 1)).T
        return target

    def fit_step_exact(self, res, target, i, alpha=None):
        A, B = self.get_LS_problem(res, target, i)
        if alpha is not None:
            A = A + alpha * np.eye(A.shape[0])
        beta = LS_solve(A, B)
        return beta.T

    def fit_step_ls(self, res_tuple, target, i, alpha=None):
        A, B = self.get_LS_problem(res_tuple, target, i)
        if alpha is None:
            reg = LinearRegression(fit_intercept=False, positive=True)
            beta = reg.fit(A, B)
        else:
            reg = Ridge(alpha=alpha, fit_intercept=False, positive=True)
            beta = reg.fit(A, B)
        return beta.T

    def payoff_function(self, x):
        if self.opt_type == 'c':
            return np.maximum(x - self.K, 0)
        elif self.opt_type == 'p':
            return np.maximum(self.K - x, 0)
        else:
            raise ValueError('Wrong option type')

    @timing
    def fit(self, alpha=1.):
        Y_array = np.zeros((self.model.N_samples, self.model.n_timesteps, self.model.d_assets))
        Y_array[:, -1, :] = self.payoff_function(self.model.price_path[:, -1, :])

        for k in range(self.model.n_timesteps - 2, -1, -1):
            print(f'Regressing time step: {k + 1}')
            res = Reservoir(n_internal_units=100, connectivity=0.5, input_scaling=0.1, seed=k)
            beta = self.fit_step_exact(res, Y_array[:, k + 1, :], k, alpha=alpha)

            Y_array[:, k, :] = self.get_solution(res, beta, k)
            # keep the price positive
            # Y_array[:, k, :] = np.maximum(self.get_solution(res, beta, k), 0)
            # Y_array[:, k, :] = np.abs(self.get_solution(res, beta, k))

        return Y_array


if __name__ == '__main__':
    BS = BlackScholes()
    path = BS.simulate_paths(n_timesteps=21, N_samples=50000)

    trainer = Trainer(BS)
    option_price_process = trainer.fit()
    # res = Reservoir(n_internal_units=500, connectivity=0.25, input_scaling=0.1)
    # res.get_reservoir_out(path[:, -1, :])

    plt.plot(BS.time_grid, option_price_process[:, :, 0].T)
    plt.xlabel('t')
    plt.ylabel(r'$C_t$')
    plt.show()

    print(f'Theo. price: {BS.call_price(S=BS.S0, T=1)}')
    print(f'MC price: {np.maximum(path[:, -1, :] - BS.K, 0).mean(0)}')
    print(f'PDE price: {option_price_process[:, 0, :].mean(0)}')
    print(f'MSE: {((BS.call_price(S=BS.S0, T=1) - option_price_process[:, 0, :].mean(0)) ** 2).mean()}')

    print('end.')
    print('end.')
