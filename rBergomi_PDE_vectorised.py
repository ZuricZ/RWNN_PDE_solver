import numpy as np
from scipy.linalg import solve as LS_solve
from scipy import sparse
from sklearn.linear_model import Ridge, LinearRegression

from rbergomi import rBergomi
from utils import temp_seed, timing

import matplotlib.pyplot as plt

plt.style.use('dark_background')


def ReLu(X):
    return np.maximum(0, X)


def grad_ReLu(X):
    """returns a vector of derivatives"""
    return np.maximum(np.sign(X.reshape(X.shape[0], -1)), 0)


def grad_tanh(X):
    """returns a vector of derivatives"""
    X = X.reshape(X.shape[0], -1)
    return np.ones_like(X) - (X ** 2)


class Parameters:
    def __init__(self):
        super(Parameters, self).__init__()
        self.S0 = 1
        self.T = 1
        self.H = 0.3
        self.r = 0.05
        self.rho = -0.7
        self.xi = 0.235 ** 2
        self.eta = 1.9
        self.K = 1
        # self.sigma = np.linspace(0.05, 0.4, 5)
        # self.Cov = np.diag(self.sigma ** 2)
        self.opt_type = 'c'


class roughBergomi(Parameters):

    def __init__(self, N_samples, n_timesteps):
        super().__init__()

        self.N_samples = N_samples
        self.n_timesteps = n_timesteps
        self.delta = self.T/n_timesteps
        # initialised when simulate_paths() is called
        self.time_grid = None
        self.S = None
        self.V = None
        self.dW1 = None
        self.dW2 = None
        self.dB = None
        self.d_assets = 1
        self.simulate_paths(N_samples=N_samples, n_timesteps=n_timesteps)

    def call_price(self):
        S, _, _, _ = self.simulate_paths(N_samples=75000, n_timesteps=100)
        return np.maximum(S[:, -1, 0] - self.K, 0).mean()

    def put_price(self):
        S, _, _, _ = self.simulate_paths(N_samples=75000, n_timesteps=100)
        return np.maximum(self.K - S[:, -1, 0], 0).mean()

    @timing
    def simulate_paths(self, N_samples, n_timesteps):
        """
        :param N_samples:   number of samples
        :param n_timesteps: number of increments
        """
        rB = rBergomi(n=n_timesteps, N=N_samples, T=self.T, r=self.r, a=self.H-.5)

        dW1 = rB.dW1()
        dW2 = rB.dW2()

        dB = rB.dB(dW1, dW2, rho=self.rho)

        Y = rB.Y(dW1)
        V = rB.V(Y, xi=self.xi, eta=self.eta)
        S = rB.S(V, dB, S0=self.S0)
        dW1 = dW1[:, :, 0]  # the [:,:,1] part is used for convolution in the rBergomi class

        # add an extra dimension - easier extension to MV setting if needed
        S = S[:, :, None]
        V = V[:, :, None]
        dW1 = dW1[:, :, None]
        dW2 = dW2[:, :, None]
        dB = dB[:, :, None]

        if self.S is None:
            self.time_grid = rB.t
            self.S = S
            self.V = V
            self.dW1 = dW1
            self.dW2 = dW2
            self.dB = dB

        return S, V, dW1, dW2


class Reservoir(object):
    """
    Build a reservoir and evaluate internal states

    Parameters:
        n_internal_units = processing units in the reservoir
        connectivity = percentage of nonzero connection weights
        input_scaling = scaling of the input connection weights
    """

    def __init__(self, n_internal_units=1000, connectivity=0.25, input_scaling=0.2,
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
        self.a = - self.r
        self.b = 0
        self.c = 0
        self._einsum_optimize = 'greedy'  # optimal # greedy # False TODO: could save the ein_sum_path after 1st iter

    @staticmethod
    def f_tilde(*args):
        return 0.

    def Sigma_func(self, V):
        # again matrix form for easier MV extension
        return np.sqrt(V)[:, :, None]

    def Y(self, y):
        return y + self.f_tilde()*self.model.delta

    def X1(self, res_out, i):
        # \Delta W^1_{t_i}) - \phi(X_{t_i})(b*\Delta t_i
        return np.einsum('ij,ik->ij', res_out, (self.model.dW1[:, i, :]-self.b*self.model.delta),
                         optimize=self._einsum_optimize)

    def X2(self, res_out, res_grad, i):
        # (1-a\Delta t_i)\phi(X_{t_i}) + \Delta W^1_{t_i}) +
        # (\nabla_\phi(X_{t_i}))\sigma(V_{t_i})(\Delta B_{t_i}-(b\rho + c\sqrt{1-\rho^2})\Delta t_i))
        scaled_res_out = (1-self.a*self.model.delta)*res_out
        scaled_dB = (self.model.dB[:, i, :]-(self.b*self.rho + self.c*np.sqrt(1-self.rho**2))*self.model.delta)
        SigmaBM_prod = np.einsum('ijk,ik->ij', self.Sigma_func(self.model.V[:, i, :]), scaled_dB,
                                 optimize=self._einsum_optimize)
        return scaled_res_out + np.einsum('ijk,ik->ij', res_grad, SigmaBM_prod, optimize=self._einsum_optimize)

    def get_LS_problem(self, res_tuple, target, i):
        res_out_xi, _ = res_tuple[0].get_reservoir_out(self.model.S[:, i, :])
        res_out_theta, res_grad_theta = res_tuple[1].get_reservoir_out(self.model.S[:, i, :])
        X1_vec = self.X1(res_out=res_out_xi, i=i)
        X2_vec = self.X2(res_out=res_out_theta, res_grad=res_grad_theta, i=i)

        # perform the summation of the outer-products https://stackoverflow.com/q/35549082/5285408
        A11 = np.einsum('ki,kj->ij', X1_vec, X1_vec, optimize=self._einsum_optimize)
        A12 = np.einsum('ki,kj->ij', X1_vec, X2_vec, optimize=self._einsum_optimize)
        # A21 = np.einsum('ki,kj->ij', X2_vec, X1_vec)
        A21 = A12.T
        A22 = np.einsum('ki,kj->ij', X2_vec, X2_vec, optimize=self._einsum_optimize)

        B1 = np.einsum('ki,kj->ij', X1_vec, target, optimize=self._einsum_optimize)
        B2 = np.einsum('ki,kj->ij', X2_vec, target, optimize=self._einsum_optimize)

        A = np.block([[A11, A12], [A21, A22]])
        B = np.concatenate([B1, B2], axis=0)
        return A, B

    def get_solution(self, res_theta, beta, i):
        # evaluate the solution at time t_i
        res_out, _ = res_theta.get_reservoir_out(self.model.S[:, i, :])

        # take only \theta from \beta = [\xi, \theta]
        target = np.tensordot(beta[:, res_theta._n_internal_units:], res_out, axes=(1, 1)).T
        return target

    def fit_step_exact(self, res_tuple, target, i, alpha=None):
        A, B = self.get_LS_problem(res_tuple, target, i)
        if alpha is not None:
            A = A + alpha * np.eye(A.shape[0])
        beta = LS_solve(A, B)
        return beta.T

    def fit_step_ls(self, res_tuple, target, i, alpha=None):
        A, B = self.get_LS_problem(res_tuple, target, i)
        if alpha is None:
            reg = LinearRegression(fit_intercept=False,
                                   # positive=True
                                   )
            beta = reg.fit(A, B).coef_
        else:
            reg = Ridge(alpha=alpha, fit_intercept=False,
                        # positive=True
                        )
            beta = reg.fit(A, B).coef_
        return beta

    def payoff_function(self, x):
        if self.opt_type == 'c':
            return np.maximum(x - self.K, 0)
        elif self.opt_type == 'p':
            return np.maximum(self.K - x, 0)
        else:
            raise ValueError('Wrong option type')

    @timing
    def fit(self, alpha=None):
        Y_array = np.zeros((self.model.N_samples, self.model.n_timesteps+1, self.model.d_assets))
        Y_array[:, -1, :] = self.payoff_function(self.model.S[:, -1, :])

        for k in range(self.model.n_timesteps+1 - 2, -1, -1):
            print(f'Regressing time step: {k + 1}')

            # res_tuple = (\Phi^xi, \Phi^\theta)
            res_tuple = (Reservoir(n_internal_units=100, connectivity=0.5, input_scaling=0.25, seed=k),
                         Reservoir(n_internal_units=100, connectivity=0.5, input_scaling=0.25, seed=k+self.model.n_timesteps))

            # same realisations of random bases
            # res_tuple = (Reservoir(n_internal_units=100, connectivity=0.5, input_scaling=0.5, seed=k),)*2

            beta = self.fit_step_exact(res_tuple, Y_array[:, k + 1, :], k, alpha=alpha)

            # Y_array[:, k, :] = self.get_solution(res_theta=res_tuple[1], beta=beta, i=k)
            # keep the price positive
            Y_array[:, k, :] = np.maximum(self.get_solution(res_tuple[1], beta, k), 0)
            # Y_array[:, k, :] = np.abs(self.get_solution(res_tuple[1], beta, k))

        return Y_array


if __name__ == '__main__':
    with temp_seed(1000):
        rB = roughBergomi(n_timesteps=21, N_samples=50000)
    # path = rB.simulate_paths(n_timesteps=21, N_samples=5000)

    trainer = Trainer(rB)
    option_price_process = trainer.fit(alpha=1.)
    # res = Reservoir(n_internal_units=500, connectivity=0.25, input_scaling=0.1)
    # res.get_reservoir_out(path[:, -1, :])

    plt.plot(rB.time_grid.T,
             option_price_process[np.random.choice(option_price_process.shape[0], 1000, replace=False), :, 0].T)
    plt.xlabel('t')
    plt.ylabel(r'$C_t$')
    plt.show()

    with temp_seed(1000):
        theo_price = rB.call_price() if rB.opt_type == 'c' else rB.put_price()

    print(f'Theo. price: {theo_price}')
    print(f'MC price: {np.maximum(rB.S[:, -1, :] - rB.K, 0).mean(0)}')
    print(f'PDE price: {option_price_process[:, 0, :].mean(0)}')
    print(f'MSE: {((theo_price - option_price_process[:, 0, :].mean(0)) ** 2).mean()}')

    print('end.')
