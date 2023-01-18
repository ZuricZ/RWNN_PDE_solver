import matplotlib.pyplot as plt

# plt.style.use('dark_background')
plt.style.use('ggplot')
from mpl_toolkits.mplot3d import Axes3D

# %matplotlib notebook

import numpy as np
from scipy.stats import norm
import scipy.io
from scipy.linalg import solve as LS_solve
from scipy import sparse
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import accuracy_score, f1_score
from dataclasses import dataclass
from typing import Literal

from utils import temp_seed, timing, vanilla_payoff_function
from reservoir import Reservoir, ReLu, grad_ReLu


@dataclass
class Parameters:
    d: int
    T: float
    r: float
    S0: np.ndarray
    K: np.ndarray
    sigma: np.ndarray
    Cov: np.ndarray
    opt_type: Literal['c', 'p']

    n_hidden_nodes: int = 100
    connectivity: float = 0.5
    input_scaling: float = 0.1
    weight_compact_radius: float = 0.5


class BlackScholes:

    def __init__(self, parameters, N_samples, n_timesteps):

        for key, value in parameters.__dict__.items():
            setattr(self, key, value)

        self.N_samples = N_samples
        self.n_timesteps = n_timesteps

        # initialised when simulate_paths() is called
        self.time_grid = None
        self.delta = None
        self.diffusion = None
        self.S = None
        self.dW = None
        self.d_assets = None

        self.simulate_paths(N_samples=N_samples, n_timesteps=n_timesteps)

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
        d_assets = self.S0.shape[0]

        time_grid = np.linspace(0., self.T, n_timesteps)
        diffusion = np.linalg.cholesky(self.Cov)
        dt = time_grid[1:] - time_grid[:-1]
        dW = np.tile(np.sqrt(dt[None, :, None]), (N_samples, 1, d_assets)) * np.random.randn(N_samples,
                                                                                             n_timesteps - 1,
                                                                                             d_assets)
        drift = np.tile(((self.r - 0.5 * self.sigma[:, None] ** 2) * dt).T, (N_samples, 1, 1))

        S = np.zeros((N_samples, n_timesteps, d_assets))
        S[:, 0] = self.S0
        for i in range(1, n_timesteps):
            S[:, i, :] = S[:, i - 1, :] * np.exp(
                drift[:, i - 1, :] + np.matmul(diffusion, dW[:, i - 1, :].T).T)

        # initialise for further use
        if self.S is None:
            self.d_assets = d_assets
            self.time_grid = time_grid
            self.delta = dt
            self.diffusion = diffusion
            self.S = S
            self.dW = dW
        return S


class Trainer:

    def __init__(self, model):

        for key, value in model.__dict__.items():
            setattr(self, key, value)

        self._einsum_optimize = 'greedy'  # optimal # greedy # False

    def a(self, res_out):
        # In Black Scholes model f(u)=-r*u, therefore f_tilde(x)=-r*reservoir_out(x)
        return - self.r * res_out

    def Sigma_func(self, S):
        # broadcast across columns (same as diag(S).dot(diffusion)
        return self.diffusion * S[:, :, None]

    def X(self, res, S, dt, dW):
        # res_out(x) - f_tilde(x) + res_grad(x)*Sigma(x)*dW
        res_out, res_grad = res.get_reservoir_out(S)
        SigmaBM_prod = np.einsum('ijk,ik->ij', self.Sigma_func(S), dW, optimize=self._einsum_optimize)
        return res_out - self.a(res_out) * dt + np.einsum('ijk,ik->ij', res_grad, SigmaBM_prod,
                                                          optimize=self._einsum_optimize)

    def get_LS_problem(self, res, target, i):
        single_regr = self.X(res, S=self.S[:, i, :],
                             dt=self.time_grid[i + 1] - self.time_grid[i],
                             dW=self.dW[:, i, :])

        # perform the summation of the outer-products https://stackoverflow.com/q/35549082/5285408
        A = np.einsum('ki,kj->ij', single_regr, single_regr, optimize=self._einsum_optimize)

        B = np.einsum('ki,kj->ij', single_regr, target, optimize=self._einsum_optimize)
        return A, B

    def get_solution(self, res, beta, i):
        # evaluates the solution of the regression
        res_out, _ = res.get_reservoir_out(self.S[:, i, :])
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

    @timing
    def fit(self, alpha=1., verbose=0, seed=0):
        Y_array = np.zeros((self.N_samples, self.n_timesteps, self.d_assets))
        Y_array[:, -1, :] = vanilla_payoff_function(self.S[:, -1, :], self.K, opt_type=self.opt_type)

        for k in range(self.n_timesteps - 2, -1, -1):
            if verbose > 0: print(f'Regressing time step: {k + 1}')

            res = Reservoir(n_internal_units=self.n_hidden_nodes,
                            connectivity=self.connectivity,
                            input_scaling=self.input_scaling,
                            seed=seed+k)
            beta = self.fit_step_exact(res, Y_array[:, k + 1, :], k, alpha=alpha)

            Y_array[:, k, :] = self.get_solution(res, beta, k)

            # keep the price positive
            # Y_array[:, k, :] = np.maximum(self.get_solution(res, beta, k), 0)
            # Y_array[:, k, :] = np.abs(self.get_solution(res, beta, k))

        return Y_array


if __name__ == '__main__':
    d = 5; sigma = np.linspace(0.05, 0.4, d)
    params = Parameters(d=d, S0=np.array([1.] * d),  # np.random.uniform(0.5, 2, 5)
                        T=1.,
                        r=0.05,
                        K=np.array([1.] * d),
                        sigma=sigma,
                        Cov=np.diag(sigma ** 2),
                        opt_type='c',
                        n_hidden_nodes=1000,
                        connectivity=0.5,
                        input_scaling=0.1
                        )

    BS = BlackScholes(params, n_timesteps=21, N_samples=50000)

    trainer = Trainer(BS)
    option_price_process = trainer.fit()

    plt.plot(BS.time_grid,
             option_price_process[np.random.choice(option_price_process.shape[0], 200, replace=False), :, 0].T)
    plt.xlabel('t')
    plt.ylabel(r'$C_t$')
    # plt.savefig('./Graphics/BS_option_price.pdf', format='pdf')
    plt.show()

    print(f'Theo. price: {BS.call_price(S=BS.S0, T=1)}')
    print(f'MC price: {vanilla_payoff_function(S=BS.S[:, -1, :], K=BS.K).mean(0)}')
    print(f'PDE price: {option_price_process[:, 0, :].mean(0)}')
    print(f'MSE: {((BS.call_price(S=BS.S0, T=1) - option_price_process[:, 0, :].mean(0)) ** 2).mean()}')

    print('end.')
    print('end.')
