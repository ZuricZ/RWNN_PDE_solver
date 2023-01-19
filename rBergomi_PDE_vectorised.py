import numpy as np
from scipy.linalg import solve as LS_solve
from sklearn.linear_model import Ridge, LinearRegression
from dataclasses import dataclass
from typing import Literal

from rbergomi import rBergomi
from utils import temp_seed, timing, vanilla_payoff_function

from reservoir import Reservoir, ReLu, grad_ReLu

import matplotlib.pyplot as plt

# plt.style.use('dark_background')
plt.style.use('ggplot')


@dataclass
class Parameters:
    S0: int
    T: int
    H: float
    r: float
    rho: float
    xi: float
    eta: float
    K: int
    opt_type: Literal['c', 'p']

    n_hidden_nodes: int = 100
    connectivity: float = 0.5
    input_scaling: float = 0.25
    weight_compact_radius: float = 0.5


class roughBergomi:

    def __init__(self, parameters, N_samples, n_timesteps):

        for key, value in parameters.__dict__.items():
            setattr(self, key, value)

        self.N_samples = N_samples
        self.n_timesteps = n_timesteps
        self.delta = self.T / n_timesteps
        # initialised when simulate_paths() is called
        self.time_grid = None
        self.S = None
        self.V = None
        self.dW1 = None
        self.dW2 = None
        self.dB = None
        self.d_assets = 1
        self.simulate_paths(N_samples=N_samples, n_timesteps=n_timesteps)

    def call_price(self, N_samples=10**5, n_timesteps=100):
        S, _, _, _ = self.simulate_paths(N_samples=N_samples, n_timesteps=n_timesteps)
        return np.maximum(S[:, -1, 0] - self.K, 0).mean()

    def put_price(self, N_samples=10**5, n_timesteps=100):
        S, _, _, _ = self.simulate_paths(N_samples=N_samples, n_timesteps=n_timesteps)
        return np.maximum(self.K - S[:, -1, 0], 0).mean()

    @timing
    def simulate_paths(self, N_samples, n_timesteps):
        """
        :param N_samples:   number of samples
        :param n_timesteps: number of increments
        """
        rB = rBergomi(n=n_timesteps, N=N_samples, T=self.T, r=self.r, a=self.H - .5)

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


class Trainer:

    def __init__(self, model):

        for key, value in model.__dict__.items():
            setattr(self, key, value)

        self.a = - self.r
        self.b = 0
        self.c = 0
        self._einsum_optimize = 'greedy'  # optimal # greedy # False
        # TODO: could save the ein_sum_path after 1st iter to optimise performance

    @staticmethod
    def f_tilde(*args):
        return 0.

    def Sigma_func(self, V):
        # again matrix form for easier MV extension
        return np.sqrt(V)[:, :, None]

    def Y(self, y):
        return y + self.f_tilde() * self.delta

    def X1(self, res_out, i):
        # \Delta W^1_{t_i}) - \phi(X_{t_i})(b*\Delta t_i
        return np.einsum('ij,ik->ij', res_out, (self.dW1[:, i, :] - self.b * self.delta),
                         optimize=self._einsum_optimize)

    def X2(self, res_out, res_grad, i):
        # (1-a\Delta t_i)\phi(X_{t_i}) + \Delta W^1_{t_i}) +
        # (\nabla_\phi(X_{t_i}))\sigma(V_{t_i})(\Delta B_{t_i}-(b\rho + c\sqrt{1-\rho^2})\Delta t_i))
        scaled_res_out = (1 - self.a * self.delta) * res_out
        scaled_dB = (self.dB[:, i, :] - (
                    self.b * self.rho + self.c * np.sqrt(1 - self.rho ** 2)) * self.delta)
        SigmaBM_prod = np.einsum('ijk,ik->ij', self.Sigma_func(self.V[:, i, :]), scaled_dB,
                                 optimize=self._einsum_optimize)
        return scaled_res_out + np.einsum('ijk,ik->ij', res_grad, SigmaBM_prod, optimize=self._einsum_optimize)

    def get_LS_problem(self, res_tuple, target, i):
        res_out_xi, _ = res_tuple[0].get_reservoir_out(self.S[:, i, :])
        res_out_theta, res_grad_theta = res_tuple[1].get_reservoir_out(self.S[:, i, :])
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
        res_out, _ = res_theta.get_reservoir_out(self.S[:, i, :])

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

    @timing
    def fit(self, alpha=None, verbose=0, seed=0):
        Y_array = np.zeros((self.N_samples, self.n_timesteps + 1, self.d_assets))
        Y_array[:, -1, :] = vanilla_payoff_function(self.S[:, -1, :], self.K,
                                            opt_type=self.opt_type)

        for k in range(self.n_timesteps + 1 - 2, -1, -1):
            if verbose > 0: print(f'Regressing time step: {k + 1}')

            # res_tuple = (\Phi^xi, \Phi^\theta)
            res_tuple = (Reservoir(n_internal_units=self.n_hidden_nodes,
                                   connectivity=self.connectivity,
                                   input_scaling=self.input_scaling,
                                   weight_compact_radius=self.weight_compact_radius,
                                   seed=seed+k),
                         Reservoir(n_internal_units=self.n_hidden_nodes,
                                   connectivity=self.connectivity,
                                   input_scaling=self.input_scaling,
                                   weight_compact_radius=self.weight_compact_radius,
                                   seed=seed + k + self.n_timesteps))

            # same realisations of random bases for both RWNNs
            # res_tuple = (Reservoir(n_internal_units=100, connectivity=0.5, input_scaling=0.5, seed=k),)*2

            beta = self.fit_step_exact(res_tuple, Y_array[:, k + 1, :], k, alpha=alpha)

            Y_array[:, k, :] = self.get_solution(res_theta=res_tuple[1], beta=beta, i=k)
            # keep the price positive
            # Y_array[:, k, :] = np.maximum(self.get_solution(res_tuple[1], beta, k), 0)
            # Y_array[:, k, :] = np.abs(self.get_solution(res_tuple[1], beta, k))

        return Y_array


if __name__ == '__main__':
    params = Parameters(S0=1, T=1, H=0.3, r=0.05, rho=-0.7, xi=0.235 ** 2, eta=1.9, K=1, opt_type='c',
                        n_hidden_nodes=100, connectivity=0.5, input_scaling=0.1, weight_compact_radius=5.)
    with temp_seed(1000):
        rB = roughBergomi(parameters=params, n_timesteps=21, N_samples=50000)

    trainer = Trainer(rB)
    option_price_process = trainer.fit(alpha=1.)

    plt.plot(rB.time_grid.T,
             option_price_process[np.random.choice(option_price_process.shape[0], 100, replace=False), :, 0].T)
    plt.xlabel('t')
    plt.ylabel(r'$C_t$')
    # plt.savefig('./Graphics/rBergomi_option_price.pdf', format='pdf')
    plt.show()

    with temp_seed(1000):
        theo_price = rB.call_price() if rB.opt_type == 'c' else rB.put_price()

    print(f'Theo. price: {theo_price}')
    print(f'MC price: {vanilla_payoff_function(S=rB.S[:, -1, :], K=rB.K).mean(0)}')
    print(f'PDE price: {option_price_process[:, 0, :].mean(0)}')
    print(f'MSE: {((theo_price - option_price_process[:, 0, :].mean(0)) ** 2).mean()}')

    print('end.')
