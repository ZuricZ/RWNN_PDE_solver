from functools import wraps
from time import time
import numpy as np
import contextlib


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result
    return wrap


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def vanilla_payoff_function(S, K, opt_type='c'):
    if opt_type not in 'CcPp':
        raise ValueError('Wrong option type.')
    opt_type_ind = 1 if opt_type == 'c' else -1
    return np.maximum(opt_type_ind*(S - K), 0)


def basket_payoff_function(S, K, weight=None, opt_type='c'):
    if opt_type not in 'CcPp':
        raise ValueError('Wrong option type.')
    if weight is None:
        weight = np.ones(S.shape[1]) / S.shape[1]
    opt_type_ind = 1 if opt_type == 'c' else -1
    return np.maximum(opt_type_ind*(np.average(S, weights=weight, axis=1, keepdims=True) - K), 0)

