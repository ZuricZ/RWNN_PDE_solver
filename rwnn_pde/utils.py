from __future__ import annotations

import contextlib
from functools import wraps
from time import time
from typing import Generator

import numpy as np


def timing(f):
    """Decorator that prints the wall-clock execution time of *f*."""

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func:{f.__name__!r} took: {te - ts:.4f} sec")
        return result

    return wrap


@contextlib.contextmanager
def temp_seed(seed: int) -> Generator[None, None, None]:
    """Context manager that temporarily sets the NumPy global random seed."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
