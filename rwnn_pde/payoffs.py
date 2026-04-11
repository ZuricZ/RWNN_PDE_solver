from __future__ import annotations

from typing import Optional

import numpy as np


def vanilla_payoff(
    S: np.ndarray,
    K: float,
    opt_type: str = "c",
) -> np.ndarray:
    """European vanilla option payoff: max(±(S − K), 0).

    Parameters
    ----------
    S:
        Asset prices at maturity, shape (N,) or (N, d).
    K:
        Strike price.
    opt_type:
        ``'c'`` for call, ``'p'`` for put.
    """
    if opt_type not in "CcPp":
        raise ValueError(f"Unknown option type {opt_type!r}. Expected 'c' or 'p'.")
    sign = 1 if opt_type in "Cc" else -1
    return np.maximum(sign * (S - K), 0)


def basket_payoff(
    S: np.ndarray,
    K: float,
    weight: Optional[np.ndarray] = None,
    opt_type: str = "c",
) -> np.ndarray:
    """Basket option payoff on the weighted average of assets.

    Parameters
    ----------
    S:
        Asset prices at maturity, shape (N, d).
    K:
        Strike price.
    weight:
        Asset weights of shape (d,). Defaults to equal weights 1/d.
    opt_type:
        ``'c'`` for call, ``'p'`` for put.
    """
    if opt_type not in "CcPp":
        raise ValueError(f"Unknown option type {opt_type!r}. Expected 'c' or 'p'.")
    if weight is None:
        weight = np.ones(S.shape[1]) / S.shape[1]
    sign = 1 if opt_type in "Cc" else -1
    basket = np.average(S, weights=weight, axis=1, keepdims=True)
    return np.maximum(sign * (basket - K), 0)


def payoff(
    S: np.ndarray,
    K: float,
    opt_style: str,
    opt_type: str,
    **kwargs,
) -> np.ndarray:
    """Dispatch payoff computation by option style.

    Parameters
    ----------
    S:
        Asset prices at maturity.
    K:
        Strike price.
    opt_style:
        ``'vanilla'`` or ``'basket'``.
    opt_type:
        ``'c'`` or ``'p'``.
    **kwargs:
        Passed through to the underlying payoff function
        (e.g. ``weight`` for basket options).
    """
    if opt_style == "vanilla":
        return vanilla_payoff(S, K, opt_type=opt_type)
    elif opt_style == "basket":
        return basket_payoff(S, K, weight=kwargs.get("weight"), opt_type=opt_type)
    else:
        raise ValueError(f"Unknown option style {opt_style!r}. Expected 'vanilla' or 'basket'.")
