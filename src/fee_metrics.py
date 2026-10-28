"""Fee metrics for AMM LP implied fee compensation.

Implements the theoretical pricing identity:
    E_t^QY[e^{-r^Y(T-t)} Gamma_{t,T}^R] = g_R(S_t) - E_t^QY[e^{-r^Y(T-t)} g_R(S_T)]

Custom — Context7 found no library equivalent for AMM LP fee metrics.
"""
import numpy as np
from typing import Union

ArrayLike = Union[float, np.ndarray]


def fee_value(g0: ArrayLike, H: ArrayLike) -> ArrayLike:
    """Compute implied accumulated fee value.
    
    FeeValue = g_R(S_t) - H_{t,T}
    
    where H_{t,T} = E_t^QY[e^{-r^Y(T-t)} g_R(S_T)] is the risk-neutral
    discounted expected terminal LP payoff.
    
    Args:
        g0: Current LP pool value g_R(S_t). Must be positive.
        H:  Risk-neutral discounted expected terminal payoff H_{t,T}.
    
    Returns:
        Implied accumulated fee value FeeValue = g0 - H.
    """
    g0 = np.asarray(g0, dtype=float)
    H = np.asarray(H, dtype=float)
    return g0 - H


def fee_return(g0: ArrayLike, H: ArrayLike) -> ArrayLike:
    """Compute implied total fee return.
    
    R_{t,T}^R = FeeValue / g_R(S_t) = 1 - H_{t,T} / g_R(S_t)
    
    Under nonnegative rates and valid pricing: 0 <= R < 1.
    
    Args:
        g0: Current LP pool value g_R(S_t). Must be positive.
        H:  Risk-neutral discounted expected terminal payoff H_{t,T}.
    
    Returns:
        Implied total fee return in [0, 1).
    """
    g0 = np.asarray(g0, dtype=float)
    H = np.asarray(H, dtype=float)
    if np.any(g0 <= 0):
        raise ValueError('g0 must be strictly positive.')
    return 1.0 - H / g0


def fee_rate(R: ArrayLike, theta: ArrayLike) -> ArrayLike:
    """Compute continuously compounded implied fee rate.
    
    y(t, theta) = (1/theta) * log(1 + R_{t,t+theta}^R)
    
    Args:
        R:     Implied total fee return R_{t,T}^R. Should be in [0, 1).
        theta: Time to maturity in years (theta > 0).
    
    Returns:
        Continuously compounded implied fee rate y(t, theta).
    """
    R = np.asarray(R, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if np.any(theta <= 0):
        raise ValueError('theta must be strictly positive.')
    if np.any(R < 0):
        raise ValueError('R must be non-negative for well-defined log.')
    return np.log1p(R) / theta


def validate_bounds(g0: ArrayLike, H: ArrayLike, tol: float = 1e-12) -> dict:
    """Validate theoretical bounds on fee value and return.
    
    Under nonnegative rates:
        0 <= FeeValue < g_R(S_t)
        0 <= R_{t,T}^R < 1
    
    Args:
        g0:  Current LP pool value. Must be positive.
        H:   Terminal payoff price.
        tol: Numerical tolerance for bound checks.
    
    Returns:
        Dictionary with bound validation results.
    """
    g0 = np.asarray(g0, dtype=float)
    H = np.asarray(H, dtype=float)
    fv = fee_value(g0, H)
    R = fee_return(g0, H)
    return {
        'fee_value_nonneg': bool(np.all(fv >= -tol)),
        'fee_value_less_g0': bool(np.all(fv < g0 + tol)),
        'return_nonneg': bool(np.all(R >= -tol)),
        'return_less_one': bool(np.all(R < 1.0 + tol)),
        'max_R': float(np.max(R)),
        'min_R': float(np.min(R)),
    }
