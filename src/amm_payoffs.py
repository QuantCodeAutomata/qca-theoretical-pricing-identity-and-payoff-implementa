"""AMM LP payoff functions for Uniswap v2 and v3.

This module implements the LP pool-value payoff functions as defined in the paper:
- Uniswap v2: g_v2(P) = 2 * sqrt(P) under unit liquidity
- Uniswap v3: g_v3(P, p_a, p_b) in both compact and piecewise forms

All values are denominated in token Y units (USDC in ETH-USDC context).
Custom — Context7 found no library equivalent for AMM LP payoff formulas.
"""
import numpy as np
from typing import Union

ArrayLike = Union[float, np.ndarray]


def g_v2(P: ArrayLike) -> ArrayLike:
    """Uniswap v2 LP payoff under unit liquidity.
    
    g_v2(P) = 2 * sqrt(P)
    
    This is the infinite-range limit of the Uniswap v3 payoff.
    Do NOT approximate with an extreme finite v3 range.
    
    Args:
        P: Token X price in token Y units (e.g., ETH price in USDC).
           Must be positive.
    
    Returns:
        LP pool value payoff in token Y units.
    """
    P = np.asarray(P, dtype=float)
    if np.any(P < 0):
        raise ValueError('Price P must be non-negative.')
    return 2.0 * np.sqrt(P)


def g_v3_compact(P: ArrayLike, p_a: float, p_b: float) -> ArrayLike:
    """Uniswap v3 LP payoff (compact form) under unit liquidity.
    
    g_v3(P) = P / sqrt(clamp(P, p_a, p_b)) - P / sqrt(p_b)
              + sqrt(clamp(P, p_a, p_b)) - sqrt(p_a)
    
    where clamp(P, p_a, p_b) = max(p_a, min(P, p_b)).
    
    Paper notation: v denotes max, ^ denotes min.
    
    Args:
        P: Token X price in token Y units. Scalar or array.
        p_a: Lower price boundary of the range (p_a > 0).
        p_b: Upper price boundary of the range (p_b > p_a).
    
    Returns:
        LP pool value payoff in token Y units.
    """
    if p_a <= 0:
        raise ValueError('p_a must be positive.')
    if p_b <= p_a:
        raise ValueError('p_b must be strictly greater than p_a.')
    P = np.asarray(P, dtype=float)
    if np.any(P < 0):
        raise ValueError('Price P must be non-negative.')
    # clamp = max(p_a, min(P, p_b))
    clamped = np.clip(P, p_a, p_b)
    return P / np.sqrt(clamped) - P / np.sqrt(p_b) + np.sqrt(clamped) - np.sqrt(p_a)


def g_v3_piecewise(P: ArrayLike, p_a: float, p_b: float) -> ArrayLike:
    """Uniswap v3 LP payoff (piecewise form) under unit liquidity.
    
    Piecewise definition:
      P < p_a  => g(P) = P * (1/sqrt(p_a) - 1/sqrt(p_b))
      p_a <= P <= p_b  => g(P) = 2*sqrt(P) - P/sqrt(p_b) - sqrt(p_a)
      P > p_b  => g(P) = sqrt(p_b) - sqrt(p_a)
    
    Args:
        P: Token X price in token Y units. Scalar or array.
        p_a: Lower price boundary (p_a > 0).
        p_b: Upper price boundary (p_b > p_a).
    
    Returns:
        LP pool value payoff in token Y units.
    """
    if p_a <= 0:
        raise ValueError('p_a must be positive.')
    if p_b <= p_a:
        raise ValueError('p_b must be strictly greater than p_a.')
    P = np.asarray(P, dtype=float)
    if np.any(P < 0):
        raise ValueError('Price P must be non-negative.')
    
    result = np.empty_like(P)
    
    # Region 1: P < p_a
    mask_below = P < p_a
    result[mask_below] = P[mask_below] * (1.0 / np.sqrt(p_a) - 1.0 / np.sqrt(p_b))
    
    # Region 2: p_a <= P <= p_b
    mask_in = (P >= p_a) & (P <= p_b)
    result[mask_in] = 2.0 * np.sqrt(P[mask_in]) - P[mask_in] / np.sqrt(p_b) - np.sqrt(p_a)
    
    # Region 3: P > p_b
    mask_above = P > p_b
    result[mask_above] = np.sqrt(p_b) - np.sqrt(p_a)
    
    return result


# Convenience alias: default g_v3 uses the compact form
def g_v3(P: ArrayLike, p_a: float, p_b: float) -> ArrayLike:
    """Uniswap v3 LP payoff (uses compact form by default).
    
    Args:
        P: Token X price in token Y units. Scalar or array.
        p_a: Lower price boundary (p_a > 0).
        p_b: Upper price boundary (p_b > p_a).
    
    Returns:
        LP pool value payoff in token Y units.
    """
    return g_v3_compact(P, p_a, p_b)


def verify_v3_equivalence(P: ArrayLike, p_a: float, p_b: float, tol: float = 1e-10) -> bool:
    """Verify that compact and piecewise v3 formulas agree within tolerance.
    
    Args:
        P: Price grid for verification.
        p_a: Lower range boundary.
        p_b: Upper range boundary.
        tol: Maximum allowed absolute error.
    
    Returns:
        True if maximum absolute difference is below tol.
    """
    g_compact = g_v3_compact(P, p_a, p_b)
    g_piece = g_v3_piecewise(P, p_a, p_b)
    max_err = np.max(np.abs(g_compact - g_piece))
    return max_err < tol


def check_shape_properties(P: np.ndarray, g: np.ndarray, tol_fd: float = 1e-6) -> dict:
    """Numerically inspect positivity, monotonicity, and concavity of g(P).
    
    Uses finite-difference first and second derivatives to verify:
    - g(P) >= 0 for P > 0
    - g'(P) >= 0 (non-decreasing)
    - g''(P) <= 0 (concave)
    
    Args:
        P: Sorted price array (uniform spacing required for FD).
        g: Payoff values at P.
        tol_fd: Tolerance for numerical derivative sign checks.
    
    Returns:
        Dictionary with boolean flags and max violations.
    """
    assert np.all(np.diff(P) > 0), 'P must be strictly increasing.'
    
    positivity_ok = bool(np.all(g >= -tol_fd))
    
    # First-order finite differences
    g_prime = np.diff(g) / np.diff(P)
    monotone_ok = bool(np.all(g_prime >= -tol_fd))
    
    # Second-order finite differences (interior points)
    h = np.diff(P)
    h_mid = 0.5 * (h[:-1] + h[1:])
    g_double_prime = (g[2:] - 2 * g[1:-1] + g[:-2]) / (h_mid ** 2)
    concave_ok = bool(np.all(g_double_prime <= tol_fd))
    
    return {
        'positivity': positivity_ok,
        'monotone': monotone_ok,
        'concave': concave_ok,
        'max_g_prime_violation': float(np.min(g_prime)),
        'max_g_double_prime': float(np.max(g_double_prime)),
    }
