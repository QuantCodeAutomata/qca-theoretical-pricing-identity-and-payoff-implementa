"""Bates/SVJD model calibration to option cross-sections.

Calibrates [lambda, kappa, b_v, beta, sigma_v, V0, rho, delta, rX, rY]
by minimizing weighted squared implied-vol errors against Deribit option
quotes. Uses scipy.optimize with differential_evolution for global search
followed by L-BFGS-B local refinement.

Calibration objective: weighted sum of squared vega-weighted price errors,
or alternatively IVMSE (implied-vol MSE) if IV inversion is stable.

Custom — Context7 found no library equivalent for Bates model calibration
to AMM-context crypto option data.
"""
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import quad
from scipy.stats import norm
from typing import Optional, List, Tuple, Dict, Any
import warnings

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.bates_cf import BatesParams, bates_cf

# Parameter bounds: [lam, kappa, b_v, beta, sigma_v, V0, rho, delta, rX, rY]
PARAM_BOUNDS = [
    (1e-4, 5.0),    # lam:     jump intensity
    (-2.0, 0.5),    # kappa:   mean log-jump
    (1e-4, 100.0),  # b_v:     long-run variance drift
    (1e-4, 200.0),  # beta:    mean-reversion speed
    (1e-4, 20.0),   # sigma_v: vol-of-vol
    (1e-6, 2.0),    # V0:      initial variance
    (-0.99, 0.99),  # rho:     correlation
    (1e-4, 2.0),    # delta:   jump vol
    (0.0, 0.0),     # rX:      rate token X (fixed at 0)
    (0.0, 0.0),     # rY:      rate token Y (fixed at 0)
]


def _bates_call_price(S: float, K: float, theta: float, params: BatesParams) -> float:
    """Price a European call option using Bates CF via Lewis (2001) formula.
    
    C = S * P1 - K * exp(-rY*theta) * P2
    
    Using the single-integral form of Lewis (2001):
    C = S - (K*exp(-rY*theta)/pi) * Re[ integral_0^inf e^{-i*u*log(K/F)}
                                          / (u^2 + u*i) * phi(u - i/2) du ]
    where phi is the characteristic function of log(S_T/S).
    
    Args:
        S:      Spot price.
        K:      Strike price.
        theta:  Time to maturity.
        params: BatesParams.
    
    Returns:
        European call price (in token Y units).
    """
    # Forward price (r^X = r^Y = 0 in empirical setup)
    F = S * np.exp((params.rY - params.rX) * theta)
    disc = np.exp(-params.rY * theta)
    log_K_F = np.log(K / F)

    def integrand(u: float) -> float:
        if abs(u) < 1e-10:
            return 0.0
        z = complex(0.5, u)  # phi evaluated at u - i/2 => G at z=1/2+iu
        cf_val = bates_cf(z, S, params.V0, theta, params)
        # Denominator in Lewis formula: (u^2 + 1/4)
        denom = u ** 2 + 0.25
        kernel = np.exp(-1j * u * log_K_F) / denom * cf_val
        return float(np.real(kernel))

    # Upper limit 200 is sufficient for crypto vols; limit=200 caps sub-intervals
    integral, _ = quad(integrand, 0.0, 200.0, limit=200, epsabs=1e-6, epsrel=1e-6)
    call = S - (np.sqrt(S * K) * disc / np.pi) * integral
    return max(float(call), 0.0)


def bs_implied_vol(S: float, K: float, theta: float, price: float,
                   option_type: str = 'C', rY: float = 0.0,
                   rX: float = 0.0) -> Optional[float]:
    """Compute Black-Scholes implied volatility via bisection.
    
    Args:
        S:           Spot price.
        K:           Strike price.
        theta:       Time to maturity.
        price:       Option price.
        option_type: 'C' for call, 'P' for put.
        rY:          Short rate for token Y.
        rX:          Short rate for token X.
    
    Returns:
        Implied volatility, or None if inversion fails.
    """
    F = S * np.exp((rY - rX) * theta)
    disc = np.exp(-rY * theta)

    def bs_price(sigma: float) -> float:
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * theta) / (sigma * np.sqrt(theta))
        d2 = d1 - sigma * np.sqrt(theta)
        if option_type.upper() == 'C':
            return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    lo, hi = 1e-4, 20.0
    if bs_price(hi) < price:
        return None
    if bs_price(lo) > price:
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if bs_price(mid) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7:
            break
    return 0.5 * (lo + hi)


def calibration_objective(
    x: np.ndarray,
    options_df: pd.DataFrame,
    S: float,
) -> float:
    """Calibration objective: weighted RMSE of IV errors.
    
    Args:
        x:          Parameter vector [lam, kappa, b_v, beta, sigma_v,
                                      V0, rho, delta, rX, rY].
        options_df: DataFrame with columns [strike, theta, price, iv_market,
                                            option_type, vega_weight].
        S:          Current spot price.
    
    Returns:
        Weighted RMSE of model vs market implied volatilities.
    """
    # Fix rates at 0
    x = np.array(x, dtype=float)
    x[8] = 0.0  # rX
    x[9] = 0.0  # rY

    try:
        params = BatesParams.from_array(x)
        errors = []
        weights = []
        for _, row in options_df.iterrows():
            K, theta, iv_mkt = row['strike'], row['theta'], row['iv_market']
            otype = row.get('option_type', 'C')
            w = row.get('vega_weight', 1.0)
            model_price = _bates_call_price(S, K, theta, params)
            if otype.upper() == 'P':
                # Put-call parity: P = C - S + K*disc
                model_price = model_price - S + K * np.exp(-params.rY * theta)
            iv_model = bs_implied_vol(S, K, theta, model_price, otype, params.rY, params.rX)
            if iv_model is None or iv_model <= 0:
                return 1e6
            errors.append((iv_model - iv_mkt) ** 2 * w)
            weights.append(w)
        if not errors:
            return 1e6
        return float(np.sqrt(np.sum(errors) / np.sum(weights)))
    except Exception:
        return 1e6


def calibrate_bates(
    options_df: pd.DataFrame,
    S: float,
    x0: Optional[np.ndarray] = None,
    max_iter_de: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """Calibrate Bates/SVJD model to an option cross-section.
    
    Uses differential_evolution for global search, then L-BFGS-B.
    Fixes r^X = r^Y = 0 as per empirical convention.
    
    Args:
        options_df:  DataFrame with columns [strike, theta, price, iv_market,
                                              option_type, vega_weight].
        S:           Current spot price.
        x0:          Optional initial parameter guess.
        max_iter_de: Maximum iterations for differential evolution.
        seed:        Random seed for reproducibility.
    
    Returns:
        Dictionary with keys:
            'params': BatesParams (calibrated),
            'obj_value': float,
            'success': bool,
            'message': str,
            'x': np.ndarray (raw parameter vector),
    """
    # Fixed bounds (rates fixed at 0)
    bounds = PARAM_BOUNDS.copy()
    bounds[8] = (0.0, 0.0)
    bounds[9] = (0.0, 0.0)

    fixed_bounds_de = [(lo, hi) if lo < hi else (lo, lo + 1e-10)
                       for lo, hi in bounds]

    def obj(x: np.ndarray) -> float:
        return calibration_objective(x, options_df, S)

    # --- Global search (skipped when a warm-start x0 is already provided) ---
    if x0 is not None:
        # Warm-start: trust the caller's initial point, skip expensive DE
        x_start = np.array(x0, dtype=float)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            de_result = differential_evolution(
                obj,
                bounds=fixed_bounds_de,
                maxiter=max_iter_de,
                seed=seed,
                tol=1e-4,
                popsize=5,          # 5×Ndim evaluations per iteration (fast demo)
                mutation=(0.5, 1.5),
                recombination=0.7,
            )
        x_start = de_result.x if de_result.success else np.array(
            [0.01, -0.1, 5.0, 10.0, 2.0, 0.1, -0.5, 0.1, 0.0, 0.0]
        )

    # --- Local refinement ---
    lbfgsb_bounds = [(lo, hi) for lo, hi in bounds]
    local_result = minimize(
        obj,
        x_start,
        method='L-BFGS-B',
        bounds=lbfgsb_bounds,
        options={'maxiter': 1000, 'ftol': 1e-10},
    )

    x_best = local_result.x
    x_best[8] = 0.0
    x_best[9] = 0.0
    params = BatesParams.from_array(x_best)

    return {
        'params': params,
        'obj_value': float(local_result.fun),
        'success': bool(local_result.success),
        'message': str(local_result.message),
        'x': x_best,
    }
