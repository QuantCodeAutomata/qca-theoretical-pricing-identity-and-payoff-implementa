"""Data loading utilities for AMM LP pricing experiments.

Provides synthetic data generation (for exp_1) and option data structures
(for exp_2-5 with Deribit data).

For this repository, empirical option data from Deribit is loaded from
pre-cleaned CSV files. The Massive API does not cover Deribit crypto options,
so data must be externally sourced. Synthetic data generators are provided
for formula validation experiments.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import os


def synthetic_price_grid(
    S_t: float = 2000.0,
    n_points: int = 500,
    p_min_frac: float = 0.1,
    p_max_frac: float = 3.0,
) -> np.ndarray:
    """Generate a dense synthetic price grid for formula validation.
    
    Args:
        S_t:         Reference spot price.
        n_points:    Number of price points.
        p_min_frac:  Minimum price as fraction of S_t.
        p_max_frac:  Maximum price as fraction of S_t.
    
    Returns:
        Array of price values.
    """
    return np.linspace(S_t * p_min_frac, S_t * p_max_frac, n_points)


def synthetic_maturity_grid(
    theta_min: float = 0.05,
    theta_max: float = 1.0,
    n_points: int = 20,
) -> np.ndarray:
    """Generate a maturity grid in years.
    
    Args:
        theta_min: Minimum maturity (years).
        theta_max: Maximum maturity (years).
        n_points:  Number of maturity points.
    
    Returns:
        Array of maturity values in years.
    """
    return np.linspace(theta_min, theta_max, n_points)


def mock_lognormal_terminal_payoff(
    S_t: float,
    theta: float,
    sigma: float,
    payoff_fn,
    r: float = 0.0,
    n_samples: int = 100_000,
    seed: int = 42,
) -> float:
    """Monte Carlo H_{t,T} under a lognormal (Black-Scholes) model.
    
    H_{t,T} = E^Q[e^{-r*theta} * g_R(S_T)]
    
    Used as a mock pricing engine for formula validation in exp_1.
    
    Args:
        S_t:       Current spot price.
        theta:     Time to maturity in years.
        sigma:     Lognormal volatility.
        payoff_fn: Callable g_R(P) returning payoff.
        r:         Discount rate.
        n_samples: Number of Monte Carlo samples.
        seed:      Random seed.
    
    Returns:
        Discounted expected terminal payoff H_{t,T}.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_samples)
    S_T = S_t * np.exp((r - 0.5 * sigma ** 2) * theta + sigma * np.sqrt(theta) * Z)
    payoffs = payoff_fn(S_T)
    H = np.exp(-r * theta) * np.mean(payoffs)
    return float(H)


def load_option_data_from_csv(filepath: str) -> Optional[pd.DataFrame]:
    """Load option data from a CSV file.
    
    Expected columns: date, expiry, strike, option_type, price, spot, iv_market.
    
    Args:
        filepath: Path to CSV file.
    
    Returns:
        DataFrame or None if file not found.
    """
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath, parse_dates=['date', 'expiry'])
    required = ['date', 'expiry', 'strike', 'option_type', 'price', 'spot']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    return df
