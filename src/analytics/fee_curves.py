"""Fee-rate curve analytics for AMM LP positions.

Builds fee-rate term structures from calibrated Bates/SVJD model parameters.
Custom — Context7 found no library equivalent for AMM fee-rate curve generation.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.amm_payoffs import g_v2, g_v3
from src.fee_metrics import fee_return, fee_rate
from src.models.bates_cf import BatesParams
from src.pricing.lp_payoff_pricer import price_v2_lp, price_v3_lp


def compute_fee_curve_v2(
    S: float,
    V: float,
    params: BatesParams,
    theta_grid: np.ndarray,
) -> pd.DataFrame:
    """Compute v2 fee-rate term structure for given model parameters.
    
    Args:
        S:          Spot price at valuation date.
        V:          Variance V_t at valuation date.
        params:     Calibrated BatesParams.
        theta_grid: Array of maturities in years.
    
    Returns:
        DataFrame with columns [theta, g0, H, FeeValue, R, y].
    """
    g0 = float(g_v2(S))
    rows = []
    for theta in theta_grid:
        H = price_v2_lp(theta, S, V, params)
        R = float(fee_return(g0, H))
        y = float(fee_rate(max(R, 1e-12), theta))
        rows.append({'theta': theta, 'g0': g0, 'H': H, 'FeeValue': g0 - H, 'R': R, 'y': y})
    return pd.DataFrame(rows)


def compute_fee_curve_v3(
    S: float,
    V: float,
    params: BatesParams,
    theta_grid: np.ndarray,
    p_a: float,
    p_b: float,
) -> pd.DataFrame:
    """Compute v3 fee-rate term structure for a given range.
    
    Args:
        S:          Spot price at valuation date.
        V:          Variance V_t at valuation date.
        params:     Calibrated BatesParams.
        theta_grid: Array of maturities in years.
        p_a:        Lower range boundary.
        p_b:        Upper range boundary.
    
    Returns:
        DataFrame with columns [theta, g0, H, FeeValue, R, y, p_a, p_b].
    """
    g0 = float(g_v3(S, p_a, p_b))
    rows = []
    for theta in theta_grid:
        H = price_v3_lp(theta, S, V, p_a, p_b, params)
        R = float(fee_return(g0, H))
        y = float(fee_rate(max(R, 1e-12), theta))
        rows.append({'theta': theta, 'g0': g0, 'H': H, 'FeeValue': g0 - H,
                     'R': R, 'y': y, 'p_a': p_a, 'p_b': p_b})
    return pd.DataFrame(rows)


def compute_fee_surface_v3(
    S: float,
    V: float,
    params: BatesParams,
    theta_grid: np.ndarray,
    range_configs: List[Tuple[float, float]],
    range_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute v3 fee-rate surface across multiple ranges and maturities.
    
    Args:
        S:             Spot price.
        V:             Variance.
        params:        BatesParams.
        theta_grid:    Maturity grid.
        range_configs: List of (p_a, p_b) tuples.
        range_labels:  Optional list of range labels.
    
    Returns:
        DataFrame with columns [theta, g0, H, FeeValue, R, y, p_a, p_b, range_label].
    """
    if range_labels is None:
        range_labels = [f'[{pa:.2f},{pb:.2f}]' for pa, pb in range_configs]
    
    dfs = []
    for (p_a, p_b), label in zip(range_configs, range_labels):
        df = compute_fee_curve_v3(S, V, params, theta_grid, p_a, p_b)
        df['range_label'] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
