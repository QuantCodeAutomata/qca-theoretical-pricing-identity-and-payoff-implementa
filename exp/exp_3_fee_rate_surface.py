"""Experiment 3: Uniswap v3 fee-rate surface across maturity and moneyness.

Constructs the fee-rate surface y(t, theta) as a function of:
    - maturity theta (x-axis)
    - moneyness M = p_b / S_0 (y-axis)

Using tick-based unitary ranges from the paper's construction.

Date fixed at 2022-03-01 (uses paper example params as stand-in when
real Deribit data is unavailable).

Produces:
    - results/exp3_fee_surface.csv
    - results/exp3_fee_surface_3d.png
    - results/exp3_fee_surface_heatmap.png
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.models.bates_cf import BatesParams
from src.amm_payoffs import g_v3
from src.fee_metrics import fee_return, fee_rate
from src.pricing.lp_payoff_pricer import price_v3_lp
from src.uniswap_v3_ranges import (
    build_unitary_range, FEE_TIER_TICK_SPACING, moneyness_grid,
    price_to_tick, tick_to_price,
)

os.makedirs('results', exist_ok=True)

# ─── Parameters ───────────────────────────────────────────────────────────────
ANALYSIS_DATE = '2022-03-01'
S_0 = 2800.0   # ETH/USDC spot price circa 2022-03-01

# Use paper calibration params as stand-in for 2022-03-01 calibration
# (Real data would require Deribit historical options for this date)
PARAMS_2022_03_01 = BatesParams(
    lam=0.01, kappa=-0.05, b_v=15.0, beta=30.0,
    sigma_v=8.0, V0=0.25, rho=-0.5, delta=0.12,
    rX=0.0, rY=0.0,
)

THETA_GRID = np.linspace(0.10, 0.50, 15)
M_GRID = np.linspace(1.02, 1.50, 20)   # M = p_b / S_0

# Fee tiers to analyze
FEE_TIERS = {
    '0.3% (tick spacing 60)': 60,
    '1.0% (tick spacing 200)': 200,
}


def compute_surface_for_tier(
    params: BatesParams,
    S: float,
    theta_grid: np.ndarray,
    M_grid: np.ndarray,
    tick_spacing: int,
    tier_label: str,
) -> pd.DataFrame:
    """Compute fee-rate surface for one fee tier.

    For each moneyness M, snaps p_b = M * S to the nearest usable tick,
    sets p_a to the adjacent lower usable tick (unitary range).

    Args:
        params:       Calibrated BatesParams.
        S:            Spot price.
        theta_grid:   Maturity grid in years.
        M_grid:       Moneyness values M = p_b_target / S.
        tick_spacing: Tick spacing for the fee tier.
        tier_label:   Fee tier label.

    Returns:
        DataFrame with columns [M, p_a, p_b, theta, g0, H, R, y, fee_tier].
    """
    rows = []
    p_b_targets = M_grid * S

    for p_b_target in p_b_targets:
        # Snap to usable tick boundaries
        p_a, p_b = build_unitary_range(p_b_target, S, tick_spacing)
        if p_a <= 0 or p_b <= p_a:
            continue

        M_actual = p_b / S
        g0 = float(g_v3(S, p_a, p_b))

        # Skip if g0 is too small (numerical issues with very narrow ranges)
        if g0 < 1e-6:
            continue

        for theta in theta_grid:
            try:
                H = price_v3_lp(
                    theta, S, params.V0, p_a, p_b, params,
                    epsabs=1e-6, epsrel=1e-6,
                )
                R = float(fee_return(g0, H))
                # Clamp R to valid range
                R = max(min(R, 1.0 - 1e-10), 0.0)
                y = float(fee_rate(max(R, 1e-12), theta))
                rows.append({
                    'M': M_actual,
                    'p_a': p_a,
                    'p_b': p_b,
                    'theta': theta,
                    'g0': g0,
                    'H': H,
                    'R': R,
                    'y': y,
                    'fee_tier': tier_label,
                    'tick_spacing': tick_spacing,
                })
            except Exception as e:
                # Record failure for diagnostics
                rows.append({
                    'M': M_actual,
                    'p_a': p_a,
                    'p_b': p_b,
                    'theta': theta,
                    'g0': g0,
                    'H': np.nan,
                    'R': np.nan,
                    'y': np.nan,
                    'fee_tier': tier_label,
                    'tick_spacing': tick_spacing,
                })

    return pd.DataFrame(rows)


def plot_surface_3d(df: pd.DataFrame, tier_label: str):
    """Plot 3D fee-rate surface over moneyness and maturity."""
    df_clean = df.dropna(subset=['y'])
    if df_clean.empty:
        print(f'No valid data for {tier_label}')
        return

    fig = plt.figure(figsize=(14, 5))

    for idx, (title, col) in enumerate([('Fee Rate $y(t,\\theta)$', 'y'),
                                         ('Fee Return $R$', 'R')]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        pivot = df_clean.pivot_table(index='M', columns='theta', values=col, aggfunc='mean')
        M_vals = pivot.index.values
        T_vals = pivot.columns.values
        M_mesh, T_mesh = np.meshgrid(M_vals, T_vals)
        Z = pivot.values.T

        surf = ax.plot_surface(T_mesh, M_mesh, Z, cmap='viridis', alpha=0.85)
        ax.set_xlabel('Maturity $\\theta$')
        ax.set_ylabel('Moneyness $M = p_b/S_0$')
        ax.set_zlabel(col)
        ax.set_title(f'{title}\n{tier_label}')
        fig.colorbar(surf, ax=ax, shrink=0.5)

    plt.suptitle(f'Uniswap v3 Fee-Rate Surface — {ANALYSIS_DATE}', fontsize=12)
    plt.tight_layout()
    fname = f'results/exp3_fee_surface_3d_{tier_label.split()[0].replace(".", "")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_heatmap(df: pd.DataFrame, tier_label: str):
    """Plot fee-rate heatmap (moneyness vs maturity)."""
    df_clean = df.dropna(subset=['y'])
    if df_clean.empty:
        return

    pivot = df_clean.pivot_table(index='M', columns='theta', values='y', aggfunc='mean')

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        pivot.values,
        aspect='auto',
        origin='lower',
        cmap='plasma',
        extent=[pivot.columns.min(), pivot.columns.max(),
                pivot.index.min(), pivot.index.max()],
    )
    plt.colorbar(im, ax=ax, label='Implied fee rate $y(t,\\theta)$')
    ax.set_xlabel('Maturity $\\theta$ (years)')
    ax.set_ylabel('Moneyness $M = p_b/S_0$')
    ax.set_title(f'Uniswap v3 Implied Fee-Rate Surface Heatmap\n{tier_label} — {ANALYSIS_DATE}')

    plt.tight_layout()
    fname = f'results/exp3_fee_surface_heatmap_{tier_label.split()[0].replace(".", "")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def main():
    print('=' * 60)
    print('Experiment 3: Uniswap v3 Fee-Rate Surface')
    print(f'Analysis date: {ANALYSIS_DATE}')
    print(f'Spot S_0 = {S_0}')
    print('=' * 60)

    all_dfs = []
    for tier_label, tick_spacing in FEE_TIERS.items():
        print(f'\nComputing surface for {tier_label}...')
        df = compute_surface_for_tier(
            PARAMS_2022_03_01, S_0, THETA_GRID, M_GRID, tick_spacing, tier_label
        )
        print(f'  Computed {len(df)} surface nodes '
              f'({df["y"].notna().sum()} valid)')
        all_dfs.append(df)
        plot_surface_3d(df, tier_label)
        plot_heatmap(df, tier_label)

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv('results/exp3_fee_surface.csv', index=False)
    print('\nSaved results/exp3_fee_surface.csv')

    # --- Qualitative checks ---
    for tier_label in FEE_TIERS:
        df_tier = df_all[df_all['fee_tier'] == tier_label].dropna(subset=['y'])
        if df_tier.empty:
            continue

        # Near-spot ranges (M close to 1) should have higher fee rates
        near_spot = df_tier[df_tier['M'] < 1.10]['y'].mean()
        far_spot = df_tier[df_tier['M'] > 1.30]['y'].mean()
        print(f'\n{tier_label}:')
        print(f'  Mean fee rate (near-spot M<1.10): {near_spot:.4f}')
        print(f'  Mean fee rate (far-spot M>1.30):  {far_spot:.4f}')
        print(f'  Near > Far: {near_spot > far_spot}')

    print('\n[PASS] Experiment 3 completed successfully.')
    return df_all


if __name__ == '__main__':
    main()
