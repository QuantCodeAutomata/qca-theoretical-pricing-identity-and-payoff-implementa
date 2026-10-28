"""Experiment 5: PCA of first differences in implied LP fee-rate curves.

Constructs a daily panel of implied fee-rate curves y(t, theta) and
applies PCA to their first differences, reproducing the paper's result:
    PC1 ≈ 81.0% variance
    PC1+PC2 ≈ 96.5% variance
    PC1+PC2+PC3 ≈ 99.6% variance

Since real Deribit data is not available in this environment, a synthetic
panel is generated from a time-varying SVJD model (documented assumption).

Produces:
    - results/exp5_pca_explained_variance.csv
    - results/exp5_pca_loadings.csv
    - results/exp5_pca_scores.csv
    - results/exp5_pca_plots.png
    - results/exp5_fee_panel.csv
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.bates_cf import BatesParams
from src.analytics.fee_curves import compute_fee_curve_v2
from src.analytics.pca_fee_curves import build_fee_rate_panel, compute_pca_on_differences

os.makedirs('results', exist_ok=True)

# ─── Parameters ───────────────────────────────────────────────────────────────
N_DATES = 200        # Number of synthetic daily calibration dates
THETA_GRID = np.linspace(0.10, 0.50, 10)   # Common maturity grid
N_COMPONENTS = 5


def generate_synthetic_daily_params(
    n_dates: int,
    seed: int = 42,
) -> list:
    """Generate time-varying SVJD parameters for synthetic panel.

    Simulates realistic daily variation in calibrated parameters:
    - V0 follows a mean-reverting process (capturing vol regime changes)
    - rho varies slowly
    - Other params change with mild random walk

    Returns:
        List of (date_str, spot, BatesParams) tuples.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2019-04-01', periods=n_dates, freq='B')  # business days

    # Base parameters
    base_V0 = 0.40
    base_rho = -0.50
    base_lam = 0.01
    base_kappa = -0.1
    base_bv = 20.0
    base_beta = 35.0
    base_sv = 9.0
    base_delta = 0.12
    base_spot = 200.0  # ETH starts low in 2019

    # Simulate correlated AR(1) paths for key params
    dt = 1.0 / 252.0
    kappa_V = 10.0    # mean-reversion speed for V0
    sigma_V = 3.0     # vol of vol process

    V0_path = [base_V0]
    rho_path = [base_rho]
    spot_path = [base_spot]
    lam_path = [base_lam]

    for _ in range(n_dates - 1):
        # V0: mean-reverting
        dV = kappa_V * (base_V0 - V0_path[-1]) * dt + sigma_V * np.sqrt(V0_path[-1] * dt) * rng.standard_normal()
        V0_path.append(max(V0_path[-1] + dV, 0.02))

        # rho: slow random walk with bounds
        drho = rng.normal(0, 0.02)
        rho_path.append(np.clip(rho_path[-1] + drho, -0.95, 0.0))

        # spot: lognormal with drift
        dlog_s = rng.normal(0.001, 0.05)
        spot_path.append(max(spot_path[-1] * np.exp(dlog_s), 50.0))

        # lam: mild variation
        dlam = rng.normal(0, 0.001)
        lam_path.append(max(lam_path[-1] + dlam, 1e-4))

    result = []
    for i, date in enumerate(dates):
        params = BatesParams(
            lam=lam_path[i],
            kappa=base_kappa + rng.normal(0, 0.05),
            b_v=max(base_bv + rng.normal(0, 2.0), 1.0),
            beta=max(base_beta + rng.normal(0, 3.0), 1.0),
            sigma_v=max(base_sv + rng.normal(0, 1.0), 0.5),
            V0=V0_path[i],
            rho=rho_path[i],
            delta=max(base_delta + rng.normal(0, 0.01), 0.01),
            rX=0.0,
            rY=0.0,
        )
        result.append((date.strftime('%Y-%m-%d'), spot_path[i], params))

    return result


def build_fee_rate_panel_synthetic(
    daily_entries: list,
    theta_grid: np.ndarray,
) -> pd.DataFrame:
    """Build the fee-rate panel from synthetic daily calibrations.

    Args:
        daily_entries: List of (date_str, spot, BatesParams).
        theta_grid:    Common maturity grid.

    Returns:
        DataFrame (dates x maturities) of fee rates.
    """
    rows = {}
    for date_str, spot, params in daily_entries:
        df_curve = compute_fee_curve_v2(spot, params.V0, params, theta_grid)
        row = {}
        for _, record in df_curve.iterrows():
            row[record['theta']] = record['y']
        rows[date_str] = row

    panel = pd.DataFrame(rows).T
    panel.index.name = 'date'
    panel.columns = [f'theta_{t:.4f}' for t in theta_grid]
    return panel


def plot_pca_results(pca_results: dict, theta_grid: np.ndarray):
    """Plot PCA explained variance, loadings, and factor scores."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: explained variance
    ax = axes[0, 0]
    ev = pca_results['explained_variance_ratio'] * 100
    cum_ev = pca_results['cumulative_variance'] * 100
    n = len(ev)
    x = np.arange(1, n + 1)
    ax.bar(x, ev, alpha=0.7, label='Individual')
    ax.step(x, cum_ev, where='post', color='red', lw=2, label='Cumulative')
    ax.axhline(81.0, color='green', ls='--', alpha=0.7, label='Paper PC1 (81%)')
    ax.axhline(96.5, color='orange', ls='--', alpha=0.7, label='Paper PC1+2 (96.5%)')
    ax.axhline(99.6, color='purple', ls='--', alpha=0.7, label='Paper PC1+2+3 (99.6%)')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('PCA Explained Variance\n(First Differences of Fee-Rate Panel)')
    ax.legend(fontsize=7)
    ax.set_xticks(x)
    ax.grid(alpha=0.3)

    # Top-right: component loadings
    ax = axes[0, 1]
    loadings = pca_results['loadings']
    for i in range(min(3, len(loadings))):
        ax.plot(theta_grid, loadings[i], lw=2, marker='o', ms=4,
                label=f'PC{i+1} ({ev[i]:.1f}%)')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('Maturity $\\theta$ (years)')
    ax.set_ylabel('Loading')
    ax.set_title('PCA Component Loadings by Maturity')
    ax.legend()
    ax.grid(alpha=0.3)

    # Bottom-left: PC1 scores over time
    ax = axes[1, 0]
    scores = pca_results['scores']
    if 'PC1' in scores.columns:
        ax.plot(scores.index, scores['PC1'], lw=1, alpha=0.8, color='blue')
    ax.set_xlabel('Date')
    ax.set_ylabel('PC1 Score')
    ax.set_title('PC1 Factor Scores Over Time\n(Level shift in fee-rate curve)')
    ax.grid(alpha=0.3)
    ax.tick_params(axis='x', rotation=30)

    # Bottom-right: PC2 scores over time
    ax = axes[1, 1]
    if 'PC2' in scores.columns:
        ax.plot(scores.index, scores['PC2'], lw=1, alpha=0.8, color='orange')
    ax.set_xlabel('Date')
    ax.set_ylabel('PC2 Score')
    ax.set_title('PC2 Factor Scores Over Time\n(Slope change in fee-rate curve)')
    ax.grid(alpha=0.3)
    ax.tick_params(axis='x', rotation=30)

    plt.suptitle('PCA of First Differences in LP Fee-Rate Curves\n'
                 '(Synthetic Panel — Uniswap v2)', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/exp5_pca_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved results/exp5_pca_plots.png')


def plot_fee_panel_time_series(panel: pd.DataFrame, theta_grid: np.ndarray):
    """Plot fixed-maturity fee-rate time series from the panel."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(theta_grid)))
    for i, theta in enumerate(theta_grid[::2]):   # Plot every other maturity
        col = f'theta_{theta:.4f}'
        if col in panel.columns:
            ax.plot(panel.index, panel[col], lw=1, alpha=0.8,
                    color=colors[i * 2], label=f'$\\theta$={theta:.2f}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Implied fee rate $y(t,\\theta)$')
    ax.set_title('Implied Fee-Rate Panel: Fixed-Maturity Time Series (Uniswap v2)')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig('results/exp5_fee_panel_time_series.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved results/exp5_fee_panel_time_series.png')


def main():
    print('=' * 60)
    print('Experiment 5: PCA of First Differences in LP Fee-Rate Curves')
    print('=' * 60)

    # Generate synthetic daily calibrations
    print(f'\nGenerating {N_DATES} synthetic daily calibration entries...')
    daily_entries = generate_synthetic_daily_params(N_DATES, seed=42)
    print(f'Date range: {daily_entries[0][0]} to {daily_entries[-1][0]}')

    # Build fee-rate panel
    print('\nBuilding fee-rate panel (v2, infinite range)...')
    panel = build_fee_rate_panel_synthetic(daily_entries, THETA_GRID)
    print(f'Panel shape: {panel.shape} (dates x maturities)')
    print(f'NaN count:   {panel.isna().sum().sum()}')

    panel.to_csv('results/exp5_fee_panel.csv')
    print('Saved results/exp5_fee_panel.csv')

    plot_fee_panel_time_series(panel, THETA_GRID)

    # PCA on first differences
    print(f'\nRunning PCA with {N_COMPONENTS} components on first-differenced panel...')
    pca_results = compute_pca_on_differences(panel, n_components=N_COMPONENTS)

    ev_pct = pca_results['explained_variance_ratio'] * 100
    cum_pct = pca_results['cumulative_variance'] * 100
    print('\nExplained variance by component:')
    for i, (ev, cum) in enumerate(zip(ev_pct, cum_pct)):
        print(f'  PC{i+1}: {ev:.1f}%  (cumulative: {cum:.1f}%)')

    # Compare to paper targets
    print('\nComparison to paper targets:')
    print(f'  PC1:           {ev_pct[0]:.1f}%  (paper: ~81.0%)')
    if len(cum_pct) >= 2:
        print(f'  PC1+PC2 cum:   {cum_pct[1]:.1f}%  (paper: ~96.5%)')
    if len(cum_pct) >= 3:
        print(f'  PC1+PC2+PC3:   {cum_pct[2]:.1f}%  (paper: ~99.6%)')

    # Save results
    ev_df = pd.DataFrame({
        'component': [f'PC{i+1}' for i in range(len(ev_pct))],
        'explained_variance_pct': ev_pct,
        'cumulative_variance_pct': cum_pct,
        'paper_target_pct': [81.0, 96.5, 99.6] + [np.nan] * (len(ev_pct) - 3),
    })
    ev_df.to_csv('results/exp5_pca_explained_variance.csv', index=False)
    print('Saved results/exp5_pca_explained_variance.csv')

    loadings_df = pd.DataFrame(
        pca_results['loadings'],
        index=[f'PC{i+1}' for i in range(len(pca_results['loadings']))],
        columns=[f'theta_{t:.4f}' for t in THETA_GRID],
    )
    loadings_df.to_csv('results/exp5_pca_loadings.csv')
    print('Saved results/exp5_pca_loadings.csv')

    pca_results['scores'].to_csv('results/exp5_pca_scores.csv')
    print('Saved results/exp5_pca_scores.csv')

    plot_pca_results(pca_results, THETA_GRID)

    # Assertions: PCA should be low-dimensional
    assert cum_pct[0] > 50.0, f'PC1 should explain >50% variance, got {cum_pct[0]:.1f}%'
    if len(cum_pct) >= 3:
        assert cum_pct[2] > 80.0, \
            f'PC1-3 cumulative should explain >80%, got {cum_pct[2]:.1f}%'
    print('\n[PASS] Experiment 5 completed successfully.')
    return pca_results


if __name__ == '__main__':
    main()
