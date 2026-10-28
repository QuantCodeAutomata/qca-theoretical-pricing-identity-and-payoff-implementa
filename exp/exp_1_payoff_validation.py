"""Experiment 1: Theoretical pricing identity and payoff implementation.

Validates AMM LP payoff formulas and the no-arbitrage pricing identity:
    E_t^QY[e^{-r^Y(T-t)} Gamma_{t,T}^R] = g_R(S_t) - E_t^QY[e^{-r^Y(T-t)} g_R(S_T)]

Produces:
    - results/exp1_payoff_validation.csv: Full validation table
    - results/exp1_payoff_shapes.png: Payoff shape plots
    - results/exp1_fee_metrics.csv: Fee metrics across maturities and ranges
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.amm_payoffs import (
    g_v2, g_v3_compact, g_v3_piecewise, g_v3,
    verify_v3_equivalence, check_shape_properties,
)
from src.fee_metrics import fee_value, fee_return, fee_rate, validate_bounds
from data.data_loader import (
    synthetic_price_grid, synthetic_maturity_grid, mock_lognormal_terminal_payoff
)

os.makedirs('results', exist_ok=True)


# ─── Parameters ───────────────────────────────────────────────────────────────
S_T = 2000.0      # Reference spot price (ETH/USDC)
SIGMA = 0.80      # Lognormal vol for mock pricing engine
RANGES = {
    'wide (0.7S, 1.4S)':   (0.7 * S_T, 1.4 * S_T),
    'narrow (0.8S, 1.2S)': (0.8 * S_T, 1.2 * S_T),
}
THETA_GRID = synthetic_maturity_grid(0.05, 1.0, 20)


def run_payoff_shape_validation() -> pd.DataFrame:
    """Validate compact vs piecewise v3 equivalence and shape properties."""
    records = []
    for label, (p_a, p_b) in RANGES.items():
        P_grid = synthetic_price_grid(S_T, n_points=1000)

        # 1. Equivalence check
        equiv = verify_v3_equivalence(P_grid, p_a, p_b, tol=1e-10)
        g_compact = g_v3_compact(P_grid, p_a, p_b)
        g_piece = g_v3_piecewise(P_grid, p_a, p_b)
        max_abs_err = float(np.max(np.abs(g_compact - g_piece)))

        # 2. Shape properties (compact form)
        props = check_shape_properties(P_grid, g_compact)

        # 3. v2 payoff check at S_T: should equal 2*sqrt(S_T)
        v2_val = float(g_v2(S_T))
        expected_v2 = 2.0 * np.sqrt(S_T)
        v2_exact = np.isclose(v2_val, expected_v2, atol=1e-12)

        # 4. Edge cases: P = p_a and P = p_b (both formulas must be consistent)
        g_c_pa = float(g_v3_compact(p_a, p_a, p_b))
        g_p_pa = float(g_v3_piecewise(p_a, p_a, p_b))
        g_c_pb = float(g_v3_compact(p_b, p_a, p_b))
        g_p_pb = float(g_v3_piecewise(p_b, p_a, p_b))

        records.append({
            'range': label,
            'p_a': p_a,
            'p_b': p_b,
            'compact_piecewise_equiv': equiv,
            'max_abs_error': max_abs_err,
            'positivity': props['positivity'],
            'monotone': props['monotone'],
            'concave': props['concave'],
            'v2_exact': v2_exact,
            'g_v2(S_T)': v2_val,
            'g_compact_at_pa': g_c_pa,
            'g_piecewise_at_pa': g_p_pa,
            'g_compact_at_pb': g_c_pb,
            'g_piecewise_at_pb': g_p_pb,
        })

    df = pd.DataFrame(records)
    print('\n=== Payoff Shape Validation ===')
    print(df.to_string(index=False))
    return df


def run_pricing_identity_validation() -> pd.DataFrame:
    """Validate the pricing identity using mock lognormal pricing engine."""
    records = []

    for label, (p_a, p_b) in RANGES.items():
        g0_v3 = float(g_v3(S_T, p_a, p_b))
        g0_v2 = float(g_v2(S_T))

        for theta in THETA_GRID:
            # Mock pricing engine: lognormal MC
            payoff_fn_v3 = lambda P: g_v3(P, p_a, p_b)  # noqa: E731
            H_v3 = mock_lognormal_terminal_payoff(
                S_T, theta, SIGMA, payoff_fn_v3, r=0.0, n_samples=100_000
            )
            H_v2 = mock_lognormal_terminal_payoff(
                S_T, theta, SIGMA, g_v2, r=0.0, n_samples=100_000
            )

            # Fee metrics
            fv_v3 = float(fee_value(g0_v3, H_v3))
            R_v3 = float(fee_return(g0_v3, H_v3))
            y_v3 = float(fee_rate(max(R_v3, 1e-12), theta))

            fv_v2 = float(fee_value(g0_v2, H_v2))
            R_v2 = float(fee_return(g0_v2, H_v2))
            y_v2 = float(fee_rate(max(R_v2, 1e-12), theta))

            # Bounds validation
            bounds_v3 = validate_bounds(g0_v3, H_v3)
            bounds_v2 = validate_bounds(g0_v2, H_v2)

            records.append({
                'range': label,
                'version': 'v3',
                'p_a': p_a,
                'p_b': p_b,
                'theta': theta,
                'g0': g0_v3,
                'H': H_v3,
                'FeeValue': fv_v3,
                'R': R_v3,
                'y': y_v3,
                'bounds_ok': bounds_v3['fee_value_nonneg'] and bounds_v3['return_less_one'],
            })
            records.append({
                'range': label,
                'version': 'v2',
                'p_a': 0.0,
                'p_b': np.inf,
                'theta': theta,
                'g0': g0_v2,
                'H': H_v2,
                'FeeValue': fv_v2,
                'R': R_v2,
                'y': y_v2,
                'bounds_ok': bounds_v2['fee_value_nonneg'] and bounds_v2['return_less_one'],
            })

    df = pd.DataFrame(records)
    print('\n=== Pricing Identity Validation (sample) ===')
    print(df.head(10).to_string(index=False))
    return df


def plot_payoff_shapes():
    """Plot payoff functions g_R(P) for v2 and v3 ranges."""
    P_grid = synthetic_price_grid(S_T, n_points=500)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: payoff functions
    ax = axes[0]
    ax.plot(P_grid, g_v2(P_grid), 'k-', lw=2, label=r'Uniswap v2: $2\sqrt{P}$')
    colors = ['#1f77b4', '#ff7f0e']
    for (label, (p_a, p_b)), c in zip(RANGES.items(), colors):
        g = g_v3(P_grid, p_a, p_b)
        ax.plot(P_grid, g, color=c, lw=1.5, label=f'v3 {label}')
        ax.axvline(p_a, color=c, ls='--', alpha=0.4, lw=0.8)
        ax.axvline(p_b, color=c, ls='--', alpha=0.4, lw=0.8)
    ax.axvline(S_T, color='gray', ls=':', lw=1, label=f'$S_t={S_T}$')
    ax.set_xlabel('Price $P$ (USDC)')
    ax.set_ylabel('Payoff $g_R(P)$')
    ax.set_title('AMM LP Payoff Functions')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: fee-rate term structure
    ax = axes[1]
    sigma = SIGMA
    for (label, (p_a, p_b)), c in zip(RANGES.items(), colors):
        g0 = float(g_v3(S_T, p_a, p_b))
        ys = []
        for theta in THETA_GRID:
            H = mock_lognormal_terminal_payoff(
                S_T, theta, sigma, lambda P: g_v3(P, p_a, p_b), r=0.0,
                n_samples=50_000, seed=0
            )
            R = float(fee_return(g0, H))
            ys.append(float(fee_rate(max(R, 1e-12), theta)))
        ax.plot(THETA_GRID, ys, color=c, lw=1.5, label=f'v3 {label}')

    # v2
    g0_v2 = float(g_v2(S_T))
    ys_v2 = []
    for theta in THETA_GRID:
        H = mock_lognormal_terminal_payoff(
            S_T, theta, sigma, g_v2, r=0.0, n_samples=50_000, seed=0
        )
        R = float(fee_return(g0_v2, H))
        ys_v2.append(float(fee_rate(max(R, 1e-12), theta)))
    ax.plot(THETA_GRID, ys_v2, 'k-', lw=2, label='v2 (infinite range)')

    ax.set_xlabel('Maturity $\\theta$ (years)')
    ax.set_ylabel('Implied fee rate $y(t,\\theta)$')
    ax.set_title('Implied Fee-Rate Term Structure (Mock Lognormal)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/exp1_payoff_shapes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved results/exp1_payoff_shapes.png')


def main():
    print('=' * 60)
    print('Experiment 1: Theoretical Payoff Validation')
    print('=' * 60)

    df_shapes = run_payoff_shape_validation()
    df_shapes.to_csv('results/exp1_payoff_validation.csv', index=False)
    print('Saved results/exp1_payoff_validation.csv')

    df_fee = run_pricing_identity_validation()
    df_fee.to_csv('results/exp1_fee_metrics.csv', index=False)
    print('Saved results/exp1_fee_metrics.csv')

    plot_payoff_shapes()

    # Summary assertions
    assert df_shapes['compact_piecewise_equiv'].all(), \
        'Compact and piecewise v3 formulas must agree!'
    assert df_shapes['max_abs_error'].max() < 1e-10, \
        'Max absolute error between formulas exceeds 1e-10!'
    assert df_shapes['positivity'].all(), 'Payoff positivity violated!'
    assert df_shapes['monotone'].all(), 'Payoff monotonicity violated!'
    assert df_shapes['concave'].all(), 'Payoff concavity violated!'
    assert df_fee['bounds_ok'].all(), 'Fee return bounds violated!'

    print('\n[PASS] All Experiment 1 assertions passed.')
    return df_shapes, df_fee


if __name__ == '__main__':
    main()
