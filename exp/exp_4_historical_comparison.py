"""Experiment 4: Historical comparison of model-implied and realized v2 fee returns.

Compares model-implied accumulated fee returns from the SVJD pricing model
against realized fee accrual in Uniswap v2 ETH-USDC positions.

Two backtest windows:
    Window 1: 2022-12-13 to 2023-03-13
    Window 2: 2023-03-14 to 2023-05-17

Implementation notes:
    - Real on-chain data from The Graph/Subgraph or web3.py is required for
      exact replication. Since this data is not available in this environment,
      a synthetic realized-fee simulation is provided with documented assumptions.
    - Theoretical fee returns are computed from SVJD model calibrated at window start.
    - Synthetic realized fees are generated from a lognormal volume process
      (documented as mock data, not real chain data).

All assumptions are logged in the output.

Produces:
    - results/exp4_comparison_window1.png
    - results/exp4_comparison_window2.png
    - results/exp4_comparison_metrics.csv
    - results/exp4_assumptions.txt
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
from src.amm_payoffs import g_v2
from src.fee_metrics import fee_return, fee_rate
from src.pricing.lp_payoff_pricer import price_v2_lp

os.makedirs('results', exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 100_000.0   # USDC
V2_FEE_RATE = 0.003           # Uniswap v2 fee rate: 0.3%

# Backtest windows (start, end)
WINDOWS = [
    ('2022-12-13', '2023-03-13', 'Window 1'),
    ('2023-03-14', '2023-05-17', 'Window 2'),
]

# SVJD params calibrated at start of each window (documented as assumptions)
# Real calibration would use Deribit data for 2022-12-13 and 2023-03-14
WINDOW_PARAMS = {
    'Window 1': {
        'spot': 1250.0,   # ETH/USDC circa 2022-12-13
        'params': BatesParams(
            lam=0.02, kappa=-0.1, b_v=30.0, beta=40.0,
            sigma_v=10.0, V0=0.60, rho=-0.55, delta=0.15,
            rX=0.0, rY=0.0,
        ),
    },
    'Window 2': {
        'spot': 1810.0,   # ETH/USDC circa 2023-03-14
        'params': BatesParams(
            lam=0.01, kappa=-0.05, b_v=20.0, beta=35.0,
            sigma_v=9.0, V0=0.35, rho=-0.45, delta=0.12,
            rX=0.0, rY=0.0,
        ),
    },
}


def compute_theoretical_fee_curve(
    params: BatesParams,
    S_0: float,
    theta_grid: np.ndarray,
) -> pd.DataFrame:
    """Compute theoretical implied accumulated fee returns across horizons.

    Uses: FeeValue(T) = g_v2(S_0) - H_{0,T}
          R_theory(T) = FeeValue / g_v2(S_0)

    Args:
        params:     Calibrated BatesParams at window start.
        S_0:        Spot price at window start.
        theta_grid: Grid of horizons in years.

    Returns:
        DataFrame with [theta, g0, H, R_theory, y_theory].
    """
    g0 = float(g_v2(S_0))
    rows = []
    for theta in theta_grid:
        H = price_v2_lp(theta, S_0, params.V0, params)
        R = float(fee_return(g0, H))
        y = float(fee_rate(max(R, 1e-12), theta))
        rows.append({'theta': theta, 'g0': g0, 'H': H, 'R_theory': R, 'y_theory': y})
    return pd.DataFrame(rows)


def generate_synthetic_realized_fees(
    S_0: float,
    start_date: str,
    end_date: str,
    initial_capital: float,
    fee_rate: float,
    sigma_spot: float = 0.80,
    avg_daily_volume_ratio: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic realized fee accrual for a Uniswap v2 LP position.

    Implementation assumption:
        This is a MOCK simulation, not real on-chain data.
        Real replication requires Uniswap v2 swap event data from The Graph.

    Convention (matching paper's constant-liquidity, no-reinvestment setup):
        - LP's ownership fraction is constant (set at t_0 from initial capital).
        - For each day, LP earns: ownership_frac * daily_trading_volume * fee_rate.
        - No reinvestment of earned fees.
        - Values in USDC.

    Args:
        S_0:                   Spot price at t_0.
        start_date:            Window start (ISO format).
        end_date:              Window end (ISO format).
        initial_capital:       Initial capital in USDC.
        fee_rate:              Pool fee rate (0.003 for Uniswap v2).
        sigma_spot:            Spot price annual volatility (for volume simulation).
        avg_daily_volume_ratio: Daily volume as fraction of pool TVL.
        seed:                  Random seed.

    Returns:
        DataFrame with [date, spot, daily_volume, fee_earned, cum_fee, R_realized].
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)

    # Simulate spot price path
    dt = 1.0 / 252.0
    log_returns = rng.normal(-0.5 * sigma_spot ** 2 * dt, sigma_spot * np.sqrt(dt), n_days - 1)
    spots = S_0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))

    # Estimate pool TVL from spot (2 * sqrt(spot) * liquidity scale)
    # Ownership fraction = initial_capital / initial_pool_tvl
    initial_pool_tvl = initial_capital / 0.10  # Assume LP owns ~10% of pool (documented)
    ownership_fraction = initial_capital / initial_pool_tvl

    # Simulate daily trading volume as fraction of TVL with log-normal noise
    vol_scale = rng.lognormal(
        mean=np.log(avg_daily_volume_ratio * initial_pool_tvl),
        sigma=0.5,
        size=n_days,
    )

    daily_fees = ownership_fraction * vol_scale * fee_rate
    cum_fees = np.cumsum(daily_fees)

    # Normalize by initial pool value to get realized return
    R_realized = cum_fees / initial_capital

    df = pd.DataFrame({
        'date': dates,
        'spot': spots,
        'daily_volume': vol_scale,
        'fee_earned': daily_fees,
        'cum_fee': cum_fees,
        'R_realized': R_realized,
        'theta': np.arange(n_days) / 365.0,
    })
    return df


def compare_and_plot(
    window_label: str,
    start_date: str,
    end_date: str,
    df_theory: pd.DataFrame,
    df_realized: pd.DataFrame,
) -> dict:
    """Produce comparison plot and compute discrepancy metrics."""
    # Align on common theta axis
    theta_common = df_theory['theta'].values

    # Interpolate realized R onto theory's theta grid
    R_realized_interp = np.interp(
        theta_common, df_realized['theta'].values, df_realized['R_realized'].values
    )
    R_theory = df_theory['R_theory'].values

    # Metrics
    mad = float(np.mean(np.abs(R_theory - R_realized_interp)))
    end_diff = float(R_theory[-1] - R_realized_interp[-1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Accumulated fee return comparison
    ax = axes[0]
    ax.plot(theta_common * 365, R_theory * 100, 'b-', lw=2,
            label='Model-implied $R_{theory}(T)$')
    ax.plot(theta_common * 365, R_realized_interp * 100, 'r--', lw=2,
            label='Realized $R_{realized}(T)$ (mock)')
    ax.set_xlabel('Days since inception')
    ax.set_ylabel('Accumulated fee return (%)')
    ax.set_title(f'{window_label}\n{start_date} to {end_date}')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.text(0.05, 0.95, f'MAD = {mad * 100:.2f}%\nEnd diff = {end_diff * 100:.2f}%',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Right: Implied fee rate term structure
    ax = axes[1]
    ax.plot(theta_common, df_theory['y_theory'].values, 'b-', lw=2,
            label='Implied fee rate $y(t,\\theta)$')
    ax.set_xlabel('Maturity $\\theta$ (years)')
    ax.set_ylabel('Implied fee rate')
    ax.set_title(f'Fee-Rate Term Structure at {start_date}')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fname = f'results/exp4_comparison_{window_label.replace(" ", "").lower()}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')

    return {'window': window_label, 'MAD': mad, 'end_diff': end_diff,
            'R_theory_end': float(R_theory[-1]),
            'R_realized_end': float(R_realized_interp[-1])}


def save_assumptions():
    """Save implementation assumptions to text file."""
    assumptions = """# Experiment 4 Implementation Assumptions

## Data Sources
- Real Deribit option data for 2022-12-13 and 2023-03-14 NOT available.
  SVJD parameters are plausible approximations, not exact calibrations.
- Real Uniswap v2 on-chain swap event data NOT available.
  Realized fees are generated by a synthetic volume simulation (documented).

## Theoretical Fee Computation
- Model: SVJD/Bates under Q^Y with rX = rY = 0.
- LP payoff: g_v2(S) = 2*sqrt(S) (unit liquidity, infinite range).
- H_theory = 2 * G(1/2, S_0, V_0, theta) where G is the Bates CF.
- FeeValue = g_v2(S_0) - H_theory.
- R_theory = FeeValue / g_v2(S_0).

## Realized Fee Computation (Mock Simulation)
- Ownership fraction: initial_capital / pool_TVL, fixed at t_0.
- Daily fee: ownership_frac * daily_trading_volume * 0.003.
- No reinvestment: ownership fraction held constant.
- Daily volume: log-normal simulation with mean = 0.20 * pool_TVL.
- Pool TVL approximated from initial_capital / 0.10 (10% ownership assumption).

## Day-Count Convention
- ACT/365 for theta = (T - t_0) / 365.

## Backtest Windows
- Window 1: 2022-12-13 to 2023-03-13 (spot start ~$1,250)
- Window 2: 2023-03-14 to 2023-05-17 (spot start ~$1,810)
"""
    with open('results/exp4_assumptions.txt', 'w') as f:
        f.write(assumptions)
    print('Saved results/exp4_assumptions.txt')


def main():
    print('=' * 60)
    print('Experiment 4: Model-Implied vs Realized Uniswap v2 Fee Returns')
    print('=' * 60)

    save_assumptions()
    metrics_list = []

    for start_date, end_date, window_label in WINDOWS:
        print(f'\n--- {window_label}: {start_date} to {end_date} ---')

        config = WINDOW_PARAMS[window_label]
        S_0 = config['spot']
        params = config['params']

        # Theta grid: daily steps from t_0 to window end
        n_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
        theta_grid = np.linspace(1 / 365.0, n_days / 365.0, min(n_days, 90))

        # Theoretical fee curve
        print(f'  Spot at t_0: {S_0}')
        print(f'  Window length: {n_days} days')
        df_theory = compute_theoretical_fee_curve(params, S_0, theta_grid)

        # Synthetic realized fees (mock)
        df_realized = generate_synthetic_realized_fees(
            S_0, start_date, end_date, INITIAL_CAPITAL, V2_FEE_RATE, seed=42
        )

        print(f'  g_v2(S_0) = {df_theory["g0"].iloc[0]:.4f}')
        print(f'  R_theory (end)   = {df_theory["R_theory"].iloc[-1]:.4f}')
        print(f'  R_realized (end) = {df_realized["R_realized"].iloc[-1]:.4f}')

        metrics = compare_and_plot(window_label, start_date, end_date,
                                   df_theory, df_realized)
        metrics_list.append(metrics)

        print(f'  MAD: {metrics["MAD"]:.4f}')
        print(f'  End-of-window diff: {metrics["end_diff"]:.4f}')

    # Save metrics
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv('results/exp4_comparison_metrics.csv', index=False)
    print('\nSaved results/exp4_comparison_metrics.csv')

    print('\n[PASS] Experiment 4 completed successfully.')
    return df_metrics


if __name__ == '__main__':
    main()
