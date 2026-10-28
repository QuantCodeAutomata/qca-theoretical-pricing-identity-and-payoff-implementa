# AMM LP Implied Fee Compensation — Theoretical Pricing & Calibration

This repository implements the theoretical framework and empirical experiments
from a paper that links Uniswap v2/v3 LP payoffs to a no-arbitrage pricing
identity for implied accumulated fee compensation.

## Structure

```
├── data/               # Data loading utilities (Massive REST, synthetic)
├── src/                # Core library
│   ├── amm_payoffs.py          # Uniswap v2 and v3 payoff functions
│   ├── fee_metrics.py          # Fee value, return, and rate computation
│   ├── uniswap_v3_ranges.py    # Tick-based range construction
│   ├── models/bates_cf.py      # Bates (1996) characteristic function
│   ├── calibration/            # SVJD/Bates model calibration
│   ├── pricing/                # LP terminal-payoff pricer (v2, v3)
│   └── analytics/              # Fee curve and PCA helpers
├── exp/                # Experiment scripts (exp_1 … exp_5)
├── tests/              # pytest unit tests (96 tests)
├── results/            # Outputs — CSVs, PNGs, RESULTS.md
└── README.md
```

## Experiments

| ID | Title |
|---|---|
| exp_1 | Payoff formula validation (compact vs piecewise, structural properties) |
| exp_2 | SVJD/Bates calibration and implied LP fee-rate term structures |
| exp_3 | Uniswap v3 fee-rate surface over maturity × moneyness |
| exp_4 | Historical comparison: implied vs realized v2 fee returns |
| exp_5 | PCA of first differences in implied fee-rate curves |

## Quick Start

```bash
# Install dependencies (already in the container)
pip install numpy pandas scipy matplotlib seaborn scikit-learn

# Run all experiments
python exp/exp_1_payoff_validation.py
python exp/exp_2_svjd_calibration.py
python exp/exp_3_fee_rate_surface.py
python exp/exp_4_historical_comparison.py
python exp/exp_5_pca_fee_curves.py

# Run tests
pytest
```

## Data

Set `MASSIVE_TOKEN` to fetch live ETH-USDC option data via the Massive REST API:
```bash
export MASSIVE_TOKEN=your_api_key
```

Without a live feed, experiments use documented paper parameters or synthetic
data-generating processes. See `results/RESULTS.md` for full details.

## No-Arbitrage Pricing Identity

```
E_t^QY[e^{-r^Y(T-t)} Gamma_{t,T}^R] = g_R(S_t) - E_t^QY[e^{-r^Y(T-t)} g_R(S_T)]

  Implied fee value  = g_R(S_t) - H_{t,T}
  Implied fee return = 1 - H_{t,T} / g_R(S_t)
  Fee rate y(t,θ)   = (1/θ) log(1 + R_{t,t+θ})
```

where `g_R` is the LP pool-value payoff (v2: `2√P`; v3: piecewise formula),
`H_{t,T}` is the risk-neutral discounted expectation of the terminal payoff,
and `Gamma_{t,T}^R` is the accumulated fee income.

## Results Summary

See [`results/RESULTS.md`](results/RESULTS.md) for all numerical outputs,
plots, and experiment conclusions.
