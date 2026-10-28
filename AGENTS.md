# AGENTS.md — Repository Knowledge Base

## Project Overview
AMM LP option-pricing and calibration framework for Uniswap v2/v3 positions
modelled under the Bates (1996) Stochastic Volatility with Jumps (SVJD) process.

## Directory Structure
```
src/
  models/
    __init__.py          (empty)
    bates_cf.py          Bates/SVJD characteristic function G(z,S,V,theta)
  pricing/
    __init__.py          (empty)
    lp_payoff_pricer.py  Uniswap v2/v3 LP terminal payoff pricer
  calibration/
    __init__.py          (empty)
    bates_calibration.py SVJD calibration to option cross-sections
  analytics/             (pre-existing)
  amm_payoffs.py         (pre-existing)
  fee_metrics.py         (pre-existing)
  uniswap_v3_ranges.py   (pre-existing)
tests/
data/
exp/
results/
```

## Key Implementation Details

### BatesParams (src/models/bates_cf.py)
- Dataclass holding 10 parameters: [lam, kappa, b_v, beta, sigma_v, V0, rho, delta, rX, rY]
- `to_array()` / `from_array()` enable seamless use with scipy.optimize
- `kbar` property: `exp(kappa) - 1` (jump drift adjustment)
- rX = rY = 0 by default (empirical crypto convention)

### bates_cf (src/models/bates_cf.py)
- Computes E[e^{z*log(S_T)} | S_t=S, V_t=V] using affine structure
- Uses Lewis (2001) sign convention for the D square-root to avoid branch-cut issues:
  - D = sqrt(c1^2 + sigma_v^2*(z^2-z)), c1 = beta - rho*sigma_v*z
  - r_ratio = (c1-D)/(c1+D)
- A_disc = -rY*theta is embedded so G includes e^{-rY*T} discount
- G(z=0) = 1 (verified), G(z=real) = real

### LP Pricing (src/pricing/lp_payoff_pricer.py)
- v2: H_v2 = 2 * Re[G(0.5+0j, S, V, theta)]  (closed form, no integral)
- v3: H_v3 = (4/pi) * integral_0^{u_max} Re[integrand] du  (factor 4 = 2/pi * 2 for symmetry)
  - Singularity at u=0 handled by L'Hopital: limit = (log_pb_pa/2) * G(0.5)
  - log_sqrt_papb = 0.5*(log(p_a)+log(p_b)), log_pb_pa = log(p_b/p_a)
- v3 price increases monotonically with range width [p_a, p_b]
- v3 price < v2 price for any finite range

### Calibration (src/calibration/bates_calibration.py)
- Two-stage: differential_evolution (global) → L-BFGS-B (local refinement)
- Objective: weighted RMSE of IV errors (model IV vs market IV)
- `_bates_call_price` uses Lewis (2001) single-integral with G(0.5+iu)
  - Note: G includes S^z factor; formula normalizes via sqrt(S*K)*disc prefactor
  - Returns max(call, 0) to handle numerical negatives
- `bs_implied_vol`: bisection over [1e-4, 20.0] for IV, returns None if out of range
- PARAM_BOUNDS: rX, rY bounds are (0,0,) — fixed at 0
- calibration_objective returns 1e6 on any exception or failed IV inversion
- calibrate_bates returns dict: {params, obj_value, success, message, x}

## Import Pattern
All submodules use sys.path.insert to add the project root:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.bates_cf import BatesParams, bates_cf
```

## Testing Notes
- Use `python -m py_compile` to syntax-check all files
- Realistic parameter ranges for smoke tests:
  - lam ≈ 0.1–1.0, kappa ≈ -0.1–0, b_v ≈ 0.04–0.1
  - beta ≈ 1–5, sigma_v ≈ 0.2–0.6, V0 ≈ 0.01–0.09
  - rho ≈ -0.7–-0.3, delta ≈ 0.05–0.15
- Extreme sigma_v (> 1.0) causes numerical issues in _bates_call_price integral
- CF sanity check: G(z=0) = 1.0 always
