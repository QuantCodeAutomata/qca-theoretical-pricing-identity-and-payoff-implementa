"""LP terminal payoff pricer using the Bates/SVJD characteristic function.

Prices the terminal LP payoff claim H_{t,T} = E_t^QY[e^{-r^Y*(T-t)} g_R(S_T)]
for Uniswap v2 and v3 using the paper's formulas:

Uniswap v2:
    H_v2(theta, S, V) = 2 * G(1/2, S, V, theta)
    (discount absorbed into G via A_disc term)

Uniswap v3:
    H_v3(theta, S, V, p_a, p_b) = (2/pi) * integral over R of
        exp(-i*u/2 * log(p_a*p_b)) * sin(u/2 * log(p_b/p_a))
        / (u * (1 + 4*u^2)) * G(1/2 + i*u, S, V, theta) du

The singularity at u=0 is handled via L'Hopital/series expansion.

Custom — Context7 found no library equivalent for AMM LP terminal payoff
pricing via Fourier integration of the Bates CF.
"""
import numpy as np
from scipy.integrate import quad
from typing import Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.bates_cf import BatesParams, bates_cf


# Numerical integration parameters
_U_MAX = 200.0    # Truncation of Fourier integral
_U_LIMIT = 1e-8   # Threshold below which to use series limit at u=0


def price_v2_lp(
    theta: float, S: float, V: float, params: BatesParams
) -> float:
    """Price the Uniswap v2 LP terminal payoff claim.
    
    H_v2 = 2 * G(1/2, S, V, theta)
    
    The e^{-r^Y*theta} discount is absorbed into G via the A_disc term
    in the Bates CF.
    
    Args:
        theta:  Time to maturity in years.
        S:      Current spot price.
        V:      Current variance V_t.
        params: BatesParams instance.
    
    Returns:
        Discounted expected terminal LP payoff H_v2 in USDC units
        (per unit liquidity).
    """
    if theta <= 0:
        raise ValueError('theta must be positive.')
    if S <= 0:
        raise ValueError('S must be positive.')
    cf_val = bates_cf(0.5 + 0j, S, V, theta, params)
    return 2.0 * float(np.real(cf_val))


def _integrand_v3_real(u: float, S: float, V: float, theta: float,
                        log_sqrt_papb: float, log_pb_pa: float,
                        params: BatesParams) -> float:
    """Real part of the v3 Fourier integrand at u > 0.
    
    Integrand = Re[ exp(-i*u*log_sqrt_papb) * sin(u/2*log_pb_pa)
                    / (u*(1+4*u^2)) * G(1/2 + i*u, S, V, theta) ]
    
    Args:
        u:             Integration variable.
        S:             Current spot.
        V:             Current variance.
        theta:         Time to maturity.
        log_sqrt_papb: 0.5 * log(p_a * p_b) = (1/2)*log(p_a*p_b).
        log_pb_pa:     log(p_b / p_a).
        params:        BatesParams.
    
    Returns:
        Real part of integrand.
    """
    if abs(u) < _U_LIMIT:
        # Series limit at u -> 0:
        # sin(u/2 * L) / u -> L/2 as u->0
        # exp(-i*u*M) -> 1 as u->0
        # G(1/2 + 0j, ...) evaluated at z=0.5
        # Limit: L/(2*(1+4*0)) * G(0.5) = (log_pb_pa/2) * G(0.5)
        g_val = bates_cf(0.5 + 0j, S, V, theta, params)
        return float(np.real((log_pb_pa / 2.0) * g_val))

    z = complex(0.5, u)
    g_val = bates_cf(z, S, V, theta, params)
    sin_term = np.sin(u / 2.0 * log_pb_pa)
    exp_term = np.exp(-1j * u * log_sqrt_papb)
    integrand = exp_term * sin_term / (u * (1.0 + 4.0 * u ** 2)) * g_val
    return float(np.real(integrand))


def price_v3_lp(
    theta: float,
    S: float,
    V: float,
    p_a: float,
    p_b: float,
    params: BatesParams,
    u_max: float = _U_MAX,
    limit: int = 500,
    epsabs: float = 1e-8,
    epsrel: float = 1e-8,
) -> float:
    """Price the Uniswap v3 LP terminal payoff claim via Fourier integration.
    
    H_v3 = (2/pi) * integral_R [
        exp(-i*u/2*log(p_a*p_b)) * sin(u/2*log(p_b/p_a))
        / (u*(1+4*u^2)) * G(1/2 + i*u, S, V, theta)
    ] du
    
    Singularity at u=0 is handled by series expansion.
    
    Args:
        theta:  Time to maturity in years.
        S:      Current spot price.
        V:      Current variance.
        p_a:    Lower range boundary.
        p_b:    Upper range boundary (p_b > p_a > 0).
        params: BatesParams instance.
        u_max:  Upper truncation of Fourier integral.
        limit:  Maximum number of subintervals in quad.
        epsabs: Absolute tolerance for quad.
        epsrel: Relative tolerance for quad.
    
    Returns:
        Discounted expected terminal LP payoff H_v3 in USDC units
        (per unit liquidity).
    """
    if theta <= 0:
        raise ValueError('theta must be positive.')
    if p_a <= 0 or p_b <= p_a:
        raise ValueError('Must have 0 < p_a < p_b.')

    log_sqrt_papb = 0.5 * (np.log(p_a) + np.log(p_b))  # log(sqrt(p_a*p_b))
    log_pb_pa = np.log(p_b / p_a)

    def integrand_fn(u: float) -> float:
        return _integrand_v3_real(u, S, V, theta, log_sqrt_papb, log_pb_pa, params)

    # Integrate from 0 to u_max (integrand is even in u so multiply by 2)
    # Actually the integral is over all R but integrand_real is symmetric,
    # we integrate over [0, u_max] and multiply by 2
    result, err = quad(
        integrand_fn,
        0.0,
        u_max,
        limit=limit,
        epsabs=epsabs,
        epsrel=epsrel,
    )
    return (2.0 / np.pi) * 2.0 * result  # factor 2 for symmetry


def price_lp(
    theta: float,
    S: float,
    V: float,
    params: BatesParams,
    version: str = 'v2',
    p_a: Optional[float] = None,
    p_b: Optional[float] = None,
) -> float:
    """Unified LP payoff pricer dispatching to v2 or v3.
    
    Args:
        theta:   Time to maturity in years.
        S:       Current spot price.
        V:       Current variance.
        params:  BatesParams instance.
        version: 'v2' or 'v3'.
        p_a:     Lower range boundary (required for v3).
        p_b:     Upper range boundary (required for v3).
    
    Returns:
        Discounted expected terminal LP payoff H in USDC units.
    """
    if version == 'v2':
        return price_v2_lp(theta, S, V, params)
    elif version == 'v3':
        if p_a is None or p_b is None:
            raise ValueError('p_a and p_b required for v3 pricing.')
        return price_v3_lp(theta, S, V, p_a, p_b, params)
    else:
        raise ValueError(f'Unknown version: {version}. Use "v2" or "v3".')
