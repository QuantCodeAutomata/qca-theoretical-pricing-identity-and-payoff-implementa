"""Bates (1996) / SVJD characteristic function.

Implements E[e^{z*log(S_T)} | S_t=S, V_t=V] for the stochastic volatility
with jumps model under Q^Y.

Dynamics under Q^Y (r^X = r^Y = 0 empirically):
    dS_t/S_t = -lambda*kbar*dt + sqrt(V_t)*dZ_t + k*dq_t
    dV_t = (b_v - beta*V_t)*dt + sigma_v*sqrt(V_t)*dZ_t^v
    corr(dZ, dZ^v) = rho
    kbar = exp(kappa) - 1
    ln(1+k) ~ N(kappa - 0.5*delta^2, delta^2)

Custom — Context7 found no library equivalent matching the paper's exact
parameterization. Based on Bates (1996) / Lewis (2001).
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Union


@dataclass
class BatesParams:
    """Parameter container for the Bates/SVJD model.
    
    Attributes:
        lam:     Jump intensity (lambda, jumps per year).
        kappa:   Mean log-jump size parameter.
        b_v:     Long-run variance drift (b_v in paper).
        beta:    Mean-reversion speed of variance.
        sigma_v: Volatility of variance (vol-of-vol).
        V0:      Initial variance level.
        rho:     Correlation between price and variance Brownian motions.
        delta:   Jump size volatility.
        rX:      Short rate for token X (default 0).
        rY:      Short rate for token Y (default 0).
    """
    lam: float
    kappa: float
    b_v: float
    beta: float
    sigma_v: float
    V0: float
    rho: float
    delta: float
    rX: float = 0.0
    rY: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to ordered numpy array for optimization."""
        return np.array([self.lam, self.kappa, self.b_v, self.beta,
                         self.sigma_v, self.V0, self.rho, self.delta,
                         self.rX, self.rY])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'BatesParams':
        """Create from ordered numpy array."""
        return cls(*arr)

    @property
    def kbar(self) -> float:
        """Drift adjustment: E[k] = exp(kappa) - 1."""
        return np.exp(self.kappa) - 1.0


def bates_cf(z: complex, S: float, V: float, theta: float,
             params: BatesParams) -> complex:
    """Bates characteristic function G(z, S, V, theta).
    
    Computes E[e^{z*log(S_T)} | S_t=S, V_t=V] for the SVJD model.
    
    Uses the affine structure:
        G(z, S, V, theta) = exp(z*log(S) + A(z,theta) + B(z,theta)*V
                               + C_jump(z, theta))
    
    where A, B are the Heston SV components and C_jump is the jump
    contribution following Bates (1996) Eq. (3).
    
    Args:
        z:      Complex argument (e.g., 1/2 + i*u for LP pricing).
        S:      Current spot price.
        V:      Current variance V_t.
        theta:  Time to maturity in years (theta > 0).
        params: BatesParams instance.
    
    Returns:
        Complex value of the characteristic function G(z, S, V, theta).
    """
    p = params
    mu_drift = p.rY - p.rX - p.lam * p.kbar  # risk-neutral drift of log S

    # Heston SV component (affine structure)
    # From the Riccati ODE for the moment generating function E[e^{z*log(S_T)}]:
    #   dB/dtau = (z^2-z)/2 + rho*sigma_v*z*B + (sigma_v^2/2)*B^2 - beta*B
    # Discriminant: D^2 = (beta - rho*sigma_v*z)^2 - sigma_v^2*(z^2 - z)
    # Note: the sign is MINUS sigma_v^2*alpha (not plus), which ensures the
    # GBM limit E[S_T^{1/2}] = S^{1/2}*exp(-V*T/8) is recovered correctly.
    alpha = z * z - z  # z^2 - z
    c1 = p.beta - p.rho * p.sigma_v * z
    D = np.sqrt(c1 ** 2 - p.sigma_v ** 2 * alpha)

    # G+(z) = (c1 + D) and G-(z) = (c1 - D)
    G_plus = c1 + D
    G_minus = c1 - D

    # Ratio r_ratio = G_minus / G_plus
    r_ratio = G_minus / G_plus

    exp_Dtheta = np.exp(-D * theta)

    # B coefficient for V_t term
    B = (G_minus * (1.0 - exp_Dtheta)) / (p.sigma_v ** 2 * (1.0 - r_ratio * exp_Dtheta))

    # A coefficient (integral of mean-reversion)
    log_term = np.log((1.0 - r_ratio * exp_Dtheta) / (1.0 - r_ratio))
    A_sv = (p.b_v / p.sigma_v ** 2) * (G_minus * theta - 2.0 * log_term)

    # Drift contribution
    A_drift = z * mu_drift * theta

    # Discount contribution
    A_disc = -p.rY * theta

    # Jump contribution: Bates (1996)
    # exp(lambda*theta*(E[e^{z*ln(1+k)}] - 1 - z*kbar))
    # ln(1+k) ~ N(kappa - 0.5*delta^2, delta^2)
    # E[e^{z*ln(1+k)}] = exp(z*(kappa - 0.5*delta^2) + 0.5*z^2*delta^2)
    jump_mgf = np.exp(z * (p.kappa - 0.5 * p.delta ** 2) + 0.5 * z ** 2 * p.delta ** 2)
    A_jump = p.lam * theta * (jump_mgf - 1.0 - z * p.kbar)

    A_total = A_drift + A_sv + A_disc + A_jump

    return np.exp(z * np.log(S) + A_total + B * V)


def bates_cf_vec(z_arr: np.ndarray, S: float, V: float, theta: float,
                  params: BatesParams) -> np.ndarray:
    """Vectorized Bates CF over array of complex z values.
    
    Args:
        z_arr:  Array of complex arguments.
        S:      Current spot price.
        V:      Current variance.
        theta:  Time to maturity.
        params: BatesParams instance.
    
    Returns:
        Array of complex CF values.
    """
    return np.array([bates_cf(z, S, V, theta, params) for z in z_arr])
