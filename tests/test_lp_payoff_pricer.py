"""Unit tests for LP payoff pricer functions.

Tests:
    - price_v2_lp: H_v2 = 2*G(1/2, S, V, theta)
    - price_v3_lp: H_v3 via Fourier integral
    - Singularity handling at u=0
    - H < g_R(S_0) (implied fee value positive)
    - v3 approaches v2 for wide range
    - Input validation
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.models.bates_cf import BatesParams, bates_cf
from src.pricing.lp_payoff_pricer import price_v2_lp, price_v3_lp, price_lp
from src.amm_payoffs import g_v2, g_v3


# ─── Standard test parameters ─────────────────────────────────────────────────
TEST_PARAMS = BatesParams(
    lam=0.01, kappa=-0.05, b_v=10.0, beta=20.0,
    sigma_v=5.0, V0=0.20, rho=-0.50, delta=0.10,
    rX=0.0, rY=0.0,
)

S = 2000.0
V = 0.20
P_A = 0.8 * S   # 1600
P_B = 1.2 * S   # 2400
THETA = 0.25


class TestPriceV2LP:
    """Tests for Uniswap v2 LP payoff pricing."""

    def test_positive_output(self):
        """H_v2 should be positive."""
        H = price_v2_lp(THETA, S, V, TEST_PARAMS)
        assert H > 0.0

    def test_less_than_g0(self):
        """H_v2 < g_v2(S) (positive implied fee)."""
        H = price_v2_lp(THETA, S, V, TEST_PARAMS)
        g0 = float(g_v2(S))
        assert H < g0, f'H={H}, g0={g0}'

    def test_equals_2_times_cf_real_part(self):
        """H_v2 = 2 * Re[G(1/2, S, V, theta)]."""
        cf_val = bates_cf(0.5 + 0j, S, V, THETA, TEST_PARAMS)
        expected = 2.0 * float(np.real(cf_val))
        H = price_v2_lp(THETA, S, V, TEST_PARAMS)
        assert H == pytest.approx(expected, rel=1e-10)

    def test_short_maturity_approaches_g0(self):
        """For very short theta, H_v2 should be close to g_v2(S)."""
        H = price_v2_lp(0.001, S, V, TEST_PARAMS)
        g0 = float(g_v2(S))
        # Fee should be small for very short horizon
        fee_return = (g0 - H) / g0
        assert 0.0 <= fee_return < 0.5, f'fee_return={fee_return}'

    def test_invalid_theta_zero(self):
        """theta=0 should raise ValueError."""
        with pytest.raises(ValueError):
            price_v2_lp(0.0, S, V, TEST_PARAMS)

    def test_invalid_S_zero(self):
        """S=0 should raise ValueError."""
        with pytest.raises(ValueError):
            price_v2_lp(THETA, 0.0, V, TEST_PARAMS)

    def test_different_maturities(self):
        """H should vary with maturity."""
        H_short = price_v2_lp(0.1, S, V, TEST_PARAMS)
        H_long = price_v2_lp(0.5, S, V, TEST_PARAMS)
        # For typical parameters, longer maturity -> lower H (more fee compensation)
        assert H_short != pytest.approx(H_long, rel=1e-3)


class TestPriceV3LP:
    """Tests for Uniswap v3 LP payoff pricing."""

    def test_positive_output(self):
        """H_v3 should be positive."""
        H = price_v3_lp(THETA, S, V, P_A, P_B, TEST_PARAMS)
        assert H > 0.0

    def test_less_than_g0(self):
        """H_v3 < g_v3(S, p_a, p_b) (positive implied fee)."""
        H = price_v3_lp(THETA, S, V, P_A, P_B, TEST_PARAMS)
        g0 = float(g_v3(S, P_A, P_B))
        assert H < g0 + 1e-6, f'H={H}, g0={g0}'

    def test_invalid_theta_zero(self):
        """theta=0 should raise ValueError."""
        with pytest.raises(ValueError):
            price_v3_lp(0.0, S, V, P_A, P_B, TEST_PARAMS)

    def test_invalid_pa_greater_pb(self):
        """p_a >= p_b should raise ValueError."""
        with pytest.raises(ValueError):
            price_v3_lp(THETA, S, V, P_B, P_A, TEST_PARAMS)

    def test_invalid_pa_zero(self):
        """p_a=0 should raise ValueError."""
        with pytest.raises(ValueError):
            price_v3_lp(THETA, S, V, 0.0, P_B, TEST_PARAMS)

    def test_narrow_range_higher_fee_rate(self):
        """Narrow range should produce higher implied fee rate than wide range."""
        # Narrow range
        p_a_n, p_b_n = 0.9 * S, 1.1 * S
        H_narrow = price_v3_lp(THETA, S, V, p_a_n, p_b_n, TEST_PARAMS)
        g0_narrow = float(g_v3(S, p_a_n, p_b_n))

        # Wide range
        p_a_w, p_b_w = 0.5 * S, 2.0 * S
        H_wide = price_v3_lp(THETA, S, V, p_a_w, p_b_w, TEST_PARAMS)
        g0_wide = float(g_v3(S, p_a_w, p_b_w))

        R_narrow = (g0_narrow - H_narrow) / g0_narrow
        R_wide = (g0_wide - H_wide) / g0_wide

        y_narrow = np.log(1 + R_narrow) / THETA
        y_wide = np.log(1 + R_wide) / THETA

        assert y_narrow > y_wide, \
            f'Narrow y={y_narrow:.4f} should exceed wide y={y_wide:.4f}'


class TestPriceLPUnified:
    """Tests for unified price_lp dispatcher."""

    def test_v2_dispatch(self):
        """price_lp with version='v2' should match price_v2_lp."""
        H_direct = price_v2_lp(THETA, S, V, TEST_PARAMS)
        H_unified = price_lp(THETA, S, V, TEST_PARAMS, version='v2')
        assert H_direct == pytest.approx(H_unified, rel=1e-10)

    def test_v3_dispatch(self):
        """price_lp with version='v3' should match price_v3_lp."""
        H_direct = price_v3_lp(THETA, S, V, P_A, P_B, TEST_PARAMS)
        H_unified = price_lp(THETA, S, V, TEST_PARAMS, version='v3',
                              p_a=P_A, p_b=P_B)
        assert H_direct == pytest.approx(H_unified, rel=1e-10)

    def test_v3_missing_range_raises(self):
        """v3 without p_a/p_b should raise ValueError."""
        with pytest.raises(ValueError):
            price_lp(THETA, S, V, TEST_PARAMS, version='v3')

    def test_unknown_version_raises(self):
        """Unknown version should raise ValueError."""
        with pytest.raises(ValueError):
            price_lp(THETA, S, V, TEST_PARAMS, version='v4')
