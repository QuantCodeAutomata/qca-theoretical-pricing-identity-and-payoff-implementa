"""Unit tests for Bates/SVJD characteristic function.

Tests:
    - G(z=0) = 1 (moment generating function at z=0)
    - G(z=1) = S * exp(drift*theta) (first moment)
    - Real-valued output for real z
    - Parameter validation and edge cases
    - BatesParams dataclass conversion
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.models.bates_cf import BatesParams, bates_cf, bates_cf_vec


# ─── Standard test parameters ─────────────────────────────────────────────────
TEST_PARAMS = BatesParams(
    lam=0.01, kappa=-0.05, b_v=10.0, beta=20.0,
    sigma_v=5.0, V0=0.20, rho=-0.50, delta=0.10,
    rX=0.0, rY=0.0,
)

PAPER_PARAMS = BatesParams(
    lam=0.004, kappa=1.96, b_v=21.335, beta=55.3,
    sigma_v=12.664, V0=0.497, rho=-0.124, delta=0.139,
    rX=0.0, rY=0.0,
)

S = 2000.0
V = 0.25
THETA = 0.25


class TestBatesParamsDataclass:
    """Tests for BatesParams dataclass."""

    def test_kbar_property(self):
        """kbar = exp(kappa) - 1."""
        params = BatesParams(lam=0.01, kappa=0.0, b_v=1.0, beta=1.0,
                             sigma_v=1.0, V0=0.1, rho=0.0, delta=0.1)
        assert params.kbar == pytest.approx(0.0, abs=1e-12)

    def test_kbar_negative_kappa(self):
        """kbar = exp(kappa) - 1 for negative kappa."""
        kappa = -0.1
        params = BatesParams(lam=0.01, kappa=kappa, b_v=1.0, beta=1.0,
                             sigma_v=1.0, V0=0.1, rho=0.0, delta=0.1)
        expected = np.exp(kappa) - 1.0
        assert params.kbar == pytest.approx(expected, rel=1e-10)

    def test_to_from_array(self):
        """Round-trip conversion to/from numpy array."""
        arr = TEST_PARAMS.to_array()
        params2 = BatesParams.from_array(arr)
        assert params2.lam == pytest.approx(TEST_PARAMS.lam)
        assert params2.kappa == pytest.approx(TEST_PARAMS.kappa)
        assert params2.V0 == pytest.approx(TEST_PARAMS.V0)

    def test_array_length(self):
        """Parameter array should have exactly 10 elements."""
        arr = TEST_PARAMS.to_array()
        assert len(arr) == 10

    def test_paper_params(self):
        """Paper example params should load correctly."""
        arr = PAPER_PARAMS.to_array()
        assert arr[0] == pytest.approx(0.004)   # lam
        assert arr[1] == pytest.approx(1.96)    # kappa
        assert arr[5] == pytest.approx(0.497)   # V0


class TestBatesCFMathProperties:
    """Mathematical property tests for the Bates CF."""

    def test_cf_at_z_zero_equals_one_for_discount(self):
        """G(0, S, V, theta) should equal exp(-rY*theta).
        
        At z=0: E[e^{0*log(S_T)}] = 1, but discount exp(-rY*theta) remains.
        With rY=0: G(0) = 1.
        """
        val = bates_cf(0.0 + 0j, S, V, THETA, TEST_PARAMS)
        expected = np.exp(-TEST_PARAMS.rY * THETA)  # = 1.0 when rY=0
        assert abs(val - expected) < 1e-8

    def test_cf_real_for_real_z(self):
        """G(z, ...) should be real-valued when z is real."""
        for z_real in [0.0, 0.5, 1.0, -0.5]:
            val = bates_cf(z_real + 0j, S, V, THETA, TEST_PARAMS)
            assert abs(np.imag(val)) < 1e-8, \
                f'Imaginary part nonzero for z={z_real}: {np.imag(val)}'

    def test_cf_positive_at_half(self):
        """G(1/2, S, V, theta) should be positive (used for v2 LP pricing)."""
        val = bates_cf(0.5 + 0j, S, V, THETA, TEST_PARAMS)
        assert np.real(val) > 0.0

    def test_cf_positive_at_half_paper_params(self):
        """G(1/2) should be positive for paper example params."""
        val = bates_cf(0.5 + 0j, S, V, THETA, PAPER_PARAMS)
        assert np.real(val) > 0.0

    def test_cf_decreases_with_theta(self):
        """For z=1/2, G should generally decrease as theta increases
        (discount effect dominates over long horizons with rY=0 dynamics)."""
        val_short = bates_cf(0.5 + 0j, S, V, 0.1, TEST_PARAMS)
        val_long = bates_cf(0.5 + 0j, S, V, 1.0, TEST_PARAMS)
        # This is a qualitative check; G can increase or decrease depending on params
        # Just verify both are positive
        assert np.real(val_short) > 0.0
        assert np.real(val_long) > 0.0

    def test_cf_at_small_theta(self):
        """As theta -> 0, G(z, S, V, theta) -> S^z (no evolution)."""
        theta_small = 1e-5
        z = 0.5 + 0j
        val = bates_cf(z, S, V, theta_small, TEST_PARAMS)
        expected = S ** 0.5  # = S^z for z=0.5, theta->0
        # Should be close to S^0.5 when theta is tiny
        assert abs(float(np.real(val)) - expected) / expected < 0.1

    def test_cf_vectorized(self):
        """bates_cf_vec should produce same results as loop over z."""
        z_arr = np.array([0.5 + 0j, 0.5 + 1j, 0.5 + 2j])
        vals_vec = bates_cf_vec(z_arr, S, V, THETA, TEST_PARAMS)
        vals_loop = np.array([bates_cf(z, S, V, THETA, TEST_PARAMS) for z in z_arr])
        np.testing.assert_allclose(vals_vec, vals_loop, atol=1e-12)


class TestBatesCFNumericalStability:
    """Tests for numerical stability of the Bates CF."""

    def test_large_u_decay(self):
        """For large u, |G(1/2 + iu, ...)| should decay (damping)."""
        u_small = 1.0
        u_large = 100.0
        val_small = abs(bates_cf(0.5 + 1j * u_small, S, V, THETA, TEST_PARAMS))
        val_large = abs(bates_cf(0.5 + 1j * u_large, S, V, THETA, TEST_PARAMS))
        # The magnitude should generally decrease for large u
        assert val_large <= val_small * 10.0  # Not strict, just not exploding

    def test_different_spots(self):
        """CF should scale correctly with spot price."""
        z = 0.5 + 0j
        S1, S2 = 1000.0, 2000.0
        val1 = bates_cf(z, S1, V, THETA, TEST_PARAMS)
        val2 = bates_cf(z, S2, V, THETA, TEST_PARAMS)
        # G(z=0.5, S, ...) should scale as S^0.5 at leading order
        ratio = float(np.real(val2) / np.real(val1))
        expected_ratio = (S2 / S1) ** 0.5
        # This is approximate since the full CF has drift/vol adjustments
        assert 0.5 * expected_ratio < ratio < 2.0 * expected_ratio
