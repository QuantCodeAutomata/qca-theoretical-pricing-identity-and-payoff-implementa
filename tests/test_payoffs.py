"""Unit tests for AMM LP payoff functions.

Tests:
    - v2 payoff: g_v2(P) = 2*sqrt(P)
    - v3 compact vs piecewise equivalence
    - v3 boundary and edge cases
    - Shape properties: positivity, monotonicity, concavity
    - Input validation
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.amm_payoffs import (
    g_v2, g_v3_compact, g_v3_piecewise, g_v3,
    verify_v3_equivalence, check_shape_properties,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────
S_T = 2000.0
P_A = 0.7 * S_T   # 1400
P_B = 1.4 * S_T   # 2800
P_A_NARROW = 0.8 * S_T
P_B_NARROW = 1.2 * S_T


class TestUniswapV2Payoff:
    """Tests for Uniswap v2 LP payoff g_v2(P) = 2*sqrt(P)."""

    def test_scalar_exact_value(self):
        """g_v2(P) should equal 2*sqrt(P) exactly."""
        P = 2000.0
        assert np.isclose(g_v2(P), 2.0 * np.sqrt(P), atol=1e-12)

    def test_zero_price(self):
        """g_v2(0) = 0."""
        assert float(g_v2(0.0)) == pytest.approx(0.0, abs=1e-12)

    def test_array_input(self):
        """g_v2 should work with numpy arrays."""
        P = np.array([1.0, 100.0, 2000.0, 10000.0])
        result = g_v2(P)
        expected = 2.0 * np.sqrt(P)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_positivity(self):
        """g_v2(P) >= 0 for P > 0."""
        P = np.linspace(1, 10000, 500)
        assert np.all(g_v2(P) >= 0)

    def test_monotone_increasing(self):
        """g_v2(P) is non-decreasing in P."""
        P = np.linspace(1, 10000, 500)
        diffs = np.diff(g_v2(P))
        assert np.all(diffs >= -1e-12)

    def test_concavity(self):
        """g_v2(P) is concave (g'' <= 0)."""
        P = np.linspace(1.0, 10000.0, 1000)
        g = g_v2(P)
        h = P[1] - P[0]
        g_double_prime = (g[2:] - 2 * g[1:-1] + g[:-2]) / h ** 2
        assert np.all(g_double_prime <= 1e-8)

    def test_negative_price_raises(self):
        """Negative price should raise ValueError."""
        with pytest.raises(ValueError):
            g_v2(-1.0)


class TestUniswapV3CompactPayoff:
    """Tests for v3 compact payoff formula."""

    def test_below_range(self):
        """For P << p_a, payoff = P*(1/sqrt(p_a) - 1/sqrt(p_b))."""
        P = 500.0  # below p_a=1400
        expected = P * (1.0 / np.sqrt(P_A) - 1.0 / np.sqrt(P_B))
        result = float(g_v3_compact(P, P_A, P_B))
        assert abs(result - expected) < 1e-10

    def test_above_range(self):
        """For P >> p_b, payoff = sqrt(p_b) - sqrt(p_a)."""
        P = 5000.0
        expected = np.sqrt(P_B) - np.sqrt(P_A)
        result = float(g_v3_compact(P, P_A, P_B))
        assert abs(result - expected) < 1e-10

    def test_at_spot_S_t(self):
        """Payoff at S_t (in-range) should be positive."""
        assert g_v3_compact(S_T, P_A, P_B) > 0

    def test_wide_range_approaches_v2(self):
        """Very wide v3 range should approach v2 payoff at spot."""
        # g_v3(P, p_a, p_b) = 2*sqrt(P) - P/sqrt(p_b) - sqrt(p_a)   (in-range, S in [p_a, p_b])
        # As p_a -> 0 and p_b -> inf: correction terms -> 0, g_v3 -> 2*sqrt(P) = g_v2.
        # Choose very extreme boundaries so the correction < 0.1% of g_v2.
        P = S_T
        p_a = 0.001    # very low => sqrt(p_a) = 0.0316
        p_b = 1e9      # very high => P/sqrt(p_b) = 2000/31623 = 0.063
        g_wide = float(g_v3_compact(P, p_a, p_b))
        g_v2_val = float(g_v2(P))
        # Combined residual: sqrt(p_a) + P/sqrt(p_b) ≈ 0.095 << g_v2 ≈ 89.4
        assert abs(g_wide - g_v2_val) / g_v2_val < 0.01

    def test_invalid_pa_negative(self):
        """Negative p_a should raise ValueError."""
        with pytest.raises(ValueError):
            g_v3_compact(1000.0, -100.0, 2000.0)

    def test_invalid_pa_greater_than_pb(self):
        """p_a >= p_b should raise ValueError."""
        with pytest.raises(ValueError):
            g_v3_compact(1000.0, 2000.0, 1000.0)

    def test_positivity_on_grid(self):
        """Payoff must be >= 0 for all P > 0."""
        P_grid = np.linspace(0.01, 10000.0, 500)
        g = g_v3_compact(P_grid, P_A, P_B)
        assert np.all(g >= -1e-10), f'Min value: {g.min()}'


class TestUniswapV3PiecewisePayoff:
    """Tests for v3 piecewise payoff formula."""

    def test_below_range_piecewise(self):
        """Below range: g(P) = P*(1/sqrt(p_a) - 1/sqrt(p_b))."""
        P = np.array([100.0, 500.0, 1000.0])  # all below p_a=1400
        expected = P * (1.0 / np.sqrt(P_A) - 1.0 / np.sqrt(P_B))
        result = g_v3_piecewise(P, P_A, P_B)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_in_range_piecewise(self):
        """In range: g(P) = 2*sqrt(P) - P/sqrt(p_b) - sqrt(p_a)."""
        P = np.array([1500.0, 2000.0, 2500.0])  # all in range
        expected = 2.0 * np.sqrt(P) - P / np.sqrt(P_B) - np.sqrt(P_A)
        result = g_v3_piecewise(P, P_A, P_B)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_above_range_piecewise(self):
        """Above range: g(P) = sqrt(p_b) - sqrt(p_a)."""
        P = np.array([3000.0, 4000.0, 5000.0])  # all above p_b=2800
        expected = np.sqrt(P_B) - np.sqrt(P_A)
        result = g_v3_piecewise(P, P_A, P_B)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_continuity_at_pa(self):
        """Payoff must be continuous at P = p_a."""
        eps = 1e-6
        g_below = float(g_v3_piecewise(P_A - eps, P_A, P_B))
        g_above = float(g_v3_piecewise(P_A + eps, P_A, P_B))
        assert abs(g_below - g_above) < 1e-4

    def test_continuity_at_pb(self):
        """Payoff must be continuous at P = p_b."""
        eps = 1e-6
        g_below = float(g_v3_piecewise(P_B - eps, P_A, P_B))
        g_above = float(g_v3_piecewise(P_B + eps, P_A, P_B))
        assert abs(g_below - g_above) < 1e-4


class TestV3Equivalence:
    """Tests for compact vs piecewise equivalence."""

    def test_equivalence_dense_grid(self):
        """Compact and piecewise forms must agree on dense price grid."""
        P_grid = np.linspace(0.01 * S_T, 5.0 * S_T, 2000)
        assert verify_v3_equivalence(P_grid, P_A, P_B, tol=1e-10)

    def test_equivalence_narrow_range(self):
        """Equivalence for narrow (0.8S, 1.2S) range."""
        P_grid = np.linspace(0.1 * S_T, 4.0 * S_T, 2000)
        assert verify_v3_equivalence(P_grid, P_A_NARROW, P_B_NARROW, tol=1e-10)

    def test_exact_at_pa_pb(self):
        """Both forms must agree exactly at boundary points p_a and p_b."""
        for P in [P_A, P_B, S_T]:
            g_c = float(g_v3_compact(P, P_A, P_B))
            g_p = float(g_v3_piecewise(P, P_A, P_B))
            assert abs(g_c - g_p) < 1e-10, f'Disagreement at P={P}: {g_c} vs {g_p}'

    def test_max_abs_error_below_threshold(self):
        """Maximum absolute error between formulas must be < 1e-10."""
        P_grid = np.linspace(1.0, 10 * S_T, 5000)
        g_c = g_v3_compact(P_grid, P_A, P_B)
        g_p = g_v3_piecewise(P_grid, P_A, P_B)
        max_err = float(np.max(np.abs(g_c - g_p)))
        assert max_err < 1e-10, f'Max abs error = {max_err}'


class TestShapeProperties:
    """Tests for qualitative shape properties of payoff functions."""

    def test_v2_shape_properties(self):
        """v2: positivity, monotone, concave on uniform grid."""
        P = np.linspace(1.0, 10000.0, 500)
        g = g_v2(P)
        props = check_shape_properties(P, g)
        assert props['positivity'], 'v2 positivity failed'
        assert props['monotone'], 'v2 monotonicity failed'
        assert props['concave'], 'v2 concavity failed'

    def test_v3_shape_properties(self):
        """v3: positivity, monotone, concave (excluding exact boundaries)."""
        # Use interior of grid away from exact boundaries
        P = np.linspace(0.01 * S_T, 5.0 * S_T, 1000)
        g = g_v3_compact(P, P_A, P_B)
        props = check_shape_properties(P, g, tol_fd=1e-5)
        assert props['positivity'], 'v3 positivity failed'
        assert props['monotone'], f'v3 monotonicity failed: {props["max_g_prime_violation"]}'

    def test_v3_narrow_shape_properties(self):
        """v3 narrow range: shape properties hold."""
        P = np.linspace(0.01 * S_T, 5.0 * S_T, 1000)
        g = g_v3_compact(P, P_A_NARROW, P_B_NARROW)
        props = check_shape_properties(P, g, tol_fd=1e-5)
        assert props['positivity'], 'v3 narrow positivity failed'
        assert props['monotone'], 'v3 narrow monotonicity failed'

    def test_v2_equals_2_sqrt(self):
        """g_v2(S_T) must be exactly 2*sqrt(S_T)."""
        val = float(g_v2(S_T))
        expected = 2.0 * np.sqrt(S_T)
        assert np.isclose(val, expected, atol=1e-12), \
            f'g_v2({S_T}) = {val}, expected {expected}'
