"""Unit tests for fee metric functions.

Tests:
    - fee_value: FeeValue = g0 - H
    - fee_return: R = 1 - H/g0
    - fee_rate: y = log(1+R)/theta
    - validate_bounds: 0 <= FeeValue < g0, 0 <= R < 1
    - Edge cases: theta -> 0, P = p_a/p_b
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.fee_metrics import fee_value, fee_return, fee_rate, validate_bounds


class TestFeeValue:
    """Tests for fee_value function."""

    def test_scalar_positive(self):
        """FeeValue = g0 - H for positive H < g0."""
        g0, H = 100.0, 90.0
        assert fee_value(g0, H) == pytest.approx(10.0, abs=1e-12)

    def test_scalar_zero(self):
        """FeeValue = 0 when H = g0 (no fee compensation)."""
        g0, H = 50.0, 50.0
        assert fee_value(g0, H) == pytest.approx(0.0, abs=1e-12)

    def test_array_input(self):
        """fee_value should work with arrays."""
        g0 = np.array([100.0, 200.0])
        H = np.array([90.0, 180.0])
        expected = np.array([10.0, 20.0])
        np.testing.assert_allclose(fee_value(g0, H), expected, atol=1e-12)

    def test_broadcasting(self):
        """fee_value should broadcast scalar g0 over H array."""
        g0 = 100.0
        H = np.array([80.0, 90.0, 95.0])
        expected = np.array([20.0, 10.0, 5.0])
        np.testing.assert_allclose(fee_value(g0, H), expected, atol=1e-12)


class TestFeeReturn:
    """Tests for fee_return function."""

    def test_scalar_valid(self):
        """R = 1 - H/g0."""
        g0, H = 100.0, 90.0
        assert fee_return(g0, H) == pytest.approx(0.1, abs=1e-12)

    def test_zero_fee_return(self):
        """R = 0 when H = g0."""
        g0, H = 100.0, 100.0
        assert fee_return(g0, H) == pytest.approx(0.0, abs=1e-12)

    def test_max_fee_return(self):
        """R approaches 1 when H approaches 0."""
        g0, H = 100.0, 1e-10
        R = fee_return(g0, H)
        assert 0.0 <= R <= 1.0 + 1e-12

    def test_invalid_g0_zero(self):
        """Zero g0 should raise ValueError."""
        with pytest.raises(ValueError):
            fee_return(0.0, 50.0)

    def test_invalid_g0_negative(self):
        """Negative g0 should raise ValueError."""
        with pytest.raises(ValueError):
            fee_return(-10.0, 5.0)

    def test_return_in_0_1_range(self):
        """R should be in [0, 1) for valid inputs."""
        g0 = np.array([100.0, 200.0, 50.0])
        H = np.array([90.0, 180.0, 45.0])
        R = fee_return(g0, H)
        assert np.all(R >= 0.0)
        assert np.all(R < 1.0)


class TestFeeRate:
    """Tests for fee_rate function."""

    def test_scalar_valid(self):
        """y = log(1+R)/theta."""
        R, theta = 0.1, 0.25
        expected = np.log(1.1) / 0.25
        assert fee_rate(R, theta) == pytest.approx(expected, rel=1e-10)

    def test_zero_return(self):
        """y = 0 when R = 0 (log(1+0)/theta = 0)."""
        R = 1e-12  # near zero
        theta = 0.25
        y = fee_rate(R, theta)
        assert y >= 0.0

    def test_invalid_theta_zero(self):
        """theta=0 should raise ValueError."""
        with pytest.raises(ValueError):
            fee_rate(0.1, 0.0)

    def test_invalid_theta_negative(self):
        """Negative theta should raise ValueError."""
        with pytest.raises(ValueError):
            fee_rate(0.1, -1.0)

    def test_invalid_R_negative(self):
        """Negative R should raise ValueError."""
        with pytest.raises(ValueError):
            fee_rate(-0.01, 0.25)

    def test_annualization(self):
        """Doubling theta halves y for same R (approximate for small R)."""
        R = 0.10
        theta1, theta2 = 0.25, 0.50
        y1 = fee_rate(R, theta1)
        y2 = fee_rate(R, theta2)
        # y1/y2 should be approximately 2 (exact for log(1+R)/theta)
        assert abs(y1 / y2 - 2.0) < 1e-10

    def test_array_maturities(self):
        """fee_rate should vectorize over theta array."""
        R = 0.15
        thetas = np.linspace(0.1, 1.0, 10)
        ys = fee_rate(R, thetas)
        expected = np.log(1 + R) / thetas
        np.testing.assert_allclose(ys, expected, atol=1e-12)


class TestValidateBounds:
    """Tests for validate_bounds function."""

    def test_valid_inputs(self):
        """Valid (g0, H) pair should pass all bound checks."""
        g0, H = 100.0, 90.0
        result = validate_bounds(g0, H)
        assert result['fee_value_nonneg']
        assert result['fee_value_less_g0']
        assert result['return_nonneg']
        assert result['return_less_one']

    def test_boundary_H_equals_g0(self):
        """When H = g0, FeeValue = 0 and R = 0 (boundary valid)."""
        g0, H = 100.0, 100.0
        result = validate_bounds(g0, H)
        assert result['fee_value_nonneg']
        assert result['return_nonneg']
        assert result['return_less_one']

    def test_near_zero_H(self):
        """Near-zero H gives R close to 1 but less than 1."""
        g0, H = 100.0, 0.001
        result = validate_bounds(g0, H)
        assert result['return_nonneg']
        assert result['return_less_one']
        assert result['max_R'] < 1.0

    def test_array_inputs(self):
        """validate_bounds should work with arrays."""
        g0 = np.array([100.0, 200.0])
        H = np.array([90.0, 180.0])
        result = validate_bounds(g0, H)
        assert result['fee_value_nonneg']
        assert result['return_nonneg']
        assert result['return_less_one']

    def test_r_range_reported(self):
        """max_R and min_R should be valid."""
        g0 = np.array([100.0, 100.0, 100.0])
        H = np.array([50.0, 80.0, 95.0])
        result = validate_bounds(g0, H)
        assert result['min_R'] >= 0.0
        assert result['max_R'] < 1.0
        assert result['max_R'] >= result['min_R']
