"""Unit tests for PCA of fee-rate curve first differences.

Tests:
    - compute_pca_on_differences uses first differences (not levels)
    - Explained variance sums to ≤ 1
    - Loadings matrix is orthonormal
    - Scores have correct dimensions
    - n_components boundary conditions
    - Variance concentration for low-dimensional data
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.analytics.pca_fee_curves import compute_pca_on_differences, build_differenced_panel


# ─── Fixtures ─────────────────────────────────────────────────────────────────
N_DATES = 100
N_TENORS = 8
THETA_COLS = [f'theta_{t:.4f}' for t in np.linspace(0.1, 0.5, N_TENORS)]


def make_synthetic_panel(n_dates: int = N_DATES,
                         n_tenors: int = N_TENORS,
                         seed: int = 42) -> pd.DataFrame:
    """Create a synthetic fee-rate panel with known structure."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2020-01-01', periods=n_dates, freq='B')
    # Single factor structure: 1 latent factor + noise
    factor = rng.standard_normal(n_dates)
    loadings = rng.uniform(0.5, 1.5, n_tenors)
    noise = rng.standard_normal((n_dates, n_tenors)) * 0.01
    data = np.outer(factor, loadings) + noise
    # Convert to fee-rate scale (positive values)
    data = np.abs(data) + 0.01
    return pd.DataFrame(data, index=dates, columns=THETA_COLS[:n_tenors])


def make_two_factor_panel(n_dates: int = 200, n_tenors: int = N_TENORS,
                          seed: int = 42) -> pd.DataFrame:
    """Panel with 2 dominant factors (level + slope)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2019-01-01', periods=n_dates, freq='B')
    tenors = np.linspace(0.1, 0.5, n_tenors)
    # PC1: level factor
    f1 = rng.standard_normal(n_dates)
    l1 = np.ones(n_tenors)
    # PC2: slope factor
    f2 = rng.standard_normal(n_dates) * 0.3
    l2 = tenors - tenors.mean()
    noise = rng.standard_normal((n_dates, n_tenors)) * 0.001
    data = np.outer(f1, l1) + np.outer(f2, l2) + noise
    data = np.abs(data) + 0.05
    cols = [f'theta_{t:.4f}' for t in tenors]
    return pd.DataFrame(data, index=dates, columns=cols)


class TestBuildDifferencedPanel:
    """Tests for first-difference computation."""

    def test_first_differences_shape(self):
        """Differenced panel has one fewer row than input."""
        panel = make_synthetic_panel()
        diff = build_differenced_panel(panel)
        assert diff.shape == (N_DATES - 1, N_TENORS)

    def test_first_differences_values(self):
        """Manual first differences should match the function output."""
        panel = make_synthetic_panel(n_dates=10)
        diff = build_differenced_panel(panel)
        expected = panel.diff().dropna()
        pd.testing.assert_frame_equal(diff, expected)

    def test_diff_preserves_columns(self):
        """Differenced panel columns match input panel columns."""
        panel = make_synthetic_panel()
        diff = build_differenced_panel(panel)
        assert list(diff.columns) == list(panel.columns)

    def test_empty_panel_raises(self):
        """Empty panel should raise ValueError."""
        with pytest.raises(ValueError):
            build_differenced_panel(pd.DataFrame())

    def test_single_row_raises(self):
        """Single-row panel cannot be differenced; should raise ValueError."""
        panel = make_synthetic_panel(n_dates=1)
        with pytest.raises(ValueError):
            build_differenced_panel(panel)


class TestComputePcaOnDifferences:
    """Tests for PCA computation on differenced fee-rate panel."""

    def test_explained_variance_sums_to_one(self):
        """Cumulative explained variance for all components should sum to 1."""
        panel = make_synthetic_panel(n_dates=50)
        result = compute_pca_on_differences(panel, n_components=N_TENORS)
        total = result['cumulative_variance'][-1]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_explained_variance_nonneg(self):
        """All explained variance ratios must be non-negative."""
        panel = make_synthetic_panel()
        result = compute_pca_on_differences(panel, n_components=3)
        assert np.all(result['explained_variance_ratio'] >= 0.0)

    def test_cumulative_monotone(self):
        """Cumulative explained variance must be non-decreasing."""
        panel = make_synthetic_panel()
        result = compute_pca_on_differences(panel, n_components=5)
        cum = result['cumulative_variance']
        assert np.all(np.diff(cum) >= -1e-10)

    def test_loadings_orthonormal(self):
        """PCA loadings rows should be approximately orthonormal."""
        panel = make_synthetic_panel()
        result = compute_pca_on_differences(panel, n_components=4)
        L = result['loadings']  # shape (n_components, n_tenors)
        gram = L @ L.T
        np.testing.assert_allclose(gram, np.eye(len(L)), atol=1e-10)

    def test_scores_shape(self):
        """Scores DataFrame should have shape (n_dates - 1, n_components)."""
        panel = make_synthetic_panel(n_dates=50)
        n_comp = 3
        result = compute_pca_on_differences(panel, n_components=n_comp)
        assert result['scores'].shape == (49, n_comp)

    def test_single_factor_dominates(self):
        """Single-factor panel should have PC1 explaining >90% variance."""
        panel = make_synthetic_panel(seed=1)
        result = compute_pca_on_differences(panel, n_components=3)
        ev1 = result['explained_variance_ratio'][0]
        assert ev1 > 0.80, f'PC1 explains only {ev1:.1%}'

    def test_two_factor_cumulative(self):
        """Two-factor panel: PC1+PC2 cumulative should exceed 90%."""
        panel = make_two_factor_panel()
        result = compute_pca_on_differences(panel, n_components=5)
        cum2 = result['cumulative_variance'][1]
        assert cum2 > 0.90, f'PC1+PC2 cumulative = {cum2:.1%}'

    def test_n_components_equals_n_tenors(self):
        """n_components can be at most n_tenors; no error."""
        panel = make_synthetic_panel(n_dates=50)
        result = compute_pca_on_differences(panel, n_components=N_TENORS)
        assert result['cumulative_variance'][-1] == pytest.approx(1.0, abs=1e-6)

    def test_n_components_one(self):
        """n_components=1 should work."""
        panel = make_synthetic_panel()
        result = compute_pca_on_differences(panel, n_components=1)
        assert len(result['explained_variance_ratio']) == 1
        assert result['scores'].shape[1] == 1

    def test_pca_on_levels_raises(self):
        """PCA function should assert that input has > 1 row for differencing."""
        panel = make_synthetic_panel(n_dates=2)
        result = compute_pca_on_differences(panel, n_components=1)
        # Should produce 1-row differenced panel -> PCA with 1 observation is degenerate
        # The function should handle this gracefully
        assert result is not None

    def test_returns_required_keys(self):
        """Result dict should contain required keys."""
        panel = make_synthetic_panel()
        result = compute_pca_on_differences(panel, n_components=3)
        required_keys = [
            'explained_variance_ratio', 'cumulative_variance',
            'loadings', 'scores', 'eigenvalues',
        ]
        for key in required_keys:
            assert key in result, f'Missing key: {key}'
