"""PCA analysis of first differences in implied LP fee-rate curves.

Reproduces the paper's PCA results on changes in fee-rate term structures.
Using scikit-learn PCA — Context7 confirmed.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Optional


def build_differenced_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute first differences of the fee-rate panel.

    Delta Y_t(theta) = Y_t(theta) - Y_{t-1}(theta)

    Args:
        panel: (dates x maturities) DataFrame of fee rates. Must have >=2 rows.

    Returns:
        First-differenced panel with (n_dates - 1) rows.

    Raises:
        ValueError: If panel is empty or has fewer than 2 rows.
    """
    if panel.empty or len(panel) < 2:
        raise ValueError('Panel must have at least 2 rows to compute first differences.')
    return panel.diff().dropna()


def build_fee_rate_panel(
    daily_curves: Dict[str, pd.DataFrame],
    theta_grid: np.ndarray,
    rate_col: str = 'y',
) -> pd.DataFrame:
    """Build a (dates x maturities) panel of fee rates.
    
    Args:
        daily_curves: Dict mapping date string -> DataFrame with [theta, y, ...].
        theta_grid:   Common maturity grid (same for all dates).
        rate_col:     Column name for fee rate ('y' by default).
    
    Returns:
        DataFrame with dates as index and maturities as columns.
    """
    rows = {}
    for date, df in sorted(daily_curves.items()):
        # Interpolate/select to common theta_grid
        row = {}
        for theta in theta_grid:
            match = df[np.isclose(df['theta'], theta, atol=1e-6)]
            if len(match) > 0:
                row[theta] = float(match[rate_col].iloc[0])
            else:
                row[theta] = np.nan
        rows[date] = row
    panel = pd.DataFrame(rows).T
    panel.index.name = 'date'
    panel.columns = [f'theta_{t:.4f}' for t in theta_grid]
    return panel


def compute_pca_on_differences(
    panel: pd.DataFrame,
    n_components: int = 5,
) -> Dict:
    """Run PCA on first differences of fee-rate panel.
    
    Paper specifies PCA on Delta y(t, theta) = y(t, theta) - y(t^-, theta).
    Centers each tenor column before PCA (covariance-based PCA).
    
    Args:
        panel:        (dates x maturities) DataFrame of fee rates.
        n_components: Number of principal components to extract.
    
    Returns:
        Dictionary with:
            'diff_matrix':         First differences matrix.
            'eigenvalues':         PCA eigenvalues.
            'explained_variance_ratio': Per-component ratios.
            'cumulative_variance': Cumulative explained variance.
            'loadings':            Component loadings (n_components x n_maturities).
            'scores':              Factor scores (n_dates-1 x n_components).
            'dates':               Date index of scores.
            'pca':                 Fitted sklearn PCA object.
    """
    # Drop rows with any NaN
    panel_clean = panel.dropna()
    if len(panel_clean) < 2:
        raise ValueError('Panel must have at least 2 dates after dropping NaNs.')
    
    # First differences: Delta Y_t = Y_t - Y_{t-1}
    diff_matrix = panel_clean.diff().dropna()
    
    # Assert we are using differences, not levels
    assert len(diff_matrix) == len(panel_clean) - 1, \
        'Differenced matrix must have one fewer row than levels panel.'
    
    n_components = min(n_components, diff_matrix.shape[1], diff_matrix.shape[0])
    
    # sklearn PCA centers each column before decomposition
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(diff_matrix.values)
    
    return {
        'diff_matrix': diff_matrix,
        'eigenvalues': pca.explained_variance_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'loadings': pca.components_,  # shape (n_components, n_maturities)
        'scores': pd.DataFrame(
            scores,
            index=diff_matrix.index,
            columns=[f'PC{i+1}' for i in range(n_components)],
        ),
        'dates': diff_matrix.index,
        'pca': pca,
    }
