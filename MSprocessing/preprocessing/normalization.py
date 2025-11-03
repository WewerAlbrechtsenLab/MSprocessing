import pandas as pd
import numpy as np
import plotly.graph_objects as go
from combat.pycombat import pycombat




def batch_correction(
    df: pd.DataFrame,
    batch_column: str = "plate_nr"
) -> pd.DataFrame:
    """
    Perform batch correction using the `pycombat` algorithm.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame of imputed proteomic intensities.
        Rows represent samples (with MultiIndex metadata) and
        columns represent protein features.

    Returns
    -------
    pd.DataFrame
        Batch-corrected DataFrame with the same shape and indices as input.
    """
    batch_labels = df.reset_index()[batch_column].tolist()
    proteomes_combat = pycombat(df.T, batch_labels).T
    return proteomes_combat




def robust_zscore(
    row: np.ndarray | pd.Series
) -> np.ndarray:
    """
    Compute a robust z-score transformation for a given row or vector.

    Parameters
    ----------
    row : array-like
        Input array or Series of numeric values.

    Returns
    -------
    np.ndarray
        Array of robust z-scores, computed as (x - median) / MAD.
        Returns zeros if the median absolute deviation (MAD) is zero.
    """
    med = np.median(row)
    mad = np.median(np.abs(row - med))
    if mad == 0:
        return np.zeros_like(row)
    return (row - med) / mad




def normalize_sample(
    df: pd.DataFrame,
    method: str = "median",
    round_digits: int = 3
) -> pd.DataFrame:
    """
    Normalize sample intensities using mean, median, or robust z-score scaling.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where rows represent samples and columns represent features.
    method : {"mean", "median", "cscore"}, default="median"
        Normalization method:
          - "mean": Scale each sample to have the same mean.
          - "median": Scale each sample to have the same median.
          - "cscore": Apply robust row-wise z-score normalization (centered score).
    round_digits : int, default=3
        Number of decimal places to round the normalized values.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with the same index and columns as input.
    """
    if method not in {"mean", "median", "cscore"}:
        raise ValueError("method must be one of: 'mean', 'median', 'cscore'")

    if method in {"mean", "median"}:
        if method == "mean":
            center_values = df.mean(axis=1)
            global_center = center_values.mean()
        else:
            center_values = df.median(axis=1)
            global_center = center_values.median()

        # Normalize each sample to global center
        df_norm = df.div(center_values, axis=0).mul(global_center)

    elif method == "cscore":
        df_norm = df.apply(robust_zscore, axis=1, result_type="broadcast")

    return df_norm.round(round_digits)
