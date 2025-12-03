import pandas as pd
import numpy as np




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
    value_cols = df.select_dtypes(include=[np.number]).columns.difference(df.index.names)

    if method not in {"mean", "median", "cscore"}:
        raise ValueError("method must be one of: 'mean', 'median', 'cscore'")

    df_norm = df.copy()

    vals = df[value_cols]

    if method == "mean":
        row_means = vals.mean(axis=1)
        target = row_means.mean()
        factors = target / row_means
        df_norm[value_cols] = vals.mul(factors, axis=0)

    elif method == "median":
        row_meds = vals.median(axis=1)
        target = row_meds.median()
        factors = target / row_meds
        df_norm[value_cols] = vals.mul(factors, axis=0)

    elif method == "cscore":
        df_norm[value_cols] = vals.apply(robust_zscore, axis=1, result_type="broadcast")

    return df_norm.round(round_digits)
