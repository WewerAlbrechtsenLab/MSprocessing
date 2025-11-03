from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler


def split_data(
    df: pd.DataFrame,
    index_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a wide-format proteomic DataFrame into a feature matrix and metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame where rows correspond to samples (possibly with a MultiIndex)
        and columns correspond to protein measurements.
    index_col : str
        Name of the index level used to uniquely identify samples.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        mat : DataFrame of protein intensity values, indexed by `index_col`.
        meta : DataFrame of metadata, indexed by `index_col`.
    """
    index_cols = list(df.index.names)
    tmp = df.reset_index()
    protein_cols = [c for c in tmp.columns if c not in index_cols]

    mat = tmp.set_index(index_col)[protein_cols].copy()
    meta = tmp[index_cols].set_index(index_col).copy()

    return mat, meta


def within_group_corr(
    proteome: pd.DataFrame,
    meta: pd.DataFrame,
    id_col: str,
    method: str = "pearson"
) -> pd.DataFrame:
    """
    Compute mean pairwise correlation among samples within each group.

    Parameters
    ----------
    proteome : pd.DataFrame
        Rows are samples, columns are protein intensity features.
    meta : pd.DataFrame
        Must have the same index as `proteome` and include a grouping column.
    id_col : str
        Column in `meta` defining group membership.
    method : {"pearson", "spearman", "kendall", "cosine", "euclidean"}, default="pearson"
        Correlation or distance metric to use.

    Returns
    -------
    pd.DataFrame
        Index: group identifiers.
        Columns:
            "corr" : mean pairwise within-group correlation (NaN if < 2 samples)
            "n_samples" : number of samples in the group
    """
    if not proteome.index.equals(meta.index):
        raise ValueError("`proteome` and `meta` must have identical sample indices.")
    
    proteome_z = pd.DataFrame(
        StandardScaler().fit_transform(proteome),
        index=proteome.index,
        columns=proteome.columns
    )

    rows = []
    ids = meta[id_col]

    for group, idx in ids.groupby(ids):
        samples = proteome_z.loc[idx.index]
        n_samples = samples.shape[0]

        if n_samples < 2:
            rows.append((group, np.nan, n_samples))
            continue

        if method in {"pearson", "spearman", "kendall"}:
            corr = samples.T.corr(method=method).to_numpy()
        elif method in {"cosine", "euclidean"}:
            corr = 1 - squareform(pdist(samples, metric=method))
        else:
            raise ValueError(f"Unknown method '{method}'.")

        iu = np.triu_indices_from(corr, k=1)
        vals = corr[iu]
        mean_corr = np.nanmean(vals) if vals.size else np.nan

        rows.append((group, mean_corr, n_samples))

    return pd.DataFrame(rows, columns=[id_col, "corr", "n_samples"]).set_index(id_col)


def within_between_corr(
    proteome: pd.DataFrame,
    meta: pd.DataFrame,
    id_col: str,
    method: str = "pearson"
) -> dict[str, float]:
    """
    Compute mean within-group and between-group correlations among samples.

    Parameters
    ----------
    proteome : pd.DataFrame
        Rows are samples, columns are protein intensity features.
    meta : pd.DataFrame
        Must have the same index as `proteome` and contain `id_col`.
    id_col : str
        Column in `meta` defining the grouping variable.
    method : {"pearson", "spearman", "kendall", "cosine", "euclidean"}, default="pearson"
        Correlation method. Distance metrics ("cosine", "euclidean")
        are converted to correlations as (1 - distance).

    Returns
    -------
    dict[str, float]
        {
            "within_mean": mean within-group correlation,
            "between_mean": mean between-group correlation
        }
    """
    if not proteome.index.equals(meta.index):
        raise ValueError("`proteome` and `meta` must have identical sample indices.")
    
    proteome_z = pd.DataFrame(
        StandardScaler().fit_transform(proteome),
        index=proteome.index,
        columns=proteome.columns
    )

    if method in {"pearson", "spearman", "kendall"}:
        corr = proteome_z.T.corr(method=method)
    elif method in {"cosine", "euclidean"}:
        dmat = squareform(pdist(proteome_z, metric=method))
        corr = pd.DataFrame(1 - dmat, index=proteome_z.index, columns=proteome_z.index)
    else:
        raise ValueError(f"Unknown method '{method}'.")

    ids = meta[id_col]
    within, between = [], []

    for i, j in combinations(corr.index, 2):
        if ids[i] == ids[j]:
            within.append(corr.loc[i, j])
        else:
            between.append(corr.loc[i, j])

    return {
        "within_mean": np.nanmean(within),
        "between_mean": np.nanmean(between)
    }
