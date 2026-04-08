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



def convert_ids_from_mapping(
    df: pd.DataFrame,
    mapping_file: str,
    from_col: str,
    to_col: str,
    axis: int = 1,
    sep: str = "\t",
) -> pd.DataFrame:
    """
    Replace row or column identifiers using a two-column mapping file.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame, DiaNN output.
    mapping_file : str
        Path to mapping file.
    from_col : str
        Column name in the mapping file containing the current identifiers.
    to_col : str
        Column name in the mapping file containing the replacement identifiers.
    axis : int, default=1
        1 = replace column names, 0 = replace row names.
    sep : str, default="\\t"
        Field separator for the mapping file.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed rows or columns.
    """

    map_genes = pd.read_csv(
        mapping_file,
        sep=sep,
        usecols=[from_col, to_col],
    )
    map_genes = dict(zip(map_genes["Protein.Group"], map_genes["Genes"]))

    if axis == 1:
        return df.rename(columns=map_genes)
    else:
        return df.rename(index=map_genes)


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


import pandas as pd




def group_table(
    df,
    group,
    summary_cols=None,
    stat="median",
    q=(0.25, 0.75),
):
    """
    Build a summary table with one row per group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group : str or list[str]
        Column(s) defining groups.
    summary_cols : list[str] or None
        Columns to summarize. Numeric columns are reported as median or mean
        (with optional quantiles). Non-numeric columns are reported as
        category proportions in a single cell.
    stat : {"median", "mean"}, default="median"
        Summary statistic for numeric columns.
    q : tuple[float, float] or None, default=(0.25, 0.75)
        Quantiles for numeric columns. If None, only the summary statistic is shown.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per group and an "N" column with counts
        and cohort percentages.
    """
    summary_cols = summary_cols or []

    if isinstance(group, str):
        group = [group]

    group_col_name = ", ".join(group)

    data = df.copy().dropna(subset=group)
    total_n = len(data)

    grouped = data.groupby(group)
    counts = grouped.size()

    category_orders = {}
    for col in summary_cols:
        if not pd.api.types.is_numeric_dtype(data[col]):
            category_orders[col] = (
                data[col].dropna().astype(str).value_counts(sort=False).index.tolist()
            )

    rows = []

    for keys, n in counts.items():
        if not isinstance(keys, tuple):
            keys = (keys,)

        label = ", ".join(map(str, keys))
        subset = grouped.get_group(keys if len(group) > 1 else keys[0])

        row = {group_col_name: label}
        row["N"] = f"{n} ({100 * n / total_n:.1f}%)"

        for col in summary_cols:
            s = subset[col]

            if pd.api.types.is_numeric_dtype(data[col]):
                s_num = pd.to_numeric(s, errors="coerce").dropna()

                if len(s_num) == 0:
                    row[col] = ""
                    continue

                lo = s_num.quantile(q[0])
                hi = s_num.quantile(q[1])

                if stat == "median":
                    center = s_num.median()
                elif stat == "mean":
                    center = s_num.mean()
                else:
                    raise ValueError('stat must be "median" or "mean"')

                row[col] = f"{center:.1f} ({lo:.1f}, {hi:.1f})"

            else:
                s_cat = s.astype("object")
                denom = len(s_cat)

                if denom == 0:
                    row[col] = ""
                    continue

                counts_cat = s_cat.dropna().astype(str).value_counts()

                parts = []
                for cat in category_orders[col]:
                    k = counts_cat.get(cat, 0)
                    parts.append(f"{cat} ({100 * k / denom:.1f}%)")

                row[col] = ", ".join(parts)

        rows.append(row)

    out = pd.DataFrame(rows)
    desired_cols = [group_col_name, "N"] + summary_cols
    return out.reindex(columns=desired_cols)