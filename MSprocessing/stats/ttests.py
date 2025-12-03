import numpy as np
import pandas as pd

from scipy.stats import ttest_ind, ttest_rel
from statsmodels.stats.multitest import multipletests




def prep_paired(
    meta: pd.DataFrame,
    proteome: pd.DataFrame,
    pair_col: str,
    group_col: str,
    group1: str,
    group2: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare matched sample matrices for paired t-test.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata with sample indices and group/pairing information.
    proteome : pd.DataFrame
        Proteome data matrix indexed by sample, columns = proteins.
    pair_col : str
        Column identifying paired samples (e.g. subject ID).
    group_col : str
        Column identifying groups to compare.
    group1, group2 : str
        Group labels to compare (Y - X).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        X : matrix of group1 samples (n_subjects × n_proteins)
        Y : matrix of group2 samples (n_subjects × n_proteins)
        proteins : protein names (np.ndarray)
    """
    m = meta[[pair_col, group_col]].dropna()
    m = m[m[group_col].isin([group1, group2])]
    m = m.drop_duplicates(subset=[pair_col, group_col], keep="first")

    # Pivot so each subject has one sample per group
    wide = m.reset_index().pivot(index=pair_col, columns=group_col, values=meta.index.name)
    wide = wide.dropna(subset=[group1, group2])

    idx1 = wide[group1].astype(str)
    idx2 = wide[group2].astype(str)

    X = proteome.loc[idx1].to_numpy(float)
    Y = proteome.loc[idx2].to_numpy(float)
    proteins = proteome.columns.to_numpy()

    return X, Y, proteins



def prep_ind(
    meta: pd.DataFrame,
    proteome: pd.DataFrame,
    group_col: str,
    group1: str,
    group2: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare independent sample matrices for unpaired t-tests.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata with sample indices and group information.
    proteome : pd.DataFrame
        Proteome data matrix indexed by sample, columns = proteins.
    group_col : str
        Column in `meta` specifying group membership.
    group1, group2 : str
        Group labels to compare (Y - X).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        X : matrix of group1 samples (n_samples1 × n_proteins)
        Y : matrix of group2 samples (n_samples2 × n_proteins)
        proteins : protein names (np.ndarray)
    """
    idx1 = meta.index[meta[group_col] == group1]
    idx2 = meta.index[meta[group_col] == group2]

    X = proteome.loc[idx1].to_numpy(float)
    Y = proteome.loc[idx2].to_numpy(float)
    proteins = proteome.columns.to_numpy()

    return X, Y, proteins




def prep_deltas(
    meta: pd.DataFrame,
    proteome: pd.DataFrame,
    subject_col: str,
    delta_col: str,
    delta_group1: str,
    delta_group2: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare subject-level deltas for a delta t-test:
    computes (delta_group2 - delta_group1) per subject.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata with sample indices and subject/timepoint information.
    proteome : pd.DataFrame
        Proteome matrix indexed by sample, columns = proteins.
    subject_col : str
        Column identifying subjects (pairing variable).
    delta_col : str
        Column identifying repeated measures (e.g. timepoint).
    delta_group1, delta_group2 : str
        Levels of `delta_col` to subtract (group2 - group1).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        delta_proteome : subject-level differences.
        delta_meta : metadata aligned to delta_proteome.
    """
    m = meta[[subject_col, delta_col]].dropna()
    m = m[m[delta_col].isin([delta_group1, delta_group2])]
    m = m.drop_duplicates(subset=[subject_col, delta_col], keep="first")

    wide = m.reset_index().pivot(index=subject_col, columns=delta_col, values=meta.index.name)
    wide = wide.dropna(subset=[delta_group1, delta_group2])

    idx1 = wide[delta_group1].astype(str)
    idx2 = wide[delta_group2].astype(str)

    delta_matrix = proteome.loc[idx2].to_numpy(float) - proteome.loc[idx1].to_numpy(float)
    delta_proteome = pd.DataFrame(delta_matrix, index=wide.index, columns=proteome.columns)

    delta_meta = meta.loc[idx2].copy()
    delta_meta.index = wide.index  # align subject index

    return delta_proteome, delta_meta







def ttests(
    proteome: pd.DataFrame,
    meta: pd.DataFrame,
    group_col: str,
    group1: str,
    group2: str,
    method: str,
    pair_col: str | None = None,
    delta_col: str | None = None,
    delta_group1: str | None = None,
    delta_group2: str | None = None,
) -> pd.DataFrame:
    """
    Run group-wise t-tests (independent, paired, or delta).

    Parameters
    ----------
    proteome : pd.DataFrame
        Proteome data matrix.
    meta : pd.DataFrame
        Sample metadata.
    group_col : str
        Column identifying groups.
    group1, group2 : str
        Group labels to compare.
    method : str
        Statistical method ('student_ttest', 'welch_ttest', 'paired', 'delta').
    pair_col : str, optional
        Subject pairing column for paired tests.
    delta_col, delta_group1, delta_group2 : optional
        Delta parameters for repeated measures (if applicable).
    """
    if method in ["student_ttest", "welch_ttest"]:
        X, Y, proteins = prep_ind(meta, proteome, group_col, group1, group2)
        equal_var = (method == "student_ttest")
        _, pvals = ttest_ind(X, Y, equal_var=equal_var, nan_policy="omit")
        log2fc = np.nanmean(X, axis=0) - np.nanmean(Y, axis=0)

    elif method == "paired":
        if pair_col is None:
            raise ValueError("Paired t-test requires 'pair_col'.")
        X, Y, proteins = prep_paired(meta, proteome, pair_col, group_col, group1, group2)
        _, pvals = ttest_rel(X, Y, nan_policy="omit")
        log2fc = np.nanmean(X - Y, axis=0)

    elif method == "delta":
        proteome, meta = prep_deltas(
            meta=meta,
            proteome=proteome,
            subject_col=pair_col,
            delta_col=delta_col,
            delta_group1=delta_group1,
            delta_group2=delta_group2,
        )
        return ttests(proteome, meta, group_col, group1, group2, "student_ttest")

    else:
        raise ValueError(f"Unsupported method: {method}")

    return pd.DataFrame({
        "protein": proteins,
        "log2fc": log2fc,
        "pval": pvals,
        "padj": np.nan
    })






def perm_meta_paired(meta, pair_col, group_col, group1, group2, rng):
    """
    Create a paired-design permutation by randomly regrouping samples
    into new pairs and assigning one sample per pair to each group.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata with grouping information.
    pair_col : str
        Parameter for paired or delta designs.
    group_col : str
        Column defining experimental groups.
    group1, group2 : str
        Group labels to assign.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    pd.DataFrame
        Permuted metadata with new pairs and reassigned groups.
    """
    m = meta.copy()
    valid = m.groupby(pair_col).filter(lambda x: len(x) == 2).index.tolist()

    shuffled_idx = rng.permutation(valid)
    new_pairs = [shuffled_idx[i:i+2] for i in range(0, len(shuffled_idx), 2)]

    # assign new synthetic subject IDs
    for new_subj_id, pair_rows in enumerate(new_pairs, start=1):
        for row in pair_rows:
            m.at[row, pair_col] = f"perm_subj_{new_subj_id}"

        if len(pair_rows) == 2:
            if rng.random() < 0.5:
                m.at[pair_rows[0], group_col] = group1
                m.at[pair_rows[1], group_col] = group2
            else:
                m.at[pair_rows[0], group_col] = group2
                m.at[pair_rows[1], group_col] = group1

    return m





def resampling_adjust(
    proteome: pd.DataFrame,
    meta: pd.DataFrame,
    group_col: str,
    group1: str,
    group2: str,
    method: str,
    adjust: str,
    n_perm: int,
    df: pd.DataFrame,   
    pair_col: str | None = None,
    delta_col: str | None = None,
    delta_group1: str | None = None,
    delta_group2: str | None = None,
) -> np.ndarray:
    """
    Perform permutation-based multiple testing correction.

    Parameters
    ----------
    proteome : pd.DataFrame
        Proteome data matrix.
    meta : pd.DataFrame
        Metadata with grouping information.
    group_col : str
        Column defining experimental groups.
    group1, group2 : str
        Group labels to compare.
    method : str
        Statistical testing method.
    adjust : str
        Type of permutation adjustment ('perm' or 'stepdown_perm').
    n_perm : int
        Number of label permutations.
    df : pd.DataFrame
        Observed results table from t-tests (must include 'pval').
    pair_col, delta_col, delta_group1, delta_group2 : optional
        Optional parameters for paired or delta designs.

    Returns
    -------
    np.ndarray
        Array of adjusted p-values.
    """
    if method == "perm_test":
        raise ValueError("Resampling p pvalue adjustment not available for permmutation test")    
    rng = np.random.default_rng(seed=0)
    labels = meta[group_col].values
    n_proteins = df.shape[0]
    obs_pvals = df["pval"].values

    perm_pvals = np.zeros((n_perm, n_proteins))

    for i in range(n_perm):
        
        if method == "paired":
            perm_meta = perm_meta_paired(
                meta=meta,
                pair_col=pair_col,
                group_col=group_col,
                group1=group1,
                group2=group2,
                rng=rng
            )

        else:
            perm_meta = meta.copy()
            perm_meta[group_col] = rng.permutation(meta[group_col].values)

        try:
            perm_df = ttests(
                proteome=proteome,
                meta=perm_meta,
                group_col=group_col,
                group1=group1,
                group2=group2,
                method=method,
                pair_col=pair_col,
                delta_col=delta_col,
                delta_group1=delta_group1,
                delta_group2=delta_group2,
            )
            perm_pvals[i, :] = perm_df["pval"].values
        except KeyError:
            perm_pvals[i, :] = np.nan
            continue

    if adjust == "perm":
        min_p = perm_pvals.min(axis=1) 
        padj = np.array([(np.sum(min_p <= p) + 1) / (n_perm + 1) for p in obs_pvals])

    elif adjust == "stepdown_perm":
        order = np.argsort(obs_pvals)
        sorted_p = obs_pvals[order]
        perm_sorted = perm_pvals[:, order]

        padj_sorted = np.zeros_like(sorted_p, dtype=float)

        for i, p in enumerate(sorted_p):
            perm_min = np.min(perm_sorted[:, : i+1], axis=1)
            padj_sorted[i] = (np.sum(perm_min <= p) + 1) / (n_perm + 1)

        padj_sorted = np.maximum.accumulate(padj_sorted)

        padj = np.empty_like(padj_sorted)
        padj[order] = padj_sorted
    else:
        raise ValueError(f"Unsupported resampling adjustment: {adjust}")

    return padj





def run_dea(
    proteome: pd.DataFrame,
    meta: pd.DataFrame,
    group_col: str,
    group1: str,
    group2: str,
    adjust: str = "fdr_bh",
    method: str = "ttest",
    n_perm: int = 10000,
    pairing: str | None = None,
    delta_col: str | None = None,
    delta_group1: str | None = None,
    delta_group2: str | None = None,
) -> pd.DataFrame:
    """
    Perform differential expression analysis (DEA).

    Executes statistical comparisons between experimental groups
    using t-tests with optional paired or delta designs and
    multiple-testing correction.

    Parameters
    ----------
    proteome : pd.DataFrame
        Quantitative proteomic data (samples × proteins).
    meta : pd.DataFrame
        Metadata describing sample annotations.
    group_col : str
        Column identifying groups to compare.
    group1, group2 : str
        Group labels to contrast.
    adjust : str, default="fdr_bh"
        Multiple-testing correction method (e.g. "fdr_bh", "perm").
    method : str, default="ttest"
        Type of statistical test to apply.
    n_perm : int, default=10000
        Number of permutations for resampling-based adjustments.
    pairing : str, optional
        Subject pairing variable for paired tests.
    delta_col, delta_group1, delta_group2 : optional
        Delta analysis parameters for repeated-measures data.

    Returns
    -------
    pd.DataFrame
        DataFrame of differential expression results including
        log2 fold changes, raw and adjusted p-values.
    """
    common = meta.index.intersection(proteome.index)
    meta = meta.loc[common]
    proteome = proteome.loc[common]

    df = ttests(
        proteome=proteome,
        meta=meta,
        group_col=group_col,
        group1=group1,
        group2=group2,
        method=method,
        pair_col=pairing,
        delta_col=delta_col,
        delta_group1=delta_group1,
        delta_group2=delta_group2,
    )

    mask = df["pval"].notna()
    if mask.any():
        if adjust in ["perm", "stepdown_perm"]:
            df.loc[mask, "padj"] = resampling_adjust(
                proteome=proteome,
                meta=meta,
                group_col=group_col,
                group1=group1,
                group2=group2,
                method=method,
                adjust=adjust,
                n_perm=n_perm,
                df=df,
                pair_col=pairing,
                delta_col=delta_col,
                delta_group1=delta_group1,
                delta_group2=delta_group2,
            )
        else:
            df.loc[mask, "padj"] = multipletests(df.loc[mask, "pval"], method=adjust)[1]

    return df.set_index("protein").sort_values("pval")







