import pandas as pd


def extract_counts(
    df: pd.DataFrame, 
    sample_name_col: str = "sample_name",
    group: str | None = None
) -> pd.DataFrame:
    """
    Count the number of detected proteins per sample and attach grouping info.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame with samples as rows (MultiIndex: metadata)
        and proteins as columns.
    sample_name_col : str, default="sample_name"
        Name of the index level corresponding to the sample identifier.
    group : str, optional
        Name of the index level or column to use for grouping.
        If None, no grouping variable is added.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - "sample_name": sample identifier
            - "proteins": number of detected (non-missing) proteins
            - <group>: grouping variable (optional)
    """
    protein_counts = df.notna().sum(axis=1)
    count_df = protein_counts.to_frame("proteins")

    # Always include sample name
    if sample_name_col in df.index.names:
        count_df["sample_name"] = df.index.get_level_values(sample_name_col)
    else:
        # Flatten MultiIndex and make sure it's a list of strings
        count_df["sample_name"] = [
            "_".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)
            for idx in df.index.to_flat_index()
        ]

    # Include group if provided
    if group is not None:
        if group in df.index.names:
            count_df[group] = df.index.get_level_values(group)
        elif group in df.columns:
            count_df[group] = df[group].values
        else:
            raise KeyError(f"Grouping variable '{group}' not found in index or columns.")

    return count_df.reset_index(drop=True)






def filter_samples_by_missingness(
    df: pd.DataFrame,
    k: float = 1.5
) -> pd.DataFrame:
    """
    Identify and remove samples with excessive missing values based on Tukey's rule.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame with samples as rows and features as columns.
    k : float, default=1.5
        Multiplier for the interquartile range (IQR) used to define the lower bound
        of acceptable missingness.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with samples exceeding the missingness threshold removed.
        Includes an additional index level named 'nan_fraction'.
    """

    non_nan_fraction = df.notna().mean(axis=1)
    Q1 = non_nan_fraction.quantile(0.25)
    Q3 = non_nan_fraction.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - k * IQR
    exclude = non_nan_fraction[non_nan_fraction < lower_bound]

    if not exclude.empty:
        print(f"Excluded samples ({len(exclude)} total):")
        for idx, val in exclude.items():
            if isinstance(idx, tuple) and "sample_name" in df.index.names:
                sample_name = idx[df.index.names.index("sample_name")]
            else:
                sample_name = idx
            print(f"  {sample_name}: non_nan_fraction = {val:.3f}")
    else:
        print("No samples excluded.")


    df = df.assign(non_nan_fraction=non_nan_fraction)
    df = df.set_index("non_nan_fraction", append=True)
    df = df.apply(pd.to_numeric, errors="coerce")

    return df.drop(index=exclude.index)




def filter_missingness(
    df: pd.DataFrame,
    feat_prevalence: float = 0.6,
    axis: int = 0
) -> pd.DataFrame:
    """
    Filter rows or columns of a DataFrame based on missing value prevalence.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to be filtered.
    feat_prevalence : float, default=0.6
        Minimum fraction of non-missing values required to retain a row or column.
    axis : int, default=0
        Axis along which to apply the filter.
        - 0: filter columns (features)
        - 1: filter rows (samples)

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only rows or columns meeting the prevalence threshold.
    """
    N = df.shape[axis]
    minimum_freq = N * feat_prevalence
    freq = df.notna().sum(axis=axis)
    mask = freq >= minimum_freq

    print(f"Dropped {(~mask).sum()} entries along axis {axis}.")

    if axis == 0:
        df = df.loc[:, mask]
    else:
        df = df.loc[mask, :]

    return df




def iter_filter_missingness(
    df: pd.DataFrame,
    feat_prevalence: float = 0.6,
    sample_prevalence: float = 0.6,
    max_iter: int = 10,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Iteratively filter a wide matrix by missingness until convergence or max iterations.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame with samples as rows and features as columns.
    feat_prevalence : float, default=0.6
        Minimum fraction of non-missing values required to keep a feature (column).
    sample_prevalence : float, default=0.6
        Minimum fraction of non-missing values required to keep a sample (row).
    max_iter : int, default=10
        Maximum number of alternating feature/sample filtering iterations.
    verbose : bool, default=True
        If True, print progress messages during iteration.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame that has converged (no further samples/features removed)
        or reached the maximum number of iterations.
    """
    df = df.copy()
    for it in range(1, max_iter + 1):
        prev_shape = df.shape

        df = filter_missingness(df, feat_prevalence=feat_prevalence, axis=0)
        if df.empty:
            if verbose:
                print(f"Iteration {it}: empty after column filter.")
            break

        df = filter_missingness(df, feat_prevalence=sample_prevalence, axis=1)
        if df.empty:
            if verbose:
                print(f"Iteration {it}: empty after row filter.")
            break

        if df.shape == prev_shape:
            if verbose:
                print(f"Converged in {it} iteration(s). Final shape: {df.shape}.")
            break
    else:
        if verbose:
            print(f"Reached max_iter={max_iter}. Final shape: {df.shape}.")

    return df
