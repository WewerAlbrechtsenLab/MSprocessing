import math
import warnings
import numpy as np
import pandas as pd
import os 

from concurrent.futures import ProcessPoolExecutor
from scipy.stats import chi2
from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway
from tqdm.auto import tqdm


def _build_formula(group_col, covariates=None):
    terms = [f"C({group_col})"]
    if covariates is not None:
        terms += list(covariates)
    return "y ~ " + " + ".join(terms)


def _prepare_multigroup_inputs(proteome, meta, group_col, covariates=None, pairing=None):
    covariates = [] if covariates is None else list(covariates)

    needed = [group_col] + covariates
    if pairing is not None:
        needed.append(pairing)

    missing = [c for c in needed if c not in meta.columns]
    if missing:
        raise ValueError(f"Columns not in meta: {missing}")

    common = meta.index.intersection(proteome.index)
    meta2 = meta.loc[common].copy()
    proteome2 = proteome.loc[common].copy()

    keep = pd.Series(True, index=meta2.index)
    for col in needed:
        keep &= meta2[col].notna()

    meta2 = meta2.loc[keep]
    proteome2 = proteome2.loc[meta2.index]

    valid_groups = meta2[group_col].value_counts()
    valid_groups = valid_groups[valid_groups > 1].index
    meta2 = meta2[meta2[group_col].isin(valid_groups)]
    proteome2 = proteome2.loc[meta2.index]

    if meta2[group_col].nunique() < 2:
        raise ValueError("At least two groups with at least two samples each are required.")

    return proteome2, meta2


def _split_columns(proteome, n_jobs, chunk_size=None):
    cols = list(proteome.columns)
    if not cols:
        return []

    if chunk_size is None:
        chunk_size = math.ceil(len(cols) / n_jobs)

    return [cols[i:i + chunk_size] for i in range(0, len(cols), chunk_size)]


def _fit_one_protein_omnibus(
    y,
    meta,
    group_col,
    method,
    covariates=None,
    pairing=None,
    reml=False,
):
    df = meta.copy()
    df["y"] = y

    covariates = [] if covariates is None else list(covariates)
    needed = [group_col] + covariates
    if pairing is not None:
        needed.append(pairing)

    df = df.dropna(subset=["y"] + needed)

    if df["y"].nunique(dropna=True) <= 1:
        return np.nan, np.nan, np.nan, np.nan, len(df), df[group_col].nunique(), {}

    if df[group_col].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan, len(df), df[group_col].nunique(), {}

    effects = _effect_summary(df, group_col)

    if method == "anova":
        formula = _build_formula(group_col, covariates=None)
        fit = ols(formula, data=df).fit()
        tab = anova_lm(fit, typ=2)
        row = f"C({group_col})"

        ss_group = tab.loc[row, "sum_sq"]
        ss_resid = tab.loc["Residual", "sum_sq"]

        effects["partial_eta_sq"] = ss_group / (ss_group + ss_resid)

        return (
            tab.loc[row, "F"],
            tab.loc[row, "PR(>F)"],
            tab.loc[row, "df"],
            fit.df_resid,
            len(df),
            df[group_col].nunique(),
            effects,
        )

    if method == "ancova":
        formula = _build_formula(group_col, covariates=covariates)
        fit = ols(formula, data=df).fit()
        tab = anova_lm(fit, typ=2)
        row = f"C({group_col})"

        ss_group = tab.loc[row, "sum_sq"]
        ss_resid = tab.loc["Residual", "sum_sq"]

        effects["partial_eta_sq"] = ss_group / (ss_group + ss_resid)

        return (
            tab.loc[row, "F"],
            tab.loc[row, "PR(>F)"],
            tab.loc[row, "df"],
            fit.df_resid,
            len(df),
            df[group_col].nunique(),
            effects,
        )

    if method == "welch_anova":
        groups = [
            g["y"].to_numpy(dtype=float)
            for _, g in df.groupby(group_col)
        ]
        res = anova_oneway(groups, use_var="unequal")

        effects["partial_eta_sq"] = np.nan

        return (
            res.statistic,
            res.pvalue,
            res.df[0],
            res.df[1],
            len(df),
            df[group_col].nunique(),
            effects,
        )

    if method == "welch_ancova":
        formula = _build_formula(group_col, covariates=covariates)

        fit = ols(formula, data=df).fit(cov_type="HC3")
        row = f"C({group_col})"

        wt = fit.wald_test_terms(skip_single=False).table
        stat = float(wt.loc[row, "statistic"])
        pval = float(wt.loc[row, "pvalue"])
        df_num = float(wt.loc[row, "df_constraint"])

        effects["partial_eta_sq"] = np.nan

        return (
            stat,
            pval,
            df_num,
            np.nan,
            len(df),
            df[group_col].nunique(),
            effects,
        )

    if method == "mixed_lm":
        if pairing is None:
            raise ValueError("mixed_lm requires `pairing`.")

        if reml:
            raise ValueError("mixed_lm likelihood-ratio tests require reml=False.")

        full_formula = _build_formula(group_col, covariates=covariates)

        reduced_terms = list(covariates)
        if reduced_terms:
            reduced_formula = "y ~ " + " + ".join(reduced_terms)
        else:
            reduced_formula = "y ~ 1"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            full = mixedlm(
                full_formula,
                data=df,
                groups=df[pairing],
                re_formula="1",
            ).fit(reml=False, method="lbfgs", disp=False)

            reduced = mixedlm(
                reduced_formula,
                data=df,
                groups=df[pairing],
                re_formula="1",
            ).fit(reml=False, method="lbfgs", disp=False)

        stat = 2 * (full.llf - reduced.llf)
        df_num = full.df_modelwc - reduced.df_modelwc
        pval = chi2.sf(stat, df_num)

        effects["partial_eta_sq"] = np.nan

        return (
            stat,
            pval,
            df_num,
            np.nan,
            len(df),
            df[group_col].nunique(),
            effects,
        )

    raise ValueError(f"Unsupported method: {method}")


def _multigroup_tests_chunk(
    proteome_chunk,
    meta,
    group_col,
    method,
    covariates=None,
    pairing=None,
    reml=False,
):
    results = []

    for protein in proteome_chunk.columns:
        try:
            stat, pval, df_num, df_denom, n, n_groups, effects = _fit_one_protein_omnibus(
                y=proteome_chunk[protein],
                meta=meta,
                group_col=group_col,
                method=method,
                covariates=covariates,
                pairing=pairing,
                reml=reml,
            )

        except Exception:
            print(f"Failed for protein: {protein}")
            stat = pval = df_num = df_denom = np.nan
            n = meta.shape[0]
            n_groups = meta[group_col].nunique(dropna=True)
            effects = {}

        results.append({
            "protein": protein,
            "stat": stat,
            "pval": pval,
            "padj": np.nan,
            "df_num": df_num,
            "df_denom": df_denom,
            "n": n,
            "n_groups": n_groups,
            "grand_mean": effects.get("grand_mean", np.nan),
            "mean_min": effects.get("mean_min", np.nan),
            "mean_max": effects.get("mean_max", np.nan),
            "mean_range": effects.get("mean_range", np.nan),
            "partial_eta_sq": effects.get("partial_eta_sq", np.nan),
        })

    return pd.DataFrame(results)


def _check_n_jobs(n_jobs):
    if n_jobs is None:
        return 1

    if not isinstance(n_jobs, int):
        raise ValueError("n_jobs must be an integer.")

    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1.")

    max_cpu = os.cpu_count() or 1

    if n_jobs > max_cpu:
        raise ValueError(f"n_jobs={n_jobs} exceeds available CPU cores: {max_cpu}.")

    return n_jobs


def multigroup_tests(
    proteome,
    meta,
    group_col,
    method,
    covariates=None,
    pairing=None,
    reml=False,
    n_jobs=1,
    chunk_size=None,
):
    if n_jobs is None or n_jobs < 2 or proteome.shape[1] < 2:
        return _multigroup_tests_chunk(
            proteome_chunk=proteome,
            meta=meta,
            group_col=group_col,
            method=method,
            covariates=covariates,
            pairing=pairing,
            reml=reml,
        )

    chunks = _split_columns(proteome, n_jobs=n_jobs, chunk_size=chunk_size)
    outputs = []

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [
            ex.submit(
                _multigroup_tests_chunk,
                proteome.loc[:, cols],
                meta,
                group_col,
                method,
                covariates,
                pairing,
                reml,
            )
            for cols in chunks
        ]

        for fut in futures:
            outputs.append(fut.result())

    return pd.concat(outputs, ignore_index=True)


def _permute_meta(meta, cols, rng):
    m = meta.copy()
    cols = [cols] if isinstance(cols, str) else list(cols)

    perm = rng.permutation(len(m))
    m.loc[:, cols] = m[cols].to_numpy()[perm]

    return m


def _effect_summary(df, group_col):
    group_means = df.groupby(group_col)["y"].mean()
    grand_mean = df["y"].mean()

    return {
        "mean_min": group_means.min(),
        "mean_max": group_means.max(),
        "mean_range": group_means.max() - group_means.min(),
        "grand_mean": grand_mean,
    }
    


def resampling_adjust_multigroup(
    proteome,
    meta,
    group_col,
    method,
    adjust,
    n_perm,
    df,
    covariates=None,
    pairing=None,
    permute_cols=None,
    seed=0,
    reml=False,
    n_jobs=1,
    chunk_size=None,
):
    if permute_cols is None:
        permute_cols = group_col

    rng = np.random.default_rng(seed)
    obs_pvals = df["pval"].to_numpy(dtype=float)
    n_proteins = df.shape[0]

    perm_pvals = np.full((n_perm, n_proteins), np.nan, dtype=float)

    bar = tqdm(
        range(n_perm),
        total=n_perm,
        desc="Permutations",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for i in bar:
        perm_meta = _permute_meta(meta, permute_cols, rng)

        perm_df = multigroup_tests(
            proteome=proteome,
            meta=perm_meta,
            group_col=group_col,
            method=method,
            covariates=covariates,
            pairing=pairing,
            reml=reml,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
        )

        perm_pvals[i, :] = perm_df.set_index("protein").reindex(df["protein"])["pval"].values

    keep = ~np.isnan(obs_pvals)
    padj = np.full_like(obs_pvals, np.nan, dtype=float)

    if not keep.any():
        return padj

    obs = obs_pvals[keep]
    perms = perm_pvals[:, keep]

    if adjust == "perm":
        min_p = np.nanmin(perms, axis=1)
        padj[keep] = np.array([
            (np.sum(min_p <= p) + 1) / (n_perm + 1)
            for p in obs
        ])

    elif adjust == "stepdown_perm":
        order = np.argsort(obs)
        sorted_p = obs[order]
        perm_sorted = perms[:, order]

        padj_sorted = np.zeros_like(sorted_p, dtype=float)

        for i, p in enumerate(sorted_p):
            perm_min = np.nanmin(perm_sorted[:, i:], axis=1)
            padj_sorted[i] = (np.sum(perm_min <= p) + 1) / (n_perm + 1)

        padj_sorted = np.maximum.accumulate(padj_sorted)

        out = np.empty_like(padj_sorted)
        out[order] = padj_sorted
        padj[keep] = out

    else:
        raise ValueError(f"Unsupported resampling adjustment: {adjust}")

    return padj





def run_multigroup(
    proteome,
    meta,
    group_col,
    method="anova",
    covariates=None,
    pairing=None,
    adjust="fdr_bh",
    n_perm=1000,
    permute_cols=None,
    seed=0,
    reml=False,
    n_jobs=1,
    chunk_size=None,
):
    """
    Run protein-wise omnibus multi-group tests.

    Parameters
    ----------
    proteome : pd.DataFrame
        Samples x proteins matrix.
    meta : pd.DataFrame
        Sample metadata.
    group_col : str
        Main categorical variable to test.
    method : str
        One of: "anova", "ancova", "welch_anova", "mixed_lm".
    covariates : list[str], optional
        Covariates for ANCOVA or mixed linear model.
    pairing : str, optional
        Subject/grouping column for mixed_lm.
    adjust : str
        Any statsmodels multipletests method, or "perm", "stepdown_perm".
    n_perm : int
        Number of permutations for resampling adjustment.
    permute_cols : str or list[str], optional
        Columns to permute. Defaults to group_col.
    seed : int
        Random seed.
    reml : bool
        REML setting for mixed_lm. For likelihood-ratio tests, False is recommended.
    n_jobs : int
        Number of worker processes.
    chunk_size : int, optional
        Protein chunk size for parallel processing.

    Returns
    -------
    pd.DataFrame
        Protein-indexed omnibus result table.
    """
    valid_methods = ["anova", "ancova", "welch_anova", "welch_ancova", "mixed_lm"]
    
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}")
    elif method == "anova" and covariates:
        raise ValueError("anova does not use covariates. Use method='ancova' instead.")
    elif method == "welch_anova" and covariates:
        raise ValueError("welch_anova does not use covariates. Use method='welch_ancova' instead.")
    elif method == "ancova" and not covariates:
        raise ValueError("ancova requires `covariates`.")
    elif method == "welch_ancova" and not covariates:
        raise ValueError("welch_ancova requires `covariates`.")
    elif method == "mixed_lm" and pairing is None:
        raise ValueError("mixed_lm requires `pairing`.")
    elif method == "mixed_lm" and reml:
        raise ValueError("mixed_lm likelihood-ratio tests require reml=False.")
    
    n_jobs = _check_n_jobs(n_jobs)

    proteome2, meta2 = _prepare_multigroup_inputs(
        proteome=proteome,
        meta=meta,
        group_col=group_col,
        covariates=covariates,
        pairing=pairing,
    )

    df = multigroup_tests(
        proteome=proteome2,
        meta=meta2,
        group_col=group_col,
        method=method,
        covariates=covariates,
        pairing=pairing,
        reml=reml,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
    )

    mask = df["pval"].notna()

    if mask.any():
        if adjust in ["perm", "stepdown_perm"]:
            df.loc[mask, "padj"] = resampling_adjust_multigroup(
                proteome=proteome2,
                meta=meta2,
                group_col=group_col,
                method=method,
                adjust=adjust,
                n_perm=n_perm,
                df=df,
                covariates=covariates,
                pairing=pairing,
                permute_cols=permute_cols,
                seed=seed,
                reml=reml,
                n_jobs=n_jobs,
                chunk_size=chunk_size,
            )[mask.to_numpy()]

        else:
            df.loc[mask, "padj"] = multipletests(
                df.loc[mask, "pval"],
                method=adjust,
            )[1]

    return df.set_index("protein").sort_values("pval")