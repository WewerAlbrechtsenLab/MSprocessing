import numpy as np
import pandas as pd
import warnings
import re

from statsmodels.formula.api import mixedlm, ols
from statsmodels.stats.multitest import multipletests


def filter_results(results_df, term):
    """
    Filter model results by term substring and sort by p-value.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing model results with at least the columns
        "term" and "pval".
    term : str
        Substring used to filter the "term" column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only rows where "term"
        includes the specified substring, sorted by ascending p-value
        and with index reset.
    """
    df = results_df[results_df["term"].str.contains(term, na=False)].copy()
    df = df.drop(columns=["term"])         
    df = df.sort_values("pval")             
    return df.reset_index(drop=True)        




def run_linear_model(proteome, meta, formula, group_col=None, adjust="fdr_bh", reml=True, filter_to=None):
    """
    Fit linear or mixed-effects models across all proteins, with automatic
    fallback to ordinary least squares (OLS) when no random-effects grouping
    structure is provided.

    Parameters
    ----------
    proteome : pd.DataFrame
        Wide-format DataFrame containing quantitative proteomic data.
        Rows represent samples, columns represent proteins.
    meta : pd.DataFrame
        Metadata table aligned with proteome samples. Must contain all variables
        referenced in the model formula.
    formula : str
        Statistical model formula (statsmodels syntax) where 'y' will be replaced
        by each protein's expression values. All variables on the right-hand side
        of '~' will be parsed and checked for missing values.
    group_col : str, optional
        Column in `meta` specifying the grouping factor for random effects in
        MixedLM models. If None, OLS regression is used for all proteins.
    adjust : str, default="fdr_bh"
        Multiple testing correction method passed to `multipletests`.
    reml : bool, default=True
        Whether to use restricted maximum likelihood (REML) for mixed models.
        Ignored when `group_col` is None.
    filter_to : str, optional
        Substring to filter output results by term name using `filter_results`.

    Returns
    -------
    pd.DataFrame
        Combined results across all proteins with columns:
        "protein", "term", "coef", "pval", and "padj".
        Index is set to "protein". If `filter_to` is provided, only matching
        terms are retained.
    """

    rhs = formula.split("~", 1)[1] if "~" in formula else ""
    tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", rhs))
    tokens = {t for t in tokens if t not in {"1"}}
    vars_in_meta = [t for t in tokens if t in meta.columns]

    if vars_in_meta:
        for v in vars_in_meta:
            meta = meta[meta[v].notna()]
        proteome = proteome.loc[meta.index]

    results = []

    common_idx = meta.index.intersection(proteome.index)
    meta = meta.loc[common_idx].copy()
    proteome = proteome.loc[common_idx].copy()

    if group_col is not None:
        if group_col not in meta.columns:
            raise ValueError(f"{group_col} not in meta columns")
        valid_groups = meta[group_col].value_counts()[lambda x: x > 1].index
        meta = meta[meta[group_col].isin(valid_groups)]
        proteome = proteome.loc[meta.index]

    meta = meta.astype(str)

    for protein in proteome.columns:
        df = meta.join(proteome[[protein]]).rename(columns={protein: "y"})

        if df["y"].isna().all() or df["y"].nunique(dropna=True) <= 1:
            results.append({"protein": protein, "term": None, "coef": np.nan, "pval": np.nan})
            continue

        try:
            if group_col is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = mixedlm(formula, df, groups=df[group_col], re_formula="1")
                    fit = m.fit(reml=reml, method="lbfgs", disp=False)
            else:
                fit = ols(formula, data=df).fit()

            for term, coef, pval in zip(fit.params.index, fit.params.values, fit.pvalues.values):
                results.append({"protein": protein, "term": term, "coef": coef, "pval": pval})

        except Exception:
            results.append({"protein": protein, "term": None, "coef": np.nan, "pval": np.nan})

    out = pd.DataFrame(results)
    if "term" in out.columns:
        out["padj"] = np.nan
        for term, sub in out.groupby("term"):
            mask = sub["pval"].notna()
            if mask.any():
                out.loc[sub.index[mask], "padj"] = multipletests(
                    sub.loc[mask, "pval"], method=adjust
                )[1]

    if filter_to:
        out = filter_results(out, filter_to)

    return out.set_index("protein")