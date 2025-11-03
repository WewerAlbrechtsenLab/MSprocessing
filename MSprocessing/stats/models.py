import numpy as np
import pandas as pd
import warnings

from statsmodels.formula.api import mixedlm, ols
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegressionCV


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


def run_mixedlm(proteome, meta, formula, group_col=None, adjust="fdr_bh", reml=True, filter_to=None):
    """
    Fit linear mixed-effects models (MixedLM) across all proteins, with automatic
    fallback to ordinary least squares (OLS) when a valid random-effects grouping
    structure is absent.

    The dependent variable 'y' in the provided `formula` is dynamically replaced
    by each protein column in the proteome DataFrame.

    Parameters
    ----------
    proteome : pd.DataFrame
        Wide-format DataFrame containing quantitative proteomic data.
        Rows represent samples, columns represent proteins.
    meta : pd.DataFrame
        Metadata table aligned with proteome samples. Must contain grouping
        variables referenced in the model formula.
    formula : str
        Statistical model formula (statsmodels syntax) where 'y' will be replaced
        by each protein's expression values.
    group_col : str, optional
        Column in `meta` specifying the grouping factor for random effects.
        If None, OLS regression is used.
    adjust : str, default="fdr_bh"
        Multiple testing correction method passed to `multipletests`.
    reml : bool, default=True
        Whether to use restricted maximum likelihood (REML) for mixed models.
    filter_to : str, optional
        Substring to filter output results by term name using `filter_results`.

    Returns
    -------
    pd.DataFrame
        Combined results across all proteins with columns:
        "protein", "term", "coef", "pval", and "padj".
        If `filter_to` is provided, only matching terms are retained.
    """
    valid_groups = meta[group_col].value_counts()[lambda x: x > 1].index
    meta = meta[meta[group_col].isin(valid_groups)]
    proteome = proteome.loc[meta.index]
    meta = meta.astype(str) 
    results = []

    if group_col not in meta.columns:
        raise ValueError(f"{group_col} not in meta columns")
    
    if group_col is not None:
        grp_counts = meta[group_col].value_counts()
        has_replication = (grp_counts >= 2).sum() >= 1
    else:
        has_replication = False

    for protein in proteome.columns:
        df = meta.join(proteome[[protein]]).rename(columns={protein: "y"})

        if df["y"].isna().all() or df["y"].nunique(dropna=True) <= 1:
            results.append({"protein": protein, "term": None, "coef": np.nan, "pval": np.nan})
            continue

        try:
            if group_col and has_replication:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = mixedlm(formula, df, groups=df[group_col], re_formula="1")
                    fit = m.fit(reml=reml, method="lbfgs", disp=False)

                gv = np.ravel(fit.cov_re)[0] if fit.cov_re.size else 0.0
                if not np.isfinite(gv) or gv <= 1e-8:
                    print(f"Defaulted to OLS for {protein}")
                    fit = ols(formula, data=df).fit()
            else:
                print(f"Defaulted to OLS for {protein}")
                fit = ols(formula, data=df).fit()

            for term, coef, pval in zip(fit.params.index, fit.params.values, fit.pvalues.values):
                results.append({"protein": protein, "term": term, "coef": coef, "pval": pval})

        except Exception as e:
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
