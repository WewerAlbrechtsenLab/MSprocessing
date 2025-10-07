import re
import numpy as np
import pandas as pd
import warnings
import plotly.express as px

from scipy.stats import norm, ttest_ind, ttest_rel, permutation_test
from statsmodels.formula.api import mixedlm, ols
from statsmodels.stats.multitest import multipletests
from alphastats.dataset.keys import Cols
from alphastats.plots.volcano_plot import VolcanoPlot
from alphastats.dataset.preprocessing import PreprocessingStateKeys as PSK
from alphastats.statistics.differential_expression_analysis import DifferentialExpressionAnalysis
from sklearn.linear_model import LogisticRegressionCV
from typing import Dict, List, Optional, Tuple, Union
from gprofiler import GProfiler



#monkeypatched deprecated numpy aliases 
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool




def split_proteome_meta(df: pd.DataFrame, index_col: str):
    """
    df: wide DataFrame with samples in the MultiIndex (metadata levels) and proteins as columns.
    Assumes metadata are exactly the index levels.
    """
    index_cols = list(df.index.names)
    tmp = df.reset_index()
    protein_cols = [c for c in tmp.columns if c not in index_cols]
    mat = tmp.set_index(index_col)[protein_cols].copy()
    meta = tmp[index_cols].set_index(index_col).copy()

    return mat, meta




def prep_for_paired_ttest(
    meta: pd.DataFrame,
    proteome: pd.DataFrame,
    pair_col: str,      #subject ID column in meta
    variable: str,      #condition column in meta (e.g., timepoint, group)
    group1: str,
    group2: str,
):
    m = (meta.reset_index()[[Cols.SAMPLE, pair_col, variable]]
              .dropna(subset=[pair_col, variable])
              .drop_duplicates())
    m = (m.sort_values([Cols.SAMPLE])
           .drop_duplicates(subset=[pair_col, variable], keep="first"))

    wide = m.pivot(index=pair_col, columns=variable, values=Cols.SAMPLE)
    wide = wide.dropna(subset=[group1, group2])

    g1 = wide[group1].tolist()
    g2 = wide[group2].tolist()

    meta_out = pd.DataFrame({
        Cols.SAMPLE: g1 + g2,
        variable: [group1]*len(g1) + [group2]*len(g2),
    })
    proteome_out = proteome.loc[meta_out[Cols.SAMPLE]]
    return proteome_out, meta_out



def prep_deltas(
    meta: pd.DataFrame,
    proteome: pd.DataFrame,
    *,
    subject_col: str,    # subject ID column
    delta_col: str,      # column with timepoint labels
    delta_group1: str,   # baseline (or earlier timepoint)
    delta_group2: str,   # follow-up (later timepoint)
):
    """
    Prepare proteome and metadata for paired t-test by computing
    subject-level deltas (delta_group2 - delta_group1) for each protein.
    """
    m = (
        meta.reset_index()[[Cols.SAMPLE, subject_col, delta_col]]
        .dropna(subset=[Cols.SAMPLE, subject_col, delta_col])
        .drop_duplicates(subset=[subject_col, delta_col], keep="first")
    )

    wide = m.pivot(index=subject_col, columns=delta_col, values=Cols.SAMPLE)
    wide = wide.dropna(subset=[delta_group1, delta_group2])

    g1_samples = wide[delta_group1].tolist()
    g2_samples = wide[delta_group2].tolist()

    delta_matrix = (
        proteome.loc[g2_samples].values - proteome.loc[g1_samples].values
    )

    proteome_delta = pd.DataFrame(
        delta_matrix,
        index=wide.index,      
        columns=proteome.columns,
    )

    meta_delta = meta.reset_index().set_index(Cols.SAMPLE).loc[g2_samples].copy()
    meta_delta.index = wide.index 

    return proteome_delta, meta_delta






def ttests(proteome, meta, group_col, group1, group2, method):
    idx1 = meta.index[meta[group_col] == group1]
    idx2 = meta.index[meta[group_col] == group2]

    X = proteome.loc[idx1].to_numpy(float)
    Y = proteome.loc[idx2].to_numpy(float)
    proteins = proteome.columns.to_numpy()

    if method == "student_ttest":
        _, pvals = ttest_ind(X, Y, equal_var=True, nan_policy="omit")
    elif method == "welch_ttest":
        _, pvals = ttest_ind(X, Y, equal_var=False, nan_policy="omit")
    elif method == "paired":
        _, pvals = ttest_rel(X, Y, nan_policy="omit")
    else:
        raise ValueError(f"Unsupported method: {method}")

    log2fc = np.nanmean(X, axis=0) - np.nanmean(Y, axis=0)

    return pd.DataFrame({
        "protein": proteins,
        "log2fc": log2fc,
        "pval": pvals,
        "padj": np.nan
    })





def resampling_adjust(
    proteome: pd.DataFrame,
    meta: pd.DataFrame,
    group_col: str,
    group1: str,
    group2: str,
    method: str,
    adjust: str,
    n_perm: int,
    df: pd.DataFrame   # pass the observed results in!
) -> np.ndarray:
    if method == "perm_test":
        raise ValueError("Resampling p pvalue adjustment not available for permmutation test")    
    rng = np.random.default_rng(seed=0)
    labels = meta[group_col].values
    n_proteins = df.shape[0]
    obs_pvals = df["pval"].values

    perm_pvals = np.zeros((n_perm, n_proteins))
    for i in range(n_perm):
        perm_labels = rng.permutation(labels)
        perm_meta = meta.copy()
        perm_meta[group_col] = perm_labels
        perm_df = ttests(proteome, perm_meta, group_col, group1, group2, method)
        perm_pvals[i, :] = perm_df["pval"].values

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
    pairing: str = None, 
    delta_col: str = None,    
    delta_group1: str = None,
    delta_group2: str = None
) -> pd.DataFrame:
   
    common = meta.index.intersection(proteome.index)
    meta = meta.loc[common]
    proteome = proteome.loc[common]

    if method == "delta":
        proteome, meta = prep_deltas(
            meta=meta,
            proteome=proteome,
            subject_col=pairing,
            delta_col=delta_col,
            delta_group1=delta_group1,
            delta_group2=delta_group2,
        )
        method = "student_ttest" 
    
    df = ttests(proteome, meta, group_col, group1, group2, method)
    
    mask = df["pval"].notna()
    if mask.any():
        if adjust in ["perm", "stepdown_perm"]:
            df.loc[mask, "padj"] = resampling_adjust(
                proteome, meta, group_col, group1, group2, method, adjust, n_perm, df
            )
        else:
            df.loc[mask, "padj"] = multipletests(df.loc[mask, "pval"], method=adjust)[1]
    
    return df.set_index("protein").sort_values("pval")





def filter_results(results_df, term):
    df = results_df[results_df["term"].str.contains(term, na=False)].copy()
    df = df.drop(columns=["term"])
    df = df.sort_values("pval")
    return df.reset_index(drop=True)




def run_mixedlm(proteome, meta, formula, group_col=None, adjust="fdr_bh", reml=True, filter_to=None):
    """
    Fits MixedLM if a valid grouping structure exists; otherwise falls back to OLS.
    The dependent variable 'y' in `formula` is replaced by each protein column.
    """
    results = []
    if group_col not in meta.columns:
        raise ValueError(f"{group_col} not in meta columns")
    
    # precompute group stats if group_col is given
    if group_col is not None:
        grp_counts = meta[group_col].value_counts()
        has_replication = (grp_counts >= 2).sum() >= 1
    else:
        has_replication = False
        print("No grouping found, defaulted to ordinary least squares")

    for protein in proteome.columns:
        df = meta.join(proteome[[protein]]).rename(columns={protein: "y"})
        # skip if y invalid
        if df["y"].isna().all() or df["y"].nunique(dropna=True) <= 1:
            results.append({"protein": protein, "term": None, "coef": np.nan, "pval": np.nan})
            continue

        try:
            if group_col and has_replication:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = mixedlm(formula, df, groups=df[group_col], re_formula="1")
                    fit = m.fit(reml=reml, method="lbfgs", disp=False)
                # boundary check: if group variance ~ 0, refit OLS for stable SEs
                gv = np.ravel(fit.cov_re)[0] if fit.cov_re.size else 0.0
                if not np.isfinite(gv) or gv <= 1e-8:
                    fit = ols(formula, data=df).fit()
            else:
                fit = ols(formula, data=df).fit()

            for term, coef, pval in zip(fit.params.index, fit.params.values, fit.pvalues.values):
                results.append({"protein": protein, "term": term, "coef": coef, "pval": pval})

        except Exception:
            # record failure for this protein
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






def run_logreg(
    proteome: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    column: str,              
    group1: str,
    group2: str,
    adjust: str = "fdr_bh",
    max_iter: int = 1000,
    cv: int = 5,
):

    meta_sub = meta[meta[column].isin([group1, group2])]
    X = proteome.loc[meta_sub.index].values
    y = (meta_sub[column] == group1).astype(int).values
    feature_names = proteome.columns

    clf = LogisticRegressionCV(
        Cs=10,                 
        cv=cv,
        penalty="l2",
        solver="lbfgs",
        max_iter=max_iter,
    )
    clf.fit(X, y)

    betas = clf.coef_.flatten()

    p = clf.predict_proba(X)[:, 1]
    W = np.diag(p * (1 - p))
    fisher_info = X.T @ W @ X
    cov_matrix = np.linalg.pinv(fisher_info)  
    se = np.sqrt(np.diag(cov_matrix))

    z = betas / se
    pvals = 2 * (1 - norm.cdf(np.abs(z)))
    _, qvals, _, _ = multipletests(pvals, method=adjust)

    results = pd.DataFrame({
        "protein": feature_names,
        "beta": betas,
        "se": se,
        "pval": pvals,
        "padj": qvals,
    }).set_index("protein").sort_values("pval")

    return clf, results







def volcano_plot(results, min_fc=0, alpha=0.05, labels=True):
    df = results.copy().reset_index()
    df = df.rename(columns={"index": "protein"})

    if "coef" in df.columns:
        df["log2fc"] = df["coef"]

    # avoid -inf for any exact zero p-values
    p = df["pval"].astype(float).clip(lower=np.finfo(float).tiny)
    df["-log10(p-value)"] = -np.log10(p)

    # classify points
    df["color"] = "non_sig"
    df.loc[(df["log2fc"] >  min_fc) & (df["padj"] < alpha), "color"] = "up"
    df.loc[(df["log2fc"] < -min_fc) & (df["padj"] < alpha), "color"] = "down"

    color_dict = {
        "non_sig": "#404040",
        "up": "#B65EAF",
        "down": "#009599",
    }

    # always create label column; fill only if labels=True
    df["label"] = ""
    if labels:
        df.loc[df["color"].isin(["up", "down"]), "label"] = df["protein"]

    fig = px.scatter(
        df,
        x="log2fc",
        y="-log10(p-value)",
        color="color",
        color_discrete_map=color_dict,
        hover_name="protein",
        text="label",  # safe: column always exists
        template="simple_white+alphastats_colors",
    )

    fig.update_traces(
        textposition="top center",
        textfont=dict(size=9)
    )

    # cutoff lines
    ycut = -np.log10(alpha)
    #fig.add_hline(y=ycut, line_width=1, line_dash="dash", line_color="#8c8c8c")
    #fig.add_vline(x=min_fc,  line_width=1, line_dash="dash", line_color="#8c8c8c")
    #fig.add_vline(x=-min_fc, line_width=1, line_dash="dash", line_color="#8c8c8c")

    fig.update_layout(
        showlegend=False,
        width=600,
        height=500,
        xaxis_title="log2 Fold Change",
        yaxis_title="-log10(p-value)",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig







def go_enrichment(
    data: pd.DataFrame,
    pval_cutoff: float = 0.05,
    organism: str = "hsapiens",
    sources = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"],
    restrict_background = True,
    adjust = "g_SCS"
) -> pd.DataFrame:
    """
    Run functional enrichment analysis on DEA results using g:Profiler.
    
    data : dataframe indexed by UniProt IDs, must contain a "pval" column
    pval_cutoff : p-value threshold for significant proteins, default = 0.05.
    organism : organism identifier, default: "hsapiens"
    sources : databases to query 
    """

    gp = GProfiler(return_dataframe=True)
    all_ids = data.index.str.split(";").str[0].dropna().unique().tolist()

    conversion = gp.convert(organism=organism, query=all_ids)
    conversion = conversion.dropna(subset=["converted"]).drop_duplicates("incoming")

    data_with_genes = data.copy()
    data_with_genes["UniProt_ID"] = data_with_genes.index.str.split(";").str[0]
    data_with_genes = data_with_genes.merge(
        conversion[["incoming", "converted", "name"]],
        left_on="UniProt_ID",
        right_on="incoming",
        how="left"
    ).dropna(subset=["converted"])

    sig_ids = data_with_genes.loc[data_with_genes["padj"] < pval_cutoff, "converted"].unique().tolist()
    background_ids = data_with_genes["converted"].unique().tolist()
    print(len(sig_ids))
    
    if restrict_background:
        results = gp.profile(
            organism=organism,
            query=sig_ids,
            domain_scope="custom",
            background=background_ids,
            sources=sources,
            all_results=True,
            significance_threshold_method = adjust
        )
    
    else:
        results = gp.profile(
            organism=organism,
            query=sig_ids,
            sources=sources,
            all_results=True,
            significance_threshold_method = adjust
        )
    
    return results








