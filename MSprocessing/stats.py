import re
import numpy as np
import pandas as pd
import plotly.express as px
import MSprocessing.stats as mss

from scipy.stats import norm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests
from alphastats.dataset.keys import Cols
from alphastats.plots.volcano_plot import VolcanoPlot
from alphastats.dataset.preprocessing import PreprocessingStateKeys as PSK
from alphastats.statistics.statistic_utils import calculate_foldchange
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
        variable:     [group1]*len(g1) + [group2]*len(g2),
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





def run_dea(
    proteome: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    method: str,                    # "ttest" | "paired-ttest" | "welch-ttest" | "sam" 
    column: str,                    # metadata column with group labels
    group1: Union[str, List[str]],
    group2: Union[str, List[str]],
    adjust: str = "fdr_bh",
    preprocessing_info: Dict,
    perm: int = 10,                  #SAM permutations
    pairing: str,
    alpha: float = 0.05,
    min_fc: float = 1,
    delta_col: str = None,
    delta_group1: str = None,
    delta_group2: str = None,
):
    meta_alpha = meta.reset_index().rename(columns={meta.index.name: Cols.SAMPLE})
    if method not in ["ttest", "delta-ttest", "paired-ttest", "welch-ttest", "anova", "sam"]:
        raise ValueError('Please select a valid method: "ttest", "delta-ttest", "paired-ttest" or "sam"')

    elif method == "paired-ttest":
        if pairing == column:
            raise ValueError("Data cannot be paired on variable to test")
        proteome, meta_alpha = prep_for_paired_ttest(
            meta_alpha, proteome, pair_col=pairing, variable=column, group1=group1, group2=group2
        )

    elif method == "delta-ttest":
        if delta_col == column:
            raise ValueError("Delta column cannot be the variable to test")
        elif delta_col == pairing:
            raise ValueError("Delta column cannot be the same as pairing column")
        elif pairing == column:
            raise ValueError("Data cannot be paired on variable to test")
        proteome, meta_alpha = prep_deltas(
            meta=meta_alpha,
            proteome=proteome,
            subject_col=pairing,
            delta_col=delta_col,
            delta_group1=delta_group1,
            delta_group2=delta_group2,
        )
        meta_alpha[Cols.SAMPLE] = meta_alpha.index
        method = "ttest"
    
    elif method == "welch-ttest" and preprocessing_info[PSK.LOG2_TRANSFORMED]:
        proteome = 2 ** proteome 

    elif method == "anova":
        proteome = proteome.transpose()

    feature_to_repr_map = {pid: pid for pid in proteome.columns}

    #VolcanoPlot runs DEA internally
    vp = VolcanoPlot(
        mat=proteome,
        rawinput=proteome,  
        metadata=meta_alpha,
        preprocessing_info=preprocessing_info,
        feature_to_repr_map=feature_to_repr_map,
        group1=group1,
        group2=group2,
        column=column,
        method=method,
        min_fc=min_fc,
        alpha=alpha,
        draw_line=True,
        perm=perm,
    )

    df = vp.res.copy()   
    volcano_fig = vp.plot  

    if method == "sam":
        vp._draw_fdr_line()
        volcano_fig = vp.plot 
        
    volcano_fig.update_layout(
        width=600,
        height=500,
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    
    effect = df["log2fc"] if "log2fc" in df.columns else df.get("fc")
    pval = (
        df["pval"] if "pval" in df.columns
        else df["pval_s0"] if "pval_s0" in df.columns
        else df["pvalue"]
    )
    padj = df["qval"] if "qval" in df.columns else pd.Series(
        multipletests(pval.values, method=adjust)[1], index=df.index
    )

    out = pd.DataFrame(
        {
            Cols.INDEX: df[Cols.INDEX],
            "log2fc": effect,
            "pval": pval.astype(float),
            "padj": padj.astype(float),
        }
    ).sort_values("pval", ascending=True).set_index(Cols.INDEX) 
    out.index.name = "protein"

    return out, volcano_fig





def run_mixedlm(
    proteome: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    var1: dict,       # e.g. {"group": ["Kontrol", "Intervention"]}
    var2: dict,       # e.g. {"timepoint": ["baseline", "w48"]}
    pairing: str,     # subject ID column
    adjust: str = "fdr_bh",
):
    # unpack the dicts
    col1, (level1a, level1b) = list(var1.items())[0]
    col2, (level2a, level2b) = list(var2.items())[0]

    meta_sub = meta[
        meta[col1].isin([level1a, level1b]) & meta[col2].isin([level2a, level2b])
    ].copy()
    meta_sub[col1] = pd.Categorical(meta_sub[col1],
                                    categories=[level1a, level1b],
                                    ordered=True)
    meta_sub[col2] = pd.Categorical(meta_sub[col2],
                                    categories=[level2a, level2b],
                                    ordered=True)
    meta_sub[pairing] = meta_sub[pairing].astype("category")
    expr = proteome.loc[meta_sub.index]

    results = []
    cols = ["protein", "beta", "pval"]
    term = f"C({col1})[T.{level1b}]:C({col2})[T.{level2b}]"

    for prot in expr.columns:
        df = meta_sub.assign(intensity=pd.to_numeric(expr[prot].values, errors="coerce"))
        df = df.dropna(subset=["intensity", col1, col2, pairing])

        try:
            fit = mixedlm(
                f"intensity ~ C({col1})*C({col2})",
                df, groups=df[pairing]
            ).fit(reml=False)

            if term in fit.params:
                results.append(dict(zip(
                    cols,
                    [prot, float(fit.params[term]), float(fit.pvalues[term])]
                )))
        except Exception as e:
            print(f"{prot}: {e}")
            continue

    out_df = pd.DataFrame(results)
    if not out_df.empty:
        out_df["padj"] = multipletests(out_df["pval"], method=adjust)[1]
        out_df = out_df.sort_values("pval", ascending=True).set_index("protein")
        out_df.index.name = "protein"

    return out_df





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



def go_enrichment(
    data: pd.DataFrame,
    pval_cutoff: float = 0.05,
    organism: str = "hsapiens",
    sources = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"],
    restrict_background = True,
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

    sig_ids = data_with_genes.loc[data_with_genes["pval"] < pval_cutoff, "converted"].unique().tolist()
    background_ids = data_with_genes["converted"].unique().tolist()
    
    if restrict_background:
        results = gp.profile(
            organism=organism,
            query=sig_ids,
            domain_scope="custom",
            background=background_ids,
            sources=sources,
            all_results=True
        )
    
    else:
        results = gp.profile(
            organism=organism,
            query=sig_ids,
            sources=sources,
            all_results=True
        )
    
    return results