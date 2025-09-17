import re
import numpy as np
import pandas as pd
import plotly.express as px
import MSprocessing.stats as mss

from alphastats.dataset.keys import Cols
from statsmodels.stats.multitest import multipletests
from alphastats.plots.volcano_plot import VolcanoPlot
from alphastats.dataset.preprocessing import PreprocessingStateKeys as PSK
from alphastats.statistics.statistic_utils import calculate_foldchange
from alphastats.statistics.differential_expression_analysis import DifferentialExpressionAnalysis
from typing import Dict, List, Optional, Tuple, Union
from gprofiler import GProfiler



#monkeypatched deprecated sumpy aliases 
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool




def split_proteome_meta(df: pd.DataFrame):
    """
    df: wide DataFrame with samples in the MultiIndex (metadata levels) and proteins as columns.
    Assumes metadata are exactly the index levels.
    """
    index_cols = list(df.index.names)
    tmp = df.reset_index()
    protein_cols = [c for c in tmp.columns if c not in index_cols]

    mat = tmp.set_index("sample_name")[protein_cols].copy()
    meta = tmp[index_cols].copy()
    meta = tmp.set_index("sample_name").copy()

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





def run_dea(
    proteome: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    method: str,                    # "ttest" | "paired-ttest" | "welch-ttest" | "sam" | "wald"
    column: str,                    # metadata column with group labels
    group1: Union[str, List[str]],
    group2: Union[str, List[str]],
    adjust: str = "fdr_bh",
    preprocessing_info: Dict,
    perm: int = 10,                  #SAM permutations
    pairing: str,
    alpha: float = 0.05,
    min_fc: float = 1,
):
    meta_alpha = meta.reset_index().rename(columns={meta.index.name: Cols.SAMPLE})

    if method == "paired-ttest":
        proteome, meta_alpha = prep_for_paired_ttest(
            meta_alpha, proteome, pair_col=pairing, variable=column, group1=group1, group2=group2
        )
    
    if method == "welch-ttest" and preprocessing_info[PSK.LOG2_TRANSFORMED]:
        proteome = proteome ** 10

    feature_to_repr_map = {pid: pid for pid in proteome.columns}

    #use VolcanoPlot instead of calling DEA directly
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

    df = vp.res.copy()   # results table
    volcano_fig = vp.plot  # plotly figure

    if method == "sam":
        vp._draw_fdr_line()
        volcano_fig = vp.plot  # refresh figure object
        
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






def go_enrichment(
    data: pd.DataFrame,
    pval_cutoff: float = 0.05,
    organism: str = "hsapiens",
    sources = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"],
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

    results = gp.profile(
        organism=organism,
        query=sig_ids,
        domain_scope="custom",
        background=background_ids,
        sources=sources,
        all_results=True
    )
    
    return results