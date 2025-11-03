
import mygene
import pandas as pd
from gprofiler import GProfiler




def convert_ids(
    df: pd.DataFrame,
    from_type: str,
    to_type: str,
    axis: int = 0,
    species: str = "human"
) -> pd.DataFrame:
    """
    Convert gene/protein identifiers in a DataFrame's index or columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with row or column identifiers to convert.
    from_type : str
        Type of the current identifiers (e.g., 'uniprot', 'symbol', 'ensembl').
    to_type : str
        Desired output identifier type (e.g., 'symbol', 'uniprot', 'entrezgene').
    axis : int, default=0
        0 = convert rownames (index), 1 = convert column names.
    species : str, default='human'
        Species name or taxonomy ID, passed to MyGeneInfo.

    Returns
    -------
    pd.DataFrame
        Same DataFrame but with converted row/column identifiers.
    """

    # Make a copy so the original isn't altered
    df_copy = df.copy()

    ids = df_copy.index if axis == 0 else df_copy.columns
    ids = ids.astype(str).tolist()
    ids = [i.split(";")[0].split(",")[0].strip() for i in ids]

    mg = mygene.MyGeneInfo()
    try:
        res = mg.querymany(ids, scopes=from_type, fields=to_type, species=species, as_dataframe=True)
        if to_type not in res.columns:
            raise KeyError(f"Requested field '{to_type}' not found in results.")
        mapping = res[[to_type]].dropna().to_dict()[to_type]
    except Exception as e:
        raise RuntimeError(f"MyGeneInfo query failed: {e}")

    new_ids = [mapping.get(i, i) for i in ids]

    if axis == 0:
        df_copy.index = new_ids
    else:
        df_copy.columns = new_ids

    return df_copy







def go_enrichment(
    data: pd.DataFrame,
    pval_cutoff: float = 0.05,
    organism: str = "hsapiens",
    sources = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"],
    restrict_background = True,
    adjust = "g_SCS"
) -> pd.DataFrame:
    """
    Perform functional enrichment analysis using g:Profiler.

    This function identifies enriched biological processes, molecular functions,
    cellular components, and pathway terms among significantly changed proteins.

    Parameters
    ----------
    data : pd.DataFrame
        Differential expression results indexed by UniProt IDs.
        Must contain a "padj" column for adjusted p-values.
    pval_cutoff : float, default=0.05
        Adjusted p-value threshold defining significantly changed proteins.
    organism : str, default="hsapiens"
        Organism code recognized by g:Profiler (e.g. "hsapiens", "mmusculus").
    sources : list of str, default=["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"]
        Functional annotation databases to include in the enrichment analysis.
    restrict_background : bool, default=True
        If True, restricts enrichment background to all tested proteins.
        If False, uses the default g:Profiler organism-wide background.
    adjust : str, default="g_SCS"
        Multiple-testing correction method used by g:Profiler.

    Returns
    -------
    pd.DataFrame
        Enrichment results as returned by g:Profiler, including term IDs,
        names, sources, p-values, and enrichment statistics.
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
