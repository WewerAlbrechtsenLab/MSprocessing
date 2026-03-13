import ast
import copy

import mygene
import networkx as nx
import pandas as pd
import requests
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
    sources=["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"],
    restrict_background=True,
    convert_id=True,
    adjust="g_SCS"
) -> pd.DataFrame:
    """
    Perform functional enrichment analysis on up- and down-regulated proteins separately.

    Splits the input data by log2 fold-change direction, identifies enriched 
    biological processes, molecular functions, cellular components, and
    pathway terms for each direction independently, then combines results.

    Parameters
    ----------
    data : pd.DataFrame
        Differential expression results indexed by UniProt IDs.
        Must contain "padj" and "log2fc" columns.
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
    convert_id : bool, default=True
        If True, converts UniProt IDs to gene symbols using g:Profiler before analysis.

    Returns
    -------
    pd.DataFrame
        Combined enrichment results for up- and down-regulated proteins, including
        columns: source, name, p_value, description, term_size, query_size,
        intersection_size, direction. Filtered to significant results only and
        sorted by p_value.
    """
    gp = GProfiler(return_dataframe=True)

    if "log2fc" in data.columns:
        fc_col = "log2fc"
    elif "coef" in data.columns:
        fc_col = "coef"
    else:
        raise ValueError("Input data must contain either 'log2fc' or 'coef' column.")

    data = data.copy()
    data["UniProt_ID"] = data.index.str.split(";").str[0]

    if convert_id:
        all_ids = data["UniProt_ID"].dropna().unique().tolist()
        conversion = gp.convert(organism=organism, query=all_ids)
        conversion = conversion.dropna(subset=["converted"]).drop_duplicates("incoming")

        id_map = conversion[["incoming", "converted"]].rename(
            columns={"incoming": "uniprot", "converted": "ontology_id"}
        )

        uniprot_by_ontology = id_map.groupby("ontology_id")["uniprot"].apply(list).to_dict()

        data = data.merge(
            conversion[["incoming", "converted", "name"]],
            left_on="UniProt_ID",
            right_on="incoming",
            how="left"
        ).dropna(subset=["converted"])
    else:
        data["converted"] = data["UniProt_ID"]
        uniprot_by_ontology = dict(zip(data["converted"], data["converted"]))

    sig_data = data[data["padj"] < pval_cutoff].copy()
    background_ids = data["converted"].unique().tolist()

    data_up = sig_data[sig_data[fc_col] > 0].copy()
    data_down = sig_data[sig_data[fc_col] < 0].copy()

    results_list = []

    for direction, direction_data in [("up", data_up), ("down", data_down)]:
        if direction_data.empty:
            continue

        sig_ids = direction_data["converted"].unique().tolist()
        if not sig_ids:
            continue

        if restrict_background:
            results = gp.profile(
                organism=organism,
                query=sig_ids,
                domain_scope="custom",
                background=background_ids,
                sources=sources,
                all_results=True,
                significance_threshold_method=adjust,
                no_evidences=False
            )
        else:
            results = gp.profile(
                organism=organism,
                query=sig_ids,
                sources=sources,
                all_results=True,
                significance_threshold_method=adjust,
                no_evidences=False
            )

        if results is None or results.empty:
            continue

        results_filtered = results[results["significant"]].copy()
        results_filtered["direction"] = direction
        results_filtered = results_filtered.drop(columns=["evidences", "query"], errors="ignore")

        results_filtered["proteins"] = results_filtered["intersections"].apply(
            lambda ids: sorted({u for oid in ids for u in uniprot_by_ontology.get(oid, [])})
        )

        results_list.append(results_filtered)

    if results_list:
        combined = pd.concat(results_list, ignore_index=True)
        combined = combined.sort_values("p_value").reset_index(drop=True)
        return combined

    print("No significant enrichment found.")




def build_reactome_graph():
    """
    Build a directed Reactome pathway hierarchy graph for human pathways.

    Returns
    -------
    nx.DiGraph
        Directed graph of the Reactome pathway hierarchy with node names
        stored in the "name" attribute.
    """
    url = "https://reactome.org/ContentService/data/eventsHierarchy/9606"
    response = requests.get(url, params={"pathwaysOnly": "true"}, timeout=60)
    response.raise_for_status()
    tree = response.json()

    graph = nx.DiGraph()
    stack = list(tree)

    while stack:
        node = stack.pop()
        node_id = node["stId"]
        graph.add_node(node_id, name=node.get("name"))

        for child in node.get("children", []):
            child_id = child["stId"]
            graph.add_node(child_id, name=child.get("name"))
            graph.add_edge(node_id, child_id)
            stack.append(child)

    return graph




def drive_proteins_up(graph: nx.DiGraph, root_id: str = "R-HSA-000000", attr: str = "proteins"):
    """
    Propagate protein annotations upward through a directed acyclic pathway graph.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed pathway graph whose nodes may contain a protein attribute.
    root_id : str, default="R-HSA-000000"
        Identifier of the root node to exclude from propagation.
    attr : str, default="proteins"
        Node attribute containing protein annotations to propagate.

    Returns
    -------
    nx.DiGraph
        Deep-copied graph with propagated protein sets and a "driven_children"
        node attribute listing descendants whose proteins were absorbed.
    """
    G = copy.deepcopy(graph)

    for n in G.nodes:
        ps = G.nodes[n].get(attr, [])
        G.nodes[n][attr] = set(ps) if ps else set()
        G.nodes[n]["driven_children"] = set()

    for n in reversed(list(nx.topological_sort(G))):
        if n == root_id:
            continue

        parents = list(G.predecessors(n))
        if not parents:
            continue

        for p in parents:
            if p == root_id:
                continue

            if G.nodes[n][attr]:
                G.nodes[p][attr] |= G.nodes[n][attr]
                G.nodes[p]["driven_children"].add(n)
                G.nodes[p]["driven_children"].update(G.nodes[n]["driven_children"])
                G.nodes[n][attr].clear()
                G.nodes[n]["driven_children"].clear()

    for n in G.nodes:
        G.nodes[n][attr] = sorted(G.nodes[n][attr])
        G.nodes[n]["driven_children"] = sorted(G.nodes[n]["driven_children"])

    return G



def make_parent_table(clean_graph, enrichment, root_id="R-HSA-000000"):
    """
    Build a summary table of parent pathway nodes and their driven proteins.

    Parameters
    ----------
    clean_graph : nx.DiGraph
        Graph after upward protein propagation.
    enrichment : pd.DataFrame
        Enrichment results containing at least "native" and "name".
    root_id : str, default="R-HSA-000000"
        Identifier of the root node to exclude from the output.

    Returns
    -------
    pd.DataFrame
        Table with parent identifiers, names, associated term names, proteins,
        and counts of proteins and terms.
    """
    e = enrichment.copy()
    e["stId"] = e["native"].str.replace(r"^REAC:", "", regex=True)

    stid_to_name = (
        e.dropna(subset=["stId", "name"])
         .drop_duplicates(subset=["stId"])
         .set_index("stId")["name"]
         .to_dict()
    )

    rows = []
    for n in clean_graph.nodes:
        prots = clean_graph.nodes[n].get("proteins", [])
        if not prots:
            continue
        if n == root_id:
            continue

        parent_name = stid_to_name.get(n, clean_graph.nodes[n].get("name"))

        driven = clean_graph.nodes[n].get("driven_children", [])
        term_ids = [n] + list(driven)

        terms = []
        for tid in term_ids:
            tname = stid_to_name.get(tid, clean_graph.nodes[tid].get("name"))
            if tname is not None:
                terms.append(tname)

        rows.append(
            {
                "parent_ID": n,
                "parent_name": parent_name,
                "terms": terms,
                "proteins": list(set(prots)),
                "n_proteins": len(prots),
                "n_terms": len(terms)
            }
        )

    return pd.DataFrame(rows).sort_values(["n_proteins", "parent_ID"], ascending=[False, True]).reset_index(drop=True)




def mean_log2fc(protein_list, log2fc_map):
    """
    Compute the mean fold-change value for a set of proteins in DEA results.

    Parameters
    ----------
    protein_list : list or str
        Protein identifiers, or a string representation of such a list.
    log2fc_map : pd.Series
        Mapping from protein identifier to fold-change value.

    Returns
    -------
    float
        Mean fold-change across matched proteins, or NaN if none are found.
    """
    if isinstance(protein_list, str):
        protein_list = ast.literal_eval(protein_list)
    vals = log2fc_map.reindex(protein_list).dropna()
    return vals.mean() if len(vals) else np.nan




def cluster_enrichment(enrichment, dea, graph):
    """
    Collapse enriched pathway terms into parent pathway clusters and annotate them
    with mean differential expression values.

    Parameters
    ----------
    enrichment : pd.DataFrame
        Enrichment results containing at least "native", "proteins", and "direction".
        The "native" column is expected to contain Reactome-style identifiers such as
        "REAC:R-HSA-...".
    dea : pd.DataFrame
        Differential expression results containing either a "coef" or "log2fc"
        column. Protein identifiers must be present either in a "protein" column
        or in the index.
    graph : nx.DiGraph
        Directed pathway graph used to propagate proteins from child terms to
        parent terms.

    Returns
    -------
    pd.DataFrame
        Clustered parent pathway table containing pathway identifiers, names,
        grouped terms, protein members, protein counts, term counts, regulation
        direction, and mean fold-change annotation in the "mean_coef" column.
    """
    enrichment = enrichment.copy()
    enrichment["stId"] = enrichment["native"].str.replace(r"^REAC:", "", regex=True)
    enrichment["proteins"] = enrichment["proteins"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if x is not None else [])
    )

    e = enrichment.copy()
    e["direction"] = e["direction"].str.lower().str.strip()
    e["stId"] = e["native"].str.replace(r"^REAC:", "", regex=True)
    parent_tables = []

    for direction in ["up", "down"]:
        enrichment_dir = e[e["direction"] == direction].copy()
        if enrichment_dir.empty:
            continue

        graph_dir = copy.deepcopy(graph)

        stid_to_proteins = (
            enrichment_dir.groupby("stId")["proteins"]
            .apply(lambda s: set(p for lst in s for p in lst))
            .to_dict()
        )

        nx.set_node_attributes(
            graph_dir,
            {k: {"proteins": sorted(v)} for k, v in stid_to_proteins.items()}
        )

        clean_graph_dir = drive_proteins_up(
            graph_dir,
            root_id="R-HSA-000000",
            attr="proteins"
        )

        parent_table_dir = make_parent_table(
            clean_graph_dir,
            enrichment_dir,
            root_id="R-HSA-000000"
        )

        parent_table_dir["direction"] = direction
        parent_tables.append(parent_table_dir)

    parent_table = (
        pd.concat(parent_tables, ignore_index=True)
        .sort_values("n_proteins", ascending=False)
        .reset_index(drop=True)
    )

    fc_col = "coef" if "coef" in dea.columns else "log2fc"

    dea = dea.copy()
    if "protein" not in dea.columns:
        dea = dea.reset_index().rename(columns={"index": "protein"})

    log2fc_map = dea.set_index("protein")[fc_col]


    parent_table["mean_coef"] = parent_table["proteins"].apply(
        mean_log2fc, log2fc_map=log2fc_map
        )

    return parent_table