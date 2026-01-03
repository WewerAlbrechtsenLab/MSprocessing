import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from matplotlib_venn import venn2, venn3
from plotly.colors import qualitative
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler








def volcano_plot(results, alpha=0.05, labels=True):
    """
    Create an interactive volcano plot for differential expression results.

    Parameters
    ----------
    results : pd.DataFrame
        Differential analysis results containing at least 'pval', 'padj',
        and either 'log2fc' or 'coef' columns.
    alpha : float, default=0.05
        Adjusted p-value significance threshold.
    labels : bool, default=True
        Whether to display labels for significant points.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly scatter plot representing log2 fold change vs. -log10(p-value),
        with significant features highlighted.
    """
    df = results.copy().reset_index()
    df = df.rename(columns={"index": "protein"})

    if "coef" in df.columns:
        df["log2fc"] = df["coef"]

    # avoid -inf for any exact zero p-values
    p = df["pval"].astype(float).clip(lower=np.finfo(float).tiny)
    df["-log10(p-value)"] = -np.log10(p)

    # classify points
    df["color"] = "non_sig"
    df.loc[(df["log2fc"] >  0) & (df["padj"] < alpha), "color"] = "up"
    df.loc[(df["log2fc"] < 0) & (df["padj"] < alpha), "color"] = "down"

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
        template="simple_white",
    )

    fig.update_traces(
        textposition="top center",
        textfont=dict(size=9)
    )


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






def plot_venn(dfs, names, alpha=0.05):
    """
    Plot a Venn diagram for the overlap of significant proteins across datasets.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        List of differential analysis results, each containing 'padj' and
        either 'protein' or index as protein identifiers.
    names : list of str
        Labels corresponding to each dataset for the Venn diagram.
    alpha : float, default=0.05
        Adjusted p-value threshold for defining significance.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object containing the Venn diagram.
    """
    sig_sets = []
    for df in dfs:
        if "protein" in df.columns:
            s = set(df.loc[df["padj"] < alpha, "protein"])
        else:
            s = set(df.index[df["padj"] < alpha])
        sig_sets.append(s)
    
    n = len(sig_sets)
    fig, ax = plt.subplots(figsize=(5, 5))

    if n == 2:
        v = venn2(sig_sets, set_labels=names,
                  set_colors=("#2ABBEB", "#B3F864"), ax=ax)
        ids = ['10', '01', '11']
        regions = {
            '10': sig_sets[0] - sig_sets[1],
            '01': sig_sets[1] - sig_sets[0],
            '11': sig_sets[0] & sig_sets[1]
        }
    elif n == 3:
        v = venn3(sig_sets, set_labels=names,
                  set_colors=("#2ABBEB", "#B3F864", "#79CFE0"), ax=ax)
        ids = ['100', '010', '001', '110', '101', '011', '111']
        A, B, C = sig_sets
        regions = {
            '100': A - B - C,
            '010': B - A - C,
            '001': C - A - B,
            '110': (A & B) - C,
            '101': (A & C) - B,
            '011': (B & C) - A,
            '111': A & B & C
        }
    else:
        raise ValueError("Only 2 or 3 sets supported in this version")
    
    # Fill in text for each region
    for rid in ids:
        subset = regions.get(rid, set())
        lbl = v.get_label_by_id(rid)
        if lbl:
            lbl.set_text("\n".join(sorted(subset)) if subset else "")

    ax.set_title(f"Significant Proteins Overlap (padj < {alpha})")
    plt.tight_layout()
    return fig





def plot_heatmap(
    proteome, 
    meta, 
    group_by=None, 
    method="average", 
    metric="correlation",
    height=800, 
    width=800
):
    """
    Generate an interactive clustered sample–sample correlation heatmap.

    Parameters
    ----------
    proteome : pd.DataFrame
        Quantitative proteomic data with samples as rows and proteins as columns.
    meta : pd.DataFrame
        Sample metadata corresponding to proteome rows.
    group_by : str or list of str, optional
        Metadata column(s) used to color and annotate samples.
    method : str, default="average"
        Linkage method for hierarchical clustering.
    metric : str, default="correlation"
        Distance metric for pairwise dissimilarity computation.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure containing a dendrogram-aligned correlation heatmap
        with optional metadata color bars.
    """
    proteome_z = pd.DataFrame(
        StandardScaler().fit_transform(proteome),
        index=proteome.index,
        columns=proteome.columns,
    )

    linkage_s = linkage(pdist(proteome_z, metric=metric), method=method)
    linkage_p = linkage(pdist(proteome_z.T, metric=metric), method=method)

    order_s = leaves_list(linkage_s)
    order_p = leaves_list(linkage_p)

    data_ordered = proteome_z.iloc[order_s, order_p]
    meta_ordered = meta.iloc[order_s]
    proteins_ordered = proteome_z.columns[order_p]
    samples_ordered = proteome_z.index[order_s]

    n_samples, n_proteins = data_ordered.shape

    if group_by is None:
        group_by_list = []
    elif isinstance(group_by, str):
        group_by_list = [group_by]
    else:
        group_by_list = list(group_by)

    n_groups = len(group_by_list)

    widths = [0.2] + [0.04] * n_groups + [0.76]
    fig = make_subplots(
        rows=2,
        cols=2 + n_groups,
        column_widths=widths,
        row_heights=[0.18, 0.82],
        vertical_spacing=0.002,
        horizontal_spacing=0.01,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    heatmap_col = 2 + n_groups

    dendro_p = dendrogram(linkage_p, no_plot=True)
    max_dp = max(max(d) for d in dendro_p["dcoord"])

    for xs, ys in zip(dendro_p["icoord"], dendro_p["dcoord"]):
        xs_fixed = [(x - 5) / 10 for x in xs]

        if ys[0] == 0 and ys[1] == 0:
            idx = int(xs_fixed[1])
            hovertemplate = "%{text}<extra></extra>"
            text = proteins_ordered[idx]
        else:
            hovertemplate = None
            text = None

        fig.add_trace(
            go.Scatter(
                x=xs_fixed,
                y=ys,
                mode="lines",
                line=dict(color="black", width=1),
                hovertemplate=hovertemplate,
                text=text,
                showlegend=False,
            ),
            row=1,
            col=heatmap_col,
        )

    dendro_s = dendrogram(linkage_s, orientation="right", no_plot=True)
    for xs, ys in zip(dendro_s["dcoord"], dendro_s["icoord"]):
        ys_fixed = [(y - 5) / 10 for y in ys]

        if xs[0] == 0 and xs[1] == 0:
            idx = int(ys_fixed[1])
            hovertemplate = "%{text}<extra></extra>"
            text = samples_ordered[idx]
        else:
            hovertemplate = None
            text = None

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys_fixed,
                mode="lines",
                line=dict(color="black", width=1),
                hovertemplate=hovertemplate,
                text=text,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(range=[0, max_dp], row=1, col=heatmap_col)
    fig.update_xaxes(autorange="reversed", row=2, col=1)

    if n_groups:
        palettes = qualitative.Plotly + qualitative.D3 + qualitative.Bold
        y_idx = np.arange(n_samples)

        for i, col in enumerate(group_by_list):
            vals = meta_ordered[col].astype(str)
            uniq = vals.unique()
            lut = dict(zip(uniq, palettes * ((len(uniq) // len(palettes)) + 1)))

            fig.add_trace(
                go.Heatmap(
                    z=y_idx.reshape(-1, 1),
                    x=[col],
                    y=y_idx,
                    showscale=False,
                    colorscale=[[j / (len(vals) - 1), lut[v]] for j, v in enumerate(vals)],
                    text=vals.to_numpy().reshape(-1, 1),
                    hovertemplate=f"{col}: %{{text}}<extra></extra>",
                    showlegend=False,
                ),
                row=2,
                col=i + 2,
            )

    group_text = (
        "<br>".join([f"{g}: %{{customdata[{i + 2}]}}" for i, g in enumerate(group_by_list)])
        if n_groups else ""
    )

    hovertemplate = (
        "Sample: %{customdata[0]}<br>"
        "Protein: %{customdata[1]}<br>"
        f"{group_text}<br>"
        "Z: %{z:.3f}<extra></extra>"
    )

    customdata = np.empty((n_samples, n_proteins, 2 + n_groups), dtype=object)
    customdata[:, :, 0] = samples_ordered.to_numpy()[:, None]
    customdata[:, :, 1] = proteins_ordered.to_numpy()

    for i, g in enumerate(group_by_list):
        customdata[:, :, i + 2] = meta_ordered[g].astype(str).to_numpy()[:, None]

    fig.add_trace(
        go.Heatmap(
            z=data_ordered.values,
            x=np.arange(n_proteins),
            y=np.arange(n_samples),
            colorscale="RdBu_r",
            zmid=0,
            showscale=False,
            hovertemplate=hovertemplate,
            customdata=customdata,
            showlegend=False,
        ),
        row=2,
        col=heatmap_col,
    )

    fig.update_yaxes(
        scaleanchor=f"x{heatmap_col}",
        scaleratio=1,
        row=2,
        col=heatmap_col,
    )

    for r in [1, 2]:
        for c in range(1, heatmap_col + 1):
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=r, col=c)
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=r, col=c)

    fig.update_layout(
        height=height,
        width=width,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
        dragmode="zoom",
        margin=dict(t=40, b=10, l=10, r=10),
        showlegend=False,
    )

    return fig






def plot_significant_boxplot(meta, proteome, group_col, protein_col, normalize_by=None):
    """
    Create an interactive boxplot of protein intensities grouped by metadata.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata containing grouping and optional normalization columns.
    proteome : pd.DataFrame
        Proteomic intensity matrix indexed by sample, columns are proteins.
    group_col : str
        Metadata column defining sample groups.
    protein_col : str
        Protein column in proteome to plot.
    normalize_by : str, optional
        Metadata column for within-group z-score normalization.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly boxplot displaying distributions per group with
        optional normalization and per-sample scatter points.
    """
    if group_col not in meta.columns:
        raise ValueError(f"'{group_col}' not found in meta columns.")
    if protein_col not in proteome.columns:
        raise ValueError(f"'{protein_col}' not found in proteome columns.")
    if normalize_by is not None and normalize_by not in meta.columns:
        raise ValueError(f"'{normalize_by}' not found in meta columns.")

    df = pd.concat(
        [meta[[group_col] + ([normalize_by] if normalize_by else [])],
         proteome[[protein_col]]],
        axis=1
    ).dropna(subset=[group_col, protein_col])

    if normalize_by is not None:
        df[protein_col] = df.groupby(normalize_by)[protein_col].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0
        )

    fig = go.Figure()
    color_seq = px.colors.qualitative.Plotly
    if pd.api.types.is_categorical_dtype(df[group_col]):
        unique_groups = df[group_col].cat.categories
    else:
        unique_groups = sorted(df[group_col].unique())

    for i, g in enumerate(unique_groups):
        group_df = df[df[group_col] == g]
        fig.add_trace(go.Box(
            y=group_df[protein_col],
            name=str(g),
            boxpoints="all",
            marker_color=color_seq[i % len(color_seq)],
            line_color=color_seq[i % len(color_seq)],
            jitter=0,
            pointpos=0
        ))

    fig.update_layout(
        width=600,
        height=500,
        template="simple_white",
        showlegend=False,
        title={
            "text": f"{protein_col}" + (f" (z-scored by {normalize_by})" if normalize_by else ""),
            "x": 0.5,                   
            "xanchor": "center",        
            "yanchor": "top",          
            "font": {"size": 20}       
        },
        xaxis_title=group_col,
        yaxis_title="Intensity (z)" if normalize_by else "Intensity",
        title_x=0.5
    )

    return fig
