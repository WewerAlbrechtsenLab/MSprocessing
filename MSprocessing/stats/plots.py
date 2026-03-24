import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from matplotlib.lines import Line2D
from matplotlib_venn import venn2, venn3
from plotly.colors import qualitative
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler




def volcano_plot(
    results,
    alpha=0.05,
    labels=True,
    width=600,
    height=500,
    legend=False,
    x_range=None,
    y_range=None,
    up_color="#B65EAF",
    down_color="#009599",
    **kwargs,
):
    """
    Create an interactive volcano plot for differential expression results.

    Additional keyword arguments are passed to plotly.express.scatter.

    Parameters
    ----------
    results : pd.DataFrame
        Differential analysis results containing at least 'pval', 'padj',
        and either 'log2fc' or 'coef' columns.
    alpha : float, default=0.05
        Adjusted p-value significance threshold.
    labels : bool, default=True
        Whether to display labels for significant points.
    width : int, default=600
        Figure width in pixels.
    height : int, default=500
        Figure height in pixels.
    showlegend : bool, default=False
        Whether to display the legend.
    x_range : list | tuple | None, default=None
        Range for x-axis, e.g. [-2, 2].
    y_range : list | tuple | None, default=None
        Range for y-axis, e.g. [0, 10].

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = results.copy().reset_index()
    df = df.rename(columns={"index": "protein"})

    if "coef" in df.columns:
        df["log2fc"] = df["coef"]

    p = df["pval"].astype(float).clip(lower=np.finfo(float).tiny)
    df["-log10(p-value)"] = -np.log10(p)

    df["color"] = "non_sig"
    df.loc[(df["log2fc"] > 0) & (df["padj"] < alpha), "color"] = "up"
    df.loc[(df["log2fc"] < 0) & (df["padj"] < alpha), "color"] = "down"

    color_dict = {
        "non_sig": "#404040",
        "up": up_color,
        "down": down_color,
    }

    df["label"] = ""
    if labels:
        df.loc[df["color"].isin(["up", "down"]), "label"] = df["protein"]

    fig = px.scatter(
        df,
        x="coef" if "coef" in df.columns else "log2fc",
        y="-log10(p-value)",
        color="color",
        color_discrete_map=color_dict,
        hover_name="protein",
        text="label",
        template="simple_white",
        **kwargs,
    )

    fig.update_traces(textposition="top center", textfont=dict(size=9))
    fig.update_xaxes(showgrid=True, zeroline=False, range=x_range)
    fig.update_yaxes(showgrid=True, zeroline=False, range=y_range)
    fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color="#5c5c5c")

    fig.update_layout(
        showlegend=legend,
        width=width,
        height=height,
        xaxis_title="log2(FC)",
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






def plot_group_boxplot(meta, proteome, group_col, protein_col, normalize_by=None):
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



def plot_cluster_coefficients(
    df,
    up_color="#B65EAF",
    down_color="#009599",
):
    """
    Plot clustered pathway summaries along a mean fold-change axis.

    This function visualizes pathway clusters as a scatter plot where the
    x-axis represents the mean fold-change per cluster, point size reflects
    the number of proteins in the cluster, and point color indicates whether
    the mean fold-change is positive or negative.

    Parameters
    ----------
    df : pd.DataFrame
        Cluster summary table containing at least "mean_coef", "n_proteins",
        and "parent_name" columns.
    up_color : str, default="#B65EAF"
        Marker color used for clusters with positive mean fold-change.
    down_color : str, default="#009599"
        Marker color used for clusters with negative mean fold-change.

    Returns
    -------
    matplotlib.figure.Figure
        Scatter plot figure showing pathway clusters ordered by mean fold-change.
    """
    d = df.copy()
    d["coef"] = d["mean_coef"].astype(float)

    d = d.sort_values("coef", ascending=False).reset_index(drop=True)
    y = np.arange(len(d))
    n = d["n_proteins"].to_numpy(dtype=float)

    ax_width_in=6.0 
    left_pad_in=3.2 
    right_pad_in=4.0

    min_s, max_s = 30, 260
    nmin, nmax = float(np.nanmin(n)), float(np.nanmax(n))
    if nmax == nmin:
        sizes = np.full_like(n, (min_s + max_s) / 2.0, dtype=float)
    else:
        sizes = min_s + (n - nmin) / (nmax - nmin) * (max_s - min_s)

    coef = d["coef"].to_numpy(dtype=float)
    face = np.where(coef > 0, up_color, np.where(coef < 0, down_color, "#666666"))

    fig_w = ax_width_in + left_pad_in + right_pad_in
    fig_h = max(3, 0.35 * len(d))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    left = left_pad_in / fig_w
    right = 1.0 - (right_pad_in / fig_w)
    fig.subplots_adjust(left=left, right=right)

    ax.scatter(
        d["coef"], y,
        c=face, s=sizes,
        edgecolors="none", linewidths=0
    )

    ax.set_yticks(y)
    ax.set_yticklabels(d["parent_name"])
    ax.set_xlabel("mean log2 fold change")
    ax.set_ylim(-1, len(d))
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles_dir = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=up_color, markeredgecolor="none",
               markersize=8, label="Upregulated"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=down_color, markeredgecolor="none",
               markersize=8, label="Downregulated"),
    ]

    leg1 = ax.legend(
        handles=handles_dir,
        title="Direction",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0.0,
        alignment="left"
    )
    leg1._legend_box.align = "left"

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    leg1_bbox_px = leg1.get_window_extent(renderer=renderer)

    if nmax == nmin:
        vals = [int(nmin)]
    else:
        vals = [int(round(nmin)), int(round((nmin + nmax) / 2.0)), int(round(nmax))]

    handles_size = []
    for v in vals:
        if nmax == nmin:
            s = (min_s + max_s) / 2.0
        else:
            s = min_s + (v - nmin) / (nmax - nmin) * (max_s - min_s)

        handles_size.append(
            Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor=down_color,
                   markeredgecolor="none",
                   markersize=float(np.sqrt(s)),
                   label=str(v))
        )

    gap_px = 10
    x_disp, y_disp = ax.transAxes.transform((1.02, 1.0))
    y2_disp = leg1_bbox_px.y0 - gap_px
    x2_axes, y2_axes = ax.transAxes.inverted().transform((x_disp, y2_disp))

    leg2 = ax.legend(
        handles=handles_size,
        title="Number of proteins",
        loc="upper left",
        bbox_to_anchor=(1.02, y2_axes),
        frameon=False,
        borderaxespad=0.0,
        alignment="left"
    )
    leg2._legend_box.align = "left"

    ax.add_artist(leg1)
    return fig