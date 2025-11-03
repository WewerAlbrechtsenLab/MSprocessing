import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import umap
from MSprocessing.preprocessing.missingness import extract_counts




def plot_count_histogram(
    df: pd.DataFrame,
    sample_type_level: str = "sample_type"
) -> go.Figure:
    """
    Plot the number of detected proteins per sample as a bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame with samples as rows and proteins as columns.
    sample_type_level : str, default="sample_type"
        Index level used to color the bars by sample type.

    Returns
    -------
    go.Figure
        Plotly bar chart figure.
    """
    count_df = extract_counts(df, sample_type_level).sort_values("proteins", ascending=False)

    fig = px.bar(
        count_df,
        x="sample_name",
        y="proteins",
        color="sample_type",
        template="simple_white",
        hover_name="sample_name",
        width=1000,
        height=500
    )
    fig.update_xaxes(showticklabels=False)

    return fig




def plot_protein_counts(
    df: pd.DataFrame,
    sample_type_level: str = "sample_type"
) -> go.Figure:
    """
    Plot a histogram of protein counts per sample with boxplot margins.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame with samples as rows and proteins as columns.
    sample_type_level : str, default="sample_type"
        Index level used for faceting and coloring.

    Returns
    -------
    go.Figure
        Plotly histogram with marginal boxplots.
    """
    count_df = extract_counts(df, sample_type_level)

    fig = px.histogram(
        count_df,
        x="proteins",
        color="sample_type",
        marginal="box",
        hover_data=count_df.columns,
        facet_col="sample_type",
        facet_col_wrap=1,
        template="simple_white",
        width=1000,
        height=500
    )
    fig.update_layout(bargap=0.1)
    fig.update_xaxes(range=[0, count_df["proteins"].max()])
    fig.update_yaxes(matches=None)
    return fig




def plot_count_boxplot(
    df: pd.DataFrame, 
    k: float = 1.5
) -> go.Figure: 
    """
    Plot missingness per sample as a boxplot and mark Tukey outliers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with samples as rows.
    k : float, default=1.5
        Multiplier for Tukey’s IQR method.

    Returns
    -------
    go.Figure
        Boxplot of missingness across samples.
    """

    nan_fraction = df.isna().mean(axis=1) 
    Q1 = nan_fraction.quantile(0.25) 
    Q3 = nan_fraction.quantile(0.75) 
    IQR = Q3 - Q1 
    upper_bound = Q3 + k * IQR 
    plot_df = pd.DataFrame({ 
        "sample": nan_fraction.index.map(lambda x: "_".join(map(str, x))), 
        "missingness": nan_fraction.values 
        }) 
    fig = px.box( 
        plot_df, 
        y="missingness", 
        points="outliers", # only show outliers 
        hover_data=["sample"], 
        title="Per-sample missingness" 
        ) 
    fig.add_hline( 
        y=upper_bound, 
        line_dash="dash", 
        line_color="red", 
        annotation_text=f"Upper bound ({upper_bound:.2f})", 
        annotation_position="top left" 
        ) 
    fig.update_layout( 
        yaxis_title="Fraction missing", 
        width=600, 
        height=500 ) 
    return fig




def plot_dynamic_range(
    df: pd.DataFrame
) -> go.Figure:
    """
    Plot the dynamic range of protein abundances (log2 scale), colored by completeness.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame with numeric protein intensities.

    Returns
    -------
    go.Figure
        Scatter plot showing dynamic range and completeness.
    """
    df = df.replace(0, np.nan).astype(float)
    mean_intensity = np.log2(df).mean(axis=0, skipna=True)
    completeness = df.notna().mean(axis=0)

    dr_df = pd.DataFrame({
        "protein": mean_intensity.index,
        "LFQ_intensity": mean_intensity.values,
        "completeness": completeness.values
    }).dropna(subset=["LFQ_intensity"]).sort_values("LFQ_intensity", ascending=False)
    dr_df["rank"] = range(1, len(dr_df) + 1)

    fig = px.scatter(
        dr_df,
        x="rank",
        y="LFQ_intensity",
        color="completeness",
        hover_data=["protein", "LFQ_intensity", "completeness"],
        template="simple_white",
        labels={"LFQ_intensity": "log2(LFQ)", "rank": "Proteins ranked by abundance"},
        title="Dynamic range",
        width=800,
        height=500
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    return fig




def plot_sample_violins(
    df: pd.DataFrame,
    z_thresh: float = 3.0
) -> go.Figure:
    """
    Plot violin plots of per-sample intensity distributions, highlighting outliers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with rows as samples and columns as proteins.
    z_thresh : float, default=3.0
        Absolute z-score threshold for flagging outlier samples.

    Returns
    -------
    go.Figure
        Violin plot figure.
    """
    index_cols = list(df.index.names)
    df = df.reset_index().drop([c for c in index_cols if c != "sample_name"], axis=1).set_index("sample_name")

    row_means = df.mean(axis=1)
    mu, sd = row_means.mean(), row_means.std(ddof=1)
    z = (row_means - mu) / sd if sd > 0 else pd.Series(0, index=row_means.index)
    is_outlier = z.abs() > z_thresh

    ordered = list(row_means[is_outlier].sort_values(ascending=False).index) + \
              list(row_means[~is_outlier].sort_values(ascending=False).index)

    long = df.reset_index().melt(id_vars=["sample_name"], var_name="protein", value_name="intensity")
    long["is_outlier"] = long["sample_name"].map(is_outlier)
    long["sample_name"] = pd.Categorical(long["sample_name"], categories=ordered, ordered=True)

    fig = px.violin(
        long,
        x="sample_name",
        y="intensity",
        color="is_outlier",
        color_discrete_sequence=["#4c78a8", "#f58518"],
        box=True,
        points=False,
        template="simple_white",
        title=f"Per-sample intensity distributions (|z| > {z_thresh} flagged)",
        width=600,
        height=500
    )
    fig.update_xaxes(showticklabels=False)
    return fig




def plot_density_heatmap(
    df: pd.DataFrame
) -> go.Figure:
    """
    Plot a heatmap of per-sample intensity histograms.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numeric intensities, indexed by samples (must include 'sample_order').

    Returns
    -------
    go.Figure
        Heatmap of intensity distributions.
    """
    bins = 50
    index_cols = list(df.index.names or [])
    dfn = df.reset_index().drop([c for c in index_cols if c != "sample_order"], axis=1)
    dfn = dfn.set_index("sample_order").sort_index()

    vmin, vmax = float(dfn.min().min()), float(dfn.max().max())
    edges = np.linspace(vmin, vmax, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    vals = dfn.to_numpy(dtype=float, copy=False)
    H = np.array([np.histogram(row, bins=edges)[0] for row in vals])

    fig = px.imshow(
        H,
        x=np.round(centers, 3),
        y=dfn.index.astype(str),
        aspect="auto",
        origin="upper",
        color_continuous_scale="Viridis",
        template="simple_white",
        labels={"x": "intensity", "y": "sample", "color": "count"},
        title="Per-sample intensity distributions",
        width=600,
        height=500
    )
    fig.update_yaxes(showticklabels=False)
    return fig




def plot_sample_means(
    df: pd.DataFrame
) -> go.Figure:
    """
    Plot mean intensity per sample with overall mean and trend line.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame with samples in the index.

    Returns
    -------
    go.Figure
        Scatter plot of mean intensities.
    """
    index_cols = list(df.index.names or [])
    dfn = df.reset_index().drop([c for c in index_cols if c != "sample_order"], axis=1)
    dfn = dfn.set_index("sample_order").sort_index()

    means = dfn.mean(axis=1, numeric_only=True)
    overall = float(means.mean())

    plot_df = pd.DataFrame({
        "sample_order": np.arange(1, len(means) + 1),
        "mean": means.values
    })

    fig = px.scatter(
        plot_df,
        x="sample_order",
        y="mean",
        trendline="lowess",
        template="simple_white",
        labels={"sample_order": "Sample order", "mean": "Mean intensity"},
        title="Per-sample mean intensity",
        width=600,
        height=500
    )
    fig.add_hline(y=overall, line=dict(dash="solid", color="red", width=2))
    fig.update_xaxes(showticklabels=False)
    return fig




def plot_sample_box(df):
    """
    Plot box plot of intensities per sample with overall mean and trend line.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame with samples in the index.

    Returns
    -------
    go.Figure
        Scatter plot of mean intensities.
    """
    if "sample_name" not in df.index.names:
        raise ValueError("'sample_order' must be one of the index levels.")

    df_sorted = df.reset_index().sort_values("sample_order")
    protein_cols = df.columns[df.columns.str.startswith("A") | df.columns.str.startswith("Q")]
    sample_names = df_sorted["sample_order"].astype(int).tolist()
    df_sorted = df_sorted.set_index("sample_order")
    
    fig = go.Figure()
    medians = []

    for i, sample in enumerate(sample_names):
        values = df_sorted.loc[sample, protein_cols]
        if isinstance(values, pd.Series):
            values = values.to_frame().T
        flat_values = values.to_numpy().flatten()
        fig.add_trace(go.Box(
            y=flat_values,
            x=[i] * len(flat_values),
            boxpoints=False,
            line=dict(width=1),
            marker=dict(size=3, color="gray"),
            showlegend=False,
            hovertext=[sample] * len(flat_values),
            hoverinfo="text+y"
        ))
        medians.append(np.nanmedian(flat_values))

    medians = np.array(medians)
    overall_median = float(np.nanmedian(medians))

    fig.add_trace(go.Scatter(
        x=[-0.5, len(sample_names) - 0.5],
        y=[overall_median, overall_median],
        mode="lines",
        name=f"overall median = {overall_median:.3g}",
        line=dict(dash="solid", width=2, color="red"),
    ))

    trend_df = pd.DataFrame({
        "sample_order": np.arange(len(medians)),
        "median": medians
    })

    try:
        trend_fig = px.scatter(trend_df, x="sample_order", y="median", trendline="lowess")
        trendline = trend_fig.data[1]
        trendline.name = "trend (median)"
        fig.add_trace(trendline)
    except Exception:
        roll = pd.Series(medians).rolling(9, center=True, min_periods=1).median()
        fig.add_trace(go.Scatter(
            x=trend_df["sample_order"],
            y=roll,
            mode="lines",
            name="trend (median)",
            line=dict(color="blue")
        ))

    fig.update_layout(
        title="Protein intensity distribution per sample",
        xaxis=dict(
            title="sample order",
            showticklabels=False,
            showgrid=False
        ),
        yaxis_title="protein intensity",
        template="simple_white",
        width=1000,
        height=500,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=80),
    )

    fig.show()



def plot_rowmean_outliers(
    df: pd.DataFrame,
    z_thresh: float = 3.0
) -> go.Figure:
    """
    Plot z-scores of per-sample mean intensities, highlighting outliers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with samples in the index and features as columns.
    z_thresh : float, default=3.0
        Absolute z-score cutoff for defining outliers.

    Returns
    -------
    go.Figure
        Scatter plot of z-scores and outliers.
    """
    index_cols = list(df.index.names or [])
    if "sample_name" in index_cols:
        dfn = df.reset_index().drop([c for c in index_cols if c != "sample_name"], axis=1).set_index("sample_name")
    else:
        dfn = df.copy()
        dfn.index.name = dfn.index.name or "sample"

    means = dfn.mean(axis=1, numeric_only=True)
    mu, sd = float(means.mean()), float(means.std(ddof=1))
    if not np.isfinite(sd) or sd == 0:
        raise ValueError("Standard deviation of row means is zero or non-finite; cannot compute z-scores.")

    z = (means - mu) / sd
    order = z.sort_values()
    outlier_mask = order.abs() > z_thresh

    plot_df = pd.DataFrame({
        "sample_order": np.arange(1, len(order) + 1),
        "z": order.values,
        "sample": order.index.astype(str),
        "outlier": np.where(outlier_mask, "outlier", "inlier")
    })

    fig = px.scatter(
        plot_df,
        x="sample_order",
        y="z",
        color="outlier",
        color_discrete_map={"inlier": "#9aa0a6", "outlier": "#d62728"},
        template="simple_white",
        labels={"sample_order": "Sample order", "z": "z-score"},
        title="Per-sample mean z-scores",
        width=600,
        height=500
    )
    for y0, dash in [(0, "dot"), (z_thresh, "dash"), (-z_thresh, "dash")]:
        fig.add_hline(y=y0, line_dash=dash, line_color="lightgrey")
    fig.update_xaxes(showticklabels=False)
    return fig




def plot_umap(
    df: pd.DataFrame,
    color_by: str
) -> go.Figure:
    """
    Plot a 2D UMAP projection of samples colored by metadata.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with MultiIndex rows and numeric proteomic features.
    color_by : str
        Index level used for coloring the points.

    Returns
    -------
    go.Figure
        2D UMAP scatter plot.
    """
    if color_by not in df.index.names:
        raise ValueError(f"'{color_by}' not found in index levels: {df.index.names}")

    color_labels = df.index.get_level_values(color_by)
    reducer = umap.UMAP(random_state=1)
    embedding = reducer.fit_transform(df.values)

    umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"], index=df.index)
    coerced = pd.to_numeric(color_labels, errors="coerce")

    color_arg = coerced if coerced.notna().all() else color_labels.astype(str)
    umap_df[color_by] = color_arg

    fig = px.scatter(
        umap_df,
        x="UMAP1",
        y="UMAP2",
        color=color_by,
        title=f"UMAP projection colored by {color_by}",
        width=600,
        height=500
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig
