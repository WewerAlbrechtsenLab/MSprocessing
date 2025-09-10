import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os, tempfile, contextlib
from sklearn.decomposition import PCA
from scipy.stats import chi2
from pathlib import Path
from typing import Dict, Iterable, Optional
from combat.pycombat import pycombat
import umap
import pimmslearn.plotting.data
import pimmslearn.sampling
from pimmslearn.sklearn.ae_transformer import AETransformer
from pimmslearn.sklearn.cf_transformer import CollaborativeFilteringTransformer




def load_proteomes(path, clean_columns = False, extension : str = None):
    """
    Load proteomes from a given TSV file path. 

    path : path to the TSV file containing proteome data
    clean_columns : iuf True, cleans column names by removing directory paths
    extension : removes this extension from the column names 
    """ 
    df = pd.read_csv(path, sep="\t")

    if clean_columns:
        
        cleaned_cols = []
        for col in df.columns:
            if isinstance(col, str):
                base = os.path.basename(col)
                if extension and base.endswith(extension):
                    base = base[: -len(extension)]
                cleaned_cols.append(base)
            else:
                cleaned_cols.append(col)

        df.columns = cleaned_cols

    return df




def extract_meta_from_columns(
    df: pd.DataFrame,
    sep: str = "_",
    plate_nr: int = -2,
    position: int = -1,
    fields: Optional[Dict[str, int]] = None,
    restrict_to: Optional[Iterable[str]] = None
    ):
    """
    Parse column names of a cleaned proteome table into a metadata table.
    df : proteomes dataframe with cleanded columns to parse
    fields : a dictionary mapping field names to their index in the column name 
    restrict_to : a list of columns to restrict the parsing to, if None, all columns are parsed
    """

    cols = list(restrict_to) if restrict_to is not None else list(df.columns)


    field_map: Dict[str, int] = {"plate_nr": plate_nr, "position": position}
    if fields:
        field_map.update(fields)

    records = []
    for col in cols:
        parts = str(col).split(sep)
        row = {"column": col}

        invalid = False
        for fname, fidx in field_map.items():
            if not (-len(parts) <= fidx < len(parts)):
                print(f"WARNING: column skipped: {col}")
                invalid = True
                break

            token = parts[fidx]  
            row[fname] = token
        
        if invalid:
            continue


        pn = row.get("plate_nr", pd.NA)
        pos = row.get("position", pd.NA)
        row["sample"] = (
            f"{pn}_{pos}" if (pd.notna(pn) and pd.notna(pos)) else pd.NA
        )

        records.append(row)

    meta = pd.DataFrame.from_records(records)

    for c in [c for c in meta.columns if c not in {"column"}]:
        meta[c] = meta[c].astype("string")

    return meta




def rename_columns_to_sample(
    df: pd.DataFrame,
    sep: str = "_",
    plate_nr: int = -2,
    position: int = -1,
    fields: Optional[Dict[str, int]] = None,
    restrict_to: Optional[Iterable[str]] = None):
    """
    Rename selected columns of a proteome table to their derived sample IDs.
    The default sample ID is constructed as "<plate_nr>_<position>",
    assuming that these are respectively at positions -2 and -1`.

    df : input dataframe (with cleaned column names)
    sep : str, default "_", separator used to split column names
    plate_nr : positional index of the plate number token.
    position : positional index of the plate position token.
    fields : optional mapping of token names to their positional indices
    restrict_to : columns to attempt to rename. If None, all columns are considered.
    """
    cols = list(restrict_to) if restrict_to is not None else list(df.columns)

    rename_map: Dict[str, str] = {}
    seen: Dict[str, int] = {}

    for col in cols:
        parts = str(col).split(sep)

        #range checks
        if not (-len(parts) <= plate_nr < len(parts)) or not (-len(parts) <= position < len(parts)):
            print(f"WARNING: column skipped (insufficient tokens): {col}")
            continue

        pn = parts[plate_nr]
        pos = parts[position]

        if pd.isna(pn) or pd.isna(pos):
            print(f"WARNING: column skipped (missing tokens): {col}")
            continue

        sample = f"{pn}_{pos}"

        target = sample
        if target in seen:
            seen[target] += 1
            target = f"{sample}.{seen[sample]}"
        else:
            seen[target] = 1

        rename_map[col] = target

    if rename_map:
        df = df.rename(columns=rename_map)

    return df




def make_long_df(proteome: pd.DataFrame, meta: pd.DataFrame, merge_on: str = "sample_name", proteome_index = None):
    """
    Create a long-format proteome dataframe by merging metadata and proteome data

    proteome : wide-format proteome matrix 
    meta : metadata dataframe
    merge_on : metadata column to merge on, default "sample_name"

    """
    if proteome_index is None:
        proteome_index = [proteome.columns[0]]
        
    pg_long = proteome.melt(id_vars=proteome_index, var_name=merge_on, value_name="LFQ")
    pg_long = pg_long.dropna(subset=["LFQ"])

    pg_long = pd.merge(meta, pg_long, on=merge_on)

    return pg_long




def extract_counts(df: pd.DataFrame):
    """
    Aggregate protein counts per sample and sort.

    df : long-format dataframe with columns 'sample_name' and 'sample_type'
    """

    count_df = (
        df.groupby(["sample_name", "sample_type"])
          .size()
          .reset_index(name="proteins")
          .sort_values(["sample_type", "proteins"], ascending=[False, False])
    )
    return count_df




def plot_protein_counts(df, save_img=False, save_path="protein_counts.png"):
    """
    Count proteins per sample and plot a bar chart

    df : long-format dataframe with columns 'sample_name', 'sample_type'.
    save_img : if True, save the figure to `save_path`
    save_path : path to save the figure (.html always works; .png/.svg need kaleido)
    """
    count_df = extract_counts(df)

    fig = px.bar(
        count_df,
        x="sample_name", y="proteins", color="sample_type",
        template="simple_white",
        hover_name="sample_name",
        width=1000,
        height=500
    )
    fig.update_xaxes(showticklabels=False)

    if save_img:
        if save_path.endswith((".png", ".svg")):
            fig.write_image(save_path)  # needs kaleido
        elif save_path.endswith(".pdf"):
            fig.write_image(save_path) 
        else:
            fig.write_html(save_path)

    fig.show()




def protein_count_histogram(df, save_img=False, save_path="protein_hist.html"):
    """
    Plot histogram of protein counts per sample with boxplot margins.

    df : long-format dataframe with columns 'sample_name', 'sample_type'.
    save_img : if True, save the figure to `save_path`
    save_path : path to save the figure (.html always works; .png/.svg need kaleido)
    """
    count_df = extract_counts(df)

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

    if save_img:
        if save_path.endswith((".png", ".svg")):
            fig.write_image(save_path)  #kaleido issue
        else:
            fig.write_html(save_path)

    fig.show()




def protein_count_boxplot(df, save_img=False, save_path="protein_box.html"):
    """
    Plot boxplot of protein counts per sample type.

    df : long-format dataframe with columns 'sample_name', 'sample_type'.
    save_img : if True, save the figure to `save_path`
    save_path : path to save the figure (.html always works; .png/.svg need kaleido)
    """
    count_df = extract_counts(df)
    
    fig = px.box(
        count_df,
        x="sample_type",
        y="proteins",
        template="simple_white",
        hover_name="sample_type",
        hover_data=["proteins", "sample_type", "sample_name"],
        width=600,
        height=500
    )

    if save_img:
        if save_path.endswith((".png", ".svg")):
            fig.write_image(save_path)  #kaleido issue
        else:
            fig.write_html(save_path)

    fig.show()




def filter_low_count_outliers(
    df: pd.DataFrame,
    sample_level: str = "sample",
    k: float = 1.5):
    """
    Filter proteome dataframe by excluding samples whose protein counts 
    are below Q1 - k*IQR, computed within the specified sample level.

    df : long-format dataframe with columns 'sample_name' and 'sample_type'.
    sample_level : level of 'sample_type' used to compute the IQR-based threshold (default "sample").
    k : multiplier for IQR in Tukey's rule, default 1.5
    """
    count_df = (
        df.groupby(["sample_name", "sample_type"])
          .size().reset_index(name="proteins")
    )

    sub = count_df.loc[count_df["sample_type"] == sample_level]
    if sub.empty:
        raise ValueError(f"No rows with sample_type == '{sample_level}' to compute IQR.")

    Q1 = sub["proteins"].quantile(0.25)
    Q3 = sub["proteins"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR

    exclude = count_df.loc[count_df["proteins"] <= lower_bound, "sample_name"].unique()

    return df.loc[~df["sample_name"].isin(exclude)]





def plot_dynamic_range(pg_long: pd.DataFrame, save_img: bool = False, save_path: str = "dynamic_range.html"):
    """
    Build a dynamic range summary and plot from a long proteomics dataframe.

    pg_long : dataframe with columns 'sample_name', 'sample_type', 'protein', 'LFQ'.
    save_img : if True, save the figure to `save_path`
    save_path : path to save the figure (.html always works; .png/.svg need kaleido)
    """
    wide = pg_long.pivot(index=["sample_name", "sample_type"], columns="protein", values="LFQ")
    wide = wide.replace(0, np.nan)
    wide = np.log2(wide)

    mean_by_group = (wide.reset_index().groupby("sample_type", dropna=False)
                         .mean(numeric_only=True)
                         .reset_index()
                         .melt(id_vars="sample_type", var_name="name", value_name="LFQ_intensity")
                         .rename(columns={"sample_type": "group"}))

    comp_by_group = (wide.notna()
                         .reset_index()
                         .groupby("sample_type", dropna=False)
                         .mean(numeric_only=True)
                         .reset_index()
                         .melt(id_vars="sample_type", var_name="name", value_name="completeness")
                         .rename(columns={"sample_type": "group"}))

    dr_df = (pd.merge(mean_by_group, comp_by_group, on=["group", "name"], how="inner")
               .sort_values(["group", "LFQ_intensity"], ascending=[True, False]))
    dr_df["rank"] = dr_df.groupby("group")["LFQ_intensity"].cumcount() + 1

    fig = px.scatter(
        dr_df,
        x="rank",
        y="LFQ_intensity",
        color="completeness",
        hover_name="name",
        hover_data=["name", "LFQ_intensity", "rank", "completeness"],
        template="simple_white",
        labels={"LFQ_intensity": "log2(LFQ)", "rank": "Proteins ranked by abundance", "completeness": "Completeness"},
        title="Dynamic range",
        facet_col="group",
        facet_col_wrap=3,
        width=1000,
        height=500
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.update_xaxes(showline=True, linewidth=2, linecolor="lightgrey", gridcolor="lightgrey",
                     zeroline=True, zerolinecolor="lightgrey")
    fig.update_yaxes(showline=True, linewidth=2, linecolor="lightgrey", gridcolor="lightgrey")

    if save_img:
        ext = (save_path.rsplit(".", 1)[-1].lower() if "." in save_path else "")
        if ext in {"png", "svg", "pdf"}:
            fig.write_image(save_path)  #KALEIDO
        else:
            fig.write_html(save_path, include_plotlyjs="cdn")

    fig.show()





def filter_missingness(df, feat_prevalence=.2, axis=0):
    """ 
    Helper function to filter rows or columns of a dataframe
    """
    N = df.shape[axis]
    minimum_freq = N * feat_prevalence
    freq = df.notna().sum(axis=axis)
    mask = freq >= minimum_freq
    print(f"Drop {(~mask).sum()} along axis {axis}.")
    freq = freq.loc[mask]
    if axis == 0:
        df = df.loc[:, mask]
    else:
        df = df.loc[mask]
    return df




def iterative_missingness_filter(
    df: pd.DataFrame,
    feat_prevalence: float = 0.6,    #keep columns present in ≥60% of samples
    sample_prevalence: float = 0.6,  #keep rows present in ≥60% of features
    max_iter: int = 10,
    verbose: bool = True):
    """
    Iteratively filter a wide matrix by missingness until convergence.

    df : wide matrix (rows = samples, columns = features).
    feat_prevalence : minimum fraction of non-missing values required to keep a column.
    sample_prevalence : minimum fraction of non-missing values required to keep a row.
    max_iter : maximum number of alternating column/row filter passes.
    """
    df = df.copy()
    for it in range(1, max_iter + 1):
        prev_shape = df.shape

        df = filter_missingness(df, feat_prevalence=feat_prevalence, axis=0)
        if df.empty:
            if verbose: print(f"Iteration {it}: empty after column filter.")
            break

        df = filter_missingness(df, feat_prevalence=sample_prevalence, axis=1)
        if df.empty:
            if verbose: print(f"Iteration {it}: empty after row filter.")
            break

        if df.shape == prev_shape:
            if verbose: print(f"Converged in {it} iteration(s). Final shape: {df.shape}.")
            break
    else:
        if verbose: print(f"Reached max_iter={max_iter}. Final shape: {df.shape}.")

    return df






def configure_fastai_local_tmp(scratch: Path | str = None) -> Path:
    """
    Route temp files and FastAI checkpoints to a local writable folder.
    Call this once per session before imputation training if you are on a network drive.
    Returns the scratch path that was configured.
    """
    from fastai.learner import defaults
    scratch_path = Path(scratch) if scratch else Path.home() / "fastai_scratch"
    scratch_path.mkdir(parents=True, exist_ok=True)

    os.environ["TMP"] = str(scratch_path)
    os.environ["TEMP"] = str(scratch_path)
    tempfile.tempdir = str(scratch_path)

    defaults.path = scratch_path
    defaults.model_dir = "."
    return scratch_path





@contextlib.contextmanager
def chdir(path: Path):
    """
    Context manager to temporarily change the working directory.
    """
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)




#######

def impute_cf(
    df_filtered: pd.DataFrame,
    epochs: int = 20,
    run_locally: bool = False,
    local_run: Path | None = None,
    non_train = 0.1,
    mnar = 0.5):
    """
    Impute missing values in a wide matrix df_filtered using PIMMSlearn collaborative filtering model.
    df_filtered : wide matrix (rows = samples, columns = features) with missing values to impute.
    epochs : maximum number of training epochs.
    run_locally : if True, change to local_run directory before fitting the model.
    """
    index_cols = list(df_filtered.index.names)
    meta = df_filtered.reset_index()[index_cols]   # keep all metadata, incl. sample_name
    df_work = (df_filtered.reset_index().drop([c for c in index_cols if c != "sample_name"], axis=1).set_index("sample_name"))
    
    proteomes_stack = df_work.stack().to_frame("intensity")
    proteomes_stack.index.set_names(["Sample ID", "protein group"], inplace=True)

    splits, _, _, _ = pimmslearn.sampling.sample_mnar_mcar(
        df_long=proteomes_stack,
        frac_non_train=non_train,
        frac_mnar=mnar,
        random_state=0,
    )
    splits = pimmslearn.sampling.check_split_integrity(splits)

    cf_model = CollaborativeFilteringTransformer(
        target_column="intensity",
        sample_column="Sample ID",
        item_column="protein group",
        out_folder="runs/scikit_interface",
    )

    if run_locally:
        local_run = local_run or (Path.home() / "fastai_run_cwd")
        local_run.mkdir(parents=True, exist_ok=True)
        with chdir(local_run):
            cf_model.fit(splits.train_X, splits.val_y, cuda=False, epochs_max=epochs)
    else:
        cf_model.fit(splits.train_X, splits.val_y, cuda=False, epochs_max=epochs)

    imputed = cf_model.transform(proteomes_stack).unstack()
    imputed.index.name = "sample_name"
    imputed.columns.name = "protein group"

    imputed = meta.merge(imputed, on="sample_name").set_index(index_cols)

    return imputed




def impute_vae(
    df_filtered: pd.DataFrame,
    epochs: int = 50,
    run_locally: bool = False,
    local_run: Path | None = None,
    frac: float = 0.1,
    random_state: int = 0):
    """
    Impute missing values in a wide matrix df_filtered using a PIMMS VAE.
    df_filtered : wide matrix (rows = samples, columns = features) with missing values to impute.
    epochs : maximum number of training epochs.
    run_locally : if True, change to local_run directory before fitting the model.
    """
    index_cols = list(df_filtered.index.names)
    meta = df_filtered.reset_index()[index_cols]   # keep all metadata, incl. sample_name
    df_work = (df_filtered.reset_index().drop([c for c in index_cols if c != "sample_name"], axis=1).set_index("sample_name"))

    val_X, train_X = pimmslearn.sampling.sample_data(
        df_work.stack(),
        sample_index_to_drop=0,
        weights=df_work.notna().sum(),
        frac=frac,
        random_state=random_state,
    )
    val_X, train_X = val_X.unstack(), train_X.unstack()
    val_X = pd.DataFrame(pd.NA, index=train_X.index, columns=train_X.columns).fillna(val_X)

    model = AETransformer(
        model="VAE",
        hidden_layers=[512],
        latent_dim=50,
        out_folder="runs/scikit_interface",
        batch_size=10,
    )

    if run_locally:
        local_run = local_run or (Path.home() / "fastai_run_cwd")
        local_run.mkdir(parents=True, exist_ok=True)
        with chdir(local_run):
            model.fit(train_X, val_X, epochs_max=epochs, cuda=False)
    else:
        model.fit(train_X, val_X, epochs_max=epochs, cuda=False)

    pred_on_train_mask = model.transform(train_X)
    mask = val_X.notna()
    val_mse = ((pred_on_train_mask[mask] - val_X[mask])**2).mean().mean()
    print(f"Masked validation MSE: {float(val_mse):.6f}") #validation MSE on masked values

    imputed =  model.transform(df_work.copy()).reindex_like(df_work)
    imputed.index.name = "sample_name"
    imputed.columns.name = "protein group"
    imputed = meta.merge(imputed, on="sample_name").set_index(index_cols)

    return imputed



def impute_dae(
    df_filtered: pd.DataFrame,
    epochs: int = 50,
    run_locally: bool = False,
    local_run: Path | None = None,
    frac: float = 0.1,
    random_state: int = 0):
    """
    Impute missing values in a wide matrix df_filtered using a PIMMS DAE.
    df_filtered : wide matrix (rows = samples, columns = features) with missing values to impute.
    epochs : maximum number of training epochs.
    run_locally : if True, change to local_run directory before fitting the model.
    """
    index_cols = list(df_filtered.index.names)
    meta = df_filtered.reset_index()[index_cols]   # keep all metadata, incl. sample_name
    df_work = (df_filtered.reset_index().drop([c for c in index_cols if c != "sample_name"], axis=1).set_index("sample_name"))


    val_X, train_X = pimmslearn.sampling.sample_data(
        df_work.stack(),
        sample_index_to_drop=0,
        weights=df_work.notna().sum(),
        frac=frac,
        random_state=random_state,
    )
    val_X, train_X = val_X.unstack(), train_X.unstack()
    val_X = pd.DataFrame(pd.NA, index=train_X.index, columns=train_X.columns).fillna(val_X)

    model = AETransformer(
        model="DAE",
        hidden_layers=[512],
        latent_dim=50,
        out_folder="runs/scikit_interface",
        batch_size=10,
    )

    if run_locally:
        local_run = local_run or (Path.home() / "fastai_run_cwd")
        local_run.mkdir(parents=True, exist_ok=True)
        with chdir(local_run):
            model.fit(train_X, val_X, epochs_max=epochs, cuda=False)
    else:
        model.fit(train_X, val_X, epochs_max=epochs, cuda=False)

    pred_on_train_mask = model.transform(train_X)
    mask = val_X.notna()
    val_mse = ((pred_on_train_mask[mask] - val_X[mask])**2).mean().mean()
    print(f"Masked validation MSE: {float(val_mse):.6f}") #validation MSE on masked values

    imputed =  model.transform(df_work.copy()).reindex_like(df_work)
    imputed.index.name = "sample_name"
    imputed.columns.name = "protein group"
    imputed = meta.merge(imputed, on="sample_name").set_index(index_cols)

    return imputed




def z_score_normalization(df):
    return df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)




def remove_zscore_outliers(
    df: pd.DataFrame,
    z_thresh: float = 3.0):
    """
    Remove rows whose row-level statistic has a z-score > z_thresh.
    Row statistic is the mean (default) or median across columns.
    """
    row_means = df.mean(axis=1)
    mu = row_means.mean()
    sd = row_means.std(ddof=1)

    z = (row_means - mu) / sd
    outliers = df.index[z.abs() > z_thresh]
    print("Outliers:", outliers.tolist())
    return df.drop(index=outliers)



###################
def plot_sample_violins(
    df: pd.DataFrame,
    z_thresh: float = 3.0,
    save_img: bool = False,
    save_path: str = "sample_violins.html",):
    """
    Violin plot of per-sample (row) intensity distributions.
    Outliers (by row-mean z-score) are highlighted.
    Assumes df has no NAs; rows=samples, cols=proteins/features.
    """
    index_cols = list(df.index.names)
    df = (df.reset_index().drop([c for c in index_cols if c != "sample_name"], axis=1).set_index("sample_name"))

    row_means = df.mean(axis=1)
    mu, sd = row_means.mean(), row_means.std(ddof=1)
    if sd == 0 or np.isclose(sd, 0.0):
        is_outlier = pd.Series(False, index=df.index)
    else:
        z = (row_means - mu) / sd
        is_outlier = z.abs() > z_thresh

    out_first = pd.Index(
        list(row_means[is_outlier].sort_values(ascending=False).index)
        + list(row_means[~is_outlier].sort_values(ascending=False).index)
    )

    idx_col = df.index.name or "sample"
    long = df.reset_index().melt(id_vars=[idx_col], var_name="protein", value_name="intensity")
    long = long.rename(columns={idx_col: "sample"})
    long["is_outlier"] = long["sample"].map(is_outlier)
    long["sample"] = pd.Categorical(long["sample"], categories=out_first, ordered=True)

    fig = px.violin(
        long,
        x="sample",
        y="intensity",
        color="is_outlier",
        color_discrete_sequence=["#4c78a8", "#f58518"],  # optional: consistent, subtle
        box=True,
        points=False,
        template="simple_white",
        hover_data=["protein", "intensity"],
        title=f"Per-sample intensity distributions (|z| > {z_thresh} flagged)",
        width=600,
        height=500
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))

    if save_img:
        if save_path.endswith((".png", ".svg", ".pdf")):
            fig.write_image(save_path)  # requires 'kaleido' installed
        else:
            fig.write_html(save_path)

    fig.show()
    print("Outliers:", row_means.index[is_outlier].tolist())






def batch_correction(df: pd.DataFrame):
    """
    Batch correction using pycombat.
    df : wide matrix with missing values imputed.
    """
    batch_labels = df.reset_index()['plate_nr'].tolist()
    proteomes_combat = pycombat(df.T, batch_labels).T
    return proteomes_combat





def sample_density_heatmap(
    df: pd.DataFrame,
    save_img: bool = False,                         
    save_path: str = "sample_density_heatmap.html",   
):
    """
    Plot per-sample intensity histograms as a heatmap (rows=samples, cols=intensity bins, color=counts).

    df : wide matrix with samples in the index (must include 'sample_name') and features as columns; no NaNs.
    save_img : write the figure to save_path when True.
    save_path : destination file for the figure.
    """
    bins = 50

    index_cols = list(df.index.names or [])
    dfn = (df.reset_index()
             .drop([c for c in index_cols if c != "sample_order"], axis=1)
             .set_index("sample_order")
             .sort_index())

    vmin = float(dfn.min().min())
    vmax = float(dfn.max().max())
    edges = np.linspace(vmin, vmax, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    vals = dfn.to_numpy(dtype=float, copy=False)
    H = np.empty((vals.shape[0], bins), dtype=float)
    for i in range(vals.shape[0]):
        H[i], _ = np.histogram(vals[i], bins=edges)

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
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))

    if save_img:
        if save_path.endswith((".png", ".svg", ".pdf")):
            fig.write_image(save_path)
        else:
            fig.write_html(save_path)

    fig.show()






def plot_sample_means(
    df: pd.DataFrame,
    save_img: bool = False, 
    save_path: str = "sample_means.html",
):
    """
    Plot per-sample means ordered by sample_order, with a global horizontal mean line and a trendline.

    df : wide matrix with samples in the index (must include 'sample_order'); features as columns; numeric values.
    save_img : write the figure to save_path when True.
    save_path : destination file for the figure.
    """

    index_cols = list(df.index.names or [])
    dfn = (df.reset_index()
             .drop([c for c in index_cols if c != "sample_order"], axis=1)
             .set_index("sample_order")
             .sort_index())

    means = dfn.mean(axis=1, numeric_only=True)   
    overall = float(means.mean()) 
    plot_df = pd.DataFrame({
        "sample_order": np.arange(1, len(means) + 1, dtype=int),
        "mean": means.to_numpy(),
    })

    # Try LOESS/OLS trendline via plotly express; fall back to rolling mean if unavailable
    try:
        fig = px.scatter(
            plot_df, x="sample_order", y="mean",
            trendline="lowess",  # requires statsmodels; switch to "ols" if preferred
            template="simple_white",
            labels={"sample_order": "sample_order", "mean": "mean intensity"},
            title="Per-sample mean intensity",
            width=600,
            height=500
        )
    except Exception:
        fig = px.scatter(
            plot_df, x="sample_order", y="mean",
            template="simple_white",
            labels={"sample_order": "sample_order", "mean": "mean intensity"},
            title="Per-sample mean intensity",
            width=600,
            height=500
        )
        roll = plot_df["mean"].rolling( nine := 9, center=True, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=plot_df["sample_order"], y=roll, mode="lines", name="trend"))

    x0, x1 = int(plot_df["sample_order"].min()), int(plot_df["sample_order"].max())
    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[overall, overall],
            mode="lines",
            name=f"overall mean = {overall:.3g}",
            line=dict(dash="solid", width=2, color="red"),
        ))

    fig.update_xaxes(showticklabels=False, title_text="sample_order")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10),
                      showlegend=False)

    if save_img:
        if save_path.endswith((".png", ".svg", ".pdf")):
            fig.write_image(save_path)
        else:
            fig.write_html(save_path)

    fig.show()



def plot_rowmean_outliers(
    df: pd.DataFrame,
    z_thresh: float = 3.0,                         # |z| above this is flagged
    save_img: bool = False,                        # save figure if True
    save_path: str = "rowmean_outliers.html",      # .html works; .png/.svg/.pdf need kaleido
):
    """
    Plot z-scores of per-sample means, ordered by z, with threshold lines and outliers highlighted.

    df : wide matrix with samples in the index (ideally includes 'sample_name'); features as columns.
    z_thresh : absolute z-score cutoff used to flag outliers.
    save_img : write the figure to save_path when True.
    save_path : destination file for the figure.
    """

    # Reduce to 'sample_name' index if present; otherwise keep current index
    index_cols = list(df.index.names or [])
    if "sample_name" in index_cols:
        dfn = (df.reset_index()
                 .drop([c for c in index_cols if c != "sample_name"], axis=1)
                 .set_index("sample_name"))
    else:
        dfn = df.copy()
        dfn.index.name = dfn.index.name or "sample"

    means = dfn.mean(axis=1, numeric_only=True)
    mu, sd = float(means.mean()), float(means.std(ddof=1))
    if not np.isfinite(sd) or sd == 0:
        raise ValueError("Standard deviation of row means is zero or non-finite; cannot compute z-scores.")

    z = (means - mu) / sd
    order = z.sort_values()  # ascending
    outlier_mask = order.abs() > z_thresh

    plot_df = pd.DataFrame({
        "sample_order": np.arange(1, len(order) + 1, dtype=int),
        "z": order.to_numpy(),
        "sample": order.index.astype(str),
        "outlier": np.where(outlier_mask.values, "outlier", "inlier"),
    })

    fig = px.scatter(
        plot_df,
        x="sample_order",
        y="z",
        color="outlier",
        hover_name="sample",
        hover_data={"sample_order": True, "z": ":.3f", "outlier": True},
        color_discrete_map={"inlier": "#9aa0a6", "outlier": "#d62728"},
        template="simple_white",
        labels={"sample_order": "sample_order", "z": "z-score (row mean)"},
        title="Per-sample mean z-scores and outliers",
        width=600,
        height=500
    )

    # Threshold lines and zero line
    for y0, dash in [(0.0, "dot"), (z_thresh, "dash"), (-z_thresh, "dash")]:
        fig.add_hline(y=y0, line_dash=dash, line_color="lightgrey")

    # Hide x tick labels for dense plots
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))

    if save_img:
        if save_path.endswith((".png", ".svg", ".pdf")):
            fig.write_image(save_path)
        else:
            fig.write_html(save_path)

    fig.show()




def _hotelling_outlier_mask(df: pd.DataFrame, var_explained: float = 0.90, alpha: float = 0.99):
    """
    Compute Hotelling-style outliers.

    df : rows=samples, cols=features; numeric, pre-imputed/logged.
    var_explained : cumulative variance threshold to choose number of PCs (0–1).
    alpha : tail probability for cutoff (e.g., 0.99).

    Returns
    -------
    mask : pd.Series[bool]  True for outliers
    D2   : np.ndarray       squared Mahalanobis distances in PC space
    k    : int              number of PCs used
    """
    X = df.to_numpy(dtype=float, copy=False)

    # Drop zero-variance columns to avoid division by zero
    col_std = X.std(axis=0, ddof=1)
    keep = col_std > 0
    if not np.any(keep):
        raise ValueError("All features have zero variance.")
    X = X[:, keep]
    col_mean = X.mean(axis=0)
    col_std = X.std(axis=0, ddof=1)
    Xz = (X - col_mean) / col_std

    pca = PCA()
    Z = pca.fit_transform(Xz)
    var_cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(var_cum, var_explained) + 1)
    k = max(1, min(k, Z.shape[1]))

    lam = pca.explained_variance_[:k]            # variances of PCs
    Zk = Z[:, :k]
    D2 = (Zk**2 / lam).sum(axis=1)               # squared Mahalanobis in PC space

    cut = chi2.ppf(alpha, df=k)                  # chi-square approximation
    mask = pd.Series(D2 > cut, index=df.index)
    return mask, D2, k




def plot_hotelling_outliers(df: pd.DataFrame, var_explained: float = 0.90, alpha: float = 0.99,
                            save_img: bool = False, save_path: str = "hotelling_outliers.html"):
    """
    Ranked D² plot with cutoff line.

    df : rows=samples, cols=features.
    var_explained : cumulative variance threshold for PCs.
    alpha : cutoff quantile for χ²_k.
    save_img : save figure if True.
    save_path : output path (.html; .png/.svg/.pdf need kaleido).
    """
    _, D2, k = _hotelling_outlier_mask(df, var_explained=var_explained, alpha=alpha)
    order = np.argsort(D2)
    D2_sorted = D2[order]
    idx_sorted = df.index.to_numpy()[order]

    cutoff = chi2.ppf(alpha, df=k)

    plot_df = pd.DataFrame({
        "sample_order": np.arange(1, len(D2_sorted) + 1),
        "D2": D2_sorted,
        "sample": idx_sorted,
        "outlier": np.where(D2_sorted > cutoff, "outlier", "inlier"),
    })

    fig = px.scatter(
        plot_df, x="sample_order", y="D2", color="outlier",
        hover_name="sample", hover_data={"sample_order": True, "D2": ":.3f"},
        color_discrete_map={"inlier": "#9aa0a6", "outlier": "#d62728"},
        template="simple_white",
        labels={"sample_order": "sample_order", "D2": "D² (Mahalanobis in PC space)"},
        title=f"Hotelling-style outliers (k={k} PCs, α={alpha})",
        width=600,
        height=500
    )

    # Horizontal cutoff line as a trace (robust across renderers)
    fig.add_trace(go.Scatter(
        x=[1, len(D2_sorted)], y=[cutoff, cutoff],
        mode="lines", name=f"cutoff (χ²₍{k}₎ @ {alpha:.2f})", line=dict(dash="dash")
    ))

    fig.update_xaxes(showticklabels=False)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))

    if save_img:
        if save_path.endswith((".png", ".svg", ".pdf")):
            fig.write_image(save_path)
        else:
            fig.write_html(save_path)
    fig.show()



def remove_hotelling_outliers(df: pd.DataFrame, var_explained: float = 0.90, alpha: float = 0.99):
    """
    Remove outlier rows per Hotelling-style criterion.

    df : rows=samples, cols=features.
    var_explained : cumulative variance threshold for PCs.
    alpha : cutoff quantile for χ²_k.

    Returns
    -------
    df_kept : pd.DataFrame  rows not flagged as outliers
    out_idx : pd.Index      indices of removed rows
    """
    mask, _, _ = _hotelling_outlier_mask(df, var_explained=var_explained, alpha=alpha)
    out_idx = df.index[mask.to_numpy()]
    df_kept = df.drop(index=out_idx)
    print(f"Removed samples: {out_idx}")
    return df_kept





def plot_umap(df: pd.DataFrame, color_by: str, save_img = False, save_path="UMAP.png"):
    """
    Generate an interactive UMAP plot from a proteomics dataframe.

    df: dataframe with MultiIndex rows (metadata) and numeric columns (proteomic features).
    color_by: name of the index level to color points by.
    """
    if color_by not in df.index.names:
        raise ValueError(f"'{color_by}' not found in index levels: {df.index.names}")

    df.index = df.index.set_levels(df.index.levels[df.index.names.index(color_by)].astype(str), level=color_by)
    color_labels = df.index.get_level_values(color_by)

    reducer = umap.UMAP(random_state=1)
    embedding = reducer.fit_transform(df.values)

    umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"], index=df.index)
    umap_df[color_by] = color_labels

    fig = px.scatter(
        umap_df,
        x="UMAP1",
        y="UMAP2",
        color=color_by,
        title=f"UMAP projection colored by {color_by}",
        width=600,
        height=500
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8), selector=dict(mode='markers'))
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    if save_img:
        if save_path.endswith((".png", ".svg")):
            fig.write_image(save_path)  #kaleido issue
        else:
            fig.write_html(save_path)

    fig.show()





def plot_sample_box(df):
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





def robust_zscore(row):
    med = np.median(row)
    mad = np.median(np.abs(row - med))
    if mad == 0:
        return np.zeros_like(row)
    return (row - med) / mad




def normalize_sample(
    df: pd.DataFrame,
    method: str = "median",  # "mean", "median", "cscore"
    round_digits: int = 3
):
    """
    Normalize a DataFrame so that all samples are adjusted based on the selected method.
    df : rows are samples, columns are features
    method : 
        - "mean": Normalize each sample to have the same mean.
        - "median": Normalize each sample to have the same median.
        - "cscore": Row-wise robust z-score (centered score).
    round_digits : number of decimal digits to round the result.
    """
    if method not in {"mean", "median", "cscore"}:
        raise ValueError("method must be one of: 'mean', 'median', 'cscore'")

    if method in {"mean", "median"}:
        if method == "mean":
            center_values = df.mean(axis=1)
            global_center = center_values.mean()
        else:
            center_values = df.median(axis=1)
            global_center = center_values.median()

        # Normalize to global center
        df_norm = df.div(center_values, axis=0).mul(global_center)

    elif method == "cscore":

        df_norm = df.apply(robust_zscore, axis=1, result_type="broadcast")

    return df_norm.round(round_digits)

    



def get_sample_order(df: pd.DataFrame) -> pd.Series:
    plate = df.index.get_level_values("plate_nr").astype(str).str.extract(r"(\d+)")[0].astype(int)
    pos   = df.index.get_level_values("plate_position").astype(str)

    rows = pos.str.extract(r"^([A-Za-z]+)")[0]
    cols = pos.str.extract(r"(\d+)$")[0].astype(int)
    

    row_num = rows.str.upper().apply(lambda s: sum((ord(c) - 64) * (26 ** i) for i, c in enumerate(s[::-1])))

    order_pos = np.lexsort((cols.to_numpy(), row_num.to_numpy(), plate.to_numpy()))
    ranks = np.empty(len(df), dtype=int)
    ranks[order_pos] = np.arange(1, len(df) + 1)

    return df.set_index(pd.Index(ranks, name="sample_order"), append=True)



