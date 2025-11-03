import pandas as pd
import numpy as np
from pathlib import Path

import pimmslearn.sampling
from pimmslearn.sklearn.ae_transformer import AETransformer
from pimmslearn.sklearn.cf_transformer import CollaborativeFilteringTransformer

from MSprocessing.preprocessing.utils import set_global_seed, chdir



def impute_minprob(
    df: pd.DataFrame,
    downshift: float = 1.8,
    width: float = 0.3,
    random_state: int = 0
) -> pd.DataFrame:
    """
    Perform minimum probability (MinProb) imputation on a proteomics DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format matrix with samples as rows and proteins as columns. Must be numeric.
    downshift : float, default=1.8
        How far to shift the distribution downwards (in standard deviation units).
    width : float, default=0.3
        Width (standard deviation scaling) of the distribution used for imputation.
    random_state : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values imputed using the MinProb approach.
    """
    rng = np.random.default_rng(random_state)
    df_imputed = df.copy()

    for col in df.columns:
        observed = df[col].dropna()
        if observed.empty:
            continue

        mean = observed.mean()
        std = observed.std()

        mu = mean - downshift * std
        sigma = width * std

        n_missing = df[col].isna().sum()
        if n_missing > 0:
            imputed_values = rng.normal(loc=mu, scale=sigma, size=n_missing)
            df_imputed.loc[df[col].isna(), col] = imputed_values

    return df_imputed




def impute_cf(
    df_filtered: pd.DataFrame,
    epochs: int = 30,
    n_factors: int = 15,
    patience: int = 5,
    run_locally: bool = False,
    local_run: Path | None = None,
    non_train: float = 0.1,
    random_state: int = 0
) -> pd.DataFrame:
    """
    Impute missing values using the PIMMSlearn Collaborative Filtering (CF) model.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        Wide-format DataFrame (rows = samples, columns = features) with missing values.
    epochs : int, default=30
        Maximum number of training epochs.
    n_factors : int, default=15
        Number of latent factors in the CF model.
    patience : int, default=5
        Early stopping patience.
    run_locally : bool, default=False
        If True, temporarily change to `local_run` directory before training.
    local_run : Path or None, default=None
        Directory to use for local run.
    non_train : float, default=0.1
        Fraction of entries to hold out for validation/testing.
    random_state : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values imputed using collaborative filtering.
    """
    set_global_seed(random_state)

    # Keep metadata from MultiIndex
    index_cols = list(df_filtered.index.names)
    meta = df_filtered.reset_index()[index_cols]

    # Work on sample_name vs. proteins
    df_work = (
        df_filtered.reset_index()
        .drop([c for c in index_cols if c != "sample_name"], axis=1)
        .set_index("sample_name")
    )

    proteomes_stack = df_work.stack().to_frame("intensity")
    proteomes_stack.index.set_names(["Sample ID", "protein"], inplace=True)

    # Sampling
    train_series, non_train_series = pimmslearn.sampling.sample_data(
        proteomes_stack["intensity"],
        sample_index_to_drop=0,  # 0 = "Sample ID" level
        frac=1.0 - non_train,
        random_state=random_state,
    )

    # Build DataSplits manually
    splits = pimmslearn.io.datasplits.DataSplits(is_wide_format=False)
    splits.train_X = train_series
    splits.val_y = non_train_series.sample(frac=0.5, random_state=random_state)
    splits.test_y = non_train_series.drop(splits.val_y.index)
    splits = pimmslearn.sampling.check_split_integrity(splits)

    # Train CF model
    cf_model = CollaborativeFilteringTransformer(
        target_column="intensity",
        sample_column="Sample ID",
        item_column="protein",
        out_folder="runs/scikit_interface",
        n_factors=n_factors,
    )

    if run_locally:
        local_run = local_run or (Path.home() / "fastai_run_cwd")
        local_run.mkdir(parents=True, exist_ok=True)
        with chdir(local_run):
            cf_model.fit(
                splits.train_X,
                splits.val_y,
                cuda=False,
                epochs_max=epochs,
                patience=patience,
            )
    else:
        cf_model.fit(
            splits.train_X,
            splits.val_y,
            cuda=False,
            epochs_max=epochs,
            patience=patience,
        )

    # Transform back to wide format
    imputed = cf_model.transform(proteomes_stack).unstack()
    imputed.index.name = "sample_name"
    imputed.columns.name = "protein"

    # Merge metadata back
    imputed = meta.merge(imputed, on="sample_name").set_index(index_cols)

    return imputed




def impute_vae(
    df_filtered: pd.DataFrame,
    epochs: int = 50,
    patience: int = 5,
    hidden_layers: list[int] | None = None,
    latent_dim: int = 50,
    run_locally: bool = False,
    local_run: Path | None = None,
    frac: float = 0.1,
    random_state: int = 0
) -> pd.DataFrame:
    """
    Impute missing values using a PIMMS Variational Autoencoder (VAE).

    Parameters
    ----------
    df_filtered : pd.DataFrame
        Wide-format DataFrame (rows = samples, columns = features) with missing values.
    epochs : int, default=50
        Maximum number of training epochs.
    patience : int, default=5
        Early stopping patience.
    hidden_layers : list[int] or None, default=None
        Layer sizes for the autoencoder; defaults to [512].
    latent_dim : int, default=50
        Dimensionality of the latent space.
    run_locally : bool, default=False
        If True, temporarily change to `local_run` directory before training.
    local_run : Path or None, default=None
        Directory to use for local run.
    frac : float, default=0.1
        Fraction of entries used for validation.
    random_state : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values imputed using a VAE.
    """
    set_global_seed(random_state)

    index_cols = list(df_filtered.index.names)
    meta = df_filtered.reset_index()[index_cols]
    df_work = (
        df_filtered.reset_index()
        .drop([c for c in index_cols if c != "sample_name"], axis=1)
        .set_index("sample_name")
    )
    hidden_layers = hidden_layers or [512]

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
        hidden_layers=hidden_layers,
        latent_dim=latent_dim,
        out_folder="runs/scikit_interface",
        batch_size=10,
    )

    if run_locally:
        local_run = local_run or (Path.home() / "fastai_run_cwd")
        local_run.mkdir(parents=True, exist_ok=True)
        with chdir(local_run):
            model.fit(train_X, val_X, epochs_max=epochs, patience=patience, cuda=False)
    else:
        model.fit(train_X, val_X, epochs_max=epochs, patience=patience, cuda=False)

    pred_on_train_mask = model.transform(train_X)
    mask = val_X.notna()
    val_mse = ((pred_on_train_mask[mask] - val_X[mask]) ** 2).mean().mean()
    print(f"Masked validation MSE: {float(val_mse):.6f}")

    imputed = model.transform(df_work.copy()).reindex_like(df_work)
    imputed.index.name = "sample_name"
    imputed.columns.name = "protein"
    imputed = meta.merge(imputed, on="sample_name").set_index(index_cols)

    return imputed




def impute_dae(
    df_filtered: pd.DataFrame,
    epochs: int = 50,
    patience: int = 5,
    hidden_layers: list[int] | None = None,
    latent_dim: int = 50,
    run_locally: bool = False,
    local_run: Path | None = None,
    frac: float = 0.1,
    random_state: int = 0
) -> pd.DataFrame:
    """
    Impute missing values using a PIMMS Denoising Autoencoder (DAE).

    Parameters
    ----------
    df_filtered : pd.DataFrame
        Wide-format DataFrame (rows = samples, columns = features) with missing values.
    epochs : int, default=50
        Maximum number of training epochs.
    patience : int, default=5
        Early stopping patience.
    hidden_layers : list[int] or None, default=None
        Layer sizes for the autoencoder; defaults to [512].
    latent_dim : int, default=50
        Dimensionality of the latent space.
    run_locally : bool, default=False
        If True, temporarily change to `local_run` directory before training.
    local_run : Path or None, default=None
        Directory to use for local run.
    frac : float, default=0.1
        Fraction of entries used for validation.
    random_state : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values imputed using a DAE.
    """
    set_global_seed(random_state)

    index_cols = list(df_filtered.index.names)
    meta = df_filtered.reset_index()[index_cols]
    df_work = (
        df_filtered.reset_index()
        .drop([c for c in index_cols if c != "sample_name"], axis=1)
        .set_index("sample_name")
    )
    hidden_layers = hidden_layers or [512]

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
        hidden_layers=hidden_layers,
        latent_dim=latent_dim,
        out_folder="runs/scikit_interface",
        batch_size=10,
    )

    if run_locally:
        local_run = local_run or (Path.home() / "fastai_run_cwd")
        local_run.mkdir(parents=True, exist_ok=True)
        with chdir(local_run):
            model.fit(train_X, val_X, epochs_max=epochs, patience=patience, cuda=False)
    else:
        model.fit(train_X, val_X, epochs_max=epochs, patience=patience, cuda=False)

    pred_on_train_mask = model.transform(train_X)
    mask = val_X.notna()
    val_mse = ((pred_on_train_mask[mask] - val_X[mask]) ** 2).mean().mean()
    print(f"Masked validation MSE: {float(val_mse):.6f}")

    imputed = model.transform(df_work.copy()).reindex_like(df_work)
    imputed.index.name = "sample_name"
    imputed.columns.name = "protein"
    imputed = meta.merge(imputed, on="sample_name").set_index(index_cols)

    return imputed
