import os
import random
import tempfile
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch




def set_global_seed(
    seed: int = 0
) -> None:
    """
    Set random seeds across Python, NumPy, and PyTorch for reproducibility.

    Parameters
    ----------
    seed : int, default=0
        Seed value for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def load_proteomes(
    path: str | Path,
    clean_columns: bool = False,
    extension: str | None = None
) -> pd.DataFrame:
    """
    Load a proteomics table from a TSV file.

    Parameters
    ----------
    path : str or Path
        Path to the TSV file containing proteome data.
    clean_columns : bool, default=False
        If True, cleans column names by removing directory paths.
    extension : str, optional
        If provided, removes this extension from column names.

    Returns
    -------
    pd.DataFrame
        Loaded proteome data with optionally cleaned column names.
    """
    df = pd.read_csv(path, sep="\t")

    if clean_columns:
        cleaned_cols = []
        for col in df.columns:
            if isinstance(col, str):
                base = os.path.basename(col)
                if extension and base.endswith(extension):
                    base = base[:-len(extension)]
                cleaned_cols.append(base)
            else:
                cleaned_cols.append(col)
        df.columns = cleaned_cols

    df = df.apply(pd.to_numeric, errors="ignore")
    return df




def rename_columns_to_sample(
    df: pd.DataFrame,
    sep: str = "_",
    keep_parts: list[int] | None = None
) -> pd.DataFrame:
    """
    Rename columns by splitting on a separator and keeping selected tokens.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame (rows = samples, columns = features).
    sep : str, default="_"
        Separator used to split column names.
    keep_parts : list[int], optional
        Indices (0-based) of the name parts to retain after splitting.
        Defaults to [-1] (keeps the last token).

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns.
    """
    keep_parts = keep_parts or [-2, -1]
    if not isinstance(keep_parts, list):
        raise ValueError("Argument 'keep_parts' must be a list of integer positions.")

    new_cols = []
    for col in df.columns:
        tokens = str(col).split(sep)
        if len(tokens) == 1:
            print(f"Column '{col}' has no separator → kept as is.")
            new_cols.append(col)
        else:
            selected = [tokens[i] for i in keep_parts if i < len(tokens)]
            if not selected:
                print(f"Warning: column '{col}' has fewer tokens than requested → kept as is.")
                new_cols.append(col)
            else:
                new_cols.append("_".join(selected))

    df_renamed = df.copy()
    df_renamed.columns = new_cols
    return df_renamed




def merge_meta_proteome(
    meta: pd.DataFrame,
    proteome: pd.DataFrame,
    meta_id_col: str,
    proteome_id_col: str
) -> pd.DataFrame:
    """
    Merge metadata and proteome data by matching sample identifiers.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata table containing a sample identifier column.
    proteome : pd.DataFrame
        Proteome table with sample identifiers as a column.
    meta_id_col : str
        Column in `meta` that identifies samples.
    proteome_id_col : str
        Column in `proteome` that identifies proteins.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame indexed by metadata columns.
    """
    proteome = proteome.set_index(proteome_id_col)
    proteome_t = proteome.T
    proteome_t.index.name = meta_id_col
    proteome_t.reset_index(inplace=True)

    meta = meta.rename(columns={meta_id_col: "sample_name"})
    proteome_t = proteome_t.rename(columns={meta_id_col: "sample_name"})

    merged = pd.merge(meta, proteome_t, on="sample_name", how="inner")

    index_cols = meta.columns.tolist()
    merged = merged.set_index(index_cols)

    return merged




def configure_fastai_local_tmp(
    scratch: Path | str | None = None
) -> Path:
    """
    Configure FastAI to use a local temporary directory for cache and checkpoints.

    Parameters
    ----------
    scratch : Path or str, optional
        Path to a writable scratch directory. If None, defaults to ~/fastai_scratch.

    Returns
    -------
    Path
        Path to the configured scratch directory.
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




def get_sample_order(
    df: pd.DataFrame,
    plate_nr: str,
    plate_pos: str,
    order: str = "horizontal"
) -> pd.DataFrame:
    """
    Compute and append a 'sample_order' index level for plate-based samples.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with MultiIndex including plate number and plate position.
    plate_nr : str
        Name of the index level containing plate numbers.
    plate_pos : str
        Name of the index level containing well positions (e.g., 'A1', 'B12').
    order : {'horizontal', 'vertical'}, default='horizontal'
        Determines lexicographic sort order within plates.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional MultiIndex level named 'sample_order'.
    """
    plate = df.index.get_level_values(plate_nr).astype(str).str.extract(r"(\d+)")[0].astype(int)
    pos = df.index.get_level_values(plate_pos).astype(str)

    rows = pos.str.extract(r"^([A-Za-z]+)")[0]
    cols = pos.str.extract(r"(\d+)$")[0].astype(int)

    row_num = rows.str.upper().apply(lambda s: sum((ord(c) - 64) * (26 ** i) for i, c in enumerate(s[::-1])))

    if order == "horizontal":
        order_pos = np.lexsort((cols.to_numpy(), row_num.to_numpy(), plate.to_numpy()))
    elif order == "vertical":
        order_pos = np.lexsort((row_num.to_numpy(), cols.to_numpy(), plate.to_numpy()))
    else:
        raise ValueError("Argument 'order' must be 'horizontal' or 'vertical'.")

    ranks = np.empty(len(df), dtype=int)
    ranks[order_pos] = np.arange(1, len(df) + 1)

    return df.set_index(pd.Index(ranks, name="sample_order"), append=True)




@contextlib.contextmanager
def chdir(
    path: Path | str
):
    """
    Temporarily change the current working directory within a context.
    """
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)
