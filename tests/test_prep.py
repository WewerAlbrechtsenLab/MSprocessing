import pandas as pd
import numpy as np
import pytest
from MSprocessing import stats as mss
from alphastats.dataset.keys import Cols

def make_dummy_data():
    index_cols = ["sample_name", "group", "study_ID", "timepoint", "sample_type"]
    df = pd.DataFrame({
        "sample_name": ["s1", "s2", "s3", "s4"],
        "group": ["A", "A", "B", "B"],
        "study_ID": ["id1", "id1", "id2", "id2"],
        "timepoint": ["baseline", "w48", "baseline", "w48"],
        "sample_type": ["sample"] * 4,
        "prot1": [1.0, 2.0, 3.0, 4.0],
        "prot2": [5.0, 6.0, 7.0, 8.0],
    })
    df = df.set_index(index_cols)
    return df

def test_split_proteome_meta():
    df = make_dummy_data()
    proteome, meta = mss.split_proteome_meta(df)
    assert isinstance(proteome, pd.DataFrame)
    assert isinstance(meta, pd.DataFrame)
    assert "prot1" in proteome.columns
    assert "group" in meta.columns

def test_prep_for_paired_ttest():
    df = make_dummy_data()
    proteome, meta = mss.split_proteome_meta(df)
    out_proteome, out_meta = mss.prep_for_paired_ttest(
        meta, proteome, pair_col="study_ID",
        variable="timepoint", group1="baseline", group2="w48"
    )
    assert isinstance(out_proteome, pd.DataFrame)
    assert isinstance(out_meta, pd.DataFrame)
    assert Cols.SAMPLE in out_meta.columns

def test_prep_deltas():
    df = make_dummy_data()
    proteome, meta = mss.split_proteome_meta(df)
    delta_proteome, delta_meta = mss.prep_deltas(
        meta, proteome, subject_col="study_ID",
        delta_col="timepoint", delta_group1="baseline", delta_group2="w48"
    )
    assert isinstance(delta_proteome, pd.DataFrame)
    assert isinstance(delta_meta, pd.DataFrame)
    assert "prot1" in delta_proteome.columns
