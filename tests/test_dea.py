import pandas as pd
import numpy as np
from MSprocessing import stats as mss
from alphastats.dataset.preprocessing import PreprocessingStateKeys as PSK

def make_dummy():
    index = ["s1", "s2", "s3", "s4"]
    proteome = pd.DataFrame(
        np.random.randn(4, 3),
        index=index,
        columns=["p1", "p2", "p3"]
    )
    meta = pd.DataFrame({
        "sample_name": index,
        "group": ["A", "A", "B", "B"],
        "study_ID": ["id1", "id1", "id2", "id2"],
        "timepoint": ["baseline", "w48", "baseline", "w48"],
    }).set_index("sample_name")
    return proteome, meta

def test_run_dea_ttest():
    proteome, meta = make_dummy()
    res, fig = mss.run_dea(
        proteome=proteome,
        meta=meta,
        method="ttest",
        column="group",
        group1="A",
        group2="B",
        pairing="study_ID",
        preprocessing_info={PSK.LOG2_TRANSFORMED: True}
    )
    assert isinstance(res, pd.DataFrame)
    assert "pval" in res.columns
    assert "padj" in res.columns

def test_run_dea_delta_ttest():
    proteome, meta = make_dummy()
    res, _ = mss.run_dea(
        proteome=proteome,
        meta=meta,
        method="delta-ttest",
        column="group",
        group1="A",
        group2="B",
        pairing="study_ID",
        delta_col="timepoint",
        delta_group1="baseline",
        delta_group2="w48",
        preprocessing_info={PSK.LOG2_TRANSFORMED: True}
    )
    assert "pval" in res.columns
