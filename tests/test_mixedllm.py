import pandas as pd
import numpy as np
from MSprocessing import stats as mss

def make_dummy():
    proteome = pd.DataFrame(
        np.random.randn(4, 2),
        index=["s1", "s2", "s3", "s4"],
        columns=["protA", "protB"]
    )
    meta = pd.DataFrame({
        "group": ["Control", "Control", "Treatment", "Treatment"],
        "timepoint": ["baseline", "w48", "baseline", "w48"],
        "study_ID": ["id1", "id1", "id2", "id2"]
    }, index=["s1", "s2", "s3", "s4"])
    return proteome, meta

def test_run_mixedlm_basic():
    proteome, meta = make_dummy()
    res = mss.run_mixedlm(
        proteome, meta,
        var1={"group": ["Control", "Treatment"]},
        var2={"timepoint": ["baseline", "w48"]},
        pairing="study_ID"
    )
    assert isinstance(res, pd.DataFrame)
    if not res.empty:
        assert "pval" in res.columns
        assert "beta" in res.columns
