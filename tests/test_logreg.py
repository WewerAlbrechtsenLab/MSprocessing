import pandas as pd
import numpy as np
from MSprocessing import stats as mss

def make_dummy():
    proteome = pd.DataFrame(
        np.random.randn(10, 3),
        index=[f"s{i}" for i in range(10)],
        columns=["prot1", "prot2", "prot3"]
    )
    meta = pd.DataFrame({
        "group": ["A"]*5 + ["B"]*5
    }, index=proteome.index)
    return proteome, meta

def test_run_logreg():
    proteome, meta = make_dummy()
    clf, res = mss.run_logreg(
        proteome, meta,
        column="group",
        group1="A",
        group2="B"
    )
    assert hasattr(clf, "coef_")
    assert "beta" in res.columns
    assert "padj" in res.columns
