import pandas as pd
from MSprocessing import stats as mss

def test_go_enrichment_runs(monkeypatch):
    # fake DEA results
    data = pd.DataFrame({
        "pval": [0.01, 0.2, 0.03],
    }, index=["P12345", "Q67890", "X11111"])

    # monkeypatch GProfiler to avoid network call
    class DummyGP:
        def convert(self, *args, **kwargs):
            return pd.DataFrame({
                "incoming": ["P12345", "Q67890", "X11111"],
                "converted": ["GENE1", "GENE2", "GENE3"],
                "name": ["n1", "n2", "n3"]
            })
        def profile(self, *a, **kw):
            return pd.DataFrame({"term": ["GO:TEST"], "p_value": [0.01]})

    monkeypatch.setattr(mss, "GProfiler", lambda return_dataframe=True: DummyGP())

    res = mss.go_enrichment(data)
    assert isinstance(res, pd.DataFrame)

