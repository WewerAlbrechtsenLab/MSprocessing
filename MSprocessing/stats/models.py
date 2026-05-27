import math
import numpy as np
import pandas as pd
import warnings
import re

from concurrent.futures import ProcessPoolExecutor
from statsmodels.formula.api import mixedlm, ols
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm


def filter_results(results_df, term):
    df = results_df[results_df["term"].str.contains(term, na=False)].copy()
    df = df.drop(columns=["term"])
    df = df.sort_values("pval")
    return df.reset_index(drop=True)


def _extract_formula_meta_vars(formula, meta_columns):
    rhs = formula.split("~", 1)[1] if "~" in formula else ""
    tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", rhs))
    tokens = {t for t in tokens if t != "1"}
    return [t for t in tokens if t in meta_columns]


def _prepare_linear_model_inputs(proteome, meta, formula, model, group_col=None):
    vars_in_meta = _extract_formula_meta_vars(formula, meta.columns)

    common_idx = meta.index.intersection(proteome.index)
    meta2 = meta.loc[common_idx]
    proteome2 = proteome.loc[common_idx]

    if vars_in_meta:
        keep = pd.Series(True, index=meta2.index)
        for v in vars_in_meta:
            keep &= meta2[v].notna()
        meta2 = meta2.loc[keep]
        proteome2 = proteome2.loc[meta2.index]

    if group_col is not None:
        if group_col not in meta2.columns:
            raise ValueError(f"{group_col} not in meta columns")
        if model == "ols":
            keep = meta2[group_col].notna()

            n_removed = (~keep).sum()

            if n_removed > 0:
                print(
                    f"Removing {n_removed} observations with missing {group_col} "
                    f"for OLS with cluster-robust SE"
                )

            meta2 = meta2.loc[keep]
            proteome2 = proteome2.loc[meta2.index]

    return proteome2.copy(), meta2.copy(), vars_in_meta


def _fit_linear_models_chunk(proteome_chunk, meta, formula, model="ols", group_col=None, reml=True):
    """
    Fit the same model across a subset of protein columns.
    """
    results = []
    df = meta.copy()

    for protein in proteome_chunk.columns:
        y = proteome_chunk[protein]
        df["y"] = y

        if y.isna().all() or y.nunique(dropna=True) <= 1:
            results.append({
                "protein": protein,
                "term": None,
                "coef": np.nan,
                "pval": np.nan,
            })
            continue

        try:
            if group_col is not None:
                if model == "lmm":
                    m = mixedlm(formula, df, groups=df[group_col], re_formula="1")
                    fit = m.fit(reml=reml, method="lbfgs", disp=False)
                elif model == "ols":
                    fit = ols(formula, data=df).fit(
                        cov_type="cluster",
                        cov_kwds={"groups": df[group_col]}
                )
                else: 
                    raise ValueError(f"Unsupported model type: {model}")
            else:
                if model == "ols":
                    fit = ols(formula, data=df).fit()
                else:
                    raise ValueError(f"Unsupported model type for no group_col: {model}")

            for term, coef, pval in zip(fit.params.index, fit.params.values, fit.pvalues.values):
                results.append({
                    "protein": protein,
                    "term": term,
                    "coef": coef,
                    "pval": pval,
                })

        except Exception:
            results.append({
                "protein": protein,
                "term": None,
                "coef": np.nan,
                "pval": np.nan,
            })

    return pd.DataFrame(results)


def _split_proteome_columns(proteome, n_jobs, chunk_size=None):
    cols = list(proteome.columns)

    if not cols:
        return []

    if chunk_size is None:
        chunk_size = math.ceil(len(cols) / n_jobs)

    return [cols[i:i + chunk_size] for i in range(0, len(cols), chunk_size)]


def _fit_linear_models_once_preprocessed(
    proteome,
    meta,
    formula,
    model="ols",
    group_col=None,
    reml=True,
    n_jobs=1,
    chunk_size=None,
):
    """
    Same model fitting as before, assuming meta/proteome have already been
    preprocessed and aligned.

    Parallelizes across protein chunks when n_jobs > 1.
    """
    if n_jobs is None or n_jobs < 2 or proteome.shape[1] < 2:
        out = _fit_linear_models_chunk(
            proteome_chunk=proteome,
            meta=meta,
            formula=formula,
            model=model,
            group_col=group_col,
            reml=reml,
        )
        return out

    col_chunks = _split_proteome_columns(proteome, n_jobs=n_jobs, chunk_size=chunk_size)

    futures = []
    outputs = []

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for cols in col_chunks:
            prot_chunk = proteome.loc[:, cols]
            futures.append(
                ex.submit(
                    _fit_linear_models_chunk,
                    prot_chunk,
                    meta,
                    formula,
                    model,
                    group_col,
                    reml,
                )
            )

        for fut in futures:
            outputs.append(fut.result())

    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def _fit_linear_models_once(
    proteome,
    meta,
    formula,
    model="ols",
    group_col=None,
    reml=True,
    n_jobs=1,
    chunk_size=None,
):
    proteome2, meta2, _ = _prepare_linear_model_inputs(
        proteome=proteome,
        meta=meta,
        formula=formula,
        model=model,
        group_col=group_col,
    )
    return _fit_linear_models_once_preprocessed(
        proteome=proteome2,
        meta=meta2,
        formula=formula,
        group_col=group_col,
        reml=reml,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
    )


def _permute_meta_columns(meta, permute_cols, rng):
    m = meta.copy()
    permute_cols = [permute_cols] if isinstance(permute_cols, str) else list(permute_cols)

    perm = rng.permutation(len(m))
    m.loc[:, permute_cols] = m[permute_cols].to_numpy()[perm]
    return m


def _resolve_permute_cols(meta, permute_cols=None, filter_to=None):
    if permute_cols is not None:
        cols = [permute_cols] if isinstance(permute_cols, str) else list(permute_cols)
        missing = [c for c in cols if c not in meta.columns]
        if missing:
            raise ValueError(f"permute_cols not in meta columns: {missing}")
        return cols

    if filter_to is not None:
        matches = [c for c in meta.columns if filter_to.lower() in c.lower()]
        if matches:
            return matches

    raise ValueError(
        "Permutation adjustment requires `permute_cols`, or `filter_to` must match "
        "at least one column in `meta`."
    )


def _resampling_adjust_linear_model(
    proteome,
    meta,
    formula,
    observed_df,
    adjust,
    n_perm,
    permute_cols,
    model,
    group_col=None,
    reml=True,
    seed=0,
    n_jobs=1,
    chunk_size=None,
):
    rng = np.random.default_rng(seed=seed)

    observed = observed_df.copy()
    observed["padj"] = np.nan

    valid_terms = observed["term"].dropna().unique().tolist()

    term_info = {}
    perm_store = {}

    for term in valid_terms:
        obs_sub = observed[observed["term"].eq(term)].copy()
        obs_sub = obs_sub[obs_sub["pval"].notna()].copy()

        if obs_sub.empty:
            continue

        proteins = obs_sub["protein"].tolist()
        obs_pvals = obs_sub["pval"].to_numpy(dtype=float)
        obs_idx = obs_sub.index.to_numpy()

        term_info[term] = {
            "proteins": proteins,
            "obs_pvals": obs_pvals,
            "obs_idx": obs_idx,
        }

        perm_store[term] = np.full((n_perm, len(proteins)), np.nan, dtype=float)

    if not term_info:
        return observed

    bar = tqdm(
        range(n_perm),
        total=n_perm,
        desc="Permutations",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for i in bar:
        perm_meta = _permute_meta_columns(meta, permute_cols, rng)

        perm_df = _fit_linear_models_once_preprocessed(
            proteome=proteome,
            meta=perm_meta,
            formula=formula,
            model=model,
            group_col=group_col,
            reml=reml,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
        )

        if perm_df.empty:
            continue

        for term, info in term_info.items():
            perm_term = perm_df[perm_df["term"].eq(term)].copy()
            if perm_term.empty:
                continue

            perm_term = perm_term.set_index("protein").reindex(info["proteins"])
            perm_store[term][i, :] = perm_term["pval"].to_numpy(dtype=float)

    for term, info in term_info.items():
        obs_pvals = info["obs_pvals"]
        obs_idx = info["obs_idx"]
        perm_pvals = perm_store[term]

        keep = ~np.isnan(obs_pvals)
        if not keep.any():
            continue

        obs_pvals_keep = obs_pvals[keep]
        obs_idx_keep = obs_idx[keep]
        perm_pvals_keep = perm_pvals[:, keep]

        if adjust == "perm":
            min_p = np.nanmin(perm_pvals_keep, axis=1)
            padj = np.array(
                [(np.sum(min_p <= p) + 1) / (n_perm + 1) for p in obs_pvals_keep],
                dtype=float,
            )

        elif adjust == "stepdown_perm":
            order = np.argsort(obs_pvals_keep)
            sorted_p = obs_pvals_keep[order]
            perm_sorted = perm_pvals_keep[:, order]

            padj_sorted = np.zeros_like(sorted_p, dtype=float)

            for j, p in enumerate(sorted_p):
                perm_min = np.nanmin(perm_sorted[:, j:], axis=1)
                padj_sorted[j] = (np.sum(perm_min <= p) + 1) / (n_perm + 1)

            padj_sorted = np.maximum.accumulate(padj_sorted)

            padj = np.empty_like(padj_sorted)
            padj[order] = padj_sorted

        else:
            raise ValueError(f"Unsupported resampling adjustment: {adjust}")

        observed.loc[obs_idx_keep, "padj"] = padj

    return observed


def run_linear_model(
    proteome,
    meta,
    formula,
    model="ols",
    group_col=None,
    adjust="fdr_bh",
    reml=True,
    filter_to=None,
    n_perm=1000,
    permute_cols=None,
    seed=0,
    n_jobs=1,
    chunk_size=None,
):
    print(f"Using {n_jobs} worker processes", flush=True)

    proteome_prepped, meta_prepped, _ = _prepare_linear_model_inputs(
        proteome=proteome,
        meta=meta,
        model=model,
        formula=formula,
        group_col=group_col,
    )

    out = _fit_linear_models_once_preprocessed(
        proteome=proteome_prepped,
        meta=meta_prepped,
        formula=formula,
        model=model,
        group_col=group_col,
        reml=reml,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
    )

    if "term" in out.columns:
        out["padj"] = np.nan

        if adjust in ["perm", "stepdown_perm"]:
            resolved_permute_cols = _resolve_permute_cols(
                meta=meta_prepped,
                permute_cols=permute_cols,
                filter_to=filter_to,
            )

            out = _resampling_adjust_linear_model(
                proteome=proteome_prepped,
                meta=meta_prepped,
                formula=formula,
                model=model,
                group_col=group_col,
                reml=reml,
                observed_df=out,
                adjust=adjust,
                n_perm=n_perm,
                permute_cols=resolved_permute_cols,
                seed=seed,
                n_jobs=n_jobs,
                chunk_size=chunk_size,
            )

        else:
            for term, sub in out.groupby("term"):
                mask = sub["pval"].notna()
                if mask.any():
                    out.loc[sub.index[mask], "padj"] = multipletests(
                        sub.loc[mask, "pval"], method=adjust
                    )[1]

    if filter_to:
        out = filter_results(out, filter_to)

    return out.set_index("protein")