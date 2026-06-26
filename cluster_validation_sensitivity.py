#!/usr/bin/env python3
"""
Post-hoc cluster validation and continuum sensitivity analyses.

This script is intentionally separate from the main clustering pipeline. It
adds no-cluster, covariance-matched, and projection-based checks requested
during methods review without changing the fitted cluster solution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import dill as _pickle_module
except Exception:  # pragma: no cover - depends on runtime environment
    import pickle as _pickle_module

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import norm as _scipy_norm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

_MAX_32BIT_SEED = 2 ** 32 - 1


def _derive_seed(*parts, base=0):
    payload = "|".join(str(part) for part in (base,) + tuple(parts)).encode("utf-8")
    seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return seed % _MAX_32BIT_SEED


@dataclass
class Solution:
    name: str
    X: np.ndarray
    labels: np.ndarray
    kind: str
    pipeline_stability_ari: float = np.nan


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_metrics(path: str) -> dict:
    with open(path, "rb") as f:
        return _pickle_module.load(f)


def _as_numeric_matrix(df: pd.DataFrame, subject_id_column: str) -> Tuple[np.ndarray, List[str]]:
    xdf = df.drop(columns=[subject_id_column], errors="ignore").copy()
    xdf = xdf.select_dtypes(include=[np.number])
    cols = list(xdf.columns)
    X = xdf.to_numpy(dtype=float, copy=True)
    good_cols = np.isfinite(X).all(axis=0)
    X = X[:, good_cols]
    cols = [c for c, keep in zip(cols, good_cols) if keep]
    if X.shape[1] == 0:
        raise ValueError("No finite numeric feature columns available.")
    return X, cols


def _standardize(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(X, dtype=float))


def _pca_reduce(X: np.ndarray, max_components: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    Xz = _standardize(X)
    n_comp = max(1, min(max_components, Xz.shape[0] - 1, Xz.shape[1]))
    pca = PCA(n_components=n_comp, random_state=42)
    return pca.fit_transform(Xz), pca.explained_variance_ratio_


def _projection_matrix(n_features: int, n_random: int, seed: int) -> Tuple[np.ndarray, List[str]]:
    """Return canonical PC axes plus random unit projection directions."""
    eye = np.eye(n_features, dtype=float)
    names = [f"PC{i + 1}" for i in range(n_features)]
    if n_random <= 0:
        return eye, names
    rng = np.random.default_rng(seed)
    random_dirs = rng.normal(size=(int(n_random), n_features))
    norms = np.linalg.norm(random_dirs, axis=1, keepdims=True)
    random_dirs = random_dirs / np.clip(norms, 1e-12, None)
    names.extend([f"random_projection_{i + 1}" for i in range(int(n_random))])
    return np.vstack([eye, random_dirs]), names


def _valid_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    return labels >= 0


def _cluster_count(labels: np.ndarray) -> int:
    v = labels[_valid_labels(labels)]
    return int(len(np.unique(v))) if v.size else 0


def _quality_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels).reshape(-1)
    valid = _valid_labels(labels)
    Xv = X[valid]
    lv = labels[valid]
    out = {
        "n": int(Xv.shape[0]),
        "n_features": int(Xv.shape[1]) if Xv.ndim == 2 else 0,
        "k": _cluster_count(labels),
        "silhouette": np.nan,
        "silhouette_norm": np.nan,
        "calinski_harabasz": np.nan,
        "calinski_harabasz_norm": np.nan,
        "davies_bouldin": np.nan,
        "davies_bouldin_inv": np.nan,
        "composite": np.nan,
    }
    if Xv.shape[0] < 3 or out["k"] <= 1:
        return out
    try:
        sil = float(silhouette_score(Xv, lv))
        out["silhouette"] = sil
        out["silhouette_norm"] = (sil + 1.0) / 2.0
    except Exception:
        pass
    try:
        ch = float(calinski_harabasz_score(Xv, lv))
        out["calinski_harabasz"] = ch
        out["calinski_harabasz_norm"] = ch / (ch + 1.0)
    except Exception:
        pass
    try:
        db = float(davies_bouldin_score(Xv, lv))
        out["davies_bouldin"] = db
        out["davies_bouldin_inv"] = 1.0 / (1.0 + db)
    except Exception:
        pass
    vals = [
        out["silhouette_norm"],
        out["calinski_harabasz_norm"],
        out["davies_bouldin_inv"],
    ]
    vals = [v for v in vals if np.isfinite(v)]
    out["composite"] = float(np.mean(vals)) if vals else np.nan
    return out


def _uni_cluster_baseline(X: np.ndarray) -> Dict[str, object]:
    labels = np.zeros(X.shape[0], dtype=int)
    quality = _quality_metrics(_standardize(X), labels)
    return {
        "k": 1,
        "quality": quality,
        "note": "Internal cluster-separation metrics are undefined for k=1; this records the explicit one-cluster baseline and the Gap statistic provides the k=1 versus k>1 comparison.",
    }


def _hierarchical_labels(X: np.ndarray, k: int, method: str = "average") -> np.ndarray:
    if k <= 1:
        return np.zeros(X.shape[0], dtype=int)
    if X.shape[0] <= k:
        return np.arange(X.shape[0], dtype=int)
    Z = linkage(X, method=method, metric="euclidean")
    return fcluster(Z, t=int(k), criterion="maxclust").astype(int) - 1


def _kmeans_within_dispersion(X: np.ndarray, k: int, seed: int) -> float:
    if k <= 1:
        center = np.mean(X, axis=0, keepdims=True)
        return float(np.sum((X - center) ** 2))
    km = KMeans(n_clusters=int(k), n_init=20, random_state=seed)
    km.fit(X)
    return float(km.inertia_)


def _resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs is None or int(n_jobs) == 0:
        return 1
    if int(n_jobs) < 0:
        return max(1, os.cpu_count() or 1)
    return max(1, int(n_jobs))


def _parallel_map(func, tasks: List[tuple], n_jobs: int):
    workers = min(_resolve_n_jobs(n_jobs), len(tasks)) if tasks else 1
    if workers <= 1 or len(tasks) <= 1:
        return [func(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(func, tasks))


def _rscript_path() -> Optional[str]:
    return shutil.which("Rscript")


def _sample_cov_matched_from_params(
    n_rows: int,
    mu: np.ndarray,
    vals: np.ndarray,
    vecs: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(int(n_rows), vals.size))
    return z @ (vecs * np.sqrt(vals)).T + mu


def _gap_reference_worker(task: tuple) -> float:
    shape, mins, maxs, k, seed = task
    rng = np.random.default_rng(seed)
    Xb = rng.uniform(mins, maxs, size=shape)
    return math.log(max(_kmeans_within_dispersion(Xb, int(k), int(seed)), 1e-12))


def _gaussian_null_worker(task: tuple) -> Dict[str, float]:
    n_rows, mu, vals, vecs, k, n_bootstrap, seed = task
    Xb = _sample_cov_matched_from_params(n_rows, mu, vals, vecs, seed)
    lb = _hierarchical_labels(Xb, k=int(k))
    q = _quality_metrics(Xb, lb)
    st = _subsample_stability(
        Xb,
        k=int(k),
        n_bootstrap=int(n_bootstrap),
        seed=_derive_seed("gaussian_null_subsample_stability", base=int(seed)),
    )
    return {
        "composite": q["composite"],
        "silhouette_norm": q["silhouette_norm"],
        "mean_ari": st["mean_ari"],
    }


def _gap_statistic(
    X: np.ndarray,
    k_max: int,
    n_refs: int,
    seed: int,
    max_components: int,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    Xr, var_ratio = _pca_reduce(X, max_components=max_components)
    k_max = max(1, min(int(k_max), Xr.shape[0] - 1))
    mins = Xr.min(axis=0)
    maxs = Xr.max(axis=0)
    rows = []
    for k in range(1, k_max + 1):
        wk = _kmeans_within_dispersion(Xr, k, seed)
        tasks = [
            (Xr.shape, mins, maxs, k, _derive_seed("gap_reference", k, b, base=seed))
            for b in range(int(n_refs))
        ]
        ref_logs = _parallel_map(_gap_reference_worker, tasks, n_jobs=n_jobs)
        ref_logs = np.asarray(ref_logs, dtype=float)
        gap = float(np.mean(ref_logs) - math.log(max(wk, 1e-12)))
        sdk = float(np.sqrt(1.0 + 1.0 / n_refs) * np.std(ref_logs, ddof=1)) if n_refs > 1 else np.nan
        rows.append({"k": k, "log_wk": math.log(max(wk, 1e-12)), "gap": gap, "gap_se": sdk})
    gap_df = pd.DataFrame(rows)
    selected = int(gap_df.loc[gap_df["gap"].idxmax(), "k"])
    for i in range(len(gap_df) - 1):
        if gap_df.loc[i, "gap"] >= gap_df.loc[i + 1, "gap"] - gap_df.loc[i + 1, "gap_se"]:
            selected = int(gap_df.loc[i, "k"])
            break
    meta = {
        "selected_k_tibshirani_rule": selected,
        "selected_k_max_gap": int(gap_df.loc[gap_df["gap"].idxmax(), "k"]),
        "pca_components": int(Xr.shape[1]),
        "pca_variance_explained": float(np.sum(var_ratio)),
    }
    return gap_df, meta


def _gap_statistic_r_clusgap(
    X: np.ndarray,
    k_max: int,
    n_refs: int,
    seed: int,
    max_components: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rscript = _rscript_path()
    if rscript is None:
        raise RuntimeError("Rscript is not available; cannot run R cluster::clusGap.")

    Xr, var_ratio = _pca_reduce(X, max_components=max_components)
    k_max = max(2, min(int(k_max), Xr.shape[0] - 1))
    with tempfile.TemporaryDirectory(prefix="multiclust_clusgap_") as tmp:
        x_path = os.path.join(tmp, "x.csv")
        out_path = os.path.join(tmp, "gap.csv")
        meta_path = os.path.join(tmp, "meta.json")
        pd.DataFrame(Xr).to_csv(x_path, index=False)
        code = r"""
args <- commandArgs(trailingOnly = TRUE)
x_path <- args[[1]]
out_path <- args[[2]]
meta_path <- args[[3]]
k_max <- as.integer(args[[4]])
b <- as.integer(args[[5]])
seed <- as.integer(args[[6]])
suppressPackageStartupMessages(library(cluster))
set.seed(seed)
x <- as.matrix(read.csv(x_path, check.names = FALSE))
fun <- function(x, k) {
  if (k <= 1) {
    return(list(cluster = rep(1L, nrow(x))))
  }
  stats::kmeans(x, centers = k, nstart = 20, iter.max = 100)
}
res <- cluster::clusGap(x, FUNcluster = fun, K.max = k_max, B = b, d.power = 2, spaceH0 = "scaledPCA", verbose = FALSE)
tab <- as.data.frame(res$Tab)
tab$k <- seq_len(nrow(tab))
names(tab) <- sub("SE.sim", "gap_se", names(tab), fixed = TRUE)
write.csv(tab[, c("k", "logW", "gap", "gap_se")], out_path, row.names = FALSE)
selected_tibs <- cluster::maxSE(tab$gap, tab$gap_se, method = "Tibs2001SEmax")
selected_max <- which.max(tab$gap)
cat(
  sprintf(
    '{"selected_k_tibshirani_rule":%d,"selected_k_max_gap":%d}\n',
    as.integer(selected_tibs),
    as.integer(selected_max)
  ),
  file = meta_path
)
"""
        cmd = [rscript, "-e", code, x_path, out_path, meta_path, str(k_max), str(int(n_refs)), str(int(seed))]
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "R cluster::clusGap failed.")
        gap_df = pd.read_csv(out_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)
    meta.update(
        {
            "method": "R cluster::clusGap",
            "package": "cluster",
            "pca_components": int(Xr.shape[1]),
            "pca_variance_explained": float(np.sum(var_ratio)),
        }
    )
    return gap_df, meta


def _dip_test_projections(
    X: np.ndarray,
    max_components: int,
    n_random_projections: int,
    n_null: int,
    seed: int,
    n_jobs: int = 1,
) -> Dict[str, object]:
    Xr, var_ratio = _pca_reduce(X, max_components=max_components)
    directions, direction_names = _projection_matrix(Xr.shape[1], n_random_projections, seed)
    out = {
        "method": "Hartigan dip tests across PCs and random projections of the multivariate PCA space",
        "max_dip": np.nan,
        "global_projection_p_value": np.nan,
        "min_projection_p_value": np.nan,
        "bonferroni_min_p_value": np.nan,
        "best_projection": "",
        "pc1_dip": np.nan,
        "pc1_p_value": np.nan,
        "n_projections": int(directions.shape[0]),
        "n_random_projections": int(n_random_projections),
        "n_null": int(n_null),
        "pca_components": int(Xr.shape[1]),
        "pca_variance_explained": float(np.sum(var_ratio)),
        "available": False,
        "note": "Install the optional 'diptest' package to compute Hartigan's dip test.",
    }
    try:
        import diptest  # type: ignore

        scores = Xr @ directions.T
        dips = []
        pvals = []
        for i in range(scores.shape[1]):
            dip, p = diptest.diptest(np.asarray(scores[:, i], dtype=float))
            dips.append(float(dip))
            pvals.append(float(p))
        dips_arr = np.asarray(dips, dtype=float)
        pvals_arr = np.asarray(pvals, dtype=float)
        best_idx = int(np.nanargmax(dips_arr))

        global_p = np.nan
        if n_null > 0:
            mu, vals, vecs = _regularized_covariance(Xr)
            tasks = [
                (Xr.shape[0], mu, vals, vecs, directions, _derive_seed("dip_projection_null", b, base=seed))
                for b in range(int(n_null))
            ]
            null_max = np.asarray(_parallel_map(_dip_projection_null_worker, tasks, n_jobs=n_jobs), dtype=float)
            global_p = float((1.0 + np.sum(null_max >= dips_arr[best_idx])) / (null_max.size + 1.0))

        out.update(
            {
                "max_dip": float(dips_arr[best_idx]),
                "global_projection_p_value": global_p,
                "min_projection_p_value": float(np.nanmin(pvals_arr)),
                "bonferroni_min_p_value": float(min(1.0, np.nanmin(pvals_arr) * len(pvals_arr))),
                "best_projection": direction_names[best_idx],
                "pc1_dip": float(dips_arr[0]) if dips_arr.size else np.nan,
                "pc1_p_value": float(pvals_arr[0]) if pvals_arr.size else np.nan,
                "projection_results": [
                    {"projection": name, "dip": dip, "p_value": p}
                    for name, dip, p in zip(direction_names, dips, pvals)
                ],
                "available": True,
                "note": "Global p-value compares the strongest observed projection dip against covariance-matched Gaussian null datasets using the same projection search.",
            }
        )
    except Exception:
        pass
    return out


def _dip_projection_null_worker(task: tuple) -> float:
    n_rows, mu, vals, vecs, directions, seed = task
    try:
        import diptest  # type: ignore
    except Exception:
        return np.nan
    Xb = _sample_cov_matched_from_params(n_rows, mu, vals, vecs, seed)
    scores = Xb @ directions.T
    dips = [diptest.diptest(np.asarray(scores[:, i], dtype=float))[0] for i in range(scores.shape[1])]
    return float(np.nanmax(dips))


def _regularized_covariance(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xz = _standardize(X)
    mu = np.mean(Xz, axis=0)
    cov = np.cov(Xz, rowvar=False)
    cov = np.atleast_2d(cov)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-6, None)
    return mu, vals, vecs


def _sigclust_python_package(
    X: np.ndarray,
    labels: np.ndarray,
    n_sim: int,
    seed: int,
    max_components: int,
    n_jobs: int = 1,
) -> Dict[str, object]:
    Xr, var_ratio = _pca_reduce(X, max_components=max_components)
    k = _cluster_count(labels)
    base = {
        "method": "Python sigclust.SigClust using the actual observed labels",
        "package": "thomaskeefe/sigclust",
        "available": False,
        "observed_k": int(k),
        "observed_cluster_index": np.nan,
        "null_mean_cluster_index": np.nan,
        "null_sd_cluster_index": np.nan,
        "p_value": np.nan,
        "p_value_normal_approx": np.nan,
        "n_simulations": int(n_sim),
        "pca_components": int(Xr.shape[1]),
        "pca_variance_explained": float(np.sum(var_ratio)),
    }
    if k != 2:
        base["note"] = "SigClust is a two-cluster-vs-one-Gaussian test. The observed solution does not have exactly two groups, so SigClust is not applicable to this actual solution."
        return base

    valid = _valid_labels(labels)
    labs = np.asarray(labels).reshape(-1)[valid]
    Xv = Xr[valid]
    uniq = np.unique(labs)
    if uniq.size != 2:
        base["note"] = "Valid observed labels do not contain exactly two groups."
        return base

    try:
        from sigclust import SigClust  # type: ignore
    except Exception as exc:
        base["note"] = f"Python sigclust package is not importable: {exc}"
        return base

    try:
        np.random.seed(int(seed))
        sc = SigClust(num_simulations=int(n_sim))
        sc.fit(data=Xv, labels=labs)
    except Exception as exc:
        base["note"] = f"Python sigclust failed: {exc}"
        return base

    sim = np.asarray(getattr(sc, "simulated_cluster_indices", []), dtype=float)
    obs_ci = float(getattr(sc, "sample_cluster_index", np.nan))
    null_mean_ci = float(np.mean(sim)) if sim.size else np.nan
    null_sd_ci = float(np.std(sim, ddof=1)) if sim.size > 1 else np.nan
    if np.isfinite(obs_ci) and np.isfinite(null_mean_ci) and np.isfinite(null_sd_ci) and null_sd_ci > 0:
        z_obs = (obs_ci - null_mean_ci) / null_sd_ci
        p_norm = float(_scipy_norm.cdf(z_obs))
    else:
        p_norm = np.nan
    base.update(
        {
            "available": True,
            "observed_cluster_index": obs_ci,
            "null_mean_cluster_index": null_mean_ci,
            "null_sd_cluster_index": null_sd_ci,
            "p_value": float(getattr(sc, "p_value", np.nan)),
            "z_score": float(getattr(sc, "z_score", np.nan)),
            "p_value_normal_approx": p_norm,
            "note": "Observed statistic uses the actual two-group final labels. The package tests whether this two-cluster split is stronger than expected under a single Gaussian null with matched covariance eigenvalues.",
        }
    )
    return base


def _subsample_stability(
    X: np.ndarray,
    k: int,
    n_bootstrap: int,
    seed: int,
    frac: float = 0.8,
) -> Dict[str, object]:
    n = X.shape[0]
    size = max(k + 1, int(round(frac * n)))
    if n_bootstrap < 2 or k <= 1 or n < 4:
        return {"mean_ari": np.nan, "sd_ari": np.nan, "n_pairs": 0}
    label_sets = []
    idx_sets = []
    for b in range(n_bootstrap):
        rng = np.random.default_rng(_derive_seed("subsample_stability_bootstrap", b, base=seed))
        idx = np.sort(rng.choice(n, size=size, replace=False))
        labs = _hierarchical_labels(X[idx], k=k)
        label_sets.append(labs)
        idx_sets.append(idx)
    scores = []
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            common, ai, bj = np.intersect1d(idx_sets[i], idx_sets[j], return_indices=True)
            if common.size > 1:
                scores.append(adjusted_rand_score(label_sets[i][ai], label_sets[j][bj]))
    return {
        "mean_ari": float(np.mean(scores)) if scores else np.nan,
        "sd_ari": float(np.std(scores)) if scores else np.nan,
        "n_pairs": int(len(scores)),
    }


def _gaussian_null_quality_and_stability(
    X: np.ndarray,
    labels: np.ndarray,
    pipeline_stability_ari: float,
    n_null: int,
    n_bootstrap: int,
    seed: int,
    max_components: int,
    n_jobs: int = 1,
) -> Dict[str, object]:
    k = _cluster_count(labels)
    Xr, var_ratio = _pca_reduce(X, max_components=max_components)
    observed_quality = _quality_metrics(Xr, labels)
    mu, vals, vecs = _regularized_covariance(Xr)
    tasks = [
        (Xr.shape[0], mu, vals, vecs, k, n_bootstrap, _derive_seed("gaussian_null_dataset", b, base=seed))
        for b in range(int(n_null))
    ]
    null_rows = _parallel_map(_gaussian_null_worker, tasks, n_jobs=n_jobs)
    null_df = pd.DataFrame(null_rows)
    comp_actual = observed_quality["composite"]
    ari_actual = float(pipeline_stability_ari) if np.isfinite(pipeline_stability_ari) else np.nan
    return {
        "k": int(k),
        "observed_solution_quality": observed_quality,
        "observed_pipeline_stability_ari": ari_actual,
        "null_quality_mean": float(null_df["composite"].mean()),
        "null_quality_sd": float(null_df["composite"].std()),
        "null_stability_mean_ari": float(null_df["mean_ari"].mean()),
        "null_stability_sd_ari": float(null_df["mean_ari"].std()),
        "p_quality_ge_observed_solution": float((1.0 + np.sum(null_df["composite"] >= comp_actual)) / (len(null_df) + 1.0)) if np.isfinite(comp_actual) else np.nan,
        "p_stability_ge_observed_pipeline": float((1.0 + np.sum(null_df["mean_ari"] >= ari_actual)) / (len(null_df) + 1.0)) if np.isfinite(ari_actual) else np.nan,
        "n_null": int(n_null),
        "n_bootstrap_per_null": int(n_bootstrap),
        "pca_components": int(Xr.shape[1]),
        "pca_variance_explained": float(np.sum(var_ratio)),
        "note": "Observed comparisons use the actual final labels and the pipeline ARI stability stored in final_metrics when available. Null datasets are covariance-matched single-Gaussian datasets reclustered at observed k to estimate how strong and stable no-cluster data can appear.",
    }


def _projection_median_split_stability(
    Xr: np.ndarray,
    directions: np.ndarray,
    n_bootstrap: int,
    seed: int,
    frac: float = 0.8,
) -> Dict[str, object]:
    n = Xr.shape[0]
    size = max(3, int(round(frac * n)))
    if n_bootstrap < 2 or n < 4:
        return {"mean_ari": np.nan, "sd_ari": np.nan, "n_pairs": 0}

    label_sets = []
    idx_sets = []
    for b in range(n_bootstrap):
        rng = np.random.default_rng(_derive_seed("projection_median_stability_bootstrap", b, base=seed))
        idx = np.sort(rng.choice(n, size=size, replace=False))
        Xb = Xr[idx]
        best_labels = None
        best_score = -np.inf
        for direction in directions:
            scores = Xb @ direction
            split = (scores > np.nanmedian(scores)).astype(int)
            q = _quality_metrics(Xb, split)["composite"]
            if np.isfinite(q) and q > best_score:
                best_score = q
                best_labels = split
        if best_labels is None:
            best_labels = np.zeros(Xb.shape[0], dtype=int)
        label_sets.append(best_labels)
        idx_sets.append(idx)

    scores = []
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            common, ai, bj = np.intersect1d(idx_sets[i], idx_sets[j], return_indices=True)
            if common.size > 1:
                scores.append(adjusted_rand_score(label_sets[i][ai], label_sets[j][bj]))
    return {
        "mean_ari": float(np.mean(scores)) if scores else np.nan,
        "sd_ari": float(np.std(scores)) if scores else np.nan,
        "n_pairs": int(len(scores)),
    }


def _projection_median_split(
    X: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int,
    max_components: int,
    n_random_projections: int,
    seed: int,
) -> Dict[str, object]:
    Xr, var_ratio = _pca_reduce(X, max_components=max_components)
    directions, direction_names = _projection_matrix(Xr.shape[1], n_random_projections, seed)
    rows = []
    valid = _valid_labels(labels)
    for name, direction in zip(direction_names, directions):
        scores = Xr @ direction
        split = (scores > np.nanmedian(scores)).astype(int)
        quality = _quality_metrics(Xr, split)
        ari = float(adjusted_rand_score(labels[valid], split[valid])) if _cluster_count(labels) > 1 else np.nan
        rows.append(
            {
                "projection": name,
                "quality": quality,
                "ari_with_observed_labels": ari,
                "class_counts": {str(k): int(v) for k, v in zip(*np.unique(split, return_counts=True))},
            }
        )

    finite_quality = [i for i, row in enumerate(rows) if np.isfinite(row["quality"]["composite"])]
    best_quality_idx = max(finite_quality, key=lambda i: rows[i]["quality"]["composite"]) if finite_quality else 0
    finite_ari = [i for i, row in enumerate(rows) if np.isfinite(row["ari_with_observed_labels"])]
    best_ari_idx = max(finite_ari, key=lambda i: rows[i]["ari_with_observed_labels"]) if finite_ari else best_quality_idx
    best_quality = rows[best_quality_idx]
    best_ari = rows[best_ari_idx]
    return {
        "method": "Median-split benchmark across PCs and random projections; split quality is scored in the multivariate PCA space.",
        "pca_components": int(Xr.shape[1]),
        "pca_variance_explained": float(np.sum(var_ratio)),
        "n_projections": int(directions.shape[0]),
        "n_random_projections": int(n_random_projections),
        "best_quality_projection": best_quality["projection"],
        "best_quality": best_quality["quality"],
        "best_quality_ari_with_observed_labels": best_quality["ari_with_observed_labels"],
        "best_quality_class_counts": best_quality["class_counts"],
        "best_ari_projection": best_ari["projection"],
        "best_ari": best_ari["ari_with_observed_labels"],
        "best_ari_quality": best_ari["quality"],
        "best_ari_class_counts": best_ari["class_counts"],
        "bootstrap_stability": _projection_median_split_stability(
            Xr,
            directions,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "projection_results": rows,
    }


def _plot_pc1_pc2(solution: Solution, out_dir: str, max_components: int) -> None:
    Xr, var_ratio = _pca_reduce(solution.X, max_components=max(2, max_components))
    if Xr.shape[1] < 2:
        return
    labels = np.asarray(solution.labels)
    valid = _valid_labels(labels)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    scatter = ax.scatter(Xr[valid, 0], Xr[valid, 1], c=labels[valid], cmap="tab10", s=22, alpha=0.78, linewidths=0)
    ax.set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}% var.)")
    ax.set_title(f"{solution.name}: PC1 vs PC2 colored by cluster label")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster label")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{solution.name}_pc1_pc2_clusters.png"), dpi=200)
    plt.close(fig)


def _plot_gap(gap_df: pd.DataFrame, name: str, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.errorbar(gap_df["k"], gap_df["gap"], yerr=gap_df["gap_se"], marker="o", linewidth=1.5, capsize=3)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Gap statistic")
    ax.set_title(f"{name}: gap statistic including k=1")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_gap_statistic.png"), dpi=200)
    plt.close(fig)


def _build_solutions(metrics: dict, modalities: Optional[List[str]], subject_id_column: str) -> List[Solution]:
    data = metrics.get("data")
    indiv = metrics.get("individual_labels")
    if indiv is None:
        indiv = metrics.get("train_individual_labels")
    final_labels = metrics.get("final_labels")
    if final_labels is None:
        final_labels = metrics.get("train_final_labels")
    if final_labels is None:
        final_labels = metrics.get("train_labels")
    per_view_stability = metrics.get("per_view_stabilities_ari")
    if per_view_stability is None:
        per_view_stability = metrics.get("per_view_stabilities")
    final_stability = metrics.get("final_stability_ari")
    if final_stability is None:
        final_stability = metrics.get("final_stability")

    try:
        final_stability_value = float(final_stability)
    except (TypeError, ValueError):
        final_stability_value = np.nan

    # singleclust stores one processed dataframe, whereas the multiview
    # pipeline stores a modality -> dataframe dictionary.
    if isinstance(data, pd.DataFrame):
        if final_labels is None:
            return []
        X, _ = _as_numeric_matrix(data, subject_id_column)
        labels = np.asarray(final_labels).reshape(-1)
        if labels.size != X.shape[0]:
            raise ValueError(
                "Single-cluster metrics contain different numbers of rows and labels: "
                f"{X.shape[0]} rows versus {labels.size} labels."
            )
        return [Solution(
            name="singleclust",
            X=X,
            labels=labels,
            kind="singleclust",
            pipeline_stability_ari=final_stability_value,
        )]

    if not isinstance(data, dict):
        raise KeyError(
            "metrics['data'] must be either a dataframe (singleclust) or a "
            "modality -> dataframe dictionary (multiview)."
        )
    if modalities is None or len(modalities) == 0:
        modalities = list(data.keys())

    solutions: List[Solution] = []
    matrices = []
    for i, mod in enumerate(modalities):
        if mod not in data:
            continue
        X, _ = _as_numeric_matrix(data[mod], subject_id_column)
        matrices.append(X)
        if indiv is not None and i < len(indiv):
            labels = np.asarray(indiv[i]).reshape(-1)
            if labels.size == X.shape[0]:
                stability = np.nan
                try:
                    if per_view_stability is not None and i < len(per_view_stability):
                        stability = float(per_view_stability[i])
                except Exception:
                    stability = np.nan
                solutions.append(Solution(name=str(mod), X=X, labels=labels, kind="modality", pipeline_stability_ari=stability))
    if final_labels is not None and matrices:
        n = min(x.shape[0] for x in matrices)
        Xc = np.hstack([x[:n] for x in matrices])
        labs = np.asarray(final_labels).reshape(-1)[:n]
        solutions.append(Solution(name="integrated", X=Xc, labels=labs, kind="integrated", pipeline_stability_ari=final_stability_value))
    return solutions


def run(args: argparse.Namespace) -> None:
    _safe_mkdir(args.output_dir)
    plot_dir = os.path.join(args.output_dir, "plots")
    _safe_mkdir(plot_dir)
    table_dir = os.path.join(args.output_dir, "tables")
    _safe_mkdir(table_dir)

    metrics = _load_metrics(args.metrics_pkl)
    solutions = _build_solutions(metrics, args.modalities, args.subject_id_column)
    if not solutions:
        raise RuntimeError("No valid modality or integrated solutions found in metrics file.")
    n_jobs = _resolve_n_jobs(args.n_jobs)
    print(f"Using {n_jobs} worker(s) for simulation-heavy validation steps.")

    summary_rows = []
    all_results: Dict[str, object] = {
        "settings": vars(args),
        "metrics_pkl": os.path.abspath(args.metrics_pkl),
        "included_tests": [
            "explicit_k1_baseline",
            "projection_median_split_quality_and_bootstrap_stability_benchmark",
            "hartigan_dip_test_multivariate_projection_search",
            "r_cluster_clusgap_statistic_including_k1",
            "python_sigclust_user_label_test_when_observed_k_is_2",
            "covariance_matched_gaussian_null_quality_and_bootstrap_stability",
            "pc1_pc2_cluster_plot",
        ],
        "solutions": {},
    }

    for idx, sol in enumerate(solutions):
        name = sol.name
        print(f"Running validation sensitivity analyses for {name}...")
        sol_dir = os.path.join(table_dir, name)
        _safe_mkdir(sol_dir)

        observed_quality = _quality_metrics(_standardize(sol.X), sol.labels)
        uni_cluster = _uni_cluster_baseline(sol.X)
        dip = _dip_test_projections(
            sol.X,
            max_components=args.max_pca_components,
            n_random_projections=args.projection_directions,
            n_null=args.gaussian_null_datasets,
            seed=_derive_seed("solution_dip_test", idx, base=args.seed),
            n_jobs=n_jobs,
        )
        projection_median = _projection_median_split(
            sol.X,
            sol.labels,
            n_bootstrap=args.null_stability_bootstraps,
            max_components=args.max_pca_components,
            n_random_projections=args.projection_directions,
            seed=_derive_seed("solution_projection_median", idx, base=args.seed),
        )
        try:
            gap_df, gap_meta = _gap_statistic_r_clusgap(
                sol.X,
                k_max=args.k_max,
                n_refs=args.gap_reference_datasets,
                seed=_derive_seed("solution_gap_r", idx, base=args.seed),
                max_components=args.max_pca_components,
            )
        except Exception as exc:
            gap_df, gap_meta = _gap_statistic(
                sol.X,
                k_max=args.k_max,
                n_refs=args.gap_reference_datasets,
                seed=_derive_seed("solution_gap_python", idx, base=args.seed),
                max_components=args.max_pca_components,
                n_jobs=n_jobs,
            )
            gap_meta["method"] = "Python fallback implementation of Tibshirani-style k-means gap statistic"
            gap_meta["package_fallback_reason"] = str(exc)
        sigclust = _sigclust_python_package(
            sol.X,
            sol.labels,
            n_sim=args.sigclust_simulations,
            seed=_derive_seed("solution_sigclust", idx, base=args.seed),
            max_components=args.max_pca_components,
            n_jobs=n_jobs,
        )
        gaussian_null = _gaussian_null_quality_and_stability(
            sol.X,
            sol.labels,
            sol.pipeline_stability_ari,
            n_null=args.gaussian_null_datasets,
            n_bootstrap=args.null_stability_bootstraps,
            seed=_derive_seed("solution_gaussian_null", idx, base=args.seed),
            max_components=args.max_pca_components,
            n_jobs=n_jobs,
        )

        gap_df.to_csv(os.path.join(sol_dir, f"{name}_gap_statistic.csv"), index=False)
        _plot_gap(gap_df, name, plot_dir)
        _plot_pc1_pc2(sol, plot_dir, args.max_pca_components)

        result = {
            "kind": sol.kind,
            "observed_quality": observed_quality,
            "uni_cluster_baseline": uni_cluster,
            "dip_test_projections": dip,
            "projection_median_split": projection_median,
            "gap_statistic": gap_meta,
            "sigclust": sigclust,
            "covariance_matched_gaussian_null": gaussian_null,
        }
        all_results["solutions"][name] = result
        with open(os.path.join(sol_dir, f"{name}_validation_sensitivity.json"), "w") as f:
            json.dump(result, f, indent=2, default=_json_default)

        summary_rows.append(
            {
                "solution": name,
                "kind": sol.kind,
                "n": observed_quality["n"],
                "n_features": observed_quality["n_features"],
                "observed_k": observed_quality["k"],
                "observed_composite": observed_quality["composite"],
                "observed_pipeline_stability_ari": sol.pipeline_stability_ari,
                "k1_composite": uni_cluster["quality"]["composite"],
                "projection_pca_components": projection_median["pca_components"],
                "projection_pca_variance_explained": projection_median["pca_variance_explained"],
                "projection_median_best_quality_projection": projection_median["best_quality_projection"],
                "projection_median_best_quality_composite": projection_median["best_quality"]["composite"],
                "projection_median_best_quality_ari_with_observed": projection_median["best_quality_ari_with_observed_labels"],
                "projection_median_best_ari_projection": projection_median["best_ari_projection"],
                "projection_median_best_ari_with_observed": projection_median["best_ari"],
                "projection_median_best_ari_composite": projection_median["best_ari_quality"]["composite"],
                "projection_median_stability_ari": projection_median["bootstrap_stability"]["mean_ari"],
                "projection_median_stability_sd_ari": projection_median["bootstrap_stability"]["sd_ari"],
                "dip_global_projection_p_value": dip["global_projection_p_value"],
                "dip_min_projection_p_value": dip["min_projection_p_value"],
                "dip_bonferroni_min_p_value": dip["bonferroni_min_p_value"],
                "dip_best_projection": dip["best_projection"],
                "dip_pc1_p_value": dip["pc1_p_value"],
                "gap_at_k1": float(gap_df.loc[gap_df["k"] == 1, "gap"].iloc[0]) if np.any(gap_df["k"] == 1) else np.nan,
                "gap_at_observed_k": float(gap_df.loc[gap_df["k"] == observed_quality["k"], "gap"].iloc[0]) if np.any(gap_df["k"] == observed_quality["k"]) else np.nan,
                "gap_selected_k_tibshirani": gap_meta["selected_k_tibshirani_rule"],
                "gap_selected_k_max_gap": gap_meta["selected_k_max_gap"],
                "gap_method": gap_meta.get("method", ""),
                "sigclust_available": sigclust["available"],
                "sigclust_p_value": sigclust["p_value"],
                "sigclust_p_value_normal_approx": sigclust["p_value_normal_approx"],
                "sigclust_observed_cluster_index": sigclust["observed_cluster_index"],
                "gaussian_null_p_quality": gaussian_null["p_quality_ge_observed_solution"],
                "gaussian_null_p_stability": gaussian_null["p_stability_ge_observed_pipeline"],
                "gaussian_null_mean_stability_ari": gaussian_null["null_stability_mean_ari"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.output_dir, "cluster_validation_sensitivity_summary.csv"), index=False)
    with open(os.path.join(args.output_dir, "cluster_validation_sensitivity_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    print(f"Saved validation sensitivity summary to {args.output_dir}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_pkl", required=True, help="Path to final_metrics.pkl or fold metrics.pkl.")
    parser.add_argument("--output_dir", required=True, help="Directory for validation outputs.")
    parser.add_argument("--subject_id_column", default="src_subject_id")
    parser.add_argument("--modalities", nargs="*", default=None)
    parser.add_argument("--k_max", type=int, default=10)
    parser.add_argument("--gap_reference_datasets", type=int, default=100)
    parser.add_argument("--sigclust_simulations", type=int, default=200)
    parser.add_argument("--gaussian_null_datasets", type=int, default=100)
    parser.add_argument("--null_stability_bootstraps", type=int, default=30)
    parser.add_argument("--max_pca_components", type=int, default=20)
    parser.add_argument("--projection_directions", type=int, default=200,
                        help="Random projection directions added to the PC axes for projection-based dip and median-split checks.")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Parallel workers for simulation-heavy validation steps. Use -1 for all CPUs.")
    parser.add_argument("--seed", type=int, default=314159)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
