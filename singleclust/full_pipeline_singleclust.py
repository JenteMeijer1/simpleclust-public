#!/usr/bin/env python
"""Single-view grid-search clustering pipeline used by the simpleclust profile.

This module shares the Slurm stage interface of the multiview pipeline
(`init`, `bootstrap`, `gather`, `outer`, `merge`) so the same shell wrappers can
schedule both workflows. In the SchizBull/simpleclust profile the search itself
is not a genetic algorithm: `init` creates a deterministic linkage-by-k grid,
bootstraps score those candidates, gather ranks them, outer selects a valid fold
solution, and merge fits the final all-subject solution.

Some object and folder names still say "GA" or "population" for compatibility
with older pickles and scheduler code. Treat them as candidate-grid containers
in this singleclust path.
"""
import os
import sys

# Limit BLAS/Numexpr threading to avoid oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# full_pipeline.py — Updated to support nested SLURM pipelines

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# --- Imports ------------------------------------------------
from Utils import *
from Utils import preprocessing as multiclust_preprocessing
import time
import pandas as pd
import re
import dill
import argparse
import hashlib
from itertools import combinations
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from joblib import Parallel, delayed
import warnings
import traceback

import numpy as np
import os
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from operator import itemgetter
from deap import base, creator, tools, algorithms

# --- SciPy imports for cophenetic correlation calculation ---
import scipy.cluster.hierarchy as hierarchy
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

from sklearn.model_selection import KFold
# SVM imports retained but SVM mode will be commented out
from sklearn.svm import SVC
from sklearn.cluster import KMeans, SpectralClustering
import random
import torch.nn as nn
import torch
torch.set_num_threads(1)
import gc
from sklearn.decomposition import NMF, PCA, SparsePCA
from collections import defaultdict
import pickle

# Import own functions
from VAE import run_VAE_complete
from AE  import run_AE_complete
from clustering_functions import (
    _safe_quality,
    enforce_min_cluster_size,
    run_ensemble_clustering,
    decode_search_candidate,
)
from SVM import *
import glob

_MAX_32BIT_SEED = 2 ** 32 - 1
METRICS_SCHEMA_VERSION = 3


def _flag_enabled(value):
    """Handle flag enabled."""
    return str(value).strip().upper() == "TRUE"


def _operational_min_cluster_n(args, current_n, reference_n):
    """Handle operational min cluster n."""
    return resolve_min_cluster_n(
        args.mincluster_n,
        current_n=current_n,
        reference_n=reference_n,
        mode=getattr(args, "mincluster_resample_mode", "fixed"),
    )


def _derive_seed(*parts, base=0):
    """Handle derive seed."""
    payload = "|".join(str(part) for part in (base,) + tuple(parts)).encode("utf-8")
    seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return seed % _MAX_32BIT_SEED


def _seed_everything(seed):
    """Handle seed everything."""
    np.random.seed(int(seed))
    random.seed(int(seed))
    if torch is not None:
        torch.manual_seed(int(seed))


def _search_bootstrap_seed(fold_index, bootstrap_index):
    """Handle search bootstrap seed."""
    return _derive_seed(
        "singleclust_search_bootstrap",
        int(fold_index or 0),
        int(bootstrap_index or 0),
    )

SEARCH_OBJECTIVE_ALIASES = {
    "qual": "quality",
    "quality": "quality",
    "stab": "stab_ari",
    "stability": "stab_ari",
    "stability_ari": "stab_ari",
    "stab_ari": "stab_ari",
    "final_stability_ari": "stab_ari",
    "stability_jaccard": "stab_jaccard",
    "stab_jaccard": "stab_jaccard",
    "final_stability_jaccard": "stab_jaccard",
    "final_quality": "quality",
}
# Backward-compatible alias for legacy references.
GA_OBJECTIVE_ALIASES = SEARCH_OBJECTIVE_ALIASES
DEFAULT_SEARCH_OBJECTIVES = ["stab_ari", "quality"]


def _normalize_method_list(values):
    """Normalize method list."""
    out = []
    for val in values or []:
        if isinstance(val, str) and "," in val:
            parts = [p for p in val.split(",") if p]
        else:
            parts = [val]
        for part in parts:
            if part is None:
                continue
            text = str(part).strip().lower()
            if text:
                out.append(text)
    return out



def _normalize_objective_tokens(raw_tokens, optimisation_mode):
    """Map user-specified search objective tokens to canonical names."""
    if not raw_tokens:
        # Defaults: single-objective -> final-stability (ARI); multi -> standard multi-objective set.
        if optimisation_mode == "single":
            tokens = ["final_stability_ari"]
        else:
            tokens = DEFAULT_SEARCH_OBJECTIVES
    else:
        tokens = []
        for tok in raw_tokens:
            if isinstance(tok, str) and "," in tok:
                tokens.extend([t for t in tok.split(",") if t])
            else:
                tokens.append(tok)

    normalized = []
    for tok in tokens:
        key = str(tok).strip().lower()
        if key not in SEARCH_OBJECTIVE_ALIASES:
            valid = ", ".join(sorted(set(SEARCH_OBJECTIVE_ALIASES.values())))
            raise ValueError(f"Unknown search objective '{tok}'. Valid options: {valid}")
        normalized.append(SEARCH_OBJECTIVE_ALIASES[key])
    return normalized


# Helper to choose the primary metric keys for summary attributes based on search objectives
def _primary_metric_keys(args):
    """Return the primary single-clustering metric keys used in summaries."""
    objs = list(getattr(args, "ga_objectives", []))
    stab_key = None
    for obj in objs:
        if obj in {"stab_ari", "stab_jaccard"}:
            stab_key = obj
            break
    if stab_key is None:
        stab_key = "stab_ari"
    return stab_key, "quality"

def _ensure_multi_fitness_class(args):
    """Ensure the DEAP multi-objective fitness/candidate classes exist (DEAP internal class name kept for compatibility)."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    class_name = None
    if args.optimisation == 'multi':
        class_name = f"FitnessMulti{len(args.ga_objectives)}"
        if hasattr(creator, class_name):
            cls = getattr(creator, class_name)
        else:
            weights = tuple([1.0] * len(args.ga_objectives))
            creator.create(class_name, base.Fitness, weights=weights)
            cls = getattr(creator, class_name)
    else:
        cls = None
    if hasattr(creator, "Individual"):
        delattr(creator, "Individual")
    if args.optimisation == 'multi':
        creator.create("Individual", list, fitness=cls)
    else:
        creator.create("Individual", list, fitness=creator.FitnessMax)
    args.multi_fitness_class_name = class_name
    return cls


def _get_multi_fitness_class(args):
    """Return the DEAP multi-objective fitness class for current args."""
    name = getattr(args, "multi_fitness_class_name", None)
    if not name:
        return None
    return getattr(creator, name)


def _search_gene_names(args):
    """Canonical hyperparameter-candidate layout for the simplified search space."""
    return ["linkage", "k"]


def _build_grid_population(args):
    """
    Build a deterministic Cartesian grid of single-clustering hyperparameter candidates (linkage x k).

    We keep using DEAP Individual objects so the rest of the bootstrap/gather/outer
    pipeline (fitness assignment, Pareto front persistence, pickling) remains unchanged.
    """
    linkages = list(args.linkages)
    if not linkages:
        raise ValueError("At least one linkage must be supplied via --linkages.")
    if args.k_min > args.k_max:
        raise ValueError(f"Invalid k-range: k_min ({args.k_min}) > k_max ({args.k_max}).")

    names = _search_gene_names(args)
    k_values = list(range(args.k_min, args.k_max + 1))
    combos = [
        [linkage, k]
        for linkage in linkages
        for k in k_values
    ]

    grid_size = len(combos)
    cap = int(getattr(args, "n_population", 0) or 0)
    if cap < 0:
        raise ValueError("--grid_max_candidates/--n_population must be zero or positive.")

    if cap > 0 and cap < grid_size:
        if args.seed is not None and len(combos) > 1:
            rnd = random.Random(args.seed)
            rnd.shuffle(combos)
        print(
            f"Grid contains {grid_size} candidates; truncating to {cap} "
            f"(shuffle controlled by --seed when provided)."
        )
        combos = combos[:cap]
    else:
        cap_text = "unlimited" if cap == 0 else str(cap)
        print(f"Grid contains {grid_size} candidates; cap={cap_text}, evaluating the complete grid.")

    pop = []
    for vals in combos:
        ind = creator.Individual(vals)
        ind.gene_names = names
        pop.append(ind)
    return pop, names


# --- Utility functions ---------------------------------------
_DATA = {}
def _init_worker(data_list, subject_id_list, args=None):
    """Handle init worker."""
    _DATA.clear()
    _DATA["data_list"] = data_list
    _DATA["subject_id_list"] = subject_id_list
    if args is not None:
        _DATA["args"] = args

# Helper function for parallel clustering in bootstrap (now only takes the candidate)
def _cluster_candidate(cand, args=None):
    """Handle cluster candidate."""
    data_list = _DATA["data_list"]
    subject_id_list = _DATA["subject_id_list"]
    if args is None and "args" in _DATA:
        args = _DATA["args"]
    params = decode_search_candidate(len(data_list), cand)
    labels, scores = run_ensemble_clustering(
        data_list,
        **params,
        subject_id_list=subject_id_list,
        inner_jobs=1,
        pre_inner_jobs=1,
        mincluster=args.mincluster,
        mincluster_n=getattr(args, "mincluster_n_applied", args.mincluster_n),
        internal_ensemble_enabled=getattr(args, "internal_ensemble_enabled", "FALSE"),
        internal_ensemble_bcs=getattr(args, "internal_ensemble_bcs", 5),
        internal_ensemble_sample_frac=getattr(args, "internal_ensemble_sample_frac", 0.8),
        internal_ensemble_feature_frac=getattr(args, "internal_ensemble_feature_frac", 1.0),
        internal_ensemble_seed=_derive_seed(
            "singleclust_search_internal_ensemble",
            int(getattr(args, "fold_index", 0) or 0),
            int(getattr(args, "bootstrap_index", 0) or 0),
        ),
    )
    return labels, scores


def _init_merge_bootstrap_worker(
    X_latent,
    ids_all,
    final_params,
    mincluster,
    mincluster_n,
    mincluster_resample_mode="fixed",
    internal_ensemble_enabled="FALSE",
    internal_ensemble_bcs=5,
    internal_ensemble_sample_frac=0.8,
    internal_ensemble_feature_frac=1.0,
    preprocessing_mode="outside",
    raw_df=None,
    meta=None,
    args=None,
):
    """Handle init merge bootstrap worker."""
    _DATA.clear()
    _DATA["merge_X_latent"] = np.asarray(X_latent, dtype=np.float32, copy=False)
    _DATA["merge_ids_all"] = list(ids_all)
    _DATA["merge_final_params"] = dict(final_params)
    _DATA["merge_mincluster"] = mincluster
    _DATA["merge_mincluster_n"] = mincluster_n
    _DATA["merge_mincluster_resample_mode"] = mincluster_resample_mode
    _DATA["merge_internal_ensemble_enabled"] = internal_ensemble_enabled
    _DATA["merge_internal_ensemble_bcs"] = internal_ensemble_bcs
    _DATA["merge_internal_ensemble_sample_frac"] = internal_ensemble_sample_frac
    _DATA["merge_internal_ensemble_feature_frac"] = internal_ensemble_feature_frac
    _DATA["merge_preprocessing_mode"] = preprocessing_mode
    _DATA["merge_raw_df"] = raw_df
    _DATA["merge_meta"] = meta
    _DATA["merge_args"] = args


def _run_merge_bootstrap(bootstrap_index):
    """Run merge bootstrap."""
    X_latent = _DATA["merge_X_latent"]
    ids_all = _DATA["merge_ids_all"]
    final_params = _DATA["merge_final_params"]
    mincluster = _DATA["merge_mincluster"]
    mincluster_n = _DATA["merge_mincluster_n"]

    n_samples = len(ids_all)
    seed = _derive_seed("singleclust_merge_bootstrap", int(bootstrap_index), base=12345)
    rng = np.random.default_rng(seed)
    m = max(3, int(round(0.8 * n_samples)))
    idx = rng.choice(n_samples, size=m, replace=False)
    if _DATA.get("merge_preprocessing_mode", "outside") == "inside":
        args = _DATA["merge_args"]
        raw_df = _DATA["merge_raw_df"].iloc[idx].reset_index(drop=True)
        ae_data_b, subject_id_list_b, df_final_b = preprocessing(
            raw_df,
            _DATA["merge_meta"],
            subject_id_column=args.subject_id_column,
            col_threshold=args.col_threshold,
            row_threshold=args.row_threshold,
            skew_threshold=args.skew_threshold,
            scaler_type=args.scaler_type,
            modalities=args.modalities,
            dummy_code_modalities=args.dummy_code_modalities,
            mixed_categorical_modalities=args.mixed_categorical_modalities,
        )
        _, Xb = _build_latent_matrix(args, ae_data_b, df_final_b, seed_value=seed)
        ids_b = list(subject_id_list_b[0])
    else:
        Xb = X_latent[idx, :]
        ids_b = [ids_all[i] for i in idx]
    mincluster_n_applied = resolve_min_cluster_n(
        mincluster_n,
        current_n=len(ids_b),
        reference_n=n_samples,
        mode=_DATA.get("merge_mincluster_resample_mode", "fixed"),
    )
    labels_b, _ = run_ensemble_clustering(
        Xb,
        **final_params,
        subject_id_list=[ids_b],
        inner_jobs=1,
        pre_inner_jobs=1,
        mincluster=mincluster,
        mincluster_n=mincluster_n_applied,
        internal_ensemble_enabled=_DATA.get("merge_internal_ensemble_enabled", "FALSE"),
        internal_ensemble_bcs=_DATA.get("merge_internal_ensemble_bcs", 5),
        internal_ensemble_sample_frac=_DATA.get("merge_internal_ensemble_sample_frac", 0.8),
        internal_ensemble_feature_frac=_DATA.get("merge_internal_ensemble_feature_frac", 1.0),
        internal_ensemble_seed=_derive_seed(
            "singleclust_merge_internal_ensemble",
            int(bootstrap_index),
            base=900000,
        ),
    )
    return {
        'orig_ids': ids_b,
        'labels': np.asarray(labels_b, dtype=int),
        'requested_k': int(final_params.get('k', 2)),
        'mincluster_n_requested': int(mincluster_n),
        'mincluster_n_applied': int(mincluster_n_applied),
        'reference_n': int(n_samples),
        'current_n': int(len(ids_b)),
    }

def save_pickle(path, obj):
    """Save an object to a pickle file safely."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _dimred_n_components(args, n_samples, n_features):
    """Bound requested latent dimensionality to a valid component count."""
    max_by_samples = max(1, n_samples - 1)
    return min(args.maxPC, n_features, max_by_samples)


def _run_sparse_pca(X_df, args, seed_value):
    """Fit SparsePCA and return float32 latent embeddings plus the fitted model."""
    n_components = _dimred_n_components(args, X_df.shape[0], X_df.shape[1])
    spca = SparsePCA(
        n_components=n_components,
        alpha=args.spca_alpha,
        ridge_alpha=args.spca_ridge_alpha,
        max_iter=args.spca_max_iter,
        random_state=seed_value,
        n_jobs=1,
    )
    X_spca = spca.fit_transform(X_df.to_numpy(dtype=np.float32, copy=True))
    return np.asarray(X_spca, dtype=np.float32, copy=False), spca


def _nonnegative_matrix_for_nmf(X):
    """Handle nonnegative matrix for nmf."""
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if X.shape[0] == 0 or X.shape[1] == 0:
        return X.astype(np.float32, copy=False), np.zeros((X.shape[1],), dtype=np.float32)
    mins = np.nanmin(X, axis=0)
    shift = np.where(mins < 0, -mins, 0.0).astype(np.float32)
    X_nonnegative = np.maximum(X + shift, 0.0)
    return X_nonnegative.astype(np.float32, copy=False), shift


def _make_sparse_nmf(n_components, alpha, l1_ratio, max_iter, random_state):
    """Create sparse nmf."""
    kwargs = dict(
        n_components=int(n_components),
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        l1_ratio=float(l1_ratio),
        max_iter=int(max_iter),
        random_state=random_state,
    )
    try:
        return NMF(alpha_W=float(alpha), alpha_H=float(alpha), **kwargs)
    except TypeError:
        return NMF(alpha=float(alpha), **kwargs)


def _run_sparse_nmf(X_df, args, seed_value):
    """Fit SparseNMF and return float32 latent embeddings plus the fitted model."""
    X_nonnegative, shift = _nonnegative_matrix_for_nmf(X_df.to_numpy(dtype=np.float32, copy=True))
    n_components = min(args.maxPC, X_nonnegative.shape[0], X_nonnegative.shape[1])
    if n_components < 1:
        return X_nonnegative, {"nmf": None, "shift": shift}
    nmf = _make_sparse_nmf(
        n_components=n_components,
        alpha=args.snmf_alpha,
        l1_ratio=args.snmf_l1_ratio,
        max_iter=args.snmf_max_iter,
        random_state=seed_value,
    )
    X_nmf = nmf.fit_transform(X_nonnegative)
    return np.asarray(X_nmf, dtype=np.float32, copy=False), {"nmf": nmf, "shift": shift}


def _dimred_run_label(args):
    """Stable label for the configured dimensionality-reduction run."""
    dim = None if args.dim_reduction is None else str(args.dim_reduction).lower()
    if dim in (None, "", "none"):
        return "none"
    if dim == "pca":
        return f"pca_{int(args.maxPC)}"
    if dim == "sparsepca":
        return f"sparsepca_{int(args.maxPC)}"
    if dim == "sparsenmf":
        return f"sparsenmf_{int(args.maxPC)}_alpha_{args.snmf_alpha:g}_l1_{args.snmf_l1_ratio:g}"
    if dim in {"ae", "autoencoder"}:
        return "ae"
    if dim == "sparseae":
        return f"sparseae_l1_{args.sparse_l1_lambda:g}"
    if dim == "vae":
        return "vae"
    if dim == "sparsevae":
        return f"sparsevae_l1_{args.sparse_l1_lambda:g}"
    return str(args.dim_reduction)


def _dimred_sparse_l1(args):
    """Return the configured latent L1 penalty for sparse autoencoder variants."""
    dim = None if args.dim_reduction is None else str(args.dim_reduction).lower()
    if dim in {"sparseae", "sparsevae"}:
        return float(args.sparse_l1_lambda)
    return 0.0


def load_pickle(path):
    """Load an object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

def _resolve_path(base_dir, path):
    """Return an absolute path anchored at base_dir when a relative path is supplied."""
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))

def _output_root(base_dir, env_name, default_name):
    """Resolve a run output root using the same convention as full_pipeline.py."""
    configured = os.environ.get(env_name)
    if configured:
        return _resolve_path(base_dir, configured)
    return os.path.join(base_dir, default_name)

def _search_root(base_dir, fold_index):
    """Canonical singleclust search root used by the shared Slurm pipeline."""
    return os.path.join(
        _output_root(base_dir, "INTERMEDIATES_DIR", "intermediates"),
        f"fold{fold_index}",
        "ga",
    )

def _ga_root(base_dir, fold_index):
    """Backward-compatible alias for legacy GA naming."""
    return _search_root(base_dir, fold_index)

def preprocessing(df,
                  meta,
                  subject_id_column='src_subject_id',
                  col_threshold=0.5, row_threshold=0.5,
                  skew_threshold=0.75,
                  scaler_type='robust',
                  modalities=None,
                  dummy_code_modalities=None,
                  mixed_categorical_modalities=None,
                  impute_parea=False,
                  return_artifact=False):
    """Build one aligned feature matrix from the parent multiclust preprocessing."""
    if modalities is None:
        modalities = ['Internalising', 'Functioning', 'Cognition', 'Detachment', 'Psychoticism']

    processed = multiclust_preprocessing(
        df=df,
        meta=meta,
        subject_id_column=subject_id_column,
        col_threshold=col_threshold,
        row_threshold=row_threshold,
        skew_threshold=skew_threshold,
        scaler_type=scaler_type,
        modalities=modalities,
        dummy_code_modalities=dummy_code_modalities,
        mixed_categorical_modalities=mixed_categorical_modalities,
        impute_parea=impute_parea,
        export_preprocessing_details=return_artifact,
    )
    if return_artifact:
        _, _, modality_frames, preprocessing_details = processed
    else:
        _, _, modality_frames = processed
        preprocessing_details = None

    selected = [name for name in modalities if name in modality_frames]
    if not selected:
        raise ValueError(f"None of the requested modalities were produced: {modalities}")

    merged = None
    for modality in selected:
        frame = modality_frames[modality].copy()
        feature_columns = [column for column in frame.columns if column != subject_id_column]
        frame = frame[[subject_id_column] + feature_columns]
        frame = frame.rename(columns={column: f"{modality}__{column}" for column in feature_columns})
        merged = frame if merged is None else merged.merge(frame, on=subject_id_column, how='inner', validate='one_to_one')

    if merged is None or merged.empty:
        raise ValueError("Parent multiclust preprocessing produced no aligned simpleclust rows.")
    feature_matrix = merged.drop(columns=[subject_id_column]).to_numpy(dtype=np.float32, copy=True)
    subject_ids = merged[subject_id_column].tolist()
    ae_data = [feature_matrix]
    subject_id_list = [subject_ids]

    if return_artifact:
        artifact = {
            'preprocessing_details': preprocessing_details,
            'modalities': selected,
            'feature_columns': [column for column in merged.columns if column != subject_id_column],
        }
        return ae_data, subject_id_list, merged, artifact
    return ae_data, subject_id_list, merged




# --- Helpers: collapse duplicate subject IDs within a bootstrap (majority vote) ---
def _collapse_duplicates(orig_ids, labels):
    """
    Given per-row orig_ids (possibly with duplicates) and corresponding labels,
    collapse to a single label per unique subject via majority vote.
    Ties are broken deterministically by choosing the smallest label value.

    Returns
    -------
    unique_ids : np.ndarray (ordered by first appearance)
    collapsed_labels : np.ndarray aligned to unique_ids
    """
    orig_ids = np.asarray(orig_ids)
    labels = np.asarray(labels)
    # Preserve first-seen order of subjects
    seen = {}
    order = []
    for sid in orig_ids:
        if sid not in seen:
            seen[sid] = True
            order.append(sid)
    unique_ids = np.array(order)
    # Majority vote per subject
    collapsed = []
    for sid in unique_ids:
        mask = (orig_ids == sid)
        vals, counts = np.unique(labels[mask], return_counts=True)
        # break ties by choosing the smallest label among those with maximal count
        maxc = counts.max()
        choice = vals[counts == maxc].min()
        collapsed.append(choice)
    return unique_ids, np.asarray(collapsed)

def _unique_ids_from_orig_ids(orig_ids):
    """Return unique subject IDs in first-seen order (matches _collapse_duplicates)."""
    orig_ids = np.asarray(orig_ids)
    seen = {}
    order = []
    for sid in orig_ids:
        if sid not in seen:
            seen[sid] = True
            order.append(sid)
    return np.array(order)

def precompute_consensus_cache(label_dicts):
    """
    Precompute union IDs and per-bootstrap union indices for consensus_pac_ccc.
    This avoids rebuilding ID maps and reduces per-call overhead.
    """
    unique_ids_list = []
    id_set = set()
    for d in label_dicts:
        uids = _unique_ids_from_orig_ids(d["orig_ids"])
        unique_ids_list.append(uids)
        id_set.update(uids.tolist())

    union_ids = np.array(sorted(id_set))
    index = {sid: i for i, sid in enumerate(union_ids)}
    idxs_list = [
        np.fromiter((index[sid] for sid in uids), dtype=int, count=len(uids))
        for uids in unique_ids_list
    ]
    return {"union_ids": union_ids, "idxs_list": idxs_list}

def precompute_bootstrap_pair_alignment(label_dicts):
    """
    Precompute bootstrap-pair alignment indices based on orig_ids only.
    This avoids repeating np.intersect1d for every candidate.
    """
    unique_ids_list = []
    for d in label_dicts:
        unique_ids_list.append(_unique_ids_from_orig_ids(d["orig_ids"]))

    pair_indices = []
    n = len(unique_ids_list)
    for i in range(n):
        for j in range(i + 1, n):
            common, idx1, idx2 = np.intersect1d(
                unique_ids_list[i],
                unique_ids_list[j],
                return_indices=True
            )
            if len(common) > 1:
                pair_indices.append((i, j, idx1, idx2))

    return {
        "unique_ids_list": unique_ids_list,
        "pair_indices": pair_indices,
    }

# --- Co-association stability across bootstraps ---
def coassociation_stability(label_dicts, label_key):
    """
    Compute per-cluster co-association stability across bootstraps.

    Parameters
    ----------
    label_dicts : list of dict
        One dict per bootstrap, each with:
          - "orig_ids": 1D array-like of subject IDs in this bootstrap
          - label_key : 1D array-like of labels for these IDs
        Duplicates within a bootstrap are allowed and are collapsed by
        majority vote before use.

    label_key : str
        Key in each dict to use for labels (e.g. "labels").

    Returns
    -------
    cluster_stability: mean of individual cluster stabilities
        Stability score for each cluster in the *reference* partition,
        in ascending order of cluster label. For a given cluster k, the
        score is the mean consensus value M(i,j) over all pairs of
        subjects i,j that belong to cluster k in the reference
        clustering.
    •	✅ Report it as “fraction of times items in the cluster appear together”
	•	✅ High = stable, low = unstable

    Notes
    -----
    1. We first construct the N x N consensus matrix M where
         M(i,j) = (# times i and j are clustered together)
                  / (# times i and j co-occur in a bootstrap),
       using all bootstraps.

    2. We then take the clustering from the *first* bootstrap (after
       collapsing duplicates) as the reference partition and compute
       a per-cluster stability:
         m(k) = mean of M(i,j) over all pairs i,j in cluster k.

    3. This matches the "cluster consensus" idea from consensus
       clustering: values close to 1 mean that cluster k is very
       stable across resampling.
    """
    # --- Step 1: collapse duplicates and build union of subject IDs ---
    collapsed = []
    id_set = set()
    for d in label_dicts:
        uids, labs = _collapse_duplicates(d["orig_ids"], d[label_key])
        collapsed.append((np.asarray(uids), np.asarray(labs)))
        id_set.update(uids.tolist())

    union_ids = np.array(sorted(id_set))
    n = len(union_ids)
    if n <= 1:
        return -3, -3 # Not enough subjects for co-association stability, Error code.

    # Map subject id -> index in [0, n)
    index = {sid: i for i, sid in enumerate(union_ids)}

    # --- Step 2: accumulate co-presence and same-cluster counts ---
    same = np.zeros((n, n), dtype=np.uint32)
    co   = np.zeros((n, n), dtype=np.uint32)

    for uids, labs in collapsed:
        vec = np.full(n, -1, dtype=int)  # -1 means "absent in this bootstrap"
        idxs = np.fromiter((index[sid] for sid in uids), dtype=int, count=len(uids))
        vec[idxs] = labs

        present = (vec != -1)
        co_mask = np.outer(present, present)
        eq_mask = (vec[:, None] == vec[None, :]) & co_mask

        same += eq_mask
        co   += co_mask

    # If no pair ever co-occurs, nothing to compute
    if not np.any(co[np.triu_indices(n, k=1)] > 0):
        return -9, -9 # Not enough co-occurrences for co-association stability, Error code.

    # --- Step 3: build consensus matrix M(i,j) in [0,1] ---
    with np.errstate(divide="ignore", invalid="ignore"):
        consensus = np.zeros((n, n), dtype=float)
        mask = co > 0
        consensus[mask] = same[mask].astype(float) / co[mask].astype(float)

    # --- Step 4: choose a reference clustering (first bootstrap) ---
    ref_uids, ref_labs = collapsed[0]
    ref_vec = np.full(n, -1, dtype=int)
    ref_idxs = np.fromiter((index[sid] for sid in ref_uids), dtype=int, count=len(ref_uids))
    ref_vec[ref_idxs] = ref_labs

    valid_mask = (ref_vec != -1)
    if not np.any(valid_mask):
        return -333, -333 # No subjects in reference clustering, Error code.

    cluster_labels = np.unique(ref_vec[valid_mask])

    # --- Step 5: compute per-cluster stability m(k) ---
    cluster_stabilities = []
    for k in cluster_labels:
        members = np.where(ref_vec == k)[0]
        if len(members) < 2:
            # Cluster of size 0 or 1: define stability as 0.0
            cluster_stabilities.append(0.0)
            continue

        # Consensus submatrix for this cluster
        subM = consensus[np.ix_(members, members)]
        iu_k = np.triu_indices(len(members), k=1)
        vals = subM[iu_k]

        if vals.size == 0:
            cluster_stabilities.append(0.0)
        else:
            cluster_stabilities.append(float(np.mean(vals)))

    # --- Step 6: compute overall CCC for consensus matrix ---
    # Distances from consensus
    D = 1.0 - consensus
    # Convert to condensed vector
    dvec = squareform(D, checks=False)

    # Hierarchical linkage on these distances
    Z = linkage(dvec, method="average")

    # cophenet returns (cophenetic_correlation, cophenetic_distances)
    ccc, _ = cophenet(Z, dvec)

    cluster_stability = np.mean(cluster_stabilities) if cluster_stabilities else 0

    return float(cluster_stability), float(ccc)


import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform

def consensus_pac_ccc(
    label_dicts,
    label_key,
    range_min=0.1,
    range_max=0.9,
    linkage_method="average",
    return_consensus=False,
    return_ecdf=False,
    precomputed_cache=None,
):
    """
    Compute MATLAB-style (Doms code) consensus diagnostics for a FIXED k:
      - consensus matrix (co-association)
      - ECDF of off-diagonal consensus values
      - PAC (Proportion of Ambiguous Clustering) over [range_min, range_max]
      - Cophenetic correlation coefficient (CCC) from hierarchical clustering on (1 - consensus)

    This matches the MATLAB pipeline conceptually:
      dd_calconsens  -> consensus matrix
      dd_ecdf        -> ECDF of consensus values
      dd_ecdfmin3    -> PAC over [range_min, range_max] (here: fraction in that interval)
      dd_cophenetic  -> CCC on the consensus-derived distances

    Parameters
    ----------
    label_dicts : list of dict
        One dict per bootstrap. Each dict must have:
          - "orig_ids": 1D array-like of subject IDs in this bootstrap
          - label_key : 1D array-like of labels for these IDs
        Duplicates within a bootstrap are allowed and collapsed by majority vote.

    label_key : str
        Key in each dict to use for labels (e.g. "labels").

    range_min, range_max : float
        Ambiguity interval for PAC, typically 0.1 and 0.9.

    linkage_method : str
        Linkage method for hierarchical clustering used in CCC (e.g. "average").

    precomputed_cache : dict or None
        Optional cache from precompute_consensus_cache(label_dicts) to reuse
        union ID mapping across calls.

    Returns
    -------
    out : dict
        {
          "consensus": (n, n) float array in [0,1],
          "union_ids": (n,) array of subject IDs (sorted),
          "ecdf_x": (m,) sorted values of off-diagonal consensus entries,
          "ecdf_f": (m,) ECDF values in [0,1],
          "PAC": float,
          "CCC": float,
          "meta": {...}
        }

    Notes
    -----
    - PAC here is computed as the fraction of *off-diagonal* consensus entries
      that fall strictly between [range_min, range_max], i.e.
        PAC = P(range_min < M(i,j) < range_max) over i<j with defined co-occurrence.
      Lower PAC => crisper / more stable clustering structure.
    """

    # --- Step 1: collapse duplicates and build union of subject IDs ---
    collapsed = []
    id_set = set() if precomputed_cache is None else None
    for d in label_dicts:
        if "orig_ids" not in d or label_key not in d:
            raise KeyError(f"Each dict must contain 'orig_ids' and '{label_key}'")
        uids, labs = _collapse_duplicates(d["orig_ids"], d[label_key])
        uids = np.asarray(uids)
        labs = np.asarray(labs)
        if uids.shape[0] != labs.shape[0]:
            raise ValueError("orig_ids and labels must have the same length after collapsing.")
        collapsed.append((uids, labs))
        if id_set is not None:
            id_set.update(uids.tolist())

    if precomputed_cache is None:
        union_ids = np.array(sorted(id_set))
        idxs_list = None
    else:
        union_ids = precomputed_cache.get("union_ids")
        idxs_list = precomputed_cache.get("idxs_list")

    n = len(union_ids)
    if n <= 1:
        return {
            "consensus": None,
            "union_ids": union_ids,
            "ecdf_x": np.array([]),
            "ecdf_f": np.array([]),
            "PAC": np.nan,
            "CCC": np.nan,
            "meta": {"error": "Not enough subjects to compute consensus."},
        }

    index = None
    if idxs_list is None or len(idxs_list) != len(collapsed):
        index = {sid: i for i, sid in enumerate(union_ids)}

    # --- Step 2: accumulate co-presence and same-cluster counts ---
    same = np.zeros((n, n), dtype=np.uint32)
    co = np.zeros((n, n), dtype=np.uint32)

    for b, (uids, labs) in enumerate(collapsed):
        if idxs_list is not None and b < len(idxs_list) and len(idxs_list[b]) == len(labs):
            idxs = idxs_list[b]
        else:
            if index is None:
                index = {sid: i for i, sid in enumerate(union_ids)}
            idxs = np.fromiter((index[sid] for sid in uids), dtype=int, count=len(uids))

        if len(idxs) == 0:
            continue

        eq_mask = (labs[:, None] == labs[None, :]).astype(np.uint32)
        same[np.ix_(idxs, idxs)] += eq_mask
        co[np.ix_(idxs, idxs)] += 1

    # Only consider pairs that ever co-occurred at least once
    iu = np.triu_indices(n, k=1)
    co_u = co[iu]
    if not np.any(co_u > 0):
        return {
            "consensus": None,
            "union_ids": union_ids,
            "ecdf_x": np.array([]),
            "ecdf_f": np.array([]),
            "PAC": np.nan,
            "CCC": np.nan,
            "meta": {"error": "No co-occurring pairs across bootstraps."},
        }

    # --- Step 3: build consensus matrix M(i,j) in [0,1] ---
    consensus = np.zeros((n, n), dtype=float)
    mask = co > 0
    consensus[mask] = same[mask].astype(float) / co[mask].astype(float)
    np.fill_diagonal(consensus, 1.0)  # conventional; doesn't affect off-diagonal PAC/CCC

    # --- Step 4: ECDF of off-diagonal consensus entries (defined pairs only) ---
    # Use only pairs with co-occurrence > 0
    vals = consensus[iu][co_u > 0]
    # ECDF: x sorted, f = (1..m)/m
    ecdf_x = np.sort(vals)
    m = ecdf_x.size
    ecdf_f = (np.arange(1, m + 1) / m) if m > 0 else np.array([])

    # Optionally drop ECDF arrays if not requested
    if not return_ecdf:
        ecdf_x = np.array([])
        ecdf_f = np.array([])

    # --- Step 5: PAC over [range_min, range_max] ---
    # MATLAB PAC is “mass in the ambiguous middle” between two thresholds.
    # Use strict interior by default; if you prefer inclusive, change comparisons.
    if m == 0:
        pac = np.nan
    else:
        pac = float(np.mean((np.sort(vals) > range_min) & (np.sort(vals) < range_max)))

    # --- Step 6: Cophenetic correlation coefficient (CCC) ---
    # Build distances from consensus: D = 1 - M
    D = 1.0 - consensus
    # condensed vector for linkage/cophenet
    dvec = squareform(D, checks=False)
    Z = linkage(dvec, method=linkage_method)
    ccc, _ = cophenet(Z, dvec)

    return {
        "consensus": consensus if return_consensus else None,
        "union_ids": union_ids,
        "ecdf_x": ecdf_x,
        "ecdf_f": ecdf_f,
        "PAC": pac,
        "CCC": float(ccc),
        "meta": {
            "n_subjects": int(n),
            "n_bootstraps": int(len(collapsed)),
            "range_min": float(range_min),
            "range_max": float(range_max),
            "linkage_method": linkage_method,
            "n_pairs_used": int(m),
        },
    }



# --- ARI and Jaccard-based stability across bootstraps ---
def ari_stability_common_subjects(label_dicts, label_key, precomputed_alignment=None):
    """
    Compute mean pairwise ARI for label arrays after collapsing duplicates
    within each bootstrap via majority vote, then aligning on common subjects.
    Each dict must have "orig_ids" and the labels under label_key.
    """
    scores = []
    if precomputed_alignment is None:
        for d1, d2 in combinations(label_dicts, 2):
            u1, l1 = _collapse_duplicates(d1["orig_ids"], d1[label_key])
            u2, l2 = _collapse_duplicates(d2["orig_ids"], d2[label_key])
            common, idx1, idx2 = np.intersect1d(u1, u2, return_indices=True)
            if len(common) > 1:
                scores.append(adjusted_rand_score(l1[idx1], l2[idx2]))
            else:
                warnings.warn("No common subjects between these two bootstraps; ARI stability contribution skipped.")
        return float(np.mean(scores)) if scores else 0.0

    labels_collapsed = []
    for d in label_dicts:
        _, labs = _collapse_duplicates(d["orig_ids"], d[label_key])
        labels_collapsed.append(np.asarray(labs))

    for b1, b2, idx1, idx2 in precomputed_alignment["pair_indices"]:
        l1 = labels_collapsed[b1][idx1]
        l2 = labels_collapsed[b2][idx2]
        scores.append(adjusted_rand_score(l1, l2))
    return float(np.mean(scores)) if scores else 0.0

def _partition_jaccard_from_labels(labels1, labels2):
    """
    Compute Jaccard index between two partitions defined on the same set of items.
    Partitions are given as integer label vectors of equal length.
    Jaccard is computed over the set of co-clustered pairs.
    """
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)
    n = len(labels1)
    if n < 2:
        return 0.0
    same1 = labels1[:, None] == labels1[None, :]
    same2 = labels2[:, None] == labels2[None, :]
    iu = np.triu_indices(n, k=1)
    a = same1[iu]
    b = same2[iu]
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def jaccard_stability_common_subjects(label_dicts, label_key, precomputed_alignment=None):
    """
    Compute mean pairwise Jaccard stability for label arrays after collapsing duplicates
    within each bootstrap via majority vote, then aligning on common subjects.
    The Jaccard index is computed over co-clustered pairs for each pair of bootstraps.
    """
    scores = []
    if precomputed_alignment is None:
        for d1, d2 in combinations(label_dicts, 2):
            u1, l1 = _collapse_duplicates(d1["orig_ids"], d1[label_key])
            u2, l2 = _collapse_duplicates(d2["orig_ids"], d2[label_key])
            common, idx1, idx2 = np.intersect1d(u1, u2, return_indices=True)
            if len(common) > 1:
                l1_aligned = l1[idx1]
                l2_aligned = l2[idx2]
                scores.append(_partition_jaccard_from_labels(l1_aligned, l2_aligned))
            else:
                warnings.warn("No common subjects between these two bootstraps; Jaccard stability contribution skipped.")
        return float(np.mean(scores)) if scores else 0.0

    labels_collapsed = []
    for d in label_dicts:
        _, labs = _collapse_duplicates(d["orig_ids"], d[label_key])
        labels_collapsed.append(np.asarray(labs))

    for b1, b2, idx1, idx2 in precomputed_alignment["pair_indices"]:
        l1_aligned = labels_collapsed[b1][idx1]
        l2_aligned = labels_collapsed[b2][idx2]
        scores.append(_partition_jaccard_from_labels(l1_aligned, l2_aligned))
    return float(np.mean(scores)) if scores else 0.0


def _composite_quality_score(X, labels):
    """Mirror the ensemble-clustering composite quality score for permutation testing."""
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0 or len(np.unique(labels)) <= 1:
        return 0.0
    try:
        sil = silhouette_score(X, labels, metric='euclidean')
        sil_n = (sil + 1.0) / 2.0
    except Exception:
        sil_n = 0.0
    try:
        Xq = np.asarray(X, dtype=float)
        ch_n = calinski_harabasz_score(Xq, labels)
        ch_n = ch_n / (ch_n + 1.0)
        db_n = 1.0 / (1.0 + davies_bouldin_score(Xq, labels))
    except Exception:
        ch_n = 0.0
        db_n = 0.0
    return float(np.mean([sil_n, ch_n, db_n]))


def _precompute_bootstrap_alignment(label_dicts, label_key='labels'):
    """Collapse duplicate IDs once and cache pairwise subject alignments across bootstraps."""
    collapsed_ids = []
    collapsed_labels = []
    for d in label_dicts:
        ids_u, labs_u = _collapse_duplicates(d["orig_ids"], d[label_key])
        collapsed_ids.append(np.asarray(ids_u))
        collapsed_labels.append(np.asarray(labs_u))

    pair_indices = []
    for b1, b2 in combinations(range(len(label_dicts)), 2):
        common, idx1, idx2 = np.intersect1d(collapsed_ids[b1], collapsed_ids[b2], return_indices=True)
        if len(common) > 1:
            pair_indices.append((b1, b2, idx1, idx2))

    return {
        'collapsed_ids': collapsed_ids,
        'collapsed_labels': collapsed_labels,
        'pair_indices': pair_indices,
    }


def _permutation_pvalue_quality(X, labels, n_permutations=200, seed=42, precomputed=False):
    """Permutation p-value for the composite clustering quality score."""
    if n_permutations <= 0:
        return None
    labels = np.asarray(labels, dtype=int)
    observed = (
        _safe_quality(X, labels, precomputed=True)
        if precomputed else _composite_quality_score(X, labels)
    )
    if labels.size == 0 or len(np.unique(labels)) <= 1:
        return 1.0

    exceed = 0
    for b in range(int(n_permutations)):
        rng = np.random.default_rng(_derive_seed("singleclust_quality_permutation", b, base=seed))
        perm_labels = rng.permutation(labels)
        perm_score = (
            _safe_quality(X, perm_labels, precomputed=True)
            if precomputed else _composite_quality_score(X, perm_labels)
        )
        if perm_score >= observed:
            exceed += 1
    return float((exceed + 1) / (int(n_permutations) + 1))


def _permutation_pvalue_stability_ari(label_dicts, observed_ari=None, label_key='labels', n_permutations=200, seed=42):
    """Permutation p-value for mean pairwise bootstrap ARI, preserving cluster-size marginals."""
    if n_permutations <= 0:
        return None

    alignment = _precompute_bootstrap_alignment(label_dicts, label_key=label_key)
    pair_indices = alignment["pair_indices"]
    if not pair_indices:
        return None

    collapsed_labels = alignment["collapsed_labels"]
    if observed_ari is None:
        observed_scores = [
            adjusted_rand_score(collapsed_labels[b1][idx1], collapsed_labels[b2][idx2])
            for b1, b2, idx1, idx2 in pair_indices
        ]
        observed_ari = float(np.mean(observed_scores)) if observed_scores else 0.0

    exceed = 0
    for b in range(int(n_permutations)):
        rng = np.random.default_rng(_derive_seed("singleclust_stability_ari_permutation", b, base=seed))
        permuted = [rng.permutation(labels) for labels in collapsed_labels]
        perm_scores = [
            adjusted_rand_score(permuted[b1][idx1], permuted[b2][idx2])
            for b1, b2, idx1, idx2 in pair_indices
        ]
        perm_mean = float(np.mean(perm_scores)) if perm_scores else 0.0
        if perm_mean >= observed_ari:
            exceed += 1
    return float((exceed + 1) / (int(n_permutations) + 1))



# --- Fitness computation for gather ---
def _compute_fitness_for_ind(
    i,
    label_dicts,
    objectives,
    cache_dir=None,
    fold_index=None,
    bootstrap_index=None,
    precomputed_alignment=None,
    precomputed_consensus_cache=None,
    requested_k=None,
    ):
    """
    Compute stability and quality
    for candidate i. Used in multi-objective search optimisation.
    """


    # Decide which additional stability flavours to compute based on selected objectives
    objectives_set = set(objectives)
    need_coassoc = any("stability_coassoc" in obj for obj in objectives_set)
    need_ccc = any("stability_CCC" in obj or "stability_ccc" in obj for obj in objectives_set)
    need_jaccard = any("stability_jaccard" in obj for obj in objectives_set)

    # --- Helpers: detect degenerate (no-cluster) solutions ---
    def _n_unique_or_zero(labels):
        """Return number of unique labels; 0 if labels missing/empty."""
        try:
            a = np.asarray(labels)
        except Exception:
            return 0
        return int(len(np.unique(a))) if a.size > 0 else 0

    def _is_degenerate(labels):
        """Degenerate = no meaningful clustering (0 or 1 unique label)."""
        return _n_unique_or_zero(labels) <= 1

    observed_cluster_counts = [
        _n_unique_or_zero(d.get("labels", [])[i])
        for d in label_dicts
    ]
    degenerate_flags = [count <= 1 for count in observed_cluster_counts]
    degenerate_fraction = float(np.mean(degenerate_flags)) if degenerate_flags else 1.0
    effective_k_summary = summarize_effective_k(observed_cluster_counts, requested_k=requested_k)
    effective_k = effective_k_summary["selected_k"]


    boot_dicts = [
        {"orig_ids": d["orig_ids"], "labels": d["labels"][i]}
        for d in label_dicts
    ]
    # Final-cluster stability
    # ARI is always computed as the primary stability metric
    stab_ari = ari_stability_common_subjects(
        boot_dicts,
        label_key="labels",
        precomputed_alignment=precomputed_alignment
    )

    # Additional stability metrics
    #final_stab_coassoc, final_stab_CCC = coassociation_stability(boot_dicts_final, label_key="labels")
    stab_jaccard = jaccard_stability_common_subjects(
        boot_dicts,
        label_key="labels",
        precomputed_alignment=precomputed_alignment
    )
    stab_SUM_MAT = consensus_pac_ccc(
        boot_dicts,
        label_key="labels",
        return_consensus=False,
        return_ecdf=False,
        precomputed_cache=precomputed_consensus_cache,
    )


    # ARI/Jaccard can regard repeated one-cluster solutions as perfectly stable.
    # Penalize in proportion to their frequency instead of zeroing an otherwise
    # valid candidate because one of many bootstraps happened to collapse.
    nondegenerate_fraction = 1.0 - degenerate_fraction
    stab_ari *= nondegenerate_fraction
    stab_jaccard *= nondegenerate_fraction


    # --- Quality ---
    has_final_q = all("scores" in d for d in label_dicts)

    mean_q = float(np.mean([float(d["scores"][i]) for d in label_dicts])) if has_final_q else 0.0
    fitness_record = {
        "ind_id": i,
        "stab_ari": stab_ari,
        #"final_stab_coassoc": final_stab_coassoc,
        #"final_stab_CCC": final_stab_CCC,
        "stab_jaccard": stab_jaccard,
        "stab_SUM_MAT": stab_SUM_MAT,
        "quality": mean_q,
        "degenerate_fraction": degenerate_fraction,
        "effective_k_summary": effective_k_summary,
    }
    if cache_dir:
        if fold_index is not None and bootstrap_index is not None:
            fname = f"fitness_{fold_index}_{bootstrap_index}_{i}.pkl"
        elif fold_index is not None:
            fname = f"fitness_{fold_index}_{i}.pkl"
        else:
            fname = f"fitness_{i}.pkl"
        save_pickle(os.path.join(cache_dir, fname), fitness_record)


    # Build summary dict and return requested objectives
    summary = {
        # ARI-based stability
        "stab_ari": float(stab_ari),
        # Quality metrics
        "quality": float(mean_q),

        # Additional stability flavours for reporting
        #"final_stability_coassoc": float(final_stab_coassoc),
        #"final_stability_CCC": float(final_stab_CCC),
        "stab_jaccard": float(stab_jaccard),
        "stab_SUM_MAT": {
            "PAC": stab_SUM_MAT.get("PAC", np.nan),
            "CCC": stab_SUM_MAT.get("CCC", np.nan),
            "meta": stab_SUM_MAT.get("meta", {}),
        },
        "effective_k": effective_k,
        "effective_k_summary": effective_k_summary,
        "observed_cluster_counts": observed_cluster_counts,
        "degenerate_fraction": degenerate_fraction,
    }
    values = tuple(summary[obj] for obj in objectives)
    return values, summary



# Modes

def do_bootstrap(args):
    """
    Run one bootstrap iteration for grid-search candidate stability evaluation.
    """

    base_dir = os.path.abspath(getattr(args, "base_dir", "."))
    if args.fold_index is None:
        raise ValueError("For bootstrap mode, --fold_index must be specified")
    search_root = _search_root(base_dir, args.fold_index)
    population_file = _resolve_path(base_dir, args.population_file)
    population_initial_file = _resolve_path(base_dir, args.population_initial_file)
    output_labels_path = _resolve_path(base_dir, args.output_labels)
    if population_initial_file is None:
        population_initial_file = os.path.join(search_root, f"population_init_fold{args.fold_index}.pkl")
    if output_labels_path is None:
        boot_dir = os.path.join(search_root, f"bootstrap_{args.bootstrap_index or 0}")
        output_labels_path = os.path.join(boot_dir, f"labels_{args.bootstrap_index or 0}.pkl")

    # Deterministic seed namespace for this search bootstrap.
    boot_index = getattr(args, "bootstrap_index", 0)
    if boot_index is None:
        boot_index = 0
    seed = _search_bootstrap_seed(args.fold_index, boot_index)
    _seed_everything(seed)


    # Ensure DEAP’s creator classes exist before unpickling the candidate set
    if args.optimisation == 'multi':
        _ensure_multi_fitness_class(args)
    else:
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    # Load data and split according to outer CV
    df = pd.read_csv(args.input_csv)
    meta = pd.read_csv(args.meta_csv)
    if args.n_folds == 1:
        # No CV split: use all rows for training to allow fast synthetic-data tests
        train_df = df.reset_index(drop=True)
    else:
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        train_idx, _ = list(kf.split(df))[args.fold_index]
        train_df = df.iloc[train_idx].reset_index(drop=True)



    # Load candidate set (single-pass grid search): use scored candidates if present, otherwise the initial grid
    candidate_file = population_file if (population_file and os.path.exists(population_file)) else population_initial_file
    if not candidate_file:
        raise FileNotFoundError("No candidate file found for bootstrap mode.")
    with open(candidate_file, 'rb') as f:
        candidate_set = dill.load(f)

    # Normalize fitness container type across all candidates to match optimisation mode
    if args.optimisation == 'multi':
        multi_cls = _get_multi_fitness_class(args)
        if multi_cls is None:
            multi_cls = _ensure_multi_fitness_class(args)
        for candidate in candidate_set:
            candidate.fitness = multi_cls()
    else:
        for candidate in candidate_set:
            candidate.fitness = creator.FitnessMax()

    # --- Bootstrap / Subsampling selection ---
    mode = getattr(args, "bootstrap_mode", "bootstrap").lower()
    if mode not in ["bootstrap", "subsample"]:
        raise ValueError(f"Invalid --bootstrap_mode '{mode}'. Must be 'bootstrap' or 'subsample'.")

    if mode == "bootstrap":
        print(f"[Fold {args.fold_index}] Running standard bootstrap (with replacement).")
        bdf = train_df.sample(n=len(train_df), replace=True, random_state=seed).reset_index(drop=True)
        bdf = bdf.reset_index(drop=True)
    elif mode == "subsample":
        print(f"[Fold {args.fold_index}] Running subsampling mode (without replacement, 80% sample).")
        # Draw 80% of subjects without replacement, avoid degenerate samples
        frac = 0.8
        for attempt in range(100):
            attempt_seed = _derive_seed("singleclust_search_subsample_attempt", seed, attempt)
            bdf = train_df.sample(frac=frac, replace=False, random_state=attempt_seed).reset_index(drop=True)
            bdf = bdf.reset_index(drop=True)
            if len(bdf.drop_duplicates(subset=args.subject_id_column)) >= 3:
                break
        else:
            raise RuntimeError(f"Failed to create a valid subsample after 100 attempts.")
    bdf["orig_subject_id"] = bdf[args.subject_id_column]
    # Assign a unique processed ID to each row for alignment
    bdf["proc_subject_id"] = np.arange(len(bdf))

    # Preprocess and train AE once on the bootstrap sample
    try:
        print("Start running Preprocessing")
        t_prep_start = time.time()
        ae_data, subject_id_list, df_final = preprocessing(
            bdf, meta,
            subject_id_column='proc_subject_id',
            col_threshold=args.col_threshold,
            row_threshold=args.row_threshold,
            skew_threshold=args.skew_threshold,
            scaler_type=args.scaler_type,
            modalities=args.modalities,
            dummy_code_modalities=args.dummy_code_modalities,
            mixed_categorical_modalities=args.mixed_categorical_modalities,
        )

        # Save subject IDs in the order or the processed data
        kept_proc_ids = subject_id_list[0] if (subject_id_list and isinstance(subject_id_list[0], (list, tuple, np.ndarray))) else subject_id_list  # same order as final embeddings/labels
        # Map proc -> orig using bdf
        proc_to_orig = dict(zip(bdf["proc_subject_id"], bdf["orig_subject_id"]))
        kept_orig_ids = [proc_to_orig[p] for p in kept_proc_ids]


        t_prep_end = time.time()
        print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] Preprocessing took {t_prep_end - t_prep_start:.2f}s")

        dim = None if args.dim_reduction is None else str(args.dim_reduction).lower()
        if dim in (None, 'none'):
            print("Skipping VAE and using preprocessed features as latent representations.")
        elif dim == 'vae':
            print("Start running VAE")
        elif dim == 'sparsevae':
            print(f"Start running SparseVAE (latent L1={args.sparse_l1_lambda})")
        elif dim == 'ae':
            print("Start running AE")
        elif dim == 'sparseae':
            print(f"Start running SparseAE (latent L1={args.sparse_l1_lambda})")
        elif dim == 'pca':
            print("Start running PCA")
        elif dim == 'sparsepca':
            print("Start running SparsePCA")
        elif dim == 'sparsenmf':
            print("Start running SparseNMF")

        t_ae_start = time.time()
        ae_res, data_list = _build_latent_matrix(
            args,
            ae_data,
            df_final,
            seed_value=seed,
            subject_id_column='proc_subject_id',
        )
        t_ae_end = time.time()

        if dim == 'vae':
            print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] VAE nested CV took {t_ae_end - t_ae_start:.2f}s")
        elif dim == 'sparsevae':
            print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] SparseVAE nested CV took {t_ae_end - t_ae_start:.2f}s")
        elif dim == 'ae':
            print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] AE nested CV took {t_ae_end - t_ae_start:.2f}s")
        elif dim == 'sparseae':
            print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] SparseAE nested CV took {t_ae_end - t_ae_start:.2f}s")
        elif dim == 'pca':
            print("PCA dimensionality reduction completed.")
        elif dim == 'sparsepca':
            print("SparsePCA dimensionality reduction completed.")
        elif dim == 'sparsenmf':
            print("SparseNMF dimensionality reduction completed.")

        # Free large dimensionality-reduction results to reduce peak memory before clustering
        del ae_res
        gc.collect()

        reference_n = int(len(df))
        current_n = int(len(kept_orig_ids))
        args.mincluster_n_applied = _operational_min_cluster_n(args, current_n, reference_n)

        print("Start running ensemble clustering on bootstrap sample...")
        # Seed RNGs before clustering
        _seed_everything(seed)
        # Check candidate gene names
        #print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] Sample gene_names from first 3 individuals: {[ind.gene_names for ind in population[:3]]}")
        t_clust_start = time.time()
        # Evaluate labels for each candidate in parallel using requested n_jobs (SLURM_CPUS_PER_TASK) if provided, otherwise all CPUs
        n_workers = args.n_jobs or (os.cpu_count() or 1)
        #print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] using {n_workers} workers for clustering")

        chunksize = max(1, len(candidate_set) // (n_workers * 4) if n_workers else len(candidate_set))
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(data_list, subject_id_list, args)
        ) as executor:
            all_results = list(executor.map(partial(_cluster_candidate, args=args), candidate_set, chunksize=chunksize))

        t_clust_end = time.time()
        print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] "
              f"Clustering {len(candidate_set)} candidates took {t_clust_end - t_clust_start:.2f}s")

        # Ensure lengths of IDs and labels align
        if len(all_results) > 0:
            assert len(kept_orig_ids) == len(all_results[0][0]), "orig_ids and final_labels length mismatch"
        # Prepare output dict (one labels array and one quality score per candidate)
        labels = [res[0] for res in all_results]
        # Prepare output dict for quality scores (mean over views already in res[2])
        scores = [res[1] for res in all_results]

        to_dump = {
            "orig_ids":        kept_orig_ids,
            "labels":    labels,
            "scores":    scores,
            "requested_params": [decode_search_candidate(1, cand) for cand in candidate_set],
            "mincluster_n_requested": int(args.mincluster_n),
            "mincluster_n_applied": int(args.mincluster_n_applied),
            "reference_n": reference_n,
            "current_n": current_n,
        }

        # Save
        os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)
        with open(output_labels_path, 'wb') as f:
            dill.dump(to_dump, f)
        print(f"Bootstrap labels {args.bootstrap_index} saved to {output_labels_path}")
        return
    except Exception as e:
        # Preserve a sentinel for diagnostics, but fail the task so Slurm dependencies
        # cannot treat an unusable bootstrap as successful.
        try:
            traceback.print_exc()
            os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)
            sentinel = {
                "orig_ids": [],
                "labels": [],
                "scores": []
            }
            with open(output_labels_path, 'wb') as f:
                dill.dump(sentinel, f)
            print(f"[Fold {args.fold_index}] Bootstrap {args.bootstrap_index} marked as SKIPPED due to error: {e}")
        except Exception as ee:
            print(f"[Fold {args.fold_index}] Failed to write sentinel for bootstrap {args.bootstrap_index}: {ee}")
        raise


def do_gather(args):
    """
    Gather bootstrap outputs and score the fixed grid-search candidates (single pass).
    """
    base_dir = os.path.abspath(getattr(args, "base_dir", "."))
    if args.fold_index is None:
        raise ValueError("For gather mode, --fold_index must be specified")
    search_root = _search_root(base_dir, args.fold_index)
    bootstrap_dir = _resolve_path(base_dir, args.bootstrap_dir)
    population_dir = _resolve_path(base_dir, args.population_dir) if args.population_dir else None
    population_file = _resolve_path(base_dir, args.population_file) if args.population_file else None
    population_initial_file = _resolve_path(base_dir, args.population_initial_file) if args.population_initial_file else None
    output_population = _resolve_path(base_dir, args.output_population) if args.output_population else None
    if population_dir is None:
        population_dir = search_root
    if population_initial_file is None:
        population_initial_file = os.path.join(search_root, f"population_init_fold{args.fold_index}.pkl")
    if output_population is None:
        output_population = os.path.join(search_root, f"candidates_scored_fold{args.fold_index}.pkl")

    if args.fold_index is None:
        raise ValueError("For gather mode, --fold_index must be specified")
    if not bootstrap_dir:
        raise ValueError("For gather mode, --bootstrap_dir must be specified")
    if not population_dir:
        raise ValueError("For gather mode, --population_dir must be specified")

    # Deterministic seed namespace for this gather/scoring step.
    boot_index = getattr(args, "bootstrap_index", 0)
    if boot_index is None:
        boot_index = 0
    seed = _derive_seed("singleclust_gather", int(args.fold_index or 0))
    _seed_everything(seed)

    hof_dir = search_root
    fitness_cache_dir = os.path.join(hof_dir, "fitness_cache", "single_pass")

    # Ensure fitness class exists BEFORE unpickling populations/HOF created in prior runs
    if args.optimisation == 'multi':
        _ensure_multi_fitness_class(args)
    if args.optimisation == 'single' and not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # Find and load all bootstrap label dicts under the specified directory
    pattern = os.path.join(bootstrap_dir, 'bootstrap_*', 'labels_*.pkl')

    import re

    def numeric_boot_dirs(path):
        """Handle numeric boot dirs."""
        m = re.search(r'bootstrap_(\d+)', path)
        return int(m.group(1)) if m else -1

    discovered_files = sorted(glob.glob(pattern), key=numeric_boot_dirs)

    expected_indices = set(range(1, int(args.n_bootstrap) + 1))
    files_by_index = {}
    for path in discovered_files:
        index = numeric_boot_dirs(path)
        if index in expected_indices:
            if index in files_by_index:
                raise RuntimeError(f"Multiple label files found for bootstrap {index} in {bootstrap_dir}.")
            files_by_index[index] = path
    files = [files_by_index[index] for index in sorted(files_by_index)]

    if not files:
        raise FileNotFoundError(f"No label files found in {bootstrap_dir}; expected bootstrap_*/labels_*.pkl")

    # Each file contains a dict with keys "labels",  "scores". The scores are the qualities.
    label_dicts_all = []
    for filename in files:
        with open(filename, 'rb') as handle:
            label_dicts_all.append(dill.load(handle))
    def _usable(d):
        """Handle usable."""
        return isinstance(d, dict) and len(d.get("labels", [])) > 0 and len(d.get("orig_ids", [])) > 0
    label_dicts = [d for d in label_dicts_all if _usable(d)]

    min_needed = max(1, args.n_bootstrap - args.max_missing_bootstraps)
    if len(label_dicts) < min_needed:
        raise RuntimeError(f"Only {len(label_dicts)} usable bootstraps (min required {min_needed}). Check for sentinel/failed runs in {bootstrap_dir}.")

    # Precompute caches to avoid repeated intersect1d and consensus ID mapping
    precomputed_alignment = precompute_bootstrap_pair_alignment(label_dicts)
    precomputed_consensus_cache = precompute_consensus_cache(label_dicts)

    # Unpack lists across bootstraps
    # final_label_sets: list of lists, one per bootstrap, each list of length pop_size
    label_sets = [d["labels"] for d in label_dicts]
    candidate_counts = {len(labels) for labels in label_sets}
    if len(candidate_counts) != 1:
        raise RuntimeError(
            f"Bootstrap candidate counts are inconsistent: {sorted(candidate_counts)}. "
            "Remove stale outputs and rerun the affected bootstraps."
        )
    # Evaluate search objectives for each candidate using ProcessPoolExecutor
    pop_size = len(label_sets[0])
    requested_params = label_dicts[0].get("requested_params", [])
    requested_ks = [
        params.get("k") if isinstance(params, dict) else None
        for params in requested_params
    ]
    if len(requested_ks) != pop_size:
        requested_ks = [None] * pop_size
    n_workers = args.n_jobs or (os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(
            _compute_fitness_for_ind,
            range(pop_size),
            repeat(label_dicts),
            repeat(tuple(args.ga_objectives)),
            repeat(fitness_cache_dir),
            repeat(args.fold_index),
            repeat(boot_index),
            repeat(precomputed_alignment),
            repeat(precomputed_consensus_cache),
            requested_ks,
        ))
        fitness = [tuple(map(float, res[0])) for res in results]
        summary_metrics = [res[1] for res in results]

    # Save fitness history for this fold
    os.makedirs(hof_dir, exist_ok=True)
    history_file = os.path.join(hof_dir, "fitness_history.pkl")
    # Ensure the directory exists before writing the history file
    os.makedirs(hof_dir, exist_ok=True)
    if os.path.exists(history_file):
        with open(history_file, "rb") as hf:
            fitness_history = pickle.load(hf)
    else:
        fitness_history = {}
    # Record the single search pass fitness tuples
    fitness_history['single_pass'] = fitness
    with open(history_file, "wb") as hf:
        pickle.dump(fitness_history, hf)

    # Load fixed grid-search candidates (single gather pass)
    candidate_file = population_file if (population_file and os.path.exists(population_file)) else population_initial_file
    if not candidate_file:
        raise ValueError("For gather mode, no candidate file is available")
    with open(candidate_file, 'rb') as f:
        candidate_set = dill.load(f)
    if len(candidate_set) != pop_size:
        raise RuntimeError(
            f"Candidate-set size ({len(candidate_set)}) does not match bootstrap outputs ({pop_size}). "
            "Remove stale outputs and rerun the search."
        )

    # Re-attach canonical metadata in case older pickles are missing or had stale gene names.
    gene_names = _search_gene_names(args)
    for candidate in candidate_set:
        candidate.gene_names = gene_names

    # Ensure DEAP’s creator classes exist before unpickling the candidate set
    if args.optimisation == 'multi':
        multi_cls = _get_multi_fitness_class(args)
        if multi_cls is None:
            multi_cls = _ensure_multi_fitness_class(args)
    else:
        multi_cls = None
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    # Reset fitness container type across all individuals to match optimisation mode
    if args.optimisation == 'multi':
        for candidate in candidate_set:
            if not isinstance(candidate.fitness, multi_cls):
                candidate.fitness = multi_cls()
    else:
        for candidate in candidate_set:
            candidate.fitness = creator.FitnessMax()

    # Clear any stale fitness values to avoid mismatches
    for candidate in candidate_set:
        try:
            del candidate.fitness.values
        except AttributeError:
            pass


    # Assign fitness to each candidate based on optimisation mode
    if args.optimisation == 'single':
        # Single-objective: fitness tuples contain only final stability
        stab_key, qual_key = _primary_metric_keys(args)
        for idx, (stab,) in enumerate(fitness):
            candidate = candidate_set[idx]
            candidate.fitness.values = (stab,)
            summary = summary_metrics[idx]
            candidate.metrics_summary = summary
            candidate.stab = summary.get(stab_key)
            candidate.qual = summary.get(qual_key)
    elif args.optimisation == 'multi':
        stab_key, qual_key = _primary_metric_keys(args)
        for idx, candidate in enumerate(candidate_set):
            fitvals = tuple(map(float, fitness[idx]))
            assert len(fitvals) == len(candidate.fitness.weights), (
                f"Mismatch: {len(fitvals)} values vs {len(candidate.fitness.weights)} weights"
            )
            candidate.fitness.values = fitvals
            summary = summary_metrics[idx]
            candidate.metrics_summary = summary
            candidate.stab = summary.get(stab_key)
            candidate.qual = summary.get(qual_key)
    else:
        raise ValueError(f"Unknown optimisation mode: {args.optimisation}")


    # ---------------- Search summary + persistence (single pass; no evolution) ----------------
    # Record statistics over the evaluated candidate set
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    if args.optimisation == 'multi':
        for idx, name in enumerate(args.ga_objectives):
            stats.register(
                f"avg_{name}",
                lambda vals, idx=idx: float(np.mean([v[idx] for v in vals]))
            )
            stats.register(
                f"max_{name}",
                lambda vals, idx=idx: float(np.max([v[idx] for v in vals]))
            )
    else:
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)


    # ——— Persistent Hall-of-Fame for this fold search pass ———
    hof_path = os.path.join(hof_dir, "halloffame.pkl")
    # Singleclust is a complete single-pass grid search. Rebuild the front from
    # this run so previous runs cannot influence candidate selection.
    hall_of_fame = tools.ParetoFront() if args.optimisation == 'multi' else tools.HallOfFame(maxsize=1)

    # Update and persist Hall of Fame / Pareto front with the evaluated candidate set
    hall_of_fame.update(candidate_set)
    with open(hof_path, 'wb') as f:
        dill.dump(hall_of_fame, f)

    # --- Validation: enforce consistent fitness tuple lengths before persistence ---
    if args.optimisation == 'multi':
        expected_len = len(args.ga_objectives)
        if len(candidate_set) != len(fitness):
            raise RuntimeError(f"Candidate-set size ({len(candidate_set)}) != fitness list size ({len(fitness)}).")
        for idx, candidate in enumerate(candidate_set):
            vals = getattr(candidate.fitness, 'values', ())
            if len(vals) != expected_len:
                try:
                    fitvals = fitness[idx]
                except Exception:
                    raise RuntimeError(f"Missing or invalid fitness for candidate {idx}: {fitness[idx] if idx < len(fitness) else 'N/A'}")
                candidate.fitness.values = tuple(map(float, fitvals))
    else:
        # Single-objective must have length-1 tuples
        if len(candidate_set) != len(fitness):
            raise RuntimeError(f"Candidate-set size ({len(candidate_set)}) != fitness list size ({len(fitness)}).")
        for idx, candidate in enumerate(candidate_set):
            vals = getattr(candidate.fitness, 'values', ())
            if len(vals) != 1:
                try:
                    (fs,) = fitness[idx]
                except Exception:
                    raise RuntimeError(f"Missing or invalid single fitness for candidate {idx}: {fitness[idx] if idx < len(fitness) else 'N/A'}")
                candidate.fitness.values = (float(fs),)

    # Grid search: no breeding/mutation. Persist the same candidate list for compatibility
    # Scored candidates are persisted for downstream outer-fold selection.
    next_population = list(candidate_set)

    # Also preserve gene_names on HOF individuals
    try:
        for ind in hall_of_fame:
            ind.gene_names = gene_names
    except Exception:
        pass

    # Save the scored candidate set
    out_path = output_population
    if not population_dir:
        raise ValueError("For gather mode, --population_dir must be specified")
    if not out_path:
        raise ValueError("For gather mode, --output_population must be specified")
    os.makedirs(population_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, 'wb') as f:
        dill.dump(next_population, f)
    print(f"Saved grid-search scored candidate set ({len(next_population)} candidates; no evolution) to {out_path}")
    return

def _activation_function_map(names):
    """Handle activation function map."""
    available = {
        "ReLU": nn.ReLU(),
        "LeakyReLU": nn.LeakyReLU(),
        "selu": nn.SELU(),
        "swish": nn.SiLU(),
    }
    return {name: available[name] for name in names if name in available}


def _as_autoencoder_modalities(ae_data, df_final, subject_col):
    """Adapt singleclust's one-matrix preprocessing output to AE/VAE helpers."""
    if isinstance(ae_data, dict):
        return ae_data
    if isinstance(ae_data, (list, tuple)) and len(ae_data) == 1:
        matrix = ae_data[0]
    else:
        matrix = ae_data

    if subject_col in df_final.columns:
        ids = df_final[subject_col].tolist()
    else:
        ids = list(range(np.asarray(matrix).shape[0]))
    return {"singleclust": (ids, matrix)}


def _autoencoder_final_latent(ae_res):
    """Return the latent matrix from either flat or modality-keyed AE/VAE output."""
    if isinstance(ae_res, dict) and "final_latent" in ae_res:
        return ae_res["final_latent"]
    if isinstance(ae_res, dict) and len(ae_res) == 1:
        only_value = next(iter(ae_res.values()))
        if isinstance(only_value, dict) and "final_latent" in only_value:
            return only_value["final_latent"]
    raise KeyError("AE/VAE result did not contain a final_latent matrix")


def _build_latent_matrix(args, ae_data, df_final, seed_value, subject_id_column=None):
    """Fit the configured dimensionality reduction on one preprocessed feature matrix."""
    subject_col = subject_id_column or args.subject_id_column
    dim = None if args.dim_reduction is None else str(args.dim_reduction).lower()
    if dim in (None, 'none'):
        X = df_final.drop(columns=[subject_col], errors='ignore')
        X = X.to_numpy(dtype=np.float32, copy=True)
        return {"final_latent": X}, np.asarray(X, dtype=np.float32, copy=False)

    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)

    activation_map = _activation_function_map(args.activation_functions)
    if dim == 'vae':
        autoencoder_modalities = _as_autoencoder_modalities(ae_data, df_final, subject_col)
        ae_res = run_VAE_complete(
            autoencoder_modalities,
            hidden_dims=args.hidden_dims,
            activation_functions=activation_map,
            learning_rates=args.learning_rates,
            batch_sizes=args.batch_sizes,
            latent_dims=args.latent_dims,
            l1_reg=0.0,
        )
        X = np.asarray(_autoencoder_final_latent(ae_res), dtype=np.float32, copy=False)
        return ae_res, X
    if dim == 'sparsevae':
        autoencoder_modalities = _as_autoencoder_modalities(ae_data, df_final, subject_col)
        ae_res = run_VAE_complete(
            autoencoder_modalities,
            hidden_dims=args.hidden_dims,
            activation_functions=activation_map,
            learning_rates=args.learning_rates,
            batch_sizes=args.batch_sizes,
            latent_dims=args.latent_dims,
            l1_reg=args.sparse_l1_lambda,
        )
        X = np.asarray(_autoencoder_final_latent(ae_res), dtype=np.float32, copy=False)
        return ae_res, X
    if dim in {'ae', 'autoencoder'}:
        autoencoder_modalities = _as_autoencoder_modalities(ae_data, df_final, subject_col)
        ae_res = run_AE_complete(
            autoencoder_modalities,
            hidden_dims=args.hidden_dims,
            activation_functions=activation_map,
            learning_rates=args.learning_rates,
            batch_sizes=args.batch_sizes,
            latent_dims=args.latent_dims,
            l1_reg=0.0,
        )
        X = np.asarray(_autoencoder_final_latent(ae_res), dtype=np.float32, copy=False)
        return ae_res, X
    if dim == 'sparseae':
        autoencoder_modalities = _as_autoencoder_modalities(ae_data, df_final, subject_col)
        ae_res = run_AE_complete(
            autoencoder_modalities,
            hidden_dims=args.hidden_dims,
            activation_functions=activation_map,
            learning_rates=args.learning_rates,
            batch_sizes=args.batch_sizes,
            latent_dims=args.latent_dims,
            l1_reg=args.sparse_l1_lambda,
        )
        X = np.asarray(_autoencoder_final_latent(ae_res), dtype=np.float32, copy=False)
        return ae_res, X
    if dim == 'pca':
        X_df = df_final.drop(columns=[subject_col], errors='ignore')
        pca = PCA(
            n_components=_dimred_n_components(args, X_df.shape[0], X_df.shape[1]),
            random_state=seed_value,
        )
        X = pca.fit_transform(X_df.to_numpy(dtype=np.float32, copy=True))
        ae_res = {'final_latent': np.asarray(X, dtype=np.float32, copy=False), 'pca_model': pca}
        return ae_res, ae_res['final_latent']
    if dim == 'sparsepca':
        X_df = df_final.drop(columns=[subject_col], errors='ignore')
        X, spca = _run_sparse_pca(X_df, args, seed_value)
        ae_res = {'final_latent': X, 'spca_model': spca}
        return ae_res, ae_res['final_latent']
    if dim == 'sparsenmf':
        X_df = df_final.drop(columns=[subject_col], errors='ignore')
        X, snmf = _run_sparse_nmf(X_df, args, seed_value)
        ae_res = {'final_latent': X, 'snmf_model': snmf}
        return ae_res, ae_res['final_latent']
    raise ValueError(f"Unknown dim_reduction method: {args.dim_reduction}")


def _simpleclust_svm_training_matrix(args, ae_res, df_final, preprocessing_artifact=None):
    """Return the final SVM matrix in the same representation used for clustering."""
    dim = None if args.dim_reduction is None else str(args.dim_reduction).lower()
    if dim in (None, 'none'):
        X_train = df_final.drop(columns=[args.subject_id_column], errors='ignore').reset_index(drop=True)
        return X_train, 'preprocessed_features'

    latent = np.asarray(_autoencoder_final_latent(ae_res), dtype=np.float32)
    if latent.ndim == 1:
        latent = latent.reshape(-1, 1)

    modality_prefix = 'singleclust'
    if isinstance(preprocessing_artifact, dict):
        preprocessing_details = preprocessing_artifact.get('preprocessing_details') or {}
        modalities = preprocessing_details.get('modalities_in_output') or preprocessing_artifact.get('modalities')
        if isinstance(modalities, (list, tuple)) and len(modalities) == 1:
            modality_prefix = str(modalities[0])

    X_train = pd.DataFrame(
        latent,
        columns=[f"{modality_prefix}__latent_{i + 1}" for i in range(latent.shape[1])],
    )
    return X_train, 'dimensionality_reduced_features'


def _sorted_hof_candidates(candidates, optimisation):
    """Handle sorted hof candidates."""
    candidates = list(candidates)
    if optimisation != 'multi' or len(candidates) <= 1:
        return candidates
    vals = np.array([ind.fitness.values for ind in candidates], dtype=float)
    mins, maxs = vals.min(axis=0), vals.max(axis=0)
    rng = np.where(maxs > mins, maxs - mins, 1.0)
    norm = (vals - mins) / rng
    d = np.linalg.norm(1.0 - norm, axis=1)
    return [candidates[i] for i in np.argsort(d)]


def do_outer(args):
    """Select the best hyperparameters for one outer fold and fit one clustering on the train split."""
    base_dir = os.path.abspath(getattr(args, "base_dir", "."))
    population_file = _resolve_path(base_dir, args.population_file)
    output_metrics_path = _resolve_path(base_dir, args.output_metrics) if args.output_metrics else None
    if not population_file:
        raise ValueError("For outer mode, --population_file must be specified")
    if not output_metrics_path:
        raise ValueError("For outer mode, --output_metrics must be specified")

    np.random.seed(1000 * (args.fold_index or 0) + 17)
    random.seed(1000 * (args.fold_index or 0) + 17)
    torch.manual_seed(1000 * (args.fold_index or 0) + 17)

    df = pd.read_csv(args.input_csv)
    meta = pd.read_csv(args.meta_csv)
    if args.n_folds == 1:
        train_idx = np.arange(len(df))
        test_idx = np.array([], dtype=int)
    else:
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        train_idx, test_idx = list(kf.split(df))[args.fold_index]
    train_df = df.iloc[train_idx].reset_index(drop=True)

    ae_data, subject_id_list, df_final, preprocessing_artifact = preprocessing(
        train_df, meta,
        subject_id_column=args.subject_id_column,
        col_threshold=args.col_threshold,
        row_threshold=args.row_threshold,
        skew_threshold=args.skew_threshold,
        scaler_type=args.scaler_type,
        modalities=args.modalities,
        dummy_code_modalities=args.dummy_code_modalities,
        mixed_categorical_modalities=args.mixed_categorical_modalities,
        return_artifact=True,
    )
    ae_res, X_latent = _build_latent_matrix(args, ae_data, df_final, seed_value=42)
    if args.optimisation == 'multi':
        _ensure_multi_fitness_class(args)
    elif not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    search_root = os.path.dirname(population_file)
    hof_path = os.path.join(search_root, 'halloffame.pkl')
    with open(hof_path, 'rb') as f:
        hall_of_fame = dill.load(f)
    candidates = _sorted_hof_candidates(hall_of_fame, args.optimisation)
    if not candidates:
        raise RuntimeError(f"No candidates found in hall of fame: {hof_path}")

    chosen = None
    best_params = None
    train_labels = None
    train_quality = None
    requested_k = None
    attempted_candidates = []
    fold_mincluster_n = _operational_min_cluster_n(args, len(X_latent), len(df))
    for rank, candidate in enumerate(candidates):
        candidate_params = decode_search_candidate(1, candidate)
        attempt = {
            'rank': rank + 1,
            'params': candidate_params,
            'n_clusters': None,
            'quality': None,
            'accepted': False,
            'rejection_reason': None,
        }
        try:
            candidate_labels, candidate_quality = run_ensemble_clustering(
                X_latent,
                **candidate_params,
                subject_id_list=subject_id_list,
                inner_jobs=1,
                pre_inner_jobs=1,
                mincluster=args.mincluster,
                mincluster_n=fold_mincluster_n,
                internal_ensemble_enabled=args.internal_ensemble_enabled,
                internal_ensemble_bcs=args.internal_ensemble_bcs,
                internal_ensemble_sample_frac=args.internal_ensemble_sample_frac,
                internal_ensemble_feature_frac=args.internal_ensemble_feature_frac,
                internal_ensemble_seed=int(getattr(args, "fold_index", 0) or 0),
            )
            n_clusters = int(len(np.unique(candidate_labels)))
            attempt['n_clusters'] = n_clusters
            attempt['quality'] = float(candidate_quality) if np.isfinite(candidate_quality) else None
            if n_clusters < 2:
                attempt['rejection_reason'] = 'fewer_than_two_clusters'
            elif not np.isfinite(candidate_quality):
                attempt['rejection_reason'] = 'nonfinite_quality'
            elif candidate_quality <= 0:
                attempt['rejection_reason'] = 'nonpositive_quality'
        except Exception as exc:
            attempt['rejection_reason'] = f'clustering_error:{type(exc).__name__}'
            attempt['error'] = str(exc)
            attempted_candidates.append(attempt)
            warnings.warn(
                f"[Fold {args.fold_index}] Candidate rank {rank + 1} failed during outer selection: {exc}"
            )
            continue

        attempted_candidates.append(attempt)
        if attempt['rejection_reason'] is None:
            attempt['accepted'] = True
            chosen = candidate
            best_params = candidate_params
            train_labels = candidate_labels
            train_quality = candidate_quality
            requested_k = int(candidate_params['k'])
            if rank > 0:
                print(f"[Fold {args.fold_index}] Selected candidate rank {rank} after rejecting degenerate candidates.")
            break

    if chosen is None:
        skip_reason = 'no_candidate_with_at_least_two_clusters_and_positive_quality'
        metrics = {
            'metrics_schema_version': METRICS_SCHEMA_VERSION,
            'fold_status': 'skipped',
            'skip_reason': skip_reason,
            'dim_reduction': args.dim_reduction,
            'dim_reduction_label': _dimred_run_label(args),
            'dim_reduction_n_components': int(args.maxPC) if str(args.dim_reduction).lower() in {'pca', 'sparsepca', 'sparsenmf'} else None,
            'dim_reduction_sparse_l1': _dimred_sparse_l1(args),
            'data': df_final,
            'preprocessing': preprocessing_artifact,
            'preprocessing_details': preprocessing_artifact.get('preprocessing_details'),
            'ae_res': ae_res,
            'train_labels': None,
            'train_quality': None,
            'best_params': None,
            'best_params_requested': None,
            'best_params_effective': None,
            'requested_k': None,
            'train_effective_k': 0,
            'effective_k_summary': None,
            'effective_k_fallback_reason': None,
            'mincluster_n_requested': int(args.mincluster_n),
            'mincluster_n_applied': int(fold_mincluster_n),
            'reference_n': int(len(df)),
            'current_n': int(len(X_latent)),
            'best_fitness': {
                'stability': None,
                'quality': None,
                'metrics_summary': {},
                'fitness_values': (),
            },
            'candidate_attempts': attempted_candidates,
            'train_ids': df.iloc[train_idx][args.subject_id_column].tolist(),
            'test_ids': df.iloc[test_idx][args.subject_id_column].tolist() if len(test_idx) else [],
        }
        os.makedirs(os.path.dirname(output_metrics_path) or '.', exist_ok=True)
        with open(output_metrics_path, 'wb') as f:
            dill.dump(metrics, f)
        print(
            f"[Fold {args.fold_index}] Skipping fold: no candidate produced at least two "
            f"clusters with positive quality. Metrics saved to {output_metrics_path}"
        )
        return

    summary = getattr(chosen, 'metrics_summary', {}) or {}
    effective_k_summary = summary.get('effective_k_summary')
    fallback_reason = None
    if not isinstance(effective_k_summary, dict) or effective_k_summary.get('selected_k') is None:
        fallback_reason = 'missing_effective_k_summary'
        effective_k_summary = summarize_effective_k([], requested_k=requested_k)
        effective_k_summary['selected_k'] = requested_k
        effective_k_summary['fallback_reason'] = fallback_reason
        warnings.warn(
            f"[Fold {args.fold_index}] Missing effective-k summary; using requested k={requested_k}."
        )
    best_params_requested = dict(best_params)
    best_params_effective = dict(best_params_requested)
    best_params_effective['k'] = int(effective_k_summary['selected_k'])
    best_params_alias = (
        best_params_effective
        if _flag_enabled(getattr(args, 'use_effective_k_for_fold_merge', 'FALSE'))
        else best_params_requested
    )
    train_effective_k = int(len(np.unique(train_labels)))

    stab_key, qual_key = _primary_metric_keys(args)
    selected_stability = summary.get(stab_key)
    selected_quality = summary.get(qual_key)
    if selected_stability is None and args.optimisation == 'single' and getattr(chosen.fitness, 'values', None):
        selected_stability = float(chosen.fitness.values[0])
    if selected_quality is None:
        selected_quality = float(train_quality)

    metrics = {
        'metrics_schema_version': METRICS_SCHEMA_VERSION,
        'dim_reduction': args.dim_reduction,
        'dim_reduction_label': _dimred_run_label(args),
        'dim_reduction_n_components': int(args.maxPC) if str(args.dim_reduction).lower() in {'pca', 'sparsepca', 'sparsenmf'} else None,
        'dim_reduction_sparse_l1': _dimred_sparse_l1(args),
        'data': df_final,
        'preprocessing': preprocessing_artifact,
        'preprocessing_details': preprocessing_artifact.get('preprocessing_details'),
        'ae_res': ae_res,
        'train_labels': np.asarray(train_labels, dtype=int),
        'train_quality': float(train_quality),
        'best_params': best_params_alias,
        'best_params_requested': best_params_requested,
        'best_params_effective': best_params_effective,
        'requested_k': requested_k,
        'train_effective_k': train_effective_k,
        'effective_k_summary': effective_k_summary,
        'effective_k_fallback_reason': fallback_reason,
        'mincluster_n_requested': int(args.mincluster_n),
        'mincluster_n_applied': int(fold_mincluster_n),
        'reference_n': int(len(df)),
        'current_n': int(len(X_latent)),
        'best_fitness': {
            'stability': selected_stability,
            'quality': selected_quality,
            'metrics_summary': summary,
            'fitness_values': tuple(getattr(chosen.fitness, 'values', ())),
        },
        'fold_status': 'ok',
        'skip_reason': None,
        'candidate_attempts': attempted_candidates,
        'train_ids': df.iloc[train_idx][args.subject_id_column].tolist(),
        'test_ids': df.iloc[test_idx][args.subject_id_column].tolist() if len(test_idx) else [],
    }

    os.makedirs(os.path.dirname(output_metrics_path) or '.', exist_ok=True)
    with open(output_metrics_path, 'wb') as f:
        dill.dump(metrics, f)
    print(f"Outer metrics saved to {output_metrics_path}")
    return


def do_merge(args):
    """Merge valid folds, fit final clustering on all data, estimate stability, and run SVM."""
    base_dir = os.path.abspath(getattr(args, 'base_dir', '.'))
    results_root = _output_root(base_dir, 'RESULTS_DIR', 'results')
    output_final_metrics_path = _resolve_path(base_dir, args.output_final_metrics) if args.output_final_metrics else None
    if not output_final_metrics_path:
        raise ValueError("For merge mode, --output_final_metrics must be specified")

    df = pd.read_csv(args.input_csv)
    meta = pd.read_csv(args.meta_csv)
    expected_fold_names = {f'fold{i}' for i in range(int(args.n_folds))}
    metrics_files = sorted(glob.glob(os.path.join(results_root, 'fold*', 'metrics.pkl')))
    metrics_by_fold = {
        os.path.basename(os.path.dirname(path)): path
        for path in metrics_files
        if os.path.basename(os.path.dirname(path)) in expected_fold_names
    }
    missing_folds = sorted(expected_fold_names - set(metrics_by_fold))
    if missing_folds:
        raise FileNotFoundError(
            f"Cannot merge an incomplete run; missing metrics for {missing_folds} under {results_root}."
        )
    fold_metrics = {}
    for fold_name in sorted(expected_fold_names, key=lambda name: int(name[4:])):
        metrics_file = metrics_by_fold[fold_name]
        fold_name = os.path.basename(os.path.dirname(metrics_file))
        with open(metrics_file, 'rb') as f:
            fold_metrics[fold_name] = pickle.load(f)

    schema_versions = {metric.get('metrics_schema_version', 1) for metric in fold_metrics.values()}
    if len(schema_versions) > 1:
        warnings.warn(
            f"Mixed fold metrics schema versions detected: {sorted(schema_versions)}. "
            "Missing effective-k summaries will fall back to requested k."
        )

    skipped_folds = {
        fold_name: fold_metric.get('skip_reason', 'unknown')
        for fold_name, fold_metric in fold_metrics.items()
        if fold_metric.get('fold_status', 'ok') != 'ok'
    }
    valid_fold_metrics = {
        fold_name: fold_metric
        for fold_name, fold_metric in fold_metrics.items()
        if fold_metric.get('fold_status', 'ok') == 'ok'
    }
    if skipped_folds:
        warnings.warn(
            "Skipping invalid outer folds during merge: "
            + ", ".join(f"{fold}={reason}" for fold, reason in sorted(skipped_folds.items()))
        )
    if not valid_fold_metrics:
        warning_message = (
            "No valid outer folds were available during merge; no valid clusters were found. "
            "Skipping final clustering, stability estimation, permutation tests, and SVM."
        )
        warnings.warn(warning_message)
        print(f"Warning: {warning_message}")
        metrics_merged = {
            'metrics_schema_version': METRICS_SCHEMA_VERSION,
            'merge_status': 'skipped',
            'skip_reason': 'no_valid_outer_folds',
            'analysis_skipped': True,
            'skip_message': warning_message,
            'dim_reduction': args.dim_reduction,
            'dim_reduction_label': _dimred_run_label(args),
            'dim_reduction_n_components': int(args.maxPC) if str(args.dim_reduction).lower() in {'pca', 'sparsepca', 'sparsenmf'} else None,
            'dim_reduction_sparse_l1': _dimred_sparse_l1(args),
            'valid_outer_folds': [],
            'skipped_outer_folds': skipped_folds,
            'effective_k': {
                'requested': None,
                'fold_bootstrap': {
                    fold: fold_metrics[fold].get('effective_k_summary')
                    for fold in sorted(fold_metrics)
                },
                'cross_fold_selected': None,
                'full_bootstrap': None,
                'consensus_cut': None,
                'final_observed': None,
            },
            'final_param_attempts': [],
            'final_labels': None,
            'final_params': None,
            'final_run_params': None,
            'consensus_cut_k': None,
            'final_effective_k': 0,
            'final_quality': None,
            'final_cluster_sizes': {},
            'final_quality_pvalue': None,
            'final_stability': None,
            'final_stability_ari': None,
            'final_stability_ari_pvalue': None,
            'final_stability_jaccard': None,
            'final_stability_SUM_MAT_full': None,
            'stability_by_preprocessing': {},
            'stability_bootstrap_cluster_counts': [],
            'stability_bootstrap_degenerate_fraction': None,
            'svm_results': None,
            'svm_final_model': None,
            'svm_feature_names': None,
            'svm_train_index': None,
        }
        report_rows = []
        for fold_name, fold_metric in sorted(fold_metrics.items()):
            summary = fold_metric.get('effective_k_summary') or {}
            report_rows.append({
                'pipeline': 'singleclust',
                'fold': fold_name,
                'component': 'final',
                'level': 'fold_bootstrap',
                'fold_status': fold_metric.get('fold_status', 'ok'),
                'skip_reason': fold_metric.get('skip_reason'),
                'requested_k': summary.get('requested_k', fold_metric.get('requested_k')),
                'selected_effective_k': summary.get('selected_k'),
                'mode_support': summary.get('support'),
                'retention_rate': summary.get('retention_rate'),
                'normalized_entropy': summary.get('normalized_entropy'),
                'distribution': repr(summary.get('counts', {})),
                'mincluster_n_applied': fold_metric.get('mincluster_n_applied'),
                'reference_n': fold_metric.get('reference_n'),
                'current_n': fold_metric.get('current_n'),
            })
        report_rows.append({
            'pipeline': 'singleclust',
            'fold': 'all',
            'component': 'final',
            'level': 'full_bootstrap',
            'fold_status': 'skipped',
            'skip_reason': 'no_valid_outer_folds',
            'requested_k': None,
            'selected_effective_k': None,
            'mode_support': None,
            'retention_rate': None,
            'normalized_entropy': None,
            'distribution': '{}',
            'mincluster_n_applied': None,
            'reference_n': len(df),
            'current_n': None,
        })
        os.makedirs(os.path.dirname(output_final_metrics_path) or '.', exist_ok=True)
        effective_k_report_path = os.path.join(
            os.path.dirname(output_final_metrics_path) or '.', 'effective_k_report.csv'
        )
        pd.DataFrame(report_rows).to_csv(effective_k_report_path, index=False)
        metrics_merged['effective_k_report_csv'] = effective_k_report_path
        with open(output_final_metrics_path, 'wb') as f:
            dill.dump(metrics_merged, f)
        print(f"Final merged metrics saved to {output_final_metrics_path}")
        return

    grouped_params = {}
    for fold_name, fold_metric in valid_fold_metrics.items():
        use_effective = _flag_enabled(getattr(args, 'use_effective_k_for_fold_merge', 'FALSE'))
        params_key = 'best_params_effective' if use_effective else 'best_params_requested'
        params = dict(fold_metric.get(params_key) or fold_metric['best_params'])
        params['k'] = int(params['k'])
        params['linkage'] = str(params['linkage'])
        key = tuple(sorted(params.items()))
        best_fitness = fold_metric.get('best_fitness', {})
        stability = best_fitness.get('stability')
        quality = best_fitness.get('quality')
        grouped_params.setdefault(key, {
            'params': params,
            'folds': [],
            'scores': [],
            'requested_params': [],
            'effective_k_summaries': [],
        })
        grouped_params[key]['folds'].append(fold_name)
        grouped_params[key]['requested_params'].append(
            dict(fold_metric.get('best_params_requested') or fold_metric['best_params'])
        )
        grouped_params[key]['effective_k_summaries'].append(
            fold_metric.get('effective_k_summary')
            or fold_metric.get('best_fitness', {}).get('metrics_summary', {}).get('effective_k_summary')
        )
        grouped_params[key]['scores'].append({
            'stability': float(stability) if stability is not None and np.isfinite(stability) else 0.0,
            'quality': float(quality) if quality is not None and np.isfinite(quality) else 0.0,
        })

    final_param_candidates = list(grouped_params.values())
    if not final_param_candidates:
        raise RuntimeError(
            "Cannot merge this run because no valid outer fold supplied final parameter candidates."
        )
    for candidate in final_param_candidates:
        candidate['mean_stability'] = float(np.mean([
            score['stability'] for score in candidate['scores']
        ]))
        candidate['mean_quality'] = float(np.mean([
            score['quality'] for score in candidate['scores']
        ]))

    for count in sorted({len(item['folds']) for item in final_param_candidates}, reverse=True):
        group = [item for item in final_param_candidates if len(item['folds']) == count]
        for metric in ('mean_stability', 'mean_quality'):
            values = [item[metric] for item in group]
            metric_min = min(values)
            metric_max = max(values)
            for item in group:
                item[f'normalized_{metric}'] = (
                    1.0 if np.isclose(metric_max, metric_min)
                    else (item[metric] - metric_min) / (metric_max - metric_min)
                )
        for item in group:
            item['multiobjective_distance'] = float(np.linalg.norm([
                1.0 - item['normalized_mean_stability'],
                1.0 - item['normalized_mean_quality'],
            ]))

    final_param_candidates = sorted(
        final_param_candidates,
        key=lambda item: (
            -len(item['folds']),
            item['multiobjective_distance'],
            int(item['params']['k']),
            item['params']['linkage'],
        ),
    )

    ae_data, subject_id_list, df_final, preprocessing_artifact = preprocessing(
        df, meta,
        subject_id_column=args.subject_id_column,
        col_threshold=args.col_threshold,
        row_threshold=args.row_threshold,
        skew_threshold=args.skew_threshold,
        scaler_type=args.scaler_type,
        modalities=args.modalities,
        dummy_code_modalities=args.dummy_code_modalities,
        mixed_categorical_modalities=args.mixed_categorical_modalities,
        return_artifact=True,
    )
    ae_res, X_latent = _build_latent_matrix(args, ae_data, df_final, seed_value=42)

    ids_all = list(subject_id_list[0]) if subject_id_list and subject_id_list[0] else []
    if not ids_all:
        raise ValueError('Subject ID list is empty; cannot perform stability estimation.')
    if df[args.subject_id_column].duplicated().any():
        raise ValueError(
            f"{args.subject_id_column} must be unique for final stability preprocessing."
        )
    raw_df_aligned = (
        df.set_index(args.subject_id_column, drop=False)
        .loc[ids_all]
        .reset_index(drop=True)
    )
    n_boot_full = int(getattr(args, 'n_bootstrap', 100))
    n_workers = min(max(1, int(getattr(args, 'n_jobs', 1) or 1)), n_boot_full)
    requested_final_bootstrap_preprocessing = str(
        getattr(args, 'final_bootstrap_preprocessing', 'outside')
    )
    final_bootstrap_preprocessing = (
        'inside'
        if requested_final_bootstrap_preprocessing == 'both'
        else requested_final_bootstrap_preprocessing
    )
    print(
        'Final stability bootstrap preprocessing mode: '
        f'{requested_final_bootstrap_preprocessing} '
        f'(primary: {final_bootstrap_preprocessing})'
    )

    def _bootstrap_candidate(candidate_params, preprocessing_mode):
        """Handle bootstrap candidate."""
        worker_args = (
            X_latent,
            ids_all,
            candidate_params,
            args.mincluster,
            args.mincluster_n,
            args.mincluster_resample_mode,
            args.internal_ensemble_enabled,
            args.internal_ensemble_bcs,
            args.internal_ensemble_sample_frac,
            args.internal_ensemble_feature_frac,
            preprocessing_mode,
            raw_df_aligned if preprocessing_mode == 'inside' else None,
            meta if preprocessing_mode == 'inside' else None,
            args if preprocessing_mode == 'inside' else None,
        )
        if n_workers == 1:
            _init_merge_bootstrap_worker(*worker_args)
            return [_run_merge_bootstrap(b) for b in range(n_boot_full)]

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_merge_bootstrap_worker,
            initargs=worker_args,
        ) as executor:
            chunksize = max(1, n_boot_full // (n_workers * 4))
            return list(executor.map(_run_merge_bootstrap, range(n_boot_full), chunksize=chunksize))

    def _labels_from_consensus(candidate_params, stability_summary, consensus_cut_k):
        """Handle labels from consensus."""
        consensus = stability_summary.get('consensus') if stability_summary else None
        union_ids = stability_summary.get('union_ids') if stability_summary else None
        if consensus is None or union_ids is None:
            return None, 0.0, None

        consensus = np.asarray(consensus, dtype=float)
        consensus = 0.5 * (consensus + consensus.T)
        np.fill_diagonal(consensus, 1.0)
        distances = 1.0 - consensus
        linkage_matrix = hierarchy.linkage(
            squareform(distances, checks=False),
            method=str(candidate_params.get('linkage', 'average')),
        )
        labels_union = hierarchy.cut_tree(
            linkage_matrix,
            n_clusters=int(consensus_cut_k),
        ).reshape(-1)
        if str(args.mincluster).upper() == 'TRUE':
            labels_union = enforce_min_cluster_size(
                distances,
                labels_union,
                min_size=int(args.mincluster_n),
            )

        consensus_quality = _safe_quality(distances, labels_union, precomputed=True)
        label_by_id = {subject_id: int(label) for subject_id, label in zip(union_ids, labels_union)}
        if any(subject_id not in label_by_id for subject_id in ids_all):
            return None, 0.0, None
        aligned_labels = np.asarray([label_by_id[subject_id] for subject_id in ids_all], dtype=int)
        union_index = {subject_id: index for index, subject_id in enumerate(union_ids)}
        aligned_indices = [union_index[subject_id] for subject_id in ids_all]
        aligned_distances = distances[np.ix_(aligned_indices, aligned_indices)]
        return aligned_labels, float(consensus_quality), aligned_distances

    final_params = None
    final_labels = None
    final_quality = None
    boot_label_dicts = None
    full_stab_ari = None
    full_stab_jaccard = None
    full_stab_sum_mat = None
    final_quality_matrix = None
    attempted_final_candidates = []
    for rank, candidate in enumerate(final_param_candidates[:5]):
        candidate_params = candidate['params']
        requested_k = int(candidate['requested_params'][0].get('k', candidate_params['k']))
        cross_fold_effective_k = int(candidate_params['k'])
        use_cross_fold_k = _flag_enabled(
            getattr(args, 'use_cross_fold_effective_k_for_final_run', 'FALSE')
        )
        consensus_cut_k = cross_fold_effective_k if use_cross_fold_k else requested_k
        run_params = dict(candidate_params)
        run_params['k'] = int(consensus_cut_k)
        direct_labels, direct_quality = run_ensemble_clustering(
            X_latent,
            **run_params,
            subject_id_list=subject_id_list,
            inner_jobs=args.n_jobs,
            pre_inner_jobs=args.n_jobs,
            mincluster=args.mincluster,
            mincluster_n=args.mincluster_n,
            internal_ensemble_enabled=args.internal_ensemble_enabled,
            internal_ensemble_bcs=args.internal_ensemble_bcs,
            internal_ensemble_sample_frac=args.internal_ensemble_sample_frac,
            internal_ensemble_feature_frac=args.internal_ensemble_feature_frac,
            internal_ensemble_seed=4242,
        )
        direct_labels = np.asarray(direct_labels, dtype=int)
        direct_cluster_count = int(len(np.unique(direct_labels)))
        attempt = {
            'rank': rank + 1,
            'params': candidate_params,
            'requested_k': requested_k,
            'cross_fold_effective_k': cross_fold_effective_k,
            'consensus_cut_k': int(consensus_cut_k),
            'folds': candidate['folds'],
            'direct_n_clusters': direct_cluster_count,
            'direct_quality': float(direct_quality),
            'consensus_n_clusters': None,
            'consensus_quality': None,
            'accepted': False,
        }
        if direct_cluster_count < 2 or not np.isfinite(direct_quality) or direct_quality <= 0:
            attempted_final_candidates.append(attempt)
            continue

        candidate_bootstraps = _bootstrap_candidate(
            run_params,
            final_bootstrap_preprocessing,
        )
        full_bootstrap_effective_k_summary = summarize_effective_k(
            [len(np.unique(item['labels'])) for item in candidate_bootstraps],
            requested_k=consensus_cut_k,
        )
        if any(
            int(item.get('mincluster_n_requested', args.mincluster_n)) != int(args.mincluster_n)
            for item in candidate_bootstraps
        ):
            raise RuntimeError('Full bootstrap minimum-cluster provenance is inconsistent.')
        if any(int(item.get('requested_k', -1)) != int(consensus_cut_k) for item in candidate_bootstraps):
            raise RuntimeError('Full bootstraps were not all requested at consensus_cut_k.')
        candidate_stability = consensus_pac_ccc(
            candidate_bootstraps,
            label_key='labels',
            return_consensus=True,
            return_ecdf=True,
        )
        candidate_labels, candidate_quality, candidate_quality_matrix = _labels_from_consensus(
            run_params,
            candidate_stability,
            consensus_cut_k,
        )
        consensus_cluster_count = (
            int(len(np.unique(candidate_labels))) if candidate_labels is not None else 0
        )
        accepted = (
            consensus_cluster_count >= 2
            and np.isfinite(candidate_quality)
            and candidate_quality > 0
        )
        attempt.update({
            'consensus_n_clusters': consensus_cluster_count,
            'consensus_quality': float(candidate_quality),
            'accepted': bool(accepted),
            'full_bootstrap_effective_k': full_bootstrap_effective_k_summary.get('selected_k'),
            'full_bootstrap_effective_k_support': full_bootstrap_effective_k_summary.get('support'),
            'full_bootstrap_effective_k_summary': full_bootstrap_effective_k_summary,
            'final_effective_k': consensus_cluster_count,
        })
        attempted_final_candidates.append(attempt)
        if not accepted:
            continue

        final_params = candidate_params
        final_run_params = run_params
        final_requested_k = requested_k
        final_cross_fold_effective_k = cross_fold_effective_k
        final_consensus_cut_k = int(consensus_cut_k)
        final_full_bootstrap_effective_k_summary = full_bootstrap_effective_k_summary
        final_labels = candidate_labels
        final_quality = float(candidate_quality)
        boot_label_dicts = candidate_bootstraps
        full_stab_sum_mat = candidate_stability
        final_quality_matrix = candidate_quality_matrix
        full_stab_ari = ari_stability_common_subjects(boot_label_dicts, label_key='labels')
        full_stab_jaccard = jaccard_stability_common_subjects(boot_label_dicts, label_key='labels')
        if rank > 0:
            print(f'Selected full-data parameter candidate rank {rank + 1} after rejecting degenerate candidates.')
        break

    if final_params is None:
        raise RuntimeError(
            'None of the top five cross-fold parameter candidates produced at least two clusters '
            'with positive quality on all data.'
        )
    print('Final selected parameters across folds:', final_params)

    def _summarize_stability(preprocessing_mode, bootstrap_results, stability_summary=None):
        """Summarize stability."""
        if stability_summary is None:
            stability_summary = consensus_pac_ccc(
                bootstrap_results,
                label_key='labels',
                return_consensus=True,
                return_ecdf=True,
            )
        return {
            'preprocessing': preprocessing_mode,
            'final_stability_ari': ari_stability_common_subjects(
                bootstrap_results, label_key='labels'
            ),
            'final_stability_jaccard': jaccard_stability_common_subjects(
                bootstrap_results, label_key='labels'
            ),
            'final_stability_SUM_MAT': stability_summary,
        }

    stability_by_preprocessing = {
        final_bootstrap_preprocessing: _summarize_stability(
            final_bootstrap_preprocessing,
            boot_label_dicts,
            full_stab_sum_mat,
        )
    }
    if requested_final_bootstrap_preprocessing == 'both':
        comparison_mode = 'outside' if final_bootstrap_preprocessing == 'inside' else 'inside'
        print(f'Running comparison final stability bootstrap preprocessing mode: {comparison_mode}')
        comparison_bootstraps = _bootstrap_candidate(final_run_params, comparison_mode)
        stability_by_preprocessing[comparison_mode] = _summarize_stability(
            comparison_mode,
            comparison_bootstraps,
        )

    bootstrap_cluster_counts = [
        int(len(np.unique(item['labels'])))
        for item in boot_label_dicts
        if item.get('labels') is not None and len(item.get('labels', [])) > 0
    ]
    bootstrap_degenerate_fraction = (
        float(np.mean(np.asarray(bootstrap_cluster_counts) <= 1))
        if bootstrap_cluster_counts else None
    )
    final_unique, final_counts = np.unique(final_labels, return_counts=True)
    final_cluster_sizes = {
        int(label): int(count)
        for label, count in zip(final_unique, final_counts)
    }

    stab_key, _ = _primary_metric_keys(args)
    if stab_key == 'stab_jaccard':
        final_stability = full_stab_jaccard
    else:
        final_stability = full_stab_ari

    n_permutations = int(getattr(args, 'n_permutations_pvalue', 200))
    final_quality_pvalue = _permutation_pvalue_quality(
        final_quality_matrix,
        final_labels,
        n_permutations=n_permutations,
        seed=4242,
        precomputed=True,
    )
    final_stability_ari_pvalue = _permutation_pvalue_stability_ari(
        boot_label_dicts,
        observed_ari=full_stab_ari,
        label_key='labels',
        n_permutations=n_permutations,
        seed=4343,
    )

    X_train, svm_feature_source = _simpleclust_svm_training_matrix(
        args,
        ae_res,
        df_final,
        preprocessing_artifact=preprocessing_artifact,
    )
    svm_results = None
    svm_final_model = None
    svm_feature_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
    svm_train_index = None

    if getattr(args, 'DO_SVM', 'FALSE').upper() == 'TRUE':
        valid_mask = final_labels >= 0
        labels_valid = final_labels[valid_mask]
        if len(np.unique(labels_valid)) >= 2:
            X_train_valid = X_train.loc[valid_mask].reset_index(drop=True)
            y_train = pd.Series(labels_valid, name='cluster')
            svm_results, svm_final_model = SVM_nested_cv(X_train_valid, y_train)
            svm_train_index = X_train.loc[valid_mask].index.to_numpy() if hasattr(X_train, 'loc') else None
        else:
            print('Warning: Less than 2 clusters in final labels; skipping SVM classification.')

    metrics_merged = {
        'metrics_schema_version': METRICS_SCHEMA_VERSION,
        'dim_reduction': args.dim_reduction,
        'dim_reduction_label': _dimred_run_label(args),
        'dim_reduction_n_components': int(args.maxPC) if str(args.dim_reduction).lower() in {'pca', 'sparsepca', 'sparsenmf'} else None,
        'dim_reduction_sparse_l1': _dimred_sparse_l1(args),
        'data': df_final,
        'preprocessing': preprocessing_artifact,
        'preprocessing_details': preprocessing_artifact.get('preprocessing_details'),
        'ae_res': ae_res,
        'final_labels': final_labels,
        'final_params': final_params,
        'final_run_params': final_run_params,
        'consensus_cut_k': final_consensus_cut_k,
        'final_effective_k': int(len(final_cluster_sizes)),
        'effective_k': {
            'requested': {'k': final_requested_k},
            'fold_bootstrap': {
                fold: valid_fold_metrics[fold].get('effective_k_summary')
                for fold in sorted(valid_fold_metrics)
            },
            'cross_fold_selected': {'k': final_cross_fold_effective_k},
            'full_bootstrap': final_full_bootstrap_effective_k_summary,
            'consensus_cut': {'k': final_consensus_cut_k},
            'final_observed': {'k': int(len(final_cluster_sizes)), 'cluster_sizes': final_cluster_sizes},
        },
        'valid_outer_folds': sorted(valid_fold_metrics),
        'skipped_outer_folds': skipped_folds,
        'final_param_attempts': attempted_final_candidates,
        'final_quality': float(final_quality),
        'final_cluster_sizes': final_cluster_sizes,
        'final_quality_pvalue': final_quality_pvalue,
        'final_stability': final_stability,
        'final_stability_ari': full_stab_ari,
        'final_stability_ari_pvalue': final_stability_ari_pvalue,
        'final_stability_jaccard': full_stab_jaccard,
        'final_stability_SUM_MAT_full': full_stab_sum_mat,
        'final_bootstrap_preprocessing': final_bootstrap_preprocessing,
        'requested_final_bootstrap_preprocessing': requested_final_bootstrap_preprocessing,
        'stability_by_preprocessing': stability_by_preprocessing,
        'stability_bootstrap_cluster_counts': bootstrap_cluster_counts,
        'stability_bootstrap_degenerate_fraction': bootstrap_degenerate_fraction,
        'svm_results': svm_results,
        'svm_final_model': svm_final_model,
        'svm_feature_names': svm_feature_names,
        'svm_train_index': svm_train_index,
        'svm_feature_source': svm_feature_source,
    }

    report_rows = []
    for fold_name, fold_metric in sorted(fold_metrics.items()):
        summary = fold_metric.get('effective_k_summary') or {}
        report_rows.append({
            'pipeline': 'singleclust',
            'fold': fold_name,
            'component': 'final',
            'level': 'fold_bootstrap',
            'fold_status': fold_metric.get('fold_status', 'ok'),
            'skip_reason': fold_metric.get('skip_reason'),
            'requested_k': summary.get('requested_k', fold_metric.get('requested_k')),
            'selected_effective_k': summary.get('selected_k'),
            'mode_support': summary.get('support'),
            'retention_rate': summary.get('retention_rate'),
            'normalized_entropy': summary.get('normalized_entropy'),
            'distribution': repr(summary.get('counts', {})),
            'mincluster_n_applied': fold_metric.get('mincluster_n_applied'),
            'reference_n': fold_metric.get('reference_n'),
            'current_n': fold_metric.get('current_n'),
        })
    applied_thresholds = [
        item.get('mincluster_n_applied') for item in boot_label_dicts
        if item.get('mincluster_n_applied') is not None
    ]
    current_sizes = [
        item.get('current_n') for item in boot_label_dicts
        if item.get('current_n') is not None
    ]
    report_rows.append({
        'pipeline': 'singleclust',
        'fold': 'all',
        'component': 'final',
        'level': 'full_bootstrap',
        'fold_status': 'ok',
        'skip_reason': None,
        'requested_k': final_consensus_cut_k,
        'selected_effective_k': final_full_bootstrap_effective_k_summary.get('selected_k'),
        'mode_support': final_full_bootstrap_effective_k_summary.get('support'),
        'retention_rate': final_full_bootstrap_effective_k_summary.get('retention_rate'),
        'normalized_entropy': final_full_bootstrap_effective_k_summary.get('normalized_entropy'),
        'distribution': repr(final_full_bootstrap_effective_k_summary.get('counts', {})),
        'mincluster_n_applied': repr(sorted(set(applied_thresholds))),
        'reference_n': len(ids_all),
        'current_n': repr(sorted(set(current_sizes))),
    })
    effective_k_report_path = os.path.join(
        os.path.dirname(output_final_metrics_path) or '.', 'effective_k_report.csv'
    )
    pd.DataFrame(report_rows).to_csv(effective_k_report_path, index=False)
    metrics_merged['effective_k_report_csv'] = effective_k_report_path

    os.makedirs(os.path.dirname(output_final_metrics_path) or '.', exist_ok=True)
    with open(output_final_metrics_path, 'wb') as f:
        dill.dump(metrics_merged, f)
    print(f"Final merged metrics saved to {output_final_metrics_path}")
    return


# --- Mode: init candidate set ---
def do_init(args):
    """
    Initialize the deterministic linkage-by-k candidate grid.

    The output is still called a population file because the shared scheduler
    and older artifacts use that name. For singleclust it is simply a list of
    candidate parameter combinations.
    """
    base_dir = os.path.abspath(getattr(args, "base_dir", "."))
    search_root = _search_root(base_dir, args.fold_index if hasattr(args, "fold_index") else 0)
    population_file = _resolve_path(base_dir, args.population_file) or os.path.join(
        search_root, f"population_init_fold{getattr(args, 'fold_index', 0)}.pkl"
    )
    # Seed RNGs so the optional grid truncation order is reproducible.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        if torch is not None:
            torch.manual_seed(args.seed)

    pop, _ = _build_grid_population(args)

    os.makedirs(os.path.dirname(population_file) or '.', exist_ok=True)
    with open(population_file, 'wb') as f:
        dill.dump(pop, f)
    print(f"Initial grid-search candidate set ({len(pop)}) saved to {population_file}")
    return



# --- Command-line entry point ------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Original args
    parser.add_argument('--input_csv', default='cleaned_discovery_data.csv')
    parser.add_argument('--meta_csv', default='merged_meta.csv')
    parser.add_argument('--base_dir', default='path/to/multiclust')
    parser.add_argument('--subject_id_column', default='src_subject_id')
    parser.add_argument('--col_threshold', type=float, default=0.5)
    parser.add_argument('--row_threshold', type=float, default=0.5)
    parser.add_argument('--skew_threshold', type=float, default=0.75)
    parser.add_argument('--scaler_type', default='robust')
    parser.add_argument('--modalities', nargs='+', default=['Internalising', 'Functioning', 'Cognition', 'Detachment', 'Psychoticism'])
    parser.add_argument('--dummy_code_modalities', nargs='*', default=[])
    parser.add_argument('--mixed_categorical_modalities', nargs='*', default=[])
    parser.add_argument('--dim_reduction', choices=[None, "None", 'VAE', 'AE', 'SparseVAE', 'SparseAE', 'PCA', 'SparsePCA', 'SparseNMF', 'sparsenmf', 'Sparse_NMF', 'sparse_nmf', 'SNMF', 'snmf'], default='VAE', help='Dimensionality reduction method to use (VAE, SparseVAE, AE, SparseAE, PCA, SparsePCA, SparseNMF, or None)')
    parser.add_argument('--maxPC', type=int, default=20, help='Maximum number of components for PCA/SparsePCA/SparseNMF (ignored for other dim_reduction methods)')
    parser.add_argument('--spca_alpha', type=float, default=1.0, help='Sparsity penalty for SparsePCA (ignored unless dim_reduction=SparsePCA)')
    parser.add_argument('--spca_ridge_alpha', type=float, default=0.01, help='Ridge regularization used during SparsePCA transforms')
    parser.add_argument('--spca_max_iter', type=int, default=1000, help='Maximum iterations for SparsePCA fitting')
    parser.add_argument('--snmf_alpha', type=float, default=0.1, help='L1/L2 regularization strength for SparseNMF W and H matrices')
    parser.add_argument('--snmf_l1_ratio', type=float, default=1.0, help='SparseNMF regularization mix: 1.0 is L1, 0.0 is L2')
    parser.add_argument('--snmf_max_iter', type=int, default=1000, help='Maximum iterations for SparseNMF fitting')
    parser.add_argument('--sparse_l1_lambda', type=float, default=1e-3, help='Latent-space L1 penalty used by SparseAE/SparseVAE (ignored for other dim_reduction methods)')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128,256,512])
    parser.add_argument('--activation_functions', nargs='+', default=['ReLU','LeakyReLU','selu','swish'])
    parser.add_argument('--learning_rates', nargs='+', type=float, default=[0.001,0.0001])
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[32,64,128])
    parser.add_argument('--latent_dims', nargs='+', type=int, default=[2,5,10])
    parser.add_argument('--k_min', type=int, default=2)
    parser.add_argument('--k_max', type=int, default=10)
    parser.add_argument('--linkages', type=str, nargs='+', default=['complete','average','weighted'])
    parser.add_argument('--grid_max_candidates', '--n_population', dest='n_population', type=int, default=0,
                        help='Maximum grid-search candidates to evaluate; 0 evaluates the full grid (legacy alias: --n_population).')
    parser.add_argument('--search_rounds', '--n_generations', dest='n_generations', type=int, default=1,
                        help='Deprecated compatibility option. Grid search now runs a single scoring pass.')
    parser.add_argument('--optimisation', choices=['single','multi'], default='multi')
    parser.add_argument('--search_objectives', '--ga_objectives', dest='search_objectives', nargs='+', default=None,
                        help='Objectives optimised during multi-objective search (legacy alias: --ga_objectives).')
    parser.add_argument('--n_bootstrap', type=int, default=100)
    parser.add_argument('--final_bootstrap_preprocessing', choices=['outside', 'inside', 'both'], default='outside',
                        help='Rerun preprocessing inside final stability resamples, outside them, or report both.')
    parser.add_argument('--n_permutations_pvalue', type=int, default=200,
                        help='Number of permutations used for final clustering quality/ARI p-values in merge mode (0 disables).')
    parser.add_argument('--bootstrap_mode', choices=['bootstrap','subsample'], default='subsample')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--output_pkl', default='pipeline_results.pkl')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel workers for bootstrap clustering')
    parser.add_argument('--TEST', choices=['TRUE', 'FALSE'], default='FALSE',
                        help='Skip VAE and use preprocessed features as latent embeddings when TRUE')
    parser.add_argument('--max_missing_bootstraps', type=int, default=5,
                        help='Maximum number of missing bootstrap label files allowed before gather aborts')
    parser.add_argument('--mincluster', default='TRUE', type=str, help='Enforce minimum cluster size of 10 in final clustering (TRUE/FALSE; case-insensitive)')
    parser.add_argument('--mincluster_n', type=int, default=10, help='Minimum cluster size to enforce in final clustering')
    parser.add_argument('--mincluster_resample_mode', choices=['fixed', 'scaled'], default='fixed',
                        help='Use a fixed minimum cluster size or scale it to the current resample size.')
    parser.add_argument('--use_effective_k_for_fold_merge', choices=['TRUE', 'FALSE'], default='FALSE',
                        help='Vote across folds using bootstrap-derived effective k values.')
    parser.add_argument('--use_cross_fold_effective_k_for_final_run', choices=['TRUE', 'FALSE'], default='FALSE',
                        help='Use the cross-fold selected effective k for full-data bootstraps and consensus cutting.')
    parser.add_argument('--internal_ensemble_enabled', choices=['TRUE', 'FALSE'], default='FALSE',
                        help='Use balanced perturbed base clusterings inside the simple ensemble clusterer.')
    parser.add_argument('--internal_ensemble_bcs', type=int, default=5,
                        help='Number of internal base clusterings when --internal_ensemble_enabled=TRUE.')
    parser.add_argument('--internal_ensemble_sample_frac', type=float, default=0.8,
                        help='Fraction of subjects sampled without replacement for each internal base clustering.')
    parser.add_argument('--internal_ensemble_feature_frac', type=float, default=1.0,
                        help='Fraction of features sampled without replacement for each internal base clustering.')
    # New mode args
    parser.add_argument('--mode', choices=['bootstrap','gather','outer','init','merge'], default='init')
    parser.add_argument('--search_round', '--generation', dest='generation', type=int,
                        help='Deprecated compatibility option; ignored in single-pass grid search.')
    parser.add_argument('--population_file', '--candidate_file', dest='population_file', type=str,
                        help='Candidate set pickle file (legacy alias: --population_file).')
    parser.add_argument('--seed',            type=int, default=None,
                    help='Random seed for grid-search candidate generation (used with --mode init).')
    parser.add_argument('--population_dir', '--candidate_dir', dest='population_dir', type=str, help='Directory where candidate-set files are stored (legacy alias: --population_dir).')
    parser.add_argument('--population_initial_file', '--candidate_grid_file', dest='population_initial_file', type=str, help='Initial grid candidate-set file for bootstrap mode (legacy alias: --population_initial_file).')
    parser.add_argument('--bootstrap_index', type=int)
    parser.add_argument('--bootstrap_dir', type=str)
    parser.add_argument('--output_labels', type=str, help='Where to save bootstrap labels for stability computation')
    parser.add_argument('--output_population', '--output_candidates', dest='output_population', type=str,
                        help='Output file for scored candidate set (legacy alias: --output_population).')
    parser.add_argument('--fold_index', type=int)
    parser.add_argument('--output_metrics', type=str)
    parser.add_argument('--output_final_metrics', type=str)
    parser.add_argument('--ga_cxpb', type=float, default=0.7,
                        help='Deprecated (unused in grid-search mode).')
    parser.add_argument('--ga_mutpb', type=float, default=0.2,
                        help='Deprecated (unused in grid-search mode).')
    parser.add_argument('--ga_elitism', type=int, default=2,
                        help='Deprecated (unused in grid-search mode).')
    parser.add_argument('--DO_SVM', choices=['TRUE', 'FALSE'], default='FALSE',
                        help='In OUTER mode, whether to run SVM classification on the final clustering labels (TRUE/FALSE).')
    args = parser.parse_args()

    if args.dim_reduction is not None:
        dim_text = str(args.dim_reduction).strip()
        if dim_text.lower() in {"sparse_nmf", "sparse-nmf", "snmf", "sparsenmf"}:
            args.dim_reduction = "sparsenmf"
    if args.maxPC < 1:
        raise ValueError("--maxPC must be >= 1.")
    if args.snmf_alpha < 0:
        raise ValueError("--snmf_alpha must be >= 0.")
    if not (0.0 <= args.snmf_l1_ratio <= 1.0):
        raise ValueError("--snmf_l1_ratio must be in the interval [0, 1].")
    if args.snmf_max_iter < 1:
        raise ValueError("--snmf_max_iter must be >= 1.")
    args.ga_objectives = _normalize_objective_tokens(getattr(args, "search_objectives", []), args.optimisation)
    args.mincluster = str(args.mincluster).upper()
    if args.mincluster not in {'TRUE', 'FALSE'}:
        raise ValueError(f"Invalid --mincluster '{args.mincluster}'. Use TRUE or FALSE.")
    args.search_objectives = list(args.ga_objectives)

    # --- Clean out any old classes so we can rebuild them fresh ---
    for attr in list(vars(creator).keys()):
        if attr.startswith("FitnessMulti"):
            delattr(creator, attr)
    for cls in ("FitnessMax", "Individual"):
        if hasattr(creator, cls):
            delattr(creator, cls)

    # Ensure DEAP classes exist before loading pickled populations
    _ensure_multi_fitness_class(args)

    if args.mode == 'init':
        do_init(args)
    elif args.mode == 'bootstrap':
        do_bootstrap(args)
    elif args.mode == 'gather':
        do_gather(args)
    elif args.mode == 'outer':
        do_outer(args)
    elif args.mode == 'merge':
        do_merge(args)
