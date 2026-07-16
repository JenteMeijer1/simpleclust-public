#!/usr/bin/env python
"""
Full multi-view clustering pipeline.

This file is the command-line entry point used by the SLURM/run-profile
wrappers. The pipeline is intentionally split into modes so expensive work can
be scheduled as independent jobs:

1. init
   Create the initial genetic-algorithm (GA) population for one outer fold.
2. bootstrap
   For one fold, generation, and resample: preprocess the training data,
   create per-modality representations, cluster every GA candidate, and save
   labels/quality scores for later stability estimation.
3. gather
   Read all bootstrap label files for a fold/generation, compute each
   candidate's stability and quality objectives, update the fold hall of fame,
   and write the next GA generation.
4. outer
   Refit the fold-level representation on the outer training split, choose a
   non-degenerate hall-of-fame candidate, and save fold metrics.
5. merge
   Combine fold metrics, select the final hyperparameters, refit on all data,
   estimate final stability, and optionally train SVM classifiers.

The important invariant throughout the pipeline is that every modality must be
aligned to the same subject order before clustering. Most guard checks and
subject-ID bookkeeping below exist to preserve that invariant across
preprocessing, resampling, and final reporting.
"""
import os

# Limit BLAS/Numexpr threading to avoid oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# --- Imports ------------------------------------------------
from Utils import *
from Utils import _preprocessing_split_first
import time
import pandas as pd
import re
import dill
import argparse
import hashlib
from itertools import combinations
from itertools import repeat
import heapq
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from joblib import Parallel, delayed
import warnings

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from operator import itemgetter
from deap import base, creator, tools, algorithms

# --- SciPy imports for cophenetic correlation calculation ---
import scipy.cluster.hierarchy as hierarchy
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.cluster import KMeans, SpectralClustering
import random
import torch.nn as nn
import torch
torch.set_num_threads(1)
import sys
import gc
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from collections import defaultdict
import pickle

# Import own functions
from VAE import run_VAE_complete
from AE  import run_AE_complete
from parea_functions import *
from Utils import fit_sparse_nmf_reducer, transform_sparse_nmf_reducer
from parea_functions import convert_to_parameters
from SVM import *
import glob

_MAX_32BIT_SEED = 2 ** 32 - 1
METRICS_SCHEMA_VERSION = 2


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
    """Derive a stable 32-bit seed from a named RNG context."""
    payload = "|".join(str(part) for part in (base,) + tuple(parts)).encode("utf-8")
    seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return seed % _MAX_32BIT_SEED


def _seed_everything(seed):
    """Handle seed everything."""
    np.random.seed(int(seed))
    random.seed(int(seed))
    if torch is not None:
        torch.manual_seed(int(seed))


def _ga_bootstrap_seed(fold_index, bootstrap_index):
    """Handle ga bootstrap seed."""
    return _derive_seed(
        "ga_bootstrap",
        int(fold_index or 0),
        int(bootstrap_index or 0),
    )

GA_OBJECTIVE_ALIASES = {
    # --- Quality objectives (unchanged semantics) ---
    "av_qual_view": "mean_view_quality",
    "avg_qual_view": "mean_view_quality",
    "mean_view_quality": "mean_view_quality",
    "qual_final": "final_quality",
    "final_quality": "final_quality",
    "min_qual_view": "min_view_quality",
    "min_view_quality": "min_view_quality",

    # --- ARI-based stability (default) ---
    "av_stab_view": "mean_view_stability_ari",
    "avg_stab_view": "mean_view_stability_ari",
    "mean_view_stability": "mean_view_stability_ari",
    "mean_view_stability_ari": "mean_view_stability_ari",
    "mean_view_stab": "mean_view_stability_ari",
    "mean_view_stab_ari": "mean_view_stability_ari",

    "stab_final": "final_stability_ari",
    "final_stability": "final_stability_ari",
    "final_stability_ari": "final_stability_ari",

    "min_stab_view": "min_view_stability_ari",
    "min_view_stability": "min_view_stability_ari",
    "min_view_stability_ari": "min_view_stability_ari",

    # --- Co-association-based stability ---
    "mean_view_stability_coassoc": "mean_view_stability_coassoc",
    "final_stability_coassoc": "final_stability_coassoc",

    # --- Jaccard-based stability ---
    "mean_view_stability_jaccard": "mean_view_stability_jaccard",
    "final_stability_jaccard": "final_stability_jaccard",

    # --- CCC-based stability ---
    "mean_view_stability_ccc": "mean_view_stability_CCC",
    "mean_view_stability_CCC": "mean_view_stability_CCC",
    "final_stability_ccc": "final_stability_CCC",
    "final_stability_CCC": "final_stability_CCC",
}
DEFAULT_GA_OBJECTIVES = [
    "mean_view_stability_ari",
    "mean_view_quality",
    "final_stability_ari",
    "final_quality",
]
DEFAULT_FUSION_METHODS = ["agreement", "consensus", "disagreement"]
DIM_REDUCTION_ALIASES = {
    "none": "none",
    "vae": "vae",
    "ae": "ae",
    "pca": "pca",
    "mca": "mca",
    "famd": "famd",
    "mixed_svd": "famd",
    "mixed-svd": "famd",
    "sparsenmf": "sparsenmf",
    "sparse_nmf": "sparsenmf",
    "sparse-nmf": "sparsenmf",
    "snmf": "sparsenmf",
}


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


def _normalize_dim_reduction_method(value):
    """Normalize dim reduction method."""
    if value is None:
        return "none"
    text = str(value).strip().lower()
    if not text:
        return "none"
    if text not in DIM_REDUCTION_ALIASES:
        valid = ", ".join(sorted(DIM_REDUCTION_ALIASES))
        raise ValueError(f"Unknown dim_reduction method '{value}'. Valid options: {valid}")
    return DIM_REDUCTION_ALIASES[text]


def _parse_dim_reduction_overrides(raw_overrides, modalities, default_method):
    """Parse dim reduction overrides."""
    methods = {mod: default_method for mod in modalities}
    if not raw_overrides:
        return methods

    known_modalities = set(modalities)
    for raw in raw_overrides:
        if "=" not in str(raw):
            raise ValueError(
                "Each --dim_reduction_by_modality entry must have the form Modality=Method."
            )
        modality, method = raw.split("=", 1)
        modality = modality.strip()
        if modality not in known_modalities:
            valid = ", ".join(modalities)
            raise ValueError(
                f"Unknown modality '{modality}' in --dim_reduction_by_modality. Valid modalities: {valid}"
            )
        methods[modality] = _normalize_dim_reduction_method(method)
    return methods


def _parse_dummy_code_modalities(raw_modalities, modalities):
    """Parse dummy code modalities."""
    if raw_modalities is None:
        return list(modalities)

    selected = []
    seen = set()
    known_modalities = set(modalities)
    for raw in raw_modalities:
        modality = str(raw).strip()
        if not modality:
            continue
        if modality not in known_modalities:
            valid = ", ".join(modalities)
            raise ValueError(
                f"Unknown modality '{modality}' in --dummy_code_modalities. Valid modalities: {valid}"
            )
        if modality not in seen:
            selected.append(modality)
            seen.add(modality)
    return selected


def _validate_mixed_categorical_dim_reduction(mixed_categorical_modalities, modality_dim_reduction):
    """Validate mixed categorical dim reduction."""
    mixed = list(mixed_categorical_modalities or [])
    invalid = [
        f"{mod}={modality_dim_reduction.get(mod)}"
        for mod in mixed
        if modality_dim_reduction.get(mod) not in {"famd", "mca"}
    ]
    if invalid:
        raise ValueError(
            "Mixed categorical modalities must use FAMD, MCA, or MIXED_SVD via "
            "--dim_reduction_by_modality. Invalid settings: " + ", ".join(invalid)
        )


def _build_autoencoder_modalities(dict_final, modalities, subject_id_column, selected_modalities):
    """Build autoencoder modalities."""
    selected = set(selected_modalities)
    return {
        mod: (
            dict_final[mod][subject_id_column].tolist(),
            dict_final[mod].drop(columns=[subject_id_column], errors='ignore').to_numpy(dtype=np.float32, copy=True),
        )
        for mod in modalities
        if mod in selected
    }


def _one_hot_encoder():
    """Handle one hot encoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _low_cardinality_numeric_columns(df, max_unique=10):
    """Handle low cardinality numeric columns."""
    cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        n_unique = df[col].dropna().nunique()
        if 0 < n_unique <= max_unique:
            cols.append(col)
    return cols


def _run_mixed_type_svd(df, subject_id_column, method, random_state):
    """Run mixed type svd."""
    Xdf = df.drop(columns=[subject_id_column], errors="ignore").copy()
    if Xdf.empty:
        return np.empty((len(df), 0), dtype=np.float32)

    categorical_cols = Xdf.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if method == "mca":
        categorical_cols = Xdf.columns.tolist()
    else:
        categorical_cols = list(dict.fromkeys(categorical_cols + _low_cardinality_numeric_columns(Xdf)))

    numeric_cols = [col for col in Xdf.columns if col not in categorical_cols]
    matrices = []

    if numeric_cols:
        numeric = Xdf[numeric_cols].apply(pd.to_numeric, errors="coerce")
        numeric = SimpleImputer(strategy="median").fit_transform(numeric)
        numeric = StandardScaler().fit_transform(numeric)
        matrices.append(np.asarray(numeric, dtype=np.float32))

    if categorical_cols:
        categorical = Xdf[categorical_cols].astype("object")
        categorical = SimpleImputer(strategy="most_frequent").fit_transform(categorical)
        categorical = _one_hot_encoder().fit_transform(categorical)
        matrices.append(np.asarray(categorical, dtype=np.float32))

    if not matrices:
        return np.empty((len(df), 0), dtype=np.float32)

    X = np.hstack(matrices) if len(matrices) > 1 else matrices[0]
    max_components = min(50, X.shape[0] - 1, X.shape[1] - 1)
    if max_components < 1:
        return X.astype(np.float32, copy=False)

    svd = TruncatedSVD(n_components=max_components, random_state=random_state)
    return svd.fit_transform(X).astype(np.float32, copy=False)


def _run_dimensionality_reduction(
    dict_final,
    modalities,
    subject_id_column,
    modality_dim_reduction,
    pca_variance_threshold,
    snmf_n_components,
    snmf_alpha,
    snmf_l1_ratio,
    snmf_max_iter,
    hidden_dims,
    activation_functions,
    learning_rates,
    batch_sizes,
    latent_dims,
    random_state,
):
    """
    Build one numeric representation per modality for clustering.

    Each modality can use its own method via args.dim_reduction_by_modality:
    - none: pass preprocessed features through unchanged
    - pca: linear dimensionality reduction for numeric tables
    - sparsenmf: sparse non-negative matrix factorization for numeric tables
    - famd/mca: mixed-type SVD approximation for categorical/mixed tables
    - ae/vae: learned latent representation selected by nested model search

    The returned data_list follows args.modalities order and is the direct
    input to parea_2_mv.
    """
    ae_res = {}
    methods_used = {modality_dim_reduction[mod] for mod in modalities}

    none_modalities = [mod for mod in modalities if modality_dim_reduction[mod] == "none"]
    if none_modalities:
        print(f"Using preprocessed features directly for modalities: {', '.join(none_modalities)}")
        for mod in none_modalities:
            X = dict_final[mod].drop(columns=[subject_id_column], errors='ignore').to_numpy(dtype=np.float32, copy=True)
            ae_res[mod] = {"final_latent": X}

    pca_modalities = [mod for mod in modalities if modality_dim_reduction[mod] == "pca"]
    if pca_modalities:
        print(f"Running PCA for modalities: {', '.join(pca_modalities)}")
        for mod in pca_modalities:
            X = dict_final[mod].drop(columns=[subject_id_column], errors='ignore').to_numpy(dtype=np.float32, copy=True)
            max_components = min(X.shape[1], X.shape[0] - 1)
            if max_components >= 1:
                n_components = (
                    float(pca_variance_threshold)
                    if pca_variance_threshold is not None
                    else min(50, max_components)
                )
                pca = PCA(n_components=n_components, random_state=random_state)
                X = pca.fit_transform(X)
                ae_res[mod] = {
                    "final_latent": np.asarray(X, dtype=np.float32, copy=False),
                    "pca_model": pca,
                    "pca_n_components": int(pca.n_components_),
                    "pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
                }
            else:
                ae_res[mod] = {"final_latent": np.asarray(X, dtype=np.float32, copy=False)}
        print("PCA dimensionality reduction completed.")

    snmf_modalities = [mod for mod in modalities if modality_dim_reduction[mod] == "sparsenmf"]
    if snmf_modalities:
        print(f"Running SparseNMF for modalities: {', '.join(snmf_modalities)}")
        for mod in snmf_modalities:
            reducer = fit_sparse_nmf_reducer(
                dict_final[mod],
                subject_id_column=subject_id_column,
                n_components=snmf_n_components,
                alpha=snmf_alpha,
                l1_ratio=snmf_l1_ratio,
                max_iter=snmf_max_iter,
                random_state=random_state,
            )
            X = transform_sparse_nmf_reducer(
                dict_final[mod],
                reducer,
                subject_id_column=subject_id_column,
            )
            ae_res[mod] = {
                "final_latent": X,
                "dim_reduction_model": reducer,
                "snmf_model": reducer.get("nmf"),
                "snmf_n_components": int(reducer.get("n_components", X.shape[1])),
                "snmf_alpha": float(snmf_alpha),
                "snmf_l1_ratio": float(snmf_l1_ratio),
            }
        print("SparseNMF dimensionality reduction completed.")

    mixed_modalities = [mod for mod in modalities if modality_dim_reduction[mod] in {"famd", "mca"}]
    if mixed_modalities:
        print(f"Running mixed-type dimensionality reduction for modalities: {', '.join(mixed_modalities)}")
        for mod in mixed_modalities:
            method = modality_dim_reduction[mod]
            reducer = fit_mixed_type_svd_reducer(
                dict_final[mod],
                subject_id_column=subject_id_column,
                method=method,
                random_state=random_state,
            )
            X = transform_mixed_type_svd_reducer(
                dict_final[mod],
                reducer,
                subject_id_column=subject_id_column,
            )
            ae_res[mod] = {
                "final_latent": X,
                "dim_reduction_model": reducer,
            }
        print("Mixed-type dimensionality reduction completed.")

    ae_modalities = [mod for mod in modalities if modality_dim_reduction[mod] == "ae"]
    if ae_modalities:
        print(f"Running AE for modalities: {', '.join(ae_modalities)}")
        ae_inputs = _build_autoencoder_modalities(dict_final, modalities, subject_id_column, ae_modalities)
        ae_outputs = run_AE_complete(
            ae_inputs,
            hidden_dims=hidden_dims,
            activation_functions=activation_functions,
            learning_rates=learning_rates,
            batch_sizes=batch_sizes,
            latent_dims=latent_dims,
        )
        ae_res.update(ae_outputs)

    vae_modalities = [mod for mod in modalities if modality_dim_reduction[mod] == "vae"]
    if vae_modalities:
        print(f"Running VAE for modalities: {', '.join(vae_modalities)}")
        vae_inputs = _build_autoencoder_modalities(dict_final, modalities, subject_id_column, vae_modalities)
        vae_outputs = run_VAE_complete(
            vae_inputs,
            hidden_dims=hidden_dims,
            activation_functions=activation_functions,
            learning_rates=learning_rates,
            batch_sizes=batch_sizes,
            latent_dims=latent_dims,
        )
        ae_res.update(vae_outputs)

    missing = [mod for mod in modalities if mod not in ae_res]
    if missing:
        raise RuntimeError(f"Missing latent representations for modalities: {missing}")

    if methods_used == {"none"}:
        print("Skipping learned dimensionality reduction and using preprocessed features as latent representations.")

    data_list = [np.asarray(ae_res[mod]['final_latent'], dtype=np.float32, copy=False) for mod in modalities]
    return ae_res, data_list



def _normalize_objective_tokens(raw_tokens, optimisation_mode):
    """Map user-specified GA objective tokens to canonical names."""
    if not raw_tokens:
        # Defaults: single-objective -> final-stability (ARI); multi -> standard multi-objective set.
        if optimisation_mode == "single":
            tokens = ["final_stability_ari"]
        else:
            tokens = DEFAULT_GA_OBJECTIVES
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
        if key not in GA_OBJECTIVE_ALIASES:
            valid = ", ".join(sorted(set(GA_OBJECTIVE_ALIASES.values())))
            raise ValueError(f"Unknown GA objective '{tok}'. Valid options: {valid}")
        normalized.append(GA_OBJECTIVE_ALIASES[key])
    return normalized


# Helper to choose the primary metric keys for summary attributes based on GA objectives
def _primary_metric_keys(args):
    """
    Decide which summary keys to use for the convenience attributes
    (mean_view_stab, final_stab, etc.) based on args.ga_objectives.

    Returns
    -------
    stab_view_key, stab_final_key, qual_view_key, qual_final_key
    """
    objs = list(getattr(args, "ga_objectives", []))

    stab_view_key = None
    stab_final_key = None

    for obj in objs:
        if isinstance(obj, str):
            if obj.startswith("mean_view_stability") and stab_view_key is None:
                stab_view_key = obj
            if obj.startswith("final_stability") and stab_final_key is None:
                stab_final_key = obj

    # Fallbacks: default to ARI-based if not explicitly in objectives
    if stab_view_key is None:
        stab_view_key = "mean_view_stability_ari"
    if stab_final_key is None:
        stab_final_key = "final_stability_ari"

    # Quality keys are fixed names in `summary`
    qual_view_key = "mean_view_quality"
    qual_final_key = "final_quality"

    return stab_view_key, stab_final_key, qual_view_key, qual_final_key


def _ensure_multi_fitness_class(args):
    """Ensure the DEAP multi-objective fitness/individual classes exist."""
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
    params = convert_to_parameters(len(data_list), cand)
    final_labels, individual_labels, view_scores_per_view, view_score, final_score = parea_2_mv(
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
            "ga_bootstrap_internal_ensemble",
            int(getattr(args, "fold_index", 0) or 0),
            int(getattr(args, "bootstrap_index", 0) or 0),
        )
    )
    return final_labels, individual_labels, view_scores_per_view, view_score, final_score

def save_pickle(path, obj):
    """Save an object to a pickle file safely."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

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
    """Resolve a run output root, allowing wrappers to separate repeated runs."""
    configured = os.environ.get(env_name)
    if configured:
        return _resolve_path(base_dir, configured)
    return os.path.join(base_dir, default_name)

def _ga_root(base_dir, fold_index):
    """Canonical GA root for a given fold under intermediates."""
    return os.path.join(_output_root(base_dir, "INTERMEDIATES_DIR", "intermediates"), f"fold{fold_index}", "ga")

def preprocessing(df,
                  meta,
                  subject_id_column='src_subject_id',
                  col_threshold=0.5, row_threshold=0.5,
                  skew_threshold=0.75,
                  scaler_type='robust',
                  modalities=['Internalising', 'Functioning', 'Cognition', 'Detachment', 'Psychoticism'],
                  dummy_code_modalities=None,
                  mixed_categorical_modalities=None,
                  impute_parea=False,
                  export_preprocessing_details=False):
    """
    Pipeline-local wrapper around Utils._preprocessing_split_first.

    Preprocessing removes high-missingness rows/columns, transforms skewed
    variables, dummy-codes selected modalities, scales features, imputes within
    modality, then realigns all modalities to the same subject-ID order.

    Keep implementation changes in Utils._preprocessing_split_first; this
    wrapper exists so older code can continue importing preprocessing from
    full_pipeline.py while all modes share one preprocessing implementation.

    Returns
    -------
    ae_data, subject_id_list, dict_final
        The standard clustering inputs when export_preprocessing_details=False.
    ae_data, subject_id_list, dict_final, preprocessing_details
        The same outputs plus audit metadata when export_preprocessing_details=True.
    """
    return _preprocessing_split_first(
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
        export_preprocessing_details=export_preprocessing_details
    )


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
    This avoids repeating np.intersect1d for every individual.
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


def _distribution_summary(values, ci_level=0.95):
    """
    Reporting-friendly summary for a 1D numeric sample.
    Returns mean/SD/SE, median/IQR, min/max, and percentile CI.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "se": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "min": np.nan,
            "max": np.nan,
            "ci_level": float(ci_level),
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = float(std / np.sqrt(n)) if n > 1 else 0.0
    alpha = max(0.0, min(1.0, 1.0 - float(ci_level)))
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ci_level": float(ci_level),
        "ci_lower": float(np.percentile(arr, lo_q)),
        "ci_upper": float(np.percentile(arr, hi_q)),
    }


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



# --- Fitness computation for gather ---
def _compute_fitness_for_ind(
    i,
    label_dicts,
    modalities,
    objectives,
    cache_dir=None,
    fold_index=None,
    bootstrap_index=None,
    precomputed_alignment=None,
    precomputed_consensus_cache=None,
    candidate_params=None,
    ):
    """
    Score one GA candidate after all bootstraps have produced labels.

    label_dicts contains the labels saved by do_bootstrap for every resample.
    For candidate i, this function compares labels across resamples to estimate
    per-view and final stability, averages clustering-quality scores, and
    returns the subset requested by args.ga_objectives as the DEAP fitness tuple.
    """

    n_views = len(modalities)

    # Decide which additional stability flavours to compute based on GA objectives
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

    # Track collapse frequency instead of treating one collapsed resample as a
    # reason to discard the candidate's results from every other resample.
    final_cluster_counts = [
        _n_unique_or_zero(d.get("final_labels", [])[i])
        for d in label_dicts
    ]
    final_degenerate_fraction = float(np.mean(np.asarray(final_cluster_counts) <= 1))
    view_cluster_counts = []
    view_degenerate_fractions = []
    for v in range(n_views):
        counts_v = [
            _n_unique_or_zero(d.get("view_labels", [])[i][v])
            for d in label_dicts
        ]
        view_cluster_counts.append(counts_v)
        view_degenerate_fractions.append(float(np.mean(np.asarray(counts_v) <= 1)))
    candidate_params = candidate_params if isinstance(candidate_params, dict) else {}
    requested_final_k = candidate_params.get("k_final")
    requested_view_ks = list(candidate_params.get("k_s", []))
    final_effective_k_summary = summarize_effective_k(
        final_cluster_counts,
        requested_k=requested_final_k,
    )
    view_effective_k_summaries = [
        summarize_effective_k(
            counts,
            requested_k=requested_view_ks[v] if v < len(requested_view_ks) else None,
        )
        for v, counts in enumerate(view_cluster_counts)
    ]

    boot_dicts_final = [
        {"orig_ids": d["orig_ids"], "labels": d["final_labels"][i]}
        for d in label_dicts
    ]
    # Final-cluster stability
    # ARI is always computed as the primary stability metric
    final_stab_ari = ari_stability_common_subjects(
        boot_dicts_final,
        label_key="labels",
        precomputed_alignment=precomputed_alignment
    )

    # Additional stability metrics
    #final_stab_coassoc, final_stab_CCC = coassociation_stability(boot_dicts_final, label_key="labels")
    final_stab_jaccard = jaccard_stability_common_subjects(
        boot_dicts_final,
        label_key="labels",
        precomputed_alignment=precomputed_alignment
    )
    final_stab_SUM_MAT = consensus_pac_ccc(
        boot_dicts_final,
        label_key="labels",
        return_consensus=False,
        return_ecdf=False,
        precomputed_cache=precomputed_consensus_cache,
    )


    # ARI/Jaccard can regard repeated one-cluster solutions as perfectly stable.
    # Penalize in proportion to collapse frequency; candidates that collapse in
    # every resample still receive zero stability.
    final_nondegenerate_fraction = 1.0 - final_degenerate_fraction
    final_stab_ari *= final_nondegenerate_fraction
    final_stab_jaccard *= final_nondegenerate_fraction

    # Per-view stability
    view_stabs_ari = []
    #view_stabs_CCC = []
    #view_stabs_coassoc = []
    view_stabs_jaccard = []
    view_stabs_SUM_MAT = []

    for v in range(n_views):
        boot_dicts_view = [
            {"orig_ids": d["orig_ids"], "labels": d["view_labels"][i][v]}
            for d in label_dicts
        ]

        # ARI always computed per view (primary stability)
        stab_v_ari = ari_stability_common_subjects(
            boot_dicts_view,
            label_key="labels",
            precomputed_alignment=precomputed_alignment
        )

        #stab_v_coassoc, stab_v_CCC = coassociation_stability(boot_dicts_view, label_key="labels")
        stab_v_jaccard = jaccard_stability_common_subjects(
            boot_dicts_view,
            label_key="labels",
            precomputed_alignment=precomputed_alignment
        )
        stab_v_SUM_MAT = consensus_pac_ccc(
            boot_dicts_view,
            label_key="labels",
            return_consensus=False,
            return_ecdf=False,
            precomputed_cache=precomputed_consensus_cache,
        )

        view_nondegenerate_fraction = 1.0 - view_degenerate_fractions[v]
        stab_v_ari *= view_nondegenerate_fraction
        stab_v_jaccard *= view_nondegenerate_fraction

        #view_stabs_coassoc.append(float(stab_v_coassoc))
        #view_stabs_CCC.append(float(stab_v_CCC))
        view_stabs_ari.append(float(stab_v_ari))
        view_stabs_jaccard.append(float(stab_v_jaccard))
        view_stabs_SUM_MAT.append({
            "PAC": stab_v_SUM_MAT.get("PAC", np.nan),
            "CCC": stab_v_SUM_MAT.get("CCC", np.nan),
            "meta": stab_v_SUM_MAT.get("meta", {}),
        })

    # --- Quality ---
    has_final_q = all("final_scores" in d for d in label_dicts)
    has_view_q = all("view_scores_per_view" in d for d in label_dicts)

    # Per-view quality: mean across bootstraps
    view_quals = []
    if has_view_q:
        for v in range(n_views):
            # Collapsed resamples already contribute quality 0 at clustering time.
            q_v = np.mean([float(d["view_scores_per_view"][i][v]) for d in label_dicts])
            view_quals.append(float(q_v))
    else:
        view_quals = [0.0] * n_views

    mean_final_q = float(np.mean([float(d["final_scores"][i]) for d in label_dicts])) if has_final_q else 0.0

    mean_view_stab_ari = np.mean(view_stabs_ari) if view_stabs_ari else 0.0
    mean_view_qual = np.mean(view_quals) if view_quals else 0.0
    min_view_stab_ari = float(np.min(view_stabs_ari)) if view_stabs_ari else 0.0
    min_view_qual = float(np.min(view_quals)) if view_quals else 0.0

    # Also compute mean coassociation- and Jaccard-based stability for reporting
    #mean_view_stab_coassoc = np.mean(view_stabs_coassoc) if view_stabs_coassoc else 0.0
    #mean_view_stab_CCC = np.mean(view_stabs_CCC) if view_stabs_CCC else 0.0
    mean_view_stab_jaccard = np.mean(view_stabs_jaccard) if view_stabs_jaccard else 0.0
    if view_stabs_SUM_MAT:
        mean_view_stab_MAT_CCC = float(np.nanmean([d.get("CCC", np.nan) for d in view_stabs_SUM_MAT]))
        mean_view_stab_MAT_PAC = float(np.nanmean([d.get("PAC", np.nan) for d in view_stabs_SUM_MAT]))
    else:
        mean_view_stab_MAT_CCC = 0.0
        mean_view_stab_MAT_PAC = 0.0

    fitness_record = {
        "ind_id": i,
        "view_stabs_ari": view_stabs_ari,
        #"view_stabs_coassoc": view_stabs_coassoc,
        #"view_stabs_CCC": view_stabs_CCC,
        "view_stabs_jaccard": view_stabs_jaccard,
        "view_stabs_SUM_MAT": view_stabs_SUM_MAT,
        "view_quals": view_quals,
        "final_stab_ari": final_stab_ari,
        #"final_stab_coassoc": final_stab_coassoc,
        #"final_stab_CCC": final_stab_CCC,
        "final_stab_jaccard": final_stab_jaccard,
        "final_qual": mean_final_q,
        "final_degenerate_fraction": final_degenerate_fraction,
        "view_degenerate_fractions": view_degenerate_fractions,
        "final_cluster_counts": final_cluster_counts,
        "view_cluster_counts": view_cluster_counts,
        "final_effective_k_summary": final_effective_k_summary,
        "view_effective_k_summaries": view_effective_k_summaries,
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
        "mean_view_stability_ari": float(mean_view_stab_ari),
        "final_stability_ari": float(final_stab_ari),
        "min_view_stability_ari": float(min_view_stab_ari),
        "view_stabs_ari": tuple(view_stabs_ari),

        # Quality metrics
        "mean_view_quality": float(mean_view_qual),
        "view_quals": tuple(view_quals),
        "final_quality": float(mean_final_q),
        "min_view_quality": float(min_view_qual),
        "final_degenerate_fraction": float(final_degenerate_fraction),
        "view_degenerate_fractions": tuple(view_degenerate_fractions),
        "final_cluster_counts": tuple(final_cluster_counts),
        "view_cluster_counts": tuple(tuple(counts) for counts in view_cluster_counts),
        "final_effective_k_summary": final_effective_k_summary,
        "view_effective_k_summaries": tuple(view_effective_k_summaries),

        # Additional stability flavours for reporting
        #"mean_view_stability_coassoc": float(mean_view_stab_coassoc),
        #"view_stabs_coassoc": tuple(view_stabs_coassoc),
        #"mean_view_stability_CCC": float(mean_view_stab_CCC),
        #"view_stabs_CCC": tuple(view_stabs_CCC),
        #"final_stability_coassoc": float(final_stab_coassoc),
        #"final_stability_CCC": float(final_stab_CCC),
        "mean_view_stability_jaccard": float(mean_view_stab_jaccard),
        "view_stabs_jaccard": tuple(view_stabs_jaccard),
        "final_stability_jaccard": float(final_stab_jaccard),
        "mean_view_stability_MAT_CCC": float(mean_view_stab_MAT_CCC),
        "mean_view_stability_MAT_PAC": float(mean_view_stab_MAT_PAC),
        "view_stabs_SUM_MAT": tuple(view_stabs_SUM_MAT),
        "final_stability_SUM_MAT": {
            "PAC": final_stab_SUM_MAT.get("PAC", np.nan),
            "CCC": final_stab_SUM_MAT.get("CCC", np.nan),
            "meta": final_stab_SUM_MAT.get("meta", {}),
        }
    }
    values = tuple(summary[obj] for obj in objectives)
    return values, tuple(view_stabs_ari), tuple(view_quals), summary



# Modes

def do_bootstrap(args):
    """
    Run one GA bootstrap/subsample job.

    Inputs
    ------
    - A fold/generation population file.
    - The raw input and metadata CSVs.

    Work performed
    --------------
    1. Select the outer-fold training data.
    2. Draw one bootstrap or subsample from that training data.
    3. Preprocess and align modalities on the resample.
    4. Build per-modality representations.
    5. Cluster every GA candidate and save labels plus quality scores.

    Output
    ------
    A labels_*.pkl file consumed by do_gather.
    """

    # Resolve all paths up front. Wrapper scripts may pass either absolute paths
    # or paths relative to --base_dir.
    base_dir = os.path.abspath(getattr(args, "base_dir", "."))
    print(
        "[Bootstrap config] "
        f"internal_ensemble_enabled={getattr(args, 'internal_ensemble_enabled', 'FALSE')}, "
        f"internal_ensemble_bcs={getattr(args, 'internal_ensemble_bcs', 5)}, "
        f"internal_ensemble_sample_frac={getattr(args, 'internal_ensemble_sample_frac', 0.8)}, "
        f"internal_ensemble_feature_frac={getattr(args, 'internal_ensemble_feature_frac', 1.0)}"
    )
    if args.fold_index is None:
        raise ValueError("For bootstrap mode, --fold_index must be specified")
    ga_root = _ga_root(base_dir, args.fold_index)
    population_file = _resolve_path(base_dir, args.population_file)
    population_initial_file = _resolve_path(base_dir, args.population_initial_file)
    output_labels_path = _resolve_path(base_dir, args.output_labels)
    if population_file is None and args.generation is not None:
        population_file = os.path.join(ga_root, f"population_fold{args.fold_index}_gen{args.generation}.pkl")
    if population_initial_file is None:
        population_initial_file = os.path.join(ga_root, f"population_init_fold{args.fold_index}.pkl")
    if output_labels_path is None:
        gen_dir = os.path.join(ga_root, f"gen{args.generation or 0}", f"bootstrap_{args.bootstrap_index or 0}")
        output_labels_path = os.path.join(gen_dir, f"labels_{args.bootstrap_index or 0}.pkl")

    # Deterministic seed namespace for this GA bootstrap. A single seed is reused
    # across sampling, dimensionality reduction, and clustering for this job so
    # the label file is reproducible from fold/bootstrap identifiers.
    boot_index = getattr(args, "bootstrap_index", 0)
    if boot_index is None:
        boot_index = 0
    seed = _ga_bootstrap_seed(args.fold_index, boot_index)
    _seed_everything(seed)


    # Ensure DEAP’s creator classes exist before unpickling the population
    if args.optimisation == 'multi':
        _ensure_multi_fitness_class(args)
    else:
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    # Load the raw data and recreate the same outer-CV training split used by
    # later outer mode. Bootstrap jobs must never see the held-out fold.
    df = pd.read_csv(args.input_csv)
    meta = pd.read_csv(args.meta_csv)
    if args.n_folds == 1:
        # No CV split: use all rows for training to allow fast synthetic-data tests
        train_df = df.reset_index(drop=True)
    else:
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        train_idx, _ = list(kf.split(df))[args.fold_index]
        train_df = df.iloc[train_idx].reset_index(drop=True)



    # Load the candidate population that this resample will evaluate. Generation
    # 1 uses the init population; later generations use the previous gather
    # output.
    if args.generation == 1:
        # Load initial population from args.population_initial_file
        if not population_initial_file:
            raise ValueError(" --population_initial_file must be specified")
        with open(population_initial_file, 'rb') as f:
            population = dill.load(f)
    else:
        print(f"Loading generation {args.generation} population from {population_file}")
        # Otherwise load the specified generation
        if args.generation is None:
            raise ValueError("For bootstrap mode, --generation must be specified")
        with open(population_file, 'rb') as f:
            population = dill.load(f)

    # Pickled DEAP individuals can carry stale fitness classes from earlier
    # modes. Reset them here because bootstrap mode only needs candidate genes,
    # not previous fitness values.
    if args.optimisation == 'multi':
        multi_cls = _get_multi_fitness_class(args)
        if multi_cls is None:
            multi_cls = _ensure_multi_fitness_class(args)
        for ind in population:
            ind.fitness = multi_cls()
    else:
        for ind in population:
            ind.fitness = creator.FitnessMax()

    # --- Bootstrap / subsampling selection ---
    # Standard bootstrap keeps replacement duplicates. Subsample mode is the
    # current default for stability estimation and draws 80% without replacement.
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
            attempt_seed = _derive_seed("ga_bootstrap_subsample_attempt", seed, attempt)
            bdf = train_df.sample(frac=frac, replace=False, random_state=attempt_seed).reset_index(drop=True)
            bdf = bdf.reset_index(drop=True)
            if len(bdf.drop_duplicates(subset=args.subject_id_column)) >= 3:
                break
        else:
            raise RuntimeError(f"Failed to create a valid subsample after 100 attempts.")
    # Preserve the original ID for stability comparisons, but give every row a
    # unique processed ID so duplicate bootstrap rows are not collapsed during
    # preprocessing/alignment.
    bdf["orig_subject_id"] = bdf[args.subject_id_column]
    bdf["proc_subject_id"] = np.arange(len(bdf))

    # Preprocess once per resample. All candidates then share the same aligned
    # representation so fitness differences reflect clustering parameters, not
    # data-processing noise within this bootstrap.
    try:
        print("Start running Preprocessing")
        t_prep_start = time.time()
        ae_data, subject_id_list, dict_final = preprocessing(
            bdf, meta,
            subject_id_column='proc_subject_id',
            col_threshold=args.col_threshold,
            row_threshold=args.row_threshold,
            skew_threshold=args.skew_threshold,
            scaler_type=args.scaler_type,
            modalities=args.modalities,
            dummy_code_modalities=args.dummy_code_modalities,
            mixed_categorical_modalities=args.mixed_categorical_modalities
        )

        # Rebuild subject lists from dict_final after enforcing a canonical
        # modality order. This avoids false bootstrap skips if preprocessing
        # returned stale ordering metadata while the actual modality frames are
        # alignable.
        present_modalities = [mod for mod in args.modalities if mod in dict_final]
        missing_modalities = [mod for mod in args.modalities if mod not in dict_final]
        if missing_modalities:
            raise RuntimeError(
                "Missing modalities after preprocessing: "
                + ", ".join(missing_modalities)
            )
        id_lists = {
            mod: dict_final[mod]['proc_subject_id'].tolist()
            for mod in present_modalities
        }
        shared_proc_ids = set.intersection(*(set(ids) for ids in id_lists.values()))
        canonical_proc_ids = [sid for sid in id_lists[present_modalities[0]] if sid in shared_proc_ids]
        if not canonical_proc_ids:
            raise RuntimeError("No shared subjects remain across modalities after preprocessing.")
        for mod in present_modalities:
            dict_final[mod] = (
                dict_final[mod][dict_final[mod]['proc_subject_id'].isin(shared_proc_ids)]
                .set_index('proc_subject_id')
                .loc[canonical_proc_ids]
                .reset_index()
            )
        subject_id_list = [
            dict_final[mod]['proc_subject_id'].tolist()
            for mod in args.modalities
        ]
        for mod, ids in zip(args.modalities[1:], subject_id_list[1:]):
            if ids != subject_id_list[0]:
                raise RuntimeError(
                    "Subject-ID order mismatch across modalities after realignment: "
                    f"{args.modalities[0]} n={len(subject_id_list[0])}, "
                    f"{mod} n={len(ids)}"
                )
        # Convert back from processed row IDs to original subject IDs. These IDs
        # are what gather mode uses to compare labels across different resamples.
        kept_proc_ids = subject_id_list[0]  # same order as final embeddings/labels
        # Map proc -> orig using bdf
        proc_to_orig = dict(zip(bdf["proc_subject_id"], bdf["orig_subject_id"]))
        kept_orig_ids = [proc_to_orig[p] for p in kept_proc_ids]


        t_prep_end = time.time()
        print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] Preprocessing took {t_prep_end - t_prep_start:.2f}s")

        _seed_everything(seed)
        t_dimred_start = time.time()
        ae_res, data_list = _run_dimensionality_reduction(
            dict_final=dict_final,
            modalities=args.modalities,
            subject_id_column='proc_subject_id',
            modality_dim_reduction=args.dim_reduction_by_modality,
            pca_variance_threshold=args.pca_variance_threshold,
            snmf_n_components=args.snmf_n_components,
            snmf_alpha=args.snmf_alpha,
            snmf_l1_ratio=args.snmf_l1_ratio,
            snmf_max_iter=args.snmf_max_iter,
            hidden_dims=args.hidden_dims,
            activation_functions=activation_functions,
            learning_rates=args.learning_rates,
            batch_sizes=args.batch_sizes,
            latent_dims=args.latent_dims,
            random_state=seed,
        )
        t_dimred_end = time.time()
        if any(method == "vae" for method in args.dim_reduction_by_modality.values()):
            print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] VAE nested CV took {t_dimred_end - t_dimred_start:.2f}s")
        if any(method == "ae" for method in args.dim_reduction_by_modality.values()):
            print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] AE nested CV took {t_dimred_end - t_dimred_start:.2f}s")
        del ae_res
        gc.collect()

        reference_n = int(len(df))
        current_n = int(len(kept_orig_ids))
        args.mincluster_n_applied = _operational_min_cluster_n(args, current_n, reference_n)

        print("Start running Parea on bootstrap sample...")
        # Re-seed immediately before clustering so candidate evaluation is stable
        # even if dimensionality reduction consumed random numbers.
        _seed_everything(seed)
        # Check GA individual gene names
        #print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] Sample gene_names from first 3 individuals: {[ind.gene_names for ind in population[:3]]}")
        t_clust_start = time.time()
        # Evaluate every candidate independently. Workers share read-only data via
        # the initializer to avoid repeatedly serializing the modality matrices.
        n_workers = args.n_jobs or (os.cpu_count() or 1)
        #print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] using {n_workers} workers for clustering")
        chunksize = max(1, len(population) // (n_workers * 4) if n_workers else len(population))
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(data_list, subject_id_list, args)
        ) as executor:
            # Map candidates; workers reuse shared read-only inputs
            all_results = list(executor.map(partial(_cluster_candidate, args=args), population, chunksize=chunksize))

        t_clust_end = time.time()
        print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] "
              f"Clustering {len(population)} candidates took {t_clust_end - t_clust_start:.2f}s")

        # Ensure lengths of IDs and labels align
        if len(all_results) > 0:
            assert len(kept_orig_ids) == len(all_results[0][0]), "orig_ids and final_labels length mismatch"
        # Output shape convention:
        # - final_labels[candidate] -> final fused labels for this resample.
        # - view_labels[candidate][view] -> labels before fusion for one view.
        # - orig_ids aligns labels back to original subject IDs, including any
        #   duplicate IDs created by bootstrap resampling.
        final_labels = [res[0] for res in all_results]
        view_labels  = [res[1] for res in all_results]
        # Prepare output dict for quality scores (mean over views already in res[2])
        view_scores_per_view = [res[2] for res in all_results]
        view_scores_mean = [res[3] for res in all_results]
        final_scores = [res[4] for res in all_results]

        # Compute a bottleneck per-view quality score for each candidate. The
        # minimum helps identify candidates that look good on average but fail in
        # one modality.
        # Composite quality per view: average of three normalized indices in [0,1]:
        #  - Silhouette (normalized from [-1,1] to [0,1])
        #  - Calinski–Harabasz normalized via ch/(ch+1)
        #  - Davies–Bouldin transformed via 1/(1+db) (higher is better)
        def _per_view_composite_qualities(X_list, indiv_labs):
            """Handle per view composite qualities."""
            vals = []
            for X, labs in zip(X_list, indiv_labs):
                labs = np.asarray(labs)
                if len(np.unique(labs)) <= 1:
                    vals.append(0.0)
                    continue
                try:
                    sil = silhouette_score(X, labs)           # [-1,1]
                    sil_n = (sil + 1.0) / 2.0                 # [0,1]
                except Exception:
                    sil_n = 0.0
                try:
                    ch = calinski_harabasz_score(X, labs)     # [0, +inf)
                    ch_n = ch / (ch + 1.0)                    # (0,1)
                except Exception:
                    ch_n = 0.0
                try:
                    db = davies_bouldin_score(X, labs)        # [0, +inf), lower is better
                    db_inv = 1.0 / (1.0 + db)                 # (0,1], higher is better
                except Exception:
                    db_inv = 0.0
                cq = (sil_n + ch_n + db_inv) / 3.0
                vals.append(float(cq))
            return vals

        view_scores_min = []
        for cand_indiv_labels in view_labels:
            quals = _per_view_composite_qualities(data_list, cand_indiv_labels)
            view_scores_min.append(float(np.min(quals)) if len(quals) else 0.0)
        #proc_ids_list = [res[2] for res in all_results]
        # Ensure all proc_id_lists are identical
        #if not all(proc_ids_list[0] == pid for pid in proc_ids_list):
        #    warnings.warn("Inconsistent subject-ID lists across cluster candidates")
        #proc_ids = proc_ids_list[0]

        to_dump = {
            "orig_ids":        kept_orig_ids,
            "final_labels":    final_labels,
            "view_labels":     view_labels,
            "view_scores_mean": view_scores_mean,
            "view_scores_per_view": view_scores_per_view,
            "view_scores_min":  view_scores_min,
            "final_scores":    final_scores,
            "requested_params": [convert_to_parameters(len(args.modalities), ind) for ind in population],
            "mincluster_n_requested": int(args.mincluster_n),
            "mincluster_n_applied": int(args.mincluster_n_applied),
            "reference_n": reference_n,
            "current_n": current_n,
        }

        # Persist the only artifact gather needs from this bootstrap job.
        os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)
        with open(output_labels_path, 'wb') as f:
            dill.dump(to_dump, f)
        print(f"Bootstrap labels {args.bootstrap_index} saved to {output_labels_path}")
        return
    except Exception as e:
        # Sentinel on bootstrap failure: write an empty labels dict and exit 0 so
        # the array job can continue. Gather mode enforces the minimum usable
        # bootstrap count and fails there if too many sentinels were produced.
        try:
            os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)
            sentinel = {
                "orig_ids": [],
                "final_labels": [],
                "view_labels": [],
                "view_scores_mean": [],
                "view_scores_per_view": [],
                "view_scores_min": [],
                "final_scores": []
            }
            with open(output_labels_path, 'wb') as f:
                dill.dump(sentinel, f)
            print(f"[Fold {args.fold_index}] Bootstrap {args.bootstrap_index} marked as SKIPPED due to error: {e}")
        except Exception as ee:
            print(f"[Fold {args.fold_index}] Failed to write sentinel for bootstrap {args.bootstrap_index}: {ee}")
        return


def do_gather(args):
    """
    Score one GA generation and create the next generation.

    Inputs
    ------
    - All labels_*.pkl files from do_bootstrap for this fold/generation.
    - The population evaluated by those bootstrap jobs.

    Work performed
    --------------
    1. Drop failed/sentinel bootstrap files.
    2. Compute candidate stability across resamples and combine with quality.
    3. Attach objective values and reporting summaries to each individual.
    4. Update the fold hall of fame/Pareto front.
    5. Breed the next generation with elitism.

    Output
    ------
    population_fold*_gen*.pkl for the next bootstrap round, plus fold-level
    fitness/view history files.
    """
    # Resolve all stage inputs/outputs. Gather has several defaults because it
    # is called once per fold/generation by scheduler wrappers.
    base_dir = os.path.abspath(getattr(args, "base_dir", "."))
    if args.fold_index is None:
        raise ValueError("For gather mode, --fold_index must be specified")
    ga_root = _ga_root(base_dir, args.fold_index)
    bootstrap_dir = _resolve_path(base_dir, args.bootstrap_dir)
    population_dir = _resolve_path(base_dir, args.population_dir) if args.population_dir else None
    population_file = _resolve_path(base_dir, args.population_file) if args.population_file else None
    population_initial_file = _resolve_path(base_dir, args.population_initial_file) if args.population_initial_file else None
    output_population = _resolve_path(base_dir, args.output_population) if args.output_population else None
    if population_dir is None and args.generation is not None:
        population_dir = os.path.join(ga_root, f"gen{args.generation}")
    if population_file is None and args.generation is not None:
        population_file = os.path.join(ga_root, f"population_fold{args.fold_index}_gen{args.generation}.pkl")
    if population_initial_file is None:
        population_initial_file = os.path.join(ga_root, f"population_init_fold{args.fold_index}.pkl")
    if output_population is None and args.generation is not None:
        output_population = os.path.join(ga_root, f"population_fold{args.fold_index}_gen{args.generation + 1}.pkl")

    if args.fold_index is None:
        raise ValueError("For gather mode, --fold_index must be specified")
    if args.generation is None:
        raise ValueError("For gather mode, --generation must be specified")
    if not bootstrap_dir:
        raise ValueError("For gather mode, --bootstrap_dir must be specified")
    if not population_dir:
        raise ValueError("For gather mode, --population_dir must be specified")

    # Deterministic seed namespace for parent selection, crossover, and mutation.
    boot_index = getattr(args, "bootstrap_index", 0)
    if boot_index is None:
        boot_index = 0
    seed = _derive_seed("ga_gather", int(args.fold_index or 0))
    _seed_everything(seed)

    hof_dir = ga_root
    fitness_cache_dir = None
    if args.generation is not None:
        fitness_cache_dir = os.path.join(
            hof_dir,
            "fitness_cache",
            f"gen{args.generation:03d}"
        )

    # Ensure fitness classes exist before unpickling populations/HOF objects.
    # DEAP stores class names in pickles, so missing creator classes break loads.
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

    files = sorted(glob.glob(pattern), key=numeric_boot_dirs)

    if not files:
        raise FileNotFoundError(f"No label files found in {bootstrap_dir}; expected bootstrap_*/labels_*.pkl")

    # Each bootstrap file contains candidate labels and quality scores saved by
    # do_bootstrap. Empty sentinel files mark failed resamples and are excluded
    # from stability estimation below.
    label_dicts_all = [dill.load(open(fn, 'rb')) for fn in files]
    def _usable(d):
        """Handle usable."""
        return isinstance(d, dict) and len(d.get("final_labels", [])) > 0 and len(d.get("orig_ids", [])) > 0
    label_dicts = [d for d in label_dicts_all if _usable(d)]

    min_needed = max(1, args.n_bootstrap - 5)
    if len(label_dicts) < min_needed:
        raise RuntimeError(f"Only {len(label_dicts)} usable bootstraps (min required {min_needed}). Check for sentinel/failed runs in {bootstrap_dir}.")

    # Precompute ID alignments once. Fitness scoring calls pairwise stability many
    # times, so this avoids repeated subject-intersection work for every candidate.
    precomputed_alignment = precompute_bootstrap_pair_alignment(label_dicts)
    precomputed_consensus_cache = precompute_consensus_cache(label_dicts)

    # Unpack lists across bootstraps
    # final_label_sets: list of lists, one per bootstrap, each list of length pop_size
    final_label_sets = [d["final_labels"] for d in label_dicts]
    # Evaluate the GA objectives candidate-by-candidate. A candidate's
    # stability is defined by how consistently its labels reproduce across the
    # usable bootstrap/subsample runs.
    pop_size = len(final_label_sets[0])
    requested_params = label_dicts[0].get("requested_params", [])
    if len(requested_params) != pop_size:
        requested_params = [None] * pop_size
    n_workers = args.n_jobs or (os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(
            _compute_fitness_for_ind,
            range(pop_size),
            repeat(label_dicts),
            repeat(args.modalities),
            repeat(tuple(args.ga_objectives)),
            repeat(fitness_cache_dir),
            repeat(args.fold_index),
            repeat(boot_index),
            repeat(precomputed_alignment),
            repeat(precomputed_consensus_cache),
            requested_params,
        ))
        fitness = [tuple(map(float, res[0])) for res in results]
        per_view_stabs = [tuple(map(float, res[1])) for res in results]
        per_view_quals = [tuple(map(float, res[2])) for res in results]
        summary_metrics = [res[3] for res in results]
    # Persist raw fitness tuples for traceability across generations. This is
    # separate from the hall of fame because it records the whole population.
    os.makedirs(hof_dir, exist_ok=True)
    history_file = os.path.join(hof_dir, "fitness_history.pkl")
    # Ensure the directory exists before writing the history file
    os.makedirs(hof_dir, exist_ok=True)
    if os.path.exists(history_file):
        with open(history_file, "rb") as hf:
            fitness_history = pickle.load(hf)
    else:
        fitness_history = {}
    # Record this generation's fitness tuples
    fitness_history[args.generation] = fitness
    with open(history_file, "wb") as hf:
        pickle.dump(fitness_history, hf)

    if args.generation == 1:
        # Load initial population from args.population_initial_file
        if not args.population_initial_file:
            raise ValueError("For gather mode, --population_initial_file must be specified")
        with open(population_initial_file, 'rb') as f:
            population = dill.load(f)
    elif args.generation > 1:
        # Otherwise load the specified generation
        if args.generation is None:
            raise ValueError("For gather mode, --generation must be specified")
        with open(population_file, 'rb') as f:
            population = dill.load(f)

    # Reattach gene names in the same order used by init. Some DEAP operations and
    # pickling paths preserve list contents but not custom metadata reliably.
    n_views = len(args.modalities)
    gene_names = []
    for i in range(1, n_views + 1):
        gene_names.append(f"c_{i}_k")
        gene_names.append(f"c_{i}_method")
    gene_names.append("pre_method")
    gene_names.append("k_final")
    gene_names.append("fusion_method")
    for ind in population:
        ind.gene_names = gene_names

    # Ensure DEAP’s creator classes exist before unpickling the population
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
        for ind in population:
            if not isinstance(ind.fitness, multi_cls):
                ind.fitness = multi_cls()
    else:
        for ind in population:
            ind.fitness = creator.FitnessMax()

    # Clear any stale fitness values to avoid mismatches
    for ind in population:
        try:
            del ind.fitness.values
        except AttributeError:
            pass


    # Attach objective values and reporting summaries to the evaluated
    # population. Later outer mode uses these summaries to write fold metrics.
    if args.optimisation == 'single':
        # Single-objective: fitness tuples contain only final stability
        stab_view_key, stab_final_key, qual_view_key, qual_final_key = _primary_metric_keys(args)
        for idx, (final_stab,) in enumerate(fitness):
            ind = population[idx]
            ind.fitness.values = (final_stab,)
            ind.view_stabs_per_view = per_view_stabs[idx]
            ind.view_quals_per_view = per_view_quals[idx]
            summary = summary_metrics[idx]
            ind.metrics_summary = summary
            ind.mean_view_stab = summary.get(stab_view_key)
            ind.mean_view_qual = summary.get(qual_view_key)
            ind.final_stab = summary.get(stab_final_key)
            ind.final_qual = summary.get(qual_final_key)
    elif args.optimisation == 'multi':
        stab_view_key, stab_final_key, qual_view_key, qual_final_key = _primary_metric_keys(args)
        for idx, ind in enumerate(population):
            fitvals = tuple(map(float, fitness[idx]))
            assert len(fitvals) == len(ind.fitness.weights), (
                f"Mismatch: {len(fitvals)} values vs {len(ind.fitness.weights)} weights"
            )
            ind.fitness.values = fitvals
            ind.view_stabs_per_view = per_view_stabs[idx]
            ind.view_quals_per_view = per_view_quals[idx]
            summary = summary_metrics[idx]
            ind.metrics_summary = summary
            ind.mean_view_stab = summary.get(stab_view_key)
            ind.mean_view_qual = summary.get(qual_view_key)
            ind.final_stab = summary.get(stab_final_key)
            ind.final_qual = summary.get(qual_final_key)
    else:
        raise ValueError(f"Unknown optimisation mode: {args.optimisation}")


    # ---------------- Generate-only GA step (ask/tell style) ----------------
    # At this point the current population has been scored. The code below only
    # creates children for the next generation; those children are evaluated by
    # the next wave of do_bootstrap jobs.
    # Record statistics over the evaluated current population
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

    cur_stats = stats.compile(population)
    print("[Gather] Current-pop stats:", cur_stats)
    if args.optimisation == 'multi' and per_view_stabs and len(per_view_stabs[0]) > 0:
        stab_matrix = np.array(per_view_stabs, dtype=float)
        qual_matrix = np.array(per_view_quals, dtype=float)
        summary_rows = []
        avg_stabs_per_mod = []
        avg_quals_per_mod = []

        def _metric_matrix_from_sum_mat(metric):
            """Handle metric matrix from sum mat."""
            rows = []
            for m in summary_metrics:
                sum_mat = m.get("view_stabs_SUM_MAT")
                if sum_mat:
                    rows.append([d.get(metric, np.nan) if isinstance(d, dict) else np.nan for d in sum_mat])
                else:
                    rows.append([np.nan] * len(args.modalities))
            return np.array(rows, dtype=float)

        def _nanmax_axis0_or_nan(matrix):
            """Handle nanmax axis0 or nan."""
            matrix = np.asarray(matrix, dtype=float)
            out = np.full(matrix.shape[1], np.nan, dtype=float)
            finite_cols = np.any(np.isfinite(matrix), axis=0)
            if np.any(finite_cols):
                out[finite_cols] = np.nanmax(matrix[:, finite_cols], axis=0)
            return out

        # Build per-modality diagnostic matrices from the summaries. These are
        # not used for breeding directly, but are saved so weak modalities can be
        # inspected after each generation.
        coassoc_matrix = np.array(
            [m.get("view_stabs_coassoc", [np.nan] * len(args.modalities)) for m in summary_metrics],
            dtype=float
        )
        ccc_matrix = _metric_matrix_from_sum_mat("CCC")
        jaccard_matrix = np.array(
            [m.get("view_stabs_jaccard", [np.nan] * len(args.modalities)) for m in summary_metrics],
            dtype=float
        )

        for idx, mod in enumerate(args.modalities):
            avg_stab = float(np.mean(stab_matrix[:, idx]))
            avg_qual = float(np.mean(qual_matrix[:, idx]))
            avg_stabs_per_mod.append((mod, avg_stab))
            avg_quals_per_mod.append((mod, avg_qual))
            summary_rows.append(f"{mod}: mean_stab={avg_stab:.3f}, mean_qual={avg_qual:.3f}")
        min_stab_mod, min_stab_val = min(avg_stabs_per_mod, key=lambda x: x[1])
        min_qual_mod, min_qual_val = min(avg_quals_per_mod, key=lambda x: x[1])
        print(
            "[Gather] Per-view means -> "
            + " | ".join(summary_rows)
            + f" || min_mean_stab={min_stab_val:.3f} ({min_stab_mod}), "
              f"min_mean_qual={min_qual_val:.3f} ({min_qual_mod})"
        )
        # Persist per-modality best stability/quality for this generation
        best_stab_per_mod_ari = np.max(stab_matrix, axis=0)
        best_stab_per_mod_coassoc = _nanmax_axis0_or_nan(coassoc_matrix)
        best_stab_per_mod_ccc = _nanmax_axis0_or_nan(ccc_matrix)
        best_stab_per_mod_jaccard = _nanmax_axis0_or_nan(jaccard_matrix)

        best_qual_per_mod = np.max(qual_matrix, axis=0)

        view_hist_path = os.path.join(hof_dir, "view_history.pkl")
        if os.path.exists(view_hist_path):
            with open(view_hist_path, "rb") as vh:
                view_history = pickle.load(vh)
        else:
            view_history = {"modalities": args.modalities, "generations": {}}
        view_history["generations"][args.generation] = {
            # ARI-based best per modality (primary stability)
            "best_stab": best_stab_per_mod_ari.tolist(),
            # Additional metrics for diagnostics
            "best_stab_coassoc": best_stab_per_mod_coassoc.tolist(),
            "best_stab_ccc": best_stab_per_mod_ccc.tolist(),
            "best_stab_jaccard": best_stab_per_mod_jaccard.tolist(),
            "best_qual": best_qual_per_mod.tolist()
        }
        with open(view_hist_path, "wb") as vh:
            pickle.dump(view_history, vh)

    # Persistent hall of fame across all generations in this fold. Multi-objective
    # runs keep the Pareto front; single-objective runs keep the best individual.
    hof_path = os.path.join(hof_dir, "halloffame.pkl")
    if os.path.exists(hof_path):
        with open(hof_path, 'rb') as f:
            hall_of_fame = dill.load(f)
    else:
        hall_of_fame = tools.ParetoFront() if args.optimisation == 'multi' else tools.HallOfFame(maxsize=1)

    # Update and persist Hall of Fame / Pareto front with the evaluated population
    hall_of_fame.update(population)
    with open(hof_path, 'wb') as f:
        dill.dump(hall_of_fame, f)

    # Breeding operators. Mutation changes exactly one gene and samples from the
    # same valid search space used during initialization.
    toolbox = base.Toolbox()
    # GA mutation indices for layout: [c_1_k, c_1_method, ..., c_V_k, c_V_method, pre_method, k_final, fusion_method]
    n_views = len(args.modalities)
    k_positions = [2*i for i in range(n_views)]
    linkage_positions = [2*i + 1 for i in range(n_views)]
    pre_linkage_index = 2 * n_views
    k_final_index = 2 * n_views + 1
    f_index = 2 * n_views + 2

    fusion_methods = list(args.fusion_methods)
    linkages = list(args.linkages)
    if not fusion_methods:
        raise ValueError("At least one fusion method must be supplied via --fusion_methods.")
    #linkages = ['complete','average','weighted']

    def mutate(individual):
        """Handle mutate."""
        idx = random.randint(0, len(individual) - 1)
        if idx in k_positions:
            individual[idx] = random.randint(args.k_min, args.k_max)
        elif idx in linkage_positions:
            individual[idx] = random.choice(linkages)
        elif idx == pre_linkage_index:
            individual[idx] = random.choice(linkages)
        elif idx == k_final_index:
            individual[idx] = random.randint(args.k_min, args.k_max)
        elif idx == f_index:
            individual[idx] = random.choice(fusion_methods)
        return (individual,)

    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutate)
    if args.optimisation == 'single':
        toolbox.register("select_parents", tools.selTournament, tournsize=3)
    else:
        toolbox.register("select_parents", tools.selNSGA2)

    # Select parents and create children for the next generation. These children
    # are intentionally unevaluated until the next bootstrap wave.
    parents = toolbox.select_parents(population, k=len(population))
    offspring = algorithms.varOr(
        parents,
        toolbox,
        lambda_=len(population),
        cxpb=args.ga_cxpb,
        mutpb=args.ga_mutpb
    )

    # Preserve gene names for all offspring
    for ind in offspring:
        ind.gene_names = population[0].gene_names

    # Ensure children have empty fitness so the bootstrap stage will evaluate them
    for ind in offspring:
        try:
            del ind.fitness.values
        except AttributeError:
            pass
        if hasattr(population[0], 'gene_names') and not hasattr(ind, 'gene_names'):
            ind.gene_names = population[0].gene_names

    # Validate fitness tuple lengths before elitism. This catches mismatches
    # between requested GA objectives and DEAP fitness classes early.
    if args.optimisation == 'multi':
        expected_len = len(args.ga_objectives)
        if len(population) != len(fitness):
            raise RuntimeError(f"Population size ({len(population)}) != fitness list size ({len(fitness)}).")
        for idx, ind in enumerate(population):
            vals = getattr(ind.fitness, 'values', ())
            if len(vals) != expected_len:
                try:
                    fitvals = fitness[idx]
                except Exception:
                    raise RuntimeError(f"Missing or invalid fitness for individual {idx}: {fitness[idx] if idx < len(fitness) else 'N/A'}")
                ind.fitness.values = tuple(map(float, fitvals))
    else:
        # Single-objective must have length-1 tuples
        if len(population) != len(fitness):
            raise RuntimeError(f"Population size ({len(population)}) != fitness list size ({len(fitness)}).")
        for idx, ind in enumerate(population):
            vals = getattr(ind.fitness, 'values', ())
            if len(vals) != 1:
                try:
                    (fs,) = fitness[idx]
                except Exception:
                    raise RuntimeError(f"Missing or invalid single fitness for individual {idx}: {fitness[idx] if idx < len(fitness) else 'N/A'}")
                ind.fitness.values = (float(fs),)

    # Elitism carries the strongest already-evaluated individuals forward so a
    # good solution cannot be lost through crossover/mutation.
    elite_count = max(1, min(args.ga_elitism, len(population)))
    if args.optimisation == 'single':
        elites = tools.selBest(population, k=elite_count)
    else:
        elites = tools.selNSGA2(population, k=elite_count)

    # Build next generation: elites + (offspring trimmed to fill the rest)
    slots = max(0, len(population) - len(elites))
    next_population = list(elites) + list(offspring[:slots])

    # Also preserve gene_names on HOF individuals
    try:
        for ind in hall_of_fame:
            ind.gene_names = gene_names
    except Exception:
        pass

    # Save the newly generated population for the next generation of bootstraps
    out_path = output_population
    if not population_dir:
        raise ValueError("For gather mode, --population_dir must be specified")
    if not out_path:
        raise ValueError("For gather mode, --output_population must be specified")
    os.makedirs(population_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, 'wb') as f:
        dill.dump(next_population, f)
    print(f"Generated next population (elitism={elite_count}) saved to {out_path}")
    return


def do_outer(args):
    """
    Fit and evaluate the selected GA solution on one outer fold.

    Inputs
    ------
    - The fold hall of fame written by do_gather.
    - The raw input and metadata CSVs.

    Work performed
    --------------
    1. Recreate the outer train/test split.
    2. Preprocess and reduce dimensions using training data only.
    3. Rank hall-of-fame candidates and choose the first one that produces at
       least two final clusters on the training fold.
    4. Save labels, candidate parameters, latent representations, and fitness
       summaries for merge mode.

    Output
    ------
    results/fold*/metrics.pkl.
    """
    # Resolve fold-specific inputs/outputs. Outer mode consumes the fold hall of
    # fame and writes a compact metrics file for merge mode.
    base_dir = os.path.abspath(getattr(args, "base_dir", "."))
    if args.fold_index is None:
        raise ValueError("For outer mode, --fold_index must be specified")
    ga_root = _ga_root(base_dir, args.fold_index)
    population_file = _resolve_path(base_dir, args.population_file) if args.population_file else None
    if population_file is None:
        population_file = os.path.join(ga_root, f"population_fold{args.fold_index}_gen{args.generation or 0}.pkl")
    output_metrics_path = _resolve_path(base_dir, args.output_metrics) if args.output_metrics else None
    output_metrics_merged_path = output_metrics_path

    if not population_file:
        raise ValueError("For outer mode, --population_file must be specified")
    if not output_metrics_path:
        raise ValueError("For outer mode, --output_metrics must be specified")

    # Deterministic seed namespace for this outer fold.
    boot_index = getattr(args, "bootstrap_index", 0)
    if boot_index is None:
        boot_index = 0
    seed = _derive_seed("outer_fold", int(args.fold_index or 0), int(args.generation or 0), int(boot_index or 0))
    _seed_everything(seed)

    # Recreate the outer train/test split exactly. Only the train split is used
    # for representation fitting and fold-level candidate selection.
    df = pd.read_csv(args.input_csv)
    meta = pd.read_csv(args.meta_csv)
    if args.n_folds == 1:
        # Train on the full dataset; leave test empty for fast single-fold runs
        train_df = df.reset_index(drop=True)
        test_df  = df.iloc[0:0].copy()
        # Create dummy indices for downstream ID capture
        train_idx = train_df.index.tolist()
        test_idx  = []
    else:
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        # Extract train and test indices for this outer fold
        train_idx, test_idx = list(kf.split(df))[args.fold_index]
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx].reset_index(drop=True)

    # Fit preprocessing on the training split only. This keeps outer-fold metrics
    # honest and mirrors how the selected candidate would perform without seeing
    # held-out subjects.
    ae_data, subject_id_list, dict_final = preprocessing(
        train_df, meta,
        subject_id_column=args.subject_id_column,
        col_threshold=args.col_threshold,
        row_threshold=args.row_threshold,
        skew_threshold=args.skew_threshold,
        scaler_type=args.scaler_type,
        modalities=args.modalities,
        dummy_code_modalities=args.dummy_code_modalities,
        mixed_categorical_modalities=args.mixed_categorical_modalities
    )

    # Assert identical subject order across modalities after preprocessing
    base_ids = dict_final[args.modalities[0]][args.subject_id_column].tolist()
    for m in args.modalities[1:]:
        assert dict_final[m][args.subject_id_column].tolist() == base_ids, \
            f"Subject-ID order mismatch between {args.modalities[0]} and {m} after preprocessing"

    # Use a fixed representation seed for fold-level candidate comparison so
    # differences between hall-of-fame candidates come from clustering parameters.
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    ae_res, _ = _run_dimensionality_reduction(
        dict_final=dict_final,
        modalities=args.modalities,
        subject_id_column=args.subject_id_column,
        modality_dim_reduction=args.dim_reduction_by_modality,
        pca_variance_threshold=args.pca_variance_threshold,
        snmf_n_components=args.snmf_n_components,
        snmf_alpha=args.snmf_alpha,
        snmf_l1_ratio=args.snmf_l1_ratio,
        snmf_max_iter=args.snmf_max_iter,
        hidden_dims=args.hidden_dims,
        activation_functions=activation_functions,
        learning_rates=args.learning_rates,
        batch_sizes=args.batch_sizes,
        latent_dims=args.latent_dims,
        random_state=42,
    )

    # Ensure DEAP classes exist before loading the fold hall of fame.
    if args.optimisation == 'multi':
        _ensure_multi_fitness_class(args)
    if args.optimisation == 'single' and not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # Load the persistent Hall-of-Fame for this fold and take its champion
    ga_root = os.path.dirname(population_file)
    hof_path = os.path.join(ga_root, "halloffame.pkl")
    with open(hof_path, 'rb') as f:
        hall_of_fame = dill.load(f)

    # Select a fold champion. Multi-objective candidates are first ordered by
    # distance to the ideal normalized objective vector, then tested on the
    # actual training fold to avoid collapsed one-cluster final solutions.
    # Prepare latent representations per modality
    ae_cluster = {mod: ae_res[mod]['final_latent'] for mod in args.modalities}
    data_list = [ae_cluster[mod] for mod in args.modalities]
    # Build the candidate list from the hall of fame. Multi-objective runs are
    # sorted below; single-objective runs already contain the best individual(s).
    if args.optimisation == 'single':
        candidates = list(hall_of_fame)
    else:
        candidates = list(hall_of_fame)
    # For multi-objective: sort candidates by distance to ideal point
    if args.optimisation == 'multi':
        vals = np.array([ind.fitness.values for ind in candidates], dtype=float)
        # Normalize objectives to [0,1] across candidates (avoid div-by-zero)
        mins, maxs = vals.min(axis=0), vals.max(axis=0)
        rng = np.where(maxs > mins, maxs - mins, 1.0)
        norm = (vals - mins) / rng
        d = np.linalg.norm(1.0 - norm, axis=1)
        order = np.argsort(d)  # ascending distance to ideal
        candidates = [candidates[i] for i in order]

    # --- DIAGNOSTICS (TEST mode): evaluate top-K Pareto candidates on ARI and save CSV ---
    if getattr(args, 'TEST', 'FALSE').upper() == "TRUE":
        try:
            K = min(20, len(candidates))
            eval_rows = []
            df_truth = pd.read_csv("path/to/multiclust/synthetic_multimodal_spartan.csv")
            truth_map = df_truth.set_index(args.subject_id_column)
            ref_ids = dict_final[args.modalities[0]][args.subject_id_column].to_numpy()
            true_cols = [
                truth_map.loc[ref_ids, f"subgroup_m{i+1}"].to_numpy() for i in range(len(args.modalities))
            ]
            true_ints_list = []
            for arr in true_cols:
                uniq = np.unique(arr)
                l2i = {name: idx for idx, name in enumerate(uniq)}
                true_ints_list.append(np.array([l2i[v] for v in arr], dtype=int))

            diag_dir = ga_root
            os.makedirs(diag_dir, exist_ok=True)
            for rank, ind in enumerate(candidates[:K]):
                params = convert_to_parameters(len(args.modalities), ind)
                labels, indiv_labels, view_scores_per_view, view_q, final_q = parea_2_mv(
                    data_list,
                    **params,
                    subject_id_list=subject_id_list,
                    inner_jobs=args.n_jobs,
                    pre_inner_jobs=args.n_jobs,
                    mincluster=args.mincluster,
                    mincluster_n=args.mincluster_n,
                    internal_ensemble_enabled=args.internal_ensemble_enabled,
                    internal_ensemble_bcs=args.internal_ensemble_bcs,
                    internal_ensemble_sample_frac=args.internal_ensemble_sample_frac,
                    internal_ensemble_feature_frac=args.internal_ensemble_feature_frac,
                    internal_ensemble_seed=int(getattr(args, "fold_index", 0) or 0)
                )
                aris = []
                for i, pred in enumerate(indiv_labels):
                    pred = np.asarray(pred, dtype=int)
                    ari = adjusted_rand_score(true_ints_list[i], pred)
                    aris.append(float(ari))
                row = {"rank": rank, "fitness": ind.fitness.values}
                if args.optimisation == 'multi':
                    for name, val in zip(args.ga_objectives, ind.fitness.values):
                        row[name] = float(val)
                else:
                    row["final_stability"] = float(ind.fitness.values[0])
                # attach ARIs per modality
                for i, mod in enumerate(args.modalities):
                    row[f"ARI_{mod}"] = aris[i]
                # attach gene params
                if hasattr(ind, 'gene_names'):
                    for gname, gval in zip(ind.gene_names, list(ind)):
                        row[gname] = gval
                eval_rows.append(row)
            eval_df = pd.DataFrame(eval_rows)
            csv_path = os.path.join(diag_dir, f"pareto_eval_top{K}.csv")
            eval_df.to_csv(csv_path, index=False)
            print(f"[Fold {args.fold_index}] Pareto diagnostics written to {csv_path}")
            # Quick console summary of best ARI per view
            for i, mod in enumerate(args.modalities):
                best_idx = int(np.nanargmax(eval_df[f"ARI_{mod}"].to_numpy()))
                best_row = eval_df.iloc[best_idx]
                print(f"[Diag] Best ARI_{mod}={best_row[f'ARI_{mod}']:.3f} at rank {best_row['rank']}, params subset: k_s={[best_row.get(f'c_{j+1}_k') for j in range(len(args.modalities))]}, k_final={best_row.get('k_final')}, fusion={best_row.get('fusion_method')}")
        except Exception as e:
            print(f"[Fold {args.fold_index}] WARNING: Pareto diagnostics failed: {e}")

    # Re-evaluate sorted candidates on the full training fold. Bootstrap fitness
    # penalizes collapsed solutions, but minimum-size enforcement can still make a
    # candidate collapse on this final fold-level fit.
    if not candidates:
        raise RuntimeError(f"[Fold {args.fold_index}] Hall of fame did not contain any candidates.")

    def _fit_candidate_on_training_fold(candidate):
        """Fit candidate on training fold."""
        candidate_params = convert_to_parameters(len(args.modalities), candidate)
        fold_mincluster_n = _operational_min_cluster_n(args, len(base_ids), len(df))
        candidate_result = parea_2_mv(
            data_list,
            subject_id_list=subject_id_list,
            inner_jobs=args.n_jobs,
            pre_inner_jobs=args.n_jobs,
            mincluster=args.mincluster,
            mincluster_n=fold_mincluster_n,
            internal_ensemble_enabled=args.internal_ensemble_enabled,
            internal_ensemble_bcs=args.internal_ensemble_bcs,
            internal_ensemble_sample_frac=args.internal_ensemble_sample_frac,
            internal_ensemble_feature_frac=args.internal_ensemble_feature_frac,
            internal_ensemble_seed=int(getattr(args, "fold_index", 0) or 0),
            **candidate_params
        )
        return candidate_params, candidate_result

    ind = None
    params = None
    selected_fit = None
    for rank, candidate in enumerate(candidates):
        candidate_params, candidate_fit = _fit_candidate_on_training_fold(candidate)
        candidate_labels = np.asarray(candidate_fit[0])
        candidate_quality = candidate_fit[4]
        if (
            candidate_labels.size > 0
            and len(np.unique(candidate_labels)) >= 2
            and candidate_quality is not None
            and np.isfinite(candidate_quality)
            and candidate_quality > 0
        ):
            ind = candidate
            params = candidate_params
            selected_fit = candidate_fit
            if rank > 0:
                print(
                    f"[Fold {args.fold_index}] Selected candidate rank {rank} because "
                    "higher-ranked candidate(s) collapsed to fewer than 2 final clusters."
                )
            break

    if selected_fit is None:
        raise RuntimeError(
            f"[Fold {args.fold_index}] No candidate produced at least two final clusters "
            "with positive quality on the training fold."
        )

    labels, indiv_labels, view_scores_per_view, view_score_mean, final_score = selected_fit

    best = ind
    best_params = params
    train_final_labels = labels
    train_individual_labels = indiv_labels
    train_view_scores_per_view = view_scores_per_view
    train_view_quality_mean = view_score_mean
    train_final_quality = final_score

    # Pull the fitness summaries saved during gather onto the fold metrics
    # payload. These are cross-bootstrap estimates, while the labels below come
    # from this final fold-level fit.
    summary = getattr(ind, "metrics_summary", {})
    final_effective_k_summary = summary.get("final_effective_k_summary")
    view_effective_k_summaries = list(summary.get("view_effective_k_summaries", ()))
    fallback_reasons = []
    if not isinstance(final_effective_k_summary, dict) or final_effective_k_summary.get("selected_k") is None:
        fallback_reasons.append("missing_final_effective_k_summary")
        final_effective_k_summary = summarize_effective_k([], requested_k=params.get("k_final"))
        final_effective_k_summary["selected_k"] = int(params.get("k_final"))
        final_effective_k_summary["fallback_reason"] = "missing_effective_k_summary"
    if len(view_effective_k_summaries) != len(args.modalities):
        fallback_reasons.append("missing_view_effective_k_summaries")
        view_effective_k_summaries = []
        for requested_k in params.get("k_s", []):
            fallback = summarize_effective_k([], requested_k=requested_k)
            fallback["selected_k"] = int(requested_k)
            fallback["fallback_reason"] = "missing_effective_k_summary"
            view_effective_k_summaries.append(fallback)
    if fallback_reasons:
        warnings.warn(
            f"[Fold {args.fold_index}] Effective-k fallback: {', '.join(fallback_reasons)}."
        )
    best_params_requested = dict(params)
    best_params_effective = dict(best_params_requested)
    best_params_effective["k_final"] = int(final_effective_k_summary["selected_k"])
    best_params_effective["k_s"] = [
        int(item["selected_k"]) for item in view_effective_k_summaries
    ]
    best_params_alias = (
        best_params_effective
        if _flag_enabled(getattr(args, "use_effective_k_for_fold_merge", "FALSE"))
        else best_params_requested
    )
    stab_view_key, stab_final_key, qual_view_key, qual_final_key = _primary_metric_keys(args)
    mean_view_stab = summary.get(stab_view_key)
    mean_view_qual = summary.get(qual_view_key)
    final_stab = summary.get(stab_final_key)
    final_qual = summary.get(qual_final_key)

    # Additional stability flavours for output
    #mean_view_stab_coassoc = summary.get("mean_view_stability_coassoc")
    #mean_view_stab_ccc = summary.get("mean_view_stability_CCC")
    mean_view_stab_jaccard = summary.get("mean_view_stability_jaccard")
    #final_stab_coassoc = summary.get("final_stability_coassoc")
    #final_stab_ccc = summary.get("final_stability_CCC")
    final_stab_jaccard = summary.get("final_stability_jaccard")
    mean_view_stability_MAT_CCC = summary.get("mean_view_stability_MAT_CCC")
    mean_view_stability_MAT_PAC = summary.get("mean_view_stability_MAT_PAC")
    final_stability_SUM_MAT = summary.get("final_stability_SUM_MAT", {})
    # Per-view MATLAB-style (lightweight) diagnostics from gather
    view_stabs_SUM_MAT = summary.get("view_stabs_SUM_MAT", [])
    # Normalize to a plain Python list for dill/pickle friendliness
    view_stabs_SUM_MAT = list(view_stabs_SUM_MAT) if view_stabs_SUM_MAT is not None else []



    if args.optimisation == 'single' and final_stab is None:
        final_stab = float(ind.fitness.values[0])
        mean_view_stab = final_stab
    view_stabs = getattr(ind, "view_stabs_per_view", None)
    view_quals = getattr(ind, "view_quals_per_view", None)


    metrics_dir = os.path.dirname(output_metrics_path) or "."

    # Save chosen individual's stabilities if available
    gen = getattr(args, "generation", 0)
    chosen = ind

    if hasattr(chosen, "view_stabs_per_view") and chosen.view_stabs_per_view:
        np.save(
            os.path.join(metrics_dir, f"chosen_view_stabs_gen{gen}.npy"),
            np.array(chosen.view_stabs_per_view, dtype=float)
        )

    # (Optional) Warn if bottleneck per-view quality is low
    if args.optimisation == 'multi' and view_quals:
        try:
            min_view_qual = float(np.min(view_quals))
            if min_view_qual < 0.3:
                print(f"[Fold {args.fold_index}] WARNING: bottleneck per-view quality is low (min view qual={min_view_qual:.3f}).")
        except Exception:
            pass


    # Capture original train subject IDs before modality filtering
    train_ids = df.loc[train_idx, args.subject_id_column].tolist()
    # Also capture test IDs (may be empty when n_folds == 1)
    test_ids = df.loc[test_idx, args.subject_id_column].tolist() if len(test_idx) > 0 else []

    #If in TEST mode, test label accuracy with true labels from synthetic data (aligned by subject IDs)
    if getattr(args, 'TEST', 'FALSE').upper() == "TRUE":
        print("TEST mode: computing Adjusted Rand Index against ground truth labels (aligned by subject IDs).")
        df_truth = pd.read_csv("path/to/multiclust/synthetic_multimodal_spartan.csv")
        truth_map = df_truth.set_index(args.subject_id_column)

        # Expanded ARI diagnostics per modality
        for i, pred in enumerate(train_individual_labels):
            mod = args.modalities[i]
            ref_ids_i = dict_final[mod][args.subject_id_column].to_numpy()
            true_labels = truth_map.loc[ref_ids_i, f"subgroup_m{i+1}"].to_numpy()

            # Map true labels (possibly strings) to stable integers
            uniq = np.unique(true_labels)
            l2i = {v: k for k, v in enumerate(uniq)}
            true_ints = np.array([l2i[v] for v in true_labels], dtype=int)
            pred = np.asarray(pred, dtype=int)

            # Safety: ensure lengths match
            if true_ints.shape[0] != pred.shape[0]:
                print(f"[DEBUG] Length mismatch for {mod}: true={true_ints.shape[0]} pred={pred.shape[0]}")

            # ARI
            ari = adjusted_rand_score(true_ints, pred)
            print(f"Adjusted Rand Index modality {i} ({mod}): {ari:.3f}")

            # --- Extra diagnostics ---
            try:
                # Show first 10 rows for sanity
                preview = list(zip(ref_ids_i[:10], true_ints[:10].tolist(), pred[:10].tolist()))
                print(f"[DEBUG] First 10 (id, true, pred) for {mod}: {preview}")

                # Confusion table (true vs pred)
                conf = pd.crosstab(pd.Series(true_ints, name='true'),
                                   pd.Series(pred, name='pred'),
                                   dropna=False)
                print(f"[DEBUG] Confusion table for {mod}:\n{conf}")
            except Exception as e:
                print(f"[DEBUG] Could not print diagnostics for {mod}: {e}")


    # Save the fold payload consumed by merge. It contains enough information to
    # select cross-fold parameters without reloading bootstrap label files.
    os.makedirs(metrics_dir, exist_ok=True)
    # Per-view lists from summary (all stability types)
    view_stabs_ari = summary.get("view_stabs_ari")
    #view_stabs_coassoc = summary.get("view_stabs_coassoc")
    #view_stabs_ccc = summary.get("view_stabs_CCC")
    view_stabs_jaccard = summary.get("view_stabs_jaccard")
    view_stabs_SUM_MAT = summary.get("view_stabs_SUM_MAT", [])


    # Map primary mean-view key to its corresponding per-view key
    primary_views_stab = None
    if stab_view_key == "mean_view_stability_ari":
        primary_views_stab = view_stabs_ari
    #elif stab_view_key == "mean_view_stability_coassoc":
    #    primary_views_stab = view_stabs_coassoc
    #elif stab_view_key == "mean_view_stability_CCC":
    #    primary_views_stab = view_stabs_ccc
    elif stab_view_key == "mean_view_stability_jaccard":
        primary_views_stab = view_stabs_jaccard
    else:
        # Fallback to ARI per-view stability
        primary_views_stab = view_stabs_ari

    view_stabs_list = list(primary_views_stab) if primary_views_stab is not None else None
    view_quals_list = list(view_quals) if view_quals else None

    best_fitness_payload = {
        # Primary stability measures (aligned with GA objectives)
        'mean_view_stability': mean_view_stab,
        'final_stability': final_stab,

        # All stability variants for reporting
        'mean_view_stability_ari': summary.get("mean_view_stability_ari"),
        'final_stability_ari': summary.get("final_stability_ari"),
        #'mean_view_stability_coassoc': mean_view_stab_coassoc,
        #'final_stability_coassoc': final_stab_coassoc,
        #'mean_view_stability_CCC': mean_view_stab_ccc,
        #'final_stability_CCC': final_stab_ccc,
        'mean_view_stability_jaccard': mean_view_stab_jaccard,
        'final_stability_jaccard': final_stab_jaccard,
        # MATLAB-style stability diagnostics (from consensus_pac_ccc during GA evaluation)
        'mean_view_stability_MAT_CCC': mean_view_stability_MAT_CCC,
        'mean_view_stability_MAT_PAC': mean_view_stability_MAT_PAC,
        'final_stability_SUM_MAT': final_stability_SUM_MAT,
        'views_stability_SUM_MAT': view_stabs_SUM_MAT,

        # Per-view stability/quality (primary plus full breakdowns)
        'views_stability': view_stabs_list,
        'views_quality': view_quals_list,
        'views_stability_ari': list(view_stabs_ari) if view_stabs_ari is not None else None,
        #'views_stability_coassoc': list(view_stabs_coassoc) if view_stabs_coassoc is not None else None,
        #'views_stability_CCC': list(view_stabs_ccc) if view_stabs_ccc is not None else None,
        'views_stability_jaccard': list(view_stabs_jaccard) if view_stabs_jaccard is not None else None,
        'views_stability_SUM_MAT': list(view_stabs_SUM_MAT) if view_stabs_SUM_MAT is not None else None,

        # Collapse diagnostics across the search bootstraps
        'final_degenerate_fraction': summary.get('final_degenerate_fraction'),
        'view_degenerate_fractions': list(summary.get('view_degenerate_fractions', ())),
        'final_cluster_counts': list(summary.get('final_cluster_counts', ())),
        'view_cluster_counts': [
            list(counts) for counts in summary.get('view_cluster_counts', ())
        ],
        'final_effective_k_summary': final_effective_k_summary,
        'view_effective_k_summaries': view_effective_k_summaries,

        # Quality measures
        'mean_view_quality': mean_view_qual,
        'final_quality': final_qual,

        # Existing view-wise quality and final composite quality metric
        'view_scores_per_view': view_scores_per_view,
        'final_quality_metric': final_score
    }
    metrics = {
        'metrics_schema_version': METRICS_SCHEMA_VERSION,
        'data': dict_final,
        'ae_res': ae_res,
        'train_final_labels': train_final_labels,
        'train_individual_labels': train_individual_labels,
        'best_fitness': best_fitness_payload,
        'train_ids': train_ids,
        'test_ids': test_ids,
        'best_params': best_params_alias,
        'best_params_requested': best_params_requested,
        'best_params_effective': best_params_effective,
        'train_effective_k': int(len(np.unique(train_final_labels))),
        'train_view_effective_k': [int(len(np.unique(labels))) for labels in train_individual_labels],
        'effective_k_fallback_reasons': fallback_reasons,
        'mincluster_n_requested': int(args.mincluster_n),
        'mincluster_n_applied': int(_operational_min_cluster_n(args, len(base_ids), len(df))),
        'reference_n': int(len(df)),
        'current_n': int(len(base_ids)),
    }
    with open(output_metrics_path, 'wb') as f:
        dill.dump(metrics, f)
    print(f"Outer metrics saved to {output_metrics_path}")
    return



def do_merge(args):
    """
    Produce the final full-data clustering and reporting payload.

    Inputs
    ------
    - One results/fold*/metrics.pkl file from each outer fold.
    - The raw input and metadata CSVs.

    Work performed
    --------------
    1. Select final hyperparameters by cross-fold mode, with stability/quality
       tie-breaks.
    2. Preprocess all data and build final modality representations.
    3. Apply the selected PAREA/multi-view clustering parameters to all data.
    4. Estimate full-data stability using final bootstraps/subsamples.
    5. Package reporting diagnostics and optionally train SVM classifiers.

    Output
    ------
    The final metrics pickle requested by --output_final_metrics.
    """
    # Merge is the final production stage. It deliberately recomputes the full
    # data representation instead of reusing a fold representation.
    t_merge_start = time.time()
    base_dir = os.path.abspath(getattr(args, 'base_dir', "."))
    results_root = _output_root(base_dir, "RESULTS_DIR", "results")
    output_final_metrics_path = _resolve_path(base_dir, args.output_final_metrics) if args.output_final_metrics else None

    df = pd.read_csv(args.input_csv)
    meta = pd.read_csv(args.meta_csv)

    # Validate that every expected outer fold produced metrics. Missing folds
    # would bias the cross-fold parameter selection below.
    metrics_files = sorted(glob.glob(os.path.join(results_root, 'fold*', 'metrics.pkl')))
    expected_metrics_files = [
        os.path.join(results_root, f"fold{i}", "metrics.pkl")
        for i in range(int(args.n_folds))
    ]
    missing_metrics_files = [path for path in expected_metrics_files if not os.path.exists(path)]
    if missing_metrics_files:
        found_rel = [os.path.relpath(path, results_root) for path in metrics_files]
        missing_rel = [os.path.relpath(path, results_root) for path in missing_metrics_files]
        raise FileNotFoundError(
            "Merge requires one metrics.pkl per outer fold. "
            f"Found {len(metrics_files)}/{args.n_folds}: {found_rel}. "
            f"Missing: {missing_rel}"
        )

    # Load fold metrics into a dict keyed by fold name, e.g. fold0/fold1.
    metrics = {}
    for metrics_file in metrics_files:
        fold_name = os.path.basename(os.path.dirname(metrics_file))  # e.g., 'fold0'
        with open(metrics_file, 'rb') as f:
            metrics[fold_name] = pickle.load(f)

    schema_versions = {payload.get("metrics_schema_version", 1) for payload in metrics.values()}
    if len(schema_versions) > 1:
        warnings.warn(
            f"Mixed fold metrics schema versions detected: {sorted(schema_versions)}. "
            "Missing effective-k summaries will fall back to requested values."
        )

    # --- Cross-fold parameter selection ---
    # Choose final clustering parameters by agreement across outer folds. When
    # folds tie on a parameter value, use the relevant fold stability/quality as
    # the tie-breaker instead of arbitrary lexical order.
    # Collect parameters from folds. If the modal value is tied across folds,
    # prefer the tied value whose source folds had better relevant stability,
    # then quality. This keeps reproducibility ahead of cluster separation.
    fold_names = list(metrics.keys())
    def _finite_mean(values):
        """Handle finite mean."""
        vals = [float(v) for v in values if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else np.nan

    def _fold_part_metrics(fold_payload, part="final", view_index=None):
        """Return the stability/quality pair relevant to one parameter choice."""
        bf = fold_payload.get("best_fitness", {})

        def _first_finite(candidates):
            """Handle first finite."""
            for val in candidates:
                if val is not None and np.isfinite(val):
                    return float(val)
            return np.nan

        if part == "view":
            stabs = (
                bf.get("views_stability")
                or bf.get("views_stability_ari")
                or bf.get("per_view_stabilities")
                or []
            )
            quals = bf.get("views_quality") or bf.get("view_scores_per_view") or []
            stability = np.nan
            quality = np.nan
            if view_index is not None:
                if view_index < len(stabs):
                    stability = stabs[view_index]
                if view_index < len(quals):
                    quality = quals[view_index]
            return {
                "stability": float(stability) if np.isfinite(stability) else np.nan,
                "quality": float(quality) if np.isfinite(quality) else np.nan,
            }

        return {
            "stability": _first_finite([bf.get("final_stability"), bf.get("final_stability_ari")]),
            "quality": _first_finite([bf.get("final_quality"), bf.get("final_quality_metric")]),
        }

    def _fallback_value_key(value):
        """Handle fallback value key."""
        if isinstance(value, (int, float, np.integer, np.floating)):
            return (0, float(value))
        return (1, str(value))

    def _assign_multiobjective_distance(rows):
        """Score stability and quality jointly by distance to their ideal point."""
        for count in sorted({row["count"] for row in rows}, reverse=True):
            group = [row for row in rows if row["count"] == count]
            for metric in ("stability", "quality"):
                finite = [row[metric] for row in group if np.isfinite(row[metric])]
                metric_min = min(finite) if finite else 0.0
                metric_max = max(finite) if finite else 0.0
                for row in group:
                    value = row[metric]
                    if not np.isfinite(value):
                        normalized = 0.0
                    elif np.isclose(metric_max, metric_min):
                        normalized = 1.0
                    else:
                        normalized = (value - metric_min) / (metric_max - metric_min)
                    row[f"normalized_{metric}"] = float(normalized)
            for row in group:
                row["multiobjective_distance"] = float(np.linalg.norm([
                    1.0 - row["normalized_stability"],
                    1.0 - row["normalized_quality"],
                ]))
        return rows

    def _rank_parameter_values(col, values, view_index=None):
        """Rank parameter values."""
        part = "view" if col in ("k_s", "linkage") else "final"
        unique_values = []
        for value in values:
            if value not in unique_values:
                unique_values.append(value)

        ranked = []
        for value in unique_values:
            fold_indices = [i for i, observed in enumerate(values) if observed == value]
            fold_scores = [
                _fold_part_metrics(metrics[fold_names[i]], part=part, view_index=view_index)
                for i in fold_indices
            ]
            ranked.append({
                "value": value,
                "count": len(fold_indices),
                "folds": [fold_names[i] for i in fold_indices],
                "stability": _finite_mean([score["stability"] for score in fold_scores]),
                "quality": _finite_mean([score["quality"] for score in fold_scores]),
            })

        ranked = _assign_multiobjective_distance(ranked)
        return sorted(
            ranked,
            key=lambda row: (
                -row["count"],
                row["multiobjective_distance"],
                _fallback_value_key(row["value"]),
            ),
        )

    use_effective_fold_params = _flag_enabled(
        getattr(args, "use_effective_k_for_fold_merge", "FALSE")
    )
    param_key = "best_params_effective" if use_effective_fold_params else "best_params_requested"
    param_list = [
        metrics[fold_name].get(param_key) or metrics[fold_name]["best_params"]
        for fold_name in fold_names
    ]
    param_df = pd.DataFrame(param_list)
    parameter_slots = []
    param_selection = {}
    for col in param_df.columns:
        series = param_df[col]
        if series.apply(lambda value: isinstance(value, (list, tuple))).any():
            expanded = pd.DataFrame(series.tolist())
            param_selection[col] = []
            for position in range(expanded.shape[1]):
                ranked = _rank_parameter_values(col, expanded[position].tolist(), view_index=position)
                parameter_slots.append({"column": col, "position": position, "ranked": ranked})
                param_selection[col].append(ranked)
        else:
            ranked = _rank_parameter_values(col, series.tolist())
            parameter_slots.append({"column": col, "position": None, "ranked": ranked})
            param_selection[col] = ranked

    def _params_from_component_ranks(component_ranks):
        """Handle params from component ranks."""
        params = {}
        for slot, value_rank in zip(parameter_slots, component_ranks):
            value = slot["ranked"][value_rank]["value"]
            col = slot["column"]
            position = slot["position"]
            if position is None:
                params[col] = value
            else:
                if col not in params:
                    width = 1 + max(
                        item["position"] for item in parameter_slots
                        if item["column"] == col and item["position"] is not None
                    )
                    params[col] = [None] * width
                params[col][position] = value
        return params

    # Best-first search over component rank combinations. The first candidate
    # is the original independently selected best value for every parameter.
    start_ranks = tuple(0 for _ in parameter_slots)
    queue = [(0, 0, start_ranks)]
    seen_rank_combinations = {start_ranks}
    final_param_candidates = []
    while queue and len(final_param_candidates) < 5:
        _, _, component_ranks = heapq.heappop(queue)
        final_param_candidates.append({
            "params": _params_from_component_ranks(component_ranks),
            "component_ranks": component_ranks,
        })
        for slot_index, slot in enumerate(parameter_slots):
            next_rank = component_ranks[slot_index] + 1
            if next_rank >= len(slot["ranked"]):
                continue
            neighbor = list(component_ranks)
            neighbor[slot_index] = next_rank
            neighbor = tuple(neighbor)
            if neighbor in seen_rank_combinations:
                continue
            seen_rank_combinations.add(neighbor)
            heapq.heappush(queue, (sum(neighbor), max(neighbor), neighbor))

    final_params = final_param_candidates[0]["params"]
    if use_effective_fold_params and not _flag_enabled(
        getattr(args, "use_cross_fold_effective_k_for_final_run", "FALSE")
    ):
        warnings.warn(
            "Effective k is enabled for fold merge but disabled for the final run; "
            "the selected component vector is still retained for compatibility."
        )
    print("Ranked component-wise cross-fold parameter combinations:")
    for rank, candidate in enumerate(final_param_candidates, 1):
        print(
            f"  rank {rank}: component_ranks={candidate['component_ranks']}, "
            f"params={candidate['params']}"
        )


    # --- Full-data representation ---
    # Refit preprocessing and dimensionality reduction on all subjects. Export
    # preprocessing details so downstream reports can audit dropped subjects,
    # feature columns, scaling, and imputation.
    export_preprocessing_details = True
    t_preprocess_start = time.time()
    ae_data, subject_id_list, dict_final, preprocessing_details = preprocessing(
        df, meta,
        subject_id_column=args.subject_id_column,
        col_threshold=args.col_threshold,
        row_threshold=args.row_threshold,
        skew_threshold=args.skew_threshold,
        scaler_type=args.scaler_type,
        modalities=args.modalities,
        dummy_code_modalities=args.dummy_code_modalities,
        mixed_categorical_modalities=args.mixed_categorical_modalities,
        export_preprocessing_details=export_preprocessing_details
    )
    t_preprocess_end = time.time()
    preprocessing_seconds = float(t_preprocess_end - t_preprocess_start)

    # Assert identical subject order across modalities after preprocessing
    base_ids = dict_final[args.modalities[0]][args.subject_id_column].tolist()
    for m in args.modalities[1:]:
        assert dict_final[m][args.subject_id_column].tolist() == base_ids, \
            f"Subject-ID order mismatch between {args.modalities[0]} and {m} after preprocessing"

    t_dimred_start = time.time()
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    ae_res, _ = _run_dimensionality_reduction(
        dict_final=dict_final,
        modalities=args.modalities,
        subject_id_column=args.subject_id_column,
        modality_dim_reduction=args.dim_reduction_by_modality,
        pca_variance_threshold=args.pca_variance_threshold,
        snmf_n_components=args.snmf_n_components,
        snmf_alpha=args.snmf_alpha,
        snmf_l1_ratio=args.snmf_l1_ratio,
        snmf_max_iter=args.snmf_max_iter,
        hidden_dims=args.hidden_dims,
        activation_functions=activation_functions,
        learning_rates=args.learning_rates,
        batch_sizes=args.batch_sizes,
        latent_dims=args.latent_dims,
        random_state=42,
    )
    t_dimred_end = time.time()
    dim_reduction_seconds = float(t_dimred_end - t_dimred_start)


    # Prepare latent representations per modality in the same order as
    # args.modalities. This order must match final_params and subject_id_list.
    ae_cluster = {mod: ae_res[mod]['final_latent'] for mod in args.modalities}
    data_list = [ae_cluster[mod] for mod in args.modalities]

    # Apply Parea with best parameters on full data
    #labels, indiv_labels, view_scores_per_view, view_score_mean, final_score = parea_2_mv(
    #    data_list,
    #    subject_id_list=subject_id_list,
    #    inner_jobs=args.n_jobs,
    #    pre_inner_jobs=args.n_jobs,
    #    mincluster=args.mincluster,
    #    mincluster_n=args.mincluster_n,
    #    **final_params
    #)

    view_scores_per_view = None
    view_score_mean = None
    final_score = None


    # --- Full-data stability resampling ---
    # Re-run the final selected parameters on resampled full data. The resulting
    # label dictionaries are used to estimate final and per-view stability.
    if not subject_id_list:
        raise ValueError("Subject ID list is empty; cannot perform stability estimation.")
    try:
        ref_subject_ids = next(ids for ids in subject_id_list if ids)
    except StopIteration:
        raise ValueError("No subject IDs available across modalities; cannot perform stability estimation.")
    full_subject_ids = ref_subject_ids
    n_samples = len(full_subject_ids)
    for ids in subject_id_list:
        if ids and len(ids) != n_samples:
            raise ValueError("Subject-ID lists per modality must have identical lengths for bootstrapping.")

    requested_final_bootstrap_preprocessing = str(
        getattr(args, 'final_bootstrap_preprocessing', 'outside')
    ).strip().lower()
    if requested_final_bootstrap_preprocessing not in ("outside", "inside", "both"):
        raise ValueError(
            "--final_bootstrap_preprocessing must be 'outside', 'inside', or 'both'."
        )
    final_bootstrap_preprocessing = (
        "inside" if requested_final_bootstrap_preprocessing == "both"
        else requested_final_bootstrap_preprocessing
    )
    print(
        "Final stability bootstrap preprocessing mode: "
        f"{requested_final_bootstrap_preprocessing} "
        f"(primary: {final_bootstrap_preprocessing})"
    )

    n_boot_full = getattr(args, 'n_bootstrap', 50)
    full_label_dicts_final = []
    full_label_dicts_views = [[] for _ in args.modalities]

    # Placeholders for aggregated full-data stability estimates
    full_final_stab_ari = None
    #full_final_stab_coassoc = None
    #full_final_stab_CCC = None
    full_final_stab_jaccard = None

    full_views_stab_ari = None
    #full_views_stab_coassoc = None
    #full_views_stab_CCC = None
    full_views_stab_jaccard = None

    # MATLAB-style consensus diagnostics (PAC/CCC) for full-data stability
    full_final_stab_SUM_MAT_full = None   # full dict from consensus_pac_ccc (may include matrix)
    full_v_stab_SUM_MAT_full = None        # lightweight dict: {PAC, CCC, meta}


    # Also keep the raw coassociation-based per-cluster lists if needed
    #full_final_coassoc = None         # list-of-floats per cluster (coassoc cluster stabilities)
    #full_views_coassoc = None         # list of lists, per view -> per cluster

    # Aggregate view-level MATLAB-style summary statistics (means across views)
    full_mean_view_stab_MAT_CCC = None
    full_mean_view_stab_MAT_PAC = None
    full_final_effective_k_summary = None
    full_view_effective_k_summaries = None



    def _run_bootstrap(seed_value):
        """Run one final stability resample under the selected preprocessing mode."""
        np.random.seed(seed_value)
        random.seed(seed_value)
        if torch is not None:
            torch.manual_seed(seed_value)

        frac = 0.8
        m = max(3, int(round(frac * n_samples)))    # ensure enough points
        rng = np.random.default_rng(seed_value)
        idx = rng.choice(n_samples, size=m, replace=False)

        if final_bootstrap_preprocessing == "inside":
            # Strict mode: rebuild preprocessing and dimensionality reduction
            # inside the resample. This captures preprocessing variability but is
            # much more expensive.
            boot_ids = [full_subject_ids[i] for i in idx]
            bdf = (
                df[df[args.subject_id_column].isin(boot_ids)]
                .set_index(args.subject_id_column)
                .loc[boot_ids]
                .reset_index()
            )
            ae_data_boot, subject_id_list_boot, dict_final_boot = preprocessing(
                bdf, meta,
                subject_id_column=args.subject_id_column,
                col_threshold=args.col_threshold,
                row_threshold=args.row_threshold,
                skew_threshold=args.skew_threshold,
                scaler_type=args.scaler_type,
                modalities=args.modalities,
                dummy_code_modalities=args.dummy_code_modalities,
                mixed_categorical_modalities=args.mixed_categorical_modalities
            )
            np.random.seed(seed_value)
            random.seed(seed_value)
            if torch is not None:
                torch.manual_seed(seed_value)
            ae_res_boot, data_list_boot = _run_dimensionality_reduction(
                dict_final=dict_final_boot,
                modalities=args.modalities,
                subject_id_column=args.subject_id_column,
                modality_dim_reduction=args.dim_reduction_by_modality,
                pca_variance_threshold=args.pca_variance_threshold,
                snmf_n_components=args.snmf_n_components,
                snmf_alpha=args.snmf_alpha,
                snmf_l1_ratio=args.snmf_l1_ratio,
                snmf_max_iter=args.snmf_max_iter,
                hidden_dims=args.hidden_dims,
                activation_functions=activation_functions,
                learning_rates=args.learning_rates,
                batch_sizes=args.batch_sizes,
                latent_dims=args.latent_dims,
                random_state=seed_value,
            )
            del ae_res_boot
            gc.collect()
        else:
            # Fast/default mode: sample rows from the fixed full-data
            # representations. This isolates clustering stability after the
            # final preprocessing model has been chosen.
            data_list_boot = [X[idx, :] for X in data_list]
            subject_id_list_boot = []
            for ids in subject_id_list:
                if not ids:
                    subject_id_list_boot.append([])
                else:
                    subject_id_list_boot.append([ids[i] for i in idx])

        boot_labels, boot_indiv_labels, _, _, _ = parea_2_mv(
            data_list_boot,
            subject_id_list=subject_id_list_boot,
            inner_jobs=args.n_jobs,
            pre_inner_jobs=args.n_jobs,
            mincluster=args.mincluster,
            mincluster_n=_operational_min_cluster_n(
                args,
                current_n=len(subject_id_list_boot[0]),
                reference_n=n_samples,
            ),
            internal_ensemble_enabled=args.internal_ensemble_enabled,
            internal_ensemble_bcs=args.internal_ensemble_bcs,
            internal_ensemble_sample_frac=args.internal_ensemble_sample_frac,
            internal_ensemble_feature_frac=args.internal_ensemble_feature_frac,
            internal_ensemble_seed=seed_value,
            **final_params
        )

        final_entry = {
            "orig_ids": subject_id_list_boot[0],
            "labels": boot_labels,
            "requested_k": int(final_params.get("k_final", 2)),
            "mincluster_n_requested": int(args.mincluster_n),
            "mincluster_n_applied": int(_operational_min_cluster_n(
                args, len(subject_id_list_boot[0]), n_samples
            )),
            "reference_n": int(n_samples),
            "current_n": int(len(subject_id_list_boot[0])),
        }
        per_view_entries = []
        requested_view_ks = list(final_params.get("k_s", []))
        for v in range(len(args.modalities)):
            per_view_entries.append({
                "orig_ids": subject_id_list_boot[v],
                "labels": boot_indiv_labels[v] if v < len(boot_indiv_labels) else [],
                "requested_k": int(requested_view_ks[v]) if v < len(requested_view_ks) else None,
                "mincluster_n_requested": int(args.mincluster_n),
                "mincluster_n_applied": int(_operational_min_cluster_n(
                    args, len(subject_id_list_boot[v]), n_samples
                )),
                "reference_n": int(n_samples),
                "current_n": int(len(subject_id_list_boot[v])),
            })
        return final_entry, per_view_entries

    seeds = [
        _derive_seed("final_stability_bootstrap", b, base=12345)
        for b in range(n_boot_full)
    ]

    raw_workers = getattr(args, 'bootstrap_jobs', None)
    if raw_workers is None:
        raw_workers = getattr(args, 'n_jobs', 1)
        if raw_workers in (-1, None):
            raw_workers = os.cpu_count() or 1
    bootstrap_workers = max(1, min(int(raw_workers), n_boot_full)) if n_boot_full > 0 else 1

    def _parallel_map_merge(func, items, workers, label, batch_size="auto"):
        """Map work with process workers and fall back to threads if needed."""
        items = list(items)
        if len(items) == 0:
            return []
        workers = max(1, min(int(workers), len(items)))
        if workers == 1:
            return [func(item) for item in items]
        print(f"Running {label} with {workers} process worker(s).")
        try:
            return Parallel(n_jobs=workers, prefer="processes", batch_size=batch_size)(
                delayed(func)(item) for item in items
            )
        except Exception as exc:
            warnings.warn(
                f"Process parallelism failed for {label} ({exc}); falling back to threads."
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                return list(executor.map(func, items))

    def _chunk_items(items, n_chunks):
        """Handle chunk items."""
        items = list(items)
        if not items:
            return []
        n_chunks = max(1, min(int(n_chunks), len(items)))
        base, extra = divmod(len(items), n_chunks)
        chunks = []
        start = 0
        for i in range(n_chunks):
            stop = start + base + (1 if i < extra else 0)
            if start < stop:
                chunks.append(items[start:stop])
            start = stop
        return chunks


    final_labels = None
    indiv_labels = None

    def _consume_bootstrap(result_iter):
        """Accumulate final stability bootstraps and derive consensus labels."""
        nonlocal full_label_dicts_final, full_label_dicts_views
        nonlocal full_final_stab_ari, full_final_stab_jaccard
        nonlocal full_views_stab_ari, full_views_stab_jaccard
        # nonlocal full_final_stab_coassoc, full_final_stab_CCC, full_views_stab_coassoc, full_views_stab_CCC
        nonlocal full_final_stab_SUM_MAT_full
        nonlocal full_v_stab_SUM_MAT_full
        nonlocal full_mean_view_stab_MAT_CCC, full_mean_view_stab_MAT_PAC
        nonlocal full_final_effective_k_summary, full_view_effective_k_summaries
        # nonlocal full_final_coassoc, full_views_coassoc
        nonlocal full_final_stab_SUM_MAT_full
        nonlocal final_labels, indiv_labels
        nonlocal view_scores_per_view, view_score_mean, final_score

        # First accumulate all bootstrap label dictionaries. Stability metrics
        # need the full set, so computation happens after this loop.
        for b, (final_entry, per_view_entries) in enumerate(result_iter, 1):
            if int(final_entry.get("requested_k", -1)) != int(final_params.get("k_final", 2)):
                raise RuntimeError("A full-data bootstrap used the wrong integrated requested k.")
            full_label_dicts_final.append(final_entry)
            for v, entry in enumerate(per_view_entries):
                requested_view_ks = list(final_params.get("k_s", []))
                expected_k = requested_view_ks[v] if v < len(requested_view_ks) else None
                if expected_k is not None and int(entry.get("requested_k", -1)) != int(expected_k):
                    raise RuntimeError(
                        f"A full-data bootstrap used the wrong requested k for view {v}."
                    )
                full_label_dicts_views[v].append(entry)

        full_final_effective_k_summary = summarize_effective_k(
            [len(np.unique(item.get("labels", []))) if len(item.get("labels", [])) else None
             for item in full_label_dicts_final],
            requested_k=final_params.get("k_final"),
        )
        requested_view_ks = list(final_params.get("k_s", []))
        full_view_effective_k_summaries = [
            summarize_effective_k(
                [len(np.unique(item.get("labels", []))) if len(item.get("labels", [])) else None
                 for item in view_dicts],
                requested_k=requested_view_ks[v] if v < len(requested_view_ks) else None,
            )
            for v, view_dicts in enumerate(full_label_dicts_views)
        ]

        # After consuming all bootstraps, compute full-data stability estimates
        if full_label_dicts_final:
            # Coassociation (scalar) + CCC for final clustering
            # full_final_stab_coassoc, full_final_stab_CCC = coassociation_stability(
            #     full_label_dicts_final, label_key="labels"
            # )
            # full_final_coassoc = full_final_stab_coassoc

            # ARI and Jaccard for final clustering
            full_final_stab_ari = ari_stability_common_subjects(full_label_dicts_final, label_key="labels")
            full_final_stab_jaccard = jaccard_stability_common_subjects(full_label_dicts_final, label_key="labels")

            # MATLAB-style consensus diagnostics also provide the consensus
            # matrix used below to produce one final label vector.
            full_final_stab_SUM_MAT_full = consensus_pac_ccc(
                full_label_dicts_final,
                label_key="labels",
                return_consensus=True,
                return_ecdf=True,
            )

        if any(full_label_dicts_views):
            # Per-view coassociation (scalar) + CCC
            # view_results = [
            #     coassociation_stability(view_dicts, label_key="labels")
            #     for view_dicts in full_label_dicts_views
            # ]
            # Each element: (stab_coassoc_view, ccc_view)
            # full_views_stab_coassoc, full_views_stab_CCC = zip(*view_results)
            # full_views_stab_coassoc = [float(s) for s in full_views_stab_coassoc]
            # full_views_coassoc = full_views_stab_coassoc

            # MATLAB-style consensus diagnostics (PAC/CCC) per view (include consensus matrices)
            stab_v_SUM_MAT_full = []
            for view_dicts in full_label_dicts_views:
                diag_v = consensus_pac_ccc(
                    view_dicts,
                    label_key="labels",
                    return_consensus=True,
                    return_ecdf=False,
                )
                stab_v_SUM_MAT_full.append({
                    "consensus": diag_v.get("consensus", None),
                    "union_ids": diag_v.get("union_ids", None),
                    "PAC": diag_v.get("PAC", np.nan),
                    "CCC": diag_v.get("CCC", np.nan),
                    "meta": diag_v.get("meta", {}),
                })

            full_v_stab_SUM_MAT_full = stab_v_SUM_MAT_full

            if full_v_stab_SUM_MAT_full:
                full_mean_view_stab_MAT_CCC = float(np.nanmean([d.get("CCC", np.nan) for d in full_v_stab_SUM_MAT_full]))
                full_mean_view_stab_MAT_PAC = float(np.nanmean([d.get("PAC", np.nan) for d in full_v_stab_SUM_MAT_full]))
            else:
                full_mean_view_stab_MAT_CCC = -3  # If no views, set to error code
                full_mean_view_stab_MAT_PAC = -3

            # Per-view ARI and Jaccard
            full_views_stab_ari = [
                ari_stability_common_subjects(view_dicts, label_key="labels")
                for view_dicts in full_label_dicts_views
            ]
            full_views_stab_jaccard = [
                jaccard_stability_common_subjects(view_dicts, label_key="labels")
                for view_dicts in full_label_dicts_views
            ]

            def _align_labels_to_ids(union_ids, labels, target_ids, fill_value=-1):
                """Handle align labels to ids."""
                if union_ids is None or labels is None:
                    return None
                idx_map = {sid: i for i, sid in enumerate(union_ids)}
                aligned = np.full(len(target_ids), fill_value, dtype=int)
                for j, sid in enumerate(target_ids):
                    i = idx_map.get(sid)
                    if i is not None:
                        aligned[j] = labels[i]
                if np.any(aligned == fill_value):
                    missing = int(np.sum(aligned == fill_value))
                    warnings.warn(f"{missing} subjects missing in consensus labels; filled with {fill_value}.")
                return aligned

            def _silhouette_norm(mat, labels, precomputed=False):
                """Handle silhouette norm."""
                labels = np.asarray(labels)
                if len(np.unique(labels)) <= 1:
                    result = 0.0
                else:
                    try:
                        if precomputed:
                            sil = silhouette_score(mat, labels, metric="precomputed")
                        else:
                            sil = silhouette_score(mat, labels)
                        result = (sil + 1.0) / 2.0
                    except Exception:
                        result = 0.0
                return float(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0))

            def _ch_norm(X, labels):
                """Handle ch norm."""
                labels = np.asarray(labels)
                if len(np.unique(labels)) <= 1:
                    result = 0.0
                else:
                    try:
                        ch = calinski_harabasz_score(X, labels)
                        result = ch / (ch + 1.0)
                    except Exception:
                        result = 0.0
                return float(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0))

            def _db_inv(X, labels):
                """Handle db inv."""
                labels = np.asarray(labels)
                if len(np.unique(labels)) <= 1:
                    result = 0.0
                else:
                    try:
                        db = davies_bouldin_score(X, labels)
                        result = 1.0 / (1.0 + db)
                    except Exception:
                        result = 0.0
                return float(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0))

            def _composite_view_quality(X, labels):
                """Handle composite view quality."""
                labels = np.asarray(labels)
                if len(np.unique(labels)) <= 1:
                    return 0.0
                s = _silhouette_norm(X, labels, precomputed=False)
                c = _ch_norm(X, labels)
                d = _db_inv(X, labels)
                return (s + c + d) / 3.0

            def _classical_mds(D, p=10):
                """Handle classical mds."""
                D = np.asarray(D, dtype=float)
                n = D.shape[0]
                if n == 0:
                    return np.zeros((0, 0), dtype=float)
                J = np.eye(n) - np.ones((n, n)) / n
                D2 = D ** 2
                B = -0.5 * J.dot(D2).dot(J)
                evals, evecs = np.linalg.eigh(B)
                idx = np.argsort(evals)[::-1]
                evals, evecs = evals[idx], evecs[:, idx]
                pos = evals > 1e-12
                if not np.any(pos):
                    return np.zeros((n, 1), dtype=float)
                evals_pos = evals[pos]
                evecs_pos = evecs[:, pos]
                m = min(p, evecs_pos.shape[1])
                X = evecs_pos[:, :m] * np.sqrt(evals_pos[:m])
                return X

            # Convert the final consensus matrix into full-data labels by
            # hierarchical clustering on distance = 1 - consensus.
            M_final = full_final_stab_SUM_MAT_full.get("consensus", None)
            union_ids_final = full_final_stab_SUM_MAT_full.get("union_ids", None)
            if M_final is None:
                warnings.warn("Final consensus matrix is empty; skipping ensemble labels.")
                final_labels = None
                final_labels_union = None
            else:
                M_final = np.asarray(M_final, dtype=float)
                assert M_final.ndim == 2 and M_final.shape[0] == M_final.shape[1], "Consensus must be square."
                M_final = (M_final + M_final.T) / 2.0
                np.fill_diagonal(M_final, 1.0)
                D_final = 1.0 - M_final
                dvec_final = squareform(D_final, checks=False)
                final_linkage = final_params.get("pre_linkage", final_params.get("pre_method", "average"))
                Z_final = hierarchy.linkage(dvec_final, method=final_linkage)
                print("Hierarchical clustering with k=", final_params.get("k_final", 2))
                final_labels_union = hierarchy.cut_tree(
                    Z_final, n_clusters=final_params.get("k_final", 2)
                ).reshape(-1)
                if str(args.mincluster).strip().upper() == "TRUE":
                    print("Enforcing minimum cluster size of", args.mincluster_n, "on final consensus labels")
                    final_labels_union = enforce_min_cluster_size(
                        D_final,
                        final_labels_union,
                        min_size=int(args.mincluster_n),
                    )
                final_labels = _align_labels_to_ids(
                    union_ids_final, final_labels_union, base_ids
                )

            # Repeat the same consensus-to-label reconstruction for each
            # modality so reporting has both integrated and per-view labels.
            indiv_labels = []
            indiv_labels_union = []
            view_scores_per_view = [None] * len(full_v_stab_SUM_MAT_full)
            for i in range(len(full_v_stab_SUM_MAT_full)):
                modality = full_v_stab_SUM_MAT_full[i]
                mod_name = args.modalities[i]
                M_view = modality.get("consensus", None)
                union_ids_view = modality.get("union_ids", None)
                if M_view is None:
                    warnings.warn(f"Consensus matrix missing for view {mod_name}; skipping.")
                    indiv_labels.append(None)
                    indiv_labels_union.append(None)
                    continue
                M_view = np.asarray(M_view, dtype=float)
                assert M_view.ndim == 2 and M_view.shape[0] == M_view.shape[1], "Consensus must be square."
                M_view = (M_view + M_view.T) / 2.0
                np.fill_diagonal(M_view, 1.0)
                D_view = 1.0 - M_view
                dvec_view = squareform(D_view, checks=False)
                view_linkages = final_params.get("linkage", [])
                view_linkage = (
                    view_linkages[i]
                    if i < len(view_linkages)
                    else final_params.get(f"c_{i+1}_method", "average")
                )
                Z_view = linkage(dvec_view, method=view_linkage)
                view_cluster_counts = final_params.get("k_s", [])
                view_k = (
                    int(view_cluster_counts[i])
                    if i < len(view_cluster_counts)
                    else int(final_params.get("c_"+str(i+1)+"_k", 2))
                )
                print("Hierarchical clustering with k=", view_k)
                labels_i_union = hierarchy.cut_tree(
                    Z_view, n_clusters=view_k
                ).reshape(-1)
                if str(args.mincluster).strip().upper() == "TRUE":
                    print(
                        "Enforcing minimum cluster size of",
                        args.mincluster_n,
                        "on consensus labels for view",
                        mod_name,
                    )
                    labels_i_union = enforce_min_cluster_size(
                        D_view,
                        labels_i_union,
                        min_size=int(args.mincluster_n),
                    )
                indiv_labels_union.append(labels_i_union)
                aligned_i = _align_labels_to_ids(
                    union_ids_view, labels_i_union, base_ids
                )
                indiv_labels.append(aligned_i)

            for i, labels_i in enumerate(indiv_labels):
                if labels_i is None:
                    continue
                labels_i = np.asarray(labels_i)
                valid_mask = labels_i >= 0
                if not np.any(valid_mask):
                    warnings.warn(f"No valid labels for view {args.modalities[i]}; skipping quality.")
                    continue
                X_sub = data_list[i][valid_mask]
                labs_sub = labels_i[valid_mask]
                view_scores_per_view[i] = float(_composite_view_quality(X_sub, labs_sub))

            valid_view_scores = [v for v in view_scores_per_view if v is not None]
            view_score_mean = float(np.mean(valid_view_scores)) if valid_view_scores else None

            final_score = None
            if final_labels is not None and M_final is not None and union_ids_final is not None:
                final_labels_arr = np.asarray(final_labels)
                valid_mask = final_labels_arr >= 0
                if np.any(valid_mask):
                    target_ids = [base_ids[i] for i in np.where(valid_mask)[0]]
                    labels_valid = final_labels_arr[valid_mask]
                    idx_map = {sid: i for i, sid in enumerate(union_ids_final)}
                    idxs = []
                    labs_sub = []
                    for sid, lab in zip(target_ids, labels_valid):
                        idx = idx_map.get(sid)
                        if idx is not None:
                            idxs.append(idx)
                            labs_sub.append(lab)
                    if len(idxs) >= 2:
                        M_sub = M_final[np.ix_(idxs, idxs)]
                        D_sub = 1.0 - M_sub
                        labs_sub = np.asarray(labs_sub)
                        if len(np.unique(labs_sub)) <= 1:
                            final_score = 0.0
                        else:
                            sil_final = _silhouette_norm(D_sub, labs_sub, precomputed=True)
                            X_mds = _classical_mds(D_sub, p=min(10, D_sub.shape[0] - 1))
                            ch_final = _ch_norm(X_mds, labs_sub)
                            dbi_final = _db_inv(X_mds, labs_sub)
                            final_score = float((sil_final + ch_final + dbi_final) / 3.0)




    def _stability_summary_for_preprocessing(
        mode,
        final_ari,
        final_jaccard,
        views_ari,
        views_jaccard,
        final_sum_mat,
        views_sum_mat,
        mean_view_mat_ccc,
        mean_view_mat_pac,
        seconds=None,
    ):
        """Package stability estimates for one preprocessing-resampling mode."""
        final_sum = None
        if isinstance(final_sum_mat, dict):
            final_sum = {
                "PAC": final_sum_mat.get("PAC", np.nan),
                "CCC": final_sum_mat.get("CCC", np.nan),
                "meta": final_sum_mat.get("meta", {}),
            }
        view_sums = None
        if views_sum_mat is not None:
            view_sums = [
                {
                    "PAC": d.get("PAC", np.nan),
                    "CCC": d.get("CCC", np.nan),
                    "meta": d.get("meta", {}),
                }
                for d in views_sum_mat
            ]
        return {
            "mode": mode,
            "n_bootstrap": int(n_boot_full),
            "seconds": seconds,
            "final_stability_ari": final_ari,
            "final_stability_jaccard": final_jaccard,
            "per_view_stabilities_ari": views_ari,
            "per_view_stabilities_jaccard": views_jaccard,
            "mean_view_stability_ari": float(np.nanmean(views_ari)) if views_ari is not None and len(views_ari) > 0 else None,
            "mean_view_stability_jaccard": float(np.nanmean(views_jaccard)) if views_jaccard is not None and len(views_jaccard) > 0 else None,
            "final_stability_SUM_MAT": final_sum,
            "per_view_stabilities_SUM_MAT": view_sums,
            "mean_view_stability_MAT_CCC": mean_view_mat_ccc,
            "mean_view_stability_MAT_PAC": mean_view_mat_pac,
        }

    def _summarize_bootstrap_label_dicts(label_dicts_final, label_dicts_views):
        """Compute the scalar stability summaries for a set of label dicts."""
        final_ari = None
        final_jaccard = None
        views_ari = None
        views_jaccard = None
        final_sum_mat = None
        views_sum_mat = None
        mean_view_mat_ccc = None
        mean_view_mat_pac = None

        if label_dicts_final:
            final_ari = ari_stability_common_subjects(label_dicts_final, label_key="labels")
            final_jaccard = jaccard_stability_common_subjects(label_dicts_final, label_key="labels")
            final_sum_mat = consensus_pac_ccc(
                label_dicts_final,
                label_key="labels",
                return_consensus=False,
                return_ecdf=False,
            )

        if any(label_dicts_views):
            view_summaries = []
            for view_dicts in label_dicts_views:
                diag_v = consensus_pac_ccc(
                    view_dicts,
                    label_key="labels",
                    return_consensus=False,
                    return_ecdf=False,
                )
                view_summaries.append({
                    "PAC": diag_v.get("PAC", np.nan),
                    "CCC": diag_v.get("CCC", np.nan),
                    "meta": diag_v.get("meta", {}),
                })
            views_sum_mat = view_summaries
            mean_view_mat_ccc = float(np.nanmean([d.get("CCC", np.nan) for d in view_summaries])) if view_summaries else -3
            mean_view_mat_pac = float(np.nanmean([d.get("PAC", np.nan) for d in view_summaries])) if view_summaries else -3
            views_ari = [
                ari_stability_common_subjects(view_dicts, label_key="labels")
                for view_dicts in label_dicts_views
            ]
            views_jaccard = [
                jaccard_stability_common_subjects(view_dicts, label_key="labels")
                for view_dicts in label_dicts_views
            ]

        return final_ari, final_jaccard, views_ari, views_jaccard, final_sum_mat, views_sum_mat, mean_view_mat_ccc, mean_view_mat_pac

    def _run_stability_only_for_mode(mode):
        """Run the alternate stability mode when --final_bootstrap_preprocessing=both."""
        nonlocal final_bootstrap_preprocessing
        previous_mode = final_bootstrap_preprocessing
        final_bootstrap_preprocessing = mode
        label_dicts_final = []
        label_dicts_views = [[] for _ in args.modalities]
        t0 = time.time()
        try:
            if n_boot_full > 0:
                if bootstrap_workers == 1:
                    result_iter = (_run_bootstrap(seed) for seed in seeds)
                    for final_entry, per_view_entries in result_iter:
                        label_dicts_final.append(final_entry)
                        for v, entry in enumerate(per_view_entries):
                            label_dicts_views[v].append(entry)
                else:
                    for final_entry, per_view_entries in _parallel_map_merge(
                        _run_bootstrap,
                        seeds,
                        bootstrap_workers,
                        f"comparison stability bootstraps ({mode})",
                        batch_size=1,
                    ):
                        label_dicts_final.append(final_entry)
                        for v, entry in enumerate(per_view_entries):
                            label_dicts_views[v].append(entry)
            seconds = float(time.time() - t0)
        finally:
            final_bootstrap_preprocessing = previous_mode

        summary = _summarize_bootstrap_label_dicts(label_dicts_final, label_dicts_views)
        return _stability_summary_for_preprocessing(mode, *summary, seconds=seconds)




    t_stability_start = time.time()
    selected_candidate_rank = None
    attempted_final_candidates = []
    for attempt_index, candidate in enumerate(final_param_candidates[:5], 1):
        final_params = candidate["params"]
        print(f"Final parameter attempt {attempt_index}/5: {final_params}")

        direct_labels, _, _, _, direct_quality = parea_2_mv(
            data_list,
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
            **final_params,
        )
        direct_labels = np.asarray(direct_labels)
        direct_cluster_count = int(len(np.unique(direct_labels))) if direct_labels.size else 0
        direct_quality_valid = (
            direct_quality is not None
            and np.isfinite(direct_quality)
            and direct_quality > 0
        )
        if direct_cluster_count < 2 or not direct_quality_valid:
            attempted_final_candidates.append({
                "rank": attempt_index,
                "params": final_params,
                "component_ranks": candidate["component_ranks"],
                "direct_n_clusters": direct_cluster_count,
                "direct_quality": direct_quality,
                "consensus_n_clusters": None,
                "consensus_quality": None,
                "accepted": False,
            })
            print(
                f"Rejected final parameter attempt {attempt_index} after direct all-data fit: "
                f"n_clusters={direct_cluster_count}, quality={direct_quality}."
            )
            continue

        full_label_dicts_final = []
        full_label_dicts_views = [[] for _ in args.modalities]
        full_final_stab_ari = None
        full_final_stab_jaccard = None
        full_views_stab_ari = None
        full_views_stab_jaccard = None
        full_final_stab_SUM_MAT_full = None
        full_v_stab_SUM_MAT_full = None
        full_mean_view_stab_MAT_CCC = None
        full_mean_view_stab_MAT_PAC = None
        full_final_effective_k_summary = None
        full_view_effective_k_summaries = None
        final_labels = None
        indiv_labels = None
        view_scores_per_view = None
        view_score_mean = None
        final_score = None

        if n_boot_full > 0:
            if bootstrap_workers == 1:
                _consume_bootstrap(_run_bootstrap(seed) for seed in seeds)
            else:
                _consume_bootstrap(_parallel_map_merge(
                    _run_bootstrap,
                    seeds,
                    bootstrap_workers,
                    f"final stability bootstraps attempt {attempt_index}",
                    batch_size=1,
                ))

        valid_final_labels = np.asarray(final_labels) if final_labels is not None else np.array([])
        valid_final_labels = valid_final_labels[valid_final_labels >= 0]
        cluster_count = int(len(np.unique(valid_final_labels))) if valid_final_labels.size else 0
        quality_valid = final_score is not None and np.isfinite(final_score) and final_score > 0
        attempted_final_candidates.append({
            "rank": attempt_index,
            "params": final_params,
            "component_ranks": candidate["component_ranks"],
            "direct_n_clusters": direct_cluster_count,
            "direct_quality": direct_quality,
            "consensus_n_clusters": cluster_count,
            "consensus_quality": final_score,
            "accepted": bool(cluster_count >= 2 and quality_valid),
            "requested_k_final": final_params.get("k_final"),
            "cross_fold_effective_k_final": final_params.get("k_final"),
            "full_bootstrap_effective_k_final": (
                full_final_effective_k_summary.get("selected_k")
                if full_final_effective_k_summary else None
            ),
            "full_bootstrap_effective_k_support_final": (
                full_final_effective_k_summary.get("support")
                if full_final_effective_k_summary else None
            ),
            "consensus_cut_k_final": final_params.get("k_final"),
            "final_effective_k": cluster_count,
        })
        if cluster_count >= 2 and quality_valid:
            selected_candidate_rank = attempt_index
            break
        print(
            f"Rejected final parameter attempt {attempt_index}: "
            f"n_clusters={cluster_count}, quality={final_score}."
        )

    t_stability_end = time.time()
    full_stability_bootstrap_seconds = float(t_stability_end - t_stability_start)

    if selected_candidate_rank is None:
        raise RuntimeError(
            "None of the top five complete fold-selected parameter sets produced at least two final "
            "consensus clusters with positive quality after minimum-size enforcement."
        )
    print(f"Selected complete parameter set rank {selected_candidate_rank}: {final_params}")

    stability_by_preprocessing = {
        final_bootstrap_preprocessing: _stability_summary_for_preprocessing(
            final_bootstrap_preprocessing,
            full_final_stab_ari,
            full_final_stab_jaccard,
            full_views_stab_ari,
            full_views_stab_jaccard,
            full_final_stab_SUM_MAT_full,
            full_v_stab_SUM_MAT_full,
            full_mean_view_stab_MAT_CCC,
            full_mean_view_stab_MAT_PAC,
            seconds=full_stability_bootstrap_seconds,
        )
    }
    if requested_final_bootstrap_preprocessing == "both":
        comparison_mode = "outside" if final_bootstrap_preprocessing == "inside" else "inside"
        print(f"Running comparison final stability bootstrap preprocessing mode: {comparison_mode}")
        stability_by_preprocessing[comparison_mode] = _run_stability_only_for_mode(comparison_mode)

    print(f"Completed bootstrap for full-data stability estimation.")
    if full_final_stab_ari is not None:
        print(f"Full-data final clustering stability (ARI): {full_final_stab_ari:.4f}")
    #if full_final_stab_coassoc is not None:
    #    print(f"Full-data final clustering stability (coassoc mean): {full_final_stab_coassoc:.4f}")
    #if full_final_stab_CCC is not None:
    #    print(f"Full-data final clustering stability (CCC): {full_final_stab_CCC:.4f}")
    if full_final_stab_jaccard is not None:
        print(f"Full-data final clustering stability (Jaccard): {full_final_stab_jaccard:.4f}")
    if full_final_stab_SUM_MAT_full is not None:
        print(f"Full-data final clustering stability (CCC MAT): {full_final_stab_SUM_MAT_full['CCC']:.4f}")
    if full_final_stab_SUM_MAT_full is not None:
        print(f"Full-data final clustering stability (PAC MAT): {full_final_stab_SUM_MAT_full['PAC']:.4f}")
    if full_views_stab_ari is not None:
        print(f"Full-data per-view clustering stabilities (ARI): {[f'{s:.4f}' for s in full_views_stab_ari]}")
    #if full_views_stab_coassoc is not None:
    #    print(f"Full-data per-view clustering stabilities (coassoc mean): {[f'{s:.4f}' for s in full_views_stab_coassoc]}")
    #if full_views_stab_CCC is not None:
    #    print(f"Full-data per-view clustering stabilities (CCC): {[f'{s:.4f}' for s in full_views_stab_CCC]}")
    if full_views_stab_jaccard is not None:
        print(f"Full-data per-view clustering stabilities (Jaccard): {[f'{s:.4f}' for s in full_views_stab_jaccard]}")
    if full_v_stab_SUM_MAT_full is not None:
        ccc_mat_per_view = [d.get("CCC", np.nan) for d in (full_v_stab_SUM_MAT_full or [])]
        print(f"Full-data per-view clustering stability (CCC MAT): {[f'{v:.4f}' for v in ccc_mat_per_view]}")
    if full_v_stab_SUM_MAT_full is not None:
        pac_mat_per_view = [d.get("PAC", np.nan) for d in (full_v_stab_SUM_MAT_full or [])]
        print(f"Full-data per-view clustering stability (PAC MAT): {[f'{v:.4f}' for v in pac_mat_per_view]}")
    if requested_final_bootstrap_preprocessing == "both":
        for mode, summary in stability_by_preprocessing.items():
            print(
                f"Final stability summary [{mode}]: "
                f"ARI={summary.get('final_stability_ari')}, "
                f"Jaccard={summary.get('final_stability_jaccard')}, "
                f"CCC={summary.get('final_stability_SUM_MAT', {}).get('CCC') if summary.get('final_stability_SUM_MAT') else None}, "
                f"PAC={summary.get('final_stability_SUM_MAT', {}).get('PAC') if summary.get('final_stability_SUM_MAT') else None}"
            )




    # Decide which stability metric should be exposed as "primary" in the final
    # payload. The raw ARI/Jaccard/MAT diagnostics are still saved separately.
    stab_view_key, stab_final_key, _, _ = _primary_metric_keys(args)

    # Map scalar final stability according to chosen objective
    if stab_final_key == "final_stability_ari":
        final_stability_primary = full_final_stab_ari
    #elif stab_final_key == "final_stability_coassoc":
    #    final_stability_primary = full_final_stab_coassoc
    #elif stab_final_key == "final_stability_CCC":
    #    final_stability_primary = full_final_stab_CCC
    elif stab_final_key == "final_stability_jaccard":
        final_stability_primary = full_final_stab_jaccard
    else:
        # Fallback
        final_stability_primary = full_final_stab_ari

    # Map per-view stability list according to chosen objective
    if stab_view_key == "mean_view_stability_ari":
        per_view_stabilities_primary = full_views_stab_ari
    #elif stab_view_key == "mean_view_stability_coassoc":
    #    per_view_stabilities_primary = full_views_stab_coassoc
    #elif stab_view_key == "mean_view_stability_CCC":
    #    per_view_stabilities_primary = full_views_stab_CCC
    elif stab_view_key == "mean_view_stability_jaccard":
        per_view_stabilities_primary = full_views_stab_jaccard
    else:
        per_view_stabilities_primary = full_views_stab_ari

    # Safe defaults if no bootstraps were run
    if per_view_stabilities_primary is None:
        per_view_stabilities_primary = []
    mean_view_stability_primary = float(np.mean(per_view_stabilities_primary)) if per_view_stabilities_primary else None
    min_view_stability_primary = float(np.min(per_view_stabilities_primary)) if per_view_stabilities_primary else None

    # --- Permutation p-values for cluster quality ---
    # The null keeps the observed data fixed and shuffles labels while preserving
    # cluster sizes. This tests whether observed separation is better than random
    # labels with the same class balance.
    t_cluster_pvalues_start = time.time()
    _n_perm_shared = getattr(args, 'cluster_pvalue_permutations', 200)
    _n_perm_quality = getattr(args, 'cluster_pvalue_permutations_quality', None)
    _n_perm_ari = getattr(args, 'cluster_pvalue_permutations_ari', None)
    if _n_perm_quality is None:
        _n_perm_quality = _n_perm_shared
    if _n_perm_ari is None:
        _n_perm_ari = _n_perm_shared
    cluster_pvalues = {
        'enabled': str(getattr(args, 'compute_cluster_pvalues', 'FALSE')).upper() == 'TRUE',
        'mode': str(getattr(args, 'cluster_pvalue_mode', 'fast')).lower(),
        'quality_null_method': 'label_shuffle_fixed_data',
        'statistic': str(getattr(args, 'cluster_pvalue_stat', 'composite')).lower(),
        'n_permutations': int(_n_perm_shared),
        'n_permutations_quality': int(_n_perm_quality),
        'n_permutations_ari': int(_n_perm_ari),
        'seed': int(getattr(args, 'cluster_pvalue_seed', 314159)),
        'workers': None,
        'observed': {'modalities': None, 'final': None},
        'null_summary': {'modalities_mean': None, 'modalities_std': None, 'final_mean': None, 'final_std': None},
        'pvalues_raw': {'modalities': None, 'final': None},
        'pvalues_fdr': {'modalities': None, 'with_final': None},
        'ari_stability': {
            'method': 'label_shuffle_within_bootstrap',
            'n_permutations': None,
            'observed': {'modalities': None, 'final': None},
            'null_summary': {'modalities_mean': None, 'modalities_std': None, 'final_mean': None, 'final_std': None},
            'pvalues_raw': {'modalities': None, 'final': None},
            'pvalues_fdr': {'modalities': None, 'with_final': None},
        },
        'notes': []
    }

    def _pval_silhouette_norm(X, labels):
        """Handle pval silhouette norm."""
        labels = np.asarray(labels)
        valid = labels >= 0
        if not np.any(valid):
            return np.nan
        Xv = np.asarray(X)[valid]
        lv = labels[valid]
        if Xv.shape[0] < 3 or len(np.unique(lv)) <= 1:
            return np.nan
        try:
            sil = silhouette_score(Xv, lv)
            return float((sil + 1.0) / 2.0)
        except Exception:
            return np.nan

    def _pval_ch_norm(X, labels):
        """Handle pval ch norm."""
        labels = np.asarray(labels)
        valid = labels >= 0
        if not np.any(valid):
            return np.nan
        Xv = np.asarray(X)[valid]
        lv = labels[valid]
        if Xv.shape[0] < 3 or len(np.unique(lv)) <= 1:
            return np.nan
        try:
            ch = calinski_harabasz_score(Xv, lv)
            return float(ch / (ch + 1.0))
        except Exception:
            return np.nan

    def _pval_db_inv(X, labels):
        """Handle pval db inv."""
        labels = np.asarray(labels)
        valid = labels >= 0
        if not np.any(valid):
            return np.nan
        Xv = np.asarray(X)[valid]
        lv = labels[valid]
        if Xv.shape[0] < 3 or len(np.unique(lv)) <= 1:
            return np.nan
        try:
            db = davies_bouldin_score(Xv, lv)
            return float(1.0 / (1.0 + db))
        except Exception:
            return np.nan

    def _cluster_stat(X, labels, stat_name):
        """Quality statistic used for observed and permuted label assignments."""
        if stat_name == 'silhouette':
            return _pval_silhouette_norm(X, labels)
        s = _pval_silhouette_norm(X, labels)
        c = _pval_ch_norm(X, labels)
        d = _pval_db_inv(X, labels)
        vals = [v for v in [s, c, d] if np.isfinite(v)]
        if not vals:
            return np.nan
        return float(np.mean(vals))

    def _shuffle_labels_fixed_counts(labels, rng):
        """Shuffle valid labels while preserving the exact observed label counts."""
        labels = np.asarray(labels).reshape(-1)
        out = labels.copy()
        valid = out >= 0
        n_valid = int(np.sum(valid))
        if n_valid > 1:
            out[valid] = out[valid][rng.permutation(n_valid)]
        return out

    def _bh_fdr(pvals):
        """Benjamini-Hochberg correction for modality/final p-values."""
        pvals = np.asarray(pvals, dtype=float)
        m = pvals.size
        if m == 0:
            return []
        order = np.argsort(np.nan_to_num(pvals, nan=np.inf))
        sorted_p = pvals[order]
        q = np.full(m, np.nan, dtype=float)
        prev = 1.0
        for i in range(m - 1, -1, -1):
            p = sorted_p[i]
            rank = i + 1
            val = np.nan if not np.isfinite(p) else min(prev, (p * m) / rank)
            q[i] = val
            prev = 1.0 if not np.isfinite(val) else val
        out = np.full(m, np.nan, dtype=float)
        out[order] = q
        return out.tolist()

    if cluster_pvalues['enabled']:
        # Compute observed modality/final quality once, then compare it against
        # B shuffled-label replicates.
        mode = cluster_pvalues['mode']
        stat_name = cluster_pvalues['statistic']
        B = max(1, cluster_pvalues['n_permutations_quality'])
        seed0 = cluster_pvalues['seed']
        if mode not in ('fast', 'full'):
            cluster_pvalues['notes'].append(f"Unknown cluster_pvalue_mode '{mode}', falling back to 'fast'.")
            mode = 'fast'
            cluster_pvalues['mode'] = mode
        if mode == 'full':
            cluster_pvalues['notes'].append(
                "cluster_pvalue_mode='full' previously reclustered permuted feature data; "
                "quality p-values now use label shuffling on the fixed observed data to preserve the data distribution."
            )
        if stat_name not in ('composite', 'silhouette'):
            cluster_pvalues['notes'].append(f"Unknown cluster_pvalue_stat '{stat_name}', falling back to 'composite'.")
            stat_name = 'composite'
            cluster_pvalues['statistic'] = stat_name

        if final_labels is None or indiv_labels is None:
            cluster_pvalues['notes'].append("Missing final/individual labels; p-value computation skipped.")
        else:
            obs_view = []
            for i, X in enumerate(data_list):
                labs = indiv_labels[i] if i < len(indiv_labels) else None
                obs_view.append(np.nan if labs is None else _cluster_stat(X, labs, stat_name))
            X_concat = np.hstack([np.asarray(X) for X in data_list]) if data_list else np.empty((0, 0))
            obs_final = _cluster_stat(X_concat, final_labels, stat_name)
            cluster_pvalues['observed']['modalities'] = [float(x) if np.isfinite(x) else np.nan for x in obs_view]
            cluster_pvalues['observed']['final'] = float(obs_final) if np.isfinite(obs_final) else np.nan

            raw_workers = getattr(args, 'cluster_pvalue_jobs', 0)
            if raw_workers in (None, 0):
                raw_workers = getattr(args, 'n_jobs', 1)
                if raw_workers in (-1, None):
                    raw_workers = os.cpu_count() or 1
            workers = max(1, min(int(raw_workers), B))
            cluster_pvalues['workers'] = int(workers)

            seeds_perm = [
                _derive_seed("cluster_quality_permutation", b, base=seed0)
                for b in range(B)
            ]

            def _perm_worker(seed_value):
                """Handle perm worker."""
                rng = np.random.default_rng(seed_value)
                perm_indiv_labels = [
                    _shuffle_labels_fixed_counts(labs, rng)
                    if labs is not None else None
                    for labs in indiv_labels
                ]
                perm_final_labels = _shuffle_labels_fixed_counts(final_labels, rng)

                perm_view_stats = []
                for i, Xp in enumerate(data_list):
                    labs = perm_indiv_labels[i] if i < len(perm_indiv_labels) else None
                    perm_view_stats.append(np.nan if labs is None else _cluster_stat(Xp, labs, stat_name))

                Xc = X_concat
                perm_final_stat = _cluster_stat(Xc, perm_final_labels, stat_name)
                return perm_view_stats, perm_final_stat

            if workers == 1:
                perm_results = [_perm_worker(s) for s in seeds_perm]
            else:
                def _perm_chunk_worker(seed_values):
                    """Handle perm chunk worker."""
                    return [_perm_worker(seed_value) for seed_value in seed_values]

                perm_chunks = _chunk_items(seeds_perm, workers)
                perm_results_nested = _parallel_map_merge(
                    _perm_chunk_worker,
                    perm_chunks,
                    workers,
                    "cluster quality permutation chunks",
                    batch_size=1,
                )
                perm_results = [row for chunk in perm_results_nested for row in chunk]

            null_view = np.array([r[0] for r in perm_results], dtype=float) if perm_results else np.empty((0, len(args.modalities)))
            null_final = np.array([r[1] for r in perm_results], dtype=float) if perm_results else np.empty((0,), dtype=float)

            pvals_view = []
            for i, obs in enumerate(obs_view):
                if not np.isfinite(obs):
                    pvals_view.append(np.nan)
                    continue
                ni = null_view[:, i]
                ni = ni[np.isfinite(ni)]
                if ni.size == 0:
                    pvals_view.append(np.nan)
                    continue
                p = (1.0 + np.sum(ni >= obs)) / (ni.size + 1.0)
                pvals_view.append(float(p))

            if np.isfinite(obs_final):
                nf = null_final[np.isfinite(null_final)]
                p_final = float((1.0 + np.sum(nf >= obs_final)) / (nf.size + 1.0)) if nf.size > 0 else np.nan
            else:
                p_final = np.nan

            pvals_with_final = list(pvals_view) + [p_final]
            cluster_pvalues['null_summary']['modalities_mean'] = np.nanmean(null_view, axis=0).tolist() if null_view.size else None
            cluster_pvalues['null_summary']['modalities_std'] = np.nanstd(null_view, axis=0).tolist() if null_view.size else None
            cluster_pvalues['null_summary']['final_mean'] = float(np.nanmean(null_final)) if null_final.size else None
            cluster_pvalues['null_summary']['final_std'] = float(np.nanstd(null_final)) if null_final.size else None
            cluster_pvalues['pvalues_raw']['modalities'] = pvals_view
            cluster_pvalues['pvalues_raw']['final'] = p_final
            cluster_pvalues['pvalues_fdr']['modalities'] = _bh_fdr(pvals_view)
            cluster_pvalues['pvalues_fdr']['with_final'] = _bh_fdr(pvals_with_final)
    t_cluster_pvalues_end = time.time()
    cluster_pvalues_seconds = float(t_cluster_pvalues_end - t_cluster_pvalues_start)

    def _pairwise_scores_common_subjects(label_dicts, label_key="labels", metric="ari", precomputed_alignment=None):
        """Return pairwise bootstrap stability scores after subject alignment."""
        scores = []
        if not label_dicts or len(label_dicts) < 2:
            return scores
        if precomputed_alignment is None:
            for d1, d2 in combinations(label_dicts, 2):
                u1, l1 = _collapse_duplicates(d1["orig_ids"], d1[label_key])
                u2, l2 = _collapse_duplicates(d2["orig_ids"], d2[label_key])
                common, idx1, idx2 = np.intersect1d(u1, u2, return_indices=True)
                if len(common) <= 1:
                    continue
                a = l1[idx1]
                b = l2[idx2]
                if metric == "ari":
                    scores.append(float(adjusted_rand_score(a, b)))
                elif metric == "jaccard":
                    scores.append(float(_partition_jaccard_from_labels(a, b)))
            return scores

        labels_collapsed = []
        for d in label_dicts:
            _, labs = _collapse_duplicates(d["orig_ids"], d[label_key])
            labels_collapsed.append(np.asarray(labs))
        for b1, b2, idx1, idx2 in precomputed_alignment.get("pair_indices", []):
            a = labels_collapsed[b1][idx1]
            b = labels_collapsed[b2][idx2]
            if metric == "ari":
                scores.append(float(adjusted_rand_score(a, b)))
            elif metric == "jaccard":
                scores.append(float(_partition_jaccard_from_labels(a, b)))
        return scores

    def _collapsed_labels_from_dicts(label_dicts, label_key="labels"):
        """Collapse duplicate bootstrap rows to one label per subject."""
        labels_collapsed = []
        for d in label_dicts:
            _, labs = _collapse_duplicates(d["orig_ids"], d[label_key])
            labels_collapsed.append(np.asarray(labs))
        return labels_collapsed

    def _mean_pairwise_ari_from_alignment(labels_collapsed, precomputed_alignment):
        """Mean ARI over the precomputed common-subject bootstrap pairs."""
        if precomputed_alignment is None:
            return np.nan
        pair_indices = precomputed_alignment.get("pair_indices", [])
        if not pair_indices:
            return np.nan
        scores = []
        for b1, b2, idx1, idx2 in pair_indices:
            a = labels_collapsed[b1][idx1]
            b = labels_collapsed[b2][idx2]
            if a.size > 1:
                scores.append(float(adjusted_rand_score(a, b)))
        return float(np.mean(scores)) if scores else np.nan

    def _permute_collapsed_labels(labels_collapsed, rng):
        """Shuffle labels within each bootstrap for the ARI-stability null."""
        permuted = []
        for labs in labels_collapsed:
            if labs.size <= 1:
                permuted.append(labs.copy())
            else:
                permuted.append(labs[rng.permutation(labs.size)])
        return permuted

    def _cluster_composition(labels):
        """Count cluster sizes and proportions for final reporting."""
        if labels is None:
            return {"n_total": 0, "n_labeled": 0, "counts": {}, "proportions": {}}
        arr = np.asarray(labels)
        n_total = int(arr.size)
        valid = arr[arr >= 0]
        n_labeled = int(valid.size)
        if n_labeled == 0:
            return {"n_total": n_total, "n_labeled": 0, "counts": {}, "proportions": {}}
        uniq, cnt = np.unique(valid, return_counts=True)
        return {
            "n_total": n_total,
            "n_labeled": n_labeled,
            "counts": {str(int(k)): int(v) for k, v in zip(uniq, cnt)},
            "proportions": {str(int(k)): float(v / n_labeled) for k, v in zip(uniq, cnt)},
        }

    def _quality_components_from_features(X, labels):
        """Break composite quality into silhouette, CH, and DB components."""
        labels = np.asarray(labels) if labels is not None else np.array([])
        if labels.size == 0:
            return {"n_labeled": 0, "n_clusters": 0, "silhouette_norm": np.nan, "ch_norm": np.nan, "db_inv": np.nan, "composite": np.nan}
        valid = labels >= 0
        if not np.any(valid):
            return {"n_labeled": 0, "n_clusters": 0, "silhouette_norm": np.nan, "ch_norm": np.nan, "db_inv": np.nan, "composite": np.nan}
        Xv = np.asarray(X)[valid]
        lv = labels[valid]
        n_clusters = int(len(np.unique(lv)))
        if Xv.shape[0] < 3 or n_clusters <= 1:
            return {"n_labeled": int(Xv.shape[0]), "n_clusters": n_clusters, "silhouette_norm": np.nan, "ch_norm": np.nan, "db_inv": np.nan, "composite": np.nan}
        try:
            sil = float((silhouette_score(Xv, lv) + 1.0) / 2.0)
        except Exception:
            sil = np.nan
        try:
            ch = calinski_harabasz_score(Xv, lv)
            chn = float(ch / (ch + 1.0))
        except Exception:
            chn = np.nan
        try:
            db = davies_bouldin_score(Xv, lv)
            dbi = float(1.0 / (1.0 + db))
        except Exception:
            dbi = np.nan
        vals = [v for v in [sil, chn, dbi] if np.isfinite(v)]
        return {
            "n_labeled": int(Xv.shape[0]),
            "n_clusters": n_clusters,
            "silhouette_norm": sil,
            "ch_norm": chn,
            "db_inv": dbi,
            "composite": float(np.mean(vals)) if vals else np.nan
        }

    def _classical_mds_local(D, p=10):
        """Handle classical mds local."""
        D = np.asarray(D, dtype=float)
        n = D.shape[0]
        if n == 0:
            return np.zeros((0, 0), dtype=float)
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J.dot(D ** 2).dot(J)
        evals, evecs = np.linalg.eigh(B)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        pos = evals > 1e-12
        if not np.any(pos):
            return np.zeros((n, 1), dtype=float)
        m = min(p, int(np.sum(pos)))
        return evecs[:, pos][:, :m] * np.sqrt(evals[pos][:m])

    def _quality_components_from_consensus(consensus, union_ids, target_ids, target_labels):
        """Compute quality in consensus-distance space for final labels."""
        if consensus is None or union_ids is None or target_labels is None:
            return {"n_labeled": 0, "n_clusters": 0, "silhouette_norm": np.nan, "ch_norm": np.nan, "db_inv": np.nan, "composite": np.nan}
        labels = np.asarray(target_labels)
        valid = labels >= 0
        if not np.any(valid):
            return {"n_labeled": 0, "n_clusters": 0, "silhouette_norm": np.nan, "ch_norm": np.nan, "db_inv": np.nan, "composite": np.nan}
        idx_map = {sid: i for i, sid in enumerate(union_ids)}
        idxs, labs = [], []
        for sid, lab in zip(target_ids, labels):
            if lab < 0:
                continue
            i = idx_map.get(sid)
            if i is None:
                continue
            idxs.append(i)
            labs.append(int(lab))
        if len(idxs) < 3:
            return {"n_labeled": len(idxs), "n_clusters": len(set(labs)), "silhouette_norm": np.nan, "ch_norm": np.nan, "db_inv": np.nan, "composite": np.nan}
        D = 1.0 - np.asarray(consensus, dtype=float)[np.ix_(idxs, idxs)]
        labs = np.asarray(labs)
        n_clusters = int(len(np.unique(labs)))
        if n_clusters <= 1:
            return {"n_labeled": len(idxs), "n_clusters": n_clusters, "silhouette_norm": np.nan, "ch_norm": np.nan, "db_inv": np.nan, "composite": np.nan}
        try:
            sil = float((silhouette_score(D, labs, metric="precomputed") + 1.0) / 2.0)
        except Exception:
            sil = np.nan
        X_mds = _classical_mds_local(D, p=min(10, D.shape[0] - 1))
        try:
            ch = calinski_harabasz_score(X_mds, labs)
            chn = float(ch / (ch + 1.0))
        except Exception:
            chn = np.nan
        try:
            db = davies_bouldin_score(X_mds, labs)
            dbi = float(1.0 / (1.0 + db))
        except Exception:
            dbi = np.nan
        vals = [v for v in [sil, chn, dbi] if np.isfinite(v)]
        return {"n_labeled": len(idxs), "n_clusters": n_clusters, "silhouette_norm": sil, "ch_norm": chn, "db_inv": dbi, "composite": float(np.mean(vals)) if vals else np.nan}

    def _assignment_certainty_from_consensus(consensus, union_ids, target_ids, target_labels, uncertain_threshold=0.60):
        """Summarize how strongly each subject matches its assigned consensus cluster."""
        empty = {
            "n_assessed": 0,
            "uncertain_threshold": float(uncertain_threshold),
            "uncertain_n": 0,
            "uncertain_fraction": np.nan,
            "assigned_cluster_mean_consensus": _distribution_summary([]),
            "best_cluster_mean_consensus": _distribution_summary([]),
            "second_best_cluster_mean_consensus": _distribution_summary([]),
            "margin_best_minus_second": _distribution_summary([]),
        }
        if consensus is None or union_ids is None or target_labels is None:
            return empty
        C = np.asarray(consensus, dtype=float)
        labels = np.asarray(target_labels)
        idx_map = {sid: i for i, sid in enumerate(union_ids)}
        rows = []
        assigned = []
        for sid, lab in zip(target_ids, labels):
            if lab < 0:
                continue
            ui = idx_map.get(sid)
            if ui is None:
                continue
            rows.append((ui, int(lab)))
            assigned.append(int(lab))
        if not rows:
            return empty
        assigned = np.asarray(assigned, dtype=int)
        uniq = np.unique(assigned)
        members = {c: np.where(assigned == c)[0] for c in uniq}
        assigned_scores, best_scores, second_scores, margins = [], [], [], []
        for local_i, (ui, lab) in enumerate(rows):
            means = {}
            for c in uniq:
                mem_local = members[c]
                mem_union = [rows[j][0] for j in mem_local]
                vals = C[ui, mem_union].astype(float)
                if int(c) == int(lab):
                    vals = vals[mem_local != local_i] if len(mem_local) > 1 else np.array([], dtype=float)
                means[int(c)] = float(np.mean(vals)) if vals.size > 0 else np.nan
            assigned_score = means.get(int(lab), np.nan)
            ranked = sorted([v for v in means.values() if np.isfinite(v)], reverse=True)
            best = ranked[0] if ranked else np.nan
            second = ranked[1] if len(ranked) > 1 else np.nan
            margin = (best - second) if np.isfinite(best) and np.isfinite(second) else np.nan
            assigned_scores.append(assigned_score)
            best_scores.append(best)
            second_scores.append(second)
            margins.append(margin)
        best_arr = np.asarray(best_scores, dtype=float)
        uncertain_n = int(np.sum(best_arr < float(uncertain_threshold))) if best_arr.size > 0 else 0
        n_assessed = int(len(rows))
        return {
            "n_assessed": n_assessed,
            "uncertain_threshold": float(uncertain_threshold),
            "uncertain_n": uncertain_n,
            "uncertain_fraction": float(uncertain_n / n_assessed) if n_assessed > 0 else np.nan,
            "assigned_cluster_mean_consensus": _distribution_summary(assigned_scores),
            "best_cluster_mean_consensus": _distribution_summary(best_scores),
            "second_best_cluster_mean_consensus": _distribution_summary(second_scores),
            "margin_best_minus_second": _distribution_summary(margins),
        }

    final_pair_align = precompute_bootstrap_pair_alignment(full_label_dicts_final) if len(full_label_dicts_final) >= 2 else None
    final_stab_ari_dist = _pairwise_scores_common_subjects(full_label_dicts_final, label_key="labels", metric="ari", precomputed_alignment=final_pair_align)
    final_stab_jaccard_dist = _pairwise_scores_common_subjects(full_label_dicts_final, label_key="labels", metric="jaccard", precomputed_alignment=final_pair_align)
    view_stab_ari_dist = []
    view_stab_jaccard_dist = []
    for view_dicts in full_label_dicts_views:
        align_v = precompute_bootstrap_pair_alignment(view_dicts) if len(view_dicts) >= 2 else None
        view_stab_ari_dist.append(_pairwise_scores_common_subjects(view_dicts, label_key="labels", metric="ari", precomputed_alignment=align_v))
        view_stab_jaccard_dist.append(_pairwise_scores_common_subjects(view_dicts, label_key="labels", metric="jaccard", precomputed_alignment=align_v))

    # --- Permutation p-values for ARI stability ---
    # Here the null shuffles each bootstrap's collapsed labels, then recomputes
    # pairwise ARI. This asks whether cross-bootstrap agreement exceeds random
    # partitions with the same per-bootstrap cluster sizes.
    if cluster_pvalues.get('enabled'):
        B_ari = max(1, int(cluster_pvalues.get('n_permutations_ari', cluster_pvalues.get('n_permutations', 200))))
        seed_ari0 = int(cluster_pvalues.get('seed', 314159)) + 1000003
        ari_results = cluster_pvalues.get('ari_stability', {})
        ari_results['n_permutations'] = int(B_ari)

        view_alignments = [
            precompute_bootstrap_pair_alignment(vdicts) if len(vdicts) >= 2 else None
            for vdicts in full_label_dicts_views
        ]
        final_labels_collapsed = _collapsed_labels_from_dicts(full_label_dicts_final, label_key="labels")
        view_labels_collapsed = [
            _collapsed_labels_from_dicts(vdicts, label_key="labels")
            for vdicts in full_label_dicts_views
        ]

        obs_final_ari = _mean_pairwise_ari_from_alignment(final_labels_collapsed, final_pair_align)
        obs_view_ari = [
            _mean_pairwise_ari_from_alignment(view_labels_collapsed[i], view_alignments[i])
            for i in range(len(view_alignments))
        ]

        ari_results['observed']['modalities'] = [float(x) if np.isfinite(x) else np.nan for x in obs_view_ari]
        ari_results['observed']['final'] = float(obs_final_ari) if np.isfinite(obs_final_ari) else np.nan

        ari_workers = cluster_pvalues.get('workers')
        if ari_workers in (None, 0):
            raw_workers = getattr(args, 'cluster_pvalue_jobs', 0)
            if raw_workers in (None, 0):
                raw_workers = getattr(args, 'n_jobs', 1)
                if raw_workers in (-1, None):
                    raw_workers = os.cpu_count() or 1
            ari_workers = max(1, min(int(raw_workers), B_ari))
        else:
            ari_workers = max(1, min(int(ari_workers), B_ari))

        seeds_ari = [
            _derive_seed("cluster_ari_permutation", b, base=seed_ari0)
            for b in range(B_ari)
        ]

        def _ari_perm_worker(seed_value):
            """Handle ari perm worker."""
            rng = np.random.default_rng(seed_value)
            perm_final = _permute_collapsed_labels(final_labels_collapsed, rng)
            perm_final_mean = _mean_pairwise_ari_from_alignment(perm_final, final_pair_align)
            perm_views_mean = []
            for i in range(len(view_labels_collapsed)):
                perm_view_i = _permute_collapsed_labels(view_labels_collapsed[i], rng)
                perm_views_mean.append(_mean_pairwise_ari_from_alignment(perm_view_i, view_alignments[i]))
            return perm_views_mean, perm_final_mean

        if ari_workers == 1:
            ari_perm_results = [_ari_perm_worker(s) for s in seeds_ari]
        else:
            def _ari_perm_chunk_worker(seed_values):
                """Handle ari perm chunk worker."""
                return [_ari_perm_worker(seed_value) for seed_value in seed_values]

            ari_chunks = _chunk_items(seeds_ari, ari_workers)
            ari_perm_results_nested = _parallel_map_merge(
                _ari_perm_chunk_worker,
                ari_chunks,
                ari_workers,
                "ARI stability permutation chunks",
                batch_size=1,
            )
            ari_perm_results = [row for chunk in ari_perm_results_nested for row in chunk]

        n_modalities = len(args.modalities)
        null_view_ari = np.array([r[0] for r in ari_perm_results], dtype=float) if ari_perm_results else np.empty((0, n_modalities), dtype=float)
        null_final_ari = np.array([r[1] for r in ari_perm_results], dtype=float) if ari_perm_results else np.empty((0,), dtype=float)

        pvals_view_ari = []
        for i, obs in enumerate(obs_view_ari):
            if not np.isfinite(obs):
                pvals_view_ari.append(np.nan)
                continue
            ni = null_view_ari[:, i]
            ni = ni[np.isfinite(ni)]
            if ni.size == 0:
                pvals_view_ari.append(np.nan)
                continue
            pvals_view_ari.append(float((1.0 + np.sum(ni >= obs)) / (ni.size + 1.0)))

        if np.isfinite(obs_final_ari):
            nf = null_final_ari[np.isfinite(null_final_ari)]
            p_final_ari = float((1.0 + np.sum(nf >= obs_final_ari)) / (nf.size + 1.0)) if nf.size > 0 else np.nan
        else:
            p_final_ari = np.nan

        ari_with_final = list(pvals_view_ari) + [p_final_ari]
        ari_results['null_summary']['modalities_mean'] = np.nanmean(null_view_ari, axis=0).tolist() if null_view_ari.size else None
        ari_results['null_summary']['modalities_std'] = np.nanstd(null_view_ari, axis=0).tolist() if null_view_ari.size else None
        ari_results['null_summary']['final_mean'] = float(np.nanmean(null_final_ari)) if null_final_ari.size else None
        ari_results['null_summary']['final_std'] = float(np.nanstd(null_final_ari)) if null_final_ari.size else None
        ari_results['pvalues_raw']['modalities'] = pvals_view_ari
        ari_results['pvalues_raw']['final'] = p_final_ari
        ari_results['pvalues_fdr']['modalities'] = _bh_fdr(pvals_view_ari)
        ari_results['pvalues_fdr']['with_final'] = _bh_fdr(ari_with_final)
        cluster_pvalues['ari_stability'] = ari_results

    # Convert pairwise bootstrap score distributions into means/CIs for reports.
    stability_uncertainty = {
        "final_ari": _distribution_summary(final_stab_ari_dist),
        "final_jaccard": _distribution_summary(final_stab_jaccard_dist),
        "per_view_ari": [_distribution_summary(x) for x in view_stab_ari_dist],
        "per_view_jaccard": [_distribution_summary(x) for x in view_stab_jaccard_dist],
    }

    final_consensus = (full_final_stab_SUM_MAT_full or {}).get("consensus", None)
    final_union_ids = (full_final_stab_SUM_MAT_full or {}).get("union_ids", None)
    # Reporting payloads below are intentionally plain dict/list structures so
    # notebooks and postprocessing scripts can read them without recomputation.
    quality_components = {
        "per_view_feature_space": [
            _quality_components_from_features(
                data_list[i],
                indiv_labels[i] if indiv_labels is not None and i < len(indiv_labels) else None
            )
            for i in range(len(args.modalities))
        ],
        "final_feature_space": _quality_components_from_features(
            np.hstack([np.asarray(X) for X in data_list]) if data_list else np.empty((0, 0)),
            final_labels
        ),
        "final_consensus_space": _quality_components_from_consensus(final_consensus, final_union_ids, base_ids, final_labels),
    }

    cluster_composition = {
        "final": _cluster_composition(final_labels),
        "per_view": [
            _cluster_composition(indiv_labels[i] if indiv_labels is not None and i < len(indiv_labels) else None)
            for i in range(len(args.modalities))
        ],
    }

    assignment_certainty = {
        "final": _assignment_certainty_from_consensus(final_consensus, final_union_ids, base_ids, final_labels),
        "per_view": [],
    }
    for i in range(len(args.modalities)):
        diag_v = full_v_stab_SUM_MAT_full[i] if full_v_stab_SUM_MAT_full is not None and i < len(full_v_stab_SUM_MAT_full) else {}
        c_v = diag_v.get("consensus", None) if isinstance(diag_v, dict) else None
        ids_v = diag_v.get("union_ids", None) if isinstance(diag_v, dict) else None
        lbl_v = indiv_labels[i] if indiv_labels is not None and i < len(indiv_labels) else None
        assignment_certainty["per_view"].append(_assignment_certainty_from_consensus(c_v, ids_v, base_ids, lbl_v))

    dropped_full_modality = preprocessing_details.get("subjects_dropped_full_missing_modality", []) if isinstance(preprocessing_details, dict) else []
    preprocessing_flow = {
        "n_input_rows": int(len(df)),
        "n_after_preprocessing_alignment": preprocessing_details.get("n_subjects_after_alignment") if isinstance(preprocessing_details, dict) else None,
        "n_dropped_full_missing_modality": int(len(dropped_full_modality)),
        "n_subjects_with_final_label": int(np.sum(np.asarray(final_labels) >= 0)) if final_labels is not None else 0,
    }

    runtime_context = {
        "preprocessing_seconds": preprocessing_seconds,
        "dim_reduction_seconds": dim_reduction_seconds,
        "full_stability_bootstrap_seconds": full_stability_bootstrap_seconds,
        "cluster_pvalues_seconds": cluster_pvalues_seconds,
        "n_bootstrap_full": int(n_boot_full),
        "bootstrap_workers": int(bootstrap_workers),
        "final_bootstrap_preprocessing": final_bootstrap_preprocessing,
        "requested_final_bootstrap_preprocessing": requested_final_bootstrap_preprocessing,
        "n_jobs": int(args.n_jobs) if getattr(args, "n_jobs", None) is not None else None,
    }

    # One nested reporting object groups the human-facing summaries separately
    # from raw arrays/models stored elsewhere in metrics_merged.
    final_reporting = {
        "quality": {
            "components": quality_components,
        },
        "stability": {
            "point_estimates": {
                "final_ari": full_final_stab_ari,
                "final_jaccard": full_final_stab_jaccard,
                "per_view_ari": full_views_stab_ari,
                "per_view_jaccard": full_views_stab_jaccard,
            },
            "by_preprocessing": stability_by_preprocessing,
            "uncertainty": stability_uncertainty,
        },
        "clusters": {
            "composition": cluster_composition,
            "assignment_certainty": assignment_certainty,
        },
        "preprocessing_flow": preprocessing_flow,
        "runtime_context": runtime_context,
        "compute_context": {
            "modalities": list(args.modalities),
            "scaler_type": args.scaler_type,
            "dim_reduction": args.dim_reduction,
            "dim_reduction_by_modality": dict(args.dim_reduction_by_modality),
            "pca_variance_threshold": args.pca_variance_threshold,
            "snmf_n_components": int(args.snmf_n_components),
            "snmf_alpha": float(args.snmf_alpha),
            "snmf_l1_ratio": float(args.snmf_l1_ratio),
            "snmf_max_iter": int(args.snmf_max_iter),
            "internal_ensemble_enabled": args.internal_ensemble_enabled,
            "internal_ensemble_bcs": int(args.internal_ensemble_bcs),
            "internal_ensemble_sample_frac": float(args.internal_ensemble_sample_frac),
            "internal_ensemble_feature_frac": float(args.internal_ensemble_feature_frac),
            "final_bootstrap_preprocessing": final_bootstrap_preprocessing,
            "requested_final_bootstrap_preprocessing": requested_final_bootstrap_preprocessing,
            "cluster_pvalue_settings": {
                "enabled": cluster_pvalues.get("enabled"),
                "mode": cluster_pvalues.get("mode"),
                "quality_null_method": cluster_pvalues.get("quality_null_method"),
                "statistic": cluster_pvalues.get("statistic"),
                "n_permutations": cluster_pvalues.get("n_permutations"),
                "n_permutations_quality": cluster_pvalues.get("n_permutations_quality"),
                "n_permutations_ari": cluster_pvalues.get("n_permutations_ari"),
                "workers": cluster_pvalues.get("workers"),
                "seed": cluster_pvalues.get("seed"),
            },
        },
    }

    def _final_reporting_with_runtime():
        """Handle final reporting with runtime."""
        rep = dict(final_reporting)
        rc = dict(rep.get("runtime_context", {}))
        rc["total_merge_seconds"] = float(time.time() - t_merge_start)
        rep["runtime_context"] = rc
        return rep

    def _merge_extra_metrics():
        """Merge extra metrics."""
        def _cluster_sizes(labels):
            """Handle cluster sizes."""
            if labels is None:
                return {}
            arr = np.asarray(labels)
            arr = arr[arr >= 0]
            unique, counts = np.unique(arr, return_counts=True)
            return {int(k): int(v) for k, v in zip(unique, counts)}

        final_observed_k = (
            int(len(np.unique(np.asarray(final_labels)[np.asarray(final_labels) >= 0])))
            if final_labels is not None else 0
        )
        view_observed_k = [
            int(len(np.unique(np.asarray(labels)[np.asarray(labels) >= 0]))) if labels is not None else 0
            for labels in (indiv_labels or [])
        ]
        return {
            "metrics_schema_version": METRICS_SCHEMA_VERSION,
            "final_param_selection": {
                "strategy": "rank_individual_hyperparameters_then_retry_combinations",
                "max_attempts": 5,
                "selected_rank": selected_candidate_rank,
                "individual_parameter_rankings": param_selection,
                "attempts": attempted_final_candidates,
                "fold_requested_params": {
                    fold: metrics[fold].get("best_params_requested", metrics[fold].get("best_params"))
                    for fold in fold_names
                },
                "fold_effective_params": {
                    fold: metrics[fold].get("best_params_effective", metrics[fold].get("best_params"))
                    for fold in fold_names
                },
            },
            "consensus_cut_k": {
                "final": int(final_params.get("k_final", 2)),
                "views": [int(k) for k in final_params.get("k_s", [])],
            },
            "effective_k": {
                "requested": {
                    fold: metrics[fold].get("best_params_requested", metrics[fold].get("best_params"))
                    for fold in fold_names
                },
                "fold_bootstrap": {
                    fold: {
                        "final": metrics[fold].get("best_fitness", {}).get("final_effective_k_summary"),
                        "views": metrics[fold].get("best_fitness", {}).get("view_effective_k_summaries"),
                    }
                    for fold in fold_names
                },
                "cross_fold_selected": {
                    "final": int(final_params.get("k_final", 2)),
                    "views": [int(k) for k in final_params.get("k_s", [])],
                },
                "full_bootstrap": {
                    "final": full_final_effective_k_summary,
                    "views": full_view_effective_k_summaries,
                },
                "consensus_cut": {
                    "final": int(final_params.get("k_final", 2)),
                    "views": [int(k) for k in final_params.get("k_s", [])],
                },
                "final_observed": {
                    "final": {
                        "k": final_observed_k,
                        "cluster_sizes": _cluster_sizes(final_labels),
                    },
                    "views": [
                        {
                            "k": view_observed_k[i],
                            "cluster_sizes": _cluster_sizes(labels),
                        }
                        for i, labels in enumerate(indiv_labels or [])
                    ],
                },
            },
            "stability_by_preprocessing": stability_by_preprocessing,
            "final_bootstrap_preprocessing": final_bootstrap_preprocessing,
            "requested_final_bootstrap_preprocessing": requested_final_bootstrap_preprocessing,
            "effective_k_report_csv": effective_k_report_path,
        }



    if not output_final_metrics_path:
        raise ValueError("For merge mode, --output_final_metrics must be specified")
    final_metrics_dir = os.path.dirname(output_final_metrics_path) or "."
    os.makedirs(final_metrics_dir, exist_ok=True)

    effective_k_report_rows = []
    for fold_name in fold_names:
        fold_payload = metrics[fold_name]
        best_fitness = fold_payload.get("best_fitness", {})
        component_summaries = [("final", best_fitness.get("final_effective_k_summary"))]
        component_summaries.extend(zip(args.modalities, best_fitness.get("view_effective_k_summaries", [])))
        for component, summary in component_summaries:
            summary = summary or {}
            effective_k_report_rows.append({
                "pipeline": "multiclust",
                "fold": fold_name,
                "component": component,
                "level": "fold_bootstrap",
                "requested_k": summary.get("requested_k"),
                "selected_effective_k": summary.get("selected_k"),
                "mode_support": summary.get("support"),
                "retention_rate": summary.get("retention_rate"),
                "normalized_entropy": summary.get("normalized_entropy"),
                "distribution": repr(summary.get("counts", {})),
                "mincluster_n_applied": fold_payload.get("mincluster_n_applied"),
                "reference_n": fold_payload.get("reference_n"),
                "current_n": fold_payload.get("current_n"),
            })
    full_component_summaries = [("final", full_final_effective_k_summary)]
    full_component_summaries.extend(zip(args.modalities, full_view_effective_k_summaries or []))
    for component, summary in full_component_summaries:
        summary = summary or {}
        entries = full_label_dicts_final if component == "final" else full_label_dicts_views[args.modalities.index(component)]
        effective_k_report_rows.append({
            "pipeline": "multiclust",
            "fold": "all",
            "component": component,
            "level": "full_bootstrap",
            "requested_k": summary.get("requested_k"),
            "selected_effective_k": summary.get("selected_k"),
            "mode_support": summary.get("support"),
            "retention_rate": summary.get("retention_rate"),
            "normalized_entropy": summary.get("normalized_entropy"),
            "distribution": repr(summary.get("counts", {})),
            "mincluster_n_applied": repr(sorted({
                e.get("mincluster_n_applied") for e in entries
                if e.get("mincluster_n_applied") is not None
            })),
            "reference_n": n_samples,
            "current_n": repr(sorted({
                e.get("current_n") for e in entries if e.get("current_n") is not None
            })),
        })
    effective_k_report_path = os.path.join(final_metrics_dir, "effective_k_report.csv")
    pd.DataFrame(effective_k_report_rows).to_csv(effective_k_report_path, index=False)


    # --- SVM classification ---
    # Optional post-clustering classifier. It predicts the final/in-view cluster
    # labels from the same feature representation used for clustering, mainly for
    # interpretability and out-of-fold uncertainty summaries.

    if getattr(args, 'DO_SVM', 'FALSE').upper() == 'TRUE':

        def _svm_features_for_modality(mod):
            """
            Match SVM features to the representation used for clustering.

            Modalities with explicit dimensionality reduction use their reduced
            numeric latent representation. Modalities configured as "none" keep
            the existing processed-feature behavior for interpretability.
            """
            method = args.dim_reduction_by_modality.get(mod, args.dim_reduction)
            method = _normalize_dim_reduction_method(method)
            if method != "none":
                latent = np.asarray(ae_res[mod]['final_latent'], dtype=np.float32)
                return pd.DataFrame(
                    latent,
                    columns=[f"{mod}__latent_{i + 1}" for i in range(latent.shape[1])],
                )

            df_mod = dict_final[mod]
            return (
                df_mod
                .drop(columns=[args.subject_id_column], errors='ignore')
                .reset_index(drop=True)
            )

        # Combine modality-level SVM features after selecting the appropriate
        # representation for each modality.
        X_train_list = [_svm_features_for_modality(mod) for mod in args.modalities]
        X_train = pd.concat(X_train_list, axis=1)


        # Use the final cluster labels as training labels
        clusters = final_labels
        clusters_indiv = indiv_labels

        def _svm_label_status(labels):
            """Handle svm label status."""
            if labels is None:
                return False, "labels are missing", {}
            labels = np.asarray(labels)
            valid = labels >= 0
            labels_valid = labels[valid]
            if labels_valid.size == 0:
                return False, "no valid labels", {}
            uniq, counts = np.unique(labels_valid, return_counts=True)
            counts_dict = {int(k): int(v) for k, v in zip(uniq, counts)}
            if len(uniq) < 2:
                return False, f"only {len(uniq)} cluster found", counts_dict
            min_count = int(np.min(counts))
            if min_count < 2:
                return False, f"minimum cluster size is {min_count}; need at least 2 per class for outer CV", counts_dict
            return True, "ok", counts_dict

        if clusters is None:
            print("Warning: Final consensus labels missing; skipping SVM classification.")
            results = None
            final_model = None
            metrics_merged = {
                'data': dict_final,
                'ae_res': ae_res,
                'final_labels': final_labels,
                'individual_labels': indiv_labels,
                'final_params': final_params,
                'preprocessing_details': preprocessing_details,
                'view_scores_per_view': view_scores_per_view,
                'view_quality_mean': view_score_mean,
                'final_quality': final_score,
                'cluster_pvalues': cluster_pvalues,
                'final_reporting': _final_reporting_with_runtime(),

                # Primary stability metrics
                'final_stability': final_stability_primary,
                'per_view_stabilities': per_view_stabilities_primary,
                'mean_view_stability': mean_view_stability_primary,
                'min_view_stability': min_view_stability_primary,

                # All stability variants
                'final_stability_ari': full_final_stab_ari,
                #'final_stability_coassoc': full_final_stab_coassoc,
                #'final_stability_CCC': full_final_stab_CCC,
                'final_stability_jaccard': full_final_stab_jaccard,
                'per_view_stabilities_ari': full_views_stab_ari,
                #'per_view_stabilities_coassoc': full_views_stab_coassoc,
                #'per_view_stabilities_CCC': full_views_stab_CCC,
                'per_view_stabilities_jaccard': full_views_stab_jaccard,
                # MATLAB-style consensus diagnostics (PAC/CCC)
                'final_stability_SUM_MAT_full': full_final_stab_SUM_MAT_full,
                'per_view_stabilities_SUM_MAT_full': full_v_stab_SUM_MAT_full,
                'mean_view_stability_MAT_CCC': full_mean_view_stab_MAT_CCC,
                'mean_view_stability_MAT_PAC': full_mean_view_stab_MAT_PAC

            }
            metrics_merged.update(_merge_extra_metrics())
            with open(output_final_metrics_path, 'wb') as f:
                dill.dump(metrics_merged, f)
            print(f"Outer metrics saved to {output_final_metrics_path}")
            return

        clusters = np.asarray(clusters)
        valid_mask = clusters >= 0
        clusters_valid = clusters[valid_mask]
        final_svm_ok, final_svm_reason, final_svm_counts = _svm_label_status(clusters)
        if not final_svm_ok:
            print(
                "Warning: Final labels are not suitable for SVM classification; "
                f"skipping SVM. Reason: {final_svm_reason}. Counts: {final_svm_counts}"
            )
            results = None
            final_model = None
            svm_feature_names = None
            svm_train_index = None
        else:
            print(f"SVM label counts for final clustering: {final_svm_counts}")
            print(f"Training SVM classifier on {len(set(clusters_valid))} clusters.")
            X_train_valid = X_train.loc[valid_mask].reset_index(drop=True)
            y_train = pd.Series(clusters_valid, name='cluster')

            try:
                results, final_model = SVM_nested_cv(X_train_valid, y_train)
            except ValueError as exc:
                print(f"Warning: Final SVM skipped: {exc}")
                results = None
                final_model = None

            # Persist feature names and row indices so notebook outputs can map
            # SVM importances/predictions back to the training matrix.
            svm_feature_names = list(X_train_valid.columns) if results is not None and isinstance(X_train_valid, pd.DataFrame) else None
            # Useful if you ever want to map OOF rows back to original rows on the server
            svm_train_index = X_train.loc[valid_mask].index.to_numpy() if results is not None and hasattr(X_train, "loc") else None


        results_modalities = []
        final_models_modalities = []
        svm_feature_names_modalities = []
        svm_train_index_modalities = []

        for i, mod in enumerate(args.modalities):
            if clusters_indiv is None or clusters_indiv[i] is None:
                print(f"Warning: No individual labels for modality {mod}; skipping SVM classification for this modality.")
                results_modalities.append(None)
                final_models_modalities.append(None)
                svm_feature_names_modalities.append(None)
                svm_train_index_modalities.append(None)
                continue

            labels_mod = np.asarray(clusters_indiv[i])
            valid_mask_mod = labels_mod >= 0
            labels_mod_valid = labels_mod[valid_mask_mod]
            modality_svm_ok, modality_svm_reason, modality_svm_counts = _svm_label_status(labels_mod)
            print(f"SVM label counts for modality {mod}: {modality_svm_counts}")

            if not modality_svm_ok:
                print(
                    f"Warning: Labels for modality {mod} are not suitable for SVM classification; "
                    f"skipping this modality. Reason: {modality_svm_reason}."
                )
                results_modalities.append(None)
                final_models_modalities.append(None)
                svm_feature_names_modalities.append(None)
                svm_train_index_modalities.append(None)
                continue

            print(f"Training SVM classifier on {len(set(labels_mod_valid))} clusters for modality {mod}")
            X_mod = _svm_features_for_modality(mod)
            X_train_mod = X_mod.loc[valid_mask_mod].reset_index(drop=True)
            y_train_mod = pd.Series(labels_mod_valid, name='cluster')

            try:
                results_mod, final_model_mod = SVM_nested_cv(X_train_mod, y_train_mod)
            except ValueError as exc:
                print(f"Warning: SVM skipped for modality {mod}: {exc}")
                results_mod = None
                final_model_mod = None

            results_modalities.append(results_mod)
            final_models_modalities.append(final_model_mod)

            if results_mod is None:
                svm_feature_names_modalities.append(None)
                svm_train_index_modalities.append(None)
            else:
                svm_feature_names_modalities.append(list(X_train_mod.columns) if isinstance(X_train_mod, pd.DataFrame) else None)
                svm_train_index_modalities.append(X_mod.loc[valid_mask_mod].index.to_numpy() if hasattr(X_mod, "loc") else None)

        # Store portable copies alongside raw SVM outputs. This keeps notebooks
        # robust when pandas/sklearn versions differ between server and laptop.
        def _pack_svm_results(res):
            """Convert pandas objects inside SVM results to plain python containers (more robust across machines)."""
            if res is None:
                return None

            out = dict(res)

            oof = out.get("oof_uncertainty", None)
            if isinstance(oof, pd.DataFrame):
                out["oof_uncertainty"] = oof.to_dict(orient="list")

            fim = out.get("feature_importance_mean", None)
            if isinstance(fim, pd.Series):
                out["feature_importance_mean"] = fim.to_dict()

            fis = out.get("feature_importance_std", None)
            if isinstance(fis, pd.Series):
                out["feature_importance_std"] = fis.to_dict()

            return out

        svm_results_packed = _pack_svm_results(results)
        svm_results_modalities_packed = [_pack_svm_results(r) for r in results_modalities]

        # Save outer metrics, including best individual's fitness and IDs
        metrics_merged = {
            'data': dict_final,
            'ae_res': ae_res,
            'final_labels': final_labels,
            'individual_labels': indiv_labels,
            'final_params': final_params,
            'preprocessing_details': preprocessing_details,
            'view_scores_per_view': view_scores_per_view,
            'view_quality_mean': view_score_mean,
            'final_quality': final_score,
            'cluster_pvalues': cluster_pvalues,
            'final_reporting': _final_reporting_with_runtime(),

            # Primary stability metrics
            'final_stability': final_stability_primary,
            'per_view_stabilities': per_view_stabilities_primary,
            'mean_view_stability': mean_view_stability_primary,
            'min_view_stability': min_view_stability_primary,

            # All stability variants
            'final_stability_ari': full_final_stab_ari,
            'final_stability_jaccard': full_final_stab_jaccard,
            'per_view_stabilities_ari': full_views_stab_ari,
            'per_view_stabilities_jaccard': full_views_stab_jaccard,
            'final_stability_SUM_MAT_full': full_final_stab_SUM_MAT_full,
            'per_view_stabilities_SUM_MAT_full': full_v_stab_SUM_MAT_full,
            'mean_view_stability_MAT_CCC': full_mean_view_stab_MAT_CCC,
            'mean_view_stability_MAT_PAC': full_mean_view_stab_MAT_PAC,

            # --- SVM outputs (raw) ---
            'svm_results': results,
            'svm_final_model': final_model,
            'svm_results_modalities': results_modalities,
            'svm_final_models_modalities': final_models_modalities,

            # SVM metadata needed for downstream notebooks/reporting.
            'svm_feature_names': svm_feature_names,
            'svm_train_index': svm_train_index,
            'svm_feature_names_modalities': svm_feature_names_modalities,
            'svm_train_index_modalities': svm_train_index_modalities,

            # Packed/portable copies of SVM result summaries.
            'svm_results_packed': svm_results_packed,
            'svm_results_modalities_packed': svm_results_modalities_packed,
        }

        metrics_merged.update(_merge_extra_metrics())
        with open(output_final_metrics_path, 'wb') as f:
            dill.dump(metrics_merged, f)

        print(f"Outer metrics saved to {output_final_metrics_path}")
        return

    else:
        print("SVM classification not requested; skipping SVM step.")
        metrics_merged = {
            'data': dict_final,
            'ae_res': ae_res,
            'final_labels': final_labels,
            'individual_labels': indiv_labels,
            'final_params': final_params,
            'preprocessing_details': preprocessing_details,
            'view_scores_per_view': view_scores_per_view,
            'view_quality_mean': view_score_mean,
            'final_quality': final_score,
            'cluster_pvalues': cluster_pvalues,
            'final_reporting': _final_reporting_with_runtime(),

            # Primary stability metrics
            'final_stability': final_stability_primary,
            'per_view_stabilities': per_view_stabilities_primary,
            'mean_view_stability': mean_view_stability_primary,
            'min_view_stability': min_view_stability_primary,

            # All stability variants
            'final_stability_ari': full_final_stab_ari,
            #'final_stability_coassoc': full_final_stab_coassoc,
            #'final_stability_CCC': full_final_stab_CCC,
            'final_stability_jaccard': full_final_stab_jaccard,
            'per_view_stabilities_ari': full_views_stab_ari,
            #'per_view_stabilities_coassoc': full_views_stab_coassoc,
            #'per_view_stabilities_CCC': full_views_stab_CCC,
            'per_view_stabilities_jaccard': full_views_stab_jaccard,
            # MATLAB-style consensus diagnostics (PAC/CCC)
            'final_stability_SUM_MAT_full': full_final_stab_SUM_MAT_full,
            'per_view_stabilities_SUM_MAT_full': full_v_stab_SUM_MAT_full,
            'mean_view_stability_MAT_CCC': full_mean_view_stab_MAT_CCC,
            'mean_view_stability_MAT_PAC': full_mean_view_stab_MAT_PAC
        }
        metrics_merged.update(_merge_extra_metrics())
        with open(output_final_metrics_path, 'wb') as f:
            dill.dump(metrics_merged, f)
        print(f"Outer metrics saved to {output_final_metrics_path}")
        return



# --- Mode: init population ---
def do_init(args):
    """
    Initialize one fold's GA population.

    Each individual has one pair of genes per modality (`k`, linkage method),
    followed by the pre-fusion linkage, final cluster count, and fusion method.
    Later stages attach fitness and reporting attributes to these individuals.
    """
    base_dir = os.path.abspath(getattr(args, "base_dir", "."))
    ga_root = _ga_root(base_dir, args.fold_index if hasattr(args, "fold_index") else 0)
    population_file = _resolve_path(base_dir, args.population_file) or os.path.join(
        ga_root, f"population_init_fold{getattr(args, 'fold_index', 0)}.pkl"
    )
    # Register one gene generator per clustering hyperparameter.
    toolbox = base.Toolbox()
    n_views = len(args.modalities)
    fusion_methods = list(args.fusion_methods)
    if not fusion_methods:
        raise ValueError("At least one fusion method must be supplied via --fusion_methods.")
    #linkages = ['complete','average','weighted']
    linkages = list(args.linkages)
    k_min, k_max = args.k_min, args.k_max

    # Seed RNGs so each fold's init is reproducible but can differ by wrapper.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        if torch is not None:
            torch.manual_seed(args.seed)


    names = []

    for i in range(n_views):
        toolbox.register(f"c_{i+1}_k", random.randint, k_min, k_max)
        names.append(f"c_{i+1}_k")
        toolbox.register(f"c_{i+1}_method", random.choice, linkages)
        names.append(f"c_{i+1}_method")
    toolbox.register("pre_method", random.choice, linkages)
    names.append("pre_method")
    toolbox.register("k_final", random.randint, k_min, k_max)
    names.append("k_final")
    toolbox.register("fusion_method", random.choice, fusion_methods)
    names.append("fusion_method")

    # Create DEAP individual and population initializers from the registered genes.
    to_pass = tuple(getattr(toolbox, name) for name in names)
    toolbox.register("individual", tools.initCycle, creator.Individual, to_pass, 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Generate and save the initial unevaluated population.
    pop = toolbox.population(n=args.n_population)
    # Attach gene_names metadata so convert_to_parameters can decode individuals
    # without relying on positional knowledge elsewhere in the pipeline.
    for ind in pop:
        ind.gene_names = names

    os.makedirs(os.path.dirname(population_file) or '.', exist_ok=True)
    with open(population_file, 'wb') as f:
        dill.dump(pop, f)
    print(f"Initial population ({len(pop)}) saved to {population_file}")
    return



def do_test1(args):
    """
    Test mode 1: Run only KMeans on preprocessed data, skipping AE and Parea.
    """
    df = pd.read_csv(args.input_csv)
    meta = pd.read_csv(args.meta_csv)
    if args.n_folds == 1:
        # No CV split: use all rows for training to allow fast synthetic-data tests
        train_df = df.reset_index(drop=True)
    else:
        raise ValueError("TEST mode only supports n_folds=1 for fast testing on synthetic data")

    # Preprocess data
    ae_data, subject_id_list, dict_final = preprocessing(
        train_df, meta,
        subject_id_column=args.subject_id_column,
        col_threshold=args.col_threshold,
        row_threshold=args.row_threshold,
        skew_threshold=args.skew_threshold,
        scaler_type=args.scaler_type,
        modalities=args.modalities,
        dummy_code_modalities=args.dummy_code_modalities,
        mixed_categorical_modalities=args.mixed_categorical_modalities
    )

    print("TEST mode 1: Running KMeans on preprocessed data...")
    # Extract preprocessed & scaled per-modality dataframes
    data_list = [
        dict_final[mod].drop(columns=[args.subject_id_column]).to_numpy(dtype=np.float32, copy=True)
        for mod in args.modalities
    ]

    # Keep subject order from preprocessing (should be identical across modalities)
    # Use the IDs from the first modality as reference order
    ref_ids = dict_final[args.modalities[0]][args.subject_id_column].to_numpy()

    # Run KMeans clustering independently on each modality and store labels
    labels_list = []
    for i, X in enumerate(data_list):
        # Fixed cluster counts for synthetic test (adjust as needed)
        if i == 0:
            k = 3
        elif i == 1:
            k = 4
        elif i == 2:
            k = 2
        km = KMeans(n_clusters=k, random_state=42)
        lab = km.fit_predict(X)
        labels_list.append(lab)
        counts = np.bincount(lab)
        print(f"Modality {args.modalities[i]}: KMeans found {len(np.unique(lab))} clusters with sizes {counts}")

    # If in TEST mode, compute ARI against ground truth labels with proper alignment
    if getattr(args, 'TEST', 'FALSE').upper() == "TRUE":
        print("TEST mode: computing Adjusted Rand Index against ground truth labels.")
        df_truth = pd.read_csv("path/to/multiclust/synthetic_multimodal_spartan.csv")
        # Align truth to the exact subject order used during preprocessing
        # Assumes the same subject ID column exists in df_truth
        truth_map = df_truth.set_index(args.subject_id_column)
        # Build truth arrays in the same order as ref_ids
        true_m1 = truth_map.loc[ref_ids, "subgroup_m1"].to_numpy()
        true_m2 = truth_map.loc[ref_ids, "subgroup_m2"].to_numpy()
        true_m3 = truth_map.loc[ref_ids, "subgroup_m3"].to_numpy()
        truth_cols = [true_m1, true_m2, true_m3]

        for i, pred in enumerate(labels_list):
            true_labels = truth_cols[i]
            # Map true labels (strings) to integers deterministically
            uniq = np.unique(true_labels)
            l2i = {name: idx for idx, name in enumerate(uniq)}
            true_ints = np.array([l2i[v] for v in true_labels], dtype=int)
            ari = adjusted_rand_score(true_ints, pred)
            print(f"Adjusted Rand Index modality {i} ({args.modalities[i]}): {ari:.3f}")
    return


def do_test2(args):
    """
    Test mode 2: Run only Spectral Clustering on preprocessed data, skipping AE and Parea.
    """
    df = pd.read_csv(args.input_csv)
    meta = pd.read_csv(args.meta_csv)
    if args.n_folds == 1:
        # No CV split: use all rows for training to allow fast synthetic-data tests
        train_df = df.reset_index(drop=True)
    else:
        raise ValueError("TEST mode only supports n_folds=1 for fast testing on synthetic data")

    # Preprocess data
    ae_data, subject_id_list, dict_final = preprocessing(
        train_df, meta,
        subject_id_column=args.subject_id_column,
        col_threshold=args.col_threshold,
        row_threshold=args.row_threshold,
        skew_threshold=args.skew_threshold,
        scaler_type=args.scaler_type,
        modalities=args.modalities,
        dummy_code_modalities=args.dummy_code_modalities,
        mixed_categorical_modalities=args.mixed_categorical_modalities
    )

    print("TEST mode 2: Running Spectral clustering on preprocessed data...")
    # Extract preprocessed & scaled per-modality dataframes
    data_list = [
        dict_final[mod].drop(columns=[args.subject_id_column]).to_numpy(dtype=np.float32, copy=True)
        for mod in args.modalities
    ]

    # Keep subject order from preprocessing (should be identical across modalities)
    # Use the IDs from the first modality as reference order
    ref_ids = dict_final[args.modalities[0]][args.subject_id_column].to_numpy()

    # Run KMeans clustering independently on each modality and store labels
    labels_list = []
    for i, X in enumerate(data_list):
        # Fixed cluster counts for synthetic test (adjust as needed)
        if i == 0:
            k = 3
        elif i == 1:
            k = 4
        elif i == 2:
            k = 2
        km = SpectralClustering(n_clusters=k, n_init=10, gamma=1.0, n_neighbors=10, eigen_tol=0.0, degree=3, coef0=1, verbose=False, assign_labels='kmeans', affinity='nearest_neighbors')
        lab = km.fit_predict(X)
        labels_list.append(lab)
        counts = np.bincount(lab)
        print(f"Modality {args.modalities[i]}: Spectral Clustering found {len(np.unique(lab))} clusters with sizes {counts}")

    # If in TEST mode, compute ARI against ground truth labels with proper alignment
    if getattr(args, 'TEST', 'FALSE').upper() == "TRUE":
        print("TEST mode: computing Adjusted Rand Index against ground truth labels.")
        df_truth = pd.read_csv("path/to/multiclust/synthetic_multimodal_spartan.csv")
        # Align truth to the exact subject order used during preprocessing
        # Assumes the same subject ID column exists in df_truth
        truth_map = df_truth.set_index(args.subject_id_column)
        # Build truth arrays in the same order as ref_ids
        true_m1 = truth_map.loc[ref_ids, "subgroup_m1"].to_numpy()
        true_m2 = truth_map.loc[ref_ids, "subgroup_m2"].to_numpy()
        true_m3 = truth_map.loc[ref_ids, "subgroup_m3"].to_numpy()
        truth_cols = [true_m1, true_m2, true_m3]

        for i, pred in enumerate(labels_list):
            true_labels = truth_cols[i]
            # Map true labels (strings) to integers deterministically
            uniq = np.unique(true_labels)
            l2i = {name: idx for idx, name in enumerate(uniq)}
            true_ints = np.array([l2i[v] for v in true_labels], dtype=int)
            ari = adjusted_rand_score(true_ints, pred)
            print(f"Adjusted Rand Index modality {i} ({args.modalities[i]}): {ari:.3f}")
    return


def do_test3(args):
    """
    Test mode 3: Test fusion matrix construction from individual labels.
    """

    df = pd.read_csv(args.input_csv)
    meta = pd.read_csv(args.meta_csv)
    if args.n_folds == 1:
        # No CV split: use all rows for training to allow fast synthetic-data tests
        train_df = df.reset_index(drop=True)
    else:
        raise ValueError("TEST mode only supports n_folds=1 for fast testing on synthetic data")

    # Preprocess data
    ae_data, subject_id_list, dict_final = preprocessing(
        train_df, meta,
        subject_id_column=args.subject_id_column,
        col_threshold=args.col_threshold,
        row_threshold=args.row_threshold,
        skew_threshold=args.skew_threshold,
        scaler_type=args.scaler_type,
        modalities=args.modalities,
        dummy_code_modalities=args.dummy_code_modalities,
        mixed_categorical_modalities=args.mixed_categorical_modalities
    )

    # Extract preprocessed & scaled per-modality dataframes
    data_list = [
        dict_final[mod].drop(columns=[args.subject_id_column]).to_numpy(dtype=np.float32, copy=True)
        for mod in args.modalities
    ]

    # Keep subject order from preprocessing (should be identical across modalities)
    # Use the IDs from the first modality as reference order
    ref_ids = dict_final[args.modalities[0]][args.subject_id_column].to_numpy()

    k_s=[3,4,2]

    n_views = len(data_list)

    clustering_algorithms = [None] * n_views

    for i in range(n_views):
        clustering_algorithms[i] = clusterer(
            'ensemble',
            n_clusters=k_s[i],
            precomputed=False,
            linkage_method='average',
            random_state=42,
            final=False
        )

    # Create the views. Initiates views as a list of view objects. Each view links one dataset and one clustering algorithm.
    views = [view(data_list[i], clustering_algorithms[i]) for i in range(n_views)]

    fusion_methods = list(args.fusion_methods) or DEFAULT_FUSION_METHODS

    for fusion_method in fusion_methods:
        print(f"Testing fusion method: {fusion_method}")

        # Create fusion algorithm
        f = fuser(fusion_method)

        # Compute fusion matrix by executing the ensemble of views directly
        fusion_matrix, individual_labels = execute_ensemble(views, f)

        print(f"Fusion matrix shape: {fusion_matrix.shape}")
        print(f"Shape of individual_labels: {len(individual_labels)} modalities, each with {len(individual_labels[0])} samples")

        # Inspect the fusion distance matrix on symmetry
        if np.allclose(fusion_matrix, fusion_matrix.T):
            print("Fusion matrix is symmetric.")
        else:
            print("Warning: Fusion matrix is not symmetric.")

        # Inspect the fusion matrix on zeros on diagonal
        if np.all(np.diag(fusion_matrix) == 0):
            print("Fusion matrix has zeros on its diagonal.")
        else:
            print("Warning: Fusion matrix diagonal has non-zero entries.")

        # Inspect values in the fusion matrix
        print(f"Fusion matrix values range from {np.min(fusion_matrix)} to {np.max(fusion_matrix)}")

        # If consensus, count singletons
        if fusion_method == 'consensus':
            # Reconstruct strict-intersection consensus labels (mirror of Consensus.execute)
            labs = [np.asarray(x) for x in individual_labels]
            n_samp = len(labs[0])
            n_cl = len(labs)
            cl_cons = np.zeros(n_samp, dtype=int)
            k = 1
            for i in range(n_samp):
                ids = np.where(labs[0] == labs[0][i])[0]
                for j in range(1, n_cl):
                    m = np.where(labs[j] == labs[j][i])[0]
                    ids = np.intersect1d(ids, m)
                if np.sum(cl_cons[ids]) == 0:
                    cl_cons[ids] = k
                    k += 1
            # Count true singletons from cl_cons
            _, counts = np.unique(cl_cons, return_counts=True)
            singleton_count = int(np.sum(counts == 1))
            print(f"Consensus produced {len(counts)} consensus clusters; true singletons = {singleton_count}.")


        # Compare clusters to true labels
        # If in TEST mode, compute ARI against ground truth labels with proper alignment
        if getattr(args, 'TEST', 'FALSE').upper() == "TRUE":
            print("TEST mode: computing Adjusted Rand Index against ground truth labels.")
            df_truth = pd.read_csv("path/to/multiclust/synthetic_multimodal_spartan.csv")
            # Align truth to the exact subject order used during preprocessing
            # Assumes the same subject ID column exists in df_truth
            truth_map = df_truth.set_index(args.subject_id_column)
            # Build truth arrays in the same order as ref_ids
            true_m1 = truth_map.loc[ref_ids, "subgroup_m1"].to_numpy()
            true_m2 = truth_map.loc[ref_ids, "subgroup_m2"].to_numpy()
            true_m3 = truth_map.loc[ref_ids, "subgroup_m3"].to_numpy()
            truth_cols = [true_m1, true_m2, true_m3]

            for i, pred in enumerate(individual_labels):
                true_labels = truth_cols[i]
                # Map true labels (strings) to integers deterministically
                uniq = np.unique(true_labels)
                l2i = {name: idx for idx, name in enumerate(uniq)}
                true_ints = np.array([l2i[v] for v in true_labels], dtype=int)
                ari = adjusted_rand_score(true_ints, pred)
                print(f"Adjusted Rand Index modality {i} ({args.modalities[i]}): {ari:.3f}")


    sys.exit(0)


def do_test4(args):
    """
    Test mode 4: Test full clustering pipeline (without VAE, without genetic algorithm etc)
    """
    df = pd.read_csv(args.input_csv)
    meta = pd.read_csv(args.meta_csv)
    if args.n_folds == 1:
        # No CV split: use all rows for training to allow fast synthetic-data tests
        train_df = df.reset_index(drop=True)
    else:
        raise ValueError("TEST mode only supports n_folds=1 for fast testing on synthetic data")

    # Preprocess data
    ae_data, subject_id_list, dict_final = preprocessing(
        train_df, meta,
        subject_id_column=args.subject_id_column,
        col_threshold=args.col_threshold,
        row_threshold=args.row_threshold,
        skew_threshold=args.skew_threshold,
        scaler_type=args.scaler_type,
        modalities=args.modalities,
        dummy_code_modalities=args.dummy_code_modalities,
        mixed_categorical_modalities=args.mixed_categorical_modalities
    )

    # Extract preprocessed & scaled per-modality dataframes
    data_list = [
        dict_final[mod].drop(columns=[args.subject_id_column]).to_numpy(dtype=np.float32, copy=True)
        for mod in args.modalities
    ]

    # Keep subject order from preprocessing (should be identical across modalities)
    # Use the IDs from the first modality as reference order
    ref_ids = dict_final[args.modalities[0]][args.subject_id_column].to_numpy()

    k_s=[3,4,2]

    n_views = len(data_list)

    clustering_algorithms = [None] * n_views

    for i in range(n_views):
        clustering_algorithms[i] = clusterer(
            'ensemble',
            n_clusters=k_s[i],
            precomputed=False,
            linkage_method='average',
            random_state=42,
            final=False
        )

    # Create the views. Initiates views as a list of view objects. Each view links one dataset and one clustering algorithm.
    views = [view(data_list[i], clustering_algorithms[i]) for i in range(n_views)]

    fusion_methods = list(args.fusion_methods) or DEFAULT_FUSION_METHODS

    for fusion_method in fusion_methods:
        print(f"Testing fusion method: {fusion_method}")

        # Create fusion algorithm
        f = fuser(fusion_method)

        # Compute fusion matrix by executing the ensemble of views directly
        fusion_matrix, individual_labels = execute_ensemble(views, f)



        # Compare clusters to true labels
        # If in TEST mode, compute ARI against ground truth labels with proper alignment
        if getattr(args, 'TEST', 'FALSE').upper() == "TRUE":
            print("TEST mode: computing Adjusted Rand Index against ground truth labels.")
            df_truth = pd.read_csv("path/to/multiclust/synthetic_multimodal_spartan.csv")
            # Align truth to the exact subject order used during preprocessing
            # Assumes the same subject ID column exists in df_truth
            truth_map = df_truth.set_index(args.subject_id_column)
            # Build truth arrays in the same order as ref_ids
            true_m1 = truth_map.loc[ref_ids, "subgroup_m1"].to_numpy()
            true_m2 = truth_map.loc[ref_ids, "subgroup_m2"].to_numpy()
            true_m3 = truth_map.loc[ref_ids, "subgroup_m3"].to_numpy()
            truth_cols = [true_m1, true_m2, true_m3]

            for i, pred in enumerate(individual_labels):
                true_labels = truth_cols[i]
                # Map true labels (strings) to integers deterministically
                uniq = np.unique(true_labels)
                l2i = {name: idx for idx, name in enumerate(uniq)}
                true_ints = np.array([l2i[v] for v in true_labels], dtype=int)
                ari = adjusted_rand_score(true_ints, pred)
                print(f"Adjusted Rand Index modality {i} ({args.modalities[i]}): {ari:.3f}")


        # Final clustering on the fused distance matrix
        k_final=8
        v_res = fusion_matrix

        if not k_final:
            raise ValueError(
                "k_final must be provided for the second-step ensemble. "
                "Pass an explicit number of clusters to apply the predefined ensemble."
            )

        # Final clustering: use ensemble on the fused **distance** matrix
        c_final = clusterer(
            'ensemble',
            precomputed=True,
            n_clusters=k_final,
            linkage='average'
        )

        v_res_final = view(v_res, c_final)
        final_labels = v_res_final.execute()

        # Normalize label shape to 1D robustly
        final_labels = np.asarray(final_labels)
        if final_labels.ndim == 2:
            if final_labels.shape[1] == 1:
                final_labels = final_labels.ravel()
            else:
                # If multiple cuts/columns are returned, take the last column
                final_labels = final_labels[:, -1]
        elif final_labels.ndim != 1:
            final_labels = final_labels.reshape(-1)

        # Quality metric: Silhouette only (normalized to [0,1]).
        # For precomputed distances we use metric='precomputed'; for feature matrices we use the standard silhouette.
        def compute_quality(mat, labels, precomputed=False):
            """Calculate quality."""
            labels = np.asarray(labels)
            if len(np.unique(labels)) <= 1:
                return 0.0
            if precomputed:
                sil = silhouette_score(mat, labels, metric='precomputed')  # raw in [-1, 1]
            else:
                sil = silhouette_score(mat, labels)  # raw in [-1, 1]
            sil_n = (sil + 1.0) / 2.0  # normalize to [0,1]
            return float(sil_n)

        # View-level quality averaged across all views
        view_scores_per_view = [
            compute_quality(data_list[v], individual_labels[v], precomputed=False)
            for v in range(n_views)
        ]
        view_score = float(np.mean(view_scores_per_view))
        # Final consensus quality on the fused (precomputed distance) matrix
        final_score = compute_quality(v_res, final_labels, precomputed=True)

        print(f"Final clustering produced {len(np.unique(final_labels))} clusters with sizes {np.bincount(final_labels)}")
        print(f"View-level quality scores for fusion method {fusion_method}: {view_scores_per_view}")
        print(f"Mean view-level quality: {view_score:.4f}, final clustering quality: {final_score:.4f} for fusion method: {fusion_method}")

    sys.exit(0)



# --- Command-line entry point ------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data and preprocessing arguments.
    parser.add_argument('--input_csv', default='cleaned_discovery_data.csv')
    parser.add_argument('--meta_csv', default='merged_meta.csv')
    parser.add_argument('--base_dir', default='path/to/multiclust')
    parser.add_argument('--subject_id_column', default='src_subject_id')
    parser.add_argument('--col_threshold', type=float, default=0.5)
    parser.add_argument('--row_threshold', type=float, default=0.5)
    parser.add_argument('--skew_threshold', type=float, default=0.75)
    parser.add_argument('--scaler_type', default='robust')
    parser.add_argument('--modalities', nargs='+', default=['Internalising', 'Functioning', 'Cognition', 'Detachment', 'Psychoticism'])
    parser.add_argument('--dummy_code_modalities', nargs='*', default=None,
                        help='Subset of modalities whose raw columns should receive dummy/ordinal encoding. Defaults to all requested modalities.')
    parser.add_argument('--mixed_categorical_modalities', nargs='*', default=None,
                        help='Modalities to leave as mixed categorical/binary/numeric tables during preprocessing. Use with FAMD/MCA dim reduction.')
    parser.add_argument('--dim_reduction', choices=[None, "None", "none", 'VAE', 'vae', 'AE', 'ae', 'PCA', 'pca', 'SparseNMF', 'sparsenmf', 'Sparse_NMF', 'sparse_nmf', 'SNMF', 'snmf', 'FAMD', 'famd', 'MCA', 'mca', 'MIXED_SVD', 'mixed_svd'], default='VAE', help='Default dimensionality reduction method to use for modalities without an explicit override.')
    parser.add_argument('--dim_reduction_by_modality', nargs='*', default=None,
                        help='Optional per-modality overrides in the form Modality=Method, e.g. Internalising=PCA Psychoticism=None.')
    parser.add_argument('--pca_variance_threshold', type=float, default=None,
                        help='For PCA modalities, retain the smallest number of components reaching this explained-variance fraction, e.g. 0.95. If omitted, keep the legacy cap of up to 50 components.')
    parser.add_argument('--snmf_n_components', type=int, default=20,
                        help='Number of components for SparseNMF modalities, bounded by sample and feature count.')
    parser.add_argument('--snmf_alpha', type=float, default=0.1,
                        help='L1/L2 regularization strength for SparseNMF W and H matrices.')
    parser.add_argument('--snmf_l1_ratio', type=float, default=1.0,
                        help='SparseNMF regularization mix: 1.0 is L1, 0.0 is L2.')
    parser.add_argument('--snmf_max_iter', type=int, default=1000,
                        help='Maximum iterations for SparseNMF fitting.')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128,256,512])
    parser.add_argument('--activation_functions', nargs='+', default=['ReLU','LeakyReLU','selu','swish'])
    parser.add_argument('--learning_rates', nargs='+', type=float, default=[0.001,0.0001])
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[32,64,128])
    parser.add_argument('--latent_dims', nargs='+', type=int, default=[2,5,10])
    parser.add_argument('--k_min', type=int, default=2)
    parser.add_argument('--k_max', type=int, default=10)
    parser.add_argument('--linkages', type=str, nargs='+', default=['complete','average','weighted'])
    parser.add_argument('--n_population', type=int, default=100)
    parser.add_argument('--n_generations', type=int, default=10)
    parser.add_argument('--optimisation', choices=['single','multi'], default='multi')
    parser.add_argument('--ga_objectives', nargs='+', default=None,
                        help='Objectives optimised by GA (tokens such as mean_view_stability, mean_view_quality, '
                             'final_stability, final_quality, min_view_stability, min_view_quality).')
    parser.add_argument('--fusion_methods', nargs='+', default=DEFAULT_FUSION_METHODS,
                        help='Fusion methods available to the GA (e.g., agreement consensus disagreement).')
    parser.add_argument('--n_bootstrap', type=int, default=100)
    parser.add_argument('--bootstrap_mode', choices=['bootstrap','subsample'], default='subsample')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--output_pkl', default='pipeline_results.pkl')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel workers for bootstrap clustering')
    parser.add_argument('--bootstrap_jobs', type=int, default=None,
                        help='Optional worker cap for final merge stability bootstraps. Defaults to --n_jobs.')
    parser.add_argument('--final_bootstrap_preprocessing', choices=['outside', 'inside', 'both'], default='outside',
                        help='Final merge stability bootstrap mode: outside uses fixed full-data representations; inside reruns preprocessing/dimensionality reduction inside each bootstrap; both reports both and uses inside as primary.')
    parser.add_argument('--TEST', choices=['TRUE', 'FALSE'], default='FALSE',
                        help='Enable synthetic-data diagnostics in selected stages. Does not change dimensionality reduction by itself.')
    parser.add_argument('--max_missing_bootstraps', type=int, default=5,
                        help='Maximum number of missing bootstrap label files allowed before gather aborts')
    parser.add_argument('--mincluster', default="TRUE", help='Enforce minimum cluster size of 10 in final clustering (True/False)', choices=['TRUE','FALSE'])
    parser.add_argument('--mincluster_n', type=int, default=10, help='Minimum cluster size to enforce in final clustering')
    parser.add_argument('--mincluster_resample_mode', choices=['fixed', 'scaled'], default='fixed',
                        help='Use a fixed minimum cluster size or scale it to the current resample size.')
    parser.add_argument('--use_effective_k_for_fold_merge', choices=['TRUE', 'FALSE'], default='FALSE',
                        help='Vote across folds using bootstrap-derived effective k values.')
    parser.add_argument('--use_cross_fold_effective_k_for_final_run', choices=['TRUE', 'FALSE'], default='FALSE',
                        help='Use cross-fold effective cluster counts in full-data bootstraps and consensus cuts.')
    parser.add_argument('--internal_ensemble_enabled', choices=['TRUE', 'FALSE'], default='FALSE',
                        help='Use balanced perturbed base clusterings inside each ensemble instead of the original fixed five-method ensemble.')
    parser.add_argument('--internal_ensemble_bcs', type=int, default=5,
                        help='Number of internal base clusterings when --internal_ensemble_enabled=TRUE.')
    parser.add_argument('--internal_ensemble_sample_frac', type=float, default=0.8,
                        help='Fraction of subjects sampled without replacement for each internal base clustering.')
    parser.add_argument('--internal_ensemble_feature_frac', type=float, default=1.0,
                        help='Fraction of features sampled without replacement for each internal base clustering.')
    # Pipeline stage selector. Production runs normally execute:
    # init -> repeated bootstrap/gather generations -> outer -> merge.
    parser.add_argument('--mode', choices=['bootstrap','gather','outer','init', 'merge', 'test1', 'test2', 'test3', 'test4', 'test5'], default='init')
    parser.add_argument('--generation', type=int, help='GA generation index')
    parser.add_argument('--population_file', type=str)
    parser.add_argument('--seed',            type=int, default=None,
                    help='Random seed for GA init (only used with --mode init)')
    parser.add_argument('--population_dir', type=str, help='Directory where population files are stored')
    parser.add_argument('--population_initial_file', type=str, help='File to load initial population from in bootstrap mode')
    parser.add_argument('--bootstrap_index', type=int)
    parser.add_argument('--bootstrap_dir', type=str)
    parser.add_argument('--output_labels', type=str, help='Where to save bootstrap labels for stability computation')
    parser.add_argument('--output_population', type=str)
    parser.add_argument('--fold_index', type=int)
    parser.add_argument('--output_metrics', type=str)
    parser.add_argument('--output_final_metrics', type=str)
    parser.add_argument('--compute_cluster_pvalues', choices=['TRUE', 'FALSE'], default='FALSE',
                        help='Compute permutation-based p-values for each modality and final cluster solution in merge mode.')
    parser.add_argument('--cluster_pvalue_mode', choices=['fast', 'full'], default='fast',
                        help="Legacy option retained for compatibility. Quality p-values now shuffle labels on the fixed observed data; ARI p-values also use label shuffling.")
    parser.add_argument('--cluster_pvalue_stat', choices=['composite', 'silhouette'], default='composite',
                        help='Cluster-separation statistic used for permutation p-values.')
    parser.add_argument('--cluster_pvalue_permutations', type=int, default=200,
                        help='Number of permutations for cluster p-value estimation.')
    parser.add_argument('--cluster_pvalue_permutations_quality', type=int, default=None,
                        help='Number of permutations for quality-score p-value estimation (defaults to --cluster_pvalue_permutations).')
    parser.add_argument('--cluster_pvalue_permutations_ari', type=int, default=None,
                        help='Number of permutations for ARI-stability p-value estimation (defaults to --cluster_pvalue_permutations).')
    parser.add_argument('--cluster_pvalue_jobs', type=int, default=0,
                        help='Parallel workers for permutation p-values (0 -> use --n_jobs setting).')
    parser.add_argument('--cluster_pvalue_seed', type=int, default=314159,
                        help='Base RNG seed for permutation p-value computation.')
    parser.add_argument('--TEST-phase', choices=[0,1,2,3,4], type=int, default=0,
                        help='For TEST mode: which phase to run (0=Full pipeline, 1=Only Kmeans, 2=Only Spectral, 3=Test fusion matrix, 4=Test final clustering, 5=Test only individual labels but full ensemble.)')
    parser.add_argument('--ga_cxpb', type=float, default=0.7,
                        help='Crossover probability used during GA gather stage.')
    parser.add_argument('--ga_mutpb', type=float, default=0.2,
                        help='Mutation probability used during GA gather stage.')
    parser.add_argument('--ga_elitism', type=int, default=2,
                        help='Number of elite individuals to carry over each generation.')
    parser.add_argument('--DO_SVM', choices=['TRUE', 'FALSE'], default='FALSE',
                        help='In OUTER mode, whether to run SVM classification on the final clustering labels (TRUE/FALSE).')
    args = parser.parse_args()

    # Map activation strings to actual nn modules
    act_map = {
        "ReLU": nn.ReLU(),
        "LeakyReLU": nn.LeakyReLU(),
        "selu": nn.SELU(),
        "swish": nn.SiLU()
    }
    activation_functions = {
        name: act_map[name]
        for name in args.activation_functions
        if name in act_map
    }

    args.dim_reduction = _normalize_dim_reduction_method(args.dim_reduction)
    args.dummy_code_modalities = _parse_dummy_code_modalities(
        getattr(args, "dummy_code_modalities", None),
        args.modalities,
    )
    args.mixed_categorical_modalities = _parse_dummy_code_modalities(
        getattr(args, "mixed_categorical_modalities", None),
        args.modalities,
    ) if getattr(args, "mixed_categorical_modalities", None) is not None else []
    args.dim_reduction_by_modality = _parse_dim_reduction_overrides(
        getattr(args, "dim_reduction_by_modality", None),
        args.modalities,
        args.dim_reduction,
    )
    if args.pca_variance_threshold is not None and not (0.0 < args.pca_variance_threshold <= 1.0):
        raise ValueError("--pca_variance_threshold must be in the interval (0, 1].")
    if args.snmf_n_components < 1:
        raise ValueError("--snmf_n_components must be >= 1.")
    if args.snmf_alpha < 0:
        raise ValueError("--snmf_alpha must be >= 0.")
    if not (0.0 <= args.snmf_l1_ratio <= 1.0):
        raise ValueError("--snmf_l1_ratio must be in the interval [0, 1].")
    if args.snmf_max_iter < 1:
        raise ValueError("--snmf_max_iter must be >= 1.")
    _validate_mixed_categorical_dim_reduction(
        args.mixed_categorical_modalities,
        args.dim_reduction_by_modality,
    )

    args.ga_objectives = _normalize_objective_tokens(getattr(args, "ga_objectives", []), args.optimisation)
    args.fusion_methods = _normalize_method_list(getattr(args, "fusion_methods", DEFAULT_FUSION_METHODS))
    if not args.fusion_methods:
        args.fusion_methods = DEFAULT_FUSION_METHODS.copy()

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
    elif args.mode == 'test1':
        do_test1(args)
    elif args.mode == 'test2':
        do_test2(args)
    elif args.mode == 'test3':
        do_test3(args)
    elif args.mode == 'test4':
        do_test4(args)
    elif args.mode == 'test5':
        do_test5(args)
    else:
        parser.error(f"Unknown mode: {args.mode}")


