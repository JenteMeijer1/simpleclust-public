#!/usr/bin/env python3
"""Finalize a forced-k alternative simpleclust solution from existing search outputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from types import SimpleNamespace

import dill
import numpy as np
import pandas as pd
from deap import creator

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from effective_k import summarize_effective_k
from singleclust import full_pipeline_singleclust as fsp
from clustering_functions import decode_search_candidate


def _normalize_dim_reduction(value):
    if value is None:
        return "None"
    text = str(value).strip()
    if text.lower() in {"sparse_nmf", "sparse-nmf", "snmf", "sparsenmf"}:
        return "sparsenmf"
    return text


def _candidate_file(intermediates_dir, fold_index, generation):
    return os.path.join(
        intermediates_dir,
        f"fold{fold_index}",
        "ga",
        f"population_fold{fold_index}_gen{generation + 1}.pkl",
    )


def _fitness_values(candidate):
    values = getattr(getattr(candidate, "fitness", None), "values", ())
    try:
        return tuple(float(v) for v in values)
    except Exception:
        return ()


def _candidate_metric(candidate, key, objective_index=None):
    summary = getattr(candidate, "metrics_summary", None) or {}
    value = summary.get(key)
    if value is None and key == "quality":
        value = getattr(candidate, "qual", None)
    if value is None and key in {"stab_ari", "stab_jaccard"}:
        value = getattr(candidate, "stab", None)
    if value is None and objective_index is not None:
        values = _fitness_values(candidate)
        if objective_index < len(values):
            value = values[objective_index]
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = np.nan
    return value


def _rank_candidates(rows):
    finite_stab = [row["stability"] for row in rows if np.isfinite(row["stability"])]
    finite_qual = [row["quality"] for row in rows if np.isfinite(row["quality"])]
    stab_min, stab_max = (min(finite_stab), max(finite_stab)) if finite_stab else (0.0, 0.0)
    qual_min, qual_max = (min(finite_qual), max(finite_qual)) if finite_qual else (0.0, 0.0)

    for row in rows:
        stab = row["stability"] if np.isfinite(row["stability"]) else stab_min
        qual = row["quality"] if np.isfinite(row["quality"]) else qual_min
        row["normalized_stability"] = (
            1.0 if np.isclose(stab_max, stab_min) else (stab - stab_min) / (stab_max - stab_min)
        )
        row["normalized_quality"] = (
            1.0 if np.isclose(qual_max, qual_min) else (qual - qual_min) / (qual_max - qual_min)
        )
        row["multiobjective_distance"] = float(
            np.linalg.norm([1.0 - row["normalized_stability"], 1.0 - row["normalized_quality"]])
        )
    return sorted(
        rows,
        key=lambda row: (
            row["multiobjective_distance"],
            -row["stability"] if np.isfinite(row["stability"]) else np.inf,
            -row["quality"] if np.isfinite(row["quality"]) else np.inf,
            str(row["params"].get("linkage")),
        ),
    )


def _load_fold_candidates(args, fold_index, fsp_args):
    path = _candidate_file(args.intermediates_dir, fold_index, args.n_generations)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scored candidate file not found for fold {fold_index}: {path}")
    with open(path, "rb") as f:
        candidates = dill.load(f)

    stab_key, qual_key = fsp._primary_metric_keys(fsp_args)
    objective_index = {name: idx for idx, name in enumerate(fsp_args.ga_objectives)}
    rows = []
    for candidate_index, candidate in enumerate(candidates):
        params = decode_search_candidate(1, candidate)
        if int(params.get("k", -1)) != int(args.target_k):
            continue
        stability = _candidate_metric(candidate, stab_key, objective_index.get(stab_key))
        quality = _candidate_metric(candidate, qual_key, objective_index.get(qual_key))
        rows.append({
            "fold": f"fold{fold_index}",
            "fold_index": fold_index,
            "candidate_index": candidate_index,
            "params": {"linkage": str(params["linkage"]), "k": int(params["k"])},
            "stability": stability,
            "quality": quality,
            "fitness_values": _fitness_values(candidate),
            "metrics_summary": getattr(candidate, "metrics_summary", {}) or {},
            "candidate_file": path,
        })
    if not rows:
        raise RuntimeError(f"No candidate with k={args.target_k} found in {path}")
    return _rank_candidates(rows)


def _select_fold_candidates(args, fsp_args):
    selected = []
    all_rows = []
    for fold_index in range(int(args.n_folds)):
        ranked = _load_fold_candidates(args, fold_index, fsp_args)
        for rank, row in enumerate(ranked, start=1):
            record = dict(row)
            record["within_fold_k_rank"] = rank
            all_rows.append(record)
            if rank == 1:
                selected.append(record)
    return selected, all_rows


def _write_synthetic_fold_metrics(args, candidate_rows, source_results_dir, alternative_results_dir):
    os.makedirs(alternative_results_dir, exist_ok=True)
    written = []
    for synthetic_index, row in enumerate(candidate_rows):
        source_fold = row["fold"]
        synthetic_fold = f"fold{synthetic_index}"
        src_path = os.path.join(source_results_dir, source_fold, "metrics.pkl")
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source fold metrics not found: {src_path}")
        with open(src_path, "rb") as f:
            metrics = dill.load(f)
        params = dict(row["params"])
        metrics["alternative_solution"] = True
        metrics["alternative_target_k"] = int(args.target_k)
        metrics["alternative_source_fold"] = source_fold
        metrics["alternative_synthetic_fold"] = synthetic_fold
        metrics["alternative_selection_source"] = row["candidate_file"]
        metrics["alternative_candidate_index"] = int(row["candidate_index"])
        metrics["alternative_within_fold_k_rank"] = int(row["within_fold_k_rank"])
        metrics["alternative_within_fold_selection"] = {
            "stability": row["stability"],
            "quality": row["quality"],
            "fitness_values": row["fitness_values"],
            "multiobjective_distance": row["multiobjective_distance"],
        }
        metrics["best_params"] = params
        metrics["best_params_requested"] = params
        metrics["best_params_effective"] = params
        metrics["requested_k"] = int(args.target_k)
        metrics["effective_k_summary"] = summarize_effective_k([], requested_k=int(args.target_k))
        metrics["effective_k_summary"]["selected_k"] = int(args.target_k)
        metrics["effective_k_summary"]["fallback_reason"] = "forced_alternative_k"
        metrics["effective_k_fallback_reason"] = "forced_alternative_k"
        metrics["best_fitness"] = {
            "stability": row["stability"],
            "quality": row["quality"],
            "metrics_summary": row["metrics_summary"],
            "fitness_values": row["fitness_values"],
        }
        metrics["fold_status"] = "ok"
        metrics["skip_reason"] = None

        out_dir = os.path.join(alternative_results_dir, synthetic_fold)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics.pkl"), "wb") as f:
            dill.dump(metrics, f)
        written.append({
            "synthetic_fold": synthetic_fold,
            "source_fold": source_fold,
            "candidate_index": int(row["candidate_index"]),
            "within_fold_k_rank": int(row["within_fold_k_rank"]),
            "linkage": params["linkage"],
            "k": int(params["k"]),
            "stability": row["stability"],
            "quality": row["quality"],
        })
    return written


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


def _write_selection_outputs(output_dir, selected_rows, all_rows, args):
    os.makedirs(output_dir, exist_ok=True)
    flat_rows = []
    for row in all_rows:
        flat_rows.append({
            "fold": row["fold"],
            "candidate_index": row["candidate_index"],
            "within_fold_k_rank": row["within_fold_k_rank"],
            "linkage": row["params"]["linkage"],
            "k": row["params"]["k"],
            "stability": row["stability"],
            "quality": row["quality"],
            "normalized_stability": row["normalized_stability"],
            "normalized_quality": row["normalized_quality"],
            "multiobjective_distance": row["multiobjective_distance"],
            "fitness_values": repr(row["fitness_values"]),
            "candidate_file": row["candidate_file"],
        })
    pd.DataFrame(flat_rows).to_csv(os.path.join(output_dir, "candidate_ranking_k.csv"), index=False)
    pd.DataFrame([row for row in flat_rows if row["within_fold_k_rank"] == 1]).to_csv(
        os.path.join(output_dir, "selected_fold_candidates.csv"),
        index=False,
    )
    summary = {
        "target_k": int(args.target_k),
        "selection_rule": (
            "For each fold, filter scored search candidates to target_k and select the "
            "candidate closest to the ideal point after within-fold min-max normalization "
            "of stability and quality."
        ),
        "selected": selected_rows,
    }
    with open(os.path.join(output_dir, "selection_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)


def _write_synthetic_fold_manifest(output_dir, written_rows):
    pd.DataFrame(written_rows).to_csv(os.path.join(output_dir, "synthetic_fold_manifest.csv"), index=False)


def _build_fsp_args(args, output_final_metrics):
    search_objectives = fsp._normalize_objective_tokens(args.search_objectives, args.optimisation)
    ns = SimpleNamespace(
        input_csv=args.input_csv,
        meta_csv=args.meta_csv,
        base_dir=args.base_dir,
        subject_id_column=args.subject_id_column,
        col_threshold=args.col_threshold,
        row_threshold=args.row_threshold,
        skew_threshold=args.skew_threshold,
        scaler_type=args.scaler_type,
        modalities=args.modalities,
        dummy_code_modalities=args.dummy_code_modalities,
        mixed_categorical_modalities=args.mixed_categorical_modalities,
        dim_reduction=_normalize_dim_reduction(args.dim_reduction),
        maxPC=args.maxPC,
        spca_alpha=args.spca_alpha,
        spca_ridge_alpha=args.spca_ridge_alpha,
        spca_max_iter=args.spca_max_iter,
        snmf_alpha=args.snmf_alpha,
        snmf_l1_ratio=args.snmf_l1_ratio,
        snmf_max_iter=args.snmf_max_iter,
        sparse_l1_lambda=args.sparse_l1_lambda,
        hidden_dims=args.hidden_dims,
        activation_functions=args.activation_functions,
        learning_rates=args.learning_rates,
        batch_sizes=args.batch_sizes,
        latent_dims=args.latent_dims,
        k_min=args.target_k,
        k_max=args.target_k,
        linkages=args.linkages,
        n_population=0,
        n_generations=args.n_generations,
        optimisation=args.optimisation,
        search_objectives=list(search_objectives),
        ga_objectives=list(search_objectives),
        n_bootstrap=args.n_bootstrap,
        final_bootstrap_preprocessing=args.final_bootstrap_preprocessing,
        n_permutations_pvalue=args.n_permutations_pvalue,
        bootstrap_mode=args.bootstrap_mode,
        n_folds=args.n_folds,
        output_pkl="pipeline_results.pkl",
        n_jobs=args.n_jobs,
        TEST=args.TEST,
        max_missing_bootstraps=5,
        mincluster=args.mincluster,
        mincluster_n=args.mincluster_n,
        mincluster_resample_mode=args.mincluster_resample_mode,
        use_effective_k_for_fold_merge="FALSE",
        use_cross_fold_effective_k_for_final_run="FALSE",
        internal_ensemble_enabled=args.internal_ensemble_enabled,
        internal_ensemble_bcs=args.internal_ensemble_bcs,
        internal_ensemble_sample_frac=args.internal_ensemble_sample_frac,
        internal_ensemble_feature_frac=args.internal_ensemble_feature_frac,
        mode="merge",
        generation=args.n_generations,
        population_file=None,
        seed=None,
        population_dir=None,
        population_initial_file=None,
        bootstrap_index=None,
        bootstrap_dir=None,
        output_labels=None,
        output_population=None,
        fold_index=None,
        output_metrics=None,
        output_final_metrics=output_final_metrics,
        ga_cxpb=0.7,
        ga_mutpb=0.2,
        ga_elitism=2,
        DO_SVM=args.DO_SVM,
    )
    ns.multi_fitness_class_name = None
    return ns


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finalize a forced-k alternative simpleclust solution from existing fold search outputs."
    )
    parser.add_argument("--target_k", type=int, required=True)
    parser.add_argument("--base_dir", default=os.getcwd())
    parser.add_argument("--source_results_dir", required=True)
    parser.add_argument("--intermediates_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input_csv", default="cleaned_discovery_data_simpleclust.csv")
    parser.add_argument("--meta_csv", default="merged_meta_simpleclust.csv")
    parser.add_argument("--subject_id_column", default="src_subject_id")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_generations", type=int, default=1)
    parser.add_argument("--n_bootstrap", type=int, default=100)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--n_permutations_pvalue", type=int, default=200)
    parser.add_argument("--bootstrap_mode", choices=["bootstrap", "subsample"], default="subsample")
    parser.add_argument("--col_threshold", type=float, default=0.5)
    parser.add_argument("--row_threshold", type=float, default=0.5)
    parser.add_argument("--skew_threshold", type=float, default=0.75)
    parser.add_argument("--scaler_type", default="robust")
    parser.add_argument("--modalities", nargs="+", default=["Include_cluster"])
    parser.add_argument("--dummy_code_modalities", nargs="*", default=["Include_cluster"])
    parser.add_argument("--mixed_categorical_modalities", nargs="*", default=[])
    parser.add_argument("--dim_reduction", default="None")
    parser.add_argument("--maxPC", type=int, default=20)
    parser.add_argument("--spca_alpha", type=float, default=1.0)
    parser.add_argument("--spca_ridge_alpha", type=float, default=0.01)
    parser.add_argument("--spca_max_iter", type=int, default=1000)
    parser.add_argument("--snmf_alpha", type=float, default=0.1)
    parser.add_argument("--snmf_l1_ratio", type=float, default=1.0)
    parser.add_argument("--snmf_max_iter", type=int, default=1000)
    parser.add_argument("--sparse_l1_lambda", type=float, default=1e-3)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--activation_functions", nargs="+", default=["ReLU", "LeakyReLU", "selu", "swish"])
    parser.add_argument("--learning_rates", nargs="+", type=float, default=[0.001, 0.0001])
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--latent_dims", nargs="+", type=int, default=[2, 5, 10])
    parser.add_argument("--linkages", nargs="+", default=["average", "complete", "weighted"])
    parser.add_argument("--optimisation", choices=["single", "multi"], default="multi")
    parser.add_argument("--search_objectives", nargs="+", default=["stab_ari", "quality"])
    parser.add_argument("--mincluster", choices=["TRUE", "FALSE"], default="TRUE")
    parser.add_argument("--mincluster_n", type=int, default=50)
    parser.add_argument("--mincluster_resample_mode", choices=["fixed", "scaled"], default="fixed")
    parser.add_argument("--internal_ensemble_enabled", choices=["TRUE", "FALSE"], default="TRUE")
    parser.add_argument("--internal_ensemble_bcs", type=int, default=100)
    parser.add_argument("--internal_ensemble_sample_frac", type=float, default=0.8)
    parser.add_argument("--internal_ensemble_feature_frac", type=float, default=1.0)
    parser.add_argument("--final_bootstrap_preprocessing", choices=["outside", "inside", "both"], default="outside")
    parser.add_argument("--TEST", choices=["TRUE", "FALSE"], default="FALSE")
    parser.add_argument("--DO_SVM", choices=["TRUE", "FALSE"], default="TRUE")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--selection_mode",
        choices=["all_target_k", "best_per_fold"],
        default="all_target_k",
        help=(
            "all_target_k passes every scored target-k candidate from every source fold into "
            "the final merge, allowing all linkages to be tested on the full sample. "
            "best_per_fold preserves the original narrower behavior."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.base_dir = os.path.abspath(args.base_dir)
    args.source_results_dir = os.path.abspath(args.source_results_dir)
    args.intermediates_dir = os.path.abspath(args.intermediates_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    args.dim_reduction = _normalize_dim_reduction(args.dim_reduction)

    if args.target_k < 2:
        raise ValueError("--target_k must be at least 2 for an alternative cluster solution.")
    if os.path.exists(args.output_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {args.output_dir}. "
                "Use --overwrite to replace synthetic fold metrics/final outputs."
            )
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    for attr in list(vars(creator).keys()):
        if attr.startswith("FitnessMulti"):
            delattr(creator, attr)
    for cls in ("FitnessMax", "Individual"):
        if hasattr(creator, cls):
            delattr(creator, cls)

    preliminary_fsp_args = _build_fsp_args(args, os.path.join(args.output_dir, "final", "final_metrics.pkl"))
    fsp._ensure_multi_fitness_class(preliminary_fsp_args)
    selected_rows, all_rows = _select_fold_candidates(args, preliminary_fsp_args)

    candidate_rows = all_rows if args.selection_mode == "all_target_k" else selected_rows
    written_rows = _write_synthetic_fold_metrics(
        args,
        candidate_rows,
        source_results_dir=args.source_results_dir,
        alternative_results_dir=args.output_dir,
    )

    final_dir = os.path.join(args.output_dir, "final")
    final_metrics_path = os.path.join(final_dir, "final_metrics.pkl")
    _write_selection_outputs(final_dir, selected_rows, all_rows, args)
    _write_synthetic_fold_manifest(final_dir, written_rows)

    merge_args = _build_fsp_args(args, final_metrics_path)
    merge_args.n_folds = len(written_rows)
    old_results_dir = os.environ.get("RESULTS_DIR")
    os.environ["RESULTS_DIR"] = args.output_dir
    try:
        try:
            fsp.do_merge(merge_args)
        except Exception as exc:
            failure = {
                "target_k": int(args.target_k),
                "selection_mode": args.selection_mode,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "note": (
                    "The target-k candidates were selected from existing scored search outputs, "
                    "but the normal final full-sample merge rejected them. Common reasons are "
                    "minimum-cluster enforcement collapsing k, non-positive final quality, or "
                    "instability in the full-data consensus cut."
                ),
                "candidate_manifest_csv": os.path.join(final_dir, "synthetic_fold_manifest.csv"),
                "candidate_ranking_csv": os.path.join(final_dir, "candidate_ranking_k.csv"),
            }
            with open(os.path.join(final_dir, "merge_failure.json"), "w") as f:
                json.dump(failure, f, indent=2, default=_json_default)
            raise
    finally:
        if old_results_dir is None:
            os.environ.pop("RESULTS_DIR", None)
        else:
            os.environ["RESULTS_DIR"] = old_results_dir

    with open(final_metrics_path, "rb") as f:
        metrics = dill.load(f)
    metrics["alternative_solution"] = True
    metrics["alternative_target_k"] = int(args.target_k)
    metrics["alternative_source_results_dir"] = args.source_results_dir
    metrics["alternative_source_intermediates_dir"] = args.intermediates_dir
    metrics["alternative_selection_summary_json"] = os.path.join(final_dir, "selection_summary.json")
    with open(final_metrics_path, "wb") as f:
        dill.dump(metrics, f)

    print(f"Alternative k={args.target_k} final metrics saved to {final_metrics_path}")


if __name__ == "__main__":
    main()
