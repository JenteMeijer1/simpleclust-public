"""Provide shared preprocessing, analysis, plotting, and notebook helpers."""

import os
import glob
import re
import pandas as pd
from datetime import datetime
from itertools import combinations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import skew
from typing import Tuple

import random
from scipy.stats import chi2, chi2_contingency, kruskal
from multiprocessing import Pool, cpu_count
import functools
import warnings
from sklearn.decomposition import NMF, PCA, SparsePCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


from typing import List, Dict, Any
from effective_k import resolve_min_cluster_n, summarize_effective_k


try:
    from theme import CATEGORICAL as _THEME_CATS, CC_COLOR as _THEME_CC, THEME as _THEME_DICT
    _MUTED = _THEME_DICT.get("muted", "#7a88a0")
except ImportError:
    _THEME_CATS = ["#CC6677", "#4477AA", "#C07EAA", "#44AA99", "#7755AA", "#882255"]
    _THEME_CC = "#4A6080"
    _MUTED = "#7a88a0"

# =============================================================================
# Utils.py Section Map
# =============================================================================
# - Shared palettes and notebook stage utilities: helpers used before running
#   the pipeline and after loading final metrics in project notebooks.
# - Reporting plots and cluster-mapping helpers: reusable notebook figures.
# - Data loading and preprocessing: raw data splits, transforms, imputation,
#   scaling, and reusable preprocessing for validation/follow-up samples.
# - Dimensionality reduction helpers: PCA/FAMD/MCA projection utilities used by
#   the pipeline and notebooks.
# - Legacy diagnostics and longitudinal helpers: older notebook/reporting
#   utilities retained for backwards compatibility.
# - Project notebook helpers: task-specific SimpleClust, clinical-paper, and
#   PROSPECT helpers moved out of notebook cells.


MODALITY_CLUSTER_PALETTES = {
    "Internalising": {
        "0": "#38699A",
        "1": "#BCDDFF",
        "2": "#043B71",
        "3": "#DBEDFF",
        "4": "#1F3850",
        "5": "#5FAFFF",
        "CC": _THEME_CC,
    },
    "Functioning": {
        "0": "#B36F9C",
        "1": "#FFDAF3",
        "2": "#A93A84",
        "3": "#BEA9B7",
        "4": "#54354A",
        "5": "#E94DB5",
        "CC": _THEME_CC,
    },
    "Detachment": {
        "0": "#389687",
        "1": "#D3FFF8",
        "2": "#16685A",
        "3": "#DBEDEA",
        "4": "#2C5A52",
        "5": "#35EFD0",
        "CC": _THEME_CC,
    },
    "Psychoticism": {
        "0": "#B45868",
        "1": "#FCD0D7",
        "2": "#753742",
        "3": "#F8C6CF",
        "4": "#560311",
        "5": "#C81533",
        "CC": _THEME_CC,
    },
    "Cognition": {
        "0": "#6D4D9C",
        "1": "#E0CCFD",
        "2": "#4F2291",
        "3": "#6C607F",
        "4": "#18003D",
        "5": "#893AFF",
        "CC": _THEME_CC,
    },
    "Metabolic_Risk": {
        "0": "#00545C",
        "1": "#7FD4CC",
        "2": "#0B8790",
        "3": "#BDEBE6",
        "4": "#002F34",
        "5": "#3FB7B2",
        "CC": _THEME_CC,
    },
    "Blood_markers": {
        "0": "#0A5D1E",
        "1": "#9BE08D",
        "2": "#168A35",
        "3": "#D7F6D2",
        "4": "#063815",
        "5": "#4FBE5B",
        "CC": _THEME_CC,
    },
    "Suicidality": {
        "0": "#4C2CA8",
        "1": "#C0A9FF",
        "2": "#7056D6",
        "3": "#E3D8FF",
        "4": "#2E1B6F",
        "5": "#9B83F0",
        "CC": _THEME_CC,
    },
    "Injury": {
        "0": "#6F143F",
        "1": "#F39ABB",
        "2": "#B72A70",
        "3": "#FAEAF0",
        "4": "#3F0822",
        "5": "#E85A97",
        "CC": _THEME_CC,
    },
    "Physical_health": {
        "0": "#174D8C",
        "1": "#8CCCEF",
        "2": "#2D76C2",
        "3": "#D0E6F8",
        "4": "#0E2F57",
        "5": "#5AA9DD",
        "CC": _THEME_CC,
    },
}

DEFAULT_CLUSTER_PALETTE = {
    "0": _THEME_CATS[0],
    "1": _THEME_CATS[1],
    "2": _THEME_CATS[2],
    "3": _THEME_CATS[3],
    "4": _THEME_CATS[4],
    "5": _THEME_CATS[5],
    "CC": _THEME_CC,
}

INTEGRATED_CLUSTER_PALETTE = {
    "0": _THEME_CATS[0],
    "1": _THEME_CATS[1],
    "2": _THEME_CATS[2],
    "3": _THEME_CATS[3],
    "4": _THEME_CATS[4],
    "5": _THEME_CATS[5],
    "CC": _THEME_CC,
}

EXTRA_CLUSTER_COLORS = list(_THEME_CATS) + [
    "#5B3FBB", "#A8326D", "#1F5FA8", "#0B7C25",
    "#006D77", "#D98B27", "#F2C94C", "#2F80ED",
]


def cluster_sort_key(value):
    """Handle cluster sort key."""
    value_str = str(value)
    if value_str == "CC":
        return (2, value_str)
    try:
        return (0, float(value_str))
    except Exception:
        return (1, value_str)


def modality_cluster_palette(labels, modality=None):
    """Return a readable cross-project cluster/CC palette for any cluster count."""
    labels = [str(label) for label in labels]
    modality_str = str(modality or "")
    matched_modality = next(
        (name for name in MODALITY_CLUSTER_PALETTES if name == modality_str or name in modality_str),
        None,
    )
    palette_seed = MODALITY_CLUSTER_PALETTES.get(matched_modality, INTEGRATED_CLUSTER_PALETTE)
    color_order = list(EXTRA_CLUSTER_COLORS) + list(INTEGRATED_CLUSTER_PALETTE.values())
    color_map = {}
    extra_idx = 0
    for label in sorted(pd.unique(pd.Series(labels).dropna()), key=cluster_sort_key):
        if label == "CC":
            color_map[label] = palette_seed.get("CC", DEFAULT_CLUSTER_PALETTE["CC"])
        elif label in palette_seed:
            color_map[label] = palette_seed[label]
        else:
            used = set(color_map.values())
            while color_order[extra_idx % len(color_order)] in used:
                extra_idx += 1
            color_map[label] = color_order[extra_idx % len(color_order)]
            extra_idx += 1
    return color_map


# Backward-compatible aliases for older notebooks/scripts.
PROSPECT_MODALITY_PALETTES = MODALITY_CLUSTER_PALETTES
PROSPECT_DEFAULT_PALETTE = DEFAULT_CLUSTER_PALETTE
PROSPECT_INTEGRATED_PALETTE = INTEGRATED_CLUSTER_PALETTE
PROSPECT_EXTRA_CLUSTER_COLORS = EXTRA_CLUSTER_COLORS


def prospect_cluster_palette(labels, modality=None):
    """Backward-compatible alias for the shared cross-project palette helper."""
    return modality_cluster_palette(labels, modality=modality)


# =============================================================================
# Notebook Stage Utilities
# =============================================================================
# These helpers are shared by project notebooks. They cover the notebook stages
# before running the pipeline (profile/data checks) and after loading results
# (cluster metadata joins, pathway summaries, diagnostics, and plots).


def truthy_profile_value(value):
    """Interpret shell/profile values such as TRUE/1/yes as booleans."""
    return str(value).strip().strip('"').strip("'").upper() in {"1", "TRUE", "YES", "Y", "ON"}


def parse_profile_exports(profile_path):
    """Parse simple `export NAME=value` assignments from a run profile shell file."""
    values = {}
    profile_path = Path(profile_path)
    if not profile_path.exists():
        return values
    for raw_line in profile_path.read_text().splitlines():
        line = raw_line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[len("export "):].split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def infer_notebook_profile(default="clinical_paper"):
    """
    Infer the active run profile for notebook code.

    Priority is RUN_PROFILE from the environment, then NOTEBOOK_PROFILE, then a
    caller-supplied default. This keeps project notebooks portable across
    clinical/prospect/legacy profiles.
    """
    return (
        os.environ.get("RUN_PROFILE")
        or os.environ.get("NOTEBOOK_PROFILE")
        or default
    )


def find_repo_root(start=None, markers=("full_pipeline.py", "Utils.py", "run_profiles")):
    """Walk upward from start/current directory until the repository root is found."""
    path = Path(start or os.getcwd()).resolve()
    for candidate in [path] + list(path.parents):
        if all((candidate / marker).exists() for marker in markers):
            return candidate
    return path


def profile_enabled_for_sensitivity(repo_root, profile_name):
    """Return whether a run profile enables cluster-validation sensitivity output."""
    profile_path = Path(repo_root) / "run_profiles" / f"{profile_name}.sh"
    values = parse_profile_exports(profile_path)
    return truthy_profile_value(values.get("DO_CLUSTER_VALIDATION_SENSITIVITY", "FALSE"))


def display_if_available(obj):
    """Use IPython display when available; otherwise print the object."""
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj)


def get_nested(dct, path, default=np.nan):
    """Safely fetch a nested value from dict/list containers."""
    cur = dct
    for key in path:
        try:
            if isinstance(cur, dict):
                cur = cur[key]
            elif isinstance(cur, (list, tuple)) and isinstance(key, int):
                cur = cur[key]
            else:
                return default
        except Exception:
            return default
    return cur


def flatten_sensitivity_results(results):
    """
    Flatten cluster-validation sensitivity output into a DataFrame.

    The input can be a list/dict of nested result payloads. Scalar values are
    preserved directly; nested dict values are flattened with dotted keys.
    """
    rows = []

    def flatten(prefix, value, out):
        """Handle flatten."""
        if isinstance(value, dict):
            for k, v in value.items():
                flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
        elif isinstance(value, (list, tuple)) and all(not isinstance(x, (dict, list, tuple)) for x in value):
            out[prefix] = value
        else:
            out[prefix] = value

    iterable = results.values() if isinstance(results, dict) else results
    for item in iterable or []:
        row = {}
        flatten("", item, row)
        rows.append(row)
    return pd.DataFrame(rows)


def print_remaining_after_full_missing_modality_removal(
    df,
    df_name,
    meta,
    modalities,
    subject_id_column="src_subject_id",
):
    """Print how many subjects remain after dropping full-missing modality rows."""
    modal_dict = extract_modalities(meta, df, subject_id_column=subject_id_column)
    subjects_to_drop = set()
    rows = []
    for modality in modalities:
        df_mod = modal_dict.get(modality)
        if df_mod is None or df_mod.empty:
            rows.append({"modality": modality, "n_full_missing": np.nan})
            continue
        data_only = df_mod.drop(columns=[subject_id_column], errors="ignore")
        mask = data_only.isna().all(axis=1)
        ids = df_mod.loc[mask, subject_id_column].tolist() if subject_id_column in df_mod else []
        subjects_to_drop.update(ids)
        rows.append({"modality": modality, "n_full_missing": int(mask.sum())})
    remaining = df[~df[subject_id_column].isin(subjects_to_drop)] if subject_id_column in df else df
    summary = pd.DataFrame(rows)
    print(f"{df_name}: {len(remaining)}/{len(df)} subjects remain after full-missing modality removal.")
    return summary, remaining


def build_group_palette(modality, group_order, modality_palettes=None, default_palette=None):
    """Build a stable color map for cluster/group labels in notebooks."""
    if modality_palettes is None:
        modality_palettes = MODALITY_CLUSTER_PALETTES
    if default_palette is None:
        default_palette = DEFAULT_CLUSTER_PALETTE
    labels = [str(g) for g in group_order]
    base = modality_palettes.get(modality, default_palette)
    palette = {}
    fallback = modality_cluster_palette(labels, modality=modality)
    for label in labels:
        palette[label] = base.get(label, fallback.get(label, _THEME_CC))
    return palette


def add_metadata_and_clusters(
    cluster_source,
    data_full,
    mod_num=None,
    subject_id_column="src_subject_id",
    metadata_columns=None,
    cluster_col="Cluster",
):
    """
    Join modality/final cluster labels to a full metadata table.

    cluster_source can be final_metrics, a dict_final-style modality dictionary,
    or a single modality DataFrame. When final_metrics is supplied, mod_num
    selects `individual_labels[mod_num]`; if mod_num is None, final labels are
    used when available.
    """
    if metadata_columns is None:
        metadata_columns = [subject_id_column, "interview_age", "sex", "Site", "race"]
    data_meta = data_full[[c for c in metadata_columns if c in data_full.columns]].copy()

    if isinstance(cluster_source, dict) and "data" in cluster_source:
        final_metrics = cluster_source
        data_by_mod = final_metrics.get("data", {})
        if mod_num is None:
            labels = final_metrics.get("final_labels")
            first_mod = next(iter(data_by_mod))
            label_ids = data_by_mod[first_mod][subject_id_column].tolist()
        else:
            modalities = list(data_by_mod)
            modality = modalities[int(mod_num)]
            labels = final_metrics.get("individual_labels", [])[int(mod_num)]
            label_ids = data_by_mod[modality][subject_id_column].tolist()
    elif isinstance(cluster_source, dict):
        modalities = list(cluster_source)
        modality = modalities[int(mod_num or 0)]
        df_mod = cluster_source[modality]
        labels = df_mod[cluster_col].to_numpy() if cluster_col in df_mod else np.arange(len(df_mod))
        label_ids = df_mod[subject_id_column].tolist()
    else:
        df_mod = cluster_source
        labels = df_mod[cluster_col].to_numpy() if cluster_col in df_mod else np.arange(len(df_mod))
        label_ids = df_mod[subject_id_column].tolist()

    label_df = pd.DataFrame({
        subject_id_column: label_ids[:len(labels)],
        cluster_col: np.asarray(labels).reshape(-1),
    })
    return data_meta.merge(label_df, on=subject_id_column, how="inner")


def chi_square_comparison(df, group_col, label_col, title_prefix="", save_path=None):
    """Run and optionally plot a chi-square group comparison table."""
    table = pd.crosstab(df[group_col], df[label_col])
    if table.shape[0] < 2 or table.shape[1] < 2:
        result = {"chi2": np.nan, "p_value": np.nan, "dof": np.nan, "expected": None, "table": table}
    else:
        chi2_stat, p_value, dof, expected = chi2_contingency(table)
        result = {
            "chi2": float(chi2_stat),
            "p_value": float(p_value),
            "dof": int(dof),
            "expected": pd.DataFrame(expected, index=table.index, columns=table.columns),
            "table": table,
        }
    print(f"{title_prefix} chi-square {group_col} by {label_col}: p={result['p_value']}")
    if save_path:
        fig, ax = plt.subplots(figsize=(max(6, 0.7 * table.shape[1]), max(4, 0.45 * table.shape[0])))
        sns.heatmap(table, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(title_prefix or f"{group_col} by {label_col}")
        fig.tight_layout()
        _save_matplotlib_png_pdf(fig, os.path.splitext(save_path)[0], dpi=300)
        plt.close(fig)
    return result


def parse_stream(stream_str):
    """Parse a stream string into ordered stage tokens."""
    if pd.isna(stream_str):
        return []
    text = str(stream_str)
    if "->" in text:
        return [part.strip() for part in text.split("->") if part.strip()]
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    return [part.strip() for part in text.split("/") if part.strip()]


def infer_stage_order(df, stream_col="stream"):
    """Infer generic stage names from the maximum stream length."""
    max_len = int(df[stream_col].map(lambda x: len(parse_stream(x))).max()) if len(df) else 0
    if max_len == 0:
        return []
    if max_len == 1:
        return ["stage1"]
    return [f"stage{i + 1}" for i in range(max_len - 1)] + ["final"]


def build_prefix_next(df, stream_col="stream", n_col="n"):
    """Summarize each stream prefix and the distribution of next labels."""
    rows = []
    for _, row in df.iterrows():
        tokens = parse_stream(row[stream_col])
        n = float(row.get(n_col, 1))
        for depth in range(1, len(tokens)):
            rows.append({
                "depth": depth,
                "prefix": " -> ".join(tokens[:depth]),
                "next": tokens[depth],
                "n": n,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.groupby(["depth", "prefix", "next"], as_index=False)["n"].sum()


def compare_prefix_structure(df_disc, df_test, stream_col="stream", n_col="n", eps=1e-12):
    """Compare next-step stream distributions between discovery and test samples."""
    disc = build_prefix_next(df_disc, stream_col, n_col)
    test = build_prefix_next(df_test, stream_col, n_col)
    keys = ["depth", "prefix", "next"]
    merged = disc.merge(test, on=keys, how="outer", suffixes=("_disc", "_test")).fillna(0)
    if merged.empty:
        return merged
    merged["prefix_total_disc"] = merged.groupby(["depth", "prefix"])["n_disc"].transform("sum")
    merged["prefix_total_test"] = merged.groupby(["depth", "prefix"])["n_test"].transform("sum")
    merged["p_disc"] = merged["n_disc"] / (merged["prefix_total_disc"] + eps)
    merged["p_test"] = merged["n_test"] / (merged["prefix_total_test"] + eps)
    merged["delta"] = merged["p_test"] - merged["p_disc"]
    merged["abs_delta"] = merged["delta"].abs()
    return merged.sort_values("abs_delta", ascending=False).reset_index(drop=True)


def plot_top_prefix_differences(prefix_report, top_n=20, min_depth=1):
    """Plot the largest discovery-test prefix transition shifts."""
    data = prefix_report[prefix_report["depth"] >= min_depth].head(top_n).copy()
    if data.empty:
        return None
    data["label"] = data["prefix"] + " -> " + data["next"].astype(str)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(data))))
    sns.barplot(data=data, y="label", x="delta", ax=ax, color=_THEME_CATS[1])
    ax.axvline(0, color=_MUTED, linewidth=0.8)
    ax.set_xlabel("Test minus discovery transition probability")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def final_mapping_table(df, stream_col="stream", n_col="n", final_domain="final"):
    """Return counts of each prefix-to-final mapping."""
    rows = []
    for _, row in df.iterrows():
        tokens = parse_stream(row[stream_col])
        if not tokens:
            continue
        final = tokens[-1]
        prefix = " -> ".join(tokens[:-1]) if len(tokens) > 1 else ""
        rows.append({"prefix": prefix, final_domain: final, "n": float(row.get(n_col, 1))})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.groupby(["prefix", final_domain], as_index=False)["n"].sum()


def compare_final_mapping(df_disc, df_test, stream_col="stream", n_col="n", final_domain="final"):
    """Compare final-cluster mapping proportions between discovery and test samples."""
    disc = final_mapping_table(df_disc, stream_col, n_col, final_domain)
    test = final_mapping_table(df_test, stream_col, n_col, final_domain)
    merged = disc.merge(test, on=["prefix", final_domain], how="outer", suffixes=("_disc", "_test")).fillna(0)
    if merged.empty:
        return merged
    merged["prefix_total_disc"] = merged.groupby("prefix")["n_disc"].transform("sum")
    merged["prefix_total_test"] = merged.groupby("prefix")["n_test"].transform("sum")
    merged["p_disc"] = merged["n_disc"] / merged["prefix_total_disc"].replace(0, np.nan)
    merged["p_test"] = merged["n_test"] / merged["prefix_total_test"].replace(0, np.nan)
    merged["delta"] = merged["p_test"] - merged["p_disc"]
    merged["abs_delta"] = merged["delta"].abs()
    return merged.sort_values("abs_delta", ascending=False).reset_index(drop=True)


def plot_top_final_mapping_shifts(final_cmp, top_n=20):
    """Plot the largest shifts in final-cluster mapping."""
    data = final_cmp.head(top_n).copy()
    if data.empty:
        return None
    final_col = [c for c in data.columns if c not in {"prefix", "n_disc", "n_test", "prefix_total_disc", "prefix_total_test", "p_disc", "p_test", "delta", "abs_delta"}][0]
    data["label"] = data["prefix"].astype(str) + " -> " + data[final_col].astype(str)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(data))))
    sns.barplot(data=data, y="label", x="delta", ax=ax, color=_THEME_CATS[0])
    ax.axvline(0, color=_MUTED, linewidth=0.8)
    ax.set_xlabel("Test minus discovery final-mapping probability")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def stream_presence_and_topk(df_disc, df_test, stream_col="stream", n_col="n", topk=30):
    """Compare stream presence and top-k coverage between discovery and test."""
    disc = df_disc[[stream_col, n_col]].copy()
    test = df_test[[stream_col, n_col]].copy()
    merged = disc.merge(test, on=stream_col, how="outer", suffixes=("_disc", "_test")).fillna(0)
    merged["present_disc"] = merged[f"{n_col}_disc"] > 0
    merged["present_test"] = merged[f"{n_col}_test"] > 0
    merged["abs_count_delta"] = (merged[f"{n_col}_test"] - merged[f"{n_col}_disc"]).abs()
    top_disc = set(disc.sort_values(n_col, ascending=False).head(topk)[stream_col])
    top_test = set(test.sort_values(n_col, ascending=False).head(topk)[stream_col])
    return {
        "table": merged.sort_values("abs_count_delta", ascending=False),
        "topk_overlap": len(top_disc & top_test),
        "topk_disc_only": sorted(top_disc - top_test),
        "topk_test_only": sorted(top_test - top_disc),
    }


def sankey_from_streams(df, stream_col="stream", n_col="n", max_edges=200):
    """Create a Plotly Sankey figure from stream/count summaries."""
    try:
        import plotly.graph_objects as go
    except Exception as err:
        raise RuntimeError("Plotly is required for sankey_from_streams.") from err
    edges = []
    for _, row in df.head(max_edges).iterrows():
        tokens = parse_stream(row[stream_col])
        n = float(row.get(n_col, 1))
        for src, tgt in zip(tokens[:-1], tokens[1:]):
            edges.append((src, tgt, n))
    edge_df = pd.DataFrame(edges, columns=["src", "tgt", "n"])
    if edge_df.empty:
        return go.Figure()
    edge_df = edge_df.groupby(["src", "tgt"], as_index=False)["n"].sum()
    nodes = sorted(set(edge_df["src"]).union(edge_df["tgt"]))
    node_index = {node: i for i, node in enumerate(nodes)}
    fig = go.Figure(go.Sankey(
        node={"label": nodes},
        link={
            "source": [node_index[x] for x in edge_df["src"]],
            "target": [node_index[x] for x in edge_df["tgt"]],
            "value": edge_df["n"].tolist(),
        },
    ))
    return fig


def full_structure_report(stream_summary, stream_summary_test, stream_col="stream", n_col="n", topk=30, final_domain="final"):
    """Return all stream-comparison tables used by reporting notebooks."""
    prefix_report = compare_prefix_structure(stream_summary, stream_summary_test, stream_col, n_col)
    final_cmp = compare_final_mapping(stream_summary, stream_summary_test, stream_col, n_col, final_domain)
    presence = stream_presence_and_topk(stream_summary, stream_summary_test, stream_col, n_col, topk)
    return {"prefix_report": prefix_report, "final_mapping": final_cmp, "presence": presence}


def all_streams_table(stream_summary, stream_summary_test, stream_col="stream", n_col="n"):
    """Outer-join discovery and test stream count tables."""
    return (
        stream_summary[[stream_col, n_col]]
        .merge(stream_summary_test[[stream_col, n_col]], on=stream_col, how="outer", suffixes=("_disc", "_test"))
        .fillna(0)
        .sort_values([f"{n_col}_disc", f"{n_col}_test"], ascending=False)
        .reset_index(drop=True)
    )


def summarize_streams(df_paths, stage_order, top_k=20, sample_ids=10):
    """Summarize modality-to-final streams from a label path table."""
    rows = []
    final_col = "final" if "final" in df_paths.columns else None
    for _, row in df_paths.iterrows():
        tokens = [f"{stage}={row[stage]}" for stage in stage_order if stage in row]
        if final_col is not None:
            tokens.append(f"{final_col}={row[final_col]}")
        rows.append({"stream": " → ".join(tokens), "subject_id": row.get("src_subject_id", row.name)})
    stream_df = pd.DataFrame(rows)
    summary = stream_df.groupby("stream").agg(
        n=("stream", "size"),
        sample_ids=("subject_id", lambda x: list(x)[:sample_ids]),
    ).sort_values("n", ascending=False).reset_index()
    return summary.head(top_k), summary, stream_df


def summarize_streams_clinical(df_paths, stage_order, top_k=20, sample_ids=10):
    """Backward-compatible alias for clinical stream summaries."""
    return summarize_streams(df_paths, stage_order, top_k=top_k, sample_ids=sample_ids)


def summarize_feature_differences(final_metrics, top_k=10):
    """Return top absolute feature differences between clusters for each modality."""
    rows = []
    data = final_metrics.get("data", {}) if isinstance(final_metrics, dict) else {}
    labels_by_mod = final_metrics.get("individual_labels", []) if isinstance(final_metrics, dict) else []
    for i, (mod, df_mod) in enumerate(data.items()):
        if i >= len(labels_by_mod):
            continue
        labels = pd.Series(np.asarray(labels_by_mod[i]).reshape(-1), name="cluster")
        features = df_mod.drop(columns=[final_metrics.get("subject_id_column", "src_subject_id")], errors="ignore")
        numeric = features.apply(pd.to_numeric, errors="coerce")
        grouped = numeric.assign(cluster=labels.to_numpy()).groupby("cluster").mean()
        if grouped.shape[0] < 2:
            continue
        spread = grouped.max(axis=0) - grouped.min(axis=0)
        for feature, value in spread.abs().sort_values(ascending=False).head(top_k).items():
            rows.append({"modality": mod, "feature": feature, "absolute_mean_range": float(value)})
    return pd.DataFrame(rows)


def plot_autoencoder_diagnostics(run_name, final_metrics):
    """Plot simple AE/VAE reconstruction-loss diagnostics when available."""
    ae_res = final_metrics.get("ae_res", {}) if isinstance(final_metrics, dict) else {}
    plotted = False
    for mod, res in ae_res.items():
        history = res.get("history") or res.get("loss_history") if isinstance(res, dict) else None
        if history is None:
            continue
        vals = pd.Series(history, dtype=float)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(vals.to_numpy())
        ax.set_title(f"{run_name}: {mod} autoencoder loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        fig.tight_layout()
        plotted = True
    if not plotted:
        print(f"{run_name}: no autoencoder loss histories found.")


def _extract_final_latent(ae_res):
    """Return the latent matrix from flat or single-modality nested outputs."""
    if isinstance(ae_res, dict) and "final_latent" in ae_res:
        return ae_res["final_latent"], None
    if not isinstance(ae_res, dict):
        return None, None

    nested = [
        (name, payload["final_latent"])
        for name, payload in ae_res.items()
        if isinstance(payload, dict) and "final_latent" in payload
    ]
    if not nested:
        return None, None
    if len(nested) == 1:
        return nested[0][1], nested[0][0]

    arrays = [np.asarray(latent) for _, latent in nested]
    n_rows = {arr.shape[0] for arr in arrays if arr.ndim > 0}
    if len(n_rows) != 1:
        return None, None
    arrays = [arr.reshape(arr.shape[0], -1) if arr.ndim == 1 else arr for arr in arrays]
    return np.concatenate(arrays, axis=1), "+".join(name for name, _ in nested)


def _format_dimred_display(dimred_display):
    """Format dimred display."""
    method = str(dimred_display).strip().lower().replace("_", "").replace("-", "")
    display_names = {
        "none": "None",
        "pca": "PCA",
        "sparsepca": "SparsePCA",
        "sparsenmf": "SparseNMF",
        "snmf": "SparseNMF",
        "ae": "AE",
        "autoencoder": "AE",
        "vae": "VAE",
        "sparseae": "SparseAE",
        "sparsevae": "SparseVAE",
    }
    return display_names.get(method, str(dimred_display).strip())


def _component_axis_labels(dimred_display, dimred_method):
    """Handle component axis labels."""
    method = dimred_method.replace("_", "").replace("-", "")
    if method in ("", "none"):
        return "PC1 (PCA fitted for visualization)", "PC2 (PCA fitted for visualization)"
    if method == "pca":
        return "PC1 (applied PCA score)", "PC2 (applied PCA score)"
    if method == "sparsepca":
        return "Sparse PC1 (applied SparsePCA score)", "Sparse PC2 (applied SparsePCA score)"
    if method in ("sparsenmf", "snmf") or "nmf" in method:
        return "NMF component 1 score", "NMF component 2 score"
    if method in ("ae", "autoencoder"):
        return "AE latent dimension 1", "AE latent dimension 2"
    if method == "vae":
        return "VAE latent dimension 1", "VAE latent dimension 2"
    if method == "sparseae":
        return "SparseAE latent dimension 1", "SparseAE latent dimension 2"
    if method == "sparsevae":
        return "SparseVAE latent dimension 1", "SparseVAE latent dimension 2"
    return f"{dimred_display} dimension 1", f"{dimred_display} dimension 2"


def _save_matplotlib_png_pdf(fig, output_stem, dpi=300, **savefig_kwargs):
    """Save a Matplotlib figure as PNG and PDF using the same stem."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for extension in ("png", "pdf"):
        output_path = output_stem.with_suffix(f".{extension}")
        kwargs = dict(savefig_kwargs)
        if extension == "png":
            kwargs.setdefault("dpi", dpi)
        fig.savefig(output_path, bbox_inches="tight", **kwargs)
        output_paths.append(output_path)
    return output_paths


def plot_latent_embeddings(run_name, final_metrics, out_dir=None, file_prefix=None):
    """Plot applied dimensions, PCA projection, and t-SNE of the representation."""
    ae_res = final_metrics.get("ae_res", {}) if isinstance(final_metrics, dict) else {}
    latent, nested_name = _extract_final_latent(ae_res)
    if latent is None:
        print(f"{run_name}: no latent representation available, skipping component/t-SNE plot.")
        return None

    X = np.asarray(latent)
    labels = np.asarray(final_metrics.get("final_labels", [])) if isinstance(final_metrics, dict) else np.asarray([])
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or X.shape[1] < 2:
        print(f"{run_name}: representation has <2 dimensions, skipping component/t-SNE plot.")
        return None
    if len(labels) not in (0, len(X)):
        print(f"{run_name}: label/representation length mismatch, skipping component/t-SNE plot.")
        return None

    dimred_value = final_metrics.get(
        "dim_reduction",
        final_metrics.get(
            "dim_reduction_label",
            final_metrics.get("final_reporting", {}).get("compute_context", {}).get("dim_reduction", run_name),
        ),
    )
    dimred_display_raw = str(dimred_value if dimred_value is not None else run_name).strip()
    dimred_method = dimred_display_raw.lower()
    if nested_name and dimred_method in ("", "none"):
        dimred_display_raw = nested_name
        dimred_method = nested_name.lower()
    dimred_display = _format_dimred_display(dimred_display_raw)

    if dimred_method in ("", "none"):
        latent_proj = X[:, :2]
        latent_title = f"{run_name}: unreduced preprocessed feature dimensions 1-2"
        tsne_source_label = "unreduced preprocessed features"
    else:
        latent_proj = X[:, :2]
        latent_title = f"{run_name}: applied {dimred_display} dimensions 1-2"
        tsne_source_label = f"applied {dimred_display} representation"

    pca_proj = PCA(n_components=2).fit_transform(X)
    pca_title = f"{run_name}: PCA projection of {tsne_source_label}"
    perplexity = max(5, min(30, len(X) - 1))
    tsne_proj = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(X)
    x_label, y_label = _component_axis_labels(dimred_display, dimred_method)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if len(labels) == len(X):
        classes = np.unique(labels)
        palette = sns.color_palette(n_colors=len(classes))
        color_map = {cls: palette[j] for j, cls in enumerate(classes)}
        colors = [color_map[cls] for cls in labels]
        legend_handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", color=color_map[cls], label=str(cls))
            for cls in classes
        ]
    else:
        colors = _THEME_CATS[1]
        legend_handles = []

    axes[0].scatter(latent_proj[:, 0], latent_proj[:, 1], c=colors, s=14, alpha=0.7, edgecolors="none")
    axes[0].set_title(latent_title)
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)

    axes[1].scatter(pca_proj[:, 0], pca_proj[:, 1], c=colors, s=14, alpha=0.7, edgecolors="none")
    axes[1].set_title(pca_title)
    axes[1].set_xlabel(f"PC1 (PCA projection of {tsne_source_label})")
    axes[1].set_ylabel(f"PC2 (PCA projection of {tsne_source_label})")

    axes[2].scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=colors, s=14, alpha=0.7, edgecolors="none")
    axes[2].set_title(f"{run_name}: t-SNE on {tsne_source_label}")
    axes[2].set_xlabel(f"t-SNE 1 ({tsne_source_label})")
    axes[2].set_ylabel(f"t-SNE 2 ({tsne_source_label})")

    for ax in axes:
        if legend_handles:
            ax.legend(handles=legend_handles, title="Label", loc="best", frameon=True)
    fig.tight_layout()

    if out_dir is not None:
        stem = Path(out_dir) / f"{safe_name(file_prefix or run_name)}_latent_pca_tsne"
        saved_paths = _save_matplotlib_png_pdf(fig, stem, dpi=300)
        for output_path in saved_paths:
            print("Saved latent/PCA/t-SNE plot to:", output_path)

    return fig


def plot_pred_modality(df, name):
    """Plot quick prediction diagnostics for numeric columns in a modality result table."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        print(f"{name}: no numeric columns to plot.")
        return None
    n_cols = min(4, numeric.shape[1])
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 3.5))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, numeric.columns[:n_cols]):
        ax.hist(numeric[col].dropna(), bins=30, color=_THEME_CATS[1], alpha=0.8)
        ax.set_title(str(col))
    fig.suptitle(name)
    fig.tight_layout()
    return fig


def finite_values(frame, col):
    """Return finite numeric values from one DataFrame column."""
    vals = pd.to_numeric(frame[col], errors="coerce") if col in frame else pd.Series(dtype=float)
    vals = vals[np.isfinite(vals)]
    return vals


def hist_if_finite(frame, col, bins=30, xlabel=None, title=None):
    """Plot a histogram only when a column has finite numeric values."""
    vals = finite_values(frame, col)
    if vals.empty:
        return None
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(vals, bins=bins, color=_THEME_CATS[1], alpha=0.8)
    ax.set_xlabel(xlabel or col)
    ax.set_title(title or col)
    fig.tight_layout()
    return fig


def safe_name(value):
    """Convert a label to a filesystem/column-name friendly token."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def ordered_cluster_labels(labels):
    """Return cluster labels in stable numeric-then-text order."""
    return sorted(pd.Series(labels).dropna().astype(str).unique(), key=cluster_sort_key)


def prepare_feature_table(df, clusters, subject_id_column="src_subject_id"):
    """Align a feature DataFrame with cluster labels and drop the subject ID column."""
    features = df.drop(columns=[subject_id_column], errors="ignore").reset_index(drop=True)
    clusters = pd.Series(np.asarray(clusters).reshape(-1), name="cluster").reset_index(drop=True)
    n = min(len(features), len(clusters))
    out = features.iloc[:n].copy()
    out["cluster"] = clusters.iloc[:n].astype(str).to_numpy()
    return out


def feature_kind(series, max_categorical_levels=8):
    """Classify a feature as continuous, categorical, binary, or empty."""
    vals = pd.Series(series).dropna()
    if vals.empty:
        return "empty"
    numeric = pd.to_numeric(vals, errors="coerce")
    if numeric.notna().all():
        n_unique = numeric.nunique()
        if n_unique <= 2:
            return "binary"
        if n_unique <= max_categorical_levels and _is_integer_like(numeric):
            return "categorical"
        return "continuous"
    return "binary" if vals.astype(str).nunique() <= 2 else "categorical"


def categorical_like(series, max_discrete_levels=6):
    """Return True when a feature should be treated as categorical/discrete."""
    return feature_kind(series, max_categorical_levels=max_discrete_levels) in {"binary", "categorical"}


def eta_squared_by_category(values, categories):
    """Effect-size style score for numeric values grouped by categories."""
    frame = pd.DataFrame({"value": pd.to_numeric(pd.Series(values), errors="coerce"), "category": categories}).dropna()
    if frame.empty or frame["category"].nunique() < 2:
        return np.nan
    grand_mean = frame["value"].mean()
    ss_between = frame.groupby("category")["value"].agg(lambda x: len(x) * (x.mean() - grand_mean) ** 2).sum()
    ss_total = ((frame["value"] - grand_mean) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


def safe_spearman(values, factor_scores):
    """Spearman correlation that returns NaN on invalid or constant inputs."""
    from scipy.stats import spearmanr
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    y = pd.to_numeric(pd.Series(factor_scores), errors="coerce")
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan
    return float(spearmanr(frame["x"], frame["y"]).correlation)


def continuous_score(series, clusters, cluster_order=None):
    """Rank continuous features by between-cluster mean spread."""
    frame = pd.DataFrame({"value": pd.to_numeric(pd.Series(series), errors="coerce"), "cluster": clusters}).dropna()
    if frame.empty or frame["cluster"].nunique() < 2:
        return np.nan
    means = frame.groupby("cluster")["value"].mean()
    if cluster_order is not None:
        means = means.reindex([str(c) for c in cluster_order]).dropna()
    return float(means.max() - means.min()) if not means.empty else np.nan


def categorical_values(series, max_levels=10):
    """Return categorical values, collapsing rare overflow levels if needed."""
    vals = pd.Series(series).astype("object")
    levels = vals.dropna().astype(str).value_counts()
    keep = set(levels.head(max_levels).index)
    return vals.astype(str).where(vals.astype(str).isin(keep), other="Other")


def categorical_score(series, clusters, cluster_order=None):
    """Rank categorical features by chi-square association with clusters."""
    frame = pd.DataFrame({"value": categorical_values(series), "cluster": clusters}).dropna()
    if frame.empty or frame["value"].nunique() < 2 or frame["cluster"].nunique() < 2:
        return np.nan
    table = pd.crosstab(frame["cluster"], frame["value"])
    try:
        chi2_stat, _, _, _ = chi2_contingency(table, correction=False)
    except Exception:
        return np.nan
    n = table.to_numpy().sum()
    denom = n * (min(table.shape) - 1)
    return float(np.sqrt(chi2_stat / denom)) if denom > 0 else np.nan


def rank_features_by_cluster(features, clusters, top_k=10):
    """Rank features by how strongly they separate cluster labels."""
    clusters = pd.Series(np.asarray(clusters).reshape(-1)).astype(str)
    rows = []
    for feature in features.columns:
        kind = feature_kind(features[feature])
        if kind == "empty":
            continue
        if kind == "continuous":
            score = continuous_score(features[feature], clusters)
        else:
            score = categorical_score(features[feature], clusters)
        rows.append({"feature": feature, "kind": kind, "score": score})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("score", ascending=False, na_position="last").head(top_k).reset_index(drop=True)


_SIMPLECLUST_PUBLICATION_LABELS = {
    "CAARMS_delusions": "CAARMS delusions",
    "chrnsipr_motiv_pleas_dimension": "NSI-PR motivation/pleasure",
    "chrnsipr_motivation_and_pleasure_dimension": "NSI-PR motivation/pleasure",
    "chrnsipr_dimin_expr_dimension": "NSI-PR diminished expression",
    "chrnsipr_diminished_expression_dimension": "NSI-PR diminished expression",
    "chrnsipr_item11_rating": "NSI-PR alogia",
    "chrnsipr_avolition_domain": "NSI-PR avolition",
    "chrnsipr_asociality_domain": "NSI-PR asociality",
    "chrnsipr_anhedonia_domain": "NSI-PR anhedonia",
    "chrnsipr_blunted_affect_domain": "NSI-PR blunted affect",
    "chrsofas_currscore": "Current SOFAS",
    "chrsofas_currscore12mo": "SOFAS 12 months ago",
    "chrgfs_gf_social_scale": "Current social GF",
    "gf_social_scale": "Current social GF",
    "chrgfs_gf_social_high": "Best social GF",
    "gf_social_high": "Best social GF",
    "chrgfs_gf_social_low": "Worst social GF",
    "gf_social_low": "Worst social GF",
    "chrgfr_gf_role_scole": "Current role GF",
    "chrgfr_gf_role_score": "Current role GF",
    "gf_role_scole": "Current role GF",
    "gf_role_score": "Current role GF",
    "chrgfr_gf_role_high": "Best role GF",
    "gf_role_high": "Best role GF",
    "chrgfr_gf_role_low": "Worst role GF",
    "gf_role_low": "Worst role GF",
    "chrgfrs_global_role_decline": "Role GF decline",
    "chrgfss_global_social_decline": "Social GF decline",
    "chrsofas_lowscore": "Lowest SOFAS in past year",
    "chrsofas_premorbid": "Premorbid SOFAS",
    "chriq_fsiq": "FSIQ",
    "fsiq": "FSIQ",
    "psychs_sips_p5": "SIPS P5 disorganization",
    "psychs_pos_tot": "PSYCHS positive total",
    "sips_pos_tot": "SIPS positive total",
    "caarms_pos_tot": "CAARMS positive total",
    "psychs_caarms_p1": "CAARMS P1 delusions",
    "psychs_caarms_p2": "CAARMS P2 delusions",
    "psychs_caarms_p3": "CAARMS P3 grandiosity",
    "psychs_caarms_p4": "CAARMS P4 hallucinations",
    "psychs_sips_p1": "SIPS P1 unusual thought",
    "psychs_sips_p2": "SIPS P2 suspiciousness",
    "psychs_sips_p3": "SIPS P3 grandiosity",
    "psychs_sips_p4": "SIPS P4 perceptual abn.",
    "chrbprs_activation": "BPRS activation",
    "chrbprs_bprs_total": "BPRS total",
    "chrbprs_affect_subscale": "BPRS affect",
    "chrbprs_positive_symptom_subscale": "BPRS positive",
    "chrbprs_pos_symp_subscale": "BPRS positive",
    "chrbprs_negative_symptom_subscale": "BPRS negative",
    "bprs_factor_neg_symp": "BPRS negative",
    "chrbprs_activation_subscale": "BPRS activation",
    "chrbprs_disorganization_subscale": "BPRS disorganization",
    "chrbprs_disorg_subscale": "BPRS disorganization",
    "bprs_factor_activation": "BPRS activation",
    "chrcssrsb_intensity_lifetime": "C-SSRS lifetime intensity",
    "chrcssrs_intensity_lifetime": "C-SSRS lifetime intensity",
    "chrcssrsb_intensity_pastmonth": "C-SSRS past-month intensity",
    "chrcssrs_intensity_pastmonth": "C-SSRS past-month intensity",
    "chrcdss_total": "CDSS total",
    "cdss": "CDSS total",
    "chrpps_empty": "Subjective emptiness",
    "chrpps_sum10": "Adult life-event risk",
    "chrpps_sum13": "Trauma risk score",
    "chrpsychs_scr_ac1": "Psychosis",
    "chrpsychs_scr_ac2": "CAARMS BLIPS",
    "chrpsychs_scr_ac3": "CAARMS APS frequency",
    "chrpsychs_scr_ac4": "CAARMS APS intensity",
    "chrpsychs_scr_ac5": "CAARMS APS",
    "chrpsychs_scr_ac6": "CAARMS vulnerability",
    "chrpsychs_scr_ac7": "CAARMS UHR group",
    "sips_bips_scr_lifetime": "BIPS status",
    "sips_aps_scr_lifetime": "APS status",
    "sips_grd_scr_lifetime": "GRD status",
    "chrassist_tobacco": "ASSIST tobacco",
    "chrassist_alcohol": "ASSIST alcohol",
    "chrassist_cannabis": "ASSIST cannabis",
    "chrassist_cocaine": "ASSIST cocaine",
    "chrassist_amphetamines": "ASSIST amphetamines",
    "chrassist_inhalants": "ASSIST inhalants",
    "chrassist_sedatives": "ASSIST sedatives",
    "chrassist_hallucinogens": "ASSIST hallucinogens",
    "chrassist_opiods": "ASSIST opioids",
    "chrassist_opioids": "ASSIST opioids",
    "chrassist_other": "ASSIST other substances",
    "chroasis_oasisscore": "OASIS total",
    "chroasis_oasis_total10": "OASIS total",
    "chrpromis_total": "PROMIS total",
    "chrpss_total": "PSS total",
    "chrpss_perceived_stress_scale_total": "PSS total",
    "pss10_total": "PSS-10 total",
    "ctq_physical": "CTQ physical abuse",
    "ctq_sexual": "CTQ sexual abuse",
    "ctq_emotional": "CTQ emotional abuse",
    "ctq_other": "CTQ other adversity",
    "chrdemo_edu_max": "Highest education",
    "chrdemo_income": "Income source",
    "chrdemo_parent_occupation": "Parent occupation",
    "chrdemo_sexassigned": "Sex assigned at birth",
    "chrdemo_student": "Student status",
    "chrdemo_working": "Paid employment",
    "race": "Race",
    "sex": "Sex",
    "Site": "Site",
    "cnb_er40_cr": "Emotion recognition",
    "cnb_ctap_dom": "Finger tapping",
    "cnb_cptn_tp": "Continuous performance",
    "cnb_volt_cr": "Visual learning",
    "cnb_digsym_dscor": "Digit symbol",
    "cnb_fnb2_tp": "Fractal n-back",
    "cnb_pllt_pllttcr": "Word list recall",
    "chrrecruit" : "Recruitment source",
    "chrpps_total" : "CTQ total score",
    "chrpps_pa_sum" : "CTQ physical abuse score",
    "chrpps_sa_sum" : "CTQ sexual abuse score",
    "chrpps_ea_sum" : "CTQ emotional abuse score",
    "chrpps_en_sum" : "CTQ emotional neglect score",
    "chrpps_pn_sum" : "CTQ physical neglect score",

}


def _title_case_fallback_label(feature):
    """Handle title case fallback label."""
    label = str(feature).replace("_", " ").strip()
    if not label:
        return str(feature)
    return " ".join(
        token.upper()
        if token.casefold() in {"caarms", "sips", "psychs", "nsi", "pr", "bprs", "cdss", "oasis", "promis", "pss", "iq"}
        else token
        for token in label.split()
    )


def display_feature_name(feature):
    """Return a publication-facing plot label without internal column prefixes."""
    name = str(feature).split("__", 1)[-1]
    if name in _SIMPLECLUST_PUBLICATION_LABELS:
        return _SIMPLECLUST_PUBLICATION_LABELS[name]
    if name.casefold().startswith(("psychs_", "chrnsipr_", "chrgf", "cnb_", "chriq_")):
        labeler = globals().get("_mixed_heatmap_feature_label")
        if callable(labeler):
            return labeler(name)
    return _title_case_fallback_label(name)


def simpleclust_feature_sort_key(feature):
    """Sort Simpleclust variables into the requested clinical/cognition order."""
    name = str(feature).split("__", 1)[-1]
    lower = name.casefold()

    if lower == "caarms_delusions":
        group = 0
        item_number = -1
    elif lower.startswith("psychs_"):
        group = 1
        match = re.search(r"(?:^|_)[a-z]*p(\d+)(?:_|$)", lower)
        item_number = int(match.group(1)) if match else float("inf")
    elif lower.startswith("chrnsipr_"):
        group = 2
        item_number = -1
    elif "bprs" in lower and "activation" in lower:
        group = 3
        item_number = -1
    elif lower.startswith("chrcdss_") or lower == "cdss":
        group = 4
        item_number = -1
    elif "oasis" in lower:
        group = 5
        item_number = -1
    elif "perceived_stress_scale" in lower:
        group = 6
        item_number = -1
    elif "promis" in lower:
        group = 7
        item_number = -1
    elif lower.startswith("chrgf"):
        group = 8
        item_number = -1
    elif lower.startswith("cnb_"):
        group = 9
        item_number = -1
    elif "fsiq" in lower:
        group = 10
        item_number = -1
    else:
        group = 11
        item_number = -1

    return group, item_number, lower


def order_simpleclust_features(features):
    """Return feature names in the standard Simpleclust plotting order."""
    return sorted(list(features), key=simpleclust_feature_sort_key)


def plot_continuous_feature(ax, plot_df, feature, cluster_order, palette):
    """Plot one continuous feature by cluster on an existing axis."""
    sns.violinplot(data=plot_df, x="cluster", y=feature, order=cluster_order, palette=palette, ax=ax, cut=0)
    sns.stripplot(data=plot_df, x="cluster", y=feature, order=cluster_order, color="black", size=2, alpha=0.25, ax=ax)
    ax.set_title(display_feature_name(feature))


def plot_categorical_feature(ax, plot_df, feature, cluster_order):
    """Plot one categorical feature as cluster-normalized proportions."""
    table = pd.crosstab(plot_df["cluster"], plot_df[feature], normalize="index").reindex(cluster_order)
    table.plot(kind="bar", stacked=True, ax=ax, legend=False)
    ax.set_title(display_feature_name(feature))
    ax.set_ylabel("Proportion")


def plot_ranked_feature_grid(features, clusters, ranked, modality, output_dir=None):
    """Plot a grid of top-ranked cluster-separating features."""
    if ranked is None or ranked.empty:
        return None
    plot_df = prepare_feature_table(features, clusters, subject_id_column=None) if "cluster" not in features else features.copy()
    cluster_order = ordered_cluster_labels(plot_df["cluster"])
    palette = build_group_palette(modality, cluster_order)
    n = len(ranked)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    for ax, (_, row) in zip(axes.ravel(), ranked.iterrows()):
        feature = row["feature"]
        if row["kind"] == "continuous":
            plot_continuous_feature(ax, plot_df, feature, cluster_order, palette)
        else:
            plot_categorical_feature(ax, plot_df, feature, cluster_order)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{modality}: top cluster-separating features")
    fig.tight_layout()
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        _save_matplotlib_png_pdf(fig, Path(output_dir) / f"{safe_name(modality)}_ranked_features", dpi=300)
    return fig


def make_chr_cc_feature_boxplots(
    chr_df,
    clusters,
    cc_df,
    top_k=30,
    out_dir=None,
    file_prefix="chr_vs_cc",
    cc_label="CC",
    title_prefix="Top features by subgroup",
    subject_id_column="src_subject_id",
    plots_per_page=6,
):
    """Rank CHR features and compare CHR subgroups, with controls shown in plots.

    Only features that are numeric in both samples are plotted. The returned
    table contains an omnibus Kruskal-Wallis test across CHR subgroups and
    pairwise CHR subgroup Mann-Whitney tests, with Benjamini-Hochberg
    correction applied separately to each family of p-values. Controls are
    plotted as a visual reference only and are not included in these tests.
    """
    from scipy.stats import kruskal, mannwhitneyu

    chr_features = chr_df.drop(columns=[subject_id_column], errors="ignore").reset_index(drop=True)
    cc_features = cc_df.drop(columns=[subject_id_column], errors="ignore").reset_index(drop=True)
    cluster_values = pd.Series(np.asarray(clusters).reshape(-1), name="group")
    if len(chr_features) != len(cluster_values):
        raise ValueError(
            "chr_df and clusters must have the same number of rows "
            f"({len(chr_features)} != {len(cluster_values)})."
        )

    common_features = [column for column in chr_features.columns if column in cc_features.columns]
    numeric_features = []
    numeric_chr = {}
    numeric_cc = {}
    for feature in common_features:
        chr_values = pd.to_numeric(chr_features[feature], errors="coerce")
        cc_values = pd.to_numeric(cc_features[feature], errors="coerce")
        if chr_values.notna().any() and cc_values.notna().any():
            numeric_features.append(feature)
            numeric_chr[feature] = chr_values
            numeric_cc[feature] = cc_values

    if not numeric_features:
        chr_examples = list(chr_features.columns[:5])
        cc_examples = list(cc_features.columns[:5])
        raise ValueError(
            "No shared numeric features were found in chr_df and cc_df. "
            f"Example CHR columns: {chr_examples}; example CC columns: {cc_examples}. "
            "For simpleclust, merge all preprocessed CC modalities using the "
            "same 'Modality__feature' names as final_metrics['data']."
        )

    ranked = rank_features_by_cluster(
        pd.DataFrame({feature: numeric_chr[feature] for feature in numeric_features}),
        cluster_values,
        top_k=min(int(top_k), len(numeric_features)),
    )
    if ranked.empty:
        raise ValueError("No shared numeric features were available for boxplots.")
    ranked_order = {
        feature: index
        for index, feature in enumerate(order_simpleclust_features(ranked["feature"]))
    }
    ranked = (
        ranked.assign(_plot_order=ranked["feature"].map(ranked_order))
        .sort_values("_plot_order")
        .drop(columns="_plot_order")
        .reset_index(drop=True)
    )

    cluster_order = ordered_cluster_labels(cluster_values)
    plot_group_order = [cc_label] + cluster_order
    palette = build_group_palette("Simple clustering", plot_group_order)
    palette[cc_label] = _THEME_CC

    plot_frames = {}
    stats_rows = []
    for rank, feature in enumerate(ranked["feature"], start=1):
        chr_plot = pd.DataFrame({"value": numeric_chr[feature], "group": cluster_values.astype(str)})
        cc_plot = pd.DataFrame({"value": numeric_cc[feature], "group": cc_label})
        plot_df = pd.concat([cc_plot, chr_plot], ignore_index=True).dropna(subset=["value"])
        plot_frames[feature] = plot_df

        samples = [
            plot_df.loc[plot_df["group"] == group, "value"].to_numpy(dtype=float)
            for group in cluster_order
        ]
        valid_samples = [sample for sample in samples if len(sample) > 0]
        omnibus_stat, omnibus_p = (np.nan, np.nan)
        if len(valid_samples) >= 2:
            try:
                omnibus_stat, omnibus_p = kruskal(*valid_samples)
            except ValueError:
                pass

        for group_a, group_b in combinations(cluster_order, 2):
            values_a = plot_df.loc[plot_df["group"] == group_a, "value"].to_numpy(dtype=float)
            values_b = plot_df.loc[plot_df["group"] == group_b, "value"].to_numpy(dtype=float)
            u_stat, pairwise_p = (np.nan, np.nan)
            if len(values_a) > 0 and len(values_b) > 0:
                try:
                    u_stat, pairwise_p = mannwhitneyu(values_a, values_b, alternative="two-sided")
                except ValueError:
                    pass
            stats_rows.append(
                {
                    "feature_rank": rank,
                    "feature": feature,
                    "cluster_score": ranked.loc[rank - 1, "score"],
                    "comparison": f"{group_a} vs {group_b}",
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_a": len(values_a),
                    "n_b": len(values_b),
                    "median_a": np.median(values_a) if len(values_a) else np.nan,
                    "median_b": np.median(values_b) if len(values_b) else np.nan,
                    "median_difference": (
                        np.median(values_a) - np.median(values_b)
                        if len(values_a) and len(values_b)
                        else np.nan
                    ),
                    "mann_whitney_u": u_stat,
                    "pairwise_p": pairwise_p,
                    "omnibus_kruskal_h": omnibus_stat,
                    "omnibus_p": omnibus_p,
                }
            )

    stats_df = pd.DataFrame(stats_rows)
    stats_df["pairwise_p_fdr"] = _benjamini_hochberg(stats_df["pairwise_p"].to_numpy(dtype=float))
    omnibus = stats_df.drop_duplicates("feature")["omnibus_p"].to_numpy(dtype=float)
    omnibus_fdr = _benjamini_hochberg(omnibus)
    omnibus_map = dict(zip(stats_df.drop_duplicates("feature")["feature"], omnibus_fdr))
    stats_df["omnibus_p_fdr"] = stats_df["feature"].map(omnibus_map)

    if out_dir:
        output_path = Path(out_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        n_per_page = max(1, int(plots_per_page))
        features = ranked["feature"].tolist()
        for page_index, start in enumerate(range(0, len(features), n_per_page), start=1):
            page_features = features[start:start + n_per_page]
            n_cols = 2
            n_rows = int(np.ceil(len(page_features) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.5 * n_rows), squeeze=False)
            for ax, feature in zip(axes.ravel(), page_features):
                sns.boxplot(
                    data=plot_frames[feature], x="group", y="value", order=plot_group_order,
                    palette=palette, ax=ax, showfliers=False,
                )
                sns.stripplot(
                    data=plot_frames[feature], x="group", y="value", order=plot_group_order,
                    color="black", size=2, alpha=0.25, ax=ax,
                )
                feature_stats = stats_df.loc[stats_df["feature"] == feature].iloc[0]
                ax.set_title(
                    f"{display_feature_name(feature)}\n"
                    f"Kruskal-Wallis FDR p={feature_stats['omnibus_p_fdr']:.3g}"
                )
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=30)
            for ax in axes.ravel()[len(page_features):]:
                ax.axis("off")
            fig.suptitle(f"{title_prefix} (page {page_index})", y=1.01)
            fig.tight_layout()
            _save_matplotlib_png_pdf(
                fig,
                output_path / f"{safe_name(file_prefix)}_boxplots_page_{page_index:02d}",
                dpi=300,
            )
            plt.close(fig)

    return stats_df


# Backward-compatible names used in existing notebooks.
_truthy_profile_value = truthy_profile_value
_parse_profile_exports = parse_profile_exports
_infer_notebook_profile = infer_notebook_profile
_find_repo_root = find_repo_root
_profile_enabled_for_sensitivity = profile_enabled_for_sensitivity
_display_if_available = display_if_available
_get_nested = get_nested
_flatten_sensitivity_results = flatten_sensitivity_results
_categorical_like = categorical_like
_eta_squared_by_category = eta_squared_by_category
_safe_spearman = safe_spearman
_safe_name = safe_name
_ordered_cluster_labels = ordered_cluster_labels
_prepare_feature_table = prepare_feature_table
_feature_kind = feature_kind
_continuous_score = continuous_score
_categorical_values = categorical_values
_categorical_score = categorical_score
_rank_features_by_cluster = rank_features_by_cluster
_plot_continuous_feature = plot_continuous_feature
_plot_categorical_feature = plot_categorical_feature
_plot_ranked_feature_grid = plot_ranked_feature_grid


def _normalize(v, eps=1e-12):
    """Normalize a vector-like object to sum to one; retained for notebook compatibility."""
    arr = np.asarray(v, dtype=float)
    total = float(np.nansum(arr))
    if abs(total) <= eps:
        return np.zeros_like(arr, dtype=float)
    return arr / total


def alluvial_sankey_force_high_top(
    labels_by_modality,
    final_labels,
    stage_order,
    final_name="final",
    high_token="high_severity",
    low_token="low_severity",
    **kwargs,
):
    """
    Backward-compatible wrapper for the generalized alluvial Sankey helper.

    Older notebooks used this fixed high/low name; the implementation now
    delegates to alluvial_sankey_general, which supports any number of clusters.
    """
    return alluvial_sankey_general(
        labels_by_modality=labels_by_modality,
        final_labels=final_labels,
        stage_order=stage_order,
        final_name=final_name,
        high_token=high_token,
        low_token=low_token,
        **kwargs,
    )


def _labels_for_modality(labels_by_modality, modality, modality_index):
    """Handle labels for modality."""
    if isinstance(labels_by_modality, dict):
        if modality not in labels_by_modality:
            raise KeyError(f"Missing subgroup labels for modality '{modality}'.")
        return labels_by_modality[modality]
    if modality_index >= len(labels_by_modality):
        raise IndexError(f"Missing subgroup labels at modality index {modality_index} ({modality}).")
    return labels_by_modality[modality_index]


def plot_subgroup_feature_profiles_by_modality(
    data_by_modality,
    labels_by_modality,
    plots_dir=None,
    subject_id_column="src_subject_id",
    sample_label=None,
    subgroup_label="Cluster",
    plot_kinds=("line",),
    show=True,
    control_data_by_modality=None,
    control_label="Community controls",
    control_group="CC",
    control_color="#4A4A4A",
):
    """
    Plot subgroup mean profiles across numeric features for each modality.

    Available plot kinds are ``line`` for connected mean profiles with SD bars,
    ``dot`` for unconnected means with SD bars, and ``heatmap`` for subgroup
    mean matrices. Non-numeric and all-missing columns are skipped because their
    subgroup means are not interpretable as numeric profiles. When supplied,
    control_data_by_modality is aligned to the discovery features and shown as
    an additional dark-grey control profile.
    """
    if not isinstance(data_by_modality, dict):
        raise TypeError("data_by_modality must be a dict of modality DataFrames.")
    if isinstance(plot_kinds, str):
        plot_kinds = (plot_kinds,)
    plot_kinds = tuple(dict.fromkeys(str(kind).lower() for kind in plot_kinds))
    valid_plot_kinds = {"line", "dot", "heatmap"}
    invalid_plot_kinds = sorted(set(plot_kinds) - valid_plot_kinds)
    if invalid_plot_kinds:
        raise ValueError(
            f"Unknown profile plot kinds {invalid_plot_kinds}; use {sorted(valid_plot_kinds)}."
        )
    if not plot_kinds:
        raise ValueError("At least one profile plot kind is required.")

    out_dir = None
    if plots_dir is not None:
        out_dir = os.path.join(plots_dir, "subgroup_feature_profiles")
        os.makedirs(out_dir, exist_ok=True)

    summaries = {}
    sample_prefix = f"{sample_label} " if sample_label else ""
    filename_prefix = f"{sample_label}_" if sample_label else ""

    for modality_index, (modality, modality_df) in enumerate(data_by_modality.items()):
        labels = pd.Series(
            np.asarray(_labels_for_modality(labels_by_modality, modality, modality_index)).reshape(-1)
        ).astype(str)
        feature_df = modality_df.drop(columns=[subject_id_column], errors="ignore").reset_index(drop=True)
        if len(labels) != len(feature_df):
            raise ValueError(
                f"{modality}: labels length ({len(labels)}) does not match rows ({len(feature_df)})."
            )

        numeric_df = feature_df.apply(pd.to_numeric, errors="coerce")
        numeric_df = numeric_df.loc[:, numeric_df.notna().any(axis=0)]
        if numeric_df.empty:
            warnings.warn(f"{modality}: no numeric features available for subgroup profile plot.")
            continue
        numeric_df = numeric_df.reindex(columns=order_simpleclust_features(numeric_df.columns))

        profile_df = numeric_df.copy()
        profile_df["_subgroup"] = labels.to_numpy()
        cluster_groups = sorted(profile_df["_subgroup"].dropna().unique(), key=cluster_sort_key)

        has_controls = False
        if control_data_by_modality is not None:
            if not isinstance(control_data_by_modality, dict):
                raise TypeError("control_data_by_modality must be a dict of modality DataFrames.")
            control_df = control_data_by_modality.get(modality)
            if control_df is not None:
                control_features = (
                    control_df.drop(columns=[subject_id_column], errors="ignore")
                    .reset_index(drop=True)
                    .reindex(columns=numeric_df.columns)
                    .apply(pd.to_numeric, errors="coerce")
                )
                control_features = control_features.dropna(axis=0, how="all")
                if not control_features.empty:
                    control_features["_subgroup"] = str(control_group)
                    profile_df = pd.concat([profile_df, control_features], ignore_index=True)
                    has_controls = True

        group_order = cluster_groups + ([str(control_group)] if has_controls else [])
        means = profile_df.groupby("_subgroup", observed=False)[numeric_df.columns].mean().reindex(group_order)
        sds = profile_df.groupby("_subgroup", observed=False)[numeric_df.columns].std().reindex(group_order)
        counts = profile_df["_subgroup"].value_counts().reindex(group_order, fill_value=0)

        x = np.arange(len(numeric_df.columns))
        fig_width = min(max(12, 0.34 * len(numeric_df.columns)), 44)
        color_map = modality_cluster_palette(group_order, modality=modality)
        if has_controls:
            color_map[str(control_group)] = control_color
        safe_modality = str(modality).replace("/", "_").replace(" ", "_")
        legend_title = "Group" if has_controls else subgroup_label

        def legend_label(group):
            """Handle legend label."""
            if has_controls and str(group) == str(control_group):
                return f"{control_label} (n={int(counts.loc[group])})"
            return f"{subgroup_label} {group} (n={int(counts.loc[group])})"

        def finish_profile_figure(fig, plot_kind):
            """Handle finish profile figure."""
            fig.tight_layout()
            if out_dir is not None:
                if plot_kind == "line":
                    filename = f"{filename_prefix}{safe_modality}_subgroup_feature_profiles"
                else:
                    filename = f"{filename_prefix}{safe_modality}_subgroup_feature_profiles_{plot_kind}"
                _save_matplotlib_png_pdf(fig, os.path.join(out_dir, filename), dpi=300)
            if show:
                display_if_available(fig)
            plt.close(fig)

        if "line" in plot_kinds:
            fig, ax = plt.subplots(figsize=(fig_width, 7.5))
            for group in group_order:
                group_means = means.loc[group].to_numpy(dtype=float)
                group_sds = sds.loc[group].fillna(0).to_numpy(dtype=float)
                ax.errorbar(
                    x,
                    group_means,
                    yerr=group_sds,
                    label=legend_label(group),
                    color=color_map[str(group)],
                    marker="o",
                    markersize=3.6,
                    linewidth=1.8,
                    elinewidth=0.8,
                    capsize=1.8,
                    alpha=0.95,
                )
            ax.set_title(f"{sample_prefix}{modality}: mean numeric feature profiles by {subgroup_label.lower()}")
            ax.set_ylabel("Group mean +/- SD")
            ax.legend(title=legend_title, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
            ax.axhline(0, color=_MUTED, linewidth=0.8, alpha=0.35)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [display_feature_name(feature) for feature in numeric_df.columns],
                rotation=70,
                ha="right",
                fontsize=8,
            )
            ax.set_xlim(-0.5, len(numeric_df.columns) - 0.5)
            ax.set_xlabel("Feature")
            ax.grid(axis="y", alpha=0.18)
            finish_profile_figure(fig, "line")

        if "dot" in plot_kinds:
            fig, ax = plt.subplots(figsize=(fig_width, 7.5))
            offsets = np.linspace(-0.28, 0.28, len(group_order)) if len(group_order) > 1 else [0]
            for group, offset in zip(group_order, offsets):
                group_means = means.loc[group].to_numpy(dtype=float)
                group_sds = sds.loc[group].fillna(0).to_numpy(dtype=float)
                ax.errorbar(
                    x + offset,
                    group_means,
                    yerr=group_sds,
                    label=legend_label(group),
                    color=color_map[str(group)],
                    linestyle="none",
                    marker="o",
                    markersize=4.2,
                    elinewidth=0.9,
                    capsize=2,
                    alpha=0.95,
                )
            ax.set_title(f"{sample_prefix}{modality}: mean numeric feature dots by {subgroup_label.lower()}")
            ax.set_ylabel("Group mean +/- SD")
            ax.legend(title=legend_title, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
            ax.axhline(0, color=_MUTED, linewidth=0.8, alpha=0.35)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [display_feature_name(feature) for feature in numeric_df.columns],
                rotation=70,
                ha="right",
                fontsize=8,
            )
            ax.set_xlim(-0.6, len(numeric_df.columns) - 0.4)
            ax.set_xlabel("Feature")
            ax.grid(axis="y", alpha=0.18)
            finish_profile_figure(fig, "dot")

        if "heatmap" in plot_kinds:
            fig_height = max(3.5, 0.52 * len(group_order) + 2.2)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            heatmap = sns.heatmap(
                means,
                cmap="vlag",
                center=0,
                linewidths=0.25,
                linecolor="white",
                cbar_kws={"label": "Subgroup mean"},
                xticklabels=[display_feature_name(feature) for feature in means.columns],
                ax=ax,
            )
            heatmap.set_yticklabels(
                [legend_label(group) for group in group_order],
                rotation=0,
            )
            for group, tick in zip(group_order, ax.get_yticklabels()):
                tick.set_color(color_map[str(group)])
            ax.set_title(f"{sample_prefix}{modality}: subgroup mean numeric feature heatmap")
            ax.set_xlabel("Feature")
            ax.set_ylabel(subgroup_label)
            ax.tick_params(axis="x", labelrotation=70, labelsize=8)
            finish_profile_figure(fig, "heatmap")

        summaries[modality] = {
            "mean": means,
            "sd": sds,
            "n": counts,
            "features": numeric_df.columns.tolist(),
            "includes_controls": has_controls,
        }

    return summaries


def _mixed_heatmap_variable_type(values):
    """Classify a report variable as continuous, binary, or categorical."""
    vals = pd.Series(values).dropna()
    if vals.empty:
        return None
    numeric = pd.to_numeric(vals, errors="coerce")
    if numeric.notna().all():
        return "binary" if numeric.nunique() <= 2 else "continuous"
    return "binary" if vals.astype(str).nunique() <= 2 else "categorical"


def _mixed_heatmap_binary_values(values):
    """Handle mixed heatmap binary values."""
    vals = pd.Series(values)
    numeric = pd.to_numeric(vals, errors="coerce")
    if numeric[vals.notna()].notna().all():
        levels = sorted(numeric.dropna().unique())
        positive = levels[-1] if levels else np.nan
        return numeric.eq(positive).astype(float).where(vals.notna())

    text = vals.astype("object")
    levels = sorted(text.dropna().astype(str).unique())
    positive = levels[-1] if levels else None
    return text.astype(str).eq(positive).astype(float).where(text.notna())


def _benjamini_hochberg(pvalues):
    """Return Benjamini-Hochberg adjusted p-values without a statsmodels dependency."""
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(values))
    if finite_idx.size == 0:
        return adjusted

    ordered_idx = finite_idx[np.argsort(values[finite_idx])]
    ordered = values[ordered_idx]
    ranks = np.arange(1, len(ordered) + 1, dtype=float)
    ordered_adjusted = np.minimum.accumulate((ordered * len(ordered) / ranks)[::-1])[::-1]
    adjusted[ordered_idx] = np.clip(ordered_adjusted, 0, 1)
    return adjusted


def _mixed_heatmap_p_value(values, groups, var_type):
    """Handle mixed heatmap p value."""
    frame = pd.DataFrame({"value": values, "group": groups}).dropna(subset=["value", "group"])
    if frame["group"].nunique() < 2:
        return np.nan

    try:
        if var_type == "continuous":
            samples = [
                pd.to_numeric(part["value"], errors="coerce").dropna().to_numpy()
                for _, part in frame.groupby("group", sort=False)
            ]
            samples = [sample for sample in samples if len(sample)]
            if len(samples) < 2 or all(np.nanstd(sample) == 0 for sample in samples):
                return np.nan
            return float(kruskal(*samples, nan_policy="omit").pvalue)

        table = pd.crosstab(frame["group"], frame["value"])
        if table.shape[0] < 2 or table.shape[1] < 2:
            return np.nan
        return float(chi2_contingency(table, correction=False).pvalue)
    except ValueError:
        return np.nan


def _mixed_heatmap_p_text(pvalue):
    """Handle mixed heatmap p text."""
    if not np.isfinite(pvalue):
        return ""
    if pvalue < 0.001:
        return "<0.001"
    return f"{pvalue:.3f}".lstrip("0")


def _mixed_heatmap_sig_text(pvalue):
    """Handle mixed heatmap sig text."""
    if not np.isfinite(pvalue) or pvalue >= 0.05:
        return ""
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    return "*"


def _mixed_heatmap_feature_label(feature):
    """Derive a compact report label from a structured element name."""
    feature = str(feature)
    direct_labels = {
        "interview_age": "Age at interview",
        "interviewage": "Age at interview",
        "recording_end_age": "Age at recording",
        "sex": "Sex",
        "sexassigned": "Sex assigned at birth",
        "chrdemo_sexassigned": "Sex assigned at birth",
        "race": "Race",
        "race1": "Race",
        "Site": "Site",
        "gender": "Gender",
        "gender_identity": "Gender identity",
        "ssgndr": "Gender identity",
    }
    if feature in direct_labels:
        return direct_labels[feature]

    instrument_prefixes = [
        ("chrnsipr_", "NSI-PR"),
        ("chrbprs_", "BPRS"),
        ("bprs_factor_", "BPRS"),
        ("bprs_", "BPRS"),
        ("chrsofas_", "SOFAS"),
        ("sofas_", "SOFAS"),
        ("chrgfsfu_", "GF social"),
        ("chrgfss_", "GF social"),
        ("chrgfs_", "GF social"),
        ("chrgfrfu_", "GF role"),
        ("chrgfrs_", "GF role"),
        ("chrgfr_", "GF role"),
        ("chrcssrsfu_", "C-SSRS"),
        ("chrcssrsb_", "C-SSRS"),
        ("chrcssrs_", "C-SSRS"),
        ("chroasis_", "OASIS"),
        ("oasis_", "OASIS"),
        ("chrpromis_", "PROMIS"),
        ("chrpss_", "PSS"),
        ("pss10_", "PSS-10"),
        ("chrpsychs_", "PSYCHS"),
        ("hcpsychs_", "PSYCHS"),
        ("psychs_", "PSYCHS"),
        ("sips_", "SIPS"),
        ("caarms_", "CAARMS"),
        ("chrtbi_", "TBI"),
        ("chrdemo_", ""),
        ("chriq_", "IQ"),
        ("cnb_", ""),
    ]
    instrument = ""
    remainder = feature
    for prefix, label in instrument_prefixes:
        if feature.startswith(prefix):
            instrument = label
            remainder = feature[len(prefix):]
            break

    token_labels = {
        "ac": "AC",
        "affect": "affect",
        "age": "age",
        "anhedonia": "anhedonia",
        "aps": "APS",
        "asociality": "asociality",
        "avolition": "avolition",
        "bips": "BIPS",
        "blunted": "blunted",
        "calg": "Calgary",
        "caarms": "CAARMS",
        "cdss": "CDSS",
        "cdstotal": "CDSS total",
        "cohen": "Cohen",
        "cr": "correct responses",
        "ctap": "CTAP",
        "curr": "current",
        "currscore": "current score",
        "currscore12mo": "current score 12 months ago",
        "decline": "decline",
        "dimension": "dimension",
        "digsym": "DIGSYM",
        "dimin": "diminished",
        "diminished": "diminished",
        "disorg": "disorganization",
        "disorganization": "disorganization",
        "dom": "dominant hand",
        "domain": "domain",
        "dscor": "score",
        "edu": "education",
        "er40": "ER40",
        "expr": "expression",
        "fnb2": "FNB2",
        "fsiq": "FSIQ",
        "fu": "follow-up",
        "gbl": "global",
        "gf": "GF",
        "global": "global",
        "grd": "GRD",
        "high": "highest",
        "highest": "highest",
        "in": "in",
        "inj": "injury",
        "int": "intensity",
        "intensity": "intensity",
        "lowscore": "lowest score",
        "low": "lowest",
        "lowest": "lowest",
        "lifetime": "lifetime",
        "max": "maximum",
        "motiv": "motivation",
        "motivation": "motivation",
        "negative": "negative",
        "neg": "negative",
        "ntimes": "number of injuries",
        "oasisscore": "OASIS score",
        "occupation": "occupation",
        "p": "P",
        "parent": "parent",
        "past": "past",
        "pastmonth": "past month",
        "pleas": "pleasure",
        "pleasure": "pleasure",
        "pllt": "PLLT",
        "pllttcr": "total correct recall",
        "pos": "positive",
        "positive": "positive",
        "premorbid": "premorbid",
        "promis": "PROMIS",
        "role": "role",
        "rs": "raw score",
        "scale": "scale",
        "scole": "score",
        "score": "score",
        "scr": "screening",
        "sips": "SIPS",
        "social": "social",
        "sofas": "SOFAS",
        "student": "student",
        "subscale": "subscale",
        "sumcdrs": "CDRS total",
        "symp": "symptoms",
        "symptom": "symptom",
        "symptoms": "symptoms",
        "t": "total",
        "tp": "true positives",
        "tot": "total",
        "total": "total",
        "ts": "total score",
        "volt": "VOLT",
        "functioning": "functioning",
        "working": "employed",
    }
    tokens = [token for token in remainder.split("_") if token]
    words = []
    for token in tokens:
        if re.fullmatch(r"[ap]\d+", token, flags=re.IGNORECASE):
            words.append(token.upper())
        elif re.fullmatch(r"\d+d\d+", token, flags=re.IGNORECASE):
            words.append(token.upper())
        else:
            words.append(token_labels.get(token.lower(), token.replace("-", " ").capitalize()))
    deduplicated = []
    for word in words:
        if deduplicated and deduplicated[-1].casefold() == word.casefold():
            continue
        deduplicated.append(word)
    words = deduplicated
    detail = " ".join(words).strip()
    if instrument and detail.lower().startswith(instrument.lower()):
        label = detail
    else:
        label = " ".join(part for part in [instrument, detail] if part)
    label = label or feature.replace("_", " ")
    return f"{label[:1].upper()}{label[1:]}"


def _mixed_heatmap_display_name(feature, metadata_row):
    """Prefer a short dictionary label; otherwise derive one from the element name."""
    for column in ["DisplayName", "VariableLabel", "Label"]:
        value = metadata_row.get(column)
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text != str(feature) and len(text) <= 52:
            if text in _SIMPLECLUST_PUBLICATION_LABELS or (
                "_" in text and re.fullmatch(r"[A-Za-z0-9_]+", text)
            ):
                return display_feature_name(text)
            return text
    name = str(feature).split("__", 1)[-1]
    if name in _SIMPLECLUST_PUBLICATION_LABELS:
        return _SIMPLECLUST_PUBLICATION_LABELS[name]
    return _mixed_heatmap_feature_label(feature)


def _mixed_heatmap_label_frame(data, labels, comparison, subject_id_column):
    """Handle mixed heatmap label frame."""
    labels = pd.Series(np.asarray(labels).reshape(-1))
    ids = pd.Series(data[subject_id_column]).reset_index(drop=True)
    size = min(len(ids), len(labels))
    return pd.DataFrame(
        {
            subject_id_column: ids.iloc[:size].to_numpy(),
            comparison: labels.iloc[:size].astype(str).to_numpy(),
        }
    )


def plot_cluster_group_difference_heatmap(
    discovery_data,
    final_metrics,
    meta,
    plots_dir=None,
    subject_id_column="src_subject_id",
    comparisons=None,
    modality_order=None,
    feature_dictionary=None,
    title="Discovery cluster group differences",
    show=True,
):
    """
    Plot the mixed discovery heatmap used to compare modality and final clusters.

    Continuous cells show per-cluster means z-scored within each variable across
    plotted cluster columns. Binary cells show proportions. Categorical cells
    stay neutral while retaining the adjusted significance annotation.
    """
    if subject_id_column not in discovery_data:
        raise KeyError(f"discovery_data is missing subject column '{subject_id_column}'.")
    if not isinstance(final_metrics.get("data"), dict) or not final_metrics["data"]:
        raise ValueError("final_metrics['data'] is required to align cluster labels.")
    if not {"ElementName", "Modality"}.issubset(meta.columns):
        raise KeyError("meta must contain 'ElementName' and 'Modality' columns.")

    cluster_data = final_metrics["data"]
    discovered_modalities = list(cluster_data)
    if comparisons is None:
        comparisons = discovered_modalities + ["final"]
    comparisons = [name for name in comparisons if name == "final" or name in discovered_modalities]
    if not comparisons:
        raise ValueError("No modality or final cluster comparisons are available.")

    label_frames = {}
    labels_by_modality = final_metrics.get("individual_labels", [])
    for modality_index, modality in enumerate(discovered_modalities):
        if modality not in comparisons or modality_index >= len(labels_by_modality):
            continue
        modality_data = cluster_data[modality]
        if subject_id_column not in modality_data:
            continue
        label_frames[modality] = _mixed_heatmap_label_frame(
            modality_data,
            labels_by_modality[modality_index],
            modality,
            subject_id_column,
        )

    if "final" in comparisons:
        final_labels_by_id = final_metrics.get("final_labels_by_subject_id")
        if final_labels_by_id is not None:
            final_labels_by_id = pd.Series(final_labels_by_id)
            label_frames["final"] = pd.DataFrame(
                {
                    subject_id_column: final_labels_by_id.index.to_list(),
                    "final": final_labels_by_id.astype(str).to_list(),
                }
            )
        elif "final_labels" in final_metrics:
            first_modality = discovered_modalities[0]
            label_frames["final"] = _mixed_heatmap_label_frame(
                cluster_data[first_modality],
                final_metrics["final_labels"],
                "final",
                subject_id_column,
            )

    comparisons = [name for name in comparisons if name in label_frames]
    if not comparisons:
        raise ValueError("No aligned cluster labels were found for the requested comparisons.")

    feature_metadata = meta.copy()
    if feature_dictionary is not None:
        if "ElementName" not in feature_dictionary:
            raise KeyError("feature_dictionary must contain an 'ElementName' column.")
        label_columns = [
            column
            for column in [
                "ElementName",
                "DisplayName",
                "VariableLabel",
                "Label",
            ]
            if column in feature_dictionary
        ]
        dictionary_labels = feature_dictionary[label_columns].drop_duplicates(
            subset=["ElementName"],
            keep="first",
        )
        feature_metadata = feature_metadata.drop(
            columns=[column for column in dictionary_labels if column != "ElementName"],
            errors="ignore",
        ).merge(dictionary_labels, on="ElementName", how="left")

    meta_features = (
        feature_metadata
        .dropna(subset=["ElementName"])
        .drop_duplicates(subset=["ElementName"], keep="first")
    )
    feature_rows = []
    feature_frames = {}
    for row in meta_features.itertuples(index=False):
        feature = row.ElementName
        if feature not in discovery_data.columns:
            continue
        var_type = _mixed_heatmap_variable_type(discovery_data[feature])
        if var_type is None:
            continue
        feature_rows.append(
            {
                "feature": feature,
                "display_name": _mixed_heatmap_display_name(feature, row._asdict()),
                "modality": str(row.Modality) if pd.notna(row.Modality) else "Other / Unmapped",
                "var_type": var_type,
            }
        )
        feature_frames[feature] = discovery_data[[subject_id_column, feature]].drop_duplicates(
            subset=[subject_id_column],
            keep="first",
        )

    features = pd.DataFrame(feature_rows)
    if features.empty:
        raise ValueError("No plottable discovery variables from meta were found in discovery_data.")

    if modality_order is None:
        modality_order = list(dict.fromkeys(features["modality"].tolist()))
    else:
        observed = list(dict.fromkeys(features["modality"].tolist()))
        modality_order = list(modality_order) + [name for name in observed if name not in modality_order]
    features["modality"] = pd.Categorical(features["modality"], categories=modality_order, ordered=True)
    features = features.sort_values(["modality"], kind="stable").reset_index(drop=True)

    columns = []
    matrices = {
        "continuous": pd.DataFrame(index=features.loc[features["var_type"].eq("continuous"), "feature"]),
        "binary": pd.DataFrame(index=features.loc[features["var_type"].eq("binary"), "feature"]),
        "categorical": pd.DataFrame(index=features.loc[features["var_type"].eq("categorical"), "feature"]),
    }
    annotations = {key: {} for key in matrices}
    p_records = []

    for comparison in comparisons:
        labels = label_frames[comparison]
        group_order = sorted(labels[comparison].dropna().astype(str).unique(), key=cluster_sort_key)
        for group in group_order:
            column = (comparison, group)
            columns.append(column)
            for matrix in matrices.values():
                matrix[column] = np.nan
        for feature_row in features.itertuples(index=False):
            feature = feature_row.feature
            var_type = feature_row.var_type
            joined = feature_frames[feature].merge(labels, on=subject_id_column, how="inner")
            raw_values = joined[feature]
            value_source = _mixed_heatmap_binary_values(raw_values) if var_type == "binary" else raw_values
            p_records.append(
                {
                    "comparison": comparison,
                    "feature": feature,
                    "var_type": var_type,
                    "test": "Kruskal-Wallis" if var_type == "continuous" else "Pearson chi-square",
                    "p_value": _mixed_heatmap_p_value(raw_values, joined[comparison], var_type),
                }
            )
            for group in group_order:
                in_group = joined[comparison].astype(str).eq(group)
                value = np.nan
                if var_type == "continuous":
                    value = pd.to_numeric(raw_values[in_group], errors="coerce").mean()
                elif var_type == "binary":
                    value = value_source[in_group].mean()
                matrices[var_type].at[feature, (comparison, group)] = value

    p_table = pd.DataFrame(p_records)
    p_table["q_value"] = np.nan
    for comparison, positions in p_table.groupby("comparison").groups.items():
        p_table.loc[positions, "q_value"] = _benjamini_hochberg(
            p_table.loc[positions, "p_value"].to_numpy()
        )

    for var_type, matrix in matrices.items():
        matrix.columns = pd.MultiIndex.from_tuples(matrix.columns, names=["comparison", "group"])
        if var_type == "continuous":
            means = matrix.mean(axis=1, skipna=True)
            sds = matrix.std(axis=1, skipna=True).replace(0, np.nan)
            matrices[var_type] = matrix.sub(means, axis=0).div(sds, axis=0).fillna(0)
        for feature in matrix.index:
            for comparison in comparisons:
                matching = p_table[
                    p_table["comparison"].eq(comparison) & p_table["feature"].eq(feature)
                ]
                if matching.empty:
                    continue
                q_value = float(matching.iloc[0]["q_value"])
                if np.isfinite(q_value):
                    q_text = _mixed_heatmap_p_text(q_value)
                    stars = _mixed_heatmap_sig_text(q_value)
                    annotations[var_type][(feature, (comparison, "FDR q"))] = {
                        "text": f"{'q' if q_text.startswith('<') else 'q='}{q_text} {stars}".strip(),
                        "significant": q_value < 0.05,
                    }

    plot_columns = []
    for comparison in comparisons:
        plot_columns.extend(column for column in columns if column[0] == comparison)
        plot_columns.append((comparison, "FDR q"))
    column_widths = np.asarray(
        [1.8 if column[1] == "FDR q" else 0.6 for column in plot_columns],
        dtype=float,
    )
    x_edges = np.concatenate(([0.0], np.cumsum(column_widths)))
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2

    def panel_rows(matrix):
        """Add a masked title row above each modality block for plotting."""
        lookup = features.set_index("feature")
        values = []
        tick_labels = []
        row_features = []
        row_heights = []
        last_modality = None
        for feature, row in matrix.iterrows():
            modality = str(lookup.loc[feature, "modality"])
            if modality != last_modality:
                header = modality.replace("_", " ")
                bold_header = header.replace(" ", r"\ ")
                values.append(np.full(len(matrix.columns), np.nan))
                tick_labels.append(rf"$\bf{{{bold_header}}}$")
                row_features.append(None)
                row_heights.append(3)
            values.append(row.to_numpy(dtype=float))
            tick_labels.append(f"  {lookup.loc[feature, 'display_name']}")
            row_features.append(feature)
            row_heights.append(2)
            last_modality = modality
        row_heights = np.asarray(row_heights, dtype=float)
        y_edges = np.concatenate(([0.0], np.cumsum(row_heights)))
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        return {
            "values": np.asarray(values, dtype=float),
            "tick_labels": tick_labels,
            "row_features": row_features,
            "y_edges": y_edges,
            "y_centers": y_centers,
            "height": float(row_heights.sum()),
        }

    present_types = [key for key, matrix in matrices.items() if not matrix.empty]
    if not present_types:
        raise ValueError("The aligned discovery variables produced no heatmap panels.")
    panel_layouts = {
        var_type: panel_rows(matrices[var_type].reindex(columns=plot_columns))
        for var_type in present_types
    }
    height_ratios = [max(2.0, panel_layouts[key]["height"]) for key in present_types]
    fig_height = max(25, 0.25 * sum(height_ratios) + 2.9)
    fig_width = max(25, 0.34 * float(column_widths.sum()) + 4.7)
    fig, axes = plt.subplots(
        len(present_types),
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.18},
    )
    axes = np.atleast_1d(axes)
    panel_titles = {
        "continuous": "Continuous (means)",
        "binary": "Binary (proportions)",
        "categorical": "Categorical",
    }

    for axis_index, (ax, var_type) in enumerate(zip(axes, present_types)):
        matrix = matrices[var_type].reindex(columns=plot_columns)
        layout = panel_layouts[var_type]
        plot_values = layout["values"].copy()
        y_edges = layout["y_edges"]
        if var_type == "continuous":
            cmap = plt.get_cmap("Blues").copy()
            cmap.set_bad("#ffffff")
            image = ax.pcolormesh(
                x_edges,
                y_edges,
                np.ma.masked_invalid(plot_values),
                cmap=cmap,
                shading="flat",
                edgecolors="white",
                linewidth=0.0045,
            )
            fig.colorbar(image, ax=ax, fraction=0.018, pad=0.012, label="Mean")
        elif var_type == "binary":
            cmap = plt.get_cmap("YlOrRd").copy()
            cmap.set_bad("#ffffff")
            image = ax.pcolormesh(
                x_edges,
                y_edges,
                np.ma.masked_invalid(plot_values),
                cmap=cmap,
                shading="flat",
                edgecolors="white",
                linewidth=0.0045,
                vmin=0,
                vmax=1,
            )
            fig.colorbar(image, ax=ax, fraction=0.018, pad=0.012, label="Prop")
        else:
            for col_index, column in enumerate(plot_columns):
                if column[1] != "FDR q":
                    for row_index, feature in enumerate(layout["row_features"]):
                        if feature is not None:
                            plot_values[row_index, col_index] = 0
            cmap = plt.get_cmap("Greys").copy()
            cmap.set_bad("#ffffff")
            ax.pcolormesh(
                x_edges,
                y_edges,
                np.ma.masked_invalid(plot_values),
                cmap=cmap,
                shading="flat",
                edgecolors="white",
                linewidth=0.0045,
                vmin=-1,
                vmax=1,
            )

        ax.set_title(panel_titles[var_type], loc="left", fontsize=22, pad=1)
        ax.set_aspect("auto")
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[-1], y_edges[0])
        ax.set_yticks(layout["y_centers"])
        ax.set_yticklabels(layout["tick_labels"], fontsize=18)
        ax.tick_params(axis="both", length=0)
        ax.set_xticks(x_centers)
        for comp_index, comparison in enumerate(comparisons[:-1]):
            last_column = max(i for i, column in enumerate(plot_columns) if column[0] == comparison)
            ax.axvline(x_edges[last_column + 1], color="white", linewidth=4)
        for row_index, feature in enumerate(layout["row_features"]):
            if feature is None:
                continue
            for col_index, column in enumerate(plot_columns):
                text = annotations[var_type].get((feature, column), "")
                annotation = annotations[var_type].get((feature, column), None)

                if annotation:
                    text = annotation["text"]
                    is_significant = annotation["significant"]

                    ax.text(
                        x_centers[col_index],
                        layout["y_centers"][row_index],
                        text,
                        ha="center",
                        va="center",
                        color="#111111",
                        fontsize=16,
                        fontweight="bold" if is_significant else "normal",
                        bbox={
                            "boxstyle": "round,pad=0.10",
                            "facecolor": "#ffffff" if is_significant else "white",
                            "edgecolor": "none",
                            "alpha": 0.98 if is_significant else 0.94,
                        },
                    )
        if axis_index < len(axes) - 1:
            ax.tick_params(axis="x", labelbottom=False)

    axes[-1].set_xticklabels([f"{comp} | {group}" for comp, group in plot_columns], rotation=45, ha="right", fontsize=18)
    axes[-1].set_xlabel("Modality | group", fontsize=22)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    output_paths = {}
    if plots_dir is not None:
        output_dir = os.path.join(plots_dir, "cluster_group_differences")
        os.makedirs(output_dir, exist_ok=True)
        for output_path in _save_matplotlib_png_pdf(
            fig,
            os.path.join(output_dir, "cluster_group_difference_heatmap"),
            dpi=300,
        ):
            output_paths[output_path.suffix.lstrip(".")] = str(output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)
    return {
        "figure": fig,
        "matrices": matrices,
        "adjusted_p_values": p_table,
        "paths": output_paths,
    }


def safe_axis_range(values, pad_fraction=0.04):
    """Finite y-range helper to avoid matplotlib/plotly [nan, nan] ranges."""
    vals = pd.to_numeric(pd.Series(np.asarray(values).reshape(-1)), errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return None
    vmin = float(vals.min())
    vmax = float(vals.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if vmin == vmax:
        pad = 0.5 if vmin == 0 else abs(vmin) * 0.08
    else:
        pad = (vmax - vmin) * float(pad_fraction)
    return vmin - pad, vmax + pad


def robust_axis_range(values, quantiles=(0.01, 0.99), pad_fraction=0.06, min_n=30):
    """
    Return a readable display range without changing the underlying values.

    For sufficiently large samples, limits are based on central quantiles so
    sparse projection outliers do not collapse violin detail. The returned
    metadata records how many finite values fall below/above the visible range.
    """
    vals = pd.to_numeric(pd.Series(np.asarray(values).reshape(-1)), errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return None, {"below": 0, "above": 0, "mode": "empty"}
    if len(vals) < int(min_n) or quantiles is None:
        return safe_axis_range(vals, pad_fraction=pad_fraction), {"below": 0, "above": 0, "mode": "full"}
    q_low, q_high = quantiles
    lower = float(vals.quantile(q_low))
    upper = float(vals.quantile(q_high))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return safe_axis_range(vals, pad_fraction=pad_fraction), {"below": 0, "above": 0, "mode": "full"}
    span = upper - lower
    y_range = (lower - span * float(pad_fraction), upper + span * float(pad_fraction))
    return y_range, {
        "below": int((vals < y_range[0]).sum()),
        "above": int((vals > y_range[1]).sum()),
        "mode": "quantile",
        "quantiles": tuple(quantiles),
    }


def _annotate_clipped_component_counts(ax, x_value_pairs, y_range, fontsize=9):
    """Annotate how many scores are hidden below/above a robust display window."""
    if y_range is None:
        return 0, 0
    y_low, y_high = y_range
    span = max(float(y_high - y_low), 1e-9)
    text_low = y_low + 0.018 * span
    text_high = y_high - 0.018 * span
    total_below = 0
    total_above = 0
    for x, values in x_value_pairs:
        vals = pd.to_numeric(pd.Series(np.asarray(values).reshape(-1)), errors="coerce")
        vals = vals[np.isfinite(vals)]
        n_below = int((vals < y_low).sum())
        n_above = int((vals > y_high).sum())
        total_below += n_below
        total_above += n_above
        if n_above:
            ax.text(x, text_high, f"^{n_above}", ha="center", va="top", fontsize=fontsize, color="#111111", zorder=6)
        if n_below:
            ax.text(x, text_low, f"v{n_below}", ha="center", va="bottom", fontsize=fontsize, color="#111111", zorder=6)
    return total_below, total_above


def _get_mixed_categorical_modalities(preprocessing_details=None):
    """Read mixed-type modality names from a preprocessing-details dict."""
    params = {}
    if isinstance(preprocessing_details, dict):
        params = preprocessing_details.get("preprocessing_parameters", {}) or {}
    return set(params.get("mixed_categorical_modalities", []) or [])


def _is_integer_like(series):
    """Handle is integer like."""
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return False
    return np.all(np.isclose(vals, np.round(vals)))


def _famd_column_split(df, max_numeric_category_levels=10):
    """Heuristic split for mixed domains: object/bool/category and low-cardinality integers are categorical."""
    numeric_cols = []
    categorical_cols = []
    n_rows = max(len(df), 1)
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s):
            categorical_cols.append(col)
            continue
        s_num = pd.to_numeric(s, errors="coerce")
        n_unique = int(s_num.dropna().nunique())
        if n_unique <= max_numeric_category_levels and _is_integer_like(s_num):
            categorical_cols.append(col)
        elif n_unique <= max(2, int(0.02 * n_rows)) and _is_integer_like(s_num):
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)
    return numeric_cols, categorical_cols


def _onehot_encoder_dense():
    """Handle onehot encoder dense."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _low_cardinality_numeric_columns_for_dimred(df, max_unique=10):
    """Handle low cardinality numeric columns for dimred."""
    cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        n_unique = df[col].dropna().nunique()
        if 0 < n_unique <= max_unique:
            cols.append(col)
    return cols


def _normalize_dim_reduction_name(value):
    """Normalize dim reduction name."""
    if value is None:
        return "none"
    text = str(value).strip().lower()
    if text in ("", "none"):
        return "none"
    if text in ("mixed_svd", "mixed-svd"):
        return "famd"
    if text in ("sparse_nmf", "sparse-nmf", "snmf"):
        return "sparsenmf"
    return text


def fit_mixed_type_svd_reducer(df, subject_id_column="src_subject_id", method="famd", random_state=42):
    """
    Fit the mixed-type SVD reducer used by the clustering pipeline.

    The fitted dictionary is intentionally sklearn-only so it can be stored in
    final_metrics and reused for independent validation/test samples.
    """
    method = _normalize_dim_reduction_name(method)
    xdf = df.drop(columns=[subject_id_column], errors="ignore").copy()
    # Decide which columns should be treated as categorical. FAMD keeps numeric
    # columns numeric but treats object/bool/low-cardinality numeric columns as
    # categorical; MCA treats every column as categorical.
    categorical_cols = xdf.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if method == "mca":
        categorical_cols = xdf.columns.tolist()
    else:
        categorical_cols = list(dict.fromkeys(
            categorical_cols + _low_cardinality_numeric_columns_for_dimred(xdf)
        ))
    numeric_cols = [col for col in xdf.columns if col not in categorical_cols]

    model = {
        "method": method,
        "feature_columns": list(xdf.columns),
        "numeric_cols": list(numeric_cols),
        "categorical_cols": list(categorical_cols),
        "random_state": random_state,
        "numeric_imputer": None,
        "numeric_scaler": None,
        "categorical_imputer": None,
        "categorical_encoder": None,
        "svd": None,
    }

    # Build one numeric block and one encoded categorical block. The fitted
    # imputers/scalers/encoder are stored so new data can be projected later.
    matrices = []
    if numeric_cols:
        numeric = xdf[numeric_cols].apply(pd.to_numeric, errors="coerce")
        num_imputer = SimpleImputer(strategy="median")
        num_scaler = StandardScaler()
        numeric_matrix = num_scaler.fit_transform(num_imputer.fit_transform(numeric))
        matrices.append(np.asarray(numeric_matrix, dtype=np.float32))
        model["numeric_imputer"] = num_imputer
        model["numeric_scaler"] = num_scaler

    if categorical_cols:
        categorical = xdf[categorical_cols].astype("object")
        cat_imputer = SimpleImputer(strategy="most_frequent")
        cat_encoder = _onehot_encoder_dense()
        categorical_matrix = cat_encoder.fit_transform(cat_imputer.fit_transform(categorical))
        matrices.append(np.asarray(categorical_matrix, dtype=np.float32))
        model["categorical_imputer"] = cat_imputer
        model["categorical_encoder"] = cat_encoder

    if not matrices:
        model["raw_output"] = True
        return model

    # TruncatedSVD gives a compact reusable representation without requiring an
    # optional FAMD/MCA package.
    X = np.hstack(matrices) if len(matrices) > 1 else matrices[0]
    max_components = min(50, X.shape[0] - 1, X.shape[1] - 1)
    if max_components < 1:
        model["raw_output"] = True
        return model

    svd = TruncatedSVD(n_components=max_components, random_state=random_state)
    svd.fit(X)
    model["svd"] = svd
    model["n_components"] = int(max_components)
    return model


def transform_mixed_type_svd_reducer(df, reducer, subject_id_column="src_subject_id"):
    """Project a new modality table through a fitted mixed-type SVD reducer."""
    xdf = df.drop(columns=[subject_id_column], errors="ignore").copy()
    feature_columns = list(reducer.get("feature_columns", xdf.columns))
    # Reindex to the discovery feature schema so missing/new columns do not shift
    # the encoded matrix layout.
    xdf = xdf.reindex(columns=feature_columns)

    matrices = []
    numeric_cols = list(reducer.get("numeric_cols", []))
    if numeric_cols:
        numeric = xdf[numeric_cols].apply(pd.to_numeric, errors="coerce")
        num_imputer = reducer.get("numeric_imputer")
        num_scaler = reducer.get("numeric_scaler")
        matrices.append(np.asarray(num_scaler.transform(num_imputer.transform(numeric)), dtype=np.float32))

    categorical_cols = list(reducer.get("categorical_cols", []))
    if categorical_cols:
        categorical = xdf[categorical_cols].astype("object")
        cat_imputer = reducer.get("categorical_imputer")
        cat_encoder = reducer.get("categorical_encoder")
        matrices.append(np.asarray(cat_encoder.transform(cat_imputer.transform(categorical)), dtype=np.float32))

    if not matrices:
        return np.empty((len(df), 0), dtype=np.float32)

    X = np.hstack(matrices) if len(matrices) > 1 else matrices[0]
    svd = reducer.get("svd")
    if svd is None:
        return X.astype(np.float32, copy=False)
    return svd.transform(X).astype(np.float32, copy=False)


def _nonnegative_matrix_for_nmf(X, shift=None):
    """Handle nonnegative matrix for nmf."""
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if X.shape[0] == 0 or X.shape[1] == 0:
        shift = np.zeros((X.shape[1],), dtype=np.float32) if shift is None else np.asarray(shift, dtype=np.float32)
        return X.astype(np.float32, copy=False), shift
    if shift is None:
        mins = np.nanmin(X, axis=0)
        shift = np.where(mins < 0, -mins, 0.0).astype(np.float32)
    X_nonnegative = X + np.asarray(shift, dtype=np.float32)
    return np.maximum(X_nonnegative, 0.0).astype(np.float32, copy=False), shift


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


def fit_sparse_nmf_reducer(
    df,
    subject_id_column="src_subject_id",
    n_components=20,
    alpha=0.1,
    l1_ratio=1.0,
    max_iter=1000,
    random_state=42,
):
    """Fit the sparse NMF reducer used by the clustering pipeline."""
    xdf = df.drop(columns=[subject_id_column], errors="ignore").copy()
    X = xdf.to_numpy(dtype=np.float32, copy=True)
    X_nonnegative, shift = _nonnegative_matrix_for_nmf(X)
    max_components = min(X_nonnegative.shape[0], X_nonnegative.shape[1])

    model = {
        "method": "sparsenmf",
        "feature_columns": list(xdf.columns),
        "shift": shift,
        "nmf": None,
        "n_components": 0,
        "alpha": float(alpha),
        "l1_ratio": float(l1_ratio),
        "max_iter": int(max_iter),
        "random_state": random_state,
    }
    if max_components < 1:
        return model

    n_components = min(int(n_components), max_components)
    nmf = _make_sparse_nmf(
        n_components=n_components,
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        random_state=random_state,
    )
    nmf.fit(X_nonnegative)
    model["nmf"] = nmf
    model["n_components"] = int(n_components)
    return model


def transform_sparse_nmf_reducer(df, reducer, subject_id_column="src_subject_id"):
    """Project a new modality table through a fitted sparse NMF reducer."""
    xdf = df.drop(columns=[subject_id_column], errors="ignore").copy()
    feature_columns = list(reducer.get("feature_columns", xdf.columns))
    xdf = xdf.reindex(columns=feature_columns)
    X = xdf.to_numpy(dtype=np.float32, copy=True)
    X_nonnegative, _ = _nonnegative_matrix_for_nmf(X, shift=reducer.get("shift"))
    nmf = reducer.get("nmf")
    if nmf is None:
        return X_nonnegative.astype(np.float32, copy=False)
    return nmf.transform(X_nonnegative).astype(np.float32, copy=False)


def _dim_reduction_context_from_metrics(final_metrics):
    """Handle dim reduction context from metrics."""
    reporting = final_metrics.get("final_reporting", {}) if isinstance(final_metrics, dict) else {}
    compute_context = reporting.get("compute_context", {}) if isinstance(reporting, dict) else {}
    context = dict(compute_context or {})
    if isinstance(final_metrics, dict):
        for key in [
            "dim_reduction",
            "dim_reduction_label",
            "dim_reduction_n_components",
            "spca_alpha",
            "spca_ridge_alpha",
            "spca_max_iter",
            "pca_variance_threshold",
            "snmf_n_components",
            "snmf_alpha",
            "snmf_l1_ratio",
            "snmf_max_iter",
        ]:
            if key not in context and key in final_metrics:
                context[key] = final_metrics[key]
    return context


def _infer_dim_reduction_by_modality(final_metrics, modalities):
    """Infer dim reduction by modality."""
    context = _dim_reduction_context_from_metrics(final_metrics)
    methods = context.get("dim_reduction_by_modality", {}) or {}
    default_method = _normalize_dim_reduction_name(context.get("dim_reduction", "none"))
    inferred = {mod: _normalize_dim_reduction_name(methods.get(mod, default_method)) for mod in modalities}

    ae_res = final_metrics.get("ae_res", {}) if isinstance(final_metrics, dict) else {}
    feature_names = final_metrics.get("svm_feature_names_modalities", None) if isinstance(final_metrics, dict) else None
    for i, mod in enumerate(modalities):
        if inferred.get(mod) != "none":
            continue
        if isinstance(ae_res, dict) and ae_res.get("spca_model") is not None:
            inferred[mod] = "sparsepca"
            continue
        if isinstance(ae_res, dict) and ae_res.get("pca_model") is not None:
            inferred[mod] = "pca"
            continue
        if isinstance(ae_res, dict) and isinstance(ae_res.get(mod), dict) and ae_res[mod].get("pca_model") is not None:
            inferred[mod] = "pca"
            continue
        if isinstance(ae_res, dict) and isinstance(ae_res.get(mod), dict) and ae_res[mod].get("spca_model") is not None:
            inferred[mod] = "sparsepca"
            continue
        if feature_names is not None and i < len(feature_names) and feature_names[i]:
            names = list(feature_names[i])
            if names and all(str(name).startswith(f"{mod}__latent_") for name in names):
                inferred[mod] = "famd"
    return inferred


def apply_dimensionality_reduction_to_new_data(
    dict_final_new,
    final_metrics,
    modalities=None,
    subject_id_column="src_subject_id",
    modality_dim_reduction=None,
    random_state=42,
    fit_reducers_on_new_data=False,
):
    """
    Apply the dimensionality-reduction steps used for SVM/clustering to an
    independently preprocessed sample.

    By default this projects through fitted discovery reducers when available.
    Set fit_reducers_on_new_data=True to refit the same reducer type on the new
    sample itself, using pipeline settings such as PCA variance threshold but
    not reusing fitted discovery PCA/FAMD/MCA parameters.

    Returns (ae_res_new, X_new, X_new_by_modality), where X_new is the integrated
    SVM feature matrix and X_new_by_modality contains one SVM feature DataFrame
    per modality.
    """
    if modalities is None:
        modalities = list(dict_final_new.keys())
    modalities = list(modalities)
    discovery_data = final_metrics.get("data", {})
    ae_res_discovery = final_metrics.get("ae_res", {})

    if modality_dim_reduction is None:
        modality_dim_reduction = _infer_dim_reduction_by_modality(final_metrics, modalities)
    else:
        modality_dim_reduction = {
            mod: _normalize_dim_reduction_name(modality_dim_reduction.get(mod, "none"))
            for mod in modalities
        }

    reducers = final_metrics.get("dim_reduction_models", {}) or {}
    context = _dim_reduction_context_from_metrics(final_metrics)
    pca_variance_threshold = context.get("pca_variance_threshold", None)
    snmf_n_components = int(context.get("snmf_n_components", 20))
    snmf_alpha = float(context.get("snmf_alpha", 0.1))
    snmf_l1_ratio = float(context.get("snmf_l1_ratio", 1.0))
    snmf_max_iter = int(context.get("snmf_max_iter", 1000))
    ae_res_new = {}
    X_by_modality = {}

    # Project each modality through the method used in the discovery run, then
    # concatenate modality-level feature matrices for SVM/reporting use.
    for mod in modalities:
        if mod not in dict_final_new:
            raise KeyError(f"Modality '{mod}' not found in new preprocessed data.")

        method = _normalize_dim_reduction_name(modality_dim_reduction.get(mod, "none"))
        df_new = dict_final_new[mod]
        df_disc = discovery_data.get(mod)

        if method == "none":
            # No reducer was used in discovery, so keep processed features.
            X_mod = df_new.drop(columns=[subject_id_column], errors="ignore").reset_index(drop=True)
            ae_res_new[mod] = {"final_latent": X_mod.to_numpy(dtype=np.float32, copy=True)}
            X_by_modality[mod] = X_mod
            continue

        if method == "pca":
            # Prefer the fitted discovery PCA model. If unavailable, refit only
            # when the caller explicitly allows or discovery data is available.
            pca = None
            feature_cols = (
                list(df_disc.drop(columns=[subject_id_column], errors="ignore").columns)
                if df_disc is not None
                else list(df_new.drop(columns=[subject_id_column], errors="ignore").columns)
            )
            X_input = df_new.drop(columns=[subject_id_column], errors="ignore").reindex(columns=feature_cols)
            if fit_reducers_on_new_data:
                X_fit = X_input.to_numpy(dtype=np.float32, copy=True)
                max_components = min(X_fit.shape[1], X_fit.shape[0] - 1)
                if max_components < 1:
                    latent = X_fit.astype(np.float32, copy=False)
                else:
                    if pca_variance_threshold is not None:
                        variance_threshold = float(pca_variance_threshold)
                        n_components = variance_threshold if variance_threshold < 1.0 else max_components
                    else:
                        n_components = min(
                            ae_res_discovery.get(mod, {}).get("pca_n_components", 50)
                            if isinstance(ae_res_discovery.get(mod), dict)
                            else 50,
                            max_components,
                        )
                    pca = PCA(n_components=n_components, random_state=random_state)
                    latent = pca.fit_transform(X_fit).astype(np.float32, copy=False)
            else:
                if isinstance(ae_res_discovery, dict) and isinstance(ae_res_discovery.get(mod), dict):
                    pca = ae_res_discovery[mod].get("pca_model")
                if pca is None and isinstance(ae_res_discovery, dict):
                    pca = ae_res_discovery.get("pca_model")
                if pca is None and isinstance(reducers, dict):
                    pca = reducers.get(mod)
                if pca is None:
                    if df_disc is None:
                        raise KeyError(f"Discovery data for modality '{mod}' is required to refit PCA.")
                    X_disc = df_disc.drop(columns=[subject_id_column], errors="ignore").to_numpy(dtype=np.float32, copy=True)
                    n_components = min(
                        ae_res_discovery.get(mod, {}).get("pca_n_components", 50) if isinstance(ae_res_discovery.get(mod), dict) else 50,
                        X_disc.shape[1],
                        X_disc.shape[0] - 1,
                    )
                    pca = PCA(n_components=n_components, random_state=random_state)
                    pca.fit(X_disc)
                latent = pca.transform(X_input.to_numpy(dtype=np.float32, copy=True)).astype(np.float32, copy=False)
            if pca is not None:
                ae_res_new[mod] = {
                    "final_latent": np.asarray(latent, dtype=np.float32),
                    "pca_model": pca,
                    "pca_n_components": int(getattr(pca, "n_components_", latent.shape[1])),
                    "pca_explained_variance": float(np.sum(getattr(pca, "explained_variance_ratio_", []))),
                }
            else:
                ae_res_new[mod] = {"final_latent": np.asarray(latent, dtype=np.float32)}
            X_by_modality[mod] = pd.DataFrame(
                np.asarray(latent, dtype=np.float32),
                columns=[f"{mod}__latent_{i + 1}" for i in range(np.asarray(latent).shape[1])],
            )
            continue
        elif method == "sparsepca":
            reducer = None
            feature_cols = (
                list(df_disc.drop(columns=[subject_id_column], errors="ignore").columns)
                if isinstance(df_disc, pd.DataFrame)
                else list(df_new.drop(columns=[subject_id_column], errors="ignore").columns)
            )
            X_input = df_new.drop(columns=[subject_id_column], errors="ignore").reindex(columns=feature_cols)
            if fit_reducers_on_new_data:
                X_fit = X_input.to_numpy(dtype=np.float32, copy=True)
                n_components = min(
                    int(context.get("dim_reduction_n_components", context.get("maxPC", 2)) or 2),
                    X_fit.shape[1],
                    max(1, X_fit.shape[0] - 1),
                )
                reducer = SparsePCA(
                    n_components=n_components,
                    alpha=float(context.get("spca_alpha", 1.0)),
                    ridge_alpha=float(context.get("spca_ridge_alpha", 0.01)),
                    max_iter=int(context.get("spca_max_iter", 1000)),
                    random_state=random_state,
                    n_jobs=1,
                )
                latent = reducer.fit_transform(X_fit).astype(np.float32, copy=False)
            else:
                if isinstance(ae_res_discovery, dict) and isinstance(ae_res_discovery.get(mod), dict):
                    reducer = ae_res_discovery[mod].get("spca_model") or ae_res_discovery[mod].get("dim_reduction_model")
                if reducer is None and isinstance(ae_res_discovery, dict):
                    reducer = ae_res_discovery.get("spca_model")
                if reducer is None and isinstance(reducers, dict):
                    reducer = reducers.get(mod)
                if reducer is None:
                    if not isinstance(df_disc, pd.DataFrame):
                        raise KeyError(f"Discovery data for modality '{mod}' is required to refit SparsePCA.")
                    X_disc = (
                        df_disc
                        .drop(columns=[subject_id_column], errors="ignore")
                        .reindex(columns=feature_cols)
                        .to_numpy(dtype=np.float32, copy=True)
                    )
                    n_components = min(
                        int(context.get("dim_reduction_n_components", context.get("maxPC", 2)) or 2),
                        X_disc.shape[1],
                        max(1, X_disc.shape[0] - 1),
                    )
                    reducer = SparsePCA(
                        n_components=n_components,
                        alpha=float(context.get("spca_alpha", 1.0)),
                        ridge_alpha=float(context.get("spca_ridge_alpha", 0.01)),
                        max_iter=int(context.get("spca_max_iter", 1000)),
                        random_state=random_state,
                        n_jobs=1,
                    )
                    reducer.fit(X_disc)
                latent = reducer.transform(X_input.to_numpy(dtype=np.float32, copy=True)).astype(np.float32, copy=False)
            ae_res_new[mod] = {
                "final_latent": np.asarray(latent, dtype=np.float32),
                "spca_model": reducer,
                "dim_reduction_model": reducer,
            }
            X_by_modality[mod] = pd.DataFrame(
                np.asarray(latent, dtype=np.float32),
                columns=[f"{mod}__latent_{i + 1}" for i in range(np.asarray(latent).shape[1])],
            )
            continue
        elif method == "sparsenmf":
            reducer = None
            if fit_reducers_on_new_data:
                reducer = fit_sparse_nmf_reducer(
                    df_new,
                    subject_id_column=subject_id_column,
                    n_components=snmf_n_components,
                    alpha=snmf_alpha,
                    l1_ratio=snmf_l1_ratio,
                    max_iter=snmf_max_iter,
                    random_state=random_state,
                )
            else:
                if isinstance(ae_res_discovery, dict) and isinstance(ae_res_discovery.get(mod), dict):
                    reducer = (
                        ae_res_discovery[mod].get("dim_reduction_model")
                        or ae_res_discovery[mod].get("snmf_model")
                    )
                if reducer is None and isinstance(ae_res_discovery, dict):
                    reducer = ae_res_discovery.get("dim_reduction_model") or ae_res_discovery.get("snmf_model")
                if reducer is None and isinstance(reducers, dict):
                    reducer = reducers.get(mod)
                if reducer is None:
                    if df_disc is None:
                        raise KeyError(f"Discovery data for modality '{mod}' is required to refit SparseNMF.")
                    reducer = fit_sparse_nmf_reducer(
                        df_disc,
                        subject_id_column=subject_id_column,
                        n_components=snmf_n_components,
                        alpha=snmf_alpha,
                        l1_ratio=snmf_l1_ratio,
                        max_iter=snmf_max_iter,
                        random_state=random_state,
                    )
            latent = transform_sparse_nmf_reducer(
                df_new,
                reducer,
                subject_id_column=subject_id_column,
            )
        elif method in ("famd", "mca"):
            # Reuse the fitted mixed-type reducer when possible so categories and
            # SVD axes match the discovery representation.
            reducer = None
            if fit_reducers_on_new_data:
                reducer = fit_mixed_type_svd_reducer(
                    df_new,
                    subject_id_column=subject_id_column,
                    method=method,
                    random_state=random_state,
                )
            else:
                if isinstance(ae_res_discovery, dict) and isinstance(ae_res_discovery.get(mod), dict):
                    reducer = ae_res_discovery[mod].get("dim_reduction_model")
                if reducer is None and isinstance(reducers, dict):
                    reducer = reducers.get(mod)
                if reducer is None:
                    if df_disc is None:
                        raise KeyError(f"Discovery data for modality '{mod}' is required to refit mixed-type reducer.")
                    reducer = fit_mixed_type_svd_reducer(
                        df_disc,
                        subject_id_column=subject_id_column,
                        method=method,
                        random_state=random_state,
                    )
            latent = transform_mixed_type_svd_reducer(
                df_new,
                reducer,
                subject_id_column=subject_id_column,
            )
        else:
            raise NotImplementedError(
                f"Cannot project new data with dim_reduction='{method}' unless a reusable transformer is available."
            )

        latent = np.asarray(latent, dtype=np.float32)
        ae_res_new[mod] = {"final_latent": latent}
        X_by_modality[mod] = pd.DataFrame(
            latent,
            columns=[f"{mod}__latent_{i + 1}" for i in range(latent.shape[1])],
        )

    X_new = pd.concat([X_by_modality[mod].reset_index(drop=True) for mod in modalities], axis=1)
    return ae_res_new, X_new, X_by_modality


def _series_from_svm_result(result, key):
    """Handle series from svm result."""
    value = (result or {}).get(key)
    if value is None:
        return None
    if isinstance(value, pd.Series):
        return value.astype(float)
    if isinstance(value, dict):
        return pd.Series(value, dtype=float)
    return pd.Series(value, dtype=float)


def _svm_latent_feature_parts(feature):
    """Handle svm latent feature parts."""
    text = str(feature)
    match = re.match(r"^(?:(?P<mod>.+)__)?latent_(?P<idx>\d+)$", text)
    if match is None:
        return None, None
    mod = match.group("mod")
    return mod, int(match.group("idx")) - 1


def _metrics_data_for_modality(final_metrics, modality, subject_id_column="src_subject_id"):
    """Handle metrics data for modality."""
    data = final_metrics.get("data", {}) if isinstance(final_metrics, dict) else {}
    if isinstance(data, pd.DataFrame):
        return data.drop(columns=[subject_id_column], errors="ignore").copy()
    if isinstance(data, dict):
        if modality in data and isinstance(data[modality], pd.DataFrame):
            return data[modality].drop(columns=[subject_id_column], errors="ignore").copy()
        if len(data) == 1:
            only = next(iter(data.values()))
            if isinstance(only, pd.DataFrame):
                return only.drop(columns=[subject_id_column], errors="ignore").copy()
    return None


def _ae_payload_for_modality(ae_res, modality):
    """Handle ae payload for modality."""
    if isinstance(ae_res, dict) and "final_latent" in ae_res:
        return ae_res
    if isinstance(ae_res, dict) and modality in ae_res and isinstance(ae_res[modality], dict):
        return ae_res[modality]
    if isinstance(ae_res, dict) and len(ae_res) == 1:
        only = next(iter(ae_res.values()))
        if isinstance(only, dict):
            return only
    return None


def _component_matrix_from_payload(payload):
    """Handle component matrix from payload."""
    if not isinstance(payload, dict):
        return None, None
    for key, method in [
        ("pca_model", "loading_weighted_pca"),
        ("spca_model", "loading_weighted_sparsepca"),
        ("snmf_model", "loading_weighted_sparsenmf"),
        ("dim_reduction_model", "loading_weighted_dimensionality_reducer"),
    ]:
        model = payload.get(key)
        components = getattr(model, "components_", None)
        if components is not None:
            return np.asarray(components, dtype=float), method
    return None, None


def _normalized_abs_rows(matrix):
    """Handle normalized abs rows."""
    weights = np.abs(np.asarray(matrix, dtype=float))
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    denom = weights.sum(axis=1, keepdims=True)
    return np.divide(weights, denom, out=np.zeros_like(weights), where=denom > 0)


def _latent_original_correlation_weights(latent, original):
    """Handle latent original correlation weights."""
    latent = np.asarray(latent, dtype=float)
    original = np.asarray(original, dtype=float)
    if latent.ndim == 1:
        latent = latent.reshape(-1, 1)
    weights = np.zeros((latent.shape[1], original.shape[1]), dtype=float)
    for i in range(latent.shape[1]):
        x = latent[:, i]
        for j in range(original.shape[1]):
            y = original[:, j]
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
                continue
            weights[i, j] = abs(np.corrcoef(x[mask], y[mask])[0, 1])
    return _normalized_abs_rows(weights)


def original_feature_importance_from_svm(
    final_metrics,
    svm_result=None,
    top_n=None,
    subject_id_column="src_subject_id",
    feature_name_prefix=True,
):
    """Map trained SVM feature contributions back to original/preprocessed variables.

    If the SVM was trained on original/preprocessed variables, the returned table
    is the direct SVM importance table. If the SVM was trained on latent/PCA
    dimensions, each latent contribution is distributed across source variables
    using absolute fitted component loadings when available. For encoders without
    linear loadings, latent contributions are distributed by absolute
    latent-original correlations in the discovery data.
    """
    if svm_result is None:
        svm_result = final_metrics.get("svm_results", {}) if isinstance(final_metrics, dict) else {}

    importance = _series_from_svm_result(svm_result, "feature_importance_mean")
    importance_std = _series_from_svm_result(svm_result, "feature_importance_std")
    meta = dict((svm_result or {}).get("feature_importance_meta") or {})
    if importance is None or importance.empty:
        raise ValueError("SVM result does not contain feature_importance_mean.")

    latent_parts = {feature: _svm_latent_feature_parts(feature) for feature in importance.index}
    if not any(idx is not None for _, idx in latent_parts.values()):
        direct = pd.DataFrame({
            "feature": importance.index,
            "original_feature": importance.index,
            "importance_mean": importance.values,
            "importance_std": importance_std.reindex(importance.index).values if importance_std is not None else np.nan,
            "importance_source": "direct_svm_feature_importance",
            "svm_importance_method": meta.get("method"),
            "svm_kernel": meta.get("kernel"),
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
        return direct.head(top_n) if top_n is not None else direct

    ae_res = final_metrics.get("ae_res", {}) if isinstance(final_metrics, dict) else {}
    rows = []
    direct_rows = []
    modalities = sorted({mod for mod, idx in latent_parts.values() if idx is not None}, key=lambda x: "" if x is None else str(x))
    if not modalities:
        modalities = [None]

    for modality in modalities:
        latent_features = [
            feature for feature, (mod, idx) in latent_parts.items()
            if idx is not None and mod == modality
        ]
        if not latent_features:
            continue
        latent_indices = np.array([latent_parts[feature][1] for feature in latent_features], dtype=int)
        latent_importance = importance.reindex(latent_features).to_numpy(dtype=float)
        latent_std = (
            importance_std.reindex(latent_features).to_numpy(dtype=float)
            if importance_std is not None else None
        )

        source_df = _metrics_data_for_modality(final_metrics, modality, subject_id_column=subject_id_column)
        payload = _ae_payload_for_modality(ae_res, modality)
        if source_df is None or source_df.empty or payload is None:
            for feature in latent_features:
                direct_rows.append({
                    "feature": feature,
                    "original_feature": feature,
                    "modality": modality,
                    "importance_mean": float(importance.loc[feature]),
                    "importance_std": float(importance_std.loc[feature]) if importance_std is not None and feature in importance_std.index else np.nan,
                    "importance_source": "latent_feature_unmapped",
                    "svm_importance_method": meta.get("method"),
                    "svm_kernel": meta.get("kernel"),
                })
            continue

        source_numeric = source_df.apply(pd.to_numeric, errors="coerce")
        source_features = list(source_numeric.columns)
        components, source_method = _component_matrix_from_payload(payload)
        if components is not None:
            n_components = components.shape[0]
            valid_mask = latent_indices < n_components
            weights = _normalized_abs_rows(components[:n_components, :len(source_features)])
        else:
            latent = payload.get("final_latent") if isinstance(payload, dict) else None
            if latent is None:
                valid_mask = np.zeros_like(latent_indices, dtype=bool)
                weights = np.zeros((0, len(source_features)), dtype=float)
                source_method = "latent_feature_unmapped"
            else:
                latent = np.asarray(latent)
                n_components = latent.shape[1] if latent.ndim > 1 else 1
                valid_mask = latent_indices < n_components
                weights = _latent_original_correlation_weights(
                    latent,
                    source_numeric.to_numpy(dtype=float, copy=True),
                )
                source_method = "latent_original_correlation"

        if not np.any(valid_mask):
            continue

        selected_indices = latent_indices[valid_mask]
        selected_importance = latent_importance[valid_mask]
        feature_scores = selected_importance @ weights[selected_indices, :]
        if latent_std is not None:
            selected_std = latent_std[valid_mask]
            feature_std = selected_std @ weights[selected_indices, :]
        else:
            feature_std = np.full(len(source_features), np.nan)

        for feature_name, score, score_std in zip(source_features, feature_scores, feature_std):
            output_feature = (
                f"{modality}__{feature_name}"
                if feature_name_prefix and modality is not None and not str(feature_name).startswith(f"{modality}__")
                else feature_name
            )
            rows.append({
                "feature": output_feature,
                "original_feature": feature_name,
                "modality": modality,
                "importance_mean": float(score),
                "importance_std": float(score_std) if np.isfinite(score_std) else np.nan,
                "importance_source": source_method,
                "svm_importance_method": meta.get("method"),
                "svm_kernel": meta.get("kernel"),
            })

    out = pd.DataFrame(rows + direct_rows)
    if out.empty:
        raise ValueError("Could not map any SVM latent features back to original variables.")
    out = (
        out
        .groupby(["feature", "original_feature", "modality", "importance_source", "svm_importance_method", "svm_kernel"], dropna=False, as_index=False)
        .agg(importance_mean=("importance_mean", "sum"), importance_std=("importance_std", "sum"))
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    return out.head(top_n) if top_n is not None else out


def plot_svm_feature_contributions(
    importance_df,
    feature_col="feature",
    importance_col="importance_mean",
    std_col="importance_std",
    display_col=None,
    top_n=25,
    xlabel="Feature contribution",
    ylabel="Feature",
    title=None,
    order_features=None,
    palette=None,
    figsize=None,
    ax=None,
):
    """Plot SVM feature contributions using the compact horizontal-bar style."""
    if importance_df is None or len(importance_df) == 0:
        raise ValueError("importance_df is empty.")
    if feature_col not in importance_df or importance_col not in importance_df:
        raise KeyError(f"importance_df must contain '{feature_col}' and '{importance_col}'.")

    plot_df = importance_df.copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[importance_col])
    plot_df = plot_df.sort_values(importance_col, ascending=False).head(top_n)
    if order_features is not None and feature_col in plot_df:
        ordered = [f for f in order_features(plot_df[feature_col]) if f in set(plot_df[feature_col])]
        if ordered:
            plot_df = plot_df.set_index(feature_col).loc[ordered].reset_index()
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    if display_col is not None and display_col in plot_df:
        labels = plot_df[display_col].astype(str)
    else:
        labels = plot_df[feature_col].map(display_feature_name).astype(str)

    n_rows = max(len(plot_df), 1)
    if figsize is None:
        figsize = (plt.rcParams.get("figure.figsize", [6.4, 4.8])[0], max(4.0, 0.28 * n_rows))
    if palette is None:
        palette = list(_THEME_CATS)
    colors = [palette[i % len(palette)] for i in range(len(plot_df))]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    xerr = None
    if std_col in plot_df and plot_df[std_col].notna().any():
        xerr = plot_df[std_col].fillna(0.0).to_numpy(dtype=float)

    ax.barh(
        labels,
        plot_df[importance_col].to_numpy(dtype=float),
        xerr=xerr,
        color=colors,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#B8C0CC")
        spine.set_linewidth(0.8)
    fig.tight_layout()
    return fig, ax, plot_df


def _fit_transform_famd_like(chr_df, cc_df=None, subject_id_column="src_subject_id"):
    """
    Lightweight FAMD-style projection matrix.

    Numeric columns are median-imputed and standardised. Categorical columns are
    most-frequent-imputed, one-hot encoded, centred, and frequency-weighted
    using CHR frequencies before PCA. This avoids requiring the optional
    `prince` package while preserving the key FAMD behaviour needed here:
    mixed continuous/categorical features contribute to one shared component
    space fitted on CHR and reused for CC.
    """
    x_chr = chr_df.drop(columns=[subject_id_column], errors="ignore").copy()
    x_cc = None if cc_df is None else cc_df.drop(columns=[subject_id_column], errors="ignore").reindex(columns=x_chr.columns)

    numeric_cols, categorical_cols = _famd_column_split(x_chr)
    blocks_chr = []
    blocks_cc = []

    if numeric_cols:
        num_chr = x_chr[numeric_cols].apply(pd.to_numeric, errors="coerce")
        keep_num = [col for col in numeric_cols if not num_chr[col].isna().all()]
        if keep_num:
            num_chr = num_chr[keep_num]
            num_imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            num_chr_imp = num_imputer.fit_transform(num_chr)
            num_chr_z = scaler.fit_transform(num_chr_imp)
            blocks_chr.append(num_chr_z)
            if x_cc is not None:
                num_cc = x_cc[keep_num].apply(pd.to_numeric, errors="coerce")
                num_cc_z = scaler.transform(num_imputer.transform(num_cc))
                blocks_cc.append(num_cc_z)

    if categorical_cols:
        cat_chr = (
            x_chr[categorical_cols]
            .where(pd.notna(x_chr[categorical_cols]), "__missing__")
            .astype(str)
        )
        cat_imputer = SimpleImputer(strategy="most_frequent")
        encoder = _onehot_encoder_dense()
        cat_chr_imp = cat_imputer.fit_transform(cat_chr)
        cat_chr_oh = encoder.fit_transform(cat_chr_imp)
        freq = np.nanmean(cat_chr_oh, axis=0)
        freq = np.where(np.isfinite(freq) & (freq > 0), freq, 1.0)
        cat_chr_w = (cat_chr_oh - freq) / np.sqrt(freq)
        blocks_chr.append(cat_chr_w)
        if x_cc is not None:
            cat_cc = (
                x_cc[categorical_cols]
                .where(pd.notna(x_cc[categorical_cols]), "__missing__")
                .astype(str)
            )
            cat_cc_imp = cat_imputer.transform(cat_cc)
            cat_cc_oh = encoder.transform(cat_cc_imp)
            blocks_cc.append((cat_cc_oh - freq) / np.sqrt(freq))

    if not blocks_chr:
        empty_chr = np.zeros((len(x_chr), 1), dtype=float)
        empty_cc = None if x_cc is None else np.zeros((len(x_cc), 1), dtype=float)
        return empty_chr, empty_cc

    z_chr = np.concatenate(blocks_chr, axis=1)
    z_cc = np.concatenate(blocks_cc, axis=1) if x_cc is not None else None
    z_chr = np.nan_to_num(z_chr, nan=0.0, posinf=0.0, neginf=0.0)
    if z_cc is not None:
        z_cc = np.nan_to_num(z_cc, nan=0.0, posinf=0.0, neginf=0.0)
    return z_chr, z_cc


def chr_fit_first_component(chr_df, cc_df=None, modality=None, preprocessing_details=None, subject_id_column="src_subject_id"):
    """
    Fit the first component on CHR and project optional CC into the same space.

    Continuous modalities use PCA. Modalities listed in
    preprocessing_details['preprocessing_parameters']['mixed_categorical_modalities']
    use a lightweight FAMD-style matrix before PCA.
    """
    mixed_modalities = _get_mixed_categorical_modalities(preprocessing_details)
    feature_df = chr_df.drop(columns=[subject_id_column], errors="ignore")
    has_non_numeric_features = any(
        not pd.api.types.is_numeric_dtype(feature_df[col])
        for col in feature_df.columns
    )
    use_famd = modality in mixed_modalities or has_non_numeric_features
    if use_famd:
        z_chr, z_cc = _fit_transform_famd_like(chr_df, cc_df, subject_id_column=subject_id_column)
        method = "FAMD"
    else:
        x_chr = chr_df.drop(columns=[subject_id_column], errors="ignore").copy()
        x_cc = None if cc_df is None else cc_df.drop(columns=[subject_id_column], errors="ignore").reindex(columns=x_chr.columns)
        x_chr = x_chr.apply(pd.to_numeric, errors="coerce")
        keep = [col for col in x_chr.columns if not x_chr[col].isna().all()]
        if not keep:
            z_chr = np.zeros((len(x_chr), 1), dtype=float)
            z_cc = None if x_cc is None else np.zeros((len(x_cc), 1), dtype=float)
        else:
            x_chr = x_chr[keep]
            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            z_chr = scaler.fit_transform(imputer.fit_transform(x_chr))
            if x_cc is not None:
                z_cc = scaler.transform(imputer.transform(x_cc[keep].apply(pd.to_numeric, errors="coerce")))
            else:
                z_cc = None
        method = "PCA"

    z_chr = np.nan_to_num(z_chr, nan=0.0, posinf=0.0, neginf=0.0)
    if z_cc is not None:
        z_cc = np.nan_to_num(z_cc, nan=0.0, posinf=0.0, neginf=0.0)
    if z_chr.shape[1] == 0 or z_chr.shape[0] < 2:
        pc1_chr = np.zeros(z_chr.shape[0], dtype=float)
        pc1_cc = None if z_cc is None else np.zeros(z_cc.shape[0], dtype=float)
        return {"chr": pc1_chr, "cc": pc1_cc, "method": method, "explained_variance_ratio": np.nan}

    pca = PCA(n_components=1, random_state=0)
    pc1_chr = pca.fit_transform(z_chr)[:, 0]
    pc1_cc = None if z_cc is None else pca.transform(z_cc)[:, 0]
    evr = float(pca.explained_variance_ratio_[0]) if hasattr(pca, "explained_variance_ratio_") else np.nan
    return {"chr": pc1_chr, "cc": pc1_cc, "method": method, "explained_variance_ratio": evr}


def build_overlap_labels_by_modality(
    data_by_modality,
    labels_by_modality,
    preprocessing_details=None,
    subject_id_column="src_subject_id",
):
    """
    Build mixed-safe ordered labels for modality-overlap and cluster-mapping plots.

    Clusters are ordered by their median CHR-fitted first component score:
    - k=2: low_<modality>, high_<modality>
    - k=3: low_<modality>, intermediate_<modality>, high_<modality>
    - k>3: low_<modality>, level2_<modality>, ..., high_<modality>

    This preserves every retained cluster instead of collapsing all non-high
    clusters into "low", and it works for PCA and FAMD-style mixed domains.
    """
    overlap_labels = {}
    cluster_ordering = {}
    modalities = list(data_by_modality.keys())
    if len(labels_by_modality) != len(modalities):
        raise ValueError(
            f"Expected {len(modalities)} modality label arrays, got {len(labels_by_modality)}."
        )

    for idx, modality in enumerate(modalities):
        df_mod = data_by_modality[modality]
        labels = pd.Series(np.asarray(labels_by_modality[idx]), index=df_mod.index)
        if len(labels) != len(df_mod):
            raise ValueError(
                f"{modality}: labels length ({len(labels)}) != modality rows ({len(df_mod)})."
            )
        unique_labels = labels.dropna().unique().tolist()
        if len(unique_labels) < 2:
            continue

        comp = chr_fit_first_component(
            df_mod,
            cc_df=None,
            modality=modality,
            preprocessing_details=preprocessing_details,
            subject_id_column=subject_id_column,
        )
        score_df = pd.DataFrame({
            "cluster": labels.astype(str).to_numpy(),
            "component1": np.asarray(comp["chr"], dtype=float),
        })
        ordering = (
            score_df.groupby("cluster", dropna=False)["component1"]
            .median()
            .sort_values(ascending=True)
        )
        ordered_clusters = ordering.index.astype(str).tolist()
        n_clusters = len(ordered_clusters)
        clean_modality = str(modality).replace(" ", "_").replace("/", "_")
        if n_clusters == 2:
            rank_tokens = ["low", "high"]
        elif n_clusters == 3:
            rank_tokens = ["low", "intermediate", "high"]
        else:
            rank_tokens = ["low"] + [f"level{rank}" for rank in range(2, n_clusters)] + ["high"]
        label_map = {
            cluster: f"{token}_{clean_modality}"
            for cluster, token in zip(ordered_clusters, rank_tokens)
        }
        mapped = labels.astype(str).map(label_map)
        if mapped.isna().any():
            missing = sorted(labels.astype(str)[mapped.isna()].unique().tolist())
            raise ValueError(f"{modality}: could not map cluster labels {missing}.")
        overlap_labels[modality] = mapped.tolist()
        cluster_ordering[modality] = {
            "method": comp["method"],
            "cluster_component1_median": ordering.to_dict(),
            "label_map": label_map,
        }

    return overlap_labels, cluster_ordering


def plot_chr_cc_first_component_by_modality(
    chr_data_by_modality,
    labels_by_modality,
    cc_data_by_modality,
    plots_dir,
    preprocessing_details=None,
    subject_id_column="src_subject_id",
    sample_label="",
    display_quantiles=(0.01, 0.99),
    save_combined=True,
):
    """Create one CHR-vs-CC first-component violin per modality."""
    out_dir = os.path.join(plots_dir, "merged_feature_pca_chr_vs_cc")
    os.makedirs(out_dir, exist_ok=True)
    combined_payload = []
    for mod_num, (modality, df_chr) in enumerate(chr_data_by_modality.items()):
        if modality not in cc_data_by_modality:
            warnings.warn(f"{modality}: no CC data available; skipping CHR-vs-CC component plot.")
            continue
        labels = np.asarray(labels_by_modality[mod_num]).astype(str)
        if len(labels) != len(df_chr):
            raise ValueError(f"{modality}: labels length ({len(labels)}) != CHR rows ({len(df_chr)}).")
        comp = chr_fit_first_component(
            df_chr,
            cc_data_by_modality[modality],
            modality=modality,
            preprocessing_details=preprocessing_details,
            subject_id_column=subject_id_column,
        )
        plot_df = pd.concat(
            [
                pd.DataFrame({"group": labels, "component1": comp["chr"], "cohort": "CHR"}),
                pd.DataFrame({"group": "CC", "component1": comp["cc"], "cohort": "CC"}),
            ],
            ignore_index=True,
        )
        group_order = sorted(pd.unique(labels), key=cluster_sort_key) + ["CC"]
        palette = modality_cluster_palette(group_order, modality=modality)
        combined_payload.append({
            "modality": modality,
            "plot_df": plot_df.copy(),
            "group_order": list(group_order),
            "palette": dict(palette),
            "method": comp["method"],
            "evr": comp["explained_variance_ratio"],
        })

        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        sns.violinplot(
            data=plot_df,
            x="group",
            y="component1",
            hue="group",
            order=group_order,
            hue_order=group_order,
            palette=palette,
            dodge=False,
            inner="quartile",
            cut=0,
            bw_adjust=1.0,
            linewidth=1,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=plot_df,
            x="group",
            y="component1",
            order=group_order,
            color="black",
            size=3,
            jitter=0.22,
            alpha=0.35,
            ax=ax,
        )
        medians = plot_df.groupby("group")["component1"].median()
        ax.scatter(
            np.arange(len(group_order)),
            [medians.loc[g] for g in group_order],
            marker="D",
            s=45,
            color="white",
            edgecolor="black",
            linewidth=1.1,
            zorder=4,
        )
        counts = plot_df.groupby("group")["component1"].size()
        y_range, display_meta = robust_axis_range(plot_df["component1"], quantiles=display_quantiles)
        if y_range is not None:
            ax.set_ylim(*y_range)
            y_text = y_range[0] + (y_range[1] - y_range[0]) * 0.07
        else:
            y_text = 0.0
        clipped_below, clipped_above = _annotate_clipped_component_counts(
            ax,
            [
                (idx, plot_df.loc[plot_df["group"] == group, "component1"].to_numpy())
                for idx, group in enumerate(group_order)
            ],
            y_range,
        )
        for idx, group in enumerate(group_order):
            ax.text(idx, y_text, f"n={int(counts.get(group, 0))}", ha="center", va="bottom", fontsize=10)
        evr = comp["explained_variance_ratio"]
        evr_txt = "NA" if not np.isfinite(evr) else f"{evr:.1%}"
        ax.set_xlabel("Group", fontsize=14)
        ax.set_ylabel(f"Component 1 score (CHR-fitted {comp['method']})", fontsize=14)
        prefix = f"{sample_label} " if sample_label else ""
        ax.set_title(f"{prefix}{modality}: CHR clusters + CC ({comp['method']} C1, EVR={evr_txt})", fontsize=16)
        if display_meta.get("mode") == "quantile" and (clipped_below or clipped_above):
            ax.text(
                0.99,
                0.01,
                f"Robust display range; off-scale values marked (^ above, v below).",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color="#444444",
            )
        ax.grid(axis="y", alpha=0.18)
        sns.despine(ax=ax)
        fig.tight_layout()
        sample_suffix = f"_{sample_label.strip().replace(' ', '_')}" if sample_label else ""
        _save_matplotlib_png_pdf(
            fig,
            os.path.join(out_dir, f"{modality}_PC1_violin_CHR_vs_CC{sample_suffix}"),
            dpi=300,
        )
        plt.show()

    if save_combined and combined_payload:
        n_panels = len(combined_payload)
        n_cols = min(3, n_panels)
        n_rows = int(np.ceil(n_panels / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.3 * n_cols, 4.8 * n_rows),
            squeeze=False,
        )
        axes_flat = axes.ravel()
        for ax, payload in zip(axes_flat, combined_payload):
            plot_df = payload["plot_df"]
            group_order = payload["group_order"]
            palette = payload["palette"]
            sns.violinplot(
                data=plot_df,
                x="group",
                y="component1",
                hue="group",
                order=group_order,
                hue_order=group_order,
                palette=palette,
                dodge=False,
                inner="quartile",
                cut=0,
                bw_adjust=1.0,
                linewidth=1,
                legend=False,
                ax=ax,
            )
            sns.stripplot(
                data=plot_df,
                x="group",
                y="component1",
                order=group_order,
                color="black",
                size=2.3,
                jitter=0.2,
                alpha=0.28,
                ax=ax,
            )
            medians = plot_df.groupby("group")["component1"].median()
            ax.scatter(
                np.arange(len(group_order)),
                [medians.loc[g] for g in group_order],
                marker="D",
                s=34,
                color="white",
                edgecolor="black",
                linewidth=1,
                zorder=4,
            )
            y_range, display_meta = robust_axis_range(plot_df["component1"], quantiles=display_quantiles)
            if y_range is not None:
                ax.set_ylim(*y_range)
                y_text = y_range[0] + (y_range[1] - y_range[0]) * 0.07
            else:
                y_text = 0.0
            clipped_below, clipped_above = _annotate_clipped_component_counts(
                ax,
                [
                    (idx, plot_df.loc[plot_df["group"] == group, "component1"].to_numpy())
                    for idx, group in enumerate(group_order)
                ],
                y_range,
                fontsize=8,
            )
            counts = plot_df.groupby("group")["component1"].size()
            for idx, group in enumerate(group_order):
                ax.text(idx, y_text, f"n={int(counts.get(group, 0))}", ha="center", va="bottom", fontsize=8)
            evr = payload["evr"]
            evr_txt = "NA" if not np.isfinite(evr) else f"{evr:.1%}"
            ax.set_title(f"{payload['modality']}\n{payload['method']} C1, EVR={evr_txt}", fontsize=12, pad=8)
            ax.set_xlabel("")
            ax.set_ylabel("Component 1 score", fontsize=10)
            ax.tick_params(axis="x", labelrotation=0, labelsize=9)
            ax.tick_params(axis="y", labelsize=9)
            ax.grid(axis="y", alpha=0.16)
            sns.despine(ax=ax)
            if display_meta.get("mode") == "quantile" and (clipped_below or clipped_above):
                ax.text(
                    0.98,
                    0.02,
                    "^ / v = off-scale",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7.5,
                    color="#444444",
                )
        for ax in axes_flat[n_panels:]:
            ax.axis("off")
        prefix = f"{sample_label} " if sample_label else ""
        fig.suptitle(
            f"{prefix}CHR domain clusters and healthy controls by modality",
            fontsize=17,
            y=1.01,
        )
        fig.tight_layout()
        sample_suffix = f"_{sample_label.strip().replace(' ', '_')}" if sample_label else ""
        _save_matplotlib_png_pdf(
            fig,
            os.path.join(out_dir, f"ALL_modalities_domain_PC1_violin_CHR_vs_CC{sample_suffix}"),
            dpi=300,
        )
        plt.show()


def plot_integrated_chr_cc_first_component_by_modality(
    chr_data_by_modality,
    final_labels,
    cc_data_by_modality,
    plots_dir,
    preprocessing_details=None,
    subject_id_column="src_subject_id",
    sample_label="",
    display_quantiles=(0.01, 0.99),
    standardize_for_display=True,
):
    """Create the combined per-modality first-component plot for integrated CHR clusters plus CC."""
    out_dir = os.path.join(plots_dir, "merged_feature_pca_chr_vs_cc")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    final_labels = np.asarray(final_labels).astype(str)
    for modality, df_chr in chr_data_by_modality.items():
        if modality not in cc_data_by_modality:
            warnings.warn(f"{modality}: no CC data available; skipping in integrated CHR-vs-CC component plot.")
            continue
        if len(final_labels) != len(df_chr):
            raise ValueError(f"{modality}: final labels length ({len(final_labels)}) != CHR rows ({len(df_chr)}).")
        comp = chr_fit_first_component(
            df_chr,
            cc_data_by_modality[modality],
            modality=modality,
            preprocessing_details=preprocessing_details,
            subject_id_column=subject_id_column,
        )
        chr_component = np.asarray(comp["chr"], dtype=float)
        cc_component = np.asarray(comp["cc"], dtype=float)
        if standardize_for_display:
            chr_center = float(np.nanmean(chr_component))
            chr_scale = float(np.nanstd(chr_component))
            if not np.isfinite(chr_scale) or chr_scale <= 0:
                chr_scale = 1.0
            chr_display = (chr_component - chr_center) / chr_scale
            cc_display = (cc_component - chr_center) / chr_scale
        else:
            chr_display = chr_component
            cc_display = cc_component
        rows.append(pd.DataFrame({
            "modality": modality,
            "group": final_labels,
            "component1": chr_display,
            "component1_raw": chr_component,
            "cohort": "CHR",
            "method": comp["method"],
            "evr": comp["explained_variance_ratio"],
        }))
        rows.append(pd.DataFrame({
            "modality": modality,
            "group": "CC",
            "component1": cc_display,
            "component1_raw": cc_component,
            "cohort": "CC",
            "method": comp["method"],
            "evr": comp["explained_variance_ratio"],
        }))
        evr = comp["explained_variance_ratio"]
        evr_txt = "NA" if not np.isfinite(evr) else f"{evr:.2%}"
        print(f"{modality}: CHR-fitted {comp['method']} component 1 EVR={evr_txt}")

    if not rows:
        raise ValueError("No modality rows available for integrated CHR-vs-CC component plot.")
    plot_df = pd.concat(rows, ignore_index=True)
    modality_order = [m for m in chr_data_by_modality.keys() if m in set(plot_df["modality"])]
    group_order = sorted(pd.unique(final_labels), key=cluster_sort_key) + ["CC"]
    palette = modality_cluster_palette(group_order)

    fig_w = max(26, 4.8 * len(modality_order))
    fig, ax = plt.subplots(figsize=(fig_w, 8.5))
    sns.violinplot(
        data=plot_df,
        x="modality",
        y="component1",
        hue="group",
        order=modality_order,
        hue_order=group_order,
        palette=palette,
        inner="quartile",
        cut=0,
        bw_adjust=1.0,
        linewidth=1,
        dodge=True,
        width=0.92,
        ax=ax,
    )
    rng = np.random.default_rng(0)
    n_hue = len(group_order)
    violin_width = 0.92
    sub_width = violin_width / n_hue
    jitter_scale = sub_width * 0.28
    for i, modality in enumerate(modality_order):
        for j, group in enumerate(group_order):
            vals = plot_df.loc[
                (plot_df["modality"] == modality) & (plot_df["group"] == group),
                "component1",
            ].dropna().to_numpy()
            if vals.size == 0:
                continue
            center = i - violin_width / 2 + (j + 0.5) * sub_width
            x = center + rng.uniform(-jitter_scale, jitter_scale, size=vals.size)
            ax.scatter(x, vals, color="black", s=12, alpha=0.18, zorder=3, linewidths=0)
    y_range, display_meta = robust_axis_range(plot_df["component1"], quantiles=display_quantiles)
    if y_range is not None:
        ax.set_ylim(*y_range)
    clipped_pairs = []
    for i, modality in enumerate(modality_order):
        for j, group in enumerate(group_order):
            center = i - violin_width / 2 + (j + 0.5) * sub_width
            vals = plot_df.loc[
                (plot_df["modality"] == modality) & (plot_df["group"] == group),
                "component1",
            ].to_numpy()
            clipped_pairs.append((center, vals))
    clipped_below, clipped_above = _annotate_clipped_component_counts(ax, clipped_pairs, y_range, fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:len(group_order)],
        labels[:len(group_order)],
        title="Group",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=13,
        title_fontsize=14,
    )
    for x in np.arange(0.5, len(modality_order), 1.0):
        ax.axvline(x, linewidth=0.8, alpha=0.25)
    ax.set_xlabel("Modality", fontsize=19)
    if standardize_for_display:
        ax.set_ylabel("CHR-standardised Component 1 score", fontsize=19)
    else:
        ax.set_ylabel("Component 1 score (CHR-fitted PCA/FAMD)", fontsize=19)
    prefix = f"{sample_label} " if sample_label else ""
    ax.set_title(f"{prefix}Component 1 distributions by modality (CHR integrated clusters + CC)", fontsize=23, pad=18)
    if display_meta.get("mode") == "quantile" and (clipped_below or clipped_above):
        ax.text(
            0.99,
            0.01,
            "Robust display range; off-scale values marked (^ above, v below).",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            color="#444444",
        )
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(axis="y", alpha=0.15)
    sns.despine(ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=16)
    fig.tight_layout()
    sample_suffix = f"_{sample_label.strip().replace(' ', '_')}" if sample_label else ""
    _save_matplotlib_png_pdf(
        fig,
        os.path.join(out_dir, f"ALL_modalities_PC1_singleplot_violin_CHR_vs_CC{sample_suffix}"),
        dpi=300,
    )
    plt.show()
    return plot_df


def alluvial_sankey_general(
    labels_by_modality: dict,
    final_labels,
    stage_order: list,
    final_name="final",
    high_token="high_severity",
    low_token="low_severity",
    final_order="auto",
    arrangement="snap",
    high_y=0.10,
    low_y=0.90,
    other_y=0.50,
    node_pad=22,
    node_thickness=18,
    width=1400,
    height=650,
    title="All modalities -> final (alluvial Sankey)",
    color_by_final=True,
    save_path=None,
    show=True,
):
    """Plot modality-to-final cluster mappings for any number of clusters.

    This is a backwards-compatible replacement for the old
    ``alluvial_sankey_force_high_top`` notebook helper. The old helper only had
    two fixed vertical slots; this version computes a per-stage category order
    and distributes any extra labels evenly, so it works when domains or the
    final solution have k > 2 and when labels do not match across domains.
    """
    try:
        import plotly.graph_objects as go
    except Exception as err:
        raise RuntimeError("Plotly is required for the alluvial Sankey plot.") from err

    stages = list(stage_order) + [final_name]
    missing = [stage for stage in stage_order if stage not in labels_by_modality]
    if missing:
        raise KeyError(f"Missing labels for stage(s): {missing}")

    df = pd.DataFrame({stage: pd.Series(labels_by_modality[stage]) for stage in stage_order})
    df[final_name] = pd.Series(final_labels)
    if df.empty:
        raise ValueError("No labels available for alluvial Sankey plot.")
    if df.isna().any().any():
        df = df.fillna("<missing>")
    for col in stages:
        df[col] = df[col].astype(str)

    def norm(value):
        """Handle norm."""
        return str(value).strip().lower().replace(" ", "_")

    high_norm = norm(high_token)
    low_norm = norm(low_token)

    def sort_label_key(label):
        """Handle sort label key."""
        label_norm = norm(label)
        try:
            numeric_value = float(label)
            numeric_key = (0, numeric_value)
        except Exception:
            numeric_key = (1, str(label))
        if high_norm and high_norm in label_norm:
            return (-2, numeric_key)
        if low_norm and low_norm in label_norm:
            return (2, numeric_key)
        if "low" in label_norm and "severity" in label_norm:
            return (-1, numeric_key)
        if "high" in label_norm and "severity" in label_norm:
            return (1, numeric_key)
        return (0, numeric_key)

    def infer_final_order():
        """Infer final order."""
        vals = list(df[final_name].unique())
        if final_order == "auto":
            previous_stage = stages[-2]
            rows = []
            for value in vals:
                mask = df[final_name] == value
                n_value = int(mask.sum())
                if n_value == 0:
                    top_score = 0.0
                else:
                    prev_labels = df.loc[mask, previous_stage].map(norm)
                    top_score = float(prev_labels.str.contains(high_norm, regex=False).mean()) if high_norm else 0.0
                rows.append((value, -top_score, sort_label_key(value)))
            return [value for value, _, _ in sorted(rows, key=lambda row: (row[1], row[2]))]
        if isinstance(final_order, list):
            explicit = [str(value) for value in final_order]
            extras = [value for value in vals if value not in explicit]
            return explicit + sorted(extras, key=sort_label_key)
        return sorted(vals, key=sort_label_key)

    stage_label_order = {}
    for stage in stage_order:
        stage_label_order[stage] = sorted(df[stage].unique(), key=sort_label_key)
    stage_label_order[final_name] = infer_final_order()

    def y_positions(labels):
        """Handle y positions."""
        labels = list(labels)
        n_labels = len(labels)
        if n_labels == 0:
            return {}
        if n_labels == 1:
            return {labels[0]: other_y}
        y_min = min(float(high_y), float(low_y))
        y_max = max(float(high_y), float(low_y))
        if y_min == y_max:
            y_min, y_max = 0.05, 0.95
        return {
            label: y_min + (y_max - y_min) * (idx / max(1, n_labels - 1))
            for idx, label in enumerate(labels)
        }

    stage_y = {stage: y_positions(labels) for stage, labels in stage_label_order.items()}
    stage_x = {
        stage: (idx / max(1, len(stages) - 1))
        for idx, stage in enumerate(stages)
    }

    nodes = []
    node_index = {}
    node_x = []
    node_y = []

    def add_node(stage, label):
        """Add node."""
        key = (stage, str(label))
        if key in node_index:
            return node_index[key]
        idx = len(nodes)
        node_index[key] = idx
        nodes.append(f"{stage}:{label}")
        node_x.append(stage_x[stage])
        node_y.append(stage_y.get(stage, {}).get(str(label), other_y))
        return idx

    for stage in stages:
        for label in stage_label_order[stage]:
            add_node(stage, label)

    sources, targets, values = [], [], []
    link_final_labels = []
    for src_stage, tgt_stage in zip(stages[:-1], stages[1:]):
        group_cols = [src_stage, tgt_stage]
        if final_name not in group_cols:
            group_cols.append(final_name)
        counts = (
            df.groupby(group_cols, dropna=False)
            .size()
            .reset_index(name="count")
        )
        for _, row in counts.iterrows():
            sources.append(add_node(src_stage, row[src_stage]))
            targets.append(add_node(tgt_stage, row[tgt_stage]))
            values.append(int(row["count"]))
            link_final_labels.append(str(row[final_name]))

    link_kwargs = {"source": sources, "target": targets, "value": values}
    final_values = stage_label_order[final_name]
    if color_by_final and final_values:
        color_map = modality_cluster_palette(final_values)
        link_kwargs["color"] = [
            color_map.get(label, "rgba(150,150,150,0.35)") for label in link_final_labels
        ]

    fig = go.Figure(
        go.Sankey(
            arrangement=arrangement,
            node=dict(
                label=nodes,
                x=node_x,
                y=node_y,
                pad=node_pad,
                thickness=node_thickness,
            ),
            link=link_kwargs,
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=0.98, yanchor="top"),
        font_size=12,
        width=width,
        height=height,
        margin=dict(t=110, l=20, r=20, b=110),
    )
    if save_path:
        fig.write_image(save_path, scale=2)
    if show:
        fig.show()
    return fig


def domain_map(
    new_labels_by_modality: dict,
    final_labels,
    stage_order=None,
    final_name="final",
    top_token="low_severity",
    bottom_token="high_severity",
    color_for_top_final="#327D6D",
    color_for_bottom_final="#7FE3CD",
    width=1500,
    height=950,
    title="Parallel categories",
    infer_final_order_from_last_stage=True,
    invert_final=False,
    final_top_value=None,
    final_bottom_value=None,
    add_gap_in_final=True,
    gap_weight=0.5,
    gap_slots=2,
    save_file_name="Parcats.pdf",
    plots_dir=None,
    show=True,
):
    """Plot parallel categories for modality-to-final mappings for any k.

    The original notebook helper assumed exactly two modality categories and two
    final groups. This version keeps the same call signature but supports any
    number of categories in each modality and any number of final clusters.
    Extra labels are sorted stably and all final clusters get distinct colors.
    """
    try:
        import plotly.graph_objects as go
    except Exception as err:
        raise RuntimeError("Plotly is required for the parallel-categories plot.") from err

    if stage_order is None:
        stage_order = list(new_labels_by_modality.keys())
    else:
        stage_order = list(stage_order)
    stage_order = [stage for stage in stage_order if stage != final_name]
    missing = [stage for stage in stage_order if stage not in new_labels_by_modality]
    if missing:
        raise KeyError(f"Missing labels for stage(s): {missing}")

    df = pd.DataFrame({stage: pd.Series(new_labels_by_modality[stage]) for stage in stage_order})
    df[final_name] = pd.Series(final_labels)
    if df.empty:
        raise ValueError("No labels available for parallel-categories plot.")
    df = df.fillna("<missing>")
    for col in stage_order + [final_name]:
        df[col] = df[col].astype(str)

    def norm(value):
        """Handle norm."""
        return str(value).strip().lower().replace(" ", "_")

    top_norm = norm(top_token)
    bottom_norm = norm(bottom_token)

    def canonicalize_label(value):
        """Handle canonicalize label."""
        value = str(value).strip()
        value_norm = norm(value)
        if top_norm and top_norm in value_norm:
            return str(top_token)
        if bottom_norm and bottom_norm in value_norm:
            return str(bottom_token)
        if "low" in value_norm and "severity" in value_norm:
            return str(top_token) if "low" in top_norm else "low_severity"
        if "high" in value_norm and "severity" in value_norm:
            return str(bottom_token) if "high" in bottom_norm else "high_severity"
        return value

    for stage in stage_order:
        df[stage] = df[stage].map(canonicalize_label)

    group_cols = stage_order + [final_name]
    agg = df.groupby(group_cols, dropna=False).size().reset_index(name="count")
    agg[final_name] = agg[final_name].astype(str)

    def sort_label_key(label):
        """Handle sort label key."""
        label_norm = norm(label)
        try:
            numeric_value = float(label)
            numeric_key = (0, numeric_value)
        except Exception:
            numeric_key = (1, str(label))
        if top_norm and top_norm in label_norm:
            return (-2, numeric_key)
        if bottom_norm and bottom_norm in label_norm:
            return (2, numeric_key)
        if "low" in label_norm and "severity" in label_norm:
            return (-1, numeric_key)
        if "high" in label_norm and "severity" in label_norm:
            return (1, numeric_key)
        return (0, numeric_key)

    final_vals = [str(value) for value in agg[final_name].unique()]

    def infer_final_order():
        """Infer final order."""
        explicit = []
        if final_top_value is not None:
            explicit.append(str(final_top_value))
        if final_bottom_value is not None and str(final_bottom_value) not in explicit:
            explicit.append(str(final_bottom_value))
        if explicit:
            extras = [value for value in final_vals if value not in explicit]
            ordered = explicit + sorted(extras, key=sort_label_key)
        elif infer_final_order_from_last_stage and stage_order:
            last_stage = stage_order[-1]
            rows = []
            for value in final_vals:
                sub = agg[agg[final_name] == value]
                denom = float(sub["count"].sum())
                if denom <= 0:
                    top_rate = 0.0
                else:
                    is_top = sub[last_stage].astype(str).map(norm).str.contains(top_norm, regex=False)
                    top_rate = float((sub["count"] * is_top).sum() / denom) if top_norm else 0.0
                rows.append((value, -top_rate, sort_label_key(value)))
            ordered = [value for value, _, _ in sorted(rows, key=lambda row: (row[1], row[2]))]
        else:
            ordered = sorted(final_vals, key=sort_label_key)
        if invert_final and len(ordered) >= 2:
            ordered = [ordered[1], ordered[0]] + ordered[2:]
        return ordered

    final_order = infer_final_order()

    stage_orders = {
        stage: sorted([str(value) for value in agg[stage].unique()], key=sort_label_key)
        for stage in stage_order
    }
    max_group_gaps = max(
        [max(0, len(final_order) - 1)]
        + [max(0, len(order) - 1) for order in stage_orders.values()]
    )
    gap_slots = max(1, int(gap_slots))
    SPACERS = [
        [" " * (gap_idx * gap_slots + slot_idx + 1) for slot_idx in range(gap_slots)]
        for gap_idx in range(max_group_gaps)
    ]
    add_spacer = bool(
        add_gap_in_final
        and len(final_order) >= 2
        and SPACERS
        and gap_weight
        and gap_weight > 0
    )

    def spacers_for_order(order):
        """Handle spacers for order."""
        return SPACERS[:max(0, len(order) - 1)]

    def spacer_values_for_order(order):
        """Handle spacer values for order."""
        return [
            spacer
            for gap_spacers in spacers_for_order(order)
            for spacer in gap_spacers
        ]

    agg["_is_spacer"] = False
    if add_spacer:
        dummies = []
        padding_targets = [
            (stage, stage_orders[stage])
            for stage in stage_order
        ] + [(final_name, final_order)]
        anchor_values = {
            stage: str(order[0])
            for stage, order in padding_targets
        }
        for target_name, target_order in padding_targets:
            for spacer in spacer_values_for_order(target_order):
                dummy = dict(anchor_values)
                dummy[target_name] = spacer
                # Keep spacer categories in the Plotly dimension order without
                # taking any visible mass out of the observed subgroup blocks.
                dummy["count"] = 0.0
                dummy["_is_spacer"] = True
                dummies.append(dummy)
        agg = pd.concat([agg, pd.DataFrame(dummies)], ignore_index=True)

    def with_optional_spacers(order):
        """Handle with optional spacers."""
        order = [str(value) for value in order]
        if not add_spacer or len(order) < 2:
            return order
        spaced_order = []
        for idx, value in enumerate(order):
            spaced_order.append(value)
            if idx < len(order) - 1:
                spaced_order.extend(spacers_for_order(order)[idx])
        return spaced_order

    dimensions = []
    for stage in stage_order:
        dimensions.append(
            dict(
                label=stage,
                values=agg[stage],
                categoryorder="array",
                categoryarray=with_optional_spacers(stage_orders[stage]),
            )
        )

    dimensions.append(
        dict(
            label=final_name,
            values=agg[final_name],
            categoryorder="array",
            categoryarray=with_optional_spacers(final_order),
        )
    )

    color_map = modality_cluster_palette(final_order)
    if final_order and color_for_top_final is not None:
        color_map[final_order[0]] = color_for_top_final
    if len(final_order) >= 2 and color_for_bottom_final is not None:
        color_map[final_order[1]] = color_for_bottom_final
    color_map.update({
        spacer: "rgba(0,0,0,0)"
        for gap_spacers in SPACERS
        for spacer in gap_spacers
    })
    line_colors = agg[final_name].map(color_map).fillna("rgba(120,120,120,0.45)")
    line_colors.loc[agg["_is_spacer"]] = "rgba(0,0,0,0)"
    line_colors = line_colors.tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Parcats(
            dimensions=dimensions,
            counts=agg["count"],
            line=dict(color=line_colors),
            arrangement="freeform",
            labelfont=dict(color="#111111"),
            tickfont=dict(color="#111111"),
        )
    )

    for value in final_order:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=color_map[value]),
                name=f"{final_name} = {value}",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=title,
        width=width,
        height=height,
        template="simple_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(t=80, l=40, r=40, b=80),
        legend=dict(
            title=f"Ribbon color = {final_name}",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
    )
    fig.update_xaxes(visible=False, showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, showticklabels=False, showgrid=False, zeroline=False)

    def save_matplotlib_domain_map(output_stem):
        """Save matplotlib domain map."""
        import matplotlib.patches as mpatches
        import matplotlib.path as mpath
        import matplotlib.colors as mcolors

        plot_cols = stage_order + [final_name]
        plot_agg = agg.loc[~agg["_is_spacer"], plot_cols + ["count"]].copy()
        plot_agg["count"] = pd.to_numeric(plot_agg["count"], errors="coerce").fillna(0.0)
        plot_agg = plot_agg.loc[plot_agg["count"] > 0]
        if plot_agg.empty:
            raise ValueError("No non-zero domain-map paths are available to save.")

        dim_names = plot_cols
        dim_orders = {stage: stage_orders[stage] for stage in stage_order}
        dim_orders[final_name] = final_order
        x_positions = np.linspace(0.08, 0.92, len(dim_names))

        total_count = float(plot_agg["count"].sum())
        if total_count <= 0:
            raise ValueError("Domain-map counts sum to zero.")

        y_top = 0.88
        y_bottom = 0.12
        y_span = y_top - y_bottom
        category_gap = 0.018
        bar_width = 0.018
        dim_bands = {}
        dim_usable_spans = {}
        y_positions = {}
        for dim in dim_names:
            totals = plot_agg.groupby(dim, dropna=False)["count"].sum()
            ordered = [str(value) for value in dim_orders[dim] if str(value) in totals.index.astype(str)]
            extras = sorted(
                [str(value) for value in totals.index.astype(str) if str(value) not in ordered],
                key=sort_label_key,
            )
            ordered.extend(extras)
            gap_total = category_gap * max(0, len(ordered) - 1)
            usable_span = max(0.1, y_span - gap_total)
            dim_usable_spans[dim] = usable_span
            current_top = y_top
            dim_bands[dim] = {}
            y_positions[dim] = {}
            for value in ordered:
                value_count = float(totals.loc[value])
                height = usable_span * value_count / total_count
                lower = current_top - height
                dim_bands[dim][value] = (lower, current_top)
                y_positions[dim][value] = (lower + current_top) / 2
                current_top = lower - category_gap

        def to_mpl_color(value, alpha=0.62):
            """Handle to mpl color."""
            value = str(value)
            match = re.match(r"rgba\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)", value)
            if match:
                r, g, b, a = match.groups()
                return (int(r) / 255, int(g) / 255, int(b) / 255, float(a) * alpha)
            try:
                r, g, b, a = mcolors.to_rgba(value)
                return (r, g, b, min(a, alpha))
            except ValueError:
                return (0.47, 0.47, 0.47, alpha)

        fig_mpl, ax = plt.subplots(figsize=(max(width / 100, 6), max(height / 100, 4)))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=14, pad=16)

        for left_dim, right_dim, x0, x1 in zip(dim_names[:-1], dim_names[1:], x_positions[:-1], x_positions[1:]):
            group_cols = list(dict.fromkeys([left_dim, right_dim, final_name]))
            pair_counts = (
                plot_agg
                .groupby(group_cols, dropna=False)["count"]
                .sum()
                .reset_index()
            )
            pair_counts["_left_order"] = pair_counts[left_dim].astype(str).map(
                {value: idx for idx, value in enumerate(dim_bands[left_dim])}
            )
            pair_counts["_right_order"] = pair_counts[right_dim].astype(str).map(
                {value: idx for idx, value in enumerate(dim_bands[right_dim])}
            )
            pair_counts["_final_order"] = pair_counts[final_name].astype(str).map(
                {value: idx for idx, value in enumerate(final_order)}
            ).fillna(len(final_order))
            pair_counts = pair_counts.sort_values(
                ["_left_order", "_right_order", "_final_order"],
                kind="mergesort",
            )
            left_offsets = {
                (left_dim, value): upper
                for value, (_, upper) in dim_bands[left_dim].items()
            }
            right_offsets = {
                (right_dim, value): upper
                for value, (_, upper) in dim_bands[right_dim].items()
            }
            for _, row in pair_counts.iterrows():
                left_value = str(row[left_dim])
                right_value = str(row[right_dim])
                final_value = str(row[final_name])
                if left_value not in dim_bands[left_dim] or right_value not in dim_bands[right_dim]:
                    continue
                count = float(row["count"])
                left_height = dim_usable_spans[left_dim] * count / total_count
                right_height = dim_usable_spans[right_dim] * count / total_count
                left_upper = left_offsets[(left_dim, left_value)]
                left_lower = left_upper - left_height
                right_upper = right_offsets[(right_dim, right_value)]
                right_lower = right_upper - right_height
                left_offsets[(left_dim, left_value)] = left_lower
                right_offsets[(right_dim, right_value)] = right_lower

                x_left = x0 + bar_width / 2
                x_right = x1 - bar_width / 2
                dx = (x_right - x_left) * 0.45
                verts = [
                    (x_left, left_upper),
                    (x_left + dx, left_upper),
                    (x_right - dx, right_upper),
                    (x_right, right_upper),
                    (x_right, right_lower),
                    (x_right - dx, right_lower),
                    (x_left + dx, left_lower),
                    (x_left, left_lower),
                    (x_left, left_upper),
                ]
                codes = [
                    mpath.Path.MOVETO,
                    mpath.Path.CURVE4,
                    mpath.Path.CURVE4,
                    mpath.Path.CURVE4,
                    mpath.Path.LINETO,
                    mpath.Path.CURVE4,
                    mpath.Path.CURVE4,
                    mpath.Path.CURVE4,
                    mpath.Path.CLOSEPOLY,
                ]
                facecolor = to_mpl_color(color_map.get(final_value, "rgba(120,120,120,0.45)"))
                ax.add_patch(
                    mpatches.PathPatch(
                        mpath.Path(verts, codes),
                        facecolor=facecolor,
                        edgecolor=facecolor,
                        linewidth=0.2,
                        zorder=1,
                    )
                )

        for dim, x in zip(dim_names, x_positions):
            ax.text(x, 0.965, dim, ha="center", va="bottom", fontsize=11, fontweight="bold")
            for value, (lower, upper) in dim_bands[dim].items():
                rect = mpatches.Rectangle(
                    (x - bar_width / 2, lower),
                    bar_width,
                    upper - lower,
                    facecolor="#f6f6f6",
                    edgecolor="#111111",
                    linewidth=0.7,
                    zorder=3,
                )
                ax.add_patch(rect)
                label_x = x + bar_width * 0.85
                ax.text(
                    label_x,
                    (lower + upper) / 2,
                    str(value),
                    ha="left",
                    va="center",
                    fontsize=10,
                    color="#111111",
                    zorder=4,
                )

        legend_handles = [
            mpatches.Patch(color=to_mpl_color(color_map[value], alpha=0.9), label=f"{final_name} = {value}")
            for value in final_order
            if value in color_map
        ]
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.04),
                ncol=min(len(legend_handles), 4),
                frameon=False,
                title=f"Ribbon color = {final_name}",
            )

        saved_paths = _save_matplotlib_png_pdf(fig_mpl, output_stem, dpi=300)
        plt.close(fig_mpl)
        return saved_paths

    if show:
        fig.show()

    if plots_dir is not None and save_file_name:
        save_path = os.path.join(plots_dir, save_file_name)
        output_stem = os.path.splitext(save_path)[0]
        try:
            for extension in (".png", ".pdf"):
                fig.write_image(output_stem + extension, scale=2)
        except Exception as err:
            fallback = os.path.splitext(save_path)[0] + ".html"
            fig.write_html(fallback)
            saved_paths = save_matplotlib_domain_map(output_stem)
            print(
                "WARNING: could not write static Plotly parcats image "
                f"({err}). Saved Matplotlib PDF/PNG fallback to: "
                f"{', '.join(map(str, saved_paths))}. Interactive HTML saved to: {fallback}"
            )

    return fig, stage_order


parcats_low_on_top_with_gap_keep_final_numeric = domain_map


##############################################
# Define labels for ordinal scale
##############################################

LabelSpec = Dict[str, Any]

LABEL_SPECS: List[LabelSpec] = [
    # 1. Head injury severity
    {
        "mapping": {
            1: "No Head Injury",
            2: "Mild head injury/concussion with no loss of consciousness",
            3: "Mild head injury/concussion with brief loss of consciousness",
            4: "Mild head injury with LOC for between 2-30 minutes.... lasting 1-24 hours",
            5: "Mild head injury with LOC for between 2-30 minutes.... lasting 1-7 days",
            6: "Mild head injury with LOC for between 2-30 minutes.... lasting >7 days to 2 months",
            7: "Mild head injury with LOC for between 30 minutes-5 hours",
            8: "Head injury with LOC/coma lasting 6 hours or more",
        }
    },
    # 2. Duration in seconds/minutes
    {
        "mapping": {
            1: "Just a few seconds",
            2: "Less than a minute",
            3: "One minute or more",
        }
    },
    # 3. General “present” scale
    {
        "mapping": {
            1: "Not present",
            2: "Very mild",
            3: "Mild",
            4: "Moderate",
            5: "Moderately severe",
            6: "Severe",
            7: "Extremely severe",
        }
    },
    # 4. Affective symptoms (0 = Missing)
    {
        "mapping": {
            0: "Missing",
            1: "asymptomatic, returned to usual self",
            2: "residual/mild affective symptoms",
            3: "partial remission, moderate symptoms or impairment",
            4: "marked/major symptoms or impairment, does not meet criteria for MDE",
            5: "meets definite MDE criteria without prominent psychotic symptoms or extreme impairment",
            6: "meets definite MDE criteria with prominent psychotic symptoms or extreme impairment",
        }
    },
    # 5. Affective symptoms (1–6)
    {
        "mapping": {
            1: "asymptomatic, returned to usual self",
            2: "residual/mild affective symptoms",
            3: "partial remission, moderate symptoms or impairment",
            4: "marked/major symptoms or impairment",
            5: "meets definite criteria without prominent psychotic symptoms or extreme impairment",
            6: "meets definite criteria with prominent psychotic symptoms or extreme impairment",
        }
    },
    # 6. Probable vs definite criteria
    {
        "mapping": {
            1: "asymptomatic, returned to usual self",
            2: "meets probable criteria (mild symptoms)",
            3: "meets definite criteria (severe symptoms)",
        }
    },
    # 7. Observational severity
    {
        "mapping": {
            1: "Not observed",
            2: "Very mild",
            3: "Mild",
            4: "Moderate",
            5: "Moderately Severe",
            6: "Severe",
            7: "Very Severe",
        }
    },
    # 8. Not present → extremely severe (lowercase)
    {
        "mapping": {
            1: "not present",
            2: "very mild",
            3: "mild",
            4: "moderate",
            5: "moderate-severe",
            6: "severe",
            7: "extremely severe",
        }
    },
    # 9. Clinical global impression
    {
        "mapping": {
            1: "Normal, Not ill",
            2: "Minimally ill",
            3: "Mildly ill",
            4: "Moderately ill",
            5: "Markedly ill",
            6: "Severely ill",
            7: "Very Severely ill",
        }
    },
    # 10. First/last only, auto-fill codes 2–5 as strings
    {
        "first": 1,
        "first_label": "least important",
        "last": 6,
        "last_label": "most important",
        "fill_middle": True,
    },
    # 11. None at all → a lot
    {
        "mapping": {
            1: "None at all",
            2: "Very little",
            3: "Some",
            4: "A lot",
        }
    },
    # 12. Substance use severity
    {
        "mapping": {
            1: "Abstinent",
            2: "Use without impairment",
            3: "Abuse",
            4: "Dependence",
            5: "Dependence with institutionalization",
        }
    },
    # 13. (duplicate of #9)
    {
        "mapping": {
            1: "Normal, Not ill",
            2: "Minimally ill",
            3: "Mildly ill",
            4: "Moderately ill",
            5: "Markedly ill",
            6: "Severely ill",
            7: "Very Severely ill",
        }
    },
    # 14. Depression severity

    {
        "mapping": {
            1: "Absent",
            2: "Mild - Expresses some sadness or discouragement on questioning",
            3: "Moderate - Distinct depressed mood persisting up to half the time over last 2 weeks: present daily",
            4: "Severe - Markedly depressed mood persisting daily over half the time interfering with normal motor and social functioning",

        }
    },

    {
        "mapping": {
            1: "Absent",
            2: "Mild - Has at times felt hopeless over the last week but still has some degree of hope for the future",
            3: "Moderate - Persistent, moderate sense of hopelessness over last week. Can be persuaded to acknowledge possibility of things being better",
            4: "Severe - Persisting and distressing sense of hopelessness",
        },
    },

    {
        "mapping": {
            1: "Absent",
            2: "Mild - Some inferiority; not amounting to feelings of worthlessness",
            3: "Moderate - Subject feels worthless, but less than 50% of the time",
            4: "Severe - Subject feels worthless more than 50% of the time. May be challenged to acknowledge otherwise",
        },

    },

    {
        "mapping": {
            1: "Absent",
            2: "Mild - Subject feels blamed but not accused less than 50% of the time",
            3: "Moderate - Persisting sense of being blamed, and/or occasional sense of being accused",
            4: "Severe - Persistent sense of being accused. When challenged, acknowledges that it is not so",
        },
    },

    {
        "mapping": {
            1: "Absent",
            2: "Mild - Subject sometimes feels over guilty about some minor peccadillo, but less than 50% of time",
            3: "Moderate - Subject usually (over 50% of time) feels guilty about past actions the significance of which he exaggerates",
            4: "Severe - Subject usually feels s/he is to blame for everything that has gone wrong, even when not his/her fault",
        },
    },

    {
        "mapping": {
            1: "Absent - No Depression",
            2: "Mild - Depression present but no diurnal variation",
            3: "Moderate - Depression spontaneously mentioned to be worse in a.m.",
            4: "Severe - Depression markedly worse in a.m., with impaired functioning which improves in p.m.",
        },
    },

    {
        "mapping": {
            1: "Absent - No early wakening",
            2: "Mild - Occasionally wakes (up to twice weekly) 1 hour or more before normal time to wake or alarm time",
            3: "Moderate - Often wakes early (up to 5 times weekly) 1 hour or more before normal time to wake or alarm",
            4: "Severe - Daily wakes l hour or more before normal time",
        },
    },

    {
        "mapping": {
            1: "Absent",
            2: "Mild - Frequent thoughts of being better off dead, or occasional thoughts or occasional thoughts of suicide",
            3: "Moderate - Deliberately considered suicide with a plan, but made no attempt",
            4: "Severe - Suicidal attempt apparantly designed to end in death (i.e.: accidental discovery or inefficient means)"

        }
    },

    {
        "mapping": {
            1: "Absent",
            2: "Mild - Subject appears sad and mournful even during parts of the interview, involving affectively neutral discussion",
            3: "Moderate - Subject appears sad and mournful throughout the interview, with gloomy monotonous voice and is tearful or close to tears at times",
            4: "Severe - Subject chokes on distressing topics, frequently sighs deeply and cries openly, or is persistently in a state of frozen misery if examiner is sure that this is present"
        }
    },

    {
        "mapping": {
            1: "Normal, not at all depressed",
            2: "Borderline depressed",
            3: "Mildly depressed",
            4: "Moderately depressed",
            5: "Markedly depressed",
            6: "Severely depressed",
            7: "Among the most severely depressed patients",
        }
    },
    # 15. Clinical global (0 = not assessed)
    {
        "mapping": {
            0: "Not assessed",
            1: "Normal, not at all ill",
            2: "Borderline ill",
            3: "Mildly ill",
            4: "Moderately ill",
            5: "Markedly ill",
            6: "Severely ill",
            7: "Among the most extremely ill patients",
        }
    },
    # 16. Symptom presence
    {
        "mapping": {
            1: "Normal/No symptoms",
            2: "Mild",
            3: "Moderate",
            4: "Severe",
            5: "Very severe",
        }
    },

    # 17. Frequency of occurrence
    {
        "mapping": {
            1: "Never",
            2: "Almost Never",
            3: "Sometimes",
            4: "Fairly often",
            5: "Very often",
        }
    },
    # 18. Impact scale (0=None → 5=Death)
    {
        "mapping": {
            0: "None",
            1: "Minor",
            2: "Moderate",
            3: "Moderately Severe",
            4: "Severe",
            5: "Death",
        }
    },
    # 19. Absent → severe
    {
        "mapping": {
            1: "Absent",
            2: "Mild",
            3: "Moderate",
            4: "Severe",
        }
    },
    # 20. Weekly frequency
    {
        "mapping": {
            1: "Less than once a week",
            2: "Once a week",
            3: "2-5 times in week",
            4: "Daily or almost daily",
            5: "Many times each day",
        }
    },
    # 21. Duration within hours
    {
        "mapping": {
            1: "Fleeting - few seconds or minutes",
            2: "Less than 1 hour/some of the time",
            3: "1-4 hours/a lot of time",
            4: "4-8 hours/most of day",
            5: "More than 8 hours/persistent or continuous",
        }
    },
    # 22. Thought control ability
    {
        "mapping": {
            0: "Does not attempt to control thoughts",
            1: "Easily able to control thoughts",
            2: "Can control thoughts with little difficulty",
            3: "Can control thoughts with some difficulty",
            4: "Can control thoughts with a lot of difficulty",
            5: "Unable to control thoughts",
        }
    },
    # 23. Suicide deterrent impact
    {
        "mapping": {
            0: "Does not apply",
            1: "Deterrents definitely stopped you from attempting suicide",
            2: "Deterrents probably stopped you",
            3: "Uncertain that deterrents stopped you",
            4: "Deterrents most likely did not stop you",
            5: "Deterrents definitely did not stop you",
        }
    },
    # 24. Reason for self‐harm
    {
        "mapping": {
            0: "Does not apply",
            1: "Completely to get attention revenge or a reaction from others",
            2: "Most likely to get attention revenge or a reaction from others",
            3: "Equally to get attention revenge or a reaction from others and to stop the pain",
            4: "Mostly to end or stop the pain",
            5: "Completely to end or stop the pain",
        }
    },
    # 25. Suicidal ideation
    {
        "mapping": {
            0: "No ideation",
            1: "wish to be dead",
            2: "non-specific active suicidal thoughts",
            3: "active suicidal ideation with any methods (no plan) without intent to act",
            4: "active suicidal ideation with some intent to act, without specific plan",
            5: "active suicidal ideation with specific plan and intent",
        }
    },
    # 26. Bottom/top group (“most severe” covers 2–5)
    {
        "mapping": {
            1: "least severe",
            2: "most severe",
            3: "most severe",
            4: "most severe",
            5: "most severe",
        }
    },
    # 27. Suicidal ideation (1–5)
    {
        "mapping": {
            1: "wish to be dead",
            2: "non-specific active suicidal thoughts",
            3: "active suicidal ideation with any methods (no plan) without intent to act",
            4: "active suicidal ideation with some intent to act, without specific plan",
            5: "active suicidal ideation with specific plan and intent",
        }
    },
    # 28. Smoking frequency
    {
        "mapping": {
            1: "Not applicable, I have never smoked",
            2: "Monthly or less",
            3: "2-4 times per month",
            4: "2-3 times per week",
            5: "Daily or almost daily",
        }
    },
    # 29. Role functioning (1–10)
    {
        "mapping": {
            1: "Extreme role dysfunction",
            2: "Inability to function",
            3: "Marginal ability to function",
            4: "Major impairment in role functioning",
            5: "Serious impairment in role functioning",
            6: "Moderate impairment in role functioning",
            7: "Mild problems in role functioning",
            8: "Good role functioning",
            9: "Above average role functioning",
            10: "Superior role functioning",
        }
    },
    # 30. Social functioning (1–10, long labels)
    {
        "mapping": {
            1: "Extreme social isolation",
            2: "Inability to function socially: Unable to function socially or to maintain any interpersonal relationships",
            3: "Marginal ability to function socially or maintain interpersonal relationships",
            4: "Major Impairment in social/interpersonal functioning",
            5: "Serious Impairment in social/interpersonal functioning",
            6: "Moderate Impairment in social/interpersonal functioning",
            7: "Mild problems: Some persistent mild difficulty in social functioning",
            8: "Good: Some transient mild impairment in social functioning",
            9: "Above average: Good Functioning in all social areas, and interpersonally effective",
            10: "Superior: Superior functioning in a wide range of social and interpersonal activities",
        }
    },
    # 31. Simple 1–5 severity
    {
        "mapping": {
            1: "Slight",
            2: "Some",
            3: "Moderate",
            4: "Major",
            5: "Severe",
        }
    },
    # 32. Global functioning (0–10)
    {
        "mapping": {
            10: "Superior functioning in a wide range of activities",
            9: "Good functioning in all areas, occupationally and socially effective",
            8: "No more than a slight impairment in social, occupational or school functioning (e.g., infrequent interpersonal conflict, temporarily falling behind in schoolwork)",
            7: "Some difficulty in social, occupational, or school functioning but generally functioning well and has some meaningful, interpersonal relationships",
            6: "Moderate difficulty in social, occupational, or school functioning (e.g., few friends, conflicts with peers or co-workers)",
            5: "Serious impairment in social, occupational, or school functioning (e.g., no friends, unable to keep a job)",
            4: "Major impairment in several areas such as work or school, family relations (e.g., depressed man avoids friends, neglects family and is unable to work; child frequently beats up younger children, is defiant at home and failing at school)",
            3: "Inability to function in almost all areas (e.g., stays in bed all day; no job, home, or friends)",
            2: "Occasionally fails to maintain minimal personal hygiene; unable to function independently",
            1: "Persistent inability to maintain minimal personal hygiene. Unable to function without harming self or others or without considerable external support (e.g., nursing care and supervision)",
            0: "Inadequate information",
        }
    },
    # 33. Social functioning (1–10, short)
    {
        "mapping": {
            1: "Extreme social isolation",
            2: "Inability to function socially",
            3: "Marginal ability to function socially",
            4: "Major impairment in social and interpersonal functioning",
            5: "Serious impairment in social/interpersonal functioning",
            6: "Moderate impairment in social/interpersonal functioning",
            7: "Mild problems in social/interpersonal functioning",
            8: "Good social/interpersonal functioning",
            9: "Above average social/interpersonal functioning",
            10: "Superior social/interpersonal functioning",
        }
    },
    # 34. Role functioning (1–10, variant)
    {
        "mapping": {
            1: "Extreme role dysfunction",
            2: "Inability to function",
            3: "Marginal ability to function",
            4: "Major impairment in role functioning",
            5: "Serious impairment in role functioning",
            6: "Moderate impairment in role functioning",
            7: "Mild Impairment in role functioning",
            8: "Good role functioning",
            9: "Above average role functioning",
            10: "Superior role functioning",
        }
    },
    # 35. Role functioning (1–10, detailed variant)
    {
        "mapping": {
            1: "Extreme role dysfunction",
            2: "Inability to function",
            3: "Marginal ability to function",
            4: "Major impairment in role functioning",
            5: "Serious Impairment in Role Functioning. Serious impairment independently",
            6: "Moderate impairment in role functioning",
            7: "Mild impairment in role functioning",
            8: "Good role functioning",
            9: "Above average role functioning",
            10: "Superior role functioning",
        }
    },
    # 36. Withdrawal scale (sparse codes)
    {
        "mapping": {
            0: "Not withdrawn",
            2: "Mild withdrawal",
            4: "Moderately withdrawn",
            6: "Unrelated to others, withdrawn and isolated, avoids contacts",
        }
    },
    # 37. Education level (1–15)
    {
        "mapping": {
            1: "Less than 6th grade",
            2: "Some high school",
            3: "High school diploma or GED",
            4: "Some college, no degree",
            5: "Associates degree",
            6: "Bachelors degree",
            7: "Some graduate school",
            8: "Masters degree and above",
            9: "Some post-graduate training, no degree",
            10: "Completed 8th grade, no high school",
            11: "High school",
            12: "College or University",
            13: "Graduate school",
            14: "Other",
            15: "Less than high school",
        }
    },
    # 38. Cognitive test speed
    {
        "mapping": {
            0: "Fail",
            4: "Correct in 66-120 seconds",
            5: "Correct in 46-65 seconds",
            6: "Correct in 31-45 seconds",
            7: "Correct in 1-30 seconds",
        }
    },
    {
        "mapping": {
            0: "Fail",
            4: "Correct in 61-120 seconds",
            5: "Correct in 46-60 seconds",
            6: "Correct in 36-45 seconds",
            7: "Correct in 1-35 seconds",
        }
    },
    # 39. Simple correctness
    {
        "mapping": {
            0: "Fail",
            1: "Partially correct",
            2: "Correct",
        }
    },
    # 40. Multi-code “Correct”
    {
        "mapping": {
            0: "Fail",
            2: "Correct",
            3: "Correct",
            4: "Correct",
        }
    },
    # 41. Cognitive test speed variant
    {
        "mapping": {
            0: "Fail",
            4: "Correct in 31-60 seconds",
            5: "Correct in 21-30 seconds",
            6: "Correct in 11-20 seconds",
            7: "Correct in 1-10 seconds",
        }
    },
    {
        "mapping": {
            0: "Fail",
            4: "Correct in 76-120 seconds",
            5: "Correct in 61-75 seconds",
            6: "Correct in 31-60 seconds",
            7: "Correct in 1-30 seconds",
        }
    },
    # 43. Trial correctness
    {
        "mapping": {
            0: "Fail",
            1: "One trial correct",
            2: "Both trials correct",
        }
    },
    {
        "mapping": {
            0: "Fail",
            1: "One trial correct",
            2: "Two trials correct",
            3: "All trials correct",
        }
    },
    # 45. Completion time buckets
    {
        "mapping": {
            0: "Complete in 45 seconds",
            1: "Complete in 40-44 seconds",
            2: "Complete in 35-39 seconds",
            3: "Complet in 30-34 seconds",
            4: "Complete in 0-29 seconds",
        }
    },
    # 46. Intensity (1–5)
    {
        "mapping": {
            1: "Not at all",
            2: "A little bit",
            3: "Somewhat",
            4: "Quite a bit",
            5: "Very much",
        }
    },
    # 47. Quality rating
    {
        "mapping": {
            1: "Very Poor",
            2: "Poor",
            3: "Fair",
            4: "Good",
            5: "Very Good",
        }
    },
    # 48. Frequency (Never→Always)
    {
        "mapping": {
            1: "Never",
            2: "Almost Never",
            3: "Sometimes",
            4: "Almost Always",
            5: "Always",
        }
    },
    # 49. Custom “Moderately” grouping
    {
        "mapping": {
            1: "Not at all",
            2: "Moderately",
            3: "Moderately",
            4: "Moderately",
            5: "Moderately",
            6: "Moderately",
            7: "Very much",
        }
    },
    # 50. Disturbance scale (only labeled codes)
    {
        "mapping": {
            0: "Not at all disturbing or disabling",
            2: "Slightly disturbing but not really disabling",
            4: "definitely disturbing or disabling",
            6: "Markedly disturbing or disabling",
            8: "Very severy disturbing or disabling",
        }
    },
]


##############################################
# Extract date
##############################################

def extract_date(file_path, prefix):
    """
    Extracts the date from the filename by removing the specified prefix and the '.csv' suffix.
    Assumes the remaining part of the filename is in the format 'YYYY-MM-DD'.

    Parameters:
        file_path (str): The path to the file.
        prefix (str): The prefix to remove (e.g., 'basetable_' or 'metatable_').

    Returns:
        datetime: The extracted date.
    """
    base = os.path.basename(file_path)
    date_str = base.replace(prefix, "").replace(".csv", "")
    return datetime.strptime(date_str, "%Y-%m-%d")

##############################################
# Load most recent data
##############################################

def load_most_recent_basetable(folder_path):
    """
    Loads the most recent basetable CSV file from the specified folder.
    Assumes basetable files are named like 'basetable_YYYY-MM-DD.csv'.
    """
    pattern = os.path.join(folder_path, "basetable_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No basetable files found in the directory: {folder_path}")
    latest_file = max(files, key=lambda f: extract_date(f, "basetable_"))
    print("Loading most recent basetable file:", latest_file)
    df = pd.read_csv(latest_file)

    # Replace 'nan' and empty strings with np.nan
    df.replace("nan", np.nan, inplace=True)
    df.replace("", np.nan, inplace=True)  # Also handles empty strings
    return df

def load_most_recent_metatable(folder_path):
    """
    Loads the most recent metatable CSV file from the specified folder.
    Assumes metatable files are named like 'metatable_YYYY-MM-DD.csv'.
    """
    pattern = os.path.join(folder_path, "metatable_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No metatable files found in the directory: {folder_path}")
    latest_file = max(files, key=lambda f: extract_date(f, "metatable_"))
    print("Loading most recent metatable file:", latest_file)
    df = pd.read_csv(latest_file)
    return df


##############################################
# Split data by network for discovery and test
##############################################
def split_by_network(df, prescient_ids, id_col='src_subject_id'):
    """
    Splits a dict of Series/DataFrames into two dicts (Prescient vs Pronet)
    based on membership in prescient_ids, and drops the helper 'Network' column.
    """
    discovery_df = {}
    test_df = {}


    # 1. Ensure a DataFrame
    if isinstance(df, pd.Series):
        df_mod = df.to_frame().copy()
    else:
        df_mod = df.copy()

    # 2. Case‐insensitive lookup of the ID column
    lower_map = {col.lower(): col for col in df_mod.columns}
    actual_col = lower_map[id_col.lower()]

    # 3. Flag network membership
    mask = df_mod[actual_col].isin(prescient_ids)
    df_mod['Network'] = np.where(mask, 'Prescient', 'Pronet')

    # 4. Split out and drop the helper column
    pres_df = df_mod.loc[df_mod['Network'] == 'Prescient'].reset_index(drop=True)
    pron_df = df_mod.loc[df_mod['Network'] == 'Pronet'].reset_index(drop=True)

    discovery_df = pres_df.drop(columns='Network')
    test_df      = pron_df.drop(columns='Network')

    return discovery_df, test_df


##############################################
# Get seperate data for each modality
##############################################
def extract_modalities(
    meta: pd.DataFrame,
    data: pd.DataFrame,
    subject_id_column: str = 'src_subject_id'
) -> dict:
    """
    Extracts separate DataFrames for each modality from the data using the meta table.

    Parameters:
    - meta (pd.DataFrame): A DataFrame with at least the columns 'ElementName' and 'Modality'.
    - data (pd.DataFrame): A DataFrame containing 'src_subject_id' and variables as columns.

    Returns:
    - modality_dfs (dict): A dictionary where each key is a modality and each value is a DataFrame
      that includes 'src_subject_id' and the variables associated with that modality.
    """
    # Get unique, non-null modalities
    modalities = meta['Modality'].dropna().unique()

    # Dictionary to store the DataFrame for each modality
    modality_dfs = {}

    for modality in modalities:
        # Get the list of variable names for the current modality
        modality_vars = meta.loc[meta['Modality'] == modality, 'ElementName'].dropna().unique()

        # Ensure the variable exists in data (if meta contains variables that aren't in data),
        # including one-hot encoded columns that start with the variable name followed by an underscore.
        valid_vars = []
        for var in modality_vars:
            matches = [col for col in data.columns if col == var or col.startswith(f"{var}_")]
            valid_vars.extend(matches)

        # Always include the subject_id_column if it exists
        columns_to_include = (
            [subject_id_column] + valid_vars
            if subject_id_column in data.columns else valid_vars
        )

        # Skip if no valid variables are available for this modality
        if not valid_vars:
            preview = ", ".join(map(str, modality_vars[:5]))
            if len(modality_vars) > 5:
                preview += ", ..."
            print(
                f"Skipping modality '{modality}' - None of its metadata variables were found in data"
                + (f" (examples: {preview})." if preview else ".")
            )
            continue

        # Slice and then reset_index so every modality df has a clean 0..N-1 index
        df_mod = data[columns_to_include].copy().reset_index(drop=True)
        modality_dfs[modality] = df_mod

    return modality_dfs


##############################################
# Remove high missing columns and rows
##############################################

def remove_high_missing_data(
    df: pd.DataFrame,
    subject_id_column: str = 'src_subject_id',
    col_threshold: float = 0.5,
    row_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Removes columns (variables) with more than col_threshold fraction of missing values
    and then removes rows (subjects) with more than row_threshold fraction of missing values.

    The subject_id_column is always preserved.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        subject_id_column (str): Column to preserve.
        col_threshold (float): Max allowed fraction of missing values per column.
                               e.g., 0.5 => drop columns with over 50% missing.
        row_threshold (float): Max allowed fraction of missing values per row.
                               e.g., 0.5 => drop rows with over 50% missing.

    Returns:
        pd.DataFrame: Cleaned DataFrame with the subject_id_column preserved.
    """
    # Separate subject IDs to ensure they aren't dropped
    if subject_id_column in df.columns:
        subject_ids = df[[subject_id_column]]
        df_data = df.drop(columns=[subject_id_column])
    else:
        subject_ids = None
        df_data = df.copy()

    num_rows = df_data.shape[0]
    num_cols = df_data.shape[1]

    # 1) Drop columns with more than 'col_threshold' fraction missing.
    #    Keep columns with at least (1 - col_threshold) * num_rows non-missing values.
    min_non_missing_col = int(np.ceil((1 - col_threshold) * num_rows))
    df_data = df_data.dropna(axis=1, thresh=min_non_missing_col)

    # 2) Drop rows with more than 'row_threshold' fraction missing.
    #    Keep rows with at least (1 - row_threshold) * current_num_cols non-missing values.
    current_num_cols = df_data.shape[1]
    min_non_missing_row = int(np.ceil((1 - row_threshold) * current_num_cols))
    df_data = df_data.dropna(axis=0, thresh=min_non_missing_row)

    # Reattach subject IDs
    if subject_ids is not None:
        # Possibly filter out any subjects that were dropped
        final_df = subject_ids.loc[df_data.index].join(df_data, how='inner')
    else:
        final_df = df_data

    return final_df


def remove_high_missing_data_split(
    discovery_df: pd.DataFrame,
    test_df: pd.DataFrame,
    subject_id_column: str = 'src_subject_id',
    col_threshold: float = 0.5,
    row_threshold: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans discovery and test DataFrames by:
      1) Dropping any column that has more than col_threshold fraction missing
         in EITHER DataFrame.
      2) Keeping only the intersection of columns that survive in both sets.
      3) Dropping any row (within each DataFrame) that has more than
         row_threshold fraction missing ACROSS those shared columns.
    The subject_id_column is always preserved and re-joined at the end.

    Returns:
        (cleaned_discovery_df, cleaned_test_df)
    """
    def split_ids(df):
        """Split ids."""
        if subject_id_column in df.columns:
            return df[[subject_id_column]], df.drop(columns=[subject_id_column])
        else:
            return None, df.copy()

    disc_ids, disc_data = split_ids(discovery_df)
    test_ids, test_data = split_ids(test_df)

    # 1) Identify columns passing threshold in each
    def passing_columns(df_data):
        """Handle passing columns."""
        n_rows = df_data.shape[0]
        min_non_missing = int(np.ceil((1 - col_threshold) * n_rows))
        return set(df_data.dropna(axis=1, thresh=min_non_missing).columns)

    disc_cols = passing_columns(disc_data)
    test_cols = passing_columns(test_data)

    # 2) Keep only the intersection of those columns
    common_cols = sorted(disc_cols & test_cols)
    disc_data = disc_data[common_cols]
    test_data = test_data[common_cols]

    # 3) Drop rows exceeding the row_threshold
    def drop_bad_rows(df_data):
        """Handle drop bad rows."""
        n_cols = df_data.shape[1]
        min_non_missing = int(np.ceil((1 - row_threshold) * n_cols))
        return df_data.dropna(axis=0, thresh=min_non_missing)

    disc_data = drop_bad_rows(disc_data)
    test_data = drop_bad_rows(test_data)

    # 4) Re-attach subject IDs to the filtered rows
    def reattach(ids, data):
        """Handle reattach."""
        if ids is None:
            return data
        return ids.loc[data.index].join(data, how='inner')

    cleaned_discovery = reattach(disc_ids, disc_data)
    cleaned_test = reattach(test_ids, test_data)

    return cleaned_discovery, cleaned_test


def remove_high_missing_data_test(
    df: pd.DataFrame,
    df_discovery: pd.DataFrame,
    subject_id_column: str = 'src_subject_id',
    col_threshold: float = 0.5,
    row_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Keeps all columns that exist in discovery.
    """
    # Separate subject IDs to ensure they aren't dropped
    if subject_id_column in df.columns:
        subject_ids = df[[subject_id_column]]
        df_data = df.drop(columns=[subject_id_column])
    else:
        subject_ids = None
        df_data = df.copy()

    # Keep only columns that exist in discovery
    df_data = df_data[df_data.columns.intersection(df_discovery.columns)]


    # 2) Drop rows with more than 'row_threshold' fraction missing.
    #    Keep rows with at least (1 - row_threshold) * current_num_cols non-missing values.
    current_num_cols = df_data.shape[1]
    min_non_missing_row = int(np.ceil((1 - row_threshold) * current_num_cols))
    df_data = df_data.dropna(axis=0, thresh=min_non_missing_row)

    # Reattach subject IDs
    if subject_ids is not None:
        # Possibly filter out any subjects that were dropped
        final_df = subject_ids.loc[df_data.index].join(df_data, how='inner')
    else:
        final_df = df_data

    return final_df


def remove_missing_from_modalities(
    modalities_data: dict,
    subject_id_column: str = 'src_subject_id',
    col_threshold: float = 0.5,
    row_threshold: float = 0.5
) -> dict:
    """
    Applies remove_high_missing_data to each modality DataFrame in the modalities_data dictionary.

    Parameters:
        modalities_data (dict): Dictionary where keys are modality names and values are DataFrames.
        subject_id_column (str): Column that identifies the subject.
        col_threshold (float): Drop columns with missing fraction > col_threshold.
        row_threshold (float): Drop rows with missing fraction > row_threshold.

    Returns:
        dict: Dictionary with the same keys as modalities_data, where each DataFrame has had high-missing
              columns and rows removed.
    """
    cleaned_modalities = {}
    for modality, df in modalities_data.items():
        cleaned_modalities[modality] = remove_high_missing_data(
            df,
            subject_id_column=subject_id_column,
            col_threshold=col_threshold,
            row_threshold=row_threshold
        )
    return cleaned_modalities


##############################################
# Log transform
##############################################

def auto_power_transform(
    df: pd.DataFrame,
    skew_threshold: float = 0.75,
    return_details: bool = False
) -> pd.DataFrame:
    """
    Applies Yeo-Johnson power transformation to highly skewed numerical columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        skew_threshold (float): The absolute skewness value above which transformation is applied.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    df_transformed = df.copy()
    numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
    transformed_cols = []
    lambda_by_column = {}

    for col in numeric_cols:
        col_values = df_transformed[col].dropna()
        col_skewness = skew(col_values, bias=False)

        if abs(col_skewness) > skew_threshold:
            # Reshape required for sklearn transformers
            col_array = df_transformed[col].values.reshape(-1, 1)
            pt = PowerTransformer(method='yeo-johnson', standardize=False)

            # Fit and transform (handling NaNs by skipping rows with them)
            mask = df_transformed[col].notnull()
            transformed = np.full_like(df_transformed[col], np.nan, dtype=np.float64)
            transformed[mask] = pt.fit_transform(col_array[mask]).flatten()

            df_transformed[col] = transformed
            transformed_cols.append(col)
            lambda_by_column[col] = float(pt.lambdas_[0])

    if return_details:
        details = {
            "skew_threshold": float(skew_threshold),
            "transformed_columns": transformed_cols,
            "lambda_by_column": lambda_by_column
        }
        return df_transformed, details
    return df_transformed


def apply_power_transform_from_details(
    df: pd.DataFrame,
    power_details: dict
) -> pd.DataFrame:
    """
    Apply a previously fitted per-column Yeo-Johnson transform.
    """
    if power_details is None:
        return df.copy()

    transformed = df.copy()
    lambda_by_column = power_details.get("lambda_by_column", {}) or {}

    for col, lmbda in lambda_by_column.items():
        if col not in transformed.columns:
            continue
        x = transformed[col].astype(float)
        mask = x.notna()
        xv = x[mask].to_numpy(dtype=np.float64, copy=True)

        pos = xv >= 0
        neg = ~pos
        yv = np.empty_like(xv, dtype=np.float64)

        if abs(lmbda) > 1e-12:
            yv[pos] = (np.power(xv[pos] + 1.0, lmbda) - 1.0) / lmbda
        else:
            yv[pos] = np.log1p(xv[pos])

        if abs(lmbda - 2.0) > 1e-12:
            yv[neg] = -((np.power(1.0 - xv[neg], 2.0 - lmbda) - 1.0) / (2.0 - lmbda))
        else:
            yv[neg] = -np.log1p(-xv[neg])

        x_out = x.to_numpy(dtype=np.float64, copy=True)
        x_out[mask.to_numpy()] = yv
        transformed[col] = x_out

    return transformed


##############################################
# Imputation
##############################################

def impute_diverse_data(df: pd.DataFrame, subject_id_column: str = 'src_subject_id', n_neighbors: int = 7) -> pd.DataFrame:
    """
    Impute missing values for a DataFrame using KNN imputation:
    - Numeric columns: imputed using KNN.
    - Categorical columns: encoded, imputed using KNN, then decoded.
    - Date columns: imputed with a constant date.

    The subject_id_column is preserved and not imputed.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        subject_id_column (str): The column name for the subject identifier.
        n_neighbors (int): Number of neighbors for KNN imputation.

    Returns:
        pd.DataFrame: A DataFrame with missing values imputed.
    """
    # Separate subject ID column
    if subject_id_column in df.columns:
        subject_ids = df[[subject_id_column]]
        df_data = df.drop(columns=[subject_id_column])
    else:
        subject_ids = None
        df_data = df.copy()

    # Normalize string "nan" to real NaN in data columns
    df_data.replace("nan", np.nan, inplace=True)

    # Identify column types
    numeric_cols = df_data.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df_data.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df_data.select_dtypes(include=['datetime64[ns]']).columns.tolist()

    # Handle categorical variables (convert to numerical labels)
    label_encoders = {}
    for col in object_cols:
        le = LabelEncoder()
        df_data[col] = df_data[col].astype(str) # Convert to string
        df_data[col] = le.fit_transform(df_data[col])  # Encode as integers
        label_encoders[col] = le  # Store encoder to decode later

    # Handle missing datetime values by imputing a constant
    for col in datetime_cols:
        df_data[col] = pd.to_datetime(df_data[col], errors='coerce')
        df_data[col] = df_data[col].fillna(pd.Timestamp('1900-01-01'))

    # Apply KNN imputation.
    # keep_empty_features=True preserves columns that are entirely missing in this cohort,
    # preventing shape mismatches when reconstructing DataFrames.
    try:
        knn_imputer = KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
    except TypeError:
        # Backward compatibility with older sklearn versions.
        knn_imputer = KNNImputer(n_neighbors=n_neighbors)

    imputed_values = knn_imputer.fit_transform(df_data)
    if imputed_values.shape[1] != df_data.shape[1]:
        # Defensive alignment if sklearn drops all-empty columns in older versions.
        warnings.warn(
            "KNNImputer returned fewer columns than expected; reintroducing missing columns as NaN "
            "to preserve training feature schema alignment."
        )
        kept_cols = getattr(knn_imputer, "feature_names_in_", None)
        if kept_cols is None:
            kept_cols = list(df_data.columns[:imputed_values.shape[1]])
        tmp = pd.DataFrame(imputed_values, columns=list(kept_cols), index=df_data.index)
        df_imputed = tmp.reindex(columns=df_data.columns)
    else:
        df_imputed = pd.DataFrame(imputed_values, columns=df_data.columns, index=df_data.index)


    # Reset indexes before concatenation to avoid mismatched shapes
    df_imputed.reset_index(drop=True, inplace=True)
    if subject_ids is not None:
        subject_ids.reset_index(drop=True, inplace=True)
        df_imputed = pd.concat([subject_ids, df_imputed], axis=1)
    return df_imputed


def impute_data(modalities_data: dict, subject_id_column: str = 'src_subject_id', n_neighbors: int = 7) -> dict:
    """
    Applies KNN imputation for diverse data types to each modality DataFrame.

    Parameters:
        modalities_data (dict): A dictionary where each key is a modality and each value is a DataFrame.
        subject_id_column (str): Column to be preserved (not imputed).
        n_neighbors (int): Number of neighbors for KNN imputation.

    Returns:
        dict: A dictionary with the same keys as modalities_data, where each DataFrame has missing values imputed.
    """
    imputed_modalities = {}
    for modality, df in modalities_data.items():
        # Check for subjects with all variables missing
        df_mod = df.copy()
        # Separate out data columns (exclude subject ID if present)
        if subject_id_column in df_mod.columns:
            data_only = df_mod.drop(columns=[subject_id_column])
        else:
            data_only = df_mod
        # Identify rows where all data are missing
        all_missing_mask = data_only.isna().all(axis=1)
        num_all_missing = all_missing_mask.sum()
        if num_all_missing > 0:
            warnings.warn(
                f"{modality}: {num_all_missing} participants have missing values for all variables "
                "and will be excluded from imputation."
            )
            # Drop those subjects before imputation
            df_mod = df_mod.loc[~all_missing_mask].reset_index(drop=True)
        if df_mod.empty:
            warnings.warn(f"{modality}: no rows remain after removing all-missing rows; skipping modality.")
            continue
        imputed_modalities[modality] = impute_diverse_data(df_mod, subject_id_column, n_neighbors)
    return imputed_modalities


##############################################
# Data scaling
##############################################

def scale_diverse_data (
    df: pd.DataFrame,
    subject_id_column: str = 'src_subject_id',
    scaler_type: str = 'robust',
    return_details: bool = False
) -> pd.DataFrame:
    """
    Scales only the continuous numeric columns in a DataFrame while leaving binary/low‐cardinality
    numeric columns and non-numeric columns (categorical, dates, etc.) unchanged.

    Parameters:
      df (pd.DataFrame): The input DataFrame.
      subject_id_column (str): The name of the subject identifier column to preserve.
      scaler_type (str): 'standard' for StandardScaler or 'minmax' for MinMaxScaler.


    Returns:
      pd.DataFrame: The DataFrame with continuous numeric columns scaled.
      If return_details=True, returns (DataFrame, details_dict).
    """
    # Separate the subject identifier column if present.
    if subject_id_column in df.columns:
        subject_ids = df[[subject_id_column]]
        df_data = df.drop(columns=[subject_id_column])
    else:
        subject_ids = None
        df_data = df.copy()

    # Identify numeric columns.
    numeric_cols = df_data.select_dtypes(include=[np.number]).columns.tolist()

    # Select the scaler.
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("scaler_type must be either 'standard', 'robust' or 'minmax'.")

    # Make a copy and scale only the continuous numeric columns.
    df_data_scaled = df_data.copy()
    scaling_details = {}
    for col in numeric_cols:
        df_data_scaled[col] = scaler.fit_transform(df_data[[col]])
        col_details = {}
        if scaler_type == 'standard':
            col_details = {
                'mean': float(scaler.mean_[0]),
                'var': float(scaler.var_[0]),
                'scale': float(scaler.scale_[0])
            }
        elif scaler_type == 'minmax':
            col_details = {
                'min': float(scaler.min_[0]),
                'scale': float(scaler.scale_[0]),
                'data_min': float(scaler.data_min_[0]),
                'data_max': float(scaler.data_max_[0]),
                'data_range': float(scaler.data_range_[0])
            }
        elif scaler_type == 'robust':
            col_details = {
                'center': float(scaler.center_[0]),
                'scale': float(scaler.scale_[0])
            }
        scaling_details[col] = col_details

    # Reassemble the final DataFrame with the subject identifier column (if present).
    if subject_ids is not None:
        final_df = pd.concat([subject_ids, df_data_scaled], axis=1)
    else:
        final_df = df_data_scaled

    if return_details:
        details = {
            'scaler_type': scaler_type,
            'subject_id_column': subject_id_column,
            'scaled_numeric_columns': numeric_cols,
            'column_scaler_params': scaling_details
        }
        return final_df, details
    return final_df


def scale_data(
    modalities_data: dict,
    subject_id_column: str = 'src_subject_id',
    scaler_type: str = 'standard',
    return_details: bool = False
    ) -> dict:
    """
    Applies the scale_diverse_data function to each modality DataFrame in the modalities_data dictionary.

    Parameters:
      modalities_data (dict): Dictionary where keys are modality names and values are DataFrames.
      subject_id_column (str): Column to be preserved (not scaled).
      scaler_type (str): 'standard' for StandardScaler or 'minmax' for MinMaxScaler.

    Returns:
      dict: Dictionary with the same keys as modalities_data, where each DataFrame has its continuous
            numeric columns scaled.
      If return_details=True, returns (scaled_modalities, details_dict).
    """
    scaled_modalities = {}
    scaling_details = {}
    for modality, df in modalities_data.items():
        if return_details:
            scaled_df, mod_details = scale_diverse_data(
                df,
                subject_id_column=subject_id_column,
                scaler_type=scaler_type,
                return_details=True
            )
            scaled_modalities[modality] = scaled_df
            scaling_details[modality] = mod_details
        else:
            scaled_modalities[modality] = scale_diverse_data(
                df,
                subject_id_column=subject_id_column,
                scaler_type=scaler_type
            )
    if return_details:
        return scaled_modalities, scaling_details
    return scaled_modalities


def apply_scaling_from_details(
    df: pd.DataFrame,
    scaling_details: dict,
    subject_id_column: str = 'src_subject_id'
) -> pd.DataFrame:
    """
    Apply previously fitted per-column scaling parameters to a DataFrame.
    """
    if scaling_details is None:
        return df.copy()

    scaler_type = scaling_details.get('scaler_type', 'robust')
    scaled_cols = scaling_details.get('scaled_numeric_columns', [])
    params = scaling_details.get('column_scaler_params', {})

    if subject_id_column in df.columns:
        subject_ids = df[[subject_id_column]]
        data = df.drop(columns=[subject_id_column]).copy()
    else:
        subject_ids = None
        data = df.copy()

    for col in scaled_cols:
        if col not in data.columns:
            continue
        col_params = params.get(col, {})
        vals = data[col].astype(float)
        if scaler_type == 'standard':
            denom = col_params.get('scale', 1.0)
            if abs(denom) < 1e-12:
                data[col] = 0.0
            else:
                data[col] = (vals - col_params.get('mean', 0.0)) / denom
        elif scaler_type == 'minmax':
            data[col] = vals * col_params.get('scale', 1.0) + col_params.get('min', 0.0)
        elif scaler_type == 'robust':
            denom = col_params.get('scale', 1.0)
            if abs(denom) < 1e-12:
                data[col] = 0.0
            else:
                data[col] = (vals - col_params.get('center', 0.0)) / denom
        else:
            raise ValueError(f"Unsupported scaler_type '{scaler_type}' in scaling details.")

    if subject_ids is not None:
        return pd.concat([subject_ids, data], axis=1)
    return data


def apply_scaling_to_modalities_from_details(
    modalities_data: dict,
    modality_scaling_details: dict,
    subject_id_column: str = 'src_subject_id'
) -> dict:
    """Apply scaling to modalities from details."""
    out = {}
    for mod, df_mod in modalities_data.items():
        out[mod] = apply_scaling_from_details(
            df_mod,
            modality_scaling_details.get(mod),
            subject_id_column=subject_id_column
        )
    return out


def impute_data_with_reference(
    modalities_data: dict,
    reference_modalities: dict,
    subject_id_column: str = 'src_subject_id',
    n_neighbors: int = 7
) -> dict:
    """
    Fit KNN imputer per modality on reference (training) data and transform new data.
    """
    imputed = {}
    for modality, df_new in modalities_data.items():
        if modality not in reference_modalities:
            raise KeyError(f"Missing imputation reference for modality '{modality}'.")

        df_ref = reference_modalities[modality]
        df_new_mod = df_new.copy()

        data_only = df_new_mod.drop(columns=[subject_id_column]) if subject_id_column in df_new_mod.columns else df_new_mod
        all_missing_mask = data_only.isna().all(axis=1)
        if all_missing_mask.any():
            df_new_mod = df_new_mod.loc[~all_missing_mask].reset_index(drop=True)
        if df_new_mod.empty:
            warnings.warn(
                f"{modality}: no rows remain after removing all-missing rows; skipping modality."
            )
            continue

        if subject_id_column in df_new_mod.columns:
            ids = df_new_mod[[subject_id_column]].reset_index(drop=True)
            X_new = df_new_mod.drop(columns=[subject_id_column]).copy()
        else:
            ids = None
            X_new = df_new_mod.copy()

        X_ref = df_ref.drop(columns=[subject_id_column]).copy() if subject_id_column in df_ref.columns else df_ref.copy()
        X_ref = X_ref.replace("nan", np.nan)
        X_new = X_new.replace("nan", np.nan)

        X_new = X_new.reindex(columns=X_ref.columns)
        X_ref = X_ref.apply(pd.to_numeric, errors='coerce')
        X_new = X_new.apply(pd.to_numeric, errors='coerce')
        if X_new.empty:
            warnings.warn(f"{modality}: no follow-up rows to impute; skipping modality.")
            continue

        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputer.fit(X_ref)
        X_imp = pd.DataFrame(imputer.transform(X_new), columns=X_ref.columns)

        if ids is not None:
            imputed[modality] = pd.concat([ids, X_imp], axis=1)
        else:
            imputed[modality] = X_imp
    return imputed


##############################################
# Convert data for VAE structure
##############################################
def dummy_code(
    df: pd.DataFrame,
    subject_id_column: str = 'src_subject_id',
    columns_to_encode: list | None = None
) -> pd.DataFrame:
    """
    Preprocesses a single modality DataFrame:
      - Converts datetime columns to numeric timestamps
      - Preserves numeric values even when they were loaded as object/string
      - Ordinal encodes variables based on LABEL_SPECS
      - One-hot encodes remaining categorical (object) columns
      - Returns a fully numeric DataFrame including the subject identifier (if present)

    Parameters:
      df (pd.DataFrame): Input DataFrame.
      subject_id_column (str): Column name for the subject identifier.
      columns_to_encode (list | None): Optional subset of raw columns that should
        receive ordinal/one-hot encoding. When None, all object columns are encoded.

    Returns:
      pd.DataFrame: Preprocessed DataFrame ready for model input.
    """
    df = df.copy()

    # Separate subject ID column
    if subject_id_column in df.columns:
        subject_ids = df[[subject_id_column]]
        df_data = df.drop(columns=[subject_id_column])
    else:
        subject_ids = None
        df_data = df.copy()

    # Normalize common missing-value strings before deciding whether an object
    # column is numeric, ordinal text, or nominal text.
    object_columns = df_data.select_dtypes(include=['object', 'string']).columns.tolist()
    if object_columns:
        df_data[object_columns] = df_data[object_columns].replace(
            ["nan", "NaN", "None", "NULL", "null", ""],
            np.nan,
        )

    # Numeric scores sometimes arrive as object columns after notebook or CSV
    # handling. If every observed value is numeric-like, preserve the variable
    # as one numeric feature instead of expanding its values into dummy levels.
    for col in object_columns:
        observed = df_data[col].notna()
        if not observed.any():
            continue
        numeric_values = pd.to_numeric(df_data[col], errors='coerce')
        if numeric_values.loc[observed].notna().all():
            df_data[col] = numeric_values

    # Convert datetime columns to numeric timestamps
    datetime_cols = df_data.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns.tolist()
    for col in datetime_cols:
        df_data[col] = pd.to_datetime(df_data[col], errors='coerce')
        df_data[col] = df_data[col].apply(lambda x: x.value if pd.notnull(x) else np.nan)

    # ----------------------------------------
    # Ordinal encode variables based on LABEL_SPECS
    ordinal_specs = []
    for spec in LABEL_SPECS:
        # build full mapping of code -> label
        if 'mapping' in spec:
            mapping = spec['mapping']
        elif spec.get('fill_middle'):
            first, last = spec['first'], spec['last']
            mapping = {first: spec['first_label'], last: spec['last_label']}
            for code in range(first+1, last):
                mapping[code] = str(code)
        else:
            continue
        # reverse mapping: label -> code
        rev_map = {label: code for code, label in mapping.items()}
        label_set = set(mapping.values())
        ordinal_specs.append((label_set, rev_map))

    # identify all object columns
    object_cols = df_data.select_dtypes(include=['object', 'string']).columns.tolist()
    if columns_to_encode is None:
        columns_to_encode = object_cols
    columns_to_encode = {col for col in columns_to_encode if col in df_data.columns}
    ordinal_cols = []

    # apply ordinal encoding for matching specs
    for col in object_cols:
        if col not in columns_to_encode:
            continue
        unique_vals = set(df_data[col].dropna().unique())
        for label_set, rev_map in ordinal_specs:
            if unique_vals.issubset(label_set):
                df_data[col] = df_data[col].map(lambda x: rev_map.get(x, np.nan))
                ordinal_cols.append(col)
                break

    # one-hot encode the remaining object columns
    remaining_obj = [c for c in object_cols if c in columns_to_encode and c not in ordinal_cols]
    if remaining_obj:
        missing_masks = {
            col: df_data[col].isna() | df_data[col].astype("object").isin(["nan", "NaN", "None", ""])
            for col in remaining_obj
        }
        columns_before_dummies = set(df_data.columns)
        df_data = pd.get_dummies(df_data, columns=remaining_obj, drop_first=True, dummy_na=False)
        for col, missing_mask in missing_masks.items():
            dummy_cols = [
                c for c in df_data.columns
                if c not in columns_before_dummies and c.startswith(f"{col}_")
            ]
            if dummy_cols:
                df_data[dummy_cols] = df_data[dummy_cols].astype(float)
                df_data.loc[missing_mask.to_numpy(), dummy_cols] = np.nan
    # ----------------------------------------

    # Convert boolean columns to integers (True -> 1, False -> 0)
    bool_cols = df_data.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        df_data[col] = df_data[col].astype(int)

    # Reattach the subject ID column
    if subject_ids is not None:
        df_processed = pd.concat([subject_ids, df_data], axis=1)
    else:
        df_processed = df_data

    return df_processed


def convert_df_for_vae(df: pd.DataFrame, subject_id_column: str = 'src_subject_id'):
    """
    Converts a preprocessed DataFrame into VAE-compatible format.

    Parameters:
      df (pd.DataFrame): Preprocessed input DataFrame with numeric values.
      subject_id_column (str): Column name for the subject identifier.

    Returns:
      subject_ids (np.ndarray): Array of subject identifiers.
      numeric_array (np.ndarray): 2D NumPy array (n_samples, n_features) for VAE training.
    """
    df = df.copy()

    # Separate subject IDs if present
    if subject_id_column in df.columns:
        subject_ids = df[subject_id_column].values
        df.drop(columns=[subject_id_column], inplace=True)
    else:
        subject_ids = None

    numeric_array = df.apply(pd.to_numeric, errors='coerce').values.astype(np.float32)

    return subject_ids, numeric_array


def convert_data_for_vae(modalities_data: dict, subject_id_column: str = 'src_subject_id') -> dict:
    """
    Convert each modality DataFrame in a dictionary to a numeric format for VAE training.

    Parameters:
      modalities_data (dict): Dictionary where keys are modality names and values are DataFrames.
      subject_id_column (str): Column name for the subject identifier.

    Returns:
      converted_data (dict): Dictionary where each key is a modality and each value is a tuple:
                             (subject_ids, numeric_array)
    """
    converted_data = {}
    for modality, df in modalities_data.items():
        subject_ids, numeric_array = convert_df_for_vae(df, subject_id_column)
        converted_data[modality] = (subject_ids, numeric_array)
    return converted_data


##############################################
# PCA for each modality
##############################################

def compute_PCA(df, n_components=None):
        """
        PCA on one individual dataframe.
        """
       # Drop 'src_subject_id' and 'interview_date' if they are in the dataframe
        columns_to_exclude = ['src_subject_id', 'interview_date']
        df_clean = df.drop(columns=[col for col in columns_to_exclude if col in df.columns])

        # Optionally, keep only numeric columns if the dataframe contains non-numeric data
        df_numeric = df_clean.select_dtypes(include=['number'])

        # Initialize and fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(df_numeric)

        return pca


def run_pca_on_modalities(data_dict, n_components=None):
    """
    Runs PCA for each modality in a dictionary of dataframes.

    Parameters:
    - data_dict (dict): Dictionary where keys are modality names and values are DataFrames.
    - n_components (int or None): Number of components to keep. If None, all components are kept.

    Returns:
    - pca_results (dict): Dictionary with modality names as keys and PCA results as values.
                           Each value is a dict containing:
                           - 'pca': the fitted PCA object,
                           - 'explained_variance_ratio': explained variance ratio of each component,
                           - 'components': the principal axes in feature space.
    """
    pca_results = {}
    for modality, df in data_dict.items():
        pca = compute_PCA(df, n_components=None)

        # Store the results in the dictionary
        pca_results[modality] = {
            'pca': pca,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'components': pca.components_
        }

    return pca_results

# Example usage:
# modalities = {
#     'modality1': df1,
#     'modality2': df2,
#     # ...
# }
# pca_results = run_pca_on_modalities(modalities, n_components=5)
# print(pca_results['modality1']['explained_variance_ratio'])


##################### PLOTS #####################

def plot_latent_feature_crosscorr(
    latent_variables: np.ndarray,
    original_data: np.ndarray,
    vmin,
    vmax,
    feature_names=None
):
    """
    Creates a figure with:
      - One row per latent variable, heatmap sorted by correlation strength.
      - Features correctly mapped (x-axis values always match features).
      - A unified colorbar to indicate correlation.
      - A legend listing the actual feature names, **not reordered**.
      - Adjusted layout to prevent overlapping elements.

    Parameters
    ----------
    latent_variables : np.ndarray
        Shape (n_samples, latent_dim).
    original_data : np.ndarray
        Shape (n_samples, n_features).
    feature_names : list of str, optional
        Names for the original features. Defaults to ["Feature_1", "Feature_2", ...].
    """

    # Ensure arrays
    latent_variables = np.asarray(latent_variables)
    original_data = np.asarray(original_data)

    n_samples, latent_dim = latent_variables.shape
    _, feat_dim = original_data.shape

    # Default feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature_{j+1}" for j in range(feat_dim)]

    # Compute correlation matrix
    combined = np.hstack((latent_variables, original_data))
    corr_matrix = np.corrcoef(combined, rowvar=False)
    cross_corr = corr_matrix[:latent_dim, latent_dim:]

    # ============= Figure Setup =============
    fig_height = 4 * latent_dim  # Increased height per latent dimension
    fig_width = max(12, min(40, 1.2 * feat_dim))  # Adjust width dynamically based on features
    fig, axes = plt.subplots(
        nrows=latent_dim, ncols=1,
        figsize=(fig_width, fig_height)
    )
    axes = np.atleast_1d(axes)

    cmap = "coolwarm"

    for i in range(latent_dim):
        ax = axes[i]

        # Sort features **per latent variable**
        sorted_indices = np.argsort(-cross_corr[i])  # Sort descending
        sorted_corrs = cross_corr[i, sorted_indices]
        sorted_feature_names = [display_feature_name(feature_names[j]) for j in sorted_indices]

        # Create heatmap
        sns.heatmap(
            sorted_corrs.reshape(1, -1),
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0,
            linewidths=0.5,
            annot=False,
            cbar=False,
            xticklabels=sorted_feature_names,  # Correct feature names
            yticklabels=[f"Latent_{i+1}"]
        )

        ax.set_title(f"Latent_{i+1} vs. Features", fontsize=10, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

    # Adjust spacing between subplots
    plt.subplots_adjust(left=0.05, right=0.8, top=0.95, bottom=0.1, hspace=1.2)  # More space

    # ============= Single Colorbar =============
    from matplotlib.colors import Normalize
    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])

    # Adjust colorbar placement to avoid overlap
    cbar_ax = fig.add_axes([0.83, 0.15, 0.02, 0.7])  # Shifted right, avoid overlap
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Correlation", fontsize=10)

    # Improve layout
    plt.tight_layout(rect=[0, 0, 0.82, 1])  # Reserve space for colorbar

    plt.show()


def top10_features_per_latent(
    latent_variables: np.ndarray,
    original_data: np.ndarray,
    feature_names=None
):
    """
    Parameters
    ----------
    latent_variables : np.ndarray
        Shape (n_samples, latent_dim).
    original_data : np.ndarray
        Shape (n_samples, n_features).
    feature_names : list of str, optional
        Names for the original features. Defaults to ["Feature_0", "Feature_1", ...].
    """

    # Convert inputs to arrays
    latent_variables = np.asarray(latent_variables)
    original_data = np.asarray(original_data)

    # Figure out dimensions
    n_samples, latent_dim = latent_variables.shape
    _, feat_dim = original_data.shape

    # If no feature names provided, generate a list of placeholder names
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(feat_dim)]
    else:
        # Optionally, ensure correct length
        if len(feature_names) != feat_dim:
            raise ValueError("feature_names must match the number of columns in original_data.")

    # Compute correlation matrix (currently using Pearson)
    combined = np.hstack((latent_variables, original_data))
    corr_matrix = np.corrcoef(combined, rowvar=False)

    # Slice the correlation matrix to get cross-correlations
    # cross_corr shape: (latent_dim, feat_dim)
    cross_corr = corr_matrix[:latent_dim, latent_dim:]

    # ============= Print matrix info =============
    print("Max correlation per latent variable:", np.max(cross_corr, axis=1))
    print("Min correlation per latent variable:", np.min(cross_corr, axis=1))
    print("Mean correlation per latent variable:", np.mean(cross_corr, axis=1))

    # ============= Table =============
    # Create a list to store results
    top_correlation_rows = []

    for latent_idx in range(latent_dim):
        correlations = cross_corr[latent_idx, :]  # Correlations for a single latent variable

        # Sort by correlation values
        sorted_indices = np.argsort(correlations)
        top_neg_indices = sorted_indices[:10]    # 10 smallest (most negative)
        top_pos_indices = sorted_indices[-10:]   # 10 largest (most positive)

        # Build lists of (feature_name, correlation)
        # Reverse top_pos_indices so the highest correlation is first
        top_pos_list = [(feature_names[idx], float(correlations[idx])) for idx in reversed(top_pos_indices)]
        top_neg_list = [(feature_names[idx], float(correlations[idx])) for idx in top_neg_indices]

        top_correlation_rows.append({
            "Latent Variable": latent_idx,
            "Top Positive Correlations": top_pos_list,
            "Top Negative Correlations": top_neg_list,
        })

    # Convert results to DataFrame
    top_correlation_df = pd.DataFrame(top_correlation_rows)

    # Print nicely using tabulate
    print(tabulate(top_correlation_df, headers='keys', tablefmt='psql'))

    return cross_corr, top_correlation_df


###################################
# Plot reconstructed data
###################################

def plot_recon(VAE_results, original_data):
    """
    Plots the reconstructed data against the original data for a specific modality.

    Parameters:
        results_VAE (dict): Dictionary containing VAE results, including 'recon_data'.
        final_data (pd.DataFrame): DataFrame containing the original data.
    """
    # Extract the reconstructed data and original data

    recon_batch = VAE_results['recon_data']
    x = original_data.drop(columns=['src_subject_id'])


    # If 'x' is your original data and is a DataFrame, convert it to a NumPy array of floats.
    if isinstance(x, pd.DataFrame):
        x_np = x.to_numpy(dtype=float)
    else:
        x_np = np.asarray(x, dtype=float)

    # For 'recon_batch', if it's a torch tensor, convert it to a NumPy array.
    if isinstance(recon_batch, torch.Tensor):
        recon_np = recon_batch.detach().cpu().numpy()
    else:
        recon_np = np.asarray(recon_batch, dtype=float)

    # Flatten the arrays
    x_flat = x_np.flatten()
    recon_flat = recon_np.flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(x_flat, recon_flat, alpha=0.5)
    plt.xlabel("Original Data")
    plt.ylabel("Reconstructed Data")
    plt.title("Original vs. Reconstructed Data")

    # Plot a diagonal (y = x) line as a reference for perfect reconstruction
    min_val = min(x_flat.min(), recon_flat.min())
    max_val = max(x_flat.max(), recon_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    plt.show()


###################################
# Plot latent variable variance
###################################

def get_latent_means(latent_output):
    """
    Extracts the latent means from the output of a VAE's encoder.

    The latent_output can be in one of the following forms:
      - A tuple of (latent_means, latent_logvar)
      - A dictionary with key 'mean'
      - Directly the latent means as a NumPy array or tensor

    Parameters:
    - latent_output: The output from the VAE encoder.

    Returns:
    - A NumPy array containing the latent means.
    """
    # If the output is a tuple, assume the first element is the latent means
    if isinstance(latent_output, tuple):
        latent_means = latent_output[0]
    # If it's a dictionary, extract the 'mean' key if available
    elif isinstance(latent_output, dict) and 'mean' in latent_output:
        latent_means = latent_output['mean']
    else:
        # Otherwise, assume it's directly the latent means
        latent_means = latent_output

    # Convert to a NumPy array if the latent means are in tensor form
    if hasattr(latent_means, 'numpy'):
        latent_means = latent_means.numpy()

    return np.array(latent_means)

def plot_latent_variance(latent_means):
    """
    Creates a scree plot for the latent variables of a VAE.

    Parameters:
    - latent_means: A NumPy array of shape (num_samples, latent_dim) containing the latent means
      for all samples.
    """
    # Compute variance for each latent dimension (axis=0: across samples)
    variances = np.var(latent_means, axis=0)

    # Normalize the variances to get a variance ratio similar to PCA explained variance
    explained_variance_ratio = variances / np.sum(variances)

    # Compute cumulative variance for visualization
    cumulative_variance = np.cumsum(explained_variance_ratio)

    latent_dim = len(explained_variance_ratio)
    plt.figure(figsize=(8, 5))

    # Bar plot for individual variance ratios
    plt.bar(range(1, latent_dim + 1), explained_variance_ratio, alpha=0.6, label='Individual Variance Ratio')

    # Line plot for cumulative variance
    plt.plot(range(1, latent_dim + 1), cumulative_variance, marker='o', color='r', label='Cumulative Variance')

    plt.xlabel('Latent Variable Index')
    plt.ylabel('Variance Ratio')
    plt.title('Scree Plot for VAE Latent Variables')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming your VAE encoder returns a latent mean vector for each sample.
# For instance, if you have a function get_latent_means() that returns a NumPy array:
# latent_means = get_latent_means(your_dataset)
# plot_latent_variance(latent_means)


###################################
# Plot variance PCA
###################################
def plot_scree(pca):
    """
    Plots a scree plot showing the explained variance ratio for each principal component.
    """
    num_components = len(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))

    # Bar plot for each component's explained variance
    plt.bar(range(1, num_components + 1), pca.explained_variance_ratio_,
            alpha=0.6, color='b', label='Individual Explained Variance')

    # Line plot for cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, num_components + 1), cumulative_variance,
             marker='o', color='r', label='Cumulative Explained Variance')

    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# Example usage:
# Assume pca is a fitted PCA object (e.g., from compute_pca(df, n_components=...))
# plot_scree(pca)

###################################
# Biplot PCA
###################################

def biplot(pca, df_numeric, feature_names=None):
    """
    Creates a biplot for the first two principal components.

    Parameters:
    - pca: A fitted PCA object.
    - df_numeric: The numeric dataframe used for PCA.
    - feature_names: Optional list of feature names. If None, the column names of df_numeric are used.
    """

    columns_to_exclude = ['src_subject_id', 'interview_date']
    df_numeric = df_numeric.drop(columns=[col for col in columns_to_exclude if col in df_numeric.columns])


    # Project the data onto the first two principal components
    scores = pca.transform(df_numeric)

    if feature_names is None:
        feature_names = df_numeric.columns

    plt.figure(figsize=(10, 8))

    # Scatter plot of the projected data (scores)
    plt.scatter(scores[:, 0], scores[:, 1], alpha=0.6, edgecolor='k')

    # Scale factor for the arrows to make them visible
    arrow_scale = 3.0

    # Plot the loadings as arrows
    for i, (comp1, comp2) in enumerate(zip(pca.components_[0], pca.components_[1])):
        plt.arrow(0, 0, comp1 * arrow_scale, comp2 * arrow_scale,
                  color='r', width=0.005, head_width=0.1)
        plt.text(comp1 * arrow_scale * 1.15, comp2 * arrow_scale * 1.15,
                 feature_names[i], color='r', ha='center', va='center')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Biplot')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming df_numeric is the numeric part of your dataframe used for PCA:
# biplot(pca, df_numeric)


###################################
# Differences in latent var between groups
###################################

def plot_cluster_latent(df):
    """
    Plot the mean latent variable values for each cluster in a heatmap format. Input is the dict
    """

    # Compute mean latent variable values per cluster
    means = df.groupby('Cluster').mean()

    # Plot heatmap
    plt.figure(figsize=(12, 6))  # Make it wider to fit labels
    plt.imshow(means, aspect="auto", cmap="coolwarm")  # Change 'coolwarm' if needed
    plt.colorbar(label="Latent Variable Value")
    plt.title("Cluster Means - Latent Variables", fontsize=14)

    # Label axes
    plt.xticks(np.arange(len(means.columns)), means.columns, rotation=45, ha="right", fontsize=10)
    plt.yticks(np.arange(len(means.index)), [f"Cluster {c}" for c in means.index], fontsize=10)

    plt.xlabel("Latent Variables")
    plt.ylabel("Clusters")
    plt.show()


###################################
# Plot parallel coordinates
###################################
def plot_parallel_coordinates(df):

    """Plot parallel coordinates."""
    for cluster_id in sorted(df["Cluster"].unique()):
        subset = df[df["Cluster"] == cluster_id]
        plt.plot(subset.drop(columns=["Cluster"]).T, alpha=0.3)  # Transpose to align variables on x-axis

    plt.xticks(range(len(df.columns) - 1), df.columns[:-1], rotation=45)
    plt.title("Parallel Coordinates - Latent Variables by Cluster, Clinical")
    plt.xlabel("Latent Variables")
    plt.ylabel("Value")
    plt.show()


###################################
# Plot heatmap of means of original variables per cluster
###################################

def plot_cluster_original(df):
    # Compute means per cluster
    """Plot cluster original."""
    means = df.groupby('Cluster').mean()

    # Create the figure with a larger size
    plt.figure(figsize=(40, 6))  # Wider figure to fit labels

    # Show heatmap
    plt.imshow(means, aspect='auto', cmap="coolwarm")  # 'coolwarm' adds better contrast
    plt.colorbar(label="Feature Value")  # Add a label for color meaning
    plt.title("Cluster Means Heatmap", fontsize=14)

    # Adjust X-axis
    plt.xticks(np.arange(len(means.columns)), means.columns, rotation=45, ha="right", fontsize=10)
    plt.xlabel("Features")

    # Adjust Y-axis
    plt.yticks(np.arange(len(means.index)), [f"Cluster {c}" for c in means.index], fontsize=10)
    plt.ylabel("Clusters")

    plt.grid(False)  # Remove grid lines for better readability
    plt.show()

def get_modality_columns_for_dummy_coding(
    meta: pd.DataFrame,
    selected_modalities: list | None,
    available_columns: list,
) -> list | None:
    """
    Resolve raw input columns that belong to modalities selected for dummy coding.
    Returns None when all columns should be dummy coded.
    """
    if selected_modalities is None:
        return None

    selected_set = {str(mod).strip() for mod in selected_modalities if str(mod).strip()}
    if not selected_set:
        return []

    meta_cols = {"ElementName", "Modality"}
    if not meta_cols.issubset(meta.columns):
        raise KeyError("meta must contain 'ElementName' and 'Modality' columns.")

    available_set = set(available_columns)
    selected_vars = (
        meta.loc[meta["Modality"].isin(selected_set), "ElementName"]
        .dropna()
        .astype(str)
        .tolist()
    )
    return [col for col in selected_vars if col in available_set]


def _collinearity_pair_level(abs_r):
    """Handle collinearity pair level."""
    if pd.isna(abs_r):
        return np.nan
    if abs_r >= 0.90:
        return "severe"
    if abs_r >= 0.70:
        return "moderate_high"
    if abs_r >= 0.50:
        return "moderate"
    return "low"


def _collinearity_vif_level(vif):
    """Handle collinearity vif level."""
    if pd.isna(vif):
        return np.nan
    if np.isinf(vif) or vif >= 10:
        return "critical"
    if vif >= 5:
        return "high"
    if vif >= 2.5:
        return "moderate"
    return "low"


def _collinearity_condition_level(condition_index):
    """Handle collinearity condition level."""
    if pd.isna(condition_index):
        return np.nan
    if np.isinf(condition_index) or condition_index >= 30:
        return "critical"
    if condition_index >= 10:
        return "moderate_strong"
    return "low"


def _prepare_collinearity_matrix(data, variables, min_pairwise_n=3):
    """Prepare collinearity matrix."""
    matrix = data[variables].apply(pd.to_numeric, errors="coerce")
    valid_variables = [
        column for column in matrix.columns
        if matrix[column].notna().sum() >= min_pairwise_n
        and matrix[column].nunique(dropna=True) > 1
    ]
    matrix = matrix[valid_variables].dropna(axis=0, how="all")
    if matrix.empty:
        return matrix

    matrix = matrix.fillna(matrix.median(numeric_only=True))
    valid_variables = [
        column for column in matrix.columns
        if matrix[column].nunique(dropna=True) > 1
    ]
    return matrix[valid_variables]


def _compute_collinearity_vif(matrix, domain_name):
    """Calculate collinearity vif."""
    if matrix.shape[1] < 2:
        return []

    values = matrix.to_numpy(dtype=float)
    values = (values - np.nanmean(values, axis=0)) / np.nanstd(values, axis=0, ddof=0)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    rows = []

    for index, variable in enumerate(matrix.columns):
        target = values[:, index]
        other_variables = np.delete(values, index, axis=1)
        design = np.column_stack([np.ones(other_variables.shape[0]), other_variables])
        try:
            coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
            fitted = design @ coefficients
            residual_sum_squares = float(np.sum((target - fitted) ** 2))
            total_sum_squares = float(np.sum((target - target.mean()) ** 2))
            r_squared = (
                1.0 - residual_sum_squares / total_sum_squares
                if total_sum_squares > 0 else np.nan
            )
            if np.isfinite(r_squared):
                r_squared = min(max(r_squared, 0.0), 1.0)
            tolerance = 1.0 - r_squared if np.isfinite(r_squared) else np.nan
            vif = 1.0 / tolerance if np.isfinite(tolerance) and tolerance > 0 else np.inf
        except np.linalg.LinAlgError:
            r_squared = np.nan
            tolerance = 0.0
            vif = np.inf

        rows.append({
            "domain": domain_name,
            "variable": variable,
            "vif": vif,
            "tolerance": tolerance,
            "r_squared_with_other_variables": r_squared,
            "vif_level": _collinearity_vif_level(vif),
        })
    return rows


def _compute_collinearity_condition_index(matrix):
    """Calculate collinearity condition index."""
    if matrix.shape[1] < 2:
        return np.nan, np.nan, np.nan

    values = matrix.to_numpy(dtype=float)
    values = (values - np.nanmean(values, axis=0)) / np.nanstd(values, axis=0, ddof=0)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    correlation = np.corrcoef(values, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)

    eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
    max_eigenvalue = float(np.max(eigenvalues)) if eigenvalues.size else np.nan
    positive_eigenvalues = eigenvalues[eigenvalues > 1e-12]
    min_positive_eigenvalue = (
        float(np.min(positive_eigenvalues)) if positive_eigenvalues.size else np.nan
    )
    condition_index = (
        float(np.sqrt(max_eigenvalue / min_positive_eigenvalue))
        if np.isfinite(max_eigenvalue) and np.isfinite(min_positive_eigenvalue)
        else np.nan
    )
    return condition_index, max_eigenvalue, min_positive_eigenvalue


def assess_domain_collinearity(
    preprocessed_modalities,
    id_cols=("src_subject_id", "phenotype"),
    correlation_methods=("pearson", "spearman"),
    high_corr_threshold=0.90,
    moderate_corr_threshold=0.70,
    min_pairwise_n=3,
):
    """Assess collinearity independently within preprocessed modality matrices."""
    if not isinstance(preprocessed_modalities, dict) or not preprocessed_modalities:
        raise ValueError("preprocessed_modalities must be a non-empty dictionary.")

    id_cols = set(id_cols)
    pair_rows = []
    vif_rows = []
    condition_rows = []
    summary_rows = []
    skipped_rows = []

    for domain_name, domain_data in preprocessed_modalities.items():
        variables = [column for column in domain_data.columns if column not in id_cols]
        matrix = _prepare_collinearity_matrix(
            domain_data,
            variables,
            min_pairwise_n=min_pairwise_n,
        )
        numeric_variables = matrix.columns.tolist()

        for variable in sorted(set(variables).difference(numeric_variables)):
            skipped_rows.append({
                "domain": domain_name,
                "variable": variable,
                "reason": "not numeric/coercible or insufficient variation",
            })

        correlation_counts = {method: 0 for method in correlation_methods}
        correlation_maxima = {method: np.nan for method in correlation_methods}
        correlation_means = {method: np.nan for method in correlation_methods}

        if len(numeric_variables) >= 2:
            numeric_source = domain_data[numeric_variables].apply(pd.to_numeric, errors="coerce")
            for method in correlation_methods:
                correlation = numeric_source.corr(method=method, min_periods=min_pairwise_n)
                absolute_correlation = correlation.abs()
                upper_mask = np.triu(np.ones(absolute_correlation.shape, dtype=bool), k=1)
                upper_values = absolute_correlation.where(upper_mask).stack().dropna()
                flagged_pairs = upper_values[
                    upper_values >= moderate_corr_threshold
                ].sort_values(ascending=False)

                correlation_counts[method] = int(
                    (upper_values >= high_corr_threshold).sum()
                )
                correlation_maxima[method] = (
                    upper_values.max() if len(upper_values) else np.nan
                )
                correlation_means[method] = (
                    upper_values.mean() if len(upper_values) else np.nan
                )

                for (variable_1, variable_2), absolute_value in flagged_pairs.items():
                    pair_rows.append({
                        "domain": domain_name,
                        "method": method,
                        "variable_1": variable_1,
                        "variable_2": variable_2,
                        "correlation": correlation.loc[variable_1, variable_2],
                        "abs_correlation": absolute_value,
                        "correlation_level": _collinearity_pair_level(absolute_value),
                    })

            domain_vif_rows = _compute_collinearity_vif(matrix, domain_name)
            vif_rows.extend(domain_vif_rows)
            condition_index, max_eigenvalue, min_eigenvalue = (
                _compute_collinearity_condition_index(matrix)
            )
        else:
            domain_vif_rows = []
            condition_index, max_eigenvalue, min_eigenvalue = np.nan, np.nan, np.nan

        condition_rows.append({
            "domain": domain_name,
            "condition_index": condition_index,
            "condition_index_level": _collinearity_condition_level(condition_index),
            "max_eigenvalue": max_eigenvalue,
            "min_positive_eigenvalue": min_eigenvalue,
        })
        finite_vifs = [row["vif"] for row in domain_vif_rows if np.isfinite(row["vif"])]
        finite_tolerances = [
            row["tolerance"] for row in domain_vif_rows
            if np.isfinite(row["tolerance"])
        ]
        summary_rows.append({
            "domain": domain_name,
            "n_variables_total": len(variables),
            "n_variables_numeric": len(numeric_variables),
            "n_severe_pearson_pairs_abs_r_ge_0_90": correlation_counts.get("pearson", 0),
            "n_severe_spearman_pairs_abs_r_ge_0_90": correlation_counts.get("spearman", 0),
            "max_abs_pearson": correlation_maxima.get("pearson", np.nan),
            "max_abs_spearman": correlation_maxima.get("spearman", np.nan),
            "mean_abs_pearson": correlation_means.get("pearson", np.nan),
            "mean_abs_spearman": correlation_means.get("spearman", np.nan),
            "max_vif": max(finite_vifs) if finite_vifs else np.nan,
            "min_tolerance": min(finite_tolerances) if finite_tolerances else np.nan,
            "n_variables_vif_ge_2_5": sum(vif >= 2.5 for vif in finite_vifs),
            "n_variables_vif_ge_5": sum(vif >= 5 for vif in finite_vifs),
            "n_variables_vif_ge_10": sum(vif >= 10 for vif in finite_vifs),
            "condition_index": condition_index,
            "condition_index_level": _collinearity_condition_level(condition_index),
        })

    return {
        "summary": pd.DataFrame(summary_rows).sort_values("domain").reset_index(drop=True),
        "correlation_pairs": (
            pd.DataFrame(pair_rows)
            .sort_values("abs_correlation", ascending=False)
            .reset_index(drop=True)
            if pair_rows else pd.DataFrame(columns=[
                "domain", "method", "variable_1", "variable_2", "correlation",
                "abs_correlation", "correlation_level",
            ])
        ),
        "vif": (
            pd.DataFrame(vif_rows)
            .sort_values("vif", ascending=False)
            .reset_index(drop=True)
            if vif_rows else pd.DataFrame(columns=[
                "domain", "variable", "vif", "tolerance",
                "r_squared_with_other_variables", "vif_level",
            ])
        ),
        "condition_index": pd.DataFrame(condition_rows).sort_values("domain").reset_index(drop=True),
        "skipped_variables": (
            pd.DataFrame(skipped_rows).sort_values(["domain", "variable"]).reset_index(drop=True)
            if skipped_rows else pd.DataFrame(columns=["domain", "variable", "reason"])
        ),
    }


def plot_domain_collinearity_heatmaps(
    preprocessed_modalities,
    output_dir=None,
    subject_id_column="src_subject_id",
    correlation_methods=("pearson", "spearman"),
    min_pairwise_n=3,
    file_prefix="collinearity",
    show=True,
):
    """Plot and optionally save correlation heatmaps for each supplied domain."""
    if not isinstance(preprocessed_modalities, dict) or not preprocessed_modalities:
        raise ValueError("preprocessed_modalities must be a non-empty dictionary.")

    output_dir = Path(output_dir).expanduser() if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for domain_name, domain_data in preprocessed_modalities.items():
        variables = [column for column in domain_data.columns if column != subject_id_column]
        matrix = _prepare_collinearity_matrix(
            domain_data,
            variables,
            min_pairwise_n=min_pairwise_n,
        )
        if matrix.shape[1] < 2:
            warnings.warn(
                f"{domain_name}: fewer than two usable variables; skipping collinearity heatmaps."
            )
            continue

        figure_size = max(8.0, min(24.0, 0.55 * matrix.shape[1] + 4.0))
        safe_domain_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(domain_name)).strip("_")
        for method in correlation_methods:
            correlation = matrix.corr(method=method, min_periods=min_pairwise_n)
            mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
            figure, axis = plt.subplots(figsize=(figure_size, figure_size))
            sns.heatmap(
                correlation,
                mask=mask,
                cmap="vlag",
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                linewidths=0.25,
                cbar_kws={"label": f"{method.title()} correlation", "shrink": 0.75},
                ax=axis,
            )
            axis.set_title(f"{domain_name}: {method.title()} correlation")
            axis.set_xticklabels(
                [display_feature_name(feature) for feature in correlation.columns],
                rotation=90,
                fontsize=8,
            )
            axis.set_yticklabels(
                [display_feature_name(feature) for feature in correlation.index],
                rotation=0,
                fontsize=8,
            )
            axis.tick_params(axis="x", labelrotation=90, labelsize=8)
            axis.tick_params(axis="y", labelrotation=0, labelsize=8)
            figure.tight_layout()

            if output_dir is not None:
                output_stem = output_dir / f"{file_prefix}_{safe_domain_name}_{method}_heatmap"
                for extension in ("png", "pdf"):
                    output_path = output_stem.with_suffix(f".{extension}")
                    figure.savefig(output_path, dpi=300, bbox_inches="tight")
                    saved_paths.append(output_path)
            if show:
                plt.show()
            else:
                plt.close(figure)

    return saved_paths


def _preprocessing_split_first(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    subject_id_column: str = 'src_subject_id',
    col_threshold: float = 0.5,
    row_threshold: float = 0.5,
    skew_threshold: float = 0.75,
    scaler_type: str = 'robust',
    modalities: list | None = None,
    dummy_code_modalities: list | None = None,
    mixed_categorical_modalities: list | None = None,
    impute_parea: bool = False,
    export_preprocessing_details: bool = False
):
    """
    Preprocess raw data for the clustering pipeline, one modality at a time.

    The pipeline intentionally splits by modality before feature engineering so
    each view keeps its own schema and missingness/imputation behavior. Standard
    numeric/object modalities are transformed, dummy-coded, scaled, imputed, and
    scaled again. Mixed categorical modalities are left closer to their raw form
    because FAMD/MCA handles encoding/scaling during dimensionality reduction.

    Returns
    -------
    ae_data, subject_id_list, dict_final
        Numeric arrays and aligned modality DataFrames for clustering.
    ae_data, subject_id_list, dict_final, preprocessing_details
        Same outputs plus reusable preprocessing metadata when requested.
    """
    if modalities is None:
        modalities = ['Internalising', 'Functioning', 'Cognition', 'Detachment', 'Psychoticism']
    mixed_categorical_set = set(mixed_categorical_modalities or [])

    # Step 1: split the raw table into modality-specific views according to the
    # metadata. From this point on each modality is processed independently.
    raw_modal_dict = extract_modalities(meta, df, subject_id_column=subject_id_column)
    raw_modal_dict = {mod: raw_modal_dict[mod].copy() for mod in modalities if mod in raw_modal_dict}
    if not raw_modal_dict:
        raise ValueError("No requested modalities present in the input data.")

    power_transform_details_by_modality = {}
    dummy_feature_columns_by_modality = {}
    initial_scaling_details_by_modality = {}
    modal_dict_clean = {}
    modal_dict_for_imputation = {}
    mixed_modal_dict = {}

    # Step 2: build a pre-imputation feature table for every modality. Standard
    # modalities become numeric here; mixed categorical modalities are preserved
    # for later FAMD/MCA dimensionality reduction.
    for mod, df_mod in raw_modal_dict.items():
        df_mod = df_mod.copy().reset_index(drop=True)
        if subject_id_column not in df_mod.columns:
            if subject_id_column not in df.columns:
                raise KeyError(f"Subject ID column '{subject_id_column}' not found.")
            df_mod.insert(0, subject_id_column, df[subject_id_column].reset_index(drop=True))

        subject_ids = df_mod[[subject_id_column]].reset_index(drop=True)
        data_only = df_mod.drop(columns=[subject_id_column])

        if mod in mixed_categorical_set:
            # Mixed-type modalities must keep their raw categorical/binary and
            # continuous values. FAMD/MCA does its own imputation, scaling, and
            # encoding downstream.
            df_scaled = pd.concat([subject_ids, data_only.reset_index(drop=True)], axis=1)
            dummy_feature_columns_by_modality[mod] = [c for c in df_scaled.columns if c != subject_id_column]
            if export_preprocessing_details:
                power_transform_details_by_modality[mod] = None
                initial_scaling_details_by_modality[mod] = None
        else:
            # Standard modality path: reduce skew, encode selected categorical
            # variables, and apply an initial scale before KNN imputation.
            if export_preprocessing_details:
                data_transformed, power_details = auto_power_transform(
                    data_only,
                    skew_threshold=skew_threshold,
                    return_details=True
                )
                power_transform_details_by_modality[mod] = power_details
            else:
                data_transformed = auto_power_transform(data_only, skew_threshold=skew_threshold)

            df_transformed = pd.concat([subject_ids, data_transformed.reset_index(drop=True)], axis=1)
            dummy_columns_to_encode = get_modality_columns_for_dummy_coding(
                meta,
                dummy_code_modalities,
                [c for c in df_transformed.columns if c != subject_id_column]
            )
            df_dummy = dummy_code(
                df_transformed,
                subject_id_column=subject_id_column,
                columns_to_encode=dummy_columns_to_encode
            )
            dummy_feature_columns_by_modality[mod] = [c for c in df_dummy.columns if c != subject_id_column]

            if export_preprocessing_details:
                df_scaled, scaling_details = scale_diverse_data(
                    df_dummy,
                    subject_id_column=subject_id_column,
                    scaler_type=scaler_type,
                    return_details=True
                )
                initial_scaling_details_by_modality[mod] = scaling_details
            else:
                df_scaled = scale_diverse_data(
                    df_dummy,
                    subject_id_column=subject_id_column,
                    scaler_type=scaler_type
                )
        modal_dict_clean[mod] = df_scaled.reset_index(drop=True)

    # Step 3: enforce complete modality presence when impute_parea=False. A
    # subject missing an entire view is removed from all views so PAREA receives
    # aligned multi-view samples.
    subjects_to_drop = set()
    if impute_parea is False:
        for mod, df_mod in modal_dict_clean.items():
            data_only = df_mod.drop(columns=[subject_id_column]) if subject_id_column in df_mod.columns else df_mod
            missing_mask = data_only.isna().all(axis=1)
            if missing_mask.any():
                subjects_to_drop.update(df_mod.loc[missing_mask, subject_id_column].tolist())
        if subjects_to_drop:
            warnings.warn(
                f"Dropping {len(subjects_to_drop)} participants missing a full modality across all views: {subjects_to_drop}"
            )
            for mod in modal_dict_clean:
                df_mod = modal_dict_clean[mod]
                modal_dict_clean[mod] = (
                    df_mod[~df_mod[subject_id_column].isin(subjects_to_drop)]
                    .reset_index(drop=True)
                )

    # Step 4: impute standard modalities. Mixed categorical modalities bypass
    # KNN imputation here because their downstream reducer handles missingness.
    for mod, df_mod in modal_dict_clean.items():
        if mod in mixed_categorical_set:
            mixed_modal_dict[mod] = df_mod.copy()
        else:
            modal_dict_for_imputation[mod] = df_mod.copy()

    imputation_reference_modalities = {
        mod: modal_dict_for_imputation[mod].copy()
        for mod in modal_dict_for_imputation
    } if export_preprocessing_details else None

    imputation_n_neighbors = 7
    dict_imputed = impute_data(
        modal_dict_for_imputation,
        subject_id_column=subject_id_column,
        n_neighbors=imputation_n_neighbors
    ) if modal_dict_for_imputation else {}
    dict_imputed.update(mixed_modal_dict)

    # Step 5: final scale after imputation and merge mixed modalities back into
    # the final modality dictionary.
    if export_preprocessing_details:
        dict_scaled_standard, final_scaling_details = scale_data(
            {mod: dict_imputed[mod] for mod in dict_imputed if mod not in mixed_categorical_set},
            subject_id_column=subject_id_column,
            scaler_type=scaler_type,
            return_details=True
        ) if any(mod not in mixed_categorical_set for mod in dict_imputed) else ({}, {})
        dict_final = dict_scaled_standard
        dict_final.update({mod: dict_imputed[mod] for mod in dict_imputed if mod in mixed_categorical_set})
    else:
        dict_final = scale_data(
            {mod: dict_imputed[mod] for mod in dict_imputed if mod not in mixed_categorical_set},
            subject_id_column=subject_id_column,
            scaler_type=scaler_type
        ) if any(mod not in mixed_categorical_set for mod in dict_imputed) else {}
        dict_final.update({mod: dict_imputed[mod] for mod in dict_imputed if mod in mixed_categorical_set})
        final_scaling_details = None

    # Step 6: canonical subject alignment. Every modality must have identical
    # subject order before clustering; this is the key invariant used throughout
    # full_pipeline.py.
    id_col = subject_id_column
    mods = [m for m in modalities if m in dict_final and not dict_final[m].empty]
    if not mods:
        raise ValueError("No requested modalities present after preprocessing.")

    id_lists = {m: dict_final[m][id_col].tolist() for m in mods}
    shared = set.intersection(*(set(v) for v in id_lists.values()))
    canonical = [sid for sid in id_lists[mods[0]] if sid in shared]

    for m in mods:
        dfm = dict_final[m]
        dict_final[m] = (
            dfm[dfm[id_col].isin(shared)]
            .set_index(id_col)
            .loc[canonical]
            .reset_index()
        )

    for m in mods[1:]:
        assert dict_final[m][id_col].tolist() == dict_final[mods[0]][id_col].tolist(), \
            f"Subject-ID order mismatch between {mods[0]} and {m}"

    # Step 7: build the legacy VAE-style payload and the subject-ID lists used
    # by PAREA for label alignment.
    subject_id_list = []
    for mod in modalities:
        if mod in dict_final and subject_id_column in dict_final[mod]:
            subject_id_list.append(dict_final[mod][subject_id_column].tolist())
        else:
            subject_id_list.append([])

    ae_data = convert_data_for_vae(dict_final, subject_id_column=subject_id_column)

    # Step 8: optional audit/reuse metadata. These details let validation,
    # follow-up, and reporting code apply the same preprocessing decisions later.
    if export_preprocessing_details:
        dummy_feature_columns = []
        for mod in mods:
            dummy_feature_columns.extend(dummy_feature_columns_by_modality.get(mod, []))
        preprocessing_details = {
            'subject_id_column': subject_id_column,
            'preprocessing_order': 'split_first_by_modality',
            'preprocessing_parameters': {
                'col_threshold': col_threshold,
                'row_threshold': row_threshold,
                'skew_threshold': skew_threshold,
                'scaler_type': scaler_type,
                'modalities_requested': list(modalities),
                'dummy_code_modalities': list(dummy_code_modalities) if dummy_code_modalities is not None else list(modalities),
                'mixed_categorical_modalities': list(mixed_categorical_modalities or []),
                'impute_parea': bool(impute_parea)
            },
            'subjects_dropped_full_missing_modality': sorted(list(subjects_to_drop)),
            'power_transform_by_modality': power_transform_details_by_modality,
            'dummy_feature_columns_by_modality': dummy_feature_columns_by_modality,
            'initial_scaling_by_modality': initial_scaling_details_by_modality,
            'imputation_n_neighbors': int(imputation_n_neighbors),
            'imputation_reference_modalities': imputation_reference_modalities,
            'final_modality_scaling': final_scaling_details,
            'modalities_in_output': list(dict_final.keys()),
            'n_subjects_after_alignment': len(canonical),
            'canonical_subject_ids': list(canonical),
            'feature_columns_per_modality': {
                mod: [c for c in dict_final[mod].columns if c != subject_id_column]
                for mod in dict_final
            },
            # Legacy fields retained for readers that check key presence.
            'power_transform': None,
            'dummy_feature_columns': dummy_feature_columns,
            'initial_scaling': None,
        }
        return ae_data, subject_id_list, dict_final, preprocessing_details

    return ae_data, subject_id_list, dict_final


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
    Public preprocessing entry point used by the pipeline and notebooks.

    The implementation lives in _preprocessing_split_first so full_pipeline.py,
    validation code, and notebook helpers all share the same preprocessing path.
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


def _apply_preprocessing_to_new_data_split_first(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    preprocessing_details: dict,
    subject_id_column: str = 'src_subject_id',
    imputation_mode: str = 'independent',
    align_modalities: bool = True,
):
    """
    Apply stored split-first preprocessing decisions to a new cohort/timepoint.

    This mirrors _preprocessing_split_first, but uses preprocessing_details to
    keep the new data on the same feature schema as the original run. It is used
    by validation, control, and longitudinal code after the discovery pipeline
    has already produced final preprocessing metadata.
    """
    params = preprocessing_details.get('preprocessing_parameters', {})
    modalities = preprocessing_details.get('modalities_in_output') or params.get('modalities_requested', [])
    impute_parea = bool(params.get('impute_parea', False))
    mixed_categorical_set = set(params.get('mixed_categorical_modalities', []) or [])

    # Step 1: split the new raw table by modality and load the stored per-modality
    # preprocessing artifacts from the discovery run.
    raw_modal_dict = extract_modalities(meta, df, subject_id_column=subject_id_column)
    modal_dict_clean = {}
    power_by_mod = preprocessing_details.get('power_transform_by_modality', {}) or {}
    dummy_by_mod = preprocessing_details.get('dummy_feature_columns_by_modality', {}) or {}
    scaling_by_mod = preprocessing_details.get('initial_scaling_by_modality', {}) or {}
    dummy_code_modalities = params.get('dummy_code_modalities', modalities)

    # Step 2: reproduce the training feature schema per modality. Missing dummy
    # columns are inserted as NaN so imputation can handle absent categories.
    for mod in modalities:
        if mod not in raw_modal_dict:
            continue
        df_mod = raw_modal_dict[mod].copy().reset_index(drop=True)
        if subject_id_column not in df_mod.columns:
            if subject_id_column not in df.columns:
                raise KeyError(f"Subject ID column '{subject_id_column}' not found.")
            df_mod.insert(0, subject_id_column, df[subject_id_column].reset_index(drop=True))

        subject_ids = df_mod[[subject_id_column]].reset_index(drop=True)
        data_only = df_mod.drop(columns=[subject_id_column])
        if mod in mixed_categorical_set:
            # Mixed categorical modalities keep their raw schema for downstream
            # mixed-type dimensionality reduction.
            df_dummy = pd.concat([subject_ids, data_only.reset_index(drop=True)], axis=1)
        else:
            # Standard modalities reuse the fitted Yeo-Johnson lambdas, dummy
            # schema, and initial scaling parameters saved during discovery.
            data_transformed = apply_power_transform_from_details(data_only, power_by_mod.get(mod))
            df_transformed = pd.concat([subject_ids, data_transformed.reset_index(drop=True)], axis=1)
            dummy_columns_to_encode = get_modality_columns_for_dummy_coding(
                meta,
                dummy_code_modalities,
                [c for c in df_transformed.columns if c != subject_id_column]
            )
            df_dummy = dummy_code(
                df_transformed,
                subject_id_column=subject_id_column,
                columns_to_encode=dummy_columns_to_encode
            )

        target_features = list(dummy_by_mod.get(mod, []))
        if target_features:
            missing_features = []
            for col in target_features:
                if col not in df_dummy.columns:
                    df_dummy[col] = np.nan
                    missing_features.append(col)
            if missing_features:
                warnings.warn(
                    f"{mod}: new data missing {len(missing_features)} training dummy/features; "
                    "filled with NaN and handled during imputation."
                )
            keep_cols = [subject_id_column] + target_features
            df_dummy = df_dummy[[c for c in keep_cols if c in df_dummy.columns]].copy()

        if mod in mixed_categorical_set:
            modal_dict_clean[mod] = df_dummy.reset_index(drop=True)
        else:
            modal_dict_clean[mod] = apply_scaling_from_details(
                df_dummy,
                scaling_by_mod.get(mod),
                subject_id_column=subject_id_column
            ).reset_index(drop=True)

    # Step 3: optionally enforce the same complete-modality subject rule as the
    # discovery pipeline.
    subjects_to_drop = set()
    if impute_parea is False and align_modalities:
        for mod, df_mod in modal_dict_clean.items():
            data_only = df_mod.drop(columns=[subject_id_column]) if subject_id_column in df_mod.columns else df_mod
            missing_mask = data_only.isna().all(axis=1)
            if missing_mask.any() and subject_id_column in df_mod.columns:
                subjects_to_drop.update(df_mod.loc[missing_mask, subject_id_column].tolist())
        if subjects_to_drop:
            for mod in modal_dict_clean:
                df_mod = modal_dict_clean[mod]
                modal_dict_clean[mod] = (
                    df_mod[~df_mod[subject_id_column].isin(subjects_to_drop)]
                    .reset_index(drop=True)
                )

    # Step 4: impute standard modalities. "reference" uses discovery data as the
    # imputer fit set; "independent" fits within the new cohort.
    if imputation_mode not in ('independent', 'reference'):
        raise ValueError("imputation_mode must be either 'independent' or 'reference'.")

    reference_modalities = preprocessing_details.get('imputation_reference_modalities')
    n_neighbors = int(preprocessing_details.get('imputation_n_neighbors', 7))
    standard_modalities = {
        mod: df_mod for mod, df_mod in modal_dict_clean.items()
        if mod not in mixed_categorical_set
    }
    mixed_modalities = {
        mod: df_mod for mod, df_mod in modal_dict_clean.items()
        if mod in mixed_categorical_set
    }

    if imputation_mode == 'reference' and reference_modalities:
        dict_imputed = impute_data_with_reference(
            standard_modalities,
            reference_modalities=reference_modalities,
            subject_id_column=subject_id_column,
            n_neighbors=n_neighbors
        ) if standard_modalities else {}
    else:
        if imputation_mode == 'reference' and not reference_modalities:
            warnings.warn(
                "Reference imputation requested but no reference modalities found; "
                "falling back to independent imputation."
            )
        dict_imputed = impute_data(
            standard_modalities,
            subject_id_column=subject_id_column,
            n_neighbors=n_neighbors
        ) if standard_modalities else {}
    dict_imputed.update(mixed_modalities)

    # Step 5: apply final per-modality scaling from discovery and merge mixed
    # modalities back in unchanged.
    dict_final = apply_scaling_to_modalities_from_details(
        {mod: df_mod for mod, df_mod in dict_imputed.items() if mod not in mixed_categorical_set},
        modality_scaling_details=preprocessing_details.get('final_modality_scaling', {}),
        subject_id_column=subject_id_column
    )
    dict_final.update({mod: df_mod for mod, df_mod in dict_imputed.items() if mod in mixed_categorical_set})

    # Step 6: align modalities to common subject IDs unless the caller explicitly
    # requests unaligned modality outputs.
    id_col = subject_id_column
    mods = [m for m in modalities if m in dict_final and not dict_final[m].empty]
    if not mods:
        raise ValueError("No requested modalities present after preprocessing.")

    if not align_modalities:
        subject_id_list = []
        for mod in modalities:
            if mod in dict_final and subject_id_column in dict_final[mod]:
                subject_id_list.append(dict_final[mod][subject_id_column].tolist())
            else:
                subject_id_list.append([])
        return {}, subject_id_list, dict_final

    id_lists = {m: dict_final[m][id_col].tolist() for m in mods}
    shared = set.intersection(*(set(v) for v in id_lists.values()))
    canonical = [sid for sid in id_lists[mods[0]] if sid in shared]

    for m in mods:
        dfm = dict_final[m]
        dict_final[m] = (
            dfm[dfm[id_col].isin(shared)]
            .set_index(id_col)
            .loc[canonical]
            .reset_index()
        )

    for m in mods[1:]:
        assert dict_final[m][id_col].tolist() == dict_final[mods[0]][id_col].tolist(), \
            f"Subject-ID order mismatch between {mods[0]} and {m}"

    subject_id_list = []
    for mod in modalities:
        if mod in dict_final and subject_id_column in dict_final[mod]:
            subject_id_list.append(dict_final[mod][subject_id_column].tolist())
        else:
            subject_id_list.append([])

    ae_data = convert_data_for_vae(dict_final, subject_id_column=subject_id_column)
    return ae_data, subject_id_list, dict_final


def apply_preprocessing_to_new_data(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    preprocessing_details: dict,
    subject_id_column: str = 'src_subject_id',
    imputation_mode: str = 'independent'
):
    """
    Apply previously fitted preprocessing (from preprocessing_details) to new data.
    Returns: ae_data, subject_id_list, dict_final
    """
    if preprocessing_details is None:
        raise ValueError("preprocessing_details is required.")

    if preprocessing_details.get('preprocessing_order') == 'split_first_by_modality':
        return _apply_preprocessing_to_new_data_split_first(
            df=df,
            meta=meta,
            preprocessing_details=preprocessing_details,
            subject_id_column=subject_id_column,
            imputation_mode=imputation_mode
        )

    params = preprocessing_details.get('preprocessing_parameters', {})
    modalities = preprocessing_details.get('modalities_in_output') or params.get('modalities_requested', [])
    impute_parea = bool(params.get('impute_parea', False))

    # 1) Power transform with fitted lambdas
    df_transformed = apply_power_transform_from_details(
        df,
        preprocessing_details.get('power_transform')
    )

    dummy_code_modalities = params.get('dummy_code_modalities', modalities)
    dummy_columns_to_encode = get_modality_columns_for_dummy_coding(
        meta,
        dummy_code_modalities,
        [c for c in df_transformed.columns if c != subject_id_column]
    )

    # 2) Dummy-code and align to training schema
    df_dummy = dummy_code(
        df_transformed,
        subject_id_column=subject_id_column,
        columns_to_encode=dummy_columns_to_encode
    )
    target_features = list(preprocessing_details.get('dummy_feature_columns', []))
    if target_features:
        missing_features = []
        for col in target_features:
            if col not in df_dummy.columns:
                # Keep as NaN so reference-based imputation can recover this feature;
                # forcing 0 here can create artificial constant columns in validation data.
                df_dummy[col] = np.nan
                missing_features.append(col)
        if missing_features:
            warnings.warn(
                f"New data missing {len(missing_features)} training dummy/features; "
                "filled with NaN and handled during reference-based imputation."
            )
        keep_cols = [subject_id_column] + target_features if subject_id_column in df_dummy.columns else target_features
        df_dummy = df_dummy[[c for c in keep_cols if c in df_dummy.columns]].copy()

    # 3) Apply initial global scaling
    df_scaled = apply_scaling_from_details(
        df_dummy,
        preprocessing_details.get('initial_scaling'),
        subject_id_column=subject_id_column
    )

    # 4) Split by modality + reattach IDs
    modal_dict = extract_modalities(meta, df_scaled, subject_id_column=subject_id_column)
    modal_dict_clean = {modality: modal_dict[modality] for modality in modalities if modality in modal_dict}
    for mod in modal_dict_clean:
        if subject_id_column not in modal_dict_clean[mod].columns and subject_id_column in df_scaled.columns:
            modal_dict_clean[mod][subject_id_column] = df_scaled[subject_id_column].loc[modal_dict_clean[mod].index]

    # 5) Apply same participant-level missingness rule
    subjects_to_drop = set()
    if impute_parea is False:
        for modality, df_mod in modal_dict_clean.items():
            data_only = df_mod.drop(columns=[subject_id_column]) if subject_id_column in df_mod.columns else df_mod
            missing_mask = data_only.isna().all(axis=1)
            if missing_mask.any() and subject_id_column in df_mod.columns:
                missing_ids = df_mod.loc[missing_mask, subject_id_column]
                subjects_to_drop.update(missing_ids.tolist())
        if subjects_to_drop:
            for modality in modal_dict_clean:
                df2 = modal_dict_clean[modality]
                if subject_id_column in df2.columns:
                    modal_dict_clean[modality] = (
                        df2[~df2[subject_id_column].isin(subjects_to_drop)]
                        .reset_index(drop=True)
                    )

    # 6) Impute missing values
    # independent: fit imputer on new cohort only (recommended for cohort comparison)
    # reference: fit on training (e.g., CHR) and transform new cohort
    if imputation_mode not in ('independent', 'reference'):
        raise ValueError("imputation_mode must be either 'independent' or 'reference'.")

    reference_modalities = preprocessing_details.get('imputation_reference_modalities')
    n_neighbors = int(preprocessing_details.get('imputation_n_neighbors', 7))
    if imputation_mode == 'reference' and reference_modalities:
        dict_imputed = impute_data_with_reference(
            modal_dict_clean,
            reference_modalities=reference_modalities,
            subject_id_column=subject_id_column,
            n_neighbors=n_neighbors
        )
    else:
        if imputation_mode == 'reference' and not reference_modalities:
            warnings.warn(
                "Reference imputation requested but no reference modalities found; "
                "falling back to independent imputation."
            )
        dict_imputed = impute_data(
            modal_dict_clean,
            subject_id_column=subject_id_column,
            n_neighbors=n_neighbors
        )

    # 7) Apply final per-modality scaling and enforce canonical alignment
    dict_final = apply_scaling_to_modalities_from_details(
        dict_imputed,
        modality_scaling_details=preprocessing_details.get('final_modality_scaling', {}),
        subject_id_column=subject_id_column
    )

    id_col = subject_id_column
    mods = [m for m in modalities if m in dict_final and not dict_final[m].empty]
    if not mods:
        raise ValueError("No requested modalities present after preprocessing.")
    id_lists = {m: dict_final[m][id_col].tolist() for m in mods}
    shared = set.intersection(*(set(v) for v in id_lists.values()))
    canonical = [sid for sid in id_lists[mods[0]] if sid in shared]

    for m in mods:
        dfm = dict_final[m]
        dict_final[m] = (
            dfm[dfm[id_col].isin(shared)]
            .set_index(id_col)
            .loc[canonical]
            .reset_index()
        )

    for m in mods[1:]:
        assert dict_final[m][id_col].tolist() == dict_final[mods[0]][id_col].tolist(), \
            f"Subject-ID order mismatch between {mods[0]} and {m}"

    subject_id_list = []
    for mod in modalities:
        if mod in dict_final and subject_id_column in dict_final[mod]:
            subject_id_list.append(dict_final[mod][subject_id_column].tolist())
        else:
            subject_id_list.append([])
    ae_data = convert_data_for_vae(dict_final, subject_id_column=subject_id_column)
    return ae_data, subject_id_list, dict_final


def regression_to_mean_longitudinal_sensitivity(
    wide_df,
    label_col,
    feature_names,
    subject_id_column="src_subject_id",
    baseline_suffix="_baseline",
    followup_suffix="_month2",
    min_group_n=5,
):
    """
    Assess whether baseline-defined subgroup differences persist beyond
    regression-to-the-mean patterns.
    """
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
    except Exception as exc:
        raise ImportError("statsmodels is required for regression-to-the-mean sensitivity tests.") from exc

    rows = []
    for feat in feature_names:
        base_col = f"{feat}{baseline_suffix}"
        follow_col = f"{feat}{followup_suffix}"
        required = [label_col, base_col, follow_col]
        if subject_id_column in wide_df.columns:
            required = [subject_id_column] + required
        missing = [c for c in required if c not in wide_df.columns]
        if missing:
            rows.append({
                "feature": feat,
                "status": "missing_columns",
                "missing_columns": ",".join(missing),
            })
            continue

        tmp = wide_df[required].dropna(subset=[label_col, base_col, follow_col]).copy()
        tmp = tmp.rename(columns={label_col: "group", base_col: "baseline", follow_col: "followup"})
        tmp["group"] = tmp["group"].astype(str)
        counts = tmp["group"].value_counts()
        if tmp.shape[0] < 2 * min_group_n or counts.shape[0] < 2 or counts.min() < min_group_n:
            rows.append({
                "feature": feat,
                "status": "too_few_subjects_or_groups",
                "n_subjects": int(tmp.shape[0]),
                "n_groups": int(counts.shape[0]),
                "min_group_n": int(counts.min()) if not counts.empty else 0,
            })
            continue

        base_means = tmp.groupby("group")["baseline"].mean()
        follow_means = tmp.groupby("group")["followup"].mean()
        baseline_gap = float(base_means.max() - base_means.min())
        followup_gap = float(follow_means.max() - follow_means.min())
        abs_gap_change = float(abs(followup_gap) - abs(baseline_gap))

        tmp["change"] = tmp["followup"] - tmp["baseline"]
        out = {
            "feature": feat,
            "status": "ok",
            "n_subjects": int(tmp.shape[0]),
            "n_groups": int(counts.shape[0]),
            "min_group_n": int(counts.min()),
            "baseline_gap_max_minus_min": baseline_gap,
            "followup_gap_max_minus_min": followup_gap,
            "absolute_gap_change_followup_minus_baseline": abs_gap_change,
            "gap_converges": bool(abs(followup_gap) < abs(baseline_gap)),
            "mean_change": float(tmp["change"].mean()),
        }

        try:
            model_follow = smf.ols("followup ~ baseline + C(group)", data=tmp).fit()
            anova_follow = sm.stats.anova_lm(model_follow, typ=2)
            out["baseline_adjusted_followup_group_p"] = float(anova_follow.loc["C(group)", "PR(>F)"])
            out["baseline_adjusted_followup_baseline_p"] = float(anova_follow.loc["baseline", "PR(>F)"])
        except Exception as exc:
            out["baseline_adjusted_followup_error"] = str(exc)

        try:
            model_change = smf.ols("change ~ baseline + C(group)", data=tmp).fit()
            anova_change = sm.stats.anova_lm(model_change, typ=2)
            out["baseline_adjusted_change_group_p"] = float(anova_change.loc["C(group)", "PR(>F)"])
            out["baseline_adjusted_change_baseline_p"] = float(anova_change.loc["baseline", "PR(>F)"])
        except Exception as exc:
            out["baseline_adjusted_change_error"] = str(exc)

        rows.append(out)

    return pd.DataFrame(rows)


def analyze_cluster_change_over_time(
    baseline_df,
    month2_df,
    labels_df,
    subject_id_column='src_subject_id',
    label_col='labels',
    top_n_features=15,
):
    """
    Compare baseline clustering labels to month_2 by:
    1) assigning month_2 observations to the nearest baseline cluster centroid,
    2) summarizing subgroup transition rates,
    3) estimating within-subject movement in baseline PCA space, and
    4) summarizing mean feature change per baseline cluster.
    """

    required_cols = {subject_id_column, label_col}
    if not required_cols.issubset(labels_df.columns):
        missing = sorted(required_cols.difference(labels_df.columns))
        raise KeyError(f"labels_df is missing required columns: {missing}")

    baseline_features = [
        c for c in baseline_df.columns
        if c != subject_id_column and c in month2_df.columns
        and pd.api.types.is_numeric_dtype(baseline_df[c])
        and pd.api.types.is_numeric_dtype(month2_df[c])
    ]
    if not baseline_features:
        raise ValueError("No shared numeric features found between baseline_df and month2_df.")

    labels_use = (
        labels_df[[subject_id_column, label_col]]
        .dropna(subset=[label_col])
        .drop_duplicates(subset=[subject_id_column])
        .copy()
    )

    baseline_use = baseline_df[[subject_id_column] + baseline_features].copy()
    month2_use = month2_df[[subject_id_column] + baseline_features].copy()

    merged = (
        labels_use
        .merge(baseline_use, on=subject_id_column, how='inner')
        .merge(month2_use, on=subject_id_column, how='inner', suffixes=('_baseline', '_month2'))
    )

    baseline_cols = [f'{c}_baseline' for c in baseline_features]
    month2_cols = [f'{c}_month2' for c in baseline_features]
    merged = merged.dropna(subset=baseline_cols + month2_cols).copy()
    if merged.empty:
        raise ValueError("No paired baseline/month_2 rows remain after alignment and NA filtering.")

    X_baseline = merged[baseline_cols].to_numpy(dtype=float)
    X_month2 = merged[month2_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_baseline_z = scaler.fit_transform(X_baseline)
    scale = np.where(np.asarray(scaler.scale_) == 0, 1.0, np.asarray(scaler.scale_))
    X_month2_z = (X_month2 - np.asarray(scaler.mean_)) / scale

    merged['baseline_cluster'] = merged[label_col].values
    cluster_order = np.sort(pd.unique(merged['baseline_cluster']))

    centroid_df = (
        pd.DataFrame(X_baseline_z, columns=baseline_features, index=merged.index)
        .assign(baseline_cluster=merged['baseline_cluster'].values)
        .groupby('baseline_cluster')
        .mean()
        .reindex(cluster_order)
    )
    centroids = centroid_df.to_numpy(dtype=float)

    baseline_cluster_positions = pd.Index(cluster_order).get_indexer(merged['baseline_cluster'])
    if np.any(baseline_cluster_positions < 0):
        raise ValueError("Failed to align baseline clusters to centroid order.")

    month2_distances = np.linalg.norm(
        X_month2_z[:, None, :] - centroids[None, :, :],
        axis=2
    )
    month2_cluster = cluster_order[month2_distances.argmin(axis=1)]
    merged['month2_cluster'] = month2_cluster
    merged['switched_cluster'] = merged['baseline_cluster'] != merged['month2_cluster']

    own_baseline_centroids = centroids[baseline_cluster_positions]
    merged['distance_from_baseline_centroid_baseline'] = np.linalg.norm(
        X_baseline_z - own_baseline_centroids,
        axis=1
    )
    merged['distance_from_baseline_centroid_month2'] = np.linalg.norm(
        X_month2_z - own_baseline_centroids,
        axis=1
    )
    merged['distance_change'] = (
        merged['distance_from_baseline_centroid_month2']
        - merged['distance_from_baseline_centroid_baseline']
    )

    pca = PCA(n_components=min(2, len(baseline_features)))
    baseline_scores = pca.fit_transform(X_baseline_z)
    month2_scores = pca.transform(X_month2_z)
    merged['pc1_baseline'] = baseline_scores[:, 0]
    merged['pc1_month2'] = month2_scores[:, 0]
    merged['pc1_change'] = merged['pc1_month2'] - merged['pc1_baseline']
    if baseline_scores.shape[1] > 1:
        merged['pc2_baseline'] = baseline_scores[:, 1]
        merged['pc2_month2'] = month2_scores[:, 1]

    delta_df = pd.DataFrame(
        X_month2_z - X_baseline_z,
        columns=baseline_features,
        index=merged.index,
    )
    cluster_feature_change = (
        delta_df.assign(baseline_cluster=merged['baseline_cluster'].values)
        .groupby('baseline_cluster')
        .mean()
        .reindex(cluster_order)
    )

    transition_counts = pd.crosstab(
        merged['baseline_cluster'],
        merged['month2_cluster'],
        dropna=False
    ).reindex(index=cluster_order, columns=cluster_order, fill_value=0)
    transition_pct = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0.0)

    cluster_summary = (
        merged.groupby('baseline_cluster')
        .agg(
            n_subjects=(subject_id_column, 'nunique'),
            n_switched=('switched_cluster', 'sum'),
            switch_rate=('switched_cluster', 'mean'),
            mean_pc1_change=('pc1_change', 'mean'),
            sd_pc1_change=('pc1_change', 'std'),
            mean_distance_change=('distance_change', 'mean'),
            sd_distance_change=('distance_change', 'std'),
        )
        .reindex(cluster_order)
    )
    cluster_summary['switch_rate'] = cluster_summary['switch_rate'].fillna(0.0)

    feature_rank = cluster_feature_change.abs().max(axis=0).sort_values(ascending=False)
    top_features = feature_rank.head(min(top_n_features, len(feature_rank))).index.tolist()

    return {
        'paired_df': merged,
        'feature_list': baseline_features,
        'cluster_order': cluster_order,
        'transition_counts': transition_counts,
        'transition_pct': transition_pct,
        'cluster_summary': cluster_summary,
        'cluster_feature_change': cluster_feature_change,
        'top_feature_changes': cluster_feature_change.loc[:, top_features],
        'pca_explained_variance_ratio': pca.explained_variance_ratio_,
    }


def plot_cluster_change_over_time(
    analysis_results,
    output_dir=None,
    prefix='baseline_to_month2',
):
    """
    Visualize how baseline clusters change at month_2 using:
    - a transition heatmap,
    - a paired PC1 change plot,
    - a heatmap of mean standardized feature change.
    """

    paired_df = analysis_results['paired_df']
    cluster_order = analysis_results['cluster_order']
    transition_pct = analysis_results['transition_pct']
    top_feature_changes = analysis_results['top_feature_changes']

    palette = modality_cluster_palette(cluster_order)
    figs = {}

    fig1, ax1 = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(
        transition_pct,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Row proportion'},
        ax=ax1,
    )
    ax1.set_title('Baseline cluster to month_2 transition')
    ax1.set_xlabel('Assigned cluster at month_2')
    ax1.set_ylabel('Baseline cluster')
    fig1.tight_layout()
    figs['transition_heatmap'] = fig1

    fig2, ax2 = plt.subplots(figsize=(8.5, 6))
    x_positions = np.array([0, 1])
    for cluster in cluster_order:
        subset = paired_df.loc[paired_df['baseline_cluster'] == cluster]
        color = palette[cluster]
        for _, row in subset.iterrows():
            ax2.plot(
                x_positions,
                [row['pc1_baseline'], row['pc1_month2']],
                color=color,
                alpha=0.12,
                linewidth=1,
            )
        means = subset[['pc1_baseline', 'pc1_month2']].mean()
        ax2.plot(
            x_positions,
            means.values,
            color=color,
            linewidth=3,
            marker='o',
            label=f'Cluster {cluster} mean',
        )
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(['Baseline', 'Month 2'])
    ax2.set_ylabel('PC1 score in baseline feature space')
    ax2.set_title('Within-subject movement from baseline to month_2')
    ax2.grid(axis='y', alpha=0.2)
    ax2.legend(frameon=False)
    sns.despine(ax=ax2)
    fig2.tight_layout()
    figs['pc1_change_plot'] = fig2

    heatmap_width = max(8, 0.45 * max(1, top_feature_changes.shape[1]) + 4)
    fig3, ax3 = plt.subplots(figsize=(heatmap_width, 4.8))
    sns.heatmap(
        top_feature_changes,
        cmap='coolwarm',
        center=0,
        cbar_kws={'label': 'Mean z-scored change (month_2 - baseline)'},
        ax=ax3,
    )
    ax3.set_title('Largest mean feature changes within each baseline cluster')
    ax3.set_xlabel('Feature')
    ax3.set_ylabel('Baseline cluster')
    fig3.tight_layout()
    figs['feature_change_heatmap'] = fig3

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        _save_longitudinal_matplotlib_image(fig1, os.path.join(output_dir, f'{prefix}_transition_heatmap.png'), dpi=300, bbox_inches='tight')
        _save_longitudinal_matplotlib_image(fig2, os.path.join(output_dir, f'{prefix}_pc1_change.png'), dpi=300, bbox_inches='tight')
        _save_longitudinal_matplotlib_image(fig3, os.path.join(output_dir, f'{prefix}_feature_change_heatmap.png'), dpi=300, bbox_inches='tight')

    return figs


import pandas as pd

def align_month2_dicts_to_clustering_features(
    dict_final_disc_m2: dict,
    dict_final_test_m2: dict,
    final_metrics: dict,
    subject_id_column: str,
    modalities: list,
    verbose: bool = True,
):
    """
    Filters month2 modality dicts so they only include variables that were used in clustering.
    Uses final_metrics['data'][modality] as the authority for the feature list and order.

    Returns
    -------
    dict_final_disc_m2_filt, dict_final_test_m2_filt, report_df
    """

    if "data" not in final_metrics:
        raise KeyError("final_metrics must contain key 'data' (a dict of modality->df used for clustering).")

    clustering_dict = final_metrics["data"]
    report_rows = []

    disc_filt = {}
    test_filt = {}

    for modality in modalities:
        if modality not in clustering_dict:
            raise KeyError(f"Modality '{modality}' not found in final_metrics['data'].")

        if modality not in dict_final_disc_m2:
            raise KeyError(f"Modality '{modality}' not found in dict_final_disc_m2.")
        if modality not in dict_final_test_m2:
            raise KeyError(f"Modality '{modality}' not found in dict_final_test_m2.")

        df_clust = clustering_dict[modality]
        df_disc = dict_final_disc_m2[modality]
        df_test = dict_final_test_m2[modality]

        # --- clustering features as authority (and order) ---
        clust_cols = list(df_clust.columns)
        if subject_id_column not in clust_cols:
            raise KeyError(
                f"subject_id_column='{subject_id_column}' not found in final_metrics['data'][{modality}].columns"
            )

        clust_feat_cols = [c for c in clust_cols if c != subject_id_column]

        # --- filter month2 disc/test to those features (intersection), preserving clustering order ---
        disc_present = [c for c in clust_feat_cols if c in df_disc.columns]
        test_present = [c for c in clust_feat_cols if c in df_test.columns]

        # Extra columns currently in month2 dicts but not used in clustering
        disc_extra = [c for c in df_disc.columns if c != subject_id_column and c not in clust_feat_cols]
        test_extra = [c for c in df_test.columns if c != subject_id_column and c not in clust_feat_cols]

        # Missing columns that clustering expects but month2 lacks
        disc_missing = [c for c in clust_feat_cols if c not in df_disc.columns]
        test_missing = [c for c in clust_feat_cols if c not in df_test.columns]

        # Build filtered dfs (keep subject id + ordered features)
        disc_filt[modality] = df_disc[[subject_id_column] + disc_present].copy()
        test_filt[modality] = df_test[[subject_id_column] + test_present].copy()

        report_rows.append({
            "modality": modality,
            "n_clustering_features": len(clust_feat_cols),
            "n_disc_features_kept": len(disc_present),
            "n_test_features_kept": len(test_present),
            "n_disc_extra_dropped": len(disc_extra),
            "n_test_extra_dropped": len(test_extra),
            "n_disc_missing_vs_clustering": len(disc_missing),
            "n_test_missing_vs_clustering": len(test_missing),
            "disc_missing_examples": disc_missing[:10],
            "test_missing_examples": test_missing[:10],
        })

        if verbose:
            print(f"\n[{modality}]")
            print(f"  Clustering features: {len(clust_feat_cols)}")
            print(f"  Disc kept: {len(disc_present)} | dropped extras: {len(disc_extra)} | missing: {len(disc_missing)}")
            print(f"  Test kept: {len(test_present)} | dropped extras: {len(test_extra)} | missing: {len(test_missing)}")

    report_df = pd.DataFrame(report_rows)

    # Optional: enforce “same feature set in disc and test” per modality
    # (If you *require* exact equality, you can assert here.)
    # for modality in modalities:
    #     disc_feats = [c for c in disc_filt[modality].columns if c != subject_id_column]
    #     test_feats = [c for c in test_filt[modality].columns if c != subject_id_column]
    #     assert disc_feats == test_feats, f"Feature mismatch after filtering in modality: {modality}"

    return disc_filt, test_filt, report_df

def apply_preprocessing_to_month2(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    preprocessing_details: dict,
    subject_id_column: str = 'src_subject_id',
    imputation_mode: str = 'independent',
    align_modalities: bool = True,
):
    """
    Apply baseline-fitted preprocessing to month-2 data.
    Aligns each modality to the training clustering feature schema before imputation,
    and preserves columns that are structurally missing at month-2 as NaN.
    """
    if preprocessing_details is None:
        raise ValueError("preprocessing_details is required.")

    if preprocessing_details.get('preprocessing_order') == 'split_first_by_modality':
        return _apply_preprocessing_to_new_data_split_first(
            df=df,
            meta=meta,
            preprocessing_details=preprocessing_details,
            subject_id_column=subject_id_column,
            imputation_mode=imputation_mode,
            align_modalities=align_modalities,
        )

    params = preprocessing_details.get('preprocessing_parameters', {})
    modalities = preprocessing_details.get('modalities_in_output') or params.get('modalities_requested', [])
    impute_parea = bool(params.get('impute_parea', False))
    training_feature_cols = preprocessing_details.get('feature_columns_per_modality', {})

    df_transformed = apply_power_transform_from_details(
        df,
        preprocessing_details.get('power_transform')
    )

    dummy_code_modalities = params.get('dummy_code_modalities', modalities)
    dummy_columns_to_encode = get_modality_columns_for_dummy_coding(
        meta,
        dummy_code_modalities,
        [c for c in df_transformed.columns if c != subject_id_column]
    )

    df_dummy = dummy_code(
        df_transformed,
        subject_id_column=subject_id_column,
        columns_to_encode=dummy_columns_to_encode
    )
    target_features = list(preprocessing_details.get('dummy_feature_columns', []))
    if target_features:
        missing_features = []
        for col in target_features:
            if col not in df_dummy.columns:
                df_dummy[col] = np.nan
                missing_features.append(col)
        if missing_features:
            warnings.warn(
                f"New data missing {len(missing_features)} training dummy/features; "
                "filled with NaN before modality-level alignment."
            )
        keep_cols = [subject_id_column] + target_features if subject_id_column in df_dummy.columns else target_features
        df_dummy = df_dummy[[c for c in keep_cols if c in df_dummy.columns]].copy()

    df_scaled = apply_scaling_from_details(
        df_dummy,
        preprocessing_details.get('initial_scaling'),
        subject_id_column=subject_id_column
    )

    modal_dict = extract_modalities(meta, df_scaled, subject_id_column=subject_id_column)
    modal_dict_clean = {modality: modal_dict[modality] for modality in modalities if modality in modal_dict}
    for mod in modal_dict_clean:
        if subject_id_column not in modal_dict_clean[mod].columns and subject_id_column in df_scaled.columns:
            modal_dict_clean[mod][subject_id_column] = df_scaled[subject_id_column].loc[modal_dict_clean[mod].index]

    all_missing_cols_by_modality = {}
    for modality, df_mod in modal_dict_clean.items():
        target_cols = training_feature_cols.get(modality)
        if not target_cols:
            target_cols = [c for c in df_mod.columns if c != subject_id_column]

        ids = df_mod[[subject_id_column]].reset_index(drop=True)
        feature_df = df_mod.drop(columns=[subject_id_column]).copy()

        missing_cols = [c for c in target_cols if c not in feature_df.columns]
        extra_cols = [c for c in feature_df.columns if c not in target_cols]

        feature_df = feature_df.reindex(columns=target_cols)
        all_missing_cols = [c for c in target_cols if feature_df[c].isna().all()]
        all_missing_cols_by_modality[modality] = all_missing_cols

        if missing_cols or extra_cols or all_missing_cols:
            warnings.warn(
                f"{modality}: aligned to training schema before imputation "
                f"(missing={len(missing_cols)}, extra={len(extra_cols)}, all_missing={len(all_missing_cols)})."
            )

        modal_dict_clean[modality] = pd.concat([ids, feature_df.reset_index(drop=True)], axis=1)

    subjects_to_drop = set()
    if impute_parea is False and align_modalities:
        for modality, df_mod in modal_dict_clean.items():
            data_only = df_mod.drop(columns=[subject_id_column]) if subject_id_column in df_mod.columns else df_mod
            missing_mask = data_only.isna().all(axis=1)
            if missing_mask.any() and subject_id_column in df_mod.columns:
                missing_ids = df_mod.loc[missing_mask, subject_id_column]
                subjects_to_drop.update(missing_ids.tolist())
        if subjects_to_drop:
            for modality in modal_dict_clean:
                df2 = modal_dict_clean[modality]
                if subject_id_column in df2.columns:
                    modal_dict_clean[modality] = (
                        df2[~df2[subject_id_column].isin(subjects_to_drop)]
                        .reset_index(drop=True)
                    )

    if imputation_mode not in ('independent', 'reference'):
        raise ValueError("imputation_mode must be either 'independent' or 'reference'.")

    reference_modalities = preprocessing_details.get('imputation_reference_modalities')
    n_neighbors = int(preprocessing_details.get('imputation_n_neighbors', 7))
    if imputation_mode == 'reference' and reference_modalities:
        dict_imputed = impute_data_with_reference(
            modal_dict_clean,
            reference_modalities=reference_modalities,
            subject_id_column=subject_id_column,
            n_neighbors=n_neighbors
        )
    else:
        if imputation_mode == 'reference' and not reference_modalities:
            warnings.warn(
                "Reference imputation requested but no reference modalities found; "
                "falling back to independent imputation."
            )
        dict_imputed = impute_data(
            modal_dict_clean,
            subject_id_column=subject_id_column,
            n_neighbors=n_neighbors
        )

    for modality, cols in all_missing_cols_by_modality.items():
        if modality in dict_imputed and cols:
            dict_imputed[modality].loc[:, cols] = np.nan

    dict_final = apply_scaling_to_modalities_from_details(
        dict_imputed,
        modality_scaling_details=preprocessing_details.get('final_modality_scaling', {}),
        subject_id_column=subject_id_column
    )

    for modality, cols in all_missing_cols_by_modality.items():
        if modality in dict_final and cols:
            dict_final[modality].loc[:, cols] = np.nan

    id_col = subject_id_column
    mods = [m for m in modalities if m in dict_final and not dict_final[m].empty]
    if not mods:
        raise ValueError("No requested modalities present after preprocessing.")

    if not align_modalities:
        subject_id_list = []
        for mod in modalities:
            if mod in dict_final and subject_id_column in dict_final[mod]:
                subject_id_list.append(dict_final[mod][subject_id_column].tolist())
            else:
                subject_id_list.append([])
        return {}, subject_id_list, dict_final

    id_lists = {m: dict_final[m][id_col].tolist() for m in mods}
    shared = set.intersection(*(set(v) for v in id_lists.values()))
    canonical = [sid for sid in id_lists[mods[0]] if sid in shared]

    for m in mods:
        dfm = dict_final[m]
        dict_final[m] = (
            dfm[dfm[id_col].isin(shared)]
            .set_index(id_col)
            .loc[canonical]
            .reset_index()
        )

    for m in mods[1:]:
        assert dict_final[m][id_col].tolist() == dict_final[mods[0]][id_col].tolist(), \
            f"Subject-ID order mismatch between {mods[0]} and {m}"

    subject_id_list = []
    for mod in modalities:
        if mod in dict_final and subject_id_column in dict_final[mod]:
            subject_id_list.append(dict_final[mod][subject_id_column].tolist())
        else:
            subject_id_list.append([])
    ae_data = convert_data_for_vae(dict_final, subject_id_column=subject_id_column)
    return ae_data, subject_id_list, dict_final


def _save_longitudinal_matplotlib_image(fig, png_path, **savefig_kwargs):
    """Save a longitudinal matplotlib figure as PNG, PDF, and fully editable SVG."""
    fig.savefig(png_path, **savefig_kwargs)
    pdf_path = os.path.splitext(png_path)[0] + ".pdf"
    pdf_kwargs = dict(savefig_kwargs)
    pdf_kwargs.pop("dpi", None)
    fig.savefig(pdf_path, format="pdf", **pdf_kwargs)
    svg_path = os.path.splitext(png_path)[0] + ".svg"
    svg_kwargs = dict(savefig_kwargs)
    svg_kwargs.pop("dpi", None)
    for artist in fig.findobj():
        if hasattr(artist, "set_rasterized"):
            artist.set_rasterized(False)
    with plt.rc_context({
        "image.composite_image": False,
        "svg.fonttype": "none",
        "svg.hashsalt": "",
    }):
        fig.savefig(svg_path, format="svg", **svg_kwargs)
    return {"png": png_path, "pdf": pdf_path, "svg": svg_path}


def find_longitudinal_month_files(
    data_dirs=None,
    months=(1, 2, 3, 4, 5),
    filename_template="data_month{month}.csv",
    filename_templates=None,
):
    """Find available longitudinal data files for month 1-5 analyses."""
    templates = list(filename_templates or [])
    if filename_template and filename_template not in templates:
        templates.insert(0, filename_template)
    for template in ("data_month{month}_simpleclust.csv",):
        if template not in templates:
            templates.append(template)

    default_dirs = [
        os.environ.get("LONGITUDINAL_DATA_DIR"),
        "path/to/prospect_data",
        "path/to/multiclust_data",
        "path/to/simpleclust_data",
        "path/to/restricted_data",
    ]
    search_dirs = []
    for directory in list(data_dirs or []) + default_dirs:
        if directory and directory not in search_dirs:
            search_dirs.append(directory)

    candidates = []
    for dir_idx, directory in enumerate(search_dirs):
        for template_idx, template in enumerate(templates):
            files = {
                int(month): os.path.join(directory, template.format(month=month))
                for month in months
                if os.path.exists(os.path.join(directory, template.format(month=month)))
            }
            if files:
                candidates.append((len(files), -dir_idx, -template_idx, files))
    if not candidates:
        return {}
    # Prefer a single folder with the most requested months available. This
    # avoids accidentally mixing cohorts when several data folders exist.
    return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


def _read_longitudinal_months(month_files, categorical_columns=None):
    """Read longitudinal months."""
    data_by_month = {}
    categorical_columns = list(categorical_columns or [])
    for month, path in sorted(month_files.items()):
        df = pd.read_csv(path)
        df.replace([-3, -300, "-3", "-300"], np.nan, inplace=True)
        cols_to_convert = [col for col in categorical_columns if col in df.columns]
        if cols_to_convert:
            df[cols_to_convert] = df[cols_to_convert].astype(str)
            df.replace("nan", np.nan, inplace=True)
        data_by_month[int(month)] = df
    return data_by_month


def _apply_longitudinal_followup_aliases(df, expected_features=None):
    """
    Copy directly comparable follow-up fields onto baseline-compatible names.

    PROSPECT Suicidality follow-up fields use a `chrcssrsfu_` prefix while the
    corresponding baseline schema uses `chrcssrsb_`. Only exact suffix matches
    are aliased here; semantically different items are left untouched.
    """
    if df is None or df.empty:
        return df
    expected = set(expected_features or [])
    if not expected:
        return df

    out = df.copy()
    prefix_aliases = (
        ("chrcssrsfu_", "chrcssrsb_"),
    )
    for source_prefix, target_prefix in prefix_aliases:
        for source_col in [c for c in out.columns if c.startswith(source_prefix)]:
            target_col = f"{target_prefix}{source_col[len(source_prefix):]}"
            if target_col in expected and target_col not in out.columns:
                out[target_col] = out[source_col]
    return out


def _merge_modalities_for_longitudinal(data_dict, subject_id_column="src_subject_id"):
    """Merge modalities for longitudinal."""
    merged = None
    for modality, df in data_dict.items():
        if df is None or subject_id_column not in df.columns:
            continue
        feature_cols = [
            c for c in df.columns
            if c != subject_id_column and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not feature_cols:
            continue
        part = df[[subject_id_column] + feature_cols].copy()
        part = part.rename(columns={c: f"{modality}__{c}" for c in feature_cols})
        merged = part if merged is None else merged.merge(part, on=subject_id_column, how="inner")
    if merged is None:
        return pd.DataFrame(columns=[subject_id_column])
    return merged


def _build_mixed_longitudinal_component_frames(
    baseline_df,
    followup_by_month,
    subject_id_column="src_subject_id",
    component_name="mixed_component1",
):
    """
    Fit one FAMD-like component on baseline mixed data and project all follow-ups.

    The longitudinal mixed-model and cluster-change code expects numeric features.
    For mixed domains, using a baseline-fitted shared component keeps categorical
    information in the analysis without imposing arbitrary ordinal codes.
    """
    if baseline_df is None or baseline_df.empty or not followup_by_month:
        return baseline_df, followup_by_month

    observed_followup_features = set()
    for df_month in followup_by_month.values():
        if df_month is None or df_month.empty:
            continue
        observed_followup_features.update(
            c for c in df_month.columns
            if c != subject_id_column and not df_month[c].isna().all()
        )
    feature_cols = [
        c for c in baseline_df.columns
        if c != subject_id_column and c in observed_followup_features
    ]
    if not feature_cols:
        return baseline_df, followup_by_month

    combined_followups = []
    month_lengths = []
    for month, df_month in sorted(followup_by_month.items()):
        if df_month is None or df_month.empty or subject_id_column not in df_month.columns:
            continue
        month_use = df_month[[subject_id_column] + [c for c in feature_cols if c in df_month.columns]].copy()
        month_use = month_use.reindex(columns=[subject_id_column] + feature_cols)
        combined_followups.append(month_use)
        month_lengths.append((month, len(month_use)))

    if not combined_followups:
        return baseline_df, followup_by_month

    combined = pd.concat(combined_followups, ignore_index=True)
    z_base, z_follow = _fit_transform_famd_like(
        baseline_df[[subject_id_column] + feature_cols].copy(),
        combined,
        subject_id_column=subject_id_column,
    )
    if z_base.size == 0 or z_follow is None or z_follow.size == 0:
        return baseline_df, followup_by_month

    pca = PCA(n_components=1)
    base_component = pca.fit_transform(z_base)[:, 0]
    follow_component = pca.transform(z_follow)[:, 0]

    baseline_component_df = baseline_df[[subject_id_column]].copy()
    baseline_component_df[component_name] = base_component

    projected_followups = {}
    start = 0
    for month, length in month_lengths:
        stop = start + length
        original = followup_by_month[month]
        projected = original[[subject_id_column]].copy().reset_index(drop=True)
        projected[component_name] = follow_component[start:stop]
        projected_followups[month] = projected
        start = stop

    return baseline_component_df, projected_followups


def _has_non_numeric_longitudinal_features(df, subject_id_column="src_subject_id"):
    """Handle has non numeric longitudinal features."""
    if df is None or df.empty:
        return False
    return any(
        c != subject_id_column and not pd.api.types.is_numeric_dtype(df[c])
        for c in df.columns
    )


def _longitudinal_final_data_map(final_metrics, subject_id_column="src_subject_id"):
    """
    Return baseline analysis frames in a consistent mapping.

    Multiview runs store ``final_metrics["data"]`` as a modality dictionary.
    Singleclust runs store one merged dataframe with ``Modality__feature``
    columns. The longitudinal code uses this adapter so both result schemas
    follow the same downstream path.
    """
    data = final_metrics.get("data") if isinstance(final_metrics, dict) else None
    if isinstance(data, dict):
        frames = {
            str(name): df.copy()
            for name, df in data.items()
            if isinstance(df, pd.DataFrame) and subject_id_column in df.columns
        }
        if not frames:
            raise KeyError("final_metrics['data'] contains no modality dataframes with subject IDs.")
        return frames, False
    if isinstance(data, pd.DataFrame):
        if subject_id_column not in data.columns:
            raise KeyError(f"subject_id_column='{subject_id_column}' not found in final_metrics['data'].")
        analysis_name = str(final_metrics.get("cluster_pipeline") or final_metrics.get("pipeline") or "singleclust")
        return {analysis_name: data.copy()}, True
    raise KeyError("final_metrics['data'] must be either a modality dataframe dictionary or a singleclust dataframe.")


def _align_singleclust_longitudinal_frame(
    frame,
    training_columns,
    subject_id_column="src_subject_id",
):
    """Handle align singleclust longitudinal frame."""
    if frame is None or not isinstance(frame, pd.DataFrame) or subject_id_column not in frame.columns:
        return pd.DataFrame(columns=[subject_id_column])
    present = [c for c in training_columns if c in frame.columns]
    if not present:
        return pd.DataFrame(columns=[subject_id_column])
    return frame[[subject_id_column] + present].copy()


def _longitudinal_raw_vars_to_keep(
    preprocessing_details,
    meta,
    modalities,
    subject_id_column="src_subject_id",
    phenotype_col="phenotype",
):
    """Handle longitudinal raw vars to keep."""
    vars_to_keep = []
    if isinstance(preprocessing_details, dict):
        feature_columns_per_modality = preprocessing_details.get("feature_columns_per_modality", {}) or {}
        for modality in modalities:
            vars_to_keep.extend(feature_columns_per_modality.get(modality, []) or [])

    if not vars_to_keep and "ElementName" in meta.columns and "Modality" in meta.columns:
        vars_to_keep = meta.loc[meta["Modality"].isin(modalities), "ElementName"].tolist()

    clean = []
    for value in vars_to_keep + [subject_id_column, phenotype_col]:
        if isinstance(value, str) and value not in clean:
            clean.append(value)
    return clean


def _longitudinal_raw_available_training_columns(
    raw_df,
    training_columns,
    raw_training_columns=None,
    subject_id_column="src_subject_id",
    min_nonmissing=1,
):
    """Handle longitudinal raw available training columns."""
    if raw_df is None or raw_df.empty:
        return []
    available = []
    raw_columns = set(raw_df.columns)
    raw_training_columns = [
        c for c in (raw_training_columns or [])
        if isinstance(c, str) and c != subject_id_column
    ]
    raw_training_set = set(raw_training_columns)
    for column in training_columns:
        candidates = []
        if isinstance(column, str):
            candidates.append(column.split("__", 1)[1] if "__" in column else column)
            candidates.append(column)
            for raw_name in raw_training_columns:
                if column == raw_name or column.startswith(f"{raw_name}_") or column.startswith(f"{raw_name}__"):
                    candidates.append(raw_name)
        candidates = list(dict.fromkeys(c for c in candidates if c in raw_training_set or c in raw_columns))
        raw_name = next((c for c in candidates if c in raw_columns), None)
        if raw_name is None:
            continue
        if raw_df[raw_name].notna().sum() >= min_nonmissing:
            available.append(column)
    return available


def _longitudinal_training_to_raw_feature_map(
    raw_df,
    training_columns,
    raw_training_columns=None,
    subject_id_column="src_subject_id",
):
    """Handle longitudinal training to raw feature map."""
    if raw_df is None or raw_df.empty:
        return {}
    raw_columns = set(raw_df.columns)
    raw_training_columns = [
        c for c in (raw_training_columns or [])
        if isinstance(c, str) and c != subject_id_column
    ]
    raw_training_set = set(raw_training_columns)
    mapping = {}
    for column in training_columns:
        candidates = []
        if isinstance(column, str):
            candidates.append(column.split("__", 1)[1] if "__" in column else column)
            candidates.append(column)
            for raw_name in raw_training_columns:
                if column == raw_name or column.startswith(f"{raw_name}_") or column.startswith(f"{raw_name}__"):
                    candidates.append(raw_name)
        candidates = list(dict.fromkeys(c for c in candidates if c in raw_training_set or c in raw_columns))
        raw_name = next((c for c in candidates if c in raw_columns), None)
        if raw_name is not None:
            mapping[column] = raw_name
    return mapping


def _longitudinal_followup_availability(
    raw_df,
    training_columns,
    raw_training_columns=None,
    subject_id_column="src_subject_id",
    col_threshold=0.5,
    row_threshold=0.5,
):
    """
    Apply baseline-style raw missingness filtering to one follow-up sample/domain.

    Columns survive when their raw follow-up missing fraction is <= col_threshold.
    Rows survive when their raw missing fraction across surviving columns is
    <= row_threshold. Imputation happens later, after this raw-data gate.
    """
    if raw_df is None or raw_df.empty:
        return [], [], []
    feature_map = _longitudinal_training_to_raw_feature_map(
        raw_df,
        training_columns,
        raw_training_columns=raw_training_columns,
        subject_id_column=subject_id_column,
    )
    if not feature_map:
        return [], [], []

    n_rows = len(raw_df)
    min_nonmissing_col = int(np.ceil((1 - col_threshold) * n_rows))
    kept_pairs = []
    for training_col, raw_col in feature_map.items():
        if raw_df[raw_col].notna().sum() >= min_nonmissing_col:
            kept_pairs.append((training_col, raw_col))

    if not kept_pairs:
        return [], [], []

    kept_training = [training_col for training_col, _ in kept_pairs]
    kept_raw = list(dict.fromkeys(raw_col for _, raw_col in kept_pairs))
    raw_matrix = raw_df[kept_raw]
    min_nonmissing_row = int(np.ceil((1 - row_threshold) * len(kept_raw)))
    row_mask = raw_matrix.notna().sum(axis=1) >= min_nonmissing_row
    if subject_id_column in raw_df.columns:
        eligible_subject_ids = raw_df.loc[row_mask, subject_id_column].astype(str).tolist()
    else:
        eligible_subject_ids = raw_df.index[row_mask].astype(str).tolist()
    return kept_training, kept_raw, eligible_subject_ids


def _singleclust_raw_available_training_columns(
    raw_df,
    training_columns,
    subject_id_column="src_subject_id",
    min_nonmissing=1,
):
    """Handle singleclust raw available training columns."""
    return _longitudinal_raw_available_training_columns(
        raw_df,
        training_columns,
        subject_id_column=subject_id_column,
        min_nonmissing=min_nonmissing,
    )


def _labels_from_final_metrics(
    final_metrics,
    subject_id_column="src_subject_id",
    validation_domain_labels=None,
    validation_final_labels=None,
    validation_subject_ids=None,
    validation_baseline_data=None,
):
    """Handle labels from final metrics."""
    labels = {"discovery": {}, "validation": {}}
    data_dict = final_metrics.get("data", {})
    if isinstance(data_dict, pd.DataFrame):
        analysis_name = str(final_metrics.get("cluster_pipeline") or final_metrics.get("pipeline") or "singleclust")
        subject_ids = data_dict[subject_id_column].astype(str).tolist()
        if "final_labels" in final_metrics:
            n_labels = min(len(subject_ids), len(final_metrics["final_labels"]))
            label_df = pd.DataFrame({
                subject_id_column: subject_ids[:n_labels],
                "label": pd.Series(final_metrics["final_labels"][:n_labels]).astype(str).values,
            })
            labels["discovery"][analysis_name] = label_df
        modalities = []
    else:
        modalities = list(data_dict.keys()) if isinstance(data_dict, dict) else []
    if modalities:
        first_modality = modalities[0]
        subject_ids = data_dict[first_modality][subject_id_column].astype(str).tolist()
        individual_labels = final_metrics.get("individual_labels", [])
        for idx, modality in enumerate(modalities):
            if idx < len(individual_labels):
                labels["discovery"][modality] = pd.DataFrame({
                    subject_id_column: subject_ids,
                    "label": pd.Series(individual_labels[idx]).astype(str).values,
                })
        if "final_labels" in final_metrics:
            labels["discovery"]["integrated"] = pd.DataFrame({
                subject_id_column: subject_ids,
                "label": pd.Series(final_metrics["final_labels"]).astype(str).values,
            })

    if validation_domain_labels:
        for modality, values in validation_domain_labels.items():
            if isinstance(validation_subject_ids, dict) and modality in validation_subject_ids:
                subject_ids = list(validation_subject_ids[modality])[:len(values)]
            elif isinstance(validation_subject_ids, (list, tuple, pd.Series, np.ndarray)):
                subject_ids = list(validation_subject_ids)[:len(values)]
            else:
                subject_ids = list(range(len(values)))
            labels["validation"][modality] = pd.DataFrame({
                subject_id_column: pd.Series(subject_ids).astype(str).values,
                "label": pd.Series(values).astype(str).values,
            })
    if validation_final_labels is not None:
        if isinstance(validation_subject_ids, dict) and validation_subject_ids:
            first_ids = next(iter(validation_subject_ids.values()))
            subject_ids = list(first_ids)[:len(validation_final_labels)]
        elif isinstance(validation_subject_ids, (list, tuple, pd.Series, np.ndarray)):
            subject_ids = list(validation_subject_ids)[:len(validation_final_labels)]
        else:
            subject_ids = list(range(len(validation_final_labels)))
        labels["validation"]["integrated"] = pd.DataFrame({
            subject_id_column: pd.Series(subject_ids).astype(str).values,
            "label": pd.Series(validation_final_labels).astype(str).values,
        })
        if isinstance(data_dict, pd.DataFrame):
            analysis_name = str(final_metrics.get("cluster_pipeline") or final_metrics.get("pipeline") or "singleclust")
            labels["validation"][analysis_name] = labels["validation"]["integrated"].copy()
            labels["validation"].pop("integrated", None)
    return labels


def _build_longitudinal_label_df(
    baseline_df,
    followup_by_month,
    labels_df,
    subject_id_column="src_subject_id",
    min_followup_timepoints_per_feature=1,
    min_nonmissing_per_timepoint=8,
):
    """Build longitudinal label df."""
    labels_use = labels_df[[subject_id_column, "label"]].dropna().drop_duplicates(subject_id_column).copy()
    labels_use[subject_id_column] = labels_use[subject_id_column].astype(str)

    baseline = baseline_df.copy()
    baseline[subject_id_column] = baseline[subject_id_column].astype(str)
    baseline = baseline.merge(labels_use, on=subject_id_column, how="inner")
    if baseline.empty:
        return pd.DataFrame(), [], {}

    common_candidates = [
        c for c in baseline.columns
        if c not in (subject_id_column, "label") and pd.api.types.is_numeric_dtype(baseline[c])
    ]
    feature_rows = []
    for feat in common_candidates:
        if baseline[feat].notna().sum() < min_nonmissing_per_timepoint:
            continue
        usable_months = []
        for month, df_month in followup_by_month.items():
            if feat in df_month.columns and pd.api.types.is_numeric_dtype(df_month[feat]):
                n_obs = df_month[[subject_id_column, feat]].dropna().shape[0]
                if n_obs >= min_nonmissing_per_timepoint:
                    usable_months.append(month)
        if len(usable_months) >= min_followup_timepoints_per_feature:
            feature_rows.append((feat, usable_months))
    feature_months = {row[0]: sorted(row[1]) for row in feature_rows}
    features = list(feature_months.keys())
    if not features:
        return pd.DataFrame(), [], {}

    pieces = []
    base_long = baseline[[subject_id_column, "label"] + features].melt(
        id_vars=[subject_id_column, "label"],
        value_vars=features,
        var_name="feature",
        value_name="value",
    )
    base_long["time"] = "baseline"
    base_long["month"] = 0
    pieces.append(base_long)

    for month, df_month in sorted(followup_by_month.items()):
        month_use = df_month.copy()
        month_use[subject_id_column] = month_use[subject_id_column].astype(str)
        month_features = [
            feat for feat in features
            if month in feature_months.get(feat, []) and feat in month_use.columns
        ]
        if not month_features:
            continue
        month_use = month_use[[subject_id_column] + month_features].merge(labels_use, on=subject_id_column, how="inner")
        if month_use.empty:
            continue
        part = month_use[[subject_id_column, "label"] + month_features].melt(
            id_vars=[subject_id_column, "label"],
            value_vars=month_features,
            var_name="feature",
            value_name="value",
        )
        part["time"] = f"month{month}"
        part["month"] = int(month)
        pieces.append(part)

    long_df = pd.concat(pieces, ignore_index=True)
    long_df = long_df.dropna(subset=["value", "label"])
    return long_df, features, feature_months


def _run_mean_drift_sensitivity(
    long_df,
    features,
    output_dir,
    analysis_name,
    subject_id_column="src_subject_id",
    features_to_plot=None,
    min_group_n=4,
    top_n_plot=12,
):
    """Assess whether subgroup trajectories may reflect overall mean drift."""
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.multitest import multipletests
    except Exception as exc:
        raise ImportError("statsmodels is required for mean-drift sensitivity tests.") from exc

    rows = []
    for feat in features:
        tmp = long_df.loc[
            long_df["feature"] == feat,
            [subject_id_column, "label", "time", "month", "value"],
        ].dropna().copy()
        if tmp.empty or "baseline" not in tmp["time"].astype(str).values:
            continue
        tmp["group"] = tmp["label"].astype(str)
        baseline_vals = (
            tmp.loc[tmp["time"].astype(str) == "baseline", [subject_id_column, "group", "value"]]
            .drop_duplicates(subject_id_column)
            .rename(columns={"value": "baseline_value"})
        )
        follow = tmp.loc[tmp["month"] > 0].merge(
            baseline_vals,
            on=[subject_id_column, "group"],
            how="inner",
        )
        if follow.empty:
            continue
        follow["change_from_baseline"] = follow["value"] - follow["baseline_value"]
        follow["time"] = follow["time"].cat.remove_unused_categories() if hasattr(follow["time"], "cat") else follow["time"]
        group_counts = follow.drop_duplicates([subject_id_column, "group"])["group"].value_counts()
        if group_counts.shape[0] < 2 or group_counts.min() < min_group_n:
            rows.append({
                "analysis_name": analysis_name,
                "feature": feat,
                "status": "too_few_subjects_or_groups",
                "n_subjects": int(follow[subject_id_column].nunique()),
                "n_groups": int(group_counts.shape[0]),
                "min_group_n": int(group_counts.min()) if not group_counts.empty else 0,
            })
            continue

        baseline_group_means = baseline_vals.groupby("group")["baseline_value"].mean()
        baseline_gap = float(baseline_group_means.max() - baseline_group_means.min()) if len(baseline_group_means) > 1 else np.nan
        overall_baseline_mean = float(baseline_vals["baseline_value"].mean())
        row_base = {
            "analysis_name": analysis_name,
            "feature": feat,
            "status": "ok",
            "n_subjects": int(follow[subject_id_column].nunique()),
            "n_groups": int(group_counts.shape[0]),
            "min_group_n": int(group_counts.min()),
            "baseline_gap_max_minus_min": baseline_gap,
            "overall_baseline_mean": overall_baseline_mean,
        }

        def _wald_term_values(result, term_name):
            """Handle wald term values."""
            try:
                term_table = result.wald_test_terms(skip_single=False).table
            except Exception:
                return np.nan, np.nan, np.nan
            if term_name not in term_table.index:
                return np.nan, np.nan, np.nan
            row = term_table.loc[term_name]
            statistic = row.get("statistic", np.nan)
            if hasattr(statistic, "item"):
                statistic = statistic.item()
            p_value = row.get("pvalue", np.nan)
            if hasattr(p_value, "item"):
                p_value = p_value.item()
            df_constraint = row.get("df_constraint", np.nan)
            if hasattr(df_constraint, "item"):
                df_constraint = df_constraint.item()
            return float(statistic), float(p_value), float(df_constraint)

        try:
            adjusted = smf.mixedlm(
                "value ~ baseline_value + C(group) * C(time)",
                follow,
                groups=follow[subject_id_column],
            ).fit(reml=False, method="lbfgs", disp=False, maxiter=200)
            baseline_wald, baseline_p, baseline_df = _wald_term_values(adjusted, "baseline_value")
            group_wald, group_p, group_df = _wald_term_values(adjusted, "C(group)")
            time_wald, time_p, time_df = _wald_term_values(adjusted, "C(time)")
            interaction_wald, interaction_p, interaction_df = _wald_term_values(adjusted, "C(group):C(time)")
            row_base.update({
                "baseline_adjusted_test_type": "mixedlm_wald_terms",
                "baseline_adjusted_baseline_wald_chi2": baseline_wald,
                "baseline_adjusted_baseline_df": baseline_df,
                "baseline_adjusted_baseline_p": baseline_p,
                "baseline_adjusted_group_wald_chi2": group_wald,
                "baseline_adjusted_group_df": group_df,
                "baseline_adjusted_group_p": group_p,
                "baseline_adjusted_time_wald_chi2": time_wald,
                "baseline_adjusted_time_df": time_df,
                "baseline_adjusted_time_p": time_p,
                "baseline_adjusted_interaction_wald_chi2": interaction_wald,
                "baseline_adjusted_interaction_df": interaction_df,
                "baseline_adjusted_interaction_p": interaction_p,
            })
        except Exception as exc:
            row_base["baseline_adjusted_model_error"] = str(exc)

        try:
            full = smf.mixedlm(
                "change_from_baseline ~ C(group) * C(time)",
                follow,
                groups=follow[subject_id_column],
            ).fit(reml=False, method="lbfgs", disp=False)
            no_interaction = smf.mixedlm(
                "change_from_baseline ~ C(group) + C(time)",
                follow,
                groups=follow[subject_id_column],
            ).fit(reml=False, method="lbfgs", disp=False)
            lr = max(0.0, 2.0 * (full.llf - no_interaction.llf))
            df_diff = max(1, int(full.df_modelwc - no_interaction.df_modelwc))
            row_base["change_score_interaction_lr"] = lr
            row_base["change_score_interaction_p"] = float(chi2.sf(lr, df_diff))
        except Exception as exc:
            row_base["change_score_model_error"] = str(exc)

        for month, month_df in follow.groupby("month"):
            group_means = month_df.groupby("group")["value"].mean()
            followup_gap = float(group_means.max() - group_means.min()) if len(group_means) > 1 else np.nan
            overall_followup_mean = float(month_df["value"].mean())
            row_base[f"month{int(month)}_overall_mean"] = overall_followup_mean
            row_base[f"month{int(month)}_overall_drift"] = overall_followup_mean - overall_baseline_mean
            row_base[f"month{int(month)}_gap_max_minus_min"] = followup_gap
            row_base[f"month{int(month)}_abs_gap_change"] = (
                abs(followup_gap) - abs(baseline_gap)
                if pd.notna(followup_gap) and pd.notna(baseline_gap)
                else np.nan
            )
        rows.append(row_base)

    out = pd.DataFrame(rows)
    for col in [
        "baseline_adjusted_group_p",
        "baseline_adjusted_time_p",
        "baseline_adjusted_interaction_p",
        "change_score_interaction_p",
    ]:
        if col in out.columns:
            ok = out[col].notna()
            out[f"{col}_fdr"] = np.nan
            if ok.any():
                out.loc[ok, f"{col}_fdr"] = multipletests(out.loc[ok, col], method="fdr_bh")[1]
    if not out.empty:
        sort_cols = [
            c for c in [
                "change_score_interaction_p_fdr",
                "baseline_adjusted_interaction_p_fdr",
                "change_score_interaction_p",
                "baseline_adjusted_interaction_p",
            ]
            if c in out.columns
        ]
        if sort_cols:
            out = out.sort_values(sort_cols, na_position="last")

    csv_path = os.path.join(output_dir, f"{analysis_name}_mean_drift_sensitivity.csv")
    out.to_csv(csv_path, index=False)

    plot_features = list(features_to_plot or [])
    if not plot_features and not out.empty and "feature" in out.columns:
        plot_features = out["feature"].dropna().astype(str).head(top_n_plot).tolist()
    plot_features = [feat for feat in plot_features if feat in set(features)][:top_n_plot]
    raw_plot_path = os.path.join(output_dir, f"{analysis_name}_mean_drift_raw_trajectories.png")
    change_plot_path = os.path.join(output_dir, f"{analysis_name}_mean_drift_change_scores.png")

    if plot_features:
        def group_sort(value):
            """Handle group sort."""
            try:
                return (0, float(value))
            except Exception:
                return (1, str(value))

        ncols = min(3, len(plot_features))
        nrows = int(np.ceil(len(plot_features) / ncols))
        fig_raw, axes_raw = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.4 * ncols, 4.0 * nrows), squeeze=False)
        fig_change, axes_change = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.4 * ncols, 4.0 * nrows), squeeze=False)

        for ax_raw, ax_change, feat in zip(axes_raw.ravel(), axes_change.ravel(), plot_features):
            tmp = long_df.loc[
                long_df["feature"] == feat,
                [subject_id_column, "label", "time", "month", "value"],
            ].dropna().copy()
            if tmp.empty:
                continue
            time_order_feature = [str(t) for t in tmp["time"].cat.categories if (tmp["time"].astype(str) == str(t)).any()]
            x_positions = {time: idx for idx, time in enumerate(time_order_feature)}
            groups = sorted(tmp["label"].astype(str).dropna().unique(), key=group_sort)
            group_colors = modality_cluster_palette(groups, modality=analysis_name)

            overall = tmp.groupby("time", observed=False)["value"].mean().reindex(time_order_feature).dropna()
            ax_raw.plot(
                [x_positions[str(t)] for t in overall.index.astype(str)],
                overall.values,
                color="#222222",
                linewidth=2.4,
                marker="o",
                label="Overall mean",
            )

            baseline_vals = (
                tmp.loc[tmp["time"].astype(str) == "baseline", [subject_id_column, "label", "value"]]
                .drop_duplicates(subject_id_column)
                .rename(columns={"value": "baseline_value"})
            )
            tmp_change = tmp.merge(baseline_vals, on=[subject_id_column, "label"], how="inner")
            tmp_change["change_from_baseline"] = tmp_change["value"] - tmp_change["baseline_value"]

            for group in groups:
                group_tmp = tmp.loc[tmp["label"].astype(str) == group]
                stats = group_tmp.groupby("time", observed=False)["value"].mean().reindex(time_order_feature).dropna()
                if not stats.empty:
                    ax_raw.plot(
                        [x_positions[str(t)] for t in stats.index.astype(str)],
                        stats.values,
                        color=group_colors[group],
                        linewidth=2.0,
                        marker="o",
                        label=f"Cluster {group}",
                    )

                group_change = tmp_change.loc[tmp_change["label"].astype(str) == group]
                stats_change = (
                    group_change.groupby("time", observed=False)["change_from_baseline"]
                    .mean()
                    .reindex(time_order_feature)
                    .dropna()
                )
                if not stats_change.empty:
                    ax_change.plot(
                        [x_positions[str(t)] for t in stats_change.index.astype(str)],
                        stats_change.values,
                        color=group_colors[group],
                        linewidth=2.0,
                        marker="o",
                        label=f"Cluster {group}",
                    )

            for ax in [ax_raw, ax_change]:
                ax.set_xticks(list(range(len(time_order_feature))))
                ax.set_xticklabels(time_order_feature, rotation=35)
                ax.grid(axis="y", alpha=0.2)
                sns.despine(ax=ax)
                ax.legend(frameon=False, fontsize=8)
            feature_label = display_feature_name(feat)
            ax_raw.set_title(f"{feature_label}\nraw means with overall mean")
            ax_change.axhline(0, color="#555555", linewidth=1, linestyle="--")
            ax_change.set_title(f"{feature_label}\nchange from baseline")

        for ax in axes_raw.ravel()[len(plot_features):]:
            ax.axis("off")
        for ax in axes_change.ravel()[len(plot_features):]:
            ax.axis("off")
        fig_raw.tight_layout()
        fig_change.tight_layout()
        _save_longitudinal_matplotlib_image(fig_raw, raw_plot_path, dpi=300, bbox_inches="tight")
        _save_longitudinal_matplotlib_image(fig_change, change_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig_raw)
        plt.close(fig_change)

    return {
        "summary": out,
        "summary_path": csv_path,
        "raw_plot_path": raw_plot_path if os.path.exists(raw_plot_path) else "",
        "change_plot_path": change_plot_path if os.path.exists(change_plot_path) else "",
    }


def run_longitudinal_mixed_models(
    baseline_df,
    followup_by_month,
    labels_df,
    output_dir,
    analysis_name,
    subject_id_column="src_subject_id",
    min_followup_timepoints_per_feature=1,
    min_nonmissing_per_timepoint=8,
    min_group_n=4,
    top_n_plot=12,
    reuse_existing=False,
):
    """Run baseline-cluster linear mixed models across all available months."""
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.multitest import multipletests
    except Exception as exc:
        raise ImportError("statsmodels is required for longitudinal mixed models.") from exc

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"{analysis_name}_mixedlm_summary.csv")
    plot_path = os.path.join(output_dir, f"{analysis_name}_mixedlm_top_features.png")
    all_plot_path = os.path.join(output_dir, f"{analysis_name}_mixedlm_all_features.png")
    cache_meta_path = os.path.join(output_dir, f"{analysis_name}_mixedlm_cache_meta.csv")
    cache_params = {
        "min_followup_timepoints_per_feature": int(min_followup_timepoints_per_feature),
        "min_nonmissing_per_timepoint": int(min_nonmissing_per_timepoint),
        "min_group_n": int(min_group_n),
        "plot_version": 11,
    }
    if reuse_existing and os.path.exists(summary_path) and os.path.exists(cache_meta_path):
        try:
            cache_meta = pd.read_csv(cache_meta_path).iloc[0].to_dict()
            cache_matches = all(
                int(cache_meta.get(key, -999)) == int(value)
                for key, value in cache_params.items()
            )
        except Exception:
            cache_matches = False
        if cache_matches:
            summary = pd.read_csv(summary_path)
            features = summary["feature"].dropna().astype(str).tolist() if "feature" in summary.columns else []
            return {
                "summary": summary,
                "long_df": pd.DataFrame(),
                "features": features,
                "feature_months": {},
                "summary_path": summary_path,
                "plot_path": plot_path if os.path.exists(plot_path) else "",
                "all_plot_path": all_plot_path if os.path.exists(all_plot_path) else "",
                "mean_drift_summary_path": os.path.join(output_dir, f"{analysis_name}_mean_drift_sensitivity.csv"),
                "mean_drift_raw_plot_path": os.path.join(output_dir, f"{analysis_name}_mean_drift_raw_trajectories.png"),
                "mean_drift_change_plot_path": os.path.join(output_dir, f"{analysis_name}_mean_drift_change_scores.png"),
                "cached": True,
            }

    long_df, features, feature_months = _build_longitudinal_label_df(
        baseline_df,
        followup_by_month,
        labels_df,
        subject_id_column=subject_id_column,
        min_followup_timepoints_per_feature=min_followup_timepoints_per_feature,
        min_nonmissing_per_timepoint=min_nonmissing_per_timepoint,
    )
    if long_df.empty:
        summary = pd.DataFrame([{
            "analysis_name": analysis_name,
            "status": "no_features_after_timepoint_filter",
            "min_followup_timepoints_per_feature": min_followup_timepoints_per_feature,
            "min_nonmissing_per_timepoint": min_nonmissing_per_timepoint,
        }])
        summary.to_csv(summary_path, index=False)
        pd.DataFrame([cache_params]).to_csv(cache_meta_path, index=False)
        return {"summary": summary, "long_df": long_df, "features": features, "feature_months": feature_months}

    long_df["group"] = long_df["label"].astype(str)
    time_order = ["baseline"] + [f"month{m}" for m in sorted(followup_by_month)]
    long_df["time"] = pd.Categorical(long_df["time"], categories=time_order, ordered=True)

    rows = []
    for feat in features:
        tmp = long_df.loc[long_df["feature"] == feat, [subject_id_column, "group", "time", "month", "value"]].dropna().copy()
        tmp["time"] = tmp["time"].cat.remove_unused_categories()
        group_counts = tmp.drop_duplicates([subject_id_column, "group"])["group"].value_counts()
        followup_months_in_model = sorted(int(m) for m in tmp.loc[tmp["month"] > 0, "month"].dropna().unique())
        if tmp["time"].nunique() < 2 or not followup_months_in_model:
            rows.append({
                "analysis_name": analysis_name,
                "feature": feat,
                "status": "too_few_timepoints",
                "n_subjects": int(tmp[subject_id_column].nunique()),
                "n_timepoints": int(tmp["time"].nunique()),
                "followup_months_in_model": ",".join(map(str, followup_months_in_model)),
            })
            continue
        if group_counts.shape[0] < 2 or group_counts.min() < min_group_n:
            rows.append({
                "analysis_name": analysis_name,
                "feature": feat,
                "status": "too_few_subjects_or_groups",
                "n_subjects": int(tmp[subject_id_column].nunique()),
                "n_groups": int(group_counts.shape[0]),
                "min_group_n": int(group_counts.min()) if not group_counts.empty else 0,
                "n_timepoints": int(tmp["time"].nunique()),
                "followup_months_in_model": ",".join(map(str, followup_months_in_model)),
            })
            continue
        try:
            full = smf.mixedlm("value ~ C(group) * C(time)", tmp, groups=tmp[subject_id_column]).fit(reml=False, method="lbfgs", disp=False)
            no_interaction = smf.mixedlm("value ~ C(group) + C(time)", tmp, groups=tmp[subject_id_column]).fit(reml=False, method="lbfgs", disp=False)
            no_group = smf.mixedlm("value ~ C(time)", tmp, groups=tmp[subject_id_column]).fit(reml=False, method="lbfgs", disp=False)
            no_time = smf.mixedlm("value ~ C(group)", tmp, groups=tmp[subject_id_column]).fit(reml=False, method="lbfgs", disp=False)
            interaction_lr = max(0.0, 2.0 * (full.llf - no_interaction.llf))
            group_lr = max(0.0, 2.0 * (no_interaction.llf - no_group.llf))
            time_lr = max(0.0, 2.0 * (no_interaction.llf - no_time.llf))
            interaction_df = max(1, int(full.df_modelwc - no_interaction.df_modelwc))
            group_df = max(1, int(no_interaction.df_modelwc - no_group.df_modelwc))
            time_df = max(1, int(no_interaction.df_modelwc - no_time.df_modelwc))
            interaction_p = float(chi2.sf(interaction_lr, interaction_df))
            group_p = float(chi2.sf(group_lr, group_df))
            time_p = float(chi2.sf(time_lr, time_df))
            status = "ok"
            beta_row = {f"Beta: {k}": float(v) for k, v in full.params.items()}
        except Exception as exc:
            rows.append({
                "analysis_name": analysis_name,
                "feature": feat,
                "status": f"model_failed: {exc}",
                "n_subjects": int(tmp[subject_id_column].nunique()),
                "n_groups": int(group_counts.shape[0]),
                "min_group_n": int(group_counts.min()) if not group_counts.empty else 0,
            })
            continue

        means = tmp.groupby(["group", "time"], observed=False)["value"].mean().unstack()
        mean_row = {
            f"Mean {group} {time}": float(value)
            for group, values in means.iterrows()
            for time, value in values.items()
            if pd.notna(value)
        }
        rows.append({
            "analysis_name": analysis_name,
            "feature": feat,
            "status": status,
            "n_subjects": int(tmp[subject_id_column].nunique()),
            "n_observations": int(tmp.shape[0]),
            "n_groups": int(group_counts.shape[0]),
            "min_group_n": int(group_counts.min()),
            "n_timepoints": int(tmp["time"].nunique()),
            "n_followup_timepoints": int(len(followup_months_in_model)),
            "followup_months_in_model": ",".join(map(str, followup_months_in_model)),
            "group_lr": group_lr,
            "group_p": group_p,
            "time_lr": time_lr,
            "time_p": time_p,
            "interaction_lr": interaction_lr,
            "interaction_p": interaction_p,
            **mean_row,
            **beta_row,
        })

    summary = pd.DataFrame(rows)
    for col in ["group_p", "time_p", "interaction_p"]:
        ok = summary[col].notna() if col in summary.columns else pd.Series(False, index=summary.index)
        summary[f"{col}_fdr"] = np.nan
        if ok.any():
            summary.loc[ok, f"{col}_fdr"] = multipletests(summary.loc[ok, col], method="fdr_bh")[1]

    sort_cols = [c for c in ["interaction_p_fdr", "interaction_p", "time_p_fdr", "group_p_fdr"] if c in summary.columns]
    if sort_cols:
        summary = summary.sort_values(sort_cols, na_position="last")
    summary.to_csv(summary_path, index=False)
    pd.DataFrame([cache_params]).to_csv(cache_meta_path, index=False)

    all_ok_features = summary.loc[summary["status"].eq("ok"), "feature"].tolist() if "status" in summary else []
    ok_features = all_ok_features[:top_n_plot]
    if all_ok_features:
        def format_fdr_effect(label, p_value):
            """Format fdr effect."""
            if pd.isna(p_value):
                return f"{label}: q=NA"
            if p_value < 0.001:
                stars = "***"
            elif p_value < 0.01:
                stars = "**"
            elif p_value < 0.05:
                stars = "*"
            else:
                stars = "ns"
            return f"{label}: q={p_value:.3g} {stars}"

        def plot_feature_grid(feature_list, output_path, dpi=300):
            """Plot feature grid."""
            ncols = min(3, len(feature_list))
            nrows = int(np.ceil(len(feature_list) / ncols))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.4 * ncols, 4.25 * nrows), squeeze=False)
            for ax, feat in zip(axes.ravel(), feature_list):
                tmp = long_df.loc[long_df["feature"] == feat].copy()
                time_order_feature = [str(t) for t in tmp["time"].cat.categories if (tmp["time"].astype(str) == str(t)).any()]
                x_positions = {time: idx for idx, time in enumerate(time_order_feature)}
                groups = sorted(tmp["label"].astype(str).dropna().unique(), key=cluster_sort_key)
                group_colors = modality_cluster_palette(groups, modality=analysis_name)

                for group in groups:
                    group_tmp = tmp.loc[tmp["label"].astype(str) == group].copy()
                    stats = (
                        group_tmp.groupby("time", observed=False)["value"]
                        .agg(["mean", "count", "std"])
                        .reindex(time_order_feature)
                        .dropna(subset=["mean"])
                    )
                    if stats.empty:
                        continue
                    stats["se"] = stats["std"] / np.sqrt(stats["count"].clip(lower=1))
                    xs = [x_positions[str(t)] for t in stats.index.astype(str)]
                    ys = stats["mean"].to_numpy(dtype=float)
                    yerr = stats["se"].fillna(0.0).to_numpy(dtype=float)
                    ax.errorbar(
                        xs,
                        ys,
                        yerr=yerr,
                        marker="o",
                        linewidth=2.0,
                        markersize=5,
                        capsize=3,
                        color=group_colors[group],
                        label=f"Cluster {group}",
                    )

                feature_row = summary.loc[summary["feature"].astype(str) == str(feat)].iloc[0]
                effect_text = "\n".join([
                    format_fdr_effect("group", feature_row.get("group_p_fdr", np.nan)),
                    format_fdr_effect("time", feature_row.get("time_p_fdr", np.nan)),
                    format_fdr_effect("group*time", feature_row.get("interaction_p_fdr", np.nan)),
                ])
                ax.set_title(display_feature_name(feat), pad=28)
                ax.text(
                    0.5,
                    1.01,
                    effect_text,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
                ax.set_xlabel("")
                time_counts = (
                    tmp.groupby("time", observed=False)[subject_id_column]
                    .nunique()
                    .reindex(time_order_feature)
                    .dropna()
                    .astype(int)
                )
                ax.set_xticks(list(range(len(time_order_feature))))
                ax.set_xticklabels([f"{time}\nn={int(time_counts.get(time, 0))}" for time in time_order_feature])
                ax.tick_params(axis="x", rotation=35)
                ax.grid(axis="y", alpha=0.2)
                ax.legend(frameon=False, fontsize=8)
                sns.despine(ax=ax)
            for ax in axes.ravel()[len(feature_list):]:
                ax.axis("off")
            fig.tight_layout()
            _save_longitudinal_matplotlib_image(fig, output_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        if ok_features:
            plot_feature_grid(ok_features, plot_path)
        all_plot_nrows = int(np.ceil(len(all_ok_features) / min(3, len(all_ok_features))))
        all_plot_dpi = max(72, min(300, int(60000 / max(1, 4.25 * all_plot_nrows))))
        plot_feature_grid(all_ok_features, all_plot_path, dpi=all_plot_dpi)

    mean_drift = _run_mean_drift_sensitivity(
        long_df=long_df,
        features=features,
        output_dir=output_dir,
        analysis_name=analysis_name,
        subject_id_column=subject_id_column,
        features_to_plot=ok_features,
        min_group_n=min_group_n,
        top_n_plot=top_n_plot,
    )

    return {
        "summary": summary,
        "long_df": long_df,
        "features": features,
        "feature_months": feature_months,
        "summary_path": summary_path,
        "plot_path": plot_path if os.path.exists(plot_path) else "",
        "all_plot_path": all_plot_path if os.path.exists(all_plot_path) else "",
        "mean_drift": mean_drift,
        "mean_drift_summary_path": mean_drift.get("summary_path", ""),
        "mean_drift_raw_plot_path": mean_drift.get("raw_plot_path", ""),
        "mean_drift_change_plot_path": mean_drift.get("change_plot_path", ""),
        "cached": False,
    }


def plot_cluster_membership_sankey_over_time(
    paired_df,
    output_dir=None,
    prefix="cluster_membership_over_time",
    subject_id_column="src_subject_id",
    title=None,
    width=1100,
    height=650,
    show=False,
):
    """Plot observed cluster transitions over time with count-scaled Sankey bands."""
    try:
        import plotly.graph_objects as go
    except Exception as err:
        raise RuntimeError("Plotly is required for longitudinal mapping plots.") from err

    required = {subject_id_column, "month", "baseline_cluster", "assigned_cluster"}
    missing = required.difference(paired_df.columns)
    if missing:
        raise KeyError(f"paired_df is missing required columns: {sorted(missing)}")

    optional_cols = [col for col in ["n_features", "n_baseline_features"] if col in paired_df.columns]
    df = paired_df[list(required) + optional_cols].dropna(subset=list(required)).copy()
    if df.empty:
        raise ValueError("No paired cluster assignments available for Sankey plot.")
    df[subject_id_column] = df[subject_id_column].astype(str)
    df["month"] = df["month"].astype(int)
    df["baseline_cluster"] = df["baseline_cluster"].astype(str)
    df["assigned_cluster"] = df["assigned_cluster"].astype(str)

    months = sorted(df["month"].unique())
    stages = ["baseline"] + [f"month{month}" for month in months]
    wide = (
        df.pivot_table(
            index=subject_id_column,
            columns="month",
            values="assigned_cluster",
            aggfunc="first",
        )
        .rename(columns={month: f"month{month}" for month in months})
    )
    baseline = df.groupby(subject_id_column)["baseline_cluster"].first()
    wide.insert(0, "baseline", baseline.reindex(wide.index))

    month_availability = {}
    for month in months:
        month_df = df.loc[df["month"] == month]
        n_subjects = int(month_df[subject_id_column].nunique())
        n_features = (
            int(month_df["n_features"].max())
            if "n_features" in month_df.columns and month_df["n_features"].notna().any()
            else np.nan
        )
        month_availability[int(month)] = {"n_subjects": n_subjects, "n_features": n_features}

    n_baseline_features = (
        int(df["n_baseline_features"].max())
        if "n_baseline_features" in df.columns and df["n_baseline_features"].notna().any()
        else np.nan
    )
    baseline_feature_text = f", p={int(n_baseline_features)}" if pd.notna(n_baseline_features) else ""
    stage_labels = {
        "baseline": f"Baseline<br>n={int(wide['baseline'].notna().sum())}{baseline_feature_text}"
    }
    for month in months:
        availability = month_availability.get(int(month), {})
        n_subjects = availability.get("n_subjects", 0)
        n_features = availability.get("n_features", np.nan)
        feature_text = f", p={int(n_features)}" if pd.notna(n_features) else ""
        stage_labels[f"month{month}"] = f"Month {month}<br>n={n_subjects}{feature_text}"

    def sort_label_key(label):
        """Handle sort label key."""
        label = str(label)
        try:
            return (0, float(label))
        except Exception:
            return (1, label)

    cluster_order = sorted(
        pd.unique(
            pd.concat(
                [wide[stage].dropna().astype(str) for stage in stages if stage in wide],
                ignore_index=True,
            )
        ),
        key=sort_label_key,
    )
    if not cluster_order:
        raise ValueError("No observed cluster assignments available for longitudinal mapping plot.")

    color_map = modality_cluster_palette(cluster_order)

    def rgba_from_hex(hex_color, alpha=0.55):
        """Handle rgba from hex."""
        hex_color = str(hex_color).lstrip("#")
        if len(hex_color) != 6:
            return "rgba(120,120,120,0.45)"
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    stage_x = {stage: 0.035 + 0.93 * (idx / max(1, len(stages) - 1)) for idx, stage in enumerate(stages)}
    cluster_y = {
        cluster: 0.18 + 0.64 * (idx / max(1, len(cluster_order) - 1))
        for idx, cluster in enumerate(cluster_order)
    }
    if len(cluster_order) == 1:
        cluster_y = {cluster_order[0]: 0.5}

    node_index = {}
    node_labels = []
    node_colors = []
    node_x = []
    node_y = []

    def add_node(stage, cluster):
        """Add node."""
        key = (stage, str(cluster))
        if key in node_index:
            return node_index[key]
        idx = len(node_labels)
        node_index[key] = idx
        node_labels.append(f"C{cluster}")
        node_colors.append(color_map.get(str(cluster), "#999999"))
        node_x.append(stage_x[stage])
        node_y.append(cluster_y[str(cluster)])
        return idx

    sources, targets, values, link_colors, customdata = [], [], [], [], []
    for left_stage, right_stage in zip(stages[:-1], stages[1:]):
        if left_stage not in wide.columns or right_stage not in wide.columns:
            continue
        observed = wide[[left_stage, right_stage]].dropna().astype(str)
        if observed.empty:
            continue
        counts = observed.groupby([left_stage, right_stage]).size().reset_index(name="count")
        for _, row in counts.iterrows():
            left_cluster = str(row[left_stage])
            right_cluster = str(row[right_stage])
            count = int(row["count"])
            sources.append(add_node(left_stage, left_cluster))
            targets.append(add_node(right_stage, right_cluster))
            values.append(count)
            link_colors.append(rgba_from_hex(color_map.get(right_cluster, "#777777"), alpha=0.50))
            customdata.append(f"{left_stage} C{left_cluster} -> {right_stage} C{right_cluster}<br>n={count}")

    if not values:
        raise ValueError("No observed adjacent-timepoint transitions available for longitudinal mapping plot.")

    fig = go.Figure(
        go.Sankey(
            arrangement="fixed",
            node=dict(
                label=node_labels,
                color=node_colors,
                x=node_x,
                y=node_y,
                pad=24,
                thickness=20,
                line=dict(color="rgba(0,0,0,0.20)", width=0.5),
                hovertemplate="%{label}<extra></extra>",
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                customdata=customdata,
                hovertemplate="%{customdata}<extra></extra>",
            ),
        )
    )
    for stage in stages:
        fig.add_annotation(
            x=stage_x[stage],
            y=1.08,
            xref="paper",
            yref="paper",
            text=stage_labels.get(stage, stage),
            showarrow=False,
            align="center",
            font=dict(size=12, color="#111111"),
        )
    for idx, value in enumerate(cluster_order):
        fig.add_annotation(
            x=0.5 + idx * 0.12,
            y=-0.13,
            xref="paper",
            yref="paper",
            text=f"<span style='color:{color_map[value]}'>●</span> Cluster {value}",
            showarrow=False,
            font=dict(size=12, color="#111111"),
        )
    fig.update_layout(
        title=title or "Cluster membership mapping over time",
        width=width,
        height=height,
        template="simple_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font_size=12,
        margin=dict(t=125, l=45, r=45, b=95),
    )

    paths = {}
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, f"{prefix}_sankey_over_time.html")
        fig.write_html(html_path)
        paths["html"] = html_path
        try:
            png_path = os.path.join(output_dir, f"{prefix}_sankey_over_time.png")
            fig.write_image(png_path, scale=2)
            pdf_path = os.path.splitext(png_path)[0] + ".pdf"
            fig.write_image(pdf_path, scale=2)
            svg_path = os.path.splitext(png_path)[0] + ".svg"
            fig.write_image(svg_path)
            paths["png"] = png_path
            paths["pdf"] = pdf_path
            paths["svg"] = svg_path
        except Exception:
            pass
    if show:
        fig.show()
    return fig, paths

def analyze_cluster_change_across_time(
    baseline_df,
    followup_by_month,
    labels_df,
    output_dir,
    analysis_name,
    subject_id_column="src_subject_id",
    min_features=2,
    min_paired_subjects_per_feature=8,
    reuse_existing=False,
):
    """Assign each follow-up month to nearest baseline cluster centroids and plot transitions."""
    os.makedirs(output_dir, exist_ok=True)
    paired_path = os.path.join(output_dir, f"{analysis_name}_cluster_membership_change.csv")
    summary_path = os.path.join(output_dir, f"{analysis_name}_cluster_membership_change_summary.csv")
    transition_heatmap_path = os.path.join(output_dir, f"{analysis_name}_transition_heatmaps.png")
    switch_rate_path = os.path.join(output_dir, f"{analysis_name}_switch_rate_over_months.png")
    cache_meta_path = os.path.join(output_dir, f"{analysis_name}_cluster_membership_change_cache_meta.csv")
    cache_params = {
        "min_features": int(min_features),
        "min_paired_subjects_per_feature": int(min_paired_subjects_per_feature),
        "plot_version": 14,
    }

    if reuse_existing and os.path.exists(paired_path) and os.path.exists(cache_meta_path):
        try:
            cache_meta = pd.read_csv(cache_meta_path).iloc[0].to_dict()
            cache_matches = all(
                int(cache_meta.get(key, -999)) == int(value)
                for key, value in cache_params.items()
            )
        except Exception:
            cache_matches = False
    else:
        cache_matches = False

    if reuse_existing and cache_matches:
        paired = pd.read_csv(paired_path)
        cached_is_complete = {
            subject_id_column,
            "month",
            "baseline_cluster",
            "assigned_cluster",
        }.issubset(paired.columns)
        if not cached_is_complete:
            paired = None
        else:
            summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
            sankey_paths = {}
            try:
                _, sankey_paths = plot_cluster_membership_sankey_over_time(
                    paired,
                    output_dir=output_dir,
                    prefix=analysis_name,
                    subject_id_column=subject_id_column,
                    title=f"{analysis_name}: cluster mapping over time",
                )
            except Exception as exc:
                warnings.warn(f"Could not create longitudinal Sankey for {analysis_name}: {exc}")
            plot_paths = {
                "transition_heatmaps": transition_heatmap_path if os.path.exists(transition_heatmap_path) else "",
                "switch_rate_over_months": switch_rate_path if os.path.exists(switch_rate_path) else "",
                **{f"sankey_{k}": v for k, v in sankey_paths.items()},
            }
            return {
                "paired_df": paired,
                "summary": summary,
                "transition_tables": {},
                "plot_paths": plot_paths,
                "cached": True,
            }

    labels_use = labels_df[[subject_id_column, "label"]].dropna().drop_duplicates(subject_id_column).copy()
    labels_use[subject_id_column] = labels_use[subject_id_column].astype(str)
    baseline = baseline_df.copy()
    baseline[subject_id_column] = baseline[subject_id_column].astype(str)
    baseline = baseline.merge(labels_use, on=subject_id_column, how="inner")
    baseline_feature_count = len([
        c for c in baseline.columns
        if c not in (subject_id_column, "label")
        and pd.api.types.is_numeric_dtype(baseline[c])
    ])

    cluster_order = sorted(pd.unique(baseline["label"].astype(str)))
    all_pairs = []
    transition_tables = {}
    for month, month_df in sorted(followup_by_month.items()):
        month_use = month_df.copy()
        month_use[subject_id_column] = month_use[subject_id_column].astype(str)
        features = [
            c for c in baseline.columns
            if c not in (subject_id_column, "label")
            and c in month_use.columns
            and pd.api.types.is_numeric_dtype(baseline[c])
            and pd.api.types.is_numeric_dtype(month_use[c])
        ]
        usable_features = []
        for feat in features:
            paired_feature = (
                baseline[[subject_id_column, feat]]
                .merge(
                    month_use[[subject_id_column, feat]],
                    on=subject_id_column,
                    how="inner",
                    suffixes=("_baseline", "_followup"),
                )
                .dropna(subset=[f"{feat}_baseline", f"{feat}_followup"])
            )
            if paired_feature.shape[0] >= min_paired_subjects_per_feature:
                usable_features.append(feat)
        features = usable_features
        if len(features) < min_features:
            continue
        base_cols = [f"{feat}_baseline" for feat in features]
        follow_cols = [f"{feat}_followup" for feat in features]
        merged = (
            baseline[[subject_id_column, "label"] + features]
            .merge(month_use[[subject_id_column] + features], on=subject_id_column, how="inner", suffixes=("_baseline", "_followup"))
            .dropna(subset=["label"] + base_cols + follow_cols)
        )
        if merged.empty:
            continue
        scaler = RobustScaler()
        x_base = scaler.fit_transform(merged[base_cols].to_numpy(dtype=float))
        x_follow = scaler.transform(merged[follow_cols].to_numpy(dtype=float))
        centroid_df = (
            pd.DataFrame(x_base, columns=features)
            .assign(label=merged["label"].astype(str).values)
            .groupby("label")
            .mean()
            .reindex(cluster_order)
        )
        centroids = centroid_df.to_numpy(dtype=float)
        distances = np.linalg.norm(x_follow[:, None, :] - centroids[None, :, :], axis=2)
        assigned = np.asarray(cluster_order, dtype=object)[distances.argmin(axis=1)]
        pca = PCA(n_components=1)
        pc_base = pca.fit_transform(x_base)[:, 0]
        pc_follow = pca.transform(x_follow)[:, 0]
        pair = merged[[subject_id_column, "label"]].copy()
        pair = pair.rename(columns={"label": "baseline_cluster"})
        pair["month"] = int(month)
        pair["assigned_cluster"] = assigned
        pair["switched_cluster"] = pair["baseline_cluster"].astype(str) != pair["assigned_cluster"].astype(str)
        pair["pc1_baseline"] = pc_base
        pair["pc1_followup"] = pc_follow
        pair["pc1_change"] = pc_follow - pc_base
        pair["n_features"] = len(features)
        pair["n_baseline_features"] = int(baseline_feature_count)
        pair["min_distance_to_centroid"] = distances.min(axis=1)
        all_pairs.append(pair)
        transition_tables[int(month)] = pd.crosstab(
            pair["baseline_cluster"],
            pair["assigned_cluster"],
            dropna=False,
        ).reindex(index=cluster_order, columns=cluster_order, fill_value=0)

    if not all_pairs:
        paired = pd.DataFrame([{
            "analysis_name": analysis_name,
            "status": "no_followup_month_with_enough_features",
            "min_features": min_features,
        }])
        paired.to_csv(paired_path, index=False)
        pd.DataFrame([cache_params]).to_csv(cache_meta_path, index=False)
        return {"paired_df": paired, "transition_tables": transition_tables}

    paired = pd.concat(all_pairs, ignore_index=True)
    paired.to_csv(paired_path, index=False)
    pd.DataFrame([cache_params]).to_csv(cache_meta_path, index=False)
    summary = (
        paired.groupby(["month", "baseline_cluster"])
        .agg(
            n_subjects=(subject_id_column, "nunique"),
            n_switched=("switched_cluster", "sum"),
            switch_rate=("switched_cluster", "mean"),
            mean_pc1_change=("pc1_change", "mean"),
            mean_min_distance_to_centroid=("min_distance_to_centroid", "mean"),
            n_features=("n_features", "max"),
            n_baseline_features=("n_baseline_features", "max"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    months = sorted(transition_tables)
    fig, axes = plt.subplots(1, len(months), figsize=(5.0 * len(months), 4.5), squeeze=False)
    for ax, month in zip(axes.ravel(), months):
        pct = transition_tables[month].div(transition_tables[month].sum(axis=1), axis=0).fillna(0.0)
        sns.heatmap(pct, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax, cbar=month == months[-1])
        month_subset = paired.loc[paired["month"] == month]
        n_subjects_month = int(month_subset[subject_id_column].nunique())
        n_features_month = int(month_subset["n_features"].max()) if month_subset["n_features"].notna().any() else 0
        n_baseline_features = int(month_subset["n_baseline_features"].max()) if "n_baseline_features" in month_subset.columns and month_subset["n_baseline_features"].notna().any() else 0
        ax.set_title(f"Month {month}\nn={n_subjects_month}, p_base={n_baseline_features}, p_month={n_features_month}")
        ax.set_xlabel("Assigned cluster")
        ax.set_ylabel("Baseline cluster")
    fig.tight_layout()
    _save_longitudinal_matplotlib_image(fig, transition_heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    switch_clusters = sorted(summary["baseline_cluster"].astype(str).dropna().unique(), key=cluster_sort_key)
    sns.lineplot(
        data=summary.assign(baseline_cluster=summary["baseline_cluster"].astype(str)),
        x="month",
        y="switch_rate",
        hue="baseline_cluster",
        hue_order=switch_clusters,
        palette=modality_cluster_palette(switch_clusters, modality=analysis_name),
        marker="o",
        ax=ax,
    )
    month_availability = (
        paired.groupby("month")
        .agg(
            n_subjects=(subject_id_column, "nunique"),
            n_features=("n_features", "max"),
            n_baseline_features=("n_baseline_features", "max"),
        )
        .reset_index()
    )
    for _, row in month_availability.iterrows():
        ax.text(
            row["month"],
            1.02,
            f"n={int(row['n_subjects'])}\np_base={int(row['n_baseline_features'])}\np_month={int(row['n_features'])}",
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False,
        )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Switch rate")
    ax.set_xlabel("Month")
    ax.grid(axis="y", alpha=0.2)
    sns.despine(ax=ax)
    fig.tight_layout()
    _save_longitudinal_matplotlib_image(fig, switch_rate_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    sankey_paths = {}
    try:
        _, sankey_paths = plot_cluster_membership_sankey_over_time(
            paired,
            output_dir=output_dir,
            prefix=analysis_name,
            subject_id_column=subject_id_column,
            title=f"{analysis_name}: cluster mapping over time",
        )
    except Exception as exc:
        warnings.warn(f"Could not create longitudinal Sankey for {analysis_name}: {exc}")

    plot_paths = {
        "transition_heatmaps": transition_heatmap_path,
        "switch_rate_over_months": switch_rate_path,
        **{f"sankey_{k}": v for k, v in sankey_paths.items()},
    }

    return {
        "paired_df": paired,
        "summary": summary,
        "transition_tables": transition_tables,
        "plot_paths": plot_paths,
        "cached": False,
    }


def run_longitudinal_multiclust_report(
    final_metrics,
    meta,
    plots_dir,
    data_dirs=None,
    prescient_ids=None,
    vars_to_keep=None,
    categorical_columns=None,
    validation_domain_labels=None,
    validation_final_labels=None,
    validation_subject_ids=None,
    validation_baseline_data=None,
    months=(1, 2, 3, 4, 5),
    subject_id_column="src_subject_id",
    phenotype_col="phenotype",
    min_features_per_analysis=2,
    min_features_for_cluster_change=1,
    min_followup_timepoints_per_feature=1,
    min_nonmissing_per_timepoint=8,
    followup_col_threshold=0.5,
    followup_row_threshold=0.5,
    min_group_n=4,
    reuse_existing=True,
    skip_model_fits=False,
):
    """
    Run longitudinal postprocessing for discovery and validation samples.

    The analysis uses baseline cluster labels, applies baseline feature-space
    decisions to all available months, filters raw follow-up variables/rows with
    baseline-style missingness thresholds before timepoint-local imputation, fits
    mixed models over baseline plus months 1-5, and summarizes how follow-up
    observations map back onto baseline cluster centroids.
    """
    if "preprocessing_details" not in final_metrics:
        raise KeyError("final_metrics must contain preprocessing_details.")
    baseline_data_by_analysis, is_singleclust_data = _longitudinal_final_data_map(
        final_metrics,
        subject_id_column=subject_id_column,
    )

    out_root = os.path.join(plots_dir, "longitudinal_all_timepoints")
    os.makedirs(out_root, exist_ok=True)
    preprocessing_details = final_metrics["preprocessing_details"]
    preprocessing_modalities = list(preprocessing_details.get("modalities_in_output") or [])
    if not preprocessing_modalities:
        params = preprocessing_details.get("preprocessing_parameters", {}) or {}
        preprocessing_modalities = list(params.get("modalities_requested", []) or [])
    if not preprocessing_modalities and not is_singleclust_data:
        preprocessing_modalities = list(baseline_data_by_analysis.keys())
    month_files = find_longitudinal_month_files(data_dirs=data_dirs, months=months)
    if not month_files:
        raise FileNotFoundError("No longitudinal data_month{1-5}.csv files found.")

    month_raw = _read_longitudinal_months(month_files, categorical_columns=categorical_columns)
    if vars_to_keep is None or is_singleclust_data:
        vars_to_keep = _longitudinal_raw_vars_to_keep(
            preprocessing_details,
            meta,
            preprocessing_modalities,
            subject_id_column=subject_id_column,
            phenotype_col=phenotype_col,
        )
    else:
        vars_to_keep = [v for v in vars_to_keep if isinstance(v, str)]

    split_raw = {"discovery": {}, "validation": {}}
    for month, df in month_raw.items():
        df_alias = _apply_longitudinal_followup_aliases(df, expected_features=vars_to_keep)
        use = df_alias[[c for c in df_alias.columns if c in vars_to_keep]].copy() if vars_to_keep else df_alias.copy()
        if phenotype_col in use.columns:
            use_chr = use.loc[use[phenotype_col] != "HC"].reset_index(drop=True)
        else:
            use_chr = use
        if prescient_ids is not None:
            disc, val = split_by_network(use_chr, set(prescient_ids), id_col=subject_id_column)
        else:
            disc, val = use_chr, pd.DataFrame(columns=use_chr.columns)
        split_raw["discovery"][month] = disc
        split_raw["validation"][month] = val

    preprocessed = {"discovery": {}, "validation": {}}
    reports = []
    for sample, month_map in split_raw.items():
        for month, df in month_map.items():
            if df.empty:
                continue
            aligned = {}
            if is_singleclust_data:
                analysis_name, baseline_df = next(iter(baseline_data_by_analysis.items()))
                training_cols = [c for c in baseline_df.columns if c != subject_id_column]
                raw_available_training_cols, raw_available_vars, eligible_subject_ids = _longitudinal_followup_availability(
                    df,
                    training_cols,
                    subject_id_column=subject_id_column,
                    col_threshold=followup_col_threshold,
                    row_threshold=followup_row_threshold,
                )
                present = []
                observed = []
                preprocessing_status = None
                if len(raw_available_training_cols) >= min_features_per_analysis and eligible_subject_ids:
                    keep_cols = [subject_id_column] + [c for c in raw_available_vars if c in df.columns]
                    df_use = df.loc[
                        df[subject_id_column].astype(str).isin(set(eligible_subject_ids)),
                        keep_cols,
                    ].copy()
                    try:
                        _, _, dict_month = apply_preprocessing_to_month2(
                            df=df_use,
                            meta=meta,
                            preprocessing_details=preprocessing_details,
                            subject_id_column=subject_id_column,
                            imputation_mode="independent",
                            align_modalities=False,
                        )
                        merged_month = _merge_modalities_for_longitudinal(dict_month, subject_id_column=subject_id_column)
                        present = [c for c in raw_available_training_cols if c in merged_month.columns]
                        observed = [
                            c for c in present
                            if not merged_month[c].isna().all()
                        ]
                        if len(observed) >= min_features_per_analysis:
                            month_use = merged_month[[subject_id_column] + observed].copy()
                            if not month_use.empty:
                                aligned[analysis_name] = month_use
                    except Exception as exc:
                        preprocessing_status = f"preprocessing_failed: {exc}"
                status = (
                    "kept"
                    if analysis_name in aligned
                    else (
                        preprocessing_status
                        if preprocessing_status is not None
                        else (
                            "dropped_no_rows_after_raw_missingness"
                            if len(raw_available_training_cols) >= min_features_per_analysis
                            else "dropped_too_few_raw_observed_features"
                        )
                    )
                )
                reports.append({
                    "sample": sample,
                    "month": month,
                    "modality": analysis_name,
                    "n_training_features": len(training_cols),
                    "n_features_available": len(observed),
                    "n_features_aligned": len(present),
                    "n_raw_features_available": len(raw_available_training_cols),
                    "n_raw_variables_available": len(raw_available_vars),
                    "n_rows_passing_raw_missingness": len(eligible_subject_ids),
                    "followup_col_threshold": followup_col_threshold,
                    "followup_row_threshold": followup_row_threshold,
                    "imputation_mode": "independent",
                    "imputation_fit_scope": "followup_timepoint_after_raw_missingness_filter",
                    "status": status,
                })
            else:
                for modality, baseline_df in baseline_data_by_analysis.items():
                    training_cols = [c for c in baseline_df.columns if c != subject_id_column]
                    raw_training_cols = (
                        preprocessing_details.get("feature_columns_per_modality", {}) or {}
                    ).get(modality, training_cols)
                    raw_available_training_cols, raw_available_vars, eligible_subject_ids = _longitudinal_followup_availability(
                        df,
                        training_cols,
                        raw_training_columns=raw_training_cols,
                        subject_id_column=subject_id_column,
                        col_threshold=followup_col_threshold,
                        row_threshold=followup_row_threshold,
                    )
                    present = []
                    observed = []
                    kept = False
                    preprocessing_status = None
                    if len(raw_available_training_cols) >= min_features_per_analysis and eligible_subject_ids:
                        keep_cols = [subject_id_column] + [c for c in raw_available_vars if c in df.columns]
                        df_use = df.loc[
                            df[subject_id_column].astype(str).isin(set(eligible_subject_ids)),
                            keep_cols,
                        ].copy()
                        try:
                            _, _, dict_month = apply_preprocessing_to_month2(
                                df=df_use,
                                meta=meta,
                                preprocessing_details=preprocessing_details,
                                subject_id_column=subject_id_column,
                                imputation_mode="independent",
                                align_modalities=False,
                            )
                            if modality in dict_month:
                                present = [c for c in raw_available_training_cols if c in dict_month[modality].columns]
                                observed = [
                                    c for c in present
                                    if not dict_month[modality][c].isna().all()
                                ]
                                if len(observed) >= min_features_per_analysis:
                                    month_use = dict_month[modality][[subject_id_column] + observed].copy()
                                    if not month_use.empty:
                                        aligned[modality] = month_use
                                        kept = True
                            else:
                                preprocessing_status = "preprocessing_failed: modality absent after preprocessing"
                        except Exception as exc:
                            preprocessing_status = f"preprocessing_failed: {exc}"
                    reports.append({
                        "sample": sample,
                        "month": month,
                        "modality": modality,
                        "n_training_features": len(training_cols),
                        "n_features_available": len(observed),
                        "n_features_aligned": len(present),
                        "n_raw_features_available": len(raw_available_training_cols),
                        "n_raw_variables_available": len(raw_available_vars),
                        "n_rows_passing_raw_missingness": len(eligible_subject_ids),
                        "followup_col_threshold": followup_col_threshold,
                        "followup_row_threshold": followup_row_threshold,
                        "imputation_mode": "independent",
                        "imputation_fit_scope": "followup_timepoint_after_raw_missingness_filter",
                        "status": "kept" if kept else (
                            preprocessing_status
                            if preprocessing_status is not None
                            else (
                                "dropped_no_rows_after_raw_missingness"
                                if len(raw_available_training_cols) >= min_features_per_analysis
                                else "dropped_too_few_raw_observed_features"
                            )
                        ),
                    })
            if aligned:
                if not is_singleclust_data:
                    aligned["integrated"] = _merge_modalities_for_longitudinal(aligned, subject_id_column=subject_id_column)
                preprocessed[sample][month] = aligned

    report_columns = [
        "sample",
        "month",
        "modality",
        "n_training_features",
        "n_features_available",
        "n_features_aligned",
        "n_raw_features_available",
        "n_raw_variables_available",
        "n_rows_passing_raw_missingness",
        "followup_col_threshold",
        "followup_row_threshold",
        "imputation_mode",
        "imputation_fit_scope",
        "status",
    ]
    report_df = pd.DataFrame(reports)
    for col in report_columns:
        if col not in report_df.columns:
            report_df[col] = np.nan
    report_df = report_df[report_columns + [c for c in report_df.columns if c not in report_columns]]
    report_df.to_csv(os.path.join(out_root, "longitudinal_preprocessing_feature_report.csv"), index=False)

    label_frames = _labels_from_final_metrics(
        final_metrics,
        subject_id_column=subject_id_column,
        validation_domain_labels=validation_domain_labels,
        validation_final_labels=validation_final_labels,
        validation_subject_ids=validation_subject_ids,
    )

    results = {
        "output_dir": out_root,
        "preprocessing_report": report_df,
        "month_files": month_files,
        "analyses": {},
        "analysis_summary": pd.DataFrame(),
    }
    analysis_summary_rows = []
    discovery_baseline_by_analysis = {
        name: df.copy()
        for name, df in baseline_data_by_analysis.items()
    }
    if not is_singleclust_data:
        discovery_baseline_by_analysis["integrated"] = _merge_modalities_for_longitudinal(
            baseline_data_by_analysis,
            subject_id_column=subject_id_column,
        )

    validation_baseline_by_analysis = {}
    if isinstance(validation_baseline_data, dict) and validation_baseline_data:
        if is_singleclust_data:
            analysis_name, baseline_df = next(iter(baseline_data_by_analysis.items()))
            training_cols = [c for c in baseline_df.columns if c != subject_id_column]
            merged_validation = _merge_modalities_for_longitudinal(
                validation_baseline_data,
                subject_id_column=subject_id_column,
            )
            validation_frame = _align_singleclust_longitudinal_frame(
                merged_validation,
                training_cols,
                subject_id_column=subject_id_column,
            )
            if not validation_frame.empty:
                validation_baseline_by_analysis[analysis_name] = validation_frame
        else:
            for modality, baseline_df in baseline_data_by_analysis.items():
                if modality in validation_baseline_data:
                    training_cols = [c for c in baseline_df.columns if c != subject_id_column]
                    present = [c for c in training_cols if c in validation_baseline_data[modality].columns]
                    if present:
                        validation_baseline_by_analysis[modality] = validation_baseline_data[modality][[subject_id_column] + present].copy()
            if validation_baseline_by_analysis:
                validation_baseline_by_analysis["integrated"] = _merge_modalities_for_longitudinal(
                    validation_baseline_by_analysis,
                    subject_id_column=subject_id_column,
                )
    elif is_singleclust_data and isinstance(validation_baseline_data, pd.DataFrame):
        analysis_name, baseline_df = next(iter(baseline_data_by_analysis.items()))
        training_cols = [c for c in baseline_df.columns if c != subject_id_column]
        validation_frame = _align_singleclust_longitudinal_frame(
            validation_baseline_data,
            training_cols,
            subject_id_column=subject_id_column,
        )
        if not validation_frame.empty:
            validation_baseline_by_analysis[analysis_name] = validation_frame
    if not validation_baseline_by_analysis:
        validation_baseline_by_analysis = discovery_baseline_by_analysis

    for sample in ["discovery", "validation"]:
        if not preprocessed[sample]:
            continue
        baseline_by_analysis = (
            discovery_baseline_by_analysis
            if sample == "discovery"
            else validation_baseline_by_analysis
        )
        for analysis_name, baseline_df in baseline_by_analysis.items():
            if analysis_name not in label_frames.get(sample, {}):
                continue
            followups = {
                month: month_dict[analysis_name]
                for month, month_dict in preprocessed[sample].items()
                if analysis_name in month_dict and not month_dict[analysis_name].empty
            }
            if not followups:
                continue
            analysis_out = os.path.join(out_root, sample, analysis_name)
            labels_df = label_frames[sample][analysis_name]
            analysis_baseline_df = baseline_df
            analysis_followups = followups
            uses_mixed_component = (
                _has_non_numeric_longitudinal_features(baseline_df, subject_id_column=subject_id_column)
                or any(
                    _has_non_numeric_longitudinal_features(df_month, subject_id_column=subject_id_column)
                    for df_month in followups.values()
                )
            )
            if uses_mixed_component:
                analysis_baseline_df, analysis_followups = _build_mixed_longitudinal_component_frames(
                    baseline_df=baseline_df,
                    followup_by_month=followups,
                    subject_id_column=subject_id_column,
                    component_name="mixed_component1",
                )
            if skip_model_fits:
                long_df, features, feature_months = _build_longitudinal_label_df(
                    baseline_df=analysis_baseline_df,
                    followup_by_month=analysis_followups,
                    labels_df=labels_df,
                    subject_id_column=subject_id_column,
                    min_followup_timepoints_per_feature=min_followup_timepoints_per_feature,
                    min_nonmissing_per_timepoint=min_nonmissing_per_timepoint,
                )
                mixed = {
                    "summary": pd.DataFrame(),
                    "long_df": long_df,
                    "features": features,
                    "feature_months": feature_months,
                    "cached": False,
                    "skipped_model_fits": True,
                }
                change = {
                    "summary": pd.DataFrame(),
                    "paired_df": pd.DataFrame(),
                    "transition_tables": {},
                    "plot_paths": {},
                    "cached": False,
                    "skipped_model_fits": True,
                }
            else:
                mixed = run_longitudinal_mixed_models(
                    baseline_df=analysis_baseline_df,
                    followup_by_month=analysis_followups,
                    labels_df=labels_df,
                    output_dir=os.path.join(analysis_out, "mixedlm"),
                    analysis_name=f"{sample}_{analysis_name}",
                    subject_id_column=subject_id_column,
                    min_followup_timepoints_per_feature=min_followup_timepoints_per_feature,
                    min_nonmissing_per_timepoint=min_nonmissing_per_timepoint,
                    min_group_n=min_group_n,
                    reuse_existing=reuse_existing,
                )
                change = analyze_cluster_change_across_time(
                    baseline_df=analysis_baseline_df,
                    followup_by_month=analysis_followups,
                    labels_df=labels_df,
                    output_dir=os.path.join(analysis_out, "cluster_membership_change"),
                    analysis_name=f"{sample}_{analysis_name}",
                    subject_id_column=subject_id_column,
                    min_features=min_features_for_cluster_change,
                    reuse_existing=reuse_existing,
                )
            results["analyses"][(sample, analysis_name)] = {"mixedlm": mixed, "cluster_change": change}
            mixed_summary = mixed.get("summary", pd.DataFrame())
            change_summary = change.get("summary", pd.DataFrame())
            change_plot_paths = change.get("plot_paths", {})
            mixed_plot_path = os.path.join(analysis_out, "mixedlm", f"{sample}_{analysis_name}_mixedlm_top_features.png")
            mixed_plot_svg_path = os.path.splitext(mixed_plot_path)[0] + ".svg"
            mixed_all_plot_path = mixed.get(
                "all_plot_path",
                os.path.join(analysis_out, "mixedlm", f"{sample}_{analysis_name}_mixedlm_all_features.png"),
            )
            mean_drift_raw_plot = mixed.get("mean_drift_raw_plot_path", "")
            mean_drift_change_plot = mixed.get("mean_drift_change_plot_path", "")
            transition_heatmap_plot = change_plot_paths.get("transition_heatmaps", "")
            switch_rate_plot = change_plot_paths.get("switch_rate_over_months", "")
            analysis_summary_rows.append({
                "sample": sample,
                "analysis": analysis_name,
                "months": ",".join(map(str, sorted(followups))),
                "representation": "mixed_component1" if uses_mixed_component else "native_numeric_features",
                "n_mixedlm_rows": int(len(mixed_summary)),
                "n_mixedlm_ok": int(mixed_summary["status"].eq("ok").sum()) if "status" in mixed_summary.columns else 0,
                "n_cluster_change_rows": int(len(change.get("paired_df", pd.DataFrame()))),
                "n_cluster_change_summary_rows": int(len(change_summary)),
                "mixedlm_cached": bool(mixed.get("cached", False)),
                "cluster_change_cached": bool(change.get("cached", False)),
                "mixedlm_dir": os.path.join(analysis_out, "mixedlm"),
                "cluster_change_dir": os.path.join(analysis_out, "cluster_membership_change"),
                "mixedlm_plot": mixed_plot_path if os.path.exists(mixed_plot_path) else "",
                "mixedlm_plot_svg": mixed_plot_svg_path if os.path.exists(mixed_plot_svg_path) else "",
                "mixedlm_all_features_plot": mixed_all_plot_path if mixed_all_plot_path and os.path.exists(mixed_all_plot_path) else "",
                "mixedlm_all_features_plot_svg": os.path.splitext(mixed_all_plot_path)[0] + ".svg" if mixed_all_plot_path and os.path.exists(os.path.splitext(mixed_all_plot_path)[0] + ".svg") else "",
                "mean_drift_summary": mixed.get("mean_drift_summary_path", ""),
                "mean_drift_raw_plot": mean_drift_raw_plot,
                "mean_drift_raw_plot_svg": os.path.splitext(mean_drift_raw_plot)[0] + ".svg" if mean_drift_raw_plot and os.path.exists(os.path.splitext(mean_drift_raw_plot)[0] + ".svg") else "",
                "mean_drift_change_plot": mean_drift_change_plot,
                "mean_drift_change_plot_svg": os.path.splitext(mean_drift_change_plot)[0] + ".svg" if mean_drift_change_plot and os.path.exists(os.path.splitext(mean_drift_change_plot)[0] + ".svg") else "",
                "transition_heatmap_plot": transition_heatmap_plot,
                "transition_heatmap_plot_svg": os.path.splitext(transition_heatmap_plot)[0] + ".svg" if transition_heatmap_plot and os.path.exists(os.path.splitext(transition_heatmap_plot)[0] + ".svg") else "",
                "switch_rate_plot": switch_rate_plot,
                "switch_rate_plot_svg": os.path.splitext(switch_rate_plot)[0] + ".svg" if switch_rate_plot and os.path.exists(os.path.splitext(switch_rate_plot)[0] + ".svg") else "",
                "sankey_html": change_plot_paths.get("sankey_html", ""),
                "sankey_png": change_plot_paths.get("sankey_png", ""),
                "sankey_svg": change_plot_paths.get("sankey_svg", ""),
            })

    analysis_summary_columns = [
        "sample",
        "analysis",
        "months",
        "representation",
        "n_mixedlm_rows",
        "n_mixedlm_ok",
        "n_cluster_change_rows",
        "n_cluster_change_summary_rows",
        "mixedlm_cached",
        "cluster_change_cached",
        "mixedlm_dir",
        "cluster_change_dir",
        "mixedlm_plot",
        "mixedlm_plot_svg",
        "mixedlm_all_features_plot",
        "mixedlm_all_features_plot_svg",
        "mean_drift_raw_plot",
        "mean_drift_raw_plot_svg",
        "mean_drift_change_plot",
        "mean_drift_change_plot_svg",
        "transition_heatmap_plot",
        "transition_heatmap_plot_svg",
        "switch_rate_plot",
        "switch_rate_plot_svg",
        "sankey_html",
        "sankey_png",
        "sankey_svg",
    ]
    analysis_summary = pd.DataFrame(analysis_summary_rows)
    for col in analysis_summary_columns:
        if col not in analysis_summary.columns:
            analysis_summary[col] = pd.Series(dtype="object")
    analysis_summary = analysis_summary[
        analysis_summary_columns
        + [c for c in analysis_summary.columns if c not in analysis_summary_columns]
    ]
    analysis_summary = _dedupe_singleclust_longitudinal_summary(analysis_summary)
    results["analysis_summary"] = analysis_summary
    analysis_summary.to_csv(os.path.join(out_root, "longitudinal_analysis_summary.csv"), index=False)

    return results


def _dedupe_singleclust_longitudinal_summary(analysis_summary):
    """Handle dedupe singleclust longitudinal summary."""
    if analysis_summary is None or analysis_summary.empty or "analysis" not in analysis_summary.columns:
        return analysis_summary
    analyses = set(analysis_summary["analysis"].dropna().astype(str))
    if "singleclust" not in analyses or "integrated" not in analyses:
        return analysis_summary
    drop_mask = analysis_summary["analysis"].astype(str).eq("integrated")
    return analysis_summary.loc[~drop_mask].reset_index(drop=True)


def load_longitudinal_multiclust_results(output_dir):
    """Load saved longitudinal result paths for display without rerunning models."""
    output_dir = str(output_dir)
    analysis_summary_path = os.path.join(output_dir, "longitudinal_analysis_summary.csv")
    preprocessing_report_path = os.path.join(output_dir, "longitudinal_preprocessing_feature_report.csv")
    try:
        analysis_summary = (
            pd.read_csv(analysis_summary_path)
            if os.path.exists(analysis_summary_path)
            else pd.DataFrame()
        )
    except pd.errors.EmptyDataError:
        analysis_summary = pd.DataFrame()
    analysis_summary = _dedupe_singleclust_longitudinal_summary(analysis_summary)
    try:
        preprocessing_report = (
            pd.read_csv(preprocessing_report_path)
            if os.path.exists(preprocessing_report_path)
            else pd.DataFrame()
        )
    except pd.errors.EmptyDataError:
        preprocessing_report = pd.DataFrame()
    return {
        "output_dir": output_dir,
        "preprocessing_report": preprocessing_report,
        "month_files": {},
        "analyses": {},
        "analysis_summary": analysis_summary,
    }


def display_longitudinal_multiclust_results(
    longitudinal_results,
    max_analyses=6,
    show_sankey=True,
):
    """Display longitudinal result tables and saved plots inside a notebook."""
    try:
        from IPython.display import display, Image, HTML
    except Exception as exc:
        raise RuntimeError("IPython is required to display notebook outputs.") from exc

    analysis_summary = longitudinal_results.get("analysis_summary", pd.DataFrame())
    if analysis_summary.empty:
        print("No completed longitudinal analyses to display.")
        report = longitudinal_results.get("preprocessing_report", pd.DataFrame())
        if not report.empty:
            display(report.head(30))
        return

    display(analysis_summary)
    shown = 0
    for _, row in analysis_summary.iterrows():
        if shown >= max_analyses:
            break
        header = f"{row.get('sample', '')} / {row.get('analysis', '')}"
        print(header)
        for col in ["mixedlm_plot", "mixedlm_all_features_plot", "transition_heatmap_plot", "switch_rate_plot", "sankey_png"]:
            path = row.get(col, "")
            if isinstance(path, str) and path and os.path.exists(path):
                display(Image(filename=path))
        mean_drift_raw = row.get("mean_drift_raw_plot", "")
        mean_drift_change = row.get("mean_drift_change_plot", "")
        if (
            isinstance(mean_drift_raw, str)
            and mean_drift_raw
            and os.path.exists(mean_drift_raw)
        ) or (
            isinstance(mean_drift_change, str)
            and mean_drift_change
            and os.path.exists(mean_drift_change)
        ):
            print(
                "Mean-drift sensitivity: the raw plot overlays the overall cohort mean "
                "with subgroup means. If subgroup lines mostly follow the black overall "
                "mean, the primary LMM may reflect general drift. The change-score plot "
                "shows each subgroup's mean change from its own baseline; persistent "
                "separation here is less consistent with simple mean drift/regression to the mean."
            )
            if isinstance(mean_drift_raw, str) and mean_drift_raw and os.path.exists(mean_drift_raw):
                display(Image(filename=mean_drift_raw))
            if isinstance(mean_drift_change, str) and mean_drift_change and os.path.exists(mean_drift_change):
                display(Image(filename=mean_drift_change))
        if show_sankey:
            html_path = row.get("sankey_html", "")
            if isinstance(html_path, str) and html_path and os.path.exists(html_path):
                display(HTML(filename=html_path))
        shown += 1

# -----------------------------------------------------------------------------
# Notebook helper binding
# -----------------------------------------------------------------------------

_ACTIVE_NOTEBOOK_GLOBALS = None


def register_notebook_context(notebook_globals):
    """Register the variables used by project-specific notebook helpers."""
    global _ACTIVE_NOTEBOOK_GLOBALS
    _ACTIVE_NOTEBOOK_GLOBALS = notebook_globals

def _bind_notebook_function(function, notebook_globals, *, name=None):
    """Bind a shared helper to a notebook's variables and imported packages."""
    import functools
    import types

    rebound = types.FunctionType(
        function.__code__,
        notebook_globals,
        name or function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    rebound.__kwdefaults__ = function.__kwdefaults__
    rebound.__annotations__ = dict(function.__annotations__)
    functools.update_wrapper(rebound, function)
    if name is not None:
        rebound.__name__ = name
        rebound.__qualname__ = name
    return rebound


def _with_notebook_context(function=None, *, default_expressions=None):
    """Run a shared analysis helper with the calling notebook's variables."""
    import functools
    import inspect

    def decorate(helper):
        @functools.wraps(helper)
        def wrapped(*args, **kwargs):
            caller_globals = _ACTIVE_NOTEBOOK_GLOBALS
            if caller_globals is None:
                caller_globals = inspect.currentframe().f_back.f_globals
            if default_expressions:
                supplied = inspect.signature(helper).bind_partial(*args, **kwargs).arguments
                for parameter, expression in default_expressions.items():
                    if parameter not in supplied:
                        kwargs[parameter] = eval(expression, caller_globals)
            rebound = _bind_notebook_function(helper, caller_globals, name=helper.__name__)
            return rebound(*args, **kwargs)

        return wrapped

    return decorate(function) if function is not None else decorate

# -----------------------------------------------------------------------------
# Simpleclust analysis helpers
# Implementations are preserved from the original notebook cells.
# -----------------------------------------------------------------------------

@_with_notebook_context
def simpleclust_extract_fold_number(fold_name):
    """Process simpleclust extract fold number."""
    digits = ''.join(ch for ch in str(fold_name) if ch.isdigit())
    return int(digits) if digits else fold_name


@_with_notebook_context
def simpleclust_mode_value(series):
    """Process simpleclust mode value."""
    series = series.dropna()
    if series.empty:
        return None
    if series.apply(lambda x: isinstance(x, (list, tuple))).any():
        expanded = pd.DataFrame(series.tolist())
        return expanded.mode(dropna=True).iloc[0].tolist()
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else series.iloc[0]


@_with_notebook_context
def simpleclust_safe_numeric(value):
    """Process simpleclust safe numeric."""
    if isinstance(value, (list, tuple, dict, pd.Series, pd.DataFrame, np.ndarray)):
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


@_with_notebook_context
def simpleclust_summarise_feature_differences(final_metrics, top_k=10):
    """Process simpleclust summarise feature differences."""
    data = final_metrics.get('data')
    labels = np.asarray(final_metrics.get('final_labels'))
    if data is None or 'src_subject_id' not in data.columns:
        return pd.DataFrame()

    feature_df = data.drop(columns=['src_subject_id'], errors='ignore').select_dtypes(include=[np.number]).copy()
    if feature_df.empty or len(labels) != len(feature_df) or pd.Series(labels).nunique() < 2:
        return pd.DataFrame()

    valid_cols = feature_df.columns[feature_df.nunique(dropna=False) > 1]
    feature_df = feature_df.loc[:, valid_cols]
    if feature_df.empty:
        return pd.DataFrame()

    f_vals, p_vals = f_classif(feature_df.fillna(feature_df.mean()), labels)
    out = pd.DataFrame({
        'feature': feature_df.columns,
        'f_value': f_vals,
        'p_value': p_vals,
    })
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=['f_value'])
    return out.sort_values(['f_value', 'p_value'], ascending=[False, True]).head(top_k).reset_index(drop=True)


@_with_notebook_context
def simpleclust_save_plot_png_svg(fig, output_stem, dpi=300):
    """Process simpleclust save plot png svg."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for extension in ('png', 'svg'):
        output_path = output_stem.with_suffix(f'.{extension}')
        save_kwargs = {'bbox_inches': 'tight'}
        if extension == 'png':
            save_kwargs['dpi'] = dpi
        fig.savefig(output_path, **save_kwargs)
        saved_paths.append(output_path)
        print('Saved plot to:', output_path)
    return saved_paths


@_with_notebook_context
def simpleclust_plot_autoencoder_diagnostics(run_name, final_metrics, out_dir=None):
    """Process simpleclust plot autoencoder diagnostics."""
    ae_res = final_metrics.get('ae_res')
    if not isinstance(ae_res, dict):
        print(f'{run_name}: no ae_res payload found, skipping latent diagnostics.')
        return
    required = {'final_latent', 'all_true', 'all_pred'}
    if not required.issubset(ae_res):
        print(f'{run_name}: ae_res missing {sorted(required - set(ae_res))}, skipping latent diagnostics.')
        return

    test_latent = ae_res['final_latent']
    X_true = np.asarray(ae_res['all_true']).ravel()
    X_pred = np.asarray(ae_res['all_pred']).ravel()

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(X_true, X_pred, alpha=0.35)
    plt.xlabel('Original Data')
    plt.ylabel('Reconstructed Data')
    plt.title(f'{run_name}: Original vs Reconstructed Data')
    min_val = min(np.nanmin(X_true), np.nanmin(X_pred))
    max_val = max(np.nanmax(X_true), np.nanmax(X_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.tight_layout()
    if out_dir is not None:
        simpleclust_save_plot_png_svg(fig, Path(out_dir) / f'{run_name}_autoencoder_reconstruction')
    plt.show()

    latent_arr = np.asarray(test_latent)
    if latent_arr.ndim == 1:
        latent_arr = latent_arr.reshape(-1, 1)

    fig = plt.figure(figsize=(6, 4))
    plt.hist(latent_arr.ravel(), bins=np.linspace(-2, 2, 50))
    plt.title(f'{run_name}: Latent Distribution')
    plt.tight_layout()
    if out_dir is not None:
        simpleclust_save_plot_png_svg(fig, Path(out_dir) / f'{run_name}_latent_distribution')
    plt.show()

    if isinstance(test_latent, pd.DataFrame):
        latent_df = test_latent.copy()
    else:
        latent_df = pd.DataFrame(latent_arr, columns=[f'z{i}' for i in range(latent_arr.shape[1])])

    axes = scatter_matrix(latent_df, alpha=0.5, figsize=(8, 8), diagonal='hist')
    plt.suptitle(f'{run_name}: Pairwise Latent Scatter Plots')
    plt.tight_layout()
    if out_dir is not None:
        simpleclust_save_plot_png_svg(axes[0, 0].figure, Path(out_dir) / f'{run_name}_latent_scatter_matrix')
    plt.show()


@_with_notebook_context
def notebook_profile_value_is_enabled(value):
    """Process simpleclust truthy profile value."""
    return str(value).strip().strip('"\'').upper() in {"TRUE", "1", "YES", "Y"}


@_with_notebook_context
def read_notebook_run_profile_exports(profile_path):
    """Process simpleclust parse profile exports."""
    exports = {}
    if profile_path is None or not profile_path.exists():
        return exports
    for line in profile_path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[len("export "):].split("=", 1)
        value = value.strip()
        # Keep literal defaults such as ${VAR:-0} as-is; simple quoted values are enough for the flag check.
        exports[key.strip()] = value.strip('"\'')
    return exports


def infer_notebook_profile_from_settings(directory_profiles, *, environment_variable="RUN_PROFILE"):
    """Infer a profile using an explicit directory map and environment fallback."""
    notebook_dir = Path.cwd().name
    directory_profile = directory_profiles.get(notebook_dir)
    if directory_profile:
        return directory_profile
    environment_profile = os.environ.get(environment_variable, "")
    return environment_profile or None


@_with_notebook_context
def simpleclust_infer_notebook_profile():
    """Infer the Simpleclust profile using its original directory settings."""
    return infer_notebook_profile_from_settings({
        "clinical_paper": "clinical_paper",
        "multiclust_extended": "multiclust_extended",
        "prospect": "prospect",
        "Simpleclust": "Simpleclust",
    })


@_with_notebook_context
def find_multiclust_repository_root(start):
    """Process simpleclust find repo root."""
    start = Path(start).resolve()
    for parent in [start] + list(start.parents):
        if (parent / "run_profiles").is_dir() and (parent / "full_pipeline.py").exists():
            return parent
    return None


@_with_notebook_context
def get_cluster_sensitivity_profile_setting(repo_root, profile_name):
    """Process simpleclust profile enabled for sensitivity."""
    if not profile_name or repo_root is None:
        return None, None
    profile_path = repo_root / "run_profiles" / f"{profile_name}.sh"
    exports = read_notebook_run_profile_exports(profile_path)
    value = exports.get("DO_CLUSTER_VALIDATION_SENSITIVITY", "FALSE")
    return notebook_profile_value_is_enabled(value), profile_path


@_with_notebook_context
def display_notebook_result(obj):
    """Process simpleclust display if available."""
    if "display" in globals():
        display(obj)
    else:
        print(obj)


@_with_notebook_context
def get_nested_result_value(dct, path, default=np.nan):
    """Process simpleclust get nested."""
    cur = dct
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


@_with_notebook_context
def summarize_cluster_sensitivity_results(results):
    """Process simpleclust flatten sensitivity results."""
    rows = []
    for solution, payload in results.get("solutions", {}).items():
        rows.append({
            "solution": solution,
            "kind": payload.get("kind"),
            "observed_k": get_nested_result_value(payload, ["observed_quality", "k"]),
            "observed_n": get_nested_result_value(payload, ["observed_quality", "n"]),
            "observed_n_features": get_nested_result_value(payload, ["observed_quality", "n_features"]),
            "observed_composite": get_nested_result_value(payload, ["observed_quality", "composite"]),
            "observed_silhouette": get_nested_result_value(payload, ["observed_quality", "silhouette"]),
            "observed_calinski_harabasz": get_nested_result_value(payload, ["observed_quality", "calinski_harabasz"]),
            "observed_davies_bouldin": get_nested_result_value(payload, ["observed_quality", "davies_bouldin"]),
            "k1_composite": get_nested_result_value(payload, ["uni_cluster_baseline", "quality", "composite"]),
            "pc1_variance_explained": get_nested_result_value(payload, ["pc1_median_split", "pc1_variance_explained"]),
            "pc1_median_composite": get_nested_result_value(payload, ["pc1_median_split", "quality", "composite"]),
            "pc1_median_ari_with_observed": get_nested_result_value(payload, ["pc1_median_split", "ari_with_observed_labels"]),
            "pc1_median_stability_ari": get_nested_result_value(payload, ["pc1_median_split", "bootstrap_stability", "mean_ari"]),
            "pc1_median_stability_sd_ari": get_nested_result_value(payload, ["pc1_median_split", "bootstrap_stability", "sd_ari"]),
            "dip_available": get_nested_result_value(payload, ["dip_test_pc1", "available"], default=False),
            "dip_statistic_pc1": get_nested_result_value(payload, ["dip_test_pc1", "dip"]),
            "dip_p_value_pc1": get_nested_result_value(payload, ["dip_test_pc1", "p_value"]),
            "gap_selected_k_tibshirani": get_nested_result_value(payload, ["gap_statistic", "selected_k_tibshirani_rule"]),
            "gap_selected_k_max_gap": get_nested_result_value(payload, ["gap_statistic", "selected_k_max_gap"]),
            "sigclust_cluster_index": get_nested_result_value(payload, ["sigclust_approx", "observed_cluster_index"]),
            "sigclust_p_value": get_nested_result_value(payload, ["sigclust_approx", "p_value"]),
            "sigclust_null_mean_cluster_index": get_nested_result_value(payload, ["sigclust_approx", "null_mean_cluster_index"]),
            "gaussian_null_p_quality": get_nested_result_value(payload, ["covariance_matched_gaussian_null", "p_quality_ge_observed_recluster"]),
            "gaussian_null_p_stability": get_nested_result_value(payload, ["covariance_matched_gaussian_null", "p_stability_ge_observed_recluster"]),
            "observed_recluster_stability_ari": get_nested_result_value(payload, ["covariance_matched_gaussian_null", "observed_recluster_stability", "mean_ari"]),
            "gaussian_null_mean_stability_ari": get_nested_result_value(payload, ["covariance_matched_gaussian_null", "null_stability_mean_ari"]),
            "gaussian_null_mean_quality": get_nested_result_value(payload, ["covariance_matched_gaussian_null", "null_quality_mean"]),
        })
    return pd.DataFrame(rows)


def adjust_pvalues_benjamini_hochberg(p_values, *, coerce_numeric=False):
    """Adjust p-values with BH-FDR while preserving the selected input policy."""
    if coerce_numeric:
        values = pd.to_numeric(pd.Series(p_values), errors="coerce")
        output = pd.Series(np.nan, index=values.index, dtype=float)
        valid = values.dropna()
        if valid.empty:
            return output
        order = np.argsort(valid.values)
        ranked_values = valid.values[order]
        ranked_index = valid.index[order]
        n_tests = len(valid)
    else:
        values = pd.Series(p_values, dtype="float64")
        output = pd.Series(np.nan, index=values.index, dtype="float64")
        valid = values.dropna()
        if valid.empty:
            return output
        ranked = valid.sort_values()
        ranked_values = ranked.to_numpy()
        ranked_index = ranked.index
        n_tests = float(len(ranked))

    adjusted = ranked_values * n_tests / np.arange(1, len(ranked_values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output.loc[ranked_index] = np.clip(adjusted, 0, 1)
    return output


@_with_notebook_context
def simpleclust_bh_fdr_original_features(p_values):
    """Adjust original-feature p-values using the Simpleclust input policy."""
    return adjust_pvalues_benjamini_hochberg(p_values, coerce_numeric=False)


@_with_notebook_context
def simpleclust_format_fdr_q(q_value):
    """Process simpleclust format fdr q."""
    if pd.isna(q_value):
        return ""
    if q_value < 0.001:
        return "<0.001"
    if q_value < 0.01:
        return "<0.01"
    if q_value < 0.05:
        return "<0.05"
    return f"{q_value:.2g}"


@_with_notebook_context
def simpleclust_cohens_d(values_a, values_b):
    """Process simpleclust cohens d."""
    values_a = pd.Series(values_a, dtype="float64").dropna()
    values_b = pd.Series(values_b, dtype="float64").dropna()
    n_a, n_b = len(values_a), len(values_b)
    if n_a < 2 or n_b < 2:
        return np.nan
    var_a = values_a.var(ddof=1)
    var_b = values_b.var(ddof=1)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if not np.isfinite(pooled_var) or pooled_var <= 0:
        return np.nan
    return float((values_a.mean() - values_b.mean()) / np.sqrt(pooled_var))


@_with_notebook_context
def simpleclust_make_original_feature_pairwise_tests(chr_df, clusters, ranked_stats, subject_id_column="src_subject_id"):
    """Process simpleclust make original feature pairwise tests."""
    features = ranked_stats["feature"].drop_duplicates().tolist()
    cluster_values = pd.Series(np.asarray(clusters).reshape(-1), name="group").astype(str)
    group_order = sorted(cluster_values.dropna().unique().tolist())
    rows = []

    for feature_rank, feature in enumerate(features, start=1):
        if feature not in chr_df.columns:
            continue
        chr_values = pd.to_numeric(chr_df[feature], errors="coerce")
        plot_df = pd.DataFrame({"value": chr_values, "group": cluster_values}).dropna(subset=["value", "group"])
        if plot_df.empty:
            continue
        for group_a, group_b in combinations(group_order, 2):
            values_a = plot_df.loc[plot_df["group"].eq(group_a), "value"].dropna().astype(float)
            values_b = plot_df.loc[plot_df["group"].eq(group_b), "value"].dropna().astype(float)
            statistic, p_value = np.nan, np.nan
            if len(values_a) > 0 and len(values_b) > 0:
                try:
                    statistic, p_value = mannwhitneyu(values_a, values_b, alternative="two-sided")
                except ValueError:
                    pass
            rows.append({
                "feature_rank": feature_rank,
                "feature": feature,
                "display_feature": display_feature_name(feature),
                "comparison": f"{group_a} vs {group_b}",
                "group_a": group_a,
                "group_b": group_b,
                "n_a": int(values_a.notna().sum()),
                "n_b": int(values_b.notna().sum()),
                "median_a": values_a.median() if len(values_a) else np.nan,
                "median_b": values_b.median() if len(values_b) else np.nan,
                "median_difference_a_minus_b": (
                    values_a.median() - values_b.median() if len(values_a) and len(values_b) else np.nan
                ),
                "mann_whitney_u": statistic,
                "effect_size": simpleclust_cohens_d(values_a, values_b),
                "p_value": p_value,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["pairwise_q_value_fdr"] = simpleclust_bh_fdr_original_features(out["p_value"])
        out = out.sort_values(
            ["pairwise_q_value_fdr", "p_value", "feature_rank", "comparison"],
            na_position="last",
        ).reset_index(drop=True)
    return out


@_with_notebook_context
def simpleclust_plot_original_feature_pairwise_tiles(pairwise_df, ranked_stats, out_dir, file_prefix="baseline_chr_vs_cc", top_n=30):
    """Process simpleclust plot original feature pairwise tiles."""
    if pairwise_df.empty:
        return None
    top_features = ranked_stats["feature"].drop_duplicates().head(top_n).tolist()
    plot_df = pairwise_df[pairwise_df["feature"].isin(top_features)].dropna(subset=["effect_size"]).copy()
    if plot_df.empty:
        return None
    effect = plot_df.pivot_table(
        index="display_feature",
        columns="comparison",
        values="effect_size",
        aggfunc="first",
    )
    q_values = plot_df.pivot_table(
        index="display_feature",
        columns="comparison",
        values="pairwise_q_value_fdr",
        aggfunc="min",
    )
    feature_order = [display_feature_name(feature) for feature in top_features if display_feature_name(feature) in effect.index]
    effect = effect.reindex(feature_order)
    q_values = q_values.reindex(feature_order)
    annot = q_values.applymap(simpleclust_format_fdr_q)
    vmax = np.nanmax(np.abs(effect.to_numpy(dtype=float))) if effect.notna().any().any() else 1.0
    vmax = max(vmax, 1e-6)
    fig_height = max(5, 0.28 * len(effect) + 1.8)
    fig_width = max(7, 0.70 * effect.shape[1] + 3.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        effect,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=annot,
        fmt="",
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Effect size: Cohen's d (group A - group B)"},
        ax=ax,
    )
    ax.set_title("Original features: CHR subgroup pairwise post-hoc tests")
    ax.set_xlabel("Pairwise group comparison; tile text is FDR q")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    out_dir = Path(out_dir)
    png_path = out_dir / f"{safe_name(file_prefix)}_all_pairwise_effect_size_tiles.png"
    pdf_path = out_dir / f"{safe_name(file_prefix)}_all_pairwise_effect_size_tiles.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print("Saved original-feature pairwise effect-size tile plot to:", png_path)
    print("Saved original-feature pairwise effect-size tile plot to:", pdf_path)
    plt.show()
    return fig


@_with_notebook_context
def simpleclust_bh_fdr(p_values):
    """Adjust comparison p-values using the Simpleclust input policy."""
    return adjust_pvalues_benjamini_hochberg(p_values, coerce_numeric=False)


@_with_notebook_context
def simpleclust_clean_compare_series(series):
    """Process simpleclust clean compare series."""
    return series.replace(["nan", "NaN", "None", "NULL", "null", ""], np.nan)


@_with_notebook_context
def simpleclust_ordinal_label_maps_from_utils(label_specs):
    """Process simpleclust ordinal label maps from utils."""
    ordinal_maps = []
    for spec in label_specs or []:
        if "mapping" in spec:
            mapping = spec["mapping"]
        elif spec.get("fill_middle"):
            first, last = spec["first"], spec["last"]
            mapping = {first: spec["first_label"], last: spec["last_label"]}
            for code in range(first + 1, last):
                mapping[code] = str(code)
        else:
            continue
        reverse_map = {str(label).strip(): code for code, label in mapping.items()}
        ordinal_maps.append((set(reverse_map), reverse_map))
    return ordinal_maps


@_with_notebook_context
def simpleclust_coerce_compare_variable(series):
    """Process simpleclust coerce compare variable."""
    cleaned = simpleclust_clean_compare_series(series)
    nonmissing = cleaned.notna()
    if nonmissing.sum() == 0:
        return cleaned, "categorical"

    numeric = pd.to_numeric(cleaned, errors="coerce")
    if numeric.loc[nonmissing].notna().all():
        return numeric, "numeric"

    observed_labels = set(cleaned.loc[nonmissing].astype(str).str.strip().unique())
    for label_set, reverse_map in ORDINAL_LABEL_MAPS:
        if observed_labels.issubset(label_set):
            mapped = cleaned.astype("object").map(
                lambda value: reverse_map.get(str(value).strip(), np.nan) if pd.notna(value) else np.nan
            )
            return pd.to_numeric(mapped, errors="coerce"), "ordinal_mapped"

    return cleaned.astype("object"), "categorical"


@_with_notebook_context
def simpleclust_cramers_v_from_contingency(contingency):
    """Process simpleclust cramers v from contingency."""
    if contingency.empty:
        return np.nan
    try:
        chi2_stat, _, _, _ = chi2_contingency(contingency)
    except ValueError:
        return np.nan
    n = contingency.to_numpy().sum()
    if n <= 0:
        return np.nan
    denom = n * max(min(contingency.shape) - 1, 1)
    return float(np.sqrt(chi2_stat / denom)) if denom > 0 else np.nan


@_with_notebook_context
def simpleclust_pairwise_numeric_tests(values, groups, variable, display_name, variable_type, group_order):
    """Process simpleclust pairwise numeric tests."""
    rows = []
    for group_a, group_b in combinations(group_order, 2):
        values_a = values[groups.eq(group_a)].dropna().astype(float)
        values_b = values[groups.eq(group_b)].dropna().astype(float)
        statistic, p_value = np.nan, np.nan
        if len(values_a) > 0 and len(values_b) > 0:
            try:
                statistic, p_value = mannwhitneyu(values_a, values_b, alternative="two-sided")
            except ValueError:
                pass
        rows.append({
            "feature": variable,
            "display_feature": display_name,
            "variable_type": variable_type,
            "test": "Mann-Whitney U",
            "comparison": f"Subgroup {group_a} vs Subgroup {group_b}",
            "group_a": group_a,
            "group_b": group_b,
            "n_a": int(values_a.notna().sum()),
            "n_b": int(values_b.notna().sum()),
            "median_a": values_a.median() if len(values_a) else np.nan,
            "median_b": values_b.median() if len(values_b) else np.nan,
            "median_difference_a_minus_b": (
                values_a.median() - values_b.median() if len(values_a) and len(values_b) else np.nan
            ),
            "statistic": statistic,
            "effect_size": simpleclust_cohens_d(values_a, values_b),
            "p_value": p_value,
        })
    return rows


@_with_notebook_context
def simpleclust_pairwise_categorical_tests(values, groups, variable, display_name, variable_type, group_order):
    """Process simpleclust pairwise categorical tests."""
    rows = []
    for group_a, group_b in combinations(group_order, 2):
        pair_df = pd.DataFrame({"value": values, "group": groups})
        pair_df = pair_df[pair_df["group"].isin([group_a, group_b])].dropna()
        statistic, p_value, test_name, effect_size = np.nan, np.nan, "Chi-square", np.nan
        if pair_df["value"].nunique() >= 2 and pair_df["group"].nunique() == 2:
            contingency = pd.crosstab(pair_df["value"], pair_df["group"])
            effect_size = simpleclust_cramers_v_from_contingency(contingency)
            try:
                if contingency.shape == (2, 2):
                    _, p_value = fisher_exact(contingency.to_numpy())
                    test_name = "Fisher exact"
                else:
                    statistic, p_value, _, _ = chi2_contingency(contingency)
            except ValueError:
                pass
        rows.append({
            "feature": variable,
            "display_feature": display_name,
            "variable_type": variable_type,
            "test": test_name,
            "comparison": f"Subgroup {group_a} vs Subgroup {group_b}",
            "group_a": group_a,
            "group_b": group_b,
            "n_a": int((pair_df["group"] == group_a).sum()),
            "n_b": int((pair_df["group"] == group_b).sum()),
            "median_a": np.nan,
            "median_b": np.nan,
            "median_difference_a_minus_b": np.nan,
            "statistic": statistic,
            "effect_size": effect_size,
            "p_value": p_value,
        })
    return rows


@_with_notebook_context
def simpleclust_plot_top_feature_profile_dotplot(stats_df, long_df, out_dir, file_prefix, title, top_n=None, rows_per_page=24):
    """Process simpleclust plot top feature profile dotplot."""
    if stats_df.empty or long_df.empty:
        return []
    ordered_features = stats_df.sort_values(["q_value_fdr", "p_value"], na_position="last")["feature"].tolist()
    if top_n is not None:
        ordered_features = ordered_features[:int(top_n)]
    plot_df = long_df[long_df["feature"].isin(ordered_features)].copy()
    if plot_df.empty:
        return []
    selected_rows = []
    for feature in ordered_features:
        feature_df = plot_df[plot_df["feature"].eq(feature)]
        if feature_df.empty:
            continue
        if feature_df["variable_type"].iloc[0] == "categorical":
            row_label = (
                feature_df.groupby("row_label")["standardized_difference"]
                .apply(lambda values: values.abs().max())
                .sort_values(ascending=False)
                .index[0]
            )
            feature_df = feature_df[feature_df["row_label"].eq(row_label)]
        selected_rows.append(feature_df)
    if not selected_rows:
        return []
    plot_df = pd.concat(selected_rows, ignore_index=True)
    q_lookup = stats_df.set_index("feature")["q_value_fdr"].to_dict()
    plot_df["feature_label"] = plot_df.apply(
        lambda row: f"{row['row_label']} (q={simpleclust_format_fdr_q(q_lookup.get(row['feature'], np.nan))})",
        axis=1,
    )
    feature_order = plot_df.drop_duplicates("feature_label")["feature_label"].tolist()
    pages = [feature_order]
    figures = []
    for page_index, page_features in enumerate(pages, start=1):
        page_df = plot_df[plot_df["feature_label"].isin(page_features)].copy()
        page_order = page_features[::-1]
        page_df["feature_label"] = pd.Categorical(page_df["feature_label"], categories=page_order, ordered=True)
        y_codes = page_df["feature_label"].cat.codes.to_numpy(dtype=float)
        effect_values = page_df["standardized_difference"].to_numpy(dtype=float)
        vmax = np.nanmax(np.abs(effect_values)) if np.isfinite(effect_values).any() else 1.0
        vmax = max(vmax, 1e-6)
        subgroup_values = sorted(page_df["subgroup"].dropna().unique().tolist())
        marker_cycle = ["o", "s", "^", "D", "X", "P"]
        marker_map = {subgroup: marker_cycle[i % len(marker_cycle)] for i, subgroup in enumerate(subgroup_values)}
        fig_height = max(7, 0.30 * len(page_order) + 2.0)
        fig, ax = plt.subplots(figsize=(9.5, fig_height))
        scatter = None
        for subgroup in subgroup_values:
            mask = page_df["subgroup"].eq(subgroup).to_numpy()
            scatter = ax.scatter(
                effect_values[mask],
                y_codes[mask],
                c=effect_values[mask],
                cmap="vlag",
                vmin=-vmax,
                vmax=vmax,
                marker=marker_map[subgroup],
                s=58,
                edgecolor="black",
                linewidth=0.25,
                label=f"Subgroup {subgroup}",
            )
        ax.axvline(0, color=theme.THEME["muted"], linewidth=0.8)
        page_suffix = f" page {page_index}/{len(pages)}" if len(pages) > 1 else ""
        ax.set_title(f"{title}{page_suffix}")
        ax.set_xlabel("Effect size: standardized subgroup difference from labelled discovery sample")
        ax.set_ylabel("")
        ax.set_yticks(np.arange(len(page_order)))
        ax.set_yticklabels(page_order)
        ax.legend(title="Subgroup", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        if scatter is not None:
            cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label("Effect size")
        fig.tight_layout()
        suffix = "all_feature_profile_dotplot" if top_n is None else "top_feature_profile_dotplot"
        page_token = f"_page_{page_index:02d}" if len(pages) > 1 else ""
        png_path = Path(out_dir) / f"{file_prefix}_{suffix}{page_token}.png"
        pdf_path = Path(out_dir) / f"{file_prefix}_{suffix}{page_token}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        print("Saved feature profile dot plot to:", png_path)
        print("Saved feature profile dot plot to:", pdf_path)
        plt.show()
        figures.append(fig)
    return figures


@_with_notebook_context
def simpleclust_plot_pairwise_fdr_tiles(pairwise_df, stats_df, out_dir, file_prefix, title, top_n=None, rows_per_page=24):
    """Process simpleclust plot pairwise fdr tiles."""
    if pairwise_df.empty or stats_df.empty:
        return []
    ordered_features = stats_df.sort_values(["q_value_fdr", "p_value"], na_position="last")["feature"].tolist()
    if top_n is not None:
        ordered_features = ordered_features[:int(top_n)]
    plot_df = pairwise_df[pairwise_df["feature"].isin(ordered_features)].copy()
    plot_df = plot_df.dropna(subset=["effect_size"])
    if plot_df.empty:
        return []
    plot_df["feature_label"] = plot_df["display_feature"]
    feature_order = [display_feature_name(feature) for feature in ordered_features if display_feature_name(feature) in set(plot_df["feature_label"])]
    pages = [feature_order]
    figures = []
    for page_index, page_features in enumerate(pages, start=1):
        page_df = plot_df[plot_df["feature_label"].isin(page_features)].copy()
        effect = page_df.pivot_table(
            index="feature_label",
            columns="comparison",
            values="effect_size",
            aggfunc="first",
        ).reindex(page_features)
        q_values = page_df.pivot_table(
            index="feature_label",
            columns="comparison",
            values="pairwise_q_value_fdr",
            aggfunc="min",
        ).reindex(page_features)
        annot = q_values.applymap(simpleclust_format_fdr_q)
        vmax = np.nanmax(np.abs(effect.to_numpy(dtype=float))) if effect.notna().any().any() else 1.0
        vmax = max(vmax, 1e-6)
        fig_height = max(7, 0.28 * len(effect) + 2.0)
        fig_width = max(7.5, 0.72 * effect.shape[1] + 3.2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.heatmap(
            effect,
            cmap="vlag",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            annot=annot,
            fmt="",
            linewidths=0.35,
            linecolor="white",
            cbar_kws={"label": "Effect size: Cohen's d; Cramer's V for categorical"},
            ax=ax,
        )
        page_suffix = f" page {page_index}/{len(pages)}" if len(pages) > 1 else ""
        ax.set_title(f"{title}{page_suffix}")
        ax.set_xlabel("Pairwise subgroup comparison; tile text is FDR q")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        suffix = "pairwise_effect_size_tiles" if top_n is None else "top_pairwise_effect_size_tiles"
        page_token = f"_page_{page_index:02d}" if len(pages) > 1 else ""
        png_path = Path(out_dir) / f"{file_prefix}_{suffix}{page_token}.png"
        pdf_path = Path(out_dir) / f"{file_prefix}_{suffix}{page_token}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        print("Saved pairwise effect-size tile plot to:", png_path)
        print("Saved pairwise effect-size tile plot to:", pdf_path)
        plt.show()
        figures.append(fig)
    return figures


@_with_notebook_context
def simpleclust_compare_extra_features_by_cluster(
    source_df,
    final_data,
    final_labels,
    meta,
    modality="compare_clusters",
    subject_id_column="src_subject_id",
):
    """Process simpleclust compare extra features by cluster."""
    compare_vars = (
        meta.loc[meta["Modality"].eq(modality), "ElementName"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    available_vars = [var for var in compare_vars if var in source_df.columns]
    missing_vars = [var for var in compare_vars if var not in source_df.columns]
    if not available_vars:
        raise ValueError(
            f"No {modality!r} metadata variables were found in the source dataframe. "
            f"Examples from metadata: {compare_vars[:8]}"
        )

    labels_df = pd.DataFrame({
        subject_id_column: final_data[subject_id_column].astype(str).to_numpy(),
        "simpleclust_subgroup": np.asarray(final_labels),
    })
    analysis_df = (
        source_df[[subject_id_column] + available_vars]
        .copy()
        .assign(**{subject_id_column: lambda df: df[subject_id_column].astype(str)})
        .merge(labels_df, on=subject_id_column, how="inner", validate="one_to_one")
    )
    if analysis_df.empty:
        raise ValueError("No labelled discovery subjects were available after merging extra features by subject ID.")

    subgroup_order = sorted(analysis_df["simpleclust_subgroup"].dropna().unique())
    stats_rows = []
    heatmap_rows = []
    long_rows = []
    pairwise_rows = []

    for variable in available_vars:
        raw = simpleclust_clean_compare_series(analysis_df[variable])
        coerced_values, variable_type = simpleclust_coerce_compare_variable(raw)
        display_name = display_feature_name(variable)

        if variable_type in {"numeric", "ordinal_mapped"}:
            values = pd.to_numeric(coerced_values, errors="coerce")
            test_df = pd.DataFrame({"value": values, "subgroup": analysis_df["simpleclust_subgroup"]}).dropna()
            groups = [group["value"].to_numpy() for _, group in test_df.groupby("subgroup") if len(group) > 0]
            if len(groups) >= 2 and sum(len(group) > 1 for group in groups) >= 2:
                stat, p_value = kruskal(*groups)
            else:
                stat, p_value = np.nan, np.nan

            overall_mean = values.mean(skipna=True)
            overall_sd = values.std(skipna=True, ddof=0)
            if not np.isfinite(overall_sd) or overall_sd == 0:
                overall_sd = 1.0

            row = {"feature": variable, "display_feature": display_name, "row_label": display_name}
            for subgroup in subgroup_order:
                subgroup_values = values[analysis_df["simpleclust_subgroup"].eq(subgroup)]
                row[f"Subgroup {subgroup}"] = (subgroup_values.mean(skipna=True) - overall_mean) / overall_sd
                long_rows.append({
                    "feature": variable,
                    "display_feature": display_name,
                    "row_label": display_name,
                    "variable_type": variable_type,
                    "subgroup": subgroup,
                    "n": int(subgroup_values.notna().sum()),
                    "mean_or_proportion": subgroup_values.mean(skipna=True),
                    "standardized_difference": row[f"Subgroup {subgroup}"],
                })
            heatmap_rows.append(row)
            pairwise_rows.extend(
                simpleclust_pairwise_numeric_tests(
                    values=values,
                    groups=analysis_df["simpleclust_subgroup"],
                    variable=variable,
                    display_name=display_name,
                    variable_type=variable_type,
                    group_order=subgroup_order,
                )
            )

            stats_rows.append({
                "feature": variable,
                "display_feature": display_name,
                "variable_type": variable_type,
                "test": "Kruskal-Wallis",
                "statistic": stat,
                "p_value": p_value,
                "n": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "levels_or_categories": np.nan,
            })
            continue

        categorical = coerced_values.astype("object").where(raw.notna(), np.nan)
        test_df = pd.DataFrame({"value": categorical, "subgroup": analysis_df["simpleclust_subgroup"]}).dropna()
        if test_df.empty or test_df["value"].nunique() < 2 or test_df["subgroup"].nunique() < 2:
            stat, p_value, test_name = np.nan, np.nan, "Chi-square"
        else:
            contingency = pd.crosstab(test_df["value"], test_df["subgroup"])
            if contingency.shape == (2, 2):
                _, p_value = fisher_exact(contingency.to_numpy())
                stat, test_name = np.nan, "Fisher exact"
            else:
                stat, p_value, _, _ = chi2_contingency(contingency)
                test_name = "Chi-square"

        stats_rows.append({
            "feature": variable,
            "display_feature": display_name,
            "variable_type": variable_type,
            "test": test_name,
            "statistic": stat,
            "p_value": p_value,
            "n": int(categorical.notna().sum()),
            "missing": int(categorical.isna().sum()),
            "levels_or_categories": int(categorical.nunique(dropna=True)),
        })

        overall_props = categorical.value_counts(normalize=True, dropna=True)
        for category in sorted(overall_props.index, key=lambda value: str(value)):
            category_mask = categorical.eq(category)
            overall_p = overall_props.loc[category]
            denom = np.sqrt(max(overall_p * (1 - overall_p), 1e-6))
            row_label = f"{display_name} = {category}"
            row = {"feature": variable, "display_feature": display_name, "row_label": row_label}
            for subgroup in subgroup_order:
                subgroup_mask = analysis_df["simpleclust_subgroup"].eq(subgroup)
                subgroup_n = int((subgroup_mask & categorical.notna()).sum())
                subgroup_p = category_mask[subgroup_mask & categorical.notna()].mean() if subgroup_n else np.nan
                row[f"Subgroup {subgroup}"] = (subgroup_p - overall_p) / denom if pd.notna(subgroup_p) else np.nan
                long_rows.append({
                    "feature": variable,
                    "display_feature": display_name,
                    "row_label": row_label,
                    "variable_type": variable_type,
                    "subgroup": subgroup,
                    "n": subgroup_n,
                    "mean_or_proportion": subgroup_p,
                    "standardized_difference": row[f"Subgroup {subgroup}"],
                })
            heatmap_rows.append(row)

        pairwise_rows.extend(
            simpleclust_pairwise_categorical_tests(
                values=categorical,
                groups=analysis_df["simpleclust_subgroup"],
                variable=variable,
                display_name=display_name,
                variable_type=variable_type,
                group_order=subgroup_order,
            )
        )

    stats_df = pd.DataFrame(stats_rows)
    stats_df["q_value_fdr"] = simpleclust_bh_fdr(stats_df["p_value"])
    stats_df["significant_fdr_0_05"] = stats_df["q_value_fdr"].lt(0.05)
    stats_df = stats_df.sort_values(["q_value_fdr", "p_value", "display_feature"], na_position="last").reset_index(drop=True)

    heatmap_df = pd.DataFrame(heatmap_rows)
    if not heatmap_df.empty:
        ordering = stats_df.set_index("feature")["q_value_fdr"].to_dict()
        heatmap_df["_q_value_fdr"] = heatmap_df["feature"].map(ordering)
        heatmap_df = heatmap_df.sort_values(["_q_value_fdr", "display_feature", "row_label"], na_position="last")
        q_lookup = stats_df.set_index("feature")["q_value_fdr"].to_dict()
        p_lookup = stats_df.set_index("feature")["p_value"].to_dict()
        heatmap_df["row_label_with_q"] = heatmap_df.apply(
            lambda row: f"{row['row_label']}  (q={q_lookup.get(row['feature'], np.nan):.3g})"
            if pd.notna(q_lookup.get(row["feature"], np.nan))
            else row["row_label"],
            axis=1,
        )
        heatmap_df["p_value"] = heatmap_df["feature"].map(p_lookup)
        heatmap_df["q_value_fdr"] = heatmap_df["feature"].map(q_lookup)

    long_df = pd.DataFrame(long_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        pairwise_df["pairwise_q_value_fdr"] = simpleclust_bh_fdr(pairwise_df["p_value"])
        pairwise_df = pairwise_df.sort_values(
            ["pairwise_q_value_fdr", "p_value", "display_feature", "comparison"],
            na_position="last",
        ).reset_index(drop=True)
    return stats_df, heatmap_df, long_df, pairwise_df, missing_vars


@_with_notebook_context
def simpleclust_alternative_run_names():
    """Process simpleclust alternative run names."""
    return [
        run_name
        for run_name in sorted(dimred_runs)
        if run_name != best_dimred_option
    ]


@_with_notebook_context
def simpleclust_plot_dimred_consensus_matrix(run_name, run_metrics, out_dir=None):
    """Process simpleclust plot dimred consensus matrix."""
    diag = run_metrics.get('final_stability_SUM_MAT_full', {})
    if 'consensus' not in diag:
        print(f'{run_name}: no final consensus matrix available.')
        return None

    M = np.asarray(diag['consensus'], dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        print(f'{run_name}: final consensus matrix is not square, skipping matrix plot.')
        return None

    M = (M + M.T) / 2.0
    np.fill_diagonal(M, 1.0)
    title_suffix = 'hierarchical order'

    union_ids = np.asarray(diag.get('union_ids', []))
    final_data = run_metrics.get('data')
    final_labels = np.asarray(run_metrics.get('final_labels', []))
    if (
        isinstance(final_data, pd.DataFrame)
        and 'src_subject_id' in final_data.columns
        and len(union_ids) == M.shape[0]
        and len(final_labels) == len(final_data)
    ):
        label_by_id = dict(zip(final_data['src_subject_id'].to_numpy(), final_labels))
        labels_aligned = np.asarray([label_by_id.get(uid, -1) for uid in union_ids])
        order = np.lexsort((np.arange(len(labels_aligned)), labels_aligned))
        title_suffix = 'sorted by final labels'
    else:
        D = 1.0 - M
        order = leaves_list(linkage(squareform(D, checks=False), method='average'))

    M_ord = M[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(M_ord, aspect='auto', vmin=0, vmax=1)
    ax.set_title(f'{run_name}: consensus matrix ({title_suffix})')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Samples')
    fig.colorbar(im, ax=ax, label='Consensus')
    fig.tight_layout()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ('png', 'pdf'):
            matrix_path = os.path.join(out_dir, f'{safe_name(run_name)}_consensus_matrix.{ext}')
            fig.savefig(matrix_path, dpi=300, bbox_inches='tight')
            print('Saved matrix plot to:', matrix_path)

    plt.show()
    return fig


@_with_notebook_context
def simpleclust_merge_simpleclust_cc_preprocessed(dict_final_cc, preproc_artifact, training_feature_order):
    """Process simpleclust merge simpleclust cc preprocessed."""
    cc_preprocessed = None
    for cc_modality in preproc_artifact.get('modalities_in_output', []):
        if cc_modality not in dict_final_cc:
            continue
        cc_modality_df = dict_final_cc[cc_modality].copy()
        cc_feature_columns = [
            column for column in cc_modality_df.columns
            if column != 'src_subject_id'
        ]
        prefixed_df = cc_modality_df.rename(columns={
            column: f'{cc_modality}__{column}'
            for column in cc_feature_columns
        })
        cc_preprocessed = (
            prefixed_df
            if cc_preprocessed is None
            else cc_preprocessed.merge(
                prefixed_df,
                on='src_subject_id',
                how='inner',
                validate='one_to_one',
            )
        )

    if cc_preprocessed is None or cc_preprocessed.empty:
        raise ValueError('No aligned discovery CC modalities were available after preprocessing.')

    missing_features = [
        column for column in training_feature_order
        if column not in cc_preprocessed.columns
    ]
    if missing_features and len(dict_final_cc) == 1:
        # Support older/simple single-modality exports that did not prefix columns.
        only_modality_df = next(iter(dict_final_cc.values())).copy()
        unprefixed_missing = [
            column for column in training_feature_order
            if column not in only_modality_df.columns
        ]
        if not unprefixed_missing:
            cc_preprocessed = only_modality_df
            missing_features = []

    if missing_features:
        raise ValueError(
            f'CC preprocessing is missing {len(missing_features)} training features; '
            f'examples: {missing_features[:10]}'
        )

    return cc_preprocessed[['src_subject_id'] + training_feature_order]


@_with_notebook_context
def simpleclust_build_cc_preprocessed_for_dimred_run(run_metrics):
    """Process simpleclust build cc preprocessed for dimred run."""
    preproc_artifact = run_metrics.get('preprocessing_details')
    if preproc_artifact is None:
        raise ValueError('Run does not contain preprocessing_details.')

    _, _, dict_final_cc = apply_preprocessing_to_new_data(
        cleaned_discovery_CC.copy(),
        meta,
        preproc_artifact,
        subject_id_column='src_subject_id',
    )
    training_feature_order = [
        column for column in run_metrics['data'].columns
        if column != 'src_subject_id'
    ]
    return simpleclust_merge_simpleclust_cc_preprocessed(
        dict_final_cc=dict_final_cc,
        preproc_artifact=preproc_artifact,
        training_feature_order=training_feature_order,
    )


@_with_notebook_context
def simpleclust_display_saved_boxplot_pages(out_dir, file_prefix):
    """Process simpleclust display saved boxplot pages."""
    pattern = os.path.join(out_dir, f'{safe_name(file_prefix)}_boxplots_page_*.png')
    for png_path in sorted(glob.glob(pattern)):
        display(Image(filename=png_path))


@_with_notebook_context
def simpleclust_cluster_size_summary(labels):
    """Process simpleclust cluster size summary."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return {}
    return pd.Series(labels).value_counts().sort_index().to_dict()


@_with_notebook_context
def simpleclust_load_alternative_k_results():
    """Process simpleclust load alternative k results."""
    loaded = {}
    seen_dirs = set()
    for run_name, run_payload in sorted(dimred_runs.items()):
        search_roots = [Path(run_payload['results_dir']), Path(run_payload['base'])]
        for search_root in search_roots:
            if not search_root.exists():
                continue
            for alt_dir in sorted(search_root.glob('alternative_k*')):
                if not alt_dir.is_dir() or alt_dir.resolve() in seen_dirs:
                    continue
                seen_dirs.add(alt_dir.resolve())
                final_path = alt_dir / 'final' / 'final_metrics.pkl'
                failure_path = alt_dir / 'merge_failure.json'
                result_key = f'{run_name}:{alt_dir.name}'
                payload = {
                    'dimred_option': run_name,
                    'alternative_k': alt_dir.name,
                    'result_key': result_key,
                    'alt_dir': str(alt_dir),
                    'final_metrics_path': str(final_path),
                    'final_metrics': None,
                    'merge_failure_path': str(failure_path) if failure_path.exists() else None,
                    'provenance_files': {
                        name: str(alt_dir / name)
                        for name in [
                            'synthetic_fold_manifest.csv',
                            'candidate_ranking_k.csv',
                            'selected_fold_candidates.csv',
                            'selection_summary.json',
                        ]
                        if (alt_dir / name).exists()
                    },
                }
                if final_path.exists():
                    with open(final_path, 'rb') as f:
                        payload['final_metrics'] = pickle.load(f)
                loaded[result_key] = payload
    return loaded


@_with_notebook_context
def simpleclust_merge_validation_modalities_for_svm(feature_dict, modalities_to_use, subject_ids, subject_id_column='src_subject_id'):
    """Process simpleclust merge validation modalities for svm."""
    merged = None
    for modality in modalities_to_use:
        if modality not in feature_dict:
            continue
        modality_df = feature_dict[modality].copy()
        if subject_id_column not in modality_df.columns:
            raise ValueError(f"{modality} validation data is missing {subject_id_column}.")
        if modality_df[subject_id_column].tolist() != subject_ids:
            modality_df = (
                modality_df
                .set_index(subject_id_column)
                .loc[subject_ids]
                .reset_index()
            )
        feature_columns = [column for column in modality_df.columns if column != subject_id_column]
        modality_df = modality_df.rename(columns={
            column: f'{modality}__{column}'
            for column in feature_columns
            if not str(column).startswith(f'{modality}__')
        })
        merged = modality_df if merged is None else merged.merge(
            modality_df,
            on=subject_id_column,
            how='inner',
            validate='one_to_one',
        )
    if merged is None or merged.empty:
        raise ValueError('No preprocessed validation feature data available for SVM prediction.')
    return merged


@_with_notebook_context
def simpleclust_merge_validation_feature_dict(feature_dict, modalities_to_use, subject_ids, subject_id_column='src_subject_id'):
    """Process simpleclust merge validation feature dict."""
    merged = None
    for modality in modalities_to_use:
        if modality not in feature_dict:
            continue
        modality_df = feature_dict[modality].copy()
        if subject_id_column not in modality_df.columns:
            raise ValueError(f"{modality} validation data is missing {subject_id_column}.")
        if modality_df[subject_id_column].tolist() != subject_ids:
            modality_df = (
                modality_df
                .set_index(subject_id_column)
                .loc[subject_ids]
                .reset_index()
            )
        feature_columns = [column for column in modality_df.columns if column != subject_id_column]
        modality_df = modality_df.rename(columns={
            column: f'{modality}__{column}'
            for column in feature_columns
            if not str(column).startswith(f'{modality}__')
        })
        merged = modality_df if merged is None else merged.merge(
            modality_df,
            on=subject_id_column,
            how='inner',
            validate='one_to_one',
        )
    if merged is None or merged.empty:
        raise ValueError('No validation feature data available for ANOVA plots.')
    return merged


@_with_notebook_context
def simpleclust_validation_cc_source_dataframe():
    """Process simpleclust validation cc source dataframe."""
    if 'cleaned_test_CC' in globals() and isinstance(cleaned_test_CC, pd.DataFrame) and not cleaned_test_CC.empty:
        return cleaned_test_CC.copy(), 'held-out/test CC split'
    if 'test_data_CC' in globals() and isinstance(test_data_CC, pd.DataFrame) and not test_data_CC.empty:
        return test_data_CC.copy(), 'held-out/test CC split before missingness filtering'
    if 'cleaned_discovery_CC' in globals() and isinstance(cleaned_discovery_CC, pd.DataFrame) and not cleaned_discovery_CC.empty:
        return cleaned_discovery_CC.copy(), 'discovery CC fallback'
    return None, None


@_with_notebook_context
def simpleclust_preprocess_cc_like_validation_chr(cc_source_df, cc_source_label):
    """Process simpleclust preprocess cc like validation chr."""
    if cc_source_df is None or cc_source_df.empty:
        print('No CC dataframe is available for validation plots; CC overlays will be skipped.')
        return None
    if 'preproc_artifact' not in globals() or preproc_artifact is None:
        raise ValueError('preproc_artifact is required to align validation CC to the validation CHR feature space.')
    _, _, dict_final_validation_cc = apply_preprocessing_to_new_data(
        cc_source_df.copy(),
        meta,
        preproc_artifact,
        subject_id_column=subject_id_column,
    )
    validation_cc = simpleclust_merge_validation_feature_dict(
        dict_final_validation_cc,
        modalities_for_feature_plots,
        dict_final_validation_cc[modalities_for_feature_plots[0]][subject_id_column].tolist(),
        subject_id_column=subject_id_column,
    )
    missing_validation_cc_features = [
        column for column in validation_data_for_plots.columns
        if column != subject_id_column and column not in validation_cc.columns
    ]
    if missing_validation_cc_features:
        raise ValueError(
            f'Validation CC preprocessing is missing {len(missing_validation_cc_features)} validation CHR features; '
            f'examples: {missing_validation_cc_features[:10]}'
        )
    validation_cc = validation_cc[[subject_id_column] + [c for c in validation_data_for_plots.columns if c != subject_id_column]]
    print(f'Validation CC preprocessing complete using {cc_source_label}:', validation_cc.shape)
    return validation_cc


@_with_notebook_context
def simpleclust_plot_validation_feature_mean_heatmap(feature_matrix, labels, out_dir, file_prefix, title, features=None, control_matrix=None):
    """Process simpleclust plot validation feature mean heatmap."""
    if features is None:
        features = order_simpleclust_features(feature_matrix.columns)
    features = [feature for feature in features if feature in feature_matrix.columns]
    if not features:
        print(f'{title}: no available features for heatmap.')
        return None
    plot_matrix = feature_matrix[features].apply(pd.to_numeric, errors='coerce')
    plot_matrix = plot_matrix.loc[:, plot_matrix.notna().any(axis=0)]
    if plot_matrix.empty:
        print(f'{title}: no numeric non-empty features for heatmap.')
        return None
    use_features = plot_matrix.columns.tolist()
    combined_for_scale = plot_matrix.copy()
    control_numeric = None
    if control_matrix is not None:
        control_numeric = (
            control_matrix.reindex(columns=use_features)
            .apply(pd.to_numeric, errors='coerce')
            .dropna(axis=0, how='all')
        )
        if not control_numeric.empty:
            combined_for_scale = pd.concat([combined_for_scale, control_numeric], ignore_index=True)
    standardized = plot_matrix.sub(combined_for_scale.mean(axis=0), axis=1).div(
        combined_for_scale.std(axis=0, ddof=0).replace(0, np.nan),
        axis=1,
    ).fillna(0)
    heatmap_df = standardized.assign(_subgroup=pd.Series(labels).astype(str).to_numpy())
    mean_matrix = heatmap_df.groupby('_subgroup', observed=False)[standardized.columns].mean()
    mean_matrix = mean_matrix.reindex(sorted(mean_matrix.index, key=lambda value: str(value)))
    if control_numeric is not None and not control_numeric.empty:
        control_standardized = control_numeric.sub(combined_for_scale.mean(axis=0), axis=1).div(
            combined_for_scale.std(axis=0, ddof=0).replace(0, np.nan),
            axis=1,
        ).fillna(0)
        mean_matrix.loc['CC'] = control_standardized.mean(axis=0)
    mean_matrix.columns = [display_feature_name(feature) for feature in mean_matrix.columns]

    fig_width = min(max(12, 0.38 * mean_matrix.shape[1] + 3), 44)
    fig_height = max(3.5, 0.55 * mean_matrix.shape[0] + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        mean_matrix,
        cmap='vlag',
        center=0,
        linewidths=0.25,
        linecolor='white',
        cbar_kws={'label': 'Mean z-score'},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Predicted validation subgroup / CC')
    ax.tick_params(axis='x', rotation=70, labelsize=8)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        output_path = os.path.join(out_dir, f'{file_prefix}.{ext}')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print('Saved validation feature heatmap to:', output_path)
    plt.show()
    return mean_matrix


# -----------------------------------------------------------------------------
# clinical_paper/Clinical_main_work.ipynb analysis helpers
# Implementations are preserved from the original notebook cells.
# -----------------------------------------------------------------------------

@_with_notebook_context
def clinical_analysis_save_figure_png_pdf(fig, output_path, dpi=300, **savefig_kwargs):
    """Process clinical analysis save figure png pdf."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for suffix in (".png", ".pdf"):
        target = output_path.with_suffix(suffix)
        fig.savefig(target, dpi=dpi, **savefig_kwargs)
        saved_paths.append(target)
    return saved_paths


@_with_notebook_context
def clinical_analysis_extract_dictionary_codes(note):
    """Process clinical analysis extract dictionary codes."""
    if pd.isna(note) or note == "":
        return []
    codes = []
    for raw in re.findall(r"(-?\d+(?:\.\d+)?)\s*=\s*[^,;]+", str(note)):
        try:
            code = float(raw)
        except ValueError:
            continue
        if code not in ignore_keys:
            codes.append(code)
    return sorted(set(codes))


@_with_notebook_context
def clinical_analysis_should_convert_to_categorical(series, max_unique_numeric=20):
    """Process clinical analysis should convert to categorical."""
    non_missing = series.dropna()
    if non_missing.empty:
        return False
    if not pd.api.types.is_numeric_dtype(non_missing):
        return True

    numeric = pd.to_numeric(non_missing, errors="coerce")
    if numeric.isna().any():
        return False
    n_unique = numeric.nunique(dropna=True)
    integer_like = np.all(np.isclose(numeric, np.round(numeric)))
    return n_unique <= max_unique_numeric and integer_like


@_with_notebook_context
def clinical_analysis_pairwise_correlation_level(abs_r):
    """Process clinical analysis pairwise correlation level."""
    if pd.isna(abs_r):
        return np.nan
    if abs_r >= 0.90:
        return "severe"
    if abs_r >= 0.70:
        return "moderate_high"
    if abs_r >= 0.50:
        return "moderate"
    return "low"


@_with_notebook_context
def clinical_analysis_vif_level(vif):
    """Process clinical analysis vif level."""
    if pd.isna(vif):
        return np.nan
    if np.isinf(vif) or vif >= 10:
        return "critical"
    if vif >= 5:
        return "high"
    if vif >= 2.5:
        return "moderate"
    return "low"


@_with_notebook_context
def clinical_analysis_condition_index_level(condition_index):
    """Process clinical analysis condition index level."""
    if pd.isna(condition_index):
        return np.nan
    if np.isinf(condition_index) or condition_index >= 30:
        return "critical"
    if condition_index >= 10:
        return "moderate_strong"
    return "low"


@_with_notebook_context
def clinical_analysis_prepare_numeric_domain_matrix(data, variables, min_pairwise_n=3):
    """Process clinical analysis prepare numeric domain matrix."""
    X = data[variables].apply(pd.to_numeric, errors="coerce")
    valid_vars = [
        col for col in X.columns
        if X[col].notna().sum() >= min_pairwise_n and X[col].nunique(dropna=True) > 1
    ]
    X = X[valid_vars]
    if X.empty:
        return X

    X = X.dropna(axis=0, how="all")
    X = X.fillna(X.median(numeric_only=True))
    valid_vars = [col for col in X.columns if X[col].nunique(dropna=True) > 1]
    return X[valid_vars]


@_with_notebook_context
def clinical_analysis_compute_vif_table(X, domain_name):
    """Process clinical analysis compute vif table."""
    rows = []
    if X.shape[1] < 2:
        return rows

    X_values = X.to_numpy(dtype=float)
    X_values = (X_values - np.nanmean(X_values, axis=0)) / np.nanstd(X_values, axis=0, ddof=0)
    X_values = np.nan_to_num(X_values, nan=0.0, posinf=0.0, neginf=0.0)

    for idx, variable in enumerate(X.columns):
        y = X_values[:, idx]
        others = np.delete(X_values, idx, axis=1)
        design = np.column_stack([np.ones(others.shape[0]), others])
        try:
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            y_hat = design @ coef
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            r_squared = min(max(r_squared, 0.0), 1.0) if np.isfinite(r_squared) else np.nan
            tolerance = 1.0 - r_squared if np.isfinite(r_squared) else np.nan
            vif = 1.0 / tolerance if np.isfinite(tolerance) and tolerance > 0 else np.inf
        except np.linalg.LinAlgError:
            r_squared = np.nan
            tolerance = 0.0
            vif = np.inf

        rows.append({
            "domain": domain_name,
            "variable": variable,
            "vif": vif,
            "tolerance": tolerance,
            "r_squared_with_other_variables": r_squared,
            "vif_level": clinical_analysis_vif_level(vif),
        })
    return rows


@_with_notebook_context
def clinical_analysis_compute_condition_index(X):
    """Process clinical analysis compute condition index."""
    if X.shape[1] < 2:
        return np.nan, np.nan, np.nan

    X_values = X.to_numpy(dtype=float)
    X_values = (X_values - np.nanmean(X_values, axis=0)) / np.nanstd(X_values, axis=0, ddof=0)
    X_values = np.nan_to_num(X_values, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.corrcoef(X_values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)

    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    max_eigenvalue = float(np.max(eigenvalues)) if eigenvalues.size else np.nan
    positive_eigenvalues = eigenvalues[eigenvalues > 1e-12]
    min_positive_eigenvalue = float(np.min(positive_eigenvalues)) if positive_eigenvalues.size else np.nan

    if not np.isfinite(max_eigenvalue) or not np.isfinite(min_positive_eigenvalue):
        condition_index = np.nan
    elif min_positive_eigenvalue <= 0:
        condition_index = np.inf
    else:
        condition_index = float(np.sqrt(max_eigenvalue / min_positive_eigenvalue))
    return condition_index, max_eigenvalue, min_positive_eigenvalue


@_with_notebook_context
def clinical_analysis_assess_domain_collinearity(
    data,
    meta=None,
    preprocessed_modalities=None,
    domain_col="Modality",
    variable_col="ElementName",
    id_cols=("src_subject_id", "phenotype"),
    correlation_methods=("pearson", "spearman"),
    high_corr_threshold=0.90,
    moderate_corr_threshold=0.70,
    min_pairwise_n=3,
):
    """Summarise within-domain collinearity for raw data or preprocessed modality matrices."""
    id_cols = set(id_cols)
    if preprocessed_modalities is not None:
        domain_items = [
            (domain_name, [c for c in df_mod.columns if c not in id_cols])
            for domain_name, df_mod in preprocessed_modalities.items()
        ]
    else:
        if meta is None:
            raise ValueError("meta is required when preprocessed_modalities is not provided.")
        required_cols = {domain_col, variable_col}
        missing_meta_cols = required_cols.difference(meta.columns)
        if missing_meta_cols:
            raise KeyError(f"Missing metadata column(s): {sorted(missing_meta_cols)}")
        domain_map = (
            meta.loc[meta[variable_col].isin(data.columns), [variable_col, domain_col]]
            .dropna(subset=[domain_col])
            .drop_duplicates()
            .rename(columns={variable_col: "variable", domain_col: "domain"})
        )
        domain_items = [
            (domain_name, [v for v in domain_df["variable"].tolist() if v not in id_cols])
            for domain_name, domain_df in domain_map.groupby("domain", sort=True)
        ]
    pair_rows = []
    vif_rows = []
    condition_rows = []
    summary_rows = []
    skipped_rows = []

    for domain_name, variables in domain_items:
        source_df = preprocessed_modalities[domain_name] if preprocessed_modalities is not None else data
        X = clinical_analysis_prepare_numeric_domain_matrix(source_df, variables, min_pairwise_n=min_pairwise_n)
        numeric_vars = X.columns.tolist()

        for variable in sorted(set(variables).difference(numeric_vars)):
            skipped_rows.append({
                "domain": domain_name,
                "variable": variable,
                "reason": "not numeric/coercible or insufficient variation",
            })

        corr_counts = {method: 0 for method in correlation_methods}
        corr_max = {method: np.nan for method in correlation_methods}
        corr_mean = {method: np.nan for method in correlation_methods}

        if len(numeric_vars) >= 2:
            for method in correlation_methods:
                corr = source_df[numeric_vars].apply(pd.to_numeric, errors="coerce").corr(
                    method=method,
                    min_periods=min_pairwise_n,
                )
                abs_corr = corr.abs()
                upper_mask = np.triu(np.ones(abs_corr.shape, dtype=bool), k=1)
                upper_values = abs_corr.where(upper_mask).stack().dropna()
                flagged_pairs = upper_values[upper_values >= moderate_corr_threshold].sort_values(ascending=False)

                corr_counts[method] = int((upper_values >= high_corr_threshold).sum())
                corr_max[method] = upper_values.max() if len(upper_values) else np.nan
                corr_mean[method] = upper_values.mean() if len(upper_values) else np.nan

                for (variable_1, variable_2), abs_value in flagged_pairs.items():
                    pair_rows.append({
                        "domain": domain_name,
                        "method": method,
                        "variable_1": variable_1,
                        "variable_2": variable_2,
                        "correlation": corr.loc[variable_1, variable_2],
                        "abs_correlation": abs_value,
                        "correlation_level": clinical_analysis_pairwise_correlation_level(abs_value),
                    })

            vif_rows.extend(clinical_analysis_compute_vif_table(X, domain_name))
            condition_index, max_eigenvalue, min_eigenvalue = clinical_analysis_compute_condition_index(X)
        else:
            condition_index, max_eigenvalue, min_eigenvalue = np.nan, np.nan, np.nan

        condition_rows.append({
            "domain": domain_name,
            "condition_index": condition_index,
            "condition_index_level": clinical_analysis_condition_index_level(condition_index),
            "max_eigenvalue": max_eigenvalue,
            "min_positive_eigenvalue": min_eigenvalue,
        })

        domain_vifs = [row["vif"] for row in vif_rows if row["domain"] == domain_name and np.isfinite(row["vif"])]
        domain_tolerances = [row["tolerance"] for row in vif_rows if row["domain"] == domain_name and np.isfinite(row["tolerance"])]
        summary_rows.append({
            "domain": domain_name,
            "n_variables_total": len(variables),
            "n_variables_numeric": len(numeric_vars),
            "n_severe_pearson_pairs_abs_r_ge_0_90": corr_counts.get("pearson", 0),
            "n_severe_spearman_pairs_abs_r_ge_0_90": corr_counts.get("spearman", 0),
            "max_abs_pearson": corr_max.get("pearson", np.nan),
            "max_abs_spearman": corr_max.get("spearman", np.nan),
            "mean_abs_pearson": corr_mean.get("pearson", np.nan),
            "mean_abs_spearman": corr_mean.get("spearman", np.nan),
            "max_vif": max(domain_vifs) if domain_vifs else np.nan,
            "min_tolerance": min(domain_tolerances) if domain_tolerances else np.nan,
            "n_variables_vif_ge_2_5": sum(v >= 2.5 for v in domain_vifs),
            "n_variables_vif_ge_5": sum(v >= 5 for v in domain_vifs),
            "n_variables_vif_ge_10": sum(v >= 10 for v in domain_vifs),
            "condition_index": condition_index,
            "condition_index_level": clinical_analysis_condition_index_level(condition_index),
        })

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(
            ["n_severe_spearman_pairs_abs_r_ge_0_90", "max_vif", "condition_index", "domain"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    correlation_pairs = (
        pd.DataFrame(pair_rows)
        .sort_values(["abs_correlation", "domain", "method", "variable_1", "variable_2"], ascending=[False, True, True, True, True])
        .reset_index(drop=True)
        if pair_rows else
        pd.DataFrame(columns=["domain", "method", "variable_1", "variable_2", "correlation", "abs_correlation", "correlation_level"])
    )
    vif_table = (
        pd.DataFrame(vif_rows)
        .sort_values(["vif", "domain", "variable"], ascending=[False, True, True])
        .reset_index(drop=True)
        if vif_rows else
        pd.DataFrame(columns=["domain", "variable", "vif", "tolerance", "r_squared_with_other_variables", "vif_level"])
    )
    condition_index_table = (
        pd.DataFrame(condition_rows)
        .sort_values(["condition_index", "domain"], ascending=[False, True])
        .reset_index(drop=True)
        if condition_rows else
        pd.DataFrame(columns=["domain", "condition_index", "condition_index_level", "max_eigenvalue", "min_positive_eigenvalue"])
    )
    skipped_variables = (
        pd.DataFrame(skipped_rows)
        .sort_values(["domain", "variable"])
        .reset_index(drop=True)
        if skipped_rows else
        pd.DataFrame(columns=["domain", "variable", "reason"])
    )

    return {
        "summary": summary,
        "correlation_pairs": correlation_pairs,
        "vif": vif_table,
        "condition_index": condition_index_table,
        "skipped_variables": skipped_variables,
    }


@_with_notebook_context
def clinical_analysis_infer_notebook_profile():
    """Infer the clinical profile using its original directory settings."""
    return infer_notebook_profile_from_settings({
        "clinical_paper": "clinical_paper",
        "multiclust_extended": "multiclust_extended",
        "prospect": "prospect",
        "schizbull_legacy": "schizbull_legacy",
    })


@_with_notebook_context
def clinical_analysis_display_merged_feature_name(feature):
    """Process clinical analysis display merged feature name."""
    text = str(feature)
    if "::" in text:
        modality, name = text.split("::", 1)
        return f"{modality}: {display_feature_name(name)}"
    return display_feature_name(text)


@_with_notebook_context
def clinical_analysis_add_metadata_and_clusters_individual_labels(final_metrics, data_full, mod_num):
    """
    Merge cluster labels into full metadata DataFrame using src_subject_id
    from the actual training data stored in fold_metrics['data'].
    This ensures correct alignment even when not all train_ids were clustered.
    """
    clusters = pd.Series(final_metrics['individual_labels'][mod_num])

    # Get the first modality’s dataframe — all have the same subject order
    modality_dfs = final_metrics.get('data', {})
    if not modality_dfs:
        raise ValueError("final_metrics['data'] is empty; cannot extract subject IDs.")

    # Use the src_subject_id column from the first available modality
    first_modality = list(modality_dfs.keys())[0]
    subj_ids = modality_dfs[first_modality]['src_subject_id'].reset_index(drop=True)

    # Sanity check: should match cluster array length
    if len(subj_ids) != len(clusters):
        print(f"⚠️ Mismatch: {len(subj_ids)} subject IDs vs {len(clusters)} cluster labels.")
        min_len = min(len(subj_ids), len(clusters))
        subj_ids = subj_ids.iloc[:min_len]
        clusters = clusters.iloc[:min_len]
        print(f"Trimmed both to {min_len} entries to align.")

    # Build cluster mapping dataframe
    cluster_df = pd.DataFrame({
        'src_subject_id': subj_ids,
        'Cluster': clusters
    })

    # Merge back into full metadata
    merged = pd.merge(data_full, cluster_df, on='src_subject_id', how='left')

    print(f"✅ Merged clusters for {merged['Cluster'].notna().sum()} subjects (out of {len(merged)} total).")
    return merged


@_with_notebook_context
def clinical_analysis_chi_square_comparison_individual_labels(df, group_col, label_col, title_prefix):
    """
    Perform chi-square test and plot grouped bar chart.
    """
    # Drop missing values
    df = df.dropna(subset=[group_col, label_col])

    # Make a copy for plotting / stats so we can safely relabel
    df_plot = df.copy()

    # Rename 'HC' -> 'CC' only for phenotype (for plotting and stats)
    if label_col == 'phenotype':
        df_plot[label_col] = df_plot[label_col].replace({'HC': 'CC'})

    # 1. Summarize counts
    summary_df = (
        df_plot.groupby([group_col, label_col])
        .size()
        .reset_index(name='n')
    )

    # 2. Chi-square test
    tbl = pd.crosstab(df_plot[group_col], df_plot[label_col])
    chi2, pval, dof, expected = chi2_contingency(tbl)
    pval = round(pval, 3)

    # 3. Grouped bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=summary_df,
        x=group_col,
        y='n',
        hue=label_col,
        dodge=True
    )
    plt.title(f"{title_prefix}\n(Chi-square p = {pval})")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    sns.despine()
    plt.tight_layout()
    plt.show()


@_with_notebook_context
def clinical_analysis_add_metadata_and_clusters_final_labels(final_metrics, data_full):
    """
    Merge cluster labels into full metadata DataFrame using src_subject_id
    from the actual training data stored in fold_metrics['data'].
    This ensures correct alignment even when not all train_ids were clustered.
    """
    clusters = pd.Series(final_metrics['final_labels'])

    # Get the first modality’s dataframe — all have the same subject order
    modality_dfs = final_metrics.get('data', {})
    if not modality_dfs:
        raise ValueError("final_metrics['data'] is empty; cannot extract subject IDs.")

    # Use the src_subject_id column from the first available modality
    first_modality = list(modality_dfs.keys())[0]
    subj_ids = modality_dfs[first_modality]['src_subject_id'].reset_index(drop=True)

    # Sanity check: should match cluster array length
    if len(subj_ids) != len(clusters):
        print(f"⚠️ Mismatch: {len(subj_ids)} subject IDs vs {len(clusters)} cluster labels.")
        min_len = min(len(subj_ids), len(clusters))
        subj_ids = subj_ids.iloc[:min_len]
        clusters = clusters.iloc[:min_len]
        print(f"Trimmed both to {min_len} entries to align.")

    # Build cluster mapping dataframe
    cluster_df = pd.DataFrame({
        'src_subject_id': subj_ids,
        'Cluster': clusters
    })

    # Merge back into full metadata
    merged = pd.merge(data_full, cluster_df, on='src_subject_id', how='left')

    print(f"✅ Merged clusters for {merged['Cluster'].notna().sum()} subjects (out of {len(merged)} total).")
    return merged


@_with_notebook_context
def clinical_analysis_chi_square_comparison_final_labels(df, group_col, label_col, title_prefix, save_path):
    """
    Perform chi-square test and plot grouped bar chart.
    """
    # Drop missing values
    df = df.dropna(subset=[group_col, label_col])

    # Make a copy for plotting / stats so we can safely relabel
    df_plot = df.copy()

    # Rename 'HC' -> 'CC' only for phenotype (for plotting and stats)
    if label_col == 'phenotype':
        df_plot[label_col] = df_plot[label_col].replace({'HC': 'CC'})

    # 1. Summarize counts
    summary_df = (
        df_plot.groupby([group_col, label_col])
        .size()
        .reset_index(name='n')
    )

    # 2. Chi-square test
    tbl = pd.crosstab(df_plot[group_col], df_plot[label_col])
    chi2, pval, dof, expected = chi2_contingency(tbl)
    pval = round(pval, 3)

    # 3. Grouped bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=summary_df,
        x=group_col,
        y='n',
        hue=label_col,
        dodge=True
    )
    plt.title(f"{title_prefix}\n(Chi-square p = {pval})")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    sns.despine()
    plt.tight_layout()
    clinical_analysis_save_figure_png_pdf(plt.gcf(), save_path, dpi=1000)
    plt.show()

    # 4. Cluster summary (including CHR percentage if relevant)
    cluster_summary = (
        df_plot.groupby(group_col)
        .agg(
            Size=(group_col, 'size'),
            CHR_percentage=(
                label_col,
                lambda x: (x.eq('CHR').sum() / len(x)) * 100
                if 'CHR' in x.values else None
            )
        )
        .reset_index()
        .sort_values(group_col)
    )
    print(f"\nCluster summary for {title_prefix}")
    print(cluster_summary)
    print("\n" + "-"*60)


@_with_notebook_context
def clinical_analysis_summarize_streams(df_paths, stage_order, top_k=20, sample_ids=10):
    """Process clinical analysis summarize streams."""
    group_cols = stage_order + ["final"]
    total = len(df_paths)

    g = (df_paths
         .groupby(group_cols, dropna=False)
         .agg(
             n=("src_subject_id", "size"),
             example_ids=("src_subject_id", lambda x: list(x.head(sample_ids))),
         )
         .reset_index()
        )

    g["pct"] = (g["n"] / total * 100).round(2)

    # A readable "stream label"
    def make_stream_label(row):
        parts = [f"{c}={row[c]}" for c in stage_order] + [f"final={row['final']}"]
        return " → ".join(parts)

    g["stream"] = g.apply(make_stream_label, axis=1)

    g = g.sort_values("n", ascending=False).head(top_k)

    # Reorder columns nicely
    cols = ["n", "pct", "stream", "example_ids"] + stage_order + ["final"]
    return g[cols]


@_with_notebook_context
def clinical_analysis_safe_svm_plot_name(name):
    """Process clinical analysis safe svm plot name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "plot"


@_with_notebook_context
def clinical_analysis_save_svm_figure_png_svg(fig, filename, dpi=300, **savefig_kwargs):
    """Process clinical analysis save svm figure png svg."""
    output_stem = svm_plots_dir / Path(filename).stem
    saved_paths = []
    for suffix in (".png", ".svg"):
        target = output_stem.with_suffix(suffix)
        fig.savefig(target, dpi=dpi, **savefig_kwargs)
        saved_paths.append(target)
    return saved_paths


@_with_notebook_context
def clinical_analysis_iter_discovery_svm_results(final_metrics):
    """Process clinical analysis iter discovery svm results."""
    modality_names = list(final_metrics.get("data", {}).keys())
    raw_results = final_metrics.get("svm_results_modalities", [])

    if isinstance(raw_results, dict):
        for mod_name in modality_names:
            yield mod_name, raw_results.get(mod_name)
        extra = [name for name in raw_results if name not in set(modality_names)]
        if extra:
            print("SVM modality results not present in final_metrics['data']:", extra)
        return

    raw_results = list(raw_results or [])
    if len(raw_results) != len(modality_names):
        print(
            "Warning: svm_results_modalities has "
            f"{len(raw_results)} entries for {len(modality_names)} discovery modalities. "
            "Trailing modalities without stored SVM results will be reported explicitly."
        )
    for idx, mod_name in enumerate(modality_names):
        res = raw_results[idx] if idx < len(raw_results) else None
        yield mod_name, res


@_with_notebook_context
def clinical_analysis_plot_pred_modality(df, name):
    """
    Plots confidence/uncertainty diagnostics for a single modality DataFrame.
    Expected columns (any subset is ok): p_0, p_1, confidence, entropy, margin
    """
    cols = set(df.columns)

    # Compute missing fields if probabilities exist
    if {"p_0", "p_1"}.issubset(cols):
        p0 = df["p_0"].astype(float).to_numpy()
        p1 = df["p_1"].astype(float).to_numpy()

        if "confidence" not in cols:
            df = df.copy()
            df["confidence"] = np.maximum(p0, p1)

        if "margin" not in cols:
            df = df.copy()
            df["margin"] = np.abs(p1 - p0)

        if "entropy" not in cols:
            df = df.copy()
            eps = 1e-12
            df["entropy"] = -(p0 * np.log(p0 + eps) + p1 * np.log(p1 + eps))

    # Helper to plot a histogram if column exists
    def hist_if_exists(col, bins=30, xlabel=None):
        if col in df.columns:
            fig = plt.figure()
            plt.hist(df[col].dropna().astype(float), bins=bins)
            plt.xlabel(xlabel or col)
            plt.ylabel("Count")
            plt.title(f"{name}: {col} distribution (unlabeled hold-out)")
            clinical_analysis_save_svm_figure_png_svg(fig, f"holdout_{clinical_analysis_safe_svm_plot_name(name)}_{clinical_analysis_safe_svm_plot_name(col)}_distribution", dpi=300, bbox_inches="tight")
            plt.show()

    # 1) Histograms
    hist_if_exists("p_1", xlabel="Predicted probability p(class=1)")
    hist_if_exists("confidence", xlabel="Confidence = max(p_0, p_1)")
    hist_if_exists("entropy", xlabel="Entropy (higher = more uncertain)")
    hist_if_exists("margin", xlabel="Margin = |p_1 - p_0| (lower = more uncertain)")

    # 2) Scatter plots that are often informative
    if ("confidence" in df.columns) and ("entropy" in df.columns):
        fig = plt.figure()
        plt.scatter(df["confidence"].astype(float), df["entropy"].astype(float), s=10)
        plt.xlabel("Confidence")
        plt.ylabel("Entropy")
        plt.title(f"{name}: confidence vs entropy")
        clinical_analysis_save_svm_figure_png_svg(fig, f"holdout_{clinical_analysis_safe_svm_plot_name(name)}_confidence_vs_entropy", dpi=300, bbox_inches="tight")
        plt.show()

    if ("confidence" in df.columns) and ("margin" in df.columns):
        fig = plt.figure()
        plt.scatter(df["confidence"].astype(float), df["margin"].astype(float), s=10)
        plt.xlabel("Confidence")
        plt.ylabel("Margin")
        plt.title(f"{name}: confidence vs margin")
        clinical_analysis_save_svm_figure_png_svg(fig, f"holdout_{clinical_analysis_safe_svm_plot_name(name)}_confidence_vs_margin", dpi=300, bbox_inches="tight")
        plt.show()

    # 3) Print “most uncertain” cases (lowest confidence / lowest margin / highest entropy)
    # (useful for manual review)
    print(f"\n=== {name}: most uncertain examples ===")

    if "confidence" in df.columns:
        print("\nLowest confidence:")
        display(df.sort_values("confidence", ascending=True).head(10))

    if "margin" in df.columns:
        print("\nLowest margin:")
        display(df.sort_values("margin", ascending=True).head(10))

    if "entropy" in df.columns:
        print("\nHighest entropy:")
        display(df.sort_values("entropy", ascending=False).head(10))


@_with_notebook_context
def clinical_analysis_add_metadata_and_clusters_validation_individual_labels(modality_data, data_full, mod_num):
    """
    Merge cluster labels into full metadata using src_subject_id from the
    modality dataframe that was clustered. The notebook calls this function
    once per modality, passing dict_final[modality].
    """
    clusters = pd.Series(labels_test_modalities[mod_num]).reset_index(drop=True)

    if isinstance(modality_data, pd.DataFrame):
        modality_df = modality_data
    elif isinstance(modality_data, dict):
        if len(modality_data) == 0:
            raise ValueError("modality_data is empty; cannot extract subject IDs.")
        first_modality = next(iter(modality_data))
        modality_df = modality_data[first_modality]
    else:
        raise TypeError(
            "modality_data must be a pandas DataFrame or a dict of modality DataFrames; "
            f"got {type(modality_data).__name__}."
        )

    if modality_df.empty:
        raise ValueError("modality_data is empty; cannot extract subject IDs.")
    if "src_subject_id" not in modality_df.columns:
        raise KeyError("modality_data must include a 'src_subject_id' column.")

    subj_ids = modality_df["src_subject_id"].reset_index(drop=True)

    # Sanity check: should match cluster array length
    if len(subj_ids) != len(clusters):
        print(f"Warning: {len(subj_ids)} subject IDs vs {len(clusters)} cluster labels.")
        min_len = min(len(subj_ids), len(clusters))
        subj_ids = subj_ids.iloc[:min_len]
        clusters = clusters.iloc[:min_len]
        print(f"Trimmed both to {min_len} entries to align.")

    cluster_df = pd.DataFrame({
        "src_subject_id": subj_ids,
        "Cluster": clusters,
    })

    merged = pd.merge(data_full, cluster_df, on="src_subject_id", how="left")

    print(f"Merged clusters for {merged['Cluster'].notna().sum()} subjects (out of {len(merged)} total).")
    return merged


@_with_notebook_context
def clinical_analysis_chi_square_comparison_validation_individual_labels(df, group_col, label_col, title_prefix):
    """
    Perform chi-square test and plot grouped bar chart.
    """
    # Drop missing values
    df = df.dropna(subset=[group_col, label_col])

    # Make a copy for plotting / stats so we can safely relabel
    df_plot = df.copy()

    # Rename 'HC' -> 'CC' only for phenotype (for plotting and stats)
    if label_col == 'phenotype':
        df_plot[label_col] = df_plot[label_col].replace({'HC': 'CC'})

    # 1. Summarize counts
    summary_df = (
        df_plot.groupby([group_col, label_col])
        .size()
        .reset_index(name='n')
    )

    # 2. Chi-square test
    tbl = pd.crosstab(df_plot[group_col], df_plot[label_col])
    chi2, pval, dof, expected = chi2_contingency(tbl)
    pval = round(pval, 3)

    # 3. Grouped bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=summary_df,
        x=group_col,
        y='n',
        hue=label_col,
        dodge=True
    )
    plt.title(f"{title_prefix}\n(Chi-square p = {pval})")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    sns.despine()
    plt.tight_layout()
    plt.show()


@_with_notebook_context
def clinical_analysis_as_domain_map_label_series(values, expected_n, label_name, sample_name):
    """Process clinical analysis as domain map label series."""
    series = pd.Series(values).astype(str).reset_index(drop=True)
    if len(series) != expected_n:
        raise ValueError(
            f"{sample_name}: {label_name} has {len(series)} labels for {expected_n} subjects."
        )
    return series.str.replace(" ", "_", regex=False)


@_with_notebook_context
def clinical_analysis_domain_map_path_table(labels_by_modality, final_labels, sample_name, stage_order_for_labels, final_name="Integrated"):
    """Process clinical analysis domain map path table."""
    available_stages = [stage for stage in stage_order_for_labels if stage in labels_by_modality]
    if not available_stages:
        raise ValueError(f"{sample_name}: no domain labels were available for comparison.")

    n_subjects = len(pd.Series(labels_by_modality[available_stages[0]]))
    frame = pd.DataFrame({
        stage: clinical_analysis_as_domain_map_label_series(labels_by_modality[stage], n_subjects, stage, sample_name)
        for stage in available_stages
    })
    frame[final_name] = clinical_analysis_as_domain_map_label_series(final_labels, n_subjects, final_name, sample_name)
    frame["domain_path"] = frame[available_stages].agg(" -> ".join, axis=1)
    frame["integrated_path"] = frame[available_stages + [final_name]].agg(" -> ".join, axis=1)

    path_table = (
        frame.groupby(available_stages + [final_name, "domain_path", "integrated_path"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    path_table["sample"] = sample_name
    path_table["proportion"] = path_table["n"] / float(n_subjects)
    return frame, path_table, available_stages


@_with_notebook_context
def clinical_analysis_compare_domain_map_paths(discovery_table, validation_table, group_cols, label):
    """Process clinical analysis compare domain map paths."""
    discovery_counts = (
        discovery_table.groupby(group_cols, dropna=False)["n"]
        .sum()
        .reset_index(name="n_discovery")
    )
    validation_counts = (
        validation_table.groupby(group_cols, dropna=False)["n"]
        .sum()
        .reset_index(name="n_validation")
    )
    comparison = discovery_counts.merge(validation_counts, on=group_cols, how="outer").fillna(0)
    comparison["n_discovery"] = comparison["n_discovery"].astype(int)
    comparison["n_validation"] = comparison["n_validation"].astype(int)
    comparison["proportion_discovery"] = comparison["n_discovery"] / max(1, int(discovery_table["n"].sum()))
    comparison["proportion_validation"] = comparison["n_validation"] / max(1, int(validation_table["n"].sum()))
    comparison["proportion_difference_validation_minus_discovery"] = (
        comparison["proportion_validation"] - comparison["proportion_discovery"]
    )
    comparison["abs_proportion_difference"] = comparison[
        "proportion_difference_validation_minus_discovery"
    ].abs()
    comparison["path_present_in"] = np.select(
        [
            comparison["n_discovery"].gt(0) & comparison["n_validation"].gt(0),
            comparison["n_discovery"].gt(0),
            comparison["n_validation"].gt(0),
        ],
        ["both", "discovery_only", "validation_only"],
        default="neither",
    )
    comparison = comparison.sort_values(
        ["abs_proportion_difference", "proportion_validation", "proportion_discovery"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    comparison.to_csv(
        os.path.join(_DOMAIN_MAP_COMPARISON_DIR, f"{label}_domain_map_path_comparison.csv"),
        index=False,
    )
    return comparison


@_with_notebook_context
def clinical_analysis_plot_domain_map_path_comparison(comparison, path_col, title, filename_prefix, top_n=25):
    """Process clinical analysis plot domain map path comparison."""
    plot_df = comparison.copy()
    plot_df["max_proportion"] = plot_df[["proportion_discovery", "proportion_validation"]].max(axis=1)
    plot_df = plot_df.sort_values("max_proportion", ascending=False, kind="mergesort").head(top_n)
    if plot_df.empty:
        print(f"Skipping {filename_prefix}: no paths available to plot.")
        return None

    plot_df = plot_df.sort_values("max_proportion", ascending=True, kind="mergesort")
    y = np.arange(plot_df.shape[0])
    bar_height = 0.38
    fig_height = max(6, 0.34 * plot_df.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(y - bar_height / 2, plot_df["proportion_discovery"], height=bar_height, label="Discovery", color="#38699A")
    ax.barh(y + bar_height / 2, plot_df["proportion_validation"], height=bar_height, label="Validation", color="#B36F9C")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df[path_col], fontsize=8)
    ax.set_xlabel("Sample proportion")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    output_stem = os.path.join(_DOMAIN_MAP_COMPARISON_DIR, filename_prefix)
    fig.savefig(output_stem + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(output_stem + ".pdf", bbox_inches="tight")
    print("Saved:", output_stem + ".png")
    print("Saved:", output_stem + ".pdf")
    return fig


@_with_notebook_context
def clinical_analysis_parse_stream(stream_str):
    """
    Parse stream like:
      "Psychoticism=low_severity → Detachment=high_severity → ... → final=1"
    into ordered list of (domain, label).
    """
    if pd.isna(stream_str):
        return []
    parts = [p.strip() for p in ARROW_PAT.split(str(stream_str).strip()) if p.strip()]
    out = []
    for p in parts:
        if "=" in p:
            dom, lab = p.split("=", 1)
            out.append((dom.strip(), lab.strip()))
        else:
            out.append((p.strip(), "<NA>"))
    return out


@_with_notebook_context
def clinical_analysis_infer_stage_order(df, stream_col="stream"):
    """
    Infer stage order from the first non-null stream.
    Assumes all streams follow the same domain order.
    """
    s = df[stream_col].dropna().astype(str)
    if s.empty:
        raise ValueError("No streams found to infer stage order.")
    path = clinical_analysis_parse_stream(s.iloc[0])
    return [d for d, _ in path]


@_with_notebook_context
def clinical_analysis_normalize(v, eps=1e-12):
    """Process clinical analysis normalize."""
    v = np.asarray(v, dtype=float)
    s = v.sum()
    if s <= 0:
        return np.ones_like(v) / max(1, len(v))
    return v / (s + eps)


@_with_notebook_context
def clinical_analysis_build_prefix_next(df, stream_col="stream", n_col="n"):
    """
    Returns:
      prefix_next: dict[prefix_tuple_of_(domain,label)] -> dict[next_token_(domain,label)|<END>] -> weight
      prefix_mass: dict[prefix] -> total weight passing through prefix
    Where prefix is a tuple of (domain,label) tokens, e.g.:
      prefix = (("Psychoticism","low_severity"), ("Detachment","high_severity"))
    and next token is the next (domain,label) or "<END>".
    """
    prefix_next = {}
    prefix_mass = {}

    for _, row in df.iterrows():
        w = float(row.get(n_col, 1.0))
        path = clinical_analysis_parse_stream(row[stream_col])
        if not path:
            continue

        # For each prefix (including empty prefix), record what comes next
        for i in range(len(path)):
            prefix = tuple(path[:i])  # empty prefix for i=0
            nxt = path[i]             # next token
            prefix_mass[prefix] = prefix_mass.get(prefix, 0.0) + w
            d = prefix_next.setdefault(prefix, {})
            d[nxt] = d.get(nxt, 0.0) + w

        # terminal transition from full path to END
        full_prefix = tuple(path)
        prefix_mass[full_prefix] = prefix_mass.get(full_prefix, 0.0) + w
        d = prefix_next.setdefault(full_prefix, {})
        d["<END>"] = d.get("<END>", 0.0) + w

    return prefix_next, prefix_mass


@_with_notebook_context
def clinical_analysis_compare_prefix_structure(df_disc, df_test, stream_col="stream", n_col="n", eps=1e-12):
    """
    For each prefix, compare the conditional distribution over next tokens:
        P_disc(next | prefix)  vs  P_test(next | prefix)

    Returns DataFrame with:
      - prefix_str
      - depth
      - js_next (Jensen–Shannon distance on next-step distributions)
      - mass_disc / mass_test (how much data passes through prefix; useful for weighting but not "size-equality")
      - top_next_disc / top_next_test (most likely next step)
      - support_next_overlap (Jaccard on next-token supports)
    """
    pn_d, pm_d = clinical_analysis_build_prefix_next(df_disc, stream_col, n_col)
    pn_t, pm_t = clinical_analysis_build_prefix_next(df_test, stream_col, n_col)

    prefixes = set(pn_d.keys()) | set(pn_t.keys())

    rows = []
    for pref in prefixes:
        nd = pn_d.get(pref, {})
        nt = pn_t.get(pref, {})

        keys = set(nd.keys()) | set(nt.keys())
        # Align next-token vectors
        vd = np.array([nd.get(k, 0.0) for k in keys], dtype=float)
        vt = np.array([nt.get(k, 0.0) for k in keys], dtype=float)

        pd_ = clinical_analysis_normalize(vd, eps=eps)
        pt_ = clinical_analysis_normalize(vt, eps=eps)

        js = float(jensenshannon(pd_ + eps, pt_ + eps, base=2.0))

        # Next-token support overlap (presence/absence, structure)
        supp_d = {k for k, v in nd.items() if v > 0}
        supp_t = {k for k, v in nt.items() if v > 0}
        supp_j = len(supp_d & supp_t) / max(1, len(supp_d | supp_t))

        # Most likely next token in each
        top_d = max(nd.items(), key=lambda x: x[1])[0] if nd else None
        top_t = max(nt.items(), key=lambda x: x[1])[0] if nt else None

        def tok_str(tok):
            if tok == "<END>":
                return "<END>"
            if tok is None:
                return "<NONE>"
            return f"{tok[0]}={tok[1]}"

        prefix_str = " → ".join([f"{d}={l}" for d, l in pref]) if pref else "<START>"
        rows.append({
            "prefix_str": prefix_str,
            "depth": len(pref),
            "js_next": js,
            "mass_disc": pm_d.get(pref, 0.0),
            "mass_test": pm_t.get(pref, 0.0),
            "top_next_disc": tok_str(top_d),
            "top_next_test": tok_str(top_t),
            "support_next_jaccard": supp_j,
            "prefix_exists_in_disc": pref in pn_d,
            "prefix_exists_in_test": pref in pn_t,
        })

    out = pd.DataFrame(rows)
    # A useful default sorting: prioritize structurally-different AND commonly-used prefixes
    out["mass_min"] = np.minimum(out["mass_disc"], out["mass_test"])
    out = out.sort_values(["mass_min", "js_next"], ascending=[False, False]).reset_index(drop=True)
    return out


@_with_notebook_context
def clinical_analysis_plot_top_prefix_differences(prefix_report, top_n=20, min_depth=1):
    """
    Barh plot of top prefixes by JS(next) after filtering.
    """
    d = prefix_report[prefix_report["depth"] >= min_depth].copy()
    d = d.sort_values("js_next", ascending=False).head(top_n)

    plt.figure(figsize=(11, max(4, 0.35 * len(d))))
    y = np.arange(len(d))[::-1]
    plt.barh(y, d["js_next"].iloc[::-1].to_numpy())
    plt.yticks(y, d["prefix_str"].iloc[::-1].to_list(), fontsize=9)
    plt.xlabel("Jensen–Shannon distance of P(next | prefix)")
    plt.title("Most structurally different prefixes (next-step rule)")
    plt.tight_layout()
    plt.show()


@_with_notebook_context
def clinical_analysis_final_mapping_table(df, stream_col="stream", n_col="n", final_domain="final"):
    """
    Builds a table over modality-prefixes (everything up to but excluding final)
    with P(final=label | prefix) computed within each dataset.

    Returns DataFrame:
      prefix_str, final_label, weight
    and also a pivoted table of P(final=...) by prefix.
    """
    rows = []
    for _, row in df.iterrows():
        w = float(row.get(n_col, 1.0))
        path = clinical_analysis_parse_stream(row[stream_col])
        if not path:
            continue

        # split into prefix (before final) and final token
        final_tokens = [t for t in path if t[0] == final_domain]
        if not final_tokens:
            # If final isn't explicitly present, skip
            continue
        final_tok = final_tokens[-1]  # in case of duplicates, take last
        final_label = final_tok[1]

        # prefix = all tokens before the final token position (first occurrence)
        # more robust: take all tokens except the final-domain token(s)
        prefix = tuple([t for t in path if t[0] != final_domain])

        prefix_str = " → ".join([f"{d}={l}" for d, l in prefix]) if prefix else "<NO_MODALITIES>"
        rows.append({"prefix_str": prefix_str, "final_label": final_label, "w": w})

    long = pd.DataFrame(rows)
    if long.empty:
        return long, pd.DataFrame()

    # Compute conditional probabilities P(final_label | prefix)
    grp = long.groupby(["prefix_str", "final_label"], dropna=False)["w"].sum().reset_index()
    totals = grp.groupby("prefix_str")["w"].sum().reset_index().rename(columns={"w": "w_total"})
    grp = grp.merge(totals, on="prefix_str", how="left")
    grp["p_final_given_prefix"] = grp["w"] / grp["w_total"]

    pivot = grp.pivot_table(index="prefix_str", columns="final_label", values="p_final_given_prefix", fill_value=0.0)
    return grp, pivot


@_with_notebook_context
def clinical_analysis_compare_final_mapping(df_disc, df_test, stream_col="stream", n_col="n", final_domain="final"):
    """
    Compare P(final | modalities) between discovery and test.
    Returns a table with per-prefix deltas per final label plus summary metrics.
    """
    long_d, piv_d = clinical_analysis_final_mapping_table(df_disc, stream_col, n_col, final_domain)
    long_t, piv_t = clinical_analysis_final_mapping_table(df_test, stream_col, n_col, final_domain)

    empty_out = pd.DataFrame(columns=["prefix_str", "js_final_given_prefix", "w_disc", "w_test", "w_min"])
    if piv_d.empty or piv_t.empty:
        return empty_out, {"note": "No final mapping found (missing final tokens?)", "num_prefixes_union": 0}

    # align
    idx = sorted(set(piv_d.index) | set(piv_t.index))
    cols = sorted(set(piv_d.columns) | set(piv_t.columns))
    A = piv_d.reindex(index=idx, columns=cols).fillna(0.0)
    B = piv_t.reindex(index=idx, columns=cols).fillna(0.0)

    # Delta per final label
    delta = (B - A)
    out = delta.copy()
    out.columns = [f"delta_final={c}" for c in out.columns]
    out.insert(0, "prefix_str", out.index)

    # A per-prefix summary: JS distance between final distributions for that prefix
    js_list = []
    for i in range(A.shape[0]):
        p = clinical_analysis_normalize(A.iloc[i].to_numpy())
        q = clinical_analysis_normalize(B.iloc[i].to_numpy())
        js_list.append(float(jensenshannon(p + 1e-12, q + 1e-12, base=2.0)))
    out["js_final_given_prefix"] = js_list

    # Add weights: how common the prefix is (within each dataset)
    wD = long_d.groupby("prefix_str")["w"].sum() if not long_d.empty else pd.Series(dtype=float)
    wT = long_t.groupby("prefix_str")["w"].sum() if not long_t.empty else pd.Series(dtype=float)
    out["w_disc"] = out["prefix_str"].map(wD).fillna(0.0)
    out["w_test"] = out["prefix_str"].map(wT).fillna(0.0)
    out["w_min"] = np.minimum(out["w_disc"], out["w_test"])

    # Sort by (common prefixes) then by biggest JS shift in final mapping
    out = out.sort_values(["w_min", "js_final_given_prefix"], ascending=[False, False]).reset_index(drop=True)

    # Global summary metric: weighted average JS over prefixes (weights = w_min)
    w = out["w_min"].to_numpy()
    if w.sum() > 0:
        global_js = float(np.sum(out["js_final_given_prefix"].to_numpy() * w) / w.sum())
    else:
        global_js = float(out["js_final_given_prefix"].mean())

    metrics = {
        "num_prefixes_union": len(out),
        "weighted_js_final_given_prefix": global_js,
        "final_labels": cols,
    }
    return out, metrics


@_with_notebook_context
def clinical_analysis_plot_top_final_mapping_shifts(final_cmp, top_n=20):
    """Process clinical analysis plot top final mapping shifts."""
    required = {"prefix_str", "js_final_given_prefix"}
    if final_cmp is None or final_cmp.empty:
        print("No final-mapping shifts to plot.")
        return None
    missing = required - set(final_cmp.columns)
    if missing:
        print(f"No final-mapping shifts to plot; missing columns: {sorted(missing)}")
        return None
    d = final_cmp.head(top_n).copy()
    plt.figure(figsize=(11, max(4, 0.35 * len(d))))
    y = np.arange(len(d))[::-1]
    plt.barh(y, d["js_final_given_prefix"].iloc[::-1].to_numpy())
    plt.yticks(y, d["prefix_str"].iloc[::-1].to_list(), fontsize=9)
    plt.xlabel("JS distance between P(final | prefix) in test vs discovery")
    plt.title("Prefixes with biggest changes in final mapping")
    plt.tight_layout()
    plt.show()


@_with_notebook_context
def clinical_analysis_stream_presence_and_topk(df_disc, df_test, stream_col="stream", n_col="n", topk=30):
    """Process clinical analysis stream presence and topk."""
    A = set(df_disc[stream_col].dropna().astype(str))
    B = set(df_test[stream_col].dropna().astype(str))

    presence = {
        "unique_streams_disc": len(A),
        "unique_streams_test": len(B),
        "stream_jaccard_presence": len(A & B) / max(1, len(A | B)),
        "coverage_disc_in_test": len(A & B) / max(1, len(A)),
        "coverage_test_in_disc": len(A & B) / max(1, len(B)),
    }

    pD = df_disc.groupby(stream_col)[n_col].sum().sort_values(ascending=False)
    pT = df_test.groupby(stream_col)[n_col].sum().sort_values(ascending=False)
    pD = pD / pD.sum()
    pT = pT / pT.sum()

    topD = set(pD.index[: min(topk, len(pD))].astype(str))
    topT = set(pT.index[: min(topk, len(pT))].astype(str))

    presence[f"top{topk}_jaccard_by_rank"] = len(topD & topT) / max(1, len(topD | topT))
    presence["top_disc_only"] = sorted(list(topD - topT))[:10]
    presence["top_test_only"] = sorted(list(topT - topD))[:10]
    return presence, pD, pT


@_with_notebook_context
def clinical_analysis_sankey_from_streams(df, stream_col="stream", n_col="n", max_edges=200):
    """
    Build a Sankey graph from full streams.
    Nodes are stage-specific label tokens: f"{domain}={label}".
    Edges connect consecutive tokens. We prune to max_edges by weight.
    """
    if not _HAS_PLOTLY:
        raise RuntimeError("Plotly not installed; cannot draw sankey.")

    edge_w = {}
    for _, row in df.iterrows():
        w = float(row.get(n_col, 1.0))
        path = clinical_analysis_parse_stream(row[stream_col])
        toks = [f"{d}={l}" for d, l in path]
        for a, b in zip(toks[:-1], toks[1:]):
            edge_w[(a, b)] = edge_w.get((a, b), 0.0) + w

    # prune
    edges = sorted(edge_w.items(), key=lambda x: -x[1])[:max_edges]
    nodes = {}
    def nid(x):
        if x not in nodes:
            nodes[x] = len(nodes)
        return nodes[x]

    src, tgt, val = [], [], []
    for (a, b), w in edges:
        src.append(nid(a))
        tgt.append(nid(b))
        val.append(w)

    labels = [None] * len(nodes)
    for k, i in nodes.items():
        labels[i] = k

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=12, thickness=12),
        link=dict(source=src, target=tgt, value=val),
    )])
    return fig


@_with_notebook_context
def clinical_analysis_full_structure_report(stream_summary, stream_summary_test, stream_col="stream", n_col="n", topk=30, final_domain="final"):
    # Presence + top-k overlap
    """Process clinical analysis full structure report."""
    presence, pD, pT = clinical_analysis_stream_presence_and_topk(stream_summary, stream_summary_test, stream_col, n_col, topk=topk)

    # Prefix-tree structural differences
    prefix_report = clinical_analysis_compare_prefix_structure(stream_summary, stream_summary_test, stream_col, n_col)

    # Full mapping to final
    final_cmp, final_metrics = clinical_analysis_compare_final_mapping(stream_summary, stream_summary_test, stream_col, n_col, final_domain)

    return {
        "presence_metrics": presence,
        "p_stream_disc": pD,
        "p_stream_test": pT,
        "prefix_report": prefix_report,
        "final_mapping_compare": final_cmp,
        "final_mapping_metrics": final_metrics,
    }


@_with_notebook_context
def clinical_analysis_summarize_streams_for_comparison(df_paths, stage_order, top_k=100, sample_ids=12, final_col="final"):
    """Build canonical stream labels for discovery/test comparison."""
    group_cols = list(stage_order) + ([final_col] if final_col in df_paths.columns else [])
    total = len(df_paths)

    g = (
        df_paths
        .groupby(group_cols, dropna=False)
        .agg(
            n=("src_subject_id", "size"),
            example_ids=("src_subject_id", lambda x: list(x.head(sample_ids))),
        )
        .reset_index()
    )
    g["pct"] = (g["n"] / total * 100).round(2) if total else 0.0

    def make_stream_label(row):
        parts = [f"{stage}={row[stage]}" for stage in stage_order]
        if final_col in row.index:
            parts.append(f"{final_col}={row[final_col]}")
        return " → ".join(parts)

    g["stream"] = g.apply(make_stream_label, axis=1)
    g = g.sort_values("n", ascending=False).head(top_k)

    cols = ["n", "pct", "stream", "example_ids"] + group_cols
    return g[cols]


@_with_notebook_context
def clinical_analysis_all_streams_table(stream_summary, stream_summary_test, stream_col="stream", n_col="n"):
    # Aggregate in case there are duplicate stream rows
    """Process clinical analysis all streams table."""
    disc = (stream_summary
            .groupby(stream_col, dropna=False)[n_col]
            .sum()
            .rename("n_disc")
            .to_frame())

    test = (stream_summary_test
            .groupby(stream_col, dropna=False)[n_col]
            .sum()
            .rename("n_test")
            .to_frame())

    # Outer join gives union of streams
    tbl = disc.join(test, how="outer").fillna(0)

    # Add proportions within each dataset
    N_disc = tbl["n_disc"].sum()
    N_test = tbl["n_test"].sum()

    tbl["p_disc"] = tbl["n_disc"] / N_disc if N_disc > 0 else np.nan
    tbl["p_test"] = tbl["n_test"] / N_test if N_test > 0 else np.nan

    # Helpful comparisons
    tbl["delta_p"] = tbl["p_test"] - tbl["p_disc"]
    tbl["abs_delta_p"] = tbl["delta_p"].abs()
    tbl["log2_fc"] = np.log2((tbl["p_test"] + 1e-12) / (tbl["p_disc"] + 1e-12))

    # Make stream a real column, sort by biggest shift
    tbl = tbl.reset_index().rename(columns={stream_col: "stream"})
    tbl = tbl.sort_values("abs_delta_p", ascending=False).reset_index(drop=True)

    return tbl


@_with_notebook_context
def clinical_analysis_subgroup_display_label(name):
    """Process clinical analysis subgroup display label."""
    return SUBGROUP_DISPLAY_LABELS.get(str(name), str(name).replace("_", " "))


@_with_notebook_context
def clinical_analysis_subject_ids_from_modality(data_by_modality, stage_order, subject_id_column="src_subject_id"):
    """Process clinical analysis subject ids from modality."""
    first_stage = stage_order[0]
    if first_stage not in data_by_modality:
        raise KeyError(f"Missing {first_stage} in modality data.")
    return data_by_modality[first_stage][subject_id_column].astype(str).reset_index(drop=True)


@_with_notebook_context
def clinical_analysis_build_label_frame_from_current_objects(sample):
    """Fallback only: rebuild labels if the saved release label CSV is unavailable."""
    if sample == "discovery":
        if "df_paths" in globals():
            label_df = df_paths.copy()
        else:
            subject_ids = clinical_analysis_subject_ids_from_modality(final_metrics["data"], stage_order)
            label_df = pd.DataFrame({"src_subject_id": subject_ids})
            for stage in stage_order:
                label_df[stage] = pd.Series(new_labels_by_modality[stage]).astype(str).reset_index(drop=True)
            label_df["final"] = pd.Series(final_metrics["final_labels"]).astype(str).reset_index(drop=True)
    elif sample == "test":
        if "df_paths_test" in globals():
            label_df = df_paths_test.copy()
        else:
            subject_ids = clinical_analysis_subject_ids_from_modality(dict_final, stage_order)
            label_df = pd.DataFrame({"src_subject_id": subject_ids})
            for stage in stage_order:
                label_df[stage] = pd.Series(new_test_labels_by_modality[stage]).astype(str).reset_index(drop=True)
            label_df["final"] = pd.Series(labels_test_final).astype(str).reset_index(drop=True)
    else:
        raise ValueError("sample must be 'discovery' or 'test'")

    keep = ["src_subject_id"] + [c for c in SUBGROUP_VARS if c in label_df.columns]
    label_df = label_df[keep].copy()
    label_df["src_subject_id"] = label_df["src_subject_id"].astype(str)
    return label_df


@_with_notebook_context
def clinical_analysis_load_saved_label_frame(path, sample):
    """Match the Rmd: read the saved label CSVs produced by the final label-export cell."""
    if path.exists():
        label_df = pd.read_csv(path)
        source = str(path)
    else:
        print(f"WARNING: {path} does not exist; rebuilding {sample} labels from current notebook objects.")
        label_df = clinical_analysis_build_label_frame_from_current_objects(sample)
        label_df.to_csv(path, index=False)
        source = f"rebuilt and saved to {path}"

    required = ["src_subject_id"] + SUBGROUP_VARS
    missing = [col for col in required if col not in label_df.columns]
    if missing:
        raise KeyError(f"{sample} labels are missing required columns: {missing}. Source: {source}")

    label_df = label_df[required].copy()
    label_df["src_subject_id"] = label_df["src_subject_id"].astype(str)
    for col in SUBGROUP_VARS:
        label_df[col] = label_df[col].astype(str)
    print(f"Loaded {sample} labels from {source}: {label_df.shape}")
    return label_df


@_with_notebook_context
def clinical_analysis_resolve_post_analysis_data():
    """Return the raw merged dataframe used by the Rmd as `merged_data`."""
    for name in ["merged_data", "data_all", "data"]:
        obj = globals().get(name)
        if isinstance(obj, pd.DataFrame) and "src_subject_id" in obj.columns:
            print(f"Using `{name}` as the Rmd-style merged_data dataframe.")
            return obj.copy()

    merged_path = Path(
        "path/to/"
        "multiclust_data/merged_data.csv"
    )
    if merged_path.exists():
        print(f"Reloading Rmd-style merged_data from {merged_path}.")
        return pd.read_csv(merged_path)

    available = {
        name: type(globals().get(name)).__name__
        for name in ["merged_data", "data_all", "data"]
        if name in globals()
    }
    raise KeyError(
        "Could not find the Rmd-style merged_data dataframe with `src_subject_id`. "
        f"Available candidates: {available}"
    )


@_with_notebook_context
def clinical_analysis_bh_fdr(pvalues):
    """Adjust p-values using the clinical notebook's coercion policy."""
    return adjust_pvalues_benjamini_hochberg(pvalues, coerce_numeric=True)


@_with_notebook_context(default_expressions={'dictionary_path': 'DICTIONARY_DIR_DIFF'})
def clinical_analysis_load_dictionary_labels(dictionary_path=None):
    """Process clinical analysis load dictionary labels."""
    try:
        d = pd.read_excel(dictionary_path)
    except Exception as exc:
        print(f"Could not read dictionary at {dictionary_path}: {exc}")
        return {}, []
    name_col = "ElementName" if "ElementName" in d.columns else d.columns[0]
    label_col = next((c for c in ["ElementDescription", "Description", "Label", "Variable", "VariableName"] if c in d.columns), name_col)
    labels = dict(zip(d[name_col].astype(str), d[label_col].astype(str)))
    ordered = [v for v in d[name_col].astype(str).tolist()]
    return labels, ordered


@_with_notebook_context
def clinical_analysis_metadata_variable_table(meta_df=None):
    """Process clinical analysis metadata variable table."""
    global _METADATA_VARIABLE_TABLE_CACHE
    use_cache = meta_df is None
    if use_cache and _METADATA_VARIABLE_TABLE_CACHE is not None:
        return _METADATA_VARIABLE_TABLE_CACHE.copy()
    meta_df = globals().get("meta") if meta_df is None else meta_df
    if not isinstance(meta_df, pd.DataFrame) or "ElementName" not in meta_df.columns:
        table = pd.DataFrame(columns=["canonical_variable", "Modality", "meta_row"])
    else:
        keep_columns = [column for column in ["ElementName", "Modality"] if column in meta_df.columns]
        table = meta_df[keep_columns].copy().reset_index().rename(columns={"index": "meta_row"})
        table["canonical_variable"] = table["ElementName"].astype(str)
        table["Modality"] = table.get("Modality", pd.Series("Other / Unmapped", index=table.index)).fillna("Other / Unmapped").astype(str)
        table = table.loc[~table["canonical_variable"].isin(HEATMAP_EXCLUDED_VARIABLES)]
        table = table.drop_duplicates("canonical_variable", keep="first")[["canonical_variable", "Modality", "meta_row"]]
    if use_cache:
        _METADATA_VARIABLE_TABLE_CACHE = table.copy()
    return table


@_with_notebook_context(default_expressions={'dictionary_path': 'DICTIONARY_DIR_DIFF'})
def clinical_analysis_dictionary_alias_table(dictionary_path=None):
    """Process clinical analysis dictionary alias table."""
    cache_key = str(dictionary_path)
    if cache_key in _DICTIONARY_ALIAS_TABLE_CACHE:
        return _DICTIONARY_ALIAS_TABLE_CACHE[cache_key].copy()
    try:
        dict_df = pd.read_excel(dictionary_path)
    except Exception:
        out = pd.DataFrame(columns=["variable", "canonical_variable", "alias_priority"])
        _DICTIONARY_ALIAS_TABLE_CACHE[cache_key] = out.copy()
        return out
    if "ElementName" not in dict_df.columns:
        out = pd.DataFrame(columns=["variable", "canonical_variable", "alias_priority"])
        _DICTIONARY_ALIAS_TABLE_CACHE[cache_key] = out.copy()
        return out
    rows = []
    for _, row in dict_df.iterrows():
        canonical = str(row["ElementName"]) if pd.notna(row["ElementName"]) else ""
        if not canonical:
            continue
        rows.append({"variable": canonical, "canonical_variable": canonical, "alias_priority": 0})
        if "Aliases" in dict_df.columns and pd.notna(row.get("Aliases")):
            for alias in re.split(r"[;|,\n\r]+", str(row.get("Aliases"))):
                alias = alias.strip()
                if alias:
                    rows.append({"variable": alias, "canonical_variable": canonical, "alias_priority": 1})
    if not rows:
        out = pd.DataFrame(columns=["variable", "canonical_variable", "alias_priority"])
    else:
        out = pd.DataFrame(rows).drop_duplicates(["variable", "canonical_variable"], keep="first")
    _DICTIONARY_ALIAS_TABLE_CACHE[cache_key] = out.copy()
    return out


@_with_notebook_context
def clinical_analysis_variable_canonical_lookup(meta_df=None):
    """Process clinical analysis variable canonical lookup."""
    global _VARIABLE_CANONICAL_LOOKUP_CACHE
    use_cache = meta_df is None
    if use_cache and _VARIABLE_CANONICAL_LOOKUP_CACHE is not None:
        return dict(_VARIABLE_CANONICAL_LOOKUP_CACHE)
    meta_table = clinical_analysis_metadata_variable_table(meta_df)
    meta_canonicals = set(meta_table["canonical_variable"])
    alias_table = clinical_analysis_dictionary_alias_table()
    if meta_canonicals:
        alias_table = alias_table.loc[alias_table["canonical_variable"].isin(meta_canonicals)].copy()
    direct = pd.DataFrame({
        "variable": meta_table["canonical_variable"],
        "canonical_variable": meta_table["canonical_variable"],
        "alias_priority": 0,
    })
    lookup = pd.concat([direct, alias_table], ignore_index=True)
    if lookup.empty:
        out = {}
    else:
        lookup = lookup.sort_values(["variable", "alias_priority"]).drop_duplicates("variable", keep="first")
        out = dict(zip(lookup["variable"].astype(str), lookup["canonical_variable"].astype(str)))
    if use_cache:
        _VARIABLE_CANONICAL_LOOKUP_CACHE = dict(out)
    return out


@_with_notebook_context
def clinical_analysis_label_for_variable(variable):
    """Process clinical analysis label for variable."""
    canonical = clinical_analysis_variable_canonical_lookup().get(str(variable), str(variable))
    if "display_feature_name" in globals():
        return display_feature_name(canonical)
    return canonical


@_with_notebook_context
def clinical_analysis_analysis_variables(df, comparison_cols=None, prefer_dictionary=True):
    """Process clinical analysis analysis variables."""
    comparison_cols = set(comparison_cols or [])
    blocked = {
        "src_subject_id", "sample", "phenotype", "Site", "sites", "chrrecruit",
        *HEATMAP_EXCLUDED_VARIABLES, *SUBGROUP_VARS, *comparison_cols,
    }
    available = {str(c) for c in df.columns if str(c) not in blocked}
    canonical_lookup = clinical_analysis_variable_canonical_lookup()
    meta_table = clinical_analysis_metadata_variable_table()
    variables = []
    used_canonicals = set()

    if not meta_table.empty:
        alias_by_canonical = {}
        for alias, canonical in canonical_lookup.items():
            alias_by_canonical.setdefault(canonical, []).append(alias)
        for canonical in meta_table["canonical_variable"]:
            candidates = [canonical] + sorted(alias_by_canonical.get(canonical, []))
            chosen = next((candidate for candidate in candidates if candidate in available), None)
            if chosen is None or canonical in used_canonicals:
                continue
            variables.append(chosen)
            used_canonicals.add(canonical)
        return variables

    if prefer_dictionary and DICTIONARY_VARIABLE_ORDER:
        candidates = [v for v in DICTIONARY_VARIABLE_ORDER if v in available]
    else:
        candidates = [str(c) for c in df.columns if str(c) in available]
    for candidate in candidates:
        canonical = canonical_lookup.get(str(candidate), str(candidate))
        if canonical in used_canonicals:
            continue
        variables.append(candidate)
        used_canonicals.add(canonical)
    return variables


@_with_notebook_context
def clinical_analysis_is_binary_series(x):
    """Process clinical analysis is binary series."""
    vals = pd.Series(x).dropna()
    if vals.empty:
        return False
    if pd.api.types.is_numeric_dtype(vals):
        u = set(pd.to_numeric(vals, errors="coerce").dropna().unique())
        return len(u) <= 2 and u.issubset({0, 1})
    return vals.astype(str).nunique(dropna=True) == 2


@_with_notebook_context
def clinical_analysis_variable_type(x):
    """Process clinical analysis variable type."""
    vals = pd.Series(x).dropna()
    if vals.empty or vals.nunique(dropna=True) < 2:
        return None
    if clinical_analysis_is_binary_series(vals):
        return "binary"
    numeric = pd.to_numeric(vals, errors="coerce")
    if numeric.notna().mean() >= 0.90 and numeric.nunique(dropna=True) > 2:
        return "continuous"
    return "categorical"


@_with_notebook_context
def clinical_analysis_cramers_v_from_table(tab):
    """Process clinical analysis cramers v from table."""
    arr = np.asarray(tab, dtype=float)
    n = arr.sum()
    if n == 0 or min(arr.shape) < 2:
        return np.nan
    expected = np.outer(arr.sum(axis=1), arr.sum(axis=0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.nansum((arr - expected) ** 2 / expected)
    denom = n * (min(arr.shape) - 1)
    return float(np.sqrt(stat / denom)) if denom > 0 else np.nan


@_with_notebook_context
def clinical_analysis_chi_square_test(tab):
    """Process clinical analysis chi square test."""
    arr = np.asarray(tab, dtype=float)
    if arr.shape[0] < 2 or arr.shape[1] < 2 or arr.sum() == 0:
        return np.nan, np.nan, np.nan
    if _HAS_SCIPY_STATS:
        stat, p, _, _ = _scipy_chi2_contingency(arr)
    else:
        expected = np.outer(arr.sum(axis=1), arr.sum(axis=0)) / arr.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = np.nansum((arr - expected) ** 2 / expected)
        p = np.nan
    return float(stat), float(p), clinical_analysis_cramers_v_from_table(arr)


@_with_notebook_context
def clinical_analysis_rank_anova_or_kruskal(groups):
    """Process clinical analysis rank anova or kruskal."""
    clean_groups = [pd.to_numeric(pd.Series(g), errors="coerce").dropna().to_numpy() for g in groups]
    clean_groups = [g for g in clean_groups if len(g) > 0]
    if len(clean_groups) < 2:
        return np.nan, np.nan, np.nan
    if _HAS_SCIPY_STATS:
        stat, p = _scipy_kruskal(*clean_groups)
    else:
        values = np.concatenate(clean_groups)
        ranks = pd.Series(values).rank(method="average").to_numpy()
        labels = np.concatenate([[i] * len(g) for i, g in enumerate(clean_groups)])
        grand = ranks.mean()
        ss_between = sum((ranks[labels == i].mean() - grand) ** 2 * (labels == i).sum() for i in np.unique(labels))
        ss_total = ((ranks - grand) ** 2).sum()
        stat = ss_between / max(1, len(clean_groups) - 1)
        p = np.nan
    n = sum(len(g) for g in clean_groups)
    eta_like = float(stat / max(stat + n - len(clean_groups), 1e-12)) if np.isfinite(stat) else np.nan
    return float(stat), float(p), eta_like


@_with_notebook_context
def clinical_analysis_summarize_group_differences(df, comparisons, variables=None, output_prefix=None):
    """Process clinical analysis summarize group differences."""
    rows = []
    comparisons = [c for c in comparisons if c in df.columns]
    variables = variables or clinical_analysis_analysis_variables(df, comparison_cols=comparisons)

    for comp in comparisons:
        d = df.loc[df[comp].notna()].copy()
        groups = d[comp].astype(str)
        if groups.nunique(dropna=True) < 2:
            continue
        for variable in variables:
            if variable not in d.columns or variable == comp:
                continue
            vt = clinical_analysis_variable_type(d[variable])
            if vt is None:
                continue

            sub = pd.DataFrame({"group": groups, "value": d[variable]}).dropna(subset=["value"])
            if sub["group"].nunique() < 2:
                continue

            if vt == "continuous":
                grouped = [g["value"] for _, g in sub.groupby("group", sort=True)]
                stat, p, effect = clinical_analysis_rank_anova_or_kruskal(grouped)
                summary = sub.assign(value=pd.to_numeric(sub["value"], errors="coerce")).groupby("group")["value"].agg(["count", "mean", "std", "median"]).reset_index()
                test = "Kruskal-Wallis"
                effect_name = "eta2_H"
            else:
                tab = pd.crosstab(sub["group"].astype(str), sub["value"].astype(str))
                stat, p, effect = clinical_analysis_chi_square_test(tab)
                summary = tab.reset_index()
                test = "Chi-square"
                effect_name = "cramers_v"

            rows.append({
                "comparison": comp,
                "variable": variable,
                "label": clinical_analysis_label_for_variable(variable),
                "var_type": vt,
                "test": test,
                "statistic": stat,
                "p": p,
                "effect_size": effect,
                "effect_size_name": effect_name,
                "n": int(sub.shape[0]),
                "n_groups": int(sub["group"].nunique()),
                "summary": summary.to_json(orient="records"),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["fdr"] = out.groupby("comparison")["p"].transform(clinical_analysis_bh_fdr)
    out = out.sort_values(["comparison", "fdr", "p", "label"], na_position="last").reset_index(drop=True)
    if output_prefix:
        out.to_csv(POST_ANALYSIS_DIR / f"{output_prefix}_difference_summary_long.csv", index=False)
        compact = out.assign(
            cell=lambda x: x.apply(
                lambda r: f"ES={r['effect_size']:.3g}; FDR={r['fdr']:.3g}" if pd.notna(r["fdr"]) else "",
                axis=1,
            )
        ).pivot_table(index="label", columns="comparison", values="cell", aggfunc="first").reset_index()
        compact.to_csv(POST_ANALYSIS_DIR / f"{output_prefix}_difference_summary_compact.csv", index=False)
        display(compact)
    return out


@_with_notebook_context
def clinical_analysis_demographic_summary_table(df, comparison, variables=None, output_name=None):
    """Process clinical analysis demographic summary table."""
    variables = variables or clinical_analysis_analysis_variables(df, comparison_cols=[comparison])
    rows = []
    d = df.loc[df[comparison].notna()].copy()
    for variable in variables:
        if variable not in d.columns:
            continue
        vt = clinical_analysis_variable_type(d[variable])
        if vt is None:
            continue
        if vt == "continuous":
            for group, sub in d.groupby(d[comparison].astype(str)):
                x = pd.to_numeric(sub[variable], errors="coerce")
                rows.append({
                    "comparison": comparison,
                    "group": group,
                    "variable": variable,
                    "label": clinical_analysis_label_for_variable(variable),
                    "level": "",
                    "type": vt,
                    "n": int(x.notna().sum()),
                    "summary": f"{x.mean():.2f} ({x.std():.2f})" if x.notna().any() else "",
                })
        else:
            for (group, level), n in d.groupby([d[comparison].astype(str), d[variable].astype(str)], dropna=False).size().items():
                denom = int((d[comparison].astype(str) == group).sum())
                rows.append({
                    "comparison": comparison,
                    "group": group,
                    "variable": variable,
                    "label": clinical_analysis_label_for_variable(variable),
                    "level": level,
                    "type": vt,
                    "n": int(n),
                    "summary": f"{int(n)} ({(n / denom * 100) if denom else np.nan:.1f}%)",
                })
    out = pd.DataFrame(rows)
    if output_name:
        out.to_csv(POST_ANALYSIS_DIR / output_name, index=False)
    return out


@_with_notebook_context
def clinical_analysis_positive_binary_level(x):
    """Process clinical analysis positive binary level."""
    vals = pd.Series(x).dropna().astype(str).unique().tolist()
    if len(vals) < 2:
        return None
    positives = {"1", "yes", "true", "positive", "present", "high", "case", "chr"}
    for value in vals:
        if value.lower() in positives:
            return value
    return sorted(vals)[1] if len(vals) > 1 else sorted(vals)[0]


@_with_notebook_context
def clinical_analysis_format_fdr_text(value):
    """Process clinical analysis format fdr text."""
    if pd.isna(value):
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


@_with_notebook_context
def clinical_analysis_stars_from_fdr(value):
    """Process clinical analysis stars from fdr."""
    if pd.isna(value):
        return ""
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return ""


@_with_notebook_context
def clinical_analysis_cohens_d_group_vs_rest(values, groups, group):
    """Process clinical analysis cohens d group vs rest."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    group_mask = pd.Series(groups).astype(str).eq(str(group))
    a = numeric[group_mask].dropna().to_numpy(dtype=float)
    b = numeric[~group_mask].dropna().to_numpy(dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    mean_diff = float(np.mean(a) - np.mean(b))
    if len(a) + len(b) <= 2:
        return np.nan
    pooled_var = ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2)
    if not np.isfinite(pooled_var) or pooled_var <= 0:
        return 0.0 if np.isclose(mean_diff, 0.0) else np.nan
    return mean_diff / float(np.sqrt(pooled_var))


@_with_notebook_context
def clinical_analysis_cohens_h_group_vs_rest(values, groups, group):
    """Process clinical analysis cohens h group vs rest."""
    series = pd.Series(values)
    if pd.api.types.is_numeric_dtype(series):
        binary = pd.to_numeric(series, errors="coerce")
    else:
        positive = clinical_analysis_positive_binary_level(series)
        if positive is None:
            return np.nan
        binary = series.astype(str).eq(str(positive)).astype(float)
        binary[series.isna()] = np.nan
    group_mask = pd.Series(groups).astype(str).eq(str(group))
    p_group = binary[group_mask].dropna().mean()
    p_rest = binary[~group_mask].dropna().mean()
    if not np.isfinite(p_group) or not np.isfinite(p_rest):
        return np.nan
    p_group = float(np.clip(p_group, 0.0, 1.0))
    p_rest = float(np.clip(p_rest, 0.0, 1.0))
    return 2.0 * (np.arcsin(np.sqrt(p_group)) - np.arcsin(np.sqrt(p_rest)))


@_with_notebook_context
def clinical_analysis_cramers_v_group_vs_rest(values, groups, group):
    """Process clinical analysis cramers v group vs rest."""
    valid = pd.DataFrame({"value": values, "group": pd.Series(groups).astype(str)}).dropna()
    if valid.empty:
        return np.nan
    valid["in_group"] = np.where(valid["group"].eq(str(group)), str(group), "rest")
    if valid["in_group"].nunique() < 2 or valid["value"].astype(str).nunique() < 2:
        return np.nan
    table = pd.crosstab(valid["in_group"], valid["value"].astype(str))
    return clinical_analysis_cramers_v_from_table(table)


@_with_notebook_context
def clinical_analysis_group_vs_rest_effect_size(values, groups, group, var_type):
    """Process clinical analysis group vs rest effect size."""
    if var_type == "continuous":
        return clinical_analysis_cohens_d_group_vs_rest(values, groups, group)
    if var_type == "binary":
        return clinical_analysis_cohens_h_group_vs_rest(values, groups, group)
    if var_type == "categorical":
        return clinical_analysis_cramers_v_group_vs_rest(values, groups, group)
    return np.nan


@_with_notebook_context
def clinical_analysis_effect_size_cmap():
    """Process clinical analysis effect size cmap."""
    try:
        return sns.color_palette("vlag", as_cmap=True)
    except Exception:
        return plt.get_cmap("coolwarm")


@_with_notebook_context
def clinical_analysis_readable_text_color(face_color):
    """Process clinical analysis readable text color."""
    from matplotlib import colors as mcolors
    r, g, b, _ = mcolors.to_rgba(face_color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.48 else "black"


@_with_notebook_context(default_expressions={'dictionary_path': 'DICTIONARY_DIR_DIFF'})
def clinical_analysis_load_metadata_variable_map(meta_df=None, dictionary_path=None):
    """Process clinical analysis load metadata variable map."""
    meta_table = clinical_analysis_metadata_variable_table(meta_df)
    alias_lookup = clinical_analysis_variable_canonical_lookup(meta_df)
    label_lookup = VARIABLE_LABELS if "VARIABLE_LABELS" in globals() else {}
    rows = []
    if not meta_table.empty:
        for _, row in meta_table.iterrows():
            canonical = str(row["canonical_variable"])
            rows.append({
                "variable": canonical,
                "canonical_variable": canonical,
                "Modality": row["Modality"],
                "dict_row": row["meta_row"],
                "element_short": canonical,
                "priority": 0,
            })
        for alias, canonical in alias_lookup.items():
            if alias == canonical:
                continue
            match = meta_table.loc[meta_table["canonical_variable"].eq(canonical)]
            if match.empty:
                continue
            first = match.iloc[0]
            rows.append({
                "variable": alias,
                "canonical_variable": canonical,
                "Modality": first["Modality"],
                "dict_row": first["meta_row"],
                "element_short": canonical,
                "priority": 1,
            })
    if not rows:
        dict_df = pd.read_excel(dictionary_path)
        required = {"ElementName", "Modality"}
        missing = required.difference(dict_df.columns)
        if missing:
            raise KeyError(f"Dictionary is missing required columns for heatmap ordering: {sorted(missing)}")
        aliases_col = "Aliases" if "Aliases" in dict_df.columns else None
        for i, row in dict_df.reset_index(drop=True).iterrows():
            element = str(row["ElementName"]) if pd.notna(row["ElementName"]) else ""
            modality = str(row["Modality"]) if pd.notna(row["Modality"]) else "Other / Unmapped"
            if element and element not in HEATMAP_EXCLUDED_VARIABLES:
                rows.append({"variable": element, "canonical_variable": element, "Modality": modality, "dict_row": i, "priority": 1, "element_short": element})
            if aliases_col and pd.notna(row.get(aliases_col)):
                for alias in re.split(r"[;|,\n\r]+", str(row.get(aliases_col))):
                    alias = alias.strip()
                    if alias and element not in HEATMAP_EXCLUDED_VARIABLES:
                        rows.append({"variable": alias, "canonical_variable": element or alias, "Modality": modality, "dict_row": i, "priority": 2, "element_short": element or alias})
    lookup = pd.DataFrame(rows)
    if lookup.empty:
        return pd.DataFrame(columns=["variable", "canonical_variable", "Modality", "dict_row", "element_short"])
    lookup = lookup.sort_values(["variable", "priority", "dict_row"]).drop_duplicates("variable", keep="first")
    return lookup[["variable", "canonical_variable", "Modality", "dict_row", "element_short"]]


@_with_notebook_context(default_expressions={'comparisons': 'SUBGROUP_VARS'})
def clinical_analysis_build_mixed_heatmap_data_rmd_style(df, diff_summary, comparisons=None):
    """Python analogue of the Rmd's build_mixed_heatmap_data()."""
    if diff_summary.empty:
        return pd.DataFrame()

    rows = []
    use = diff_summary.loc[diff_summary["comparison"].isin(comparisons)].copy()
    # Keep all variables returned by the demographic/difference table, as in the Rmd.
    use = use.sort_values(["comparison", "fdr", "p", "variable"], na_position="last")

    for comp in comparisons:
        comp_rows = use.loc[use["comparison"] == comp]
        if comp not in df.columns or comp_rows.empty:
            print(f"Skipping comparison {comp!r}: not found in data or no difference rows.")
            continue

        df2 = df.loc[df[comp].notna()].copy()
        df2[comp] = df2[comp].astype(str)
        grp_levels = list(pd.unique(df2[comp]))

        for _, r in comp_rows.iterrows():
            variable = r["variable"]
            if variable not in df2.columns or variable in HEATMAP_EXCLUDED_VARIABLES:
                continue
            canonical_variable = clinical_analysis_variable_canonical_lookup().get(str(variable), str(variable))
            vt = r["var_type"]
            x = df2[variable]
            for k, group in enumerate(grp_levels, start=1):
                sub = df2.loc[df2[comp] == group]
                if vt == "continuous":
                    value = pd.to_numeric(sub[variable], errors="coerce").mean()
                elif vt == "binary":
                    if pd.api.types.is_numeric_dtype(x):
                        value = pd.to_numeric(sub[variable], errors="coerce").mean()
                    else:
                        pos = clinical_analysis_positive_binary_level(x)
                        value = (sub[variable].astype(str) == pos).mean() if pos is not None else np.nan
                else:
                    value = np.nan
                effect_size = clinical_analysis_group_vs_rest_effect_size(df2[variable], df2[comp], group, vt)
                rows.append({
                    "comparison": comp,
                    "x": f"{comp} | {group}",
                    "k": k,
                    "variable": variable,
                    "canonical_variable": canonical_variable,
                    "label": r["label"],
                    "var_type": vt,
                    "value": value,
                    "effect_size_plot": effect_size,
                    "effect_size_name": {
                        "continuous": "cohens_d_group_vs_rest",
                        "binary": "cohens_h_group_vs_rest",
                        "categorical": "cramers_v_group_vs_rest",
                    }.get(vt),
                    "fdr_num": r["fdr"],
                    "fdr_text": clinical_analysis_format_fdr_text(r["fdr"]),
                })

    heat = pd.DataFrame(rows)
    if heat.empty:
        return heat

    heat = heat.sort_values(["comparison", "fdr_num", "variable"], na_position="last").drop_duplicates(
        subset=["comparison", "x", "canonical_variable", "var_type"],
        keep="first",
    )
    heat["effect_size_plot"] = pd.to_numeric(heat["effect_size_plot"], errors="coerce")
    heat["value_plot_cont"] = np.nan
    heat["value_plot_bin"] = np.nan
    heat.loc[heat["var_type"].eq("continuous"), "value_plot_cont"] = heat.loc[
        heat["var_type"].eq("continuous"), "effect_size_plot"
    ]
    heat.loc[heat["var_type"].eq("binary"), "value_plot_bin"] = heat.loc[
        heat["var_type"].eq("binary"), "effect_size_plot"
    ]
    return heat


@_with_notebook_context(default_expressions={'modality_order': 'SUBGROUP_VARS'})
def clinical_analysis_add_modality_spacers_rmd_style(heat_df, spacer_suffix=" ", modality_order=None):
    """Process clinical analysis add modality spacers rmd style."""
    if heat_df.empty:
        return heat_df
    df = heat_df.copy()
    df["modality_x"] = df["x"].astype(str).str.replace(r"\|.*$", "", regex=True).str.strip()
    observed_modalities = list(pd.unique(df["modality_x"]))
    modalities = [m for m in modality_order if m in observed_modalities]
    modalities.extend([m for m in observed_modalities if m not in modalities])
    spacer_rows = []
    for modality in modalities[:-1]:
        labels = df.loc[df["modality_x"] == modality, "label"].drop_duplicates()
        for label in labels:
            spacer_rows.append({
                "comparison": np.nan,
                "x": f"{modality}{spacer_suffix}",
                "k": np.nan,
                "variable": np.nan,
                "canonical_variable": np.nan,
                "label": label,
                "var_type": "spacer",
                "value": np.nan,
                "fdr_num": np.nan,
                "fdr_text": "",
                "value_plot_cont": np.nan,
                "value_plot_bin": np.nan,
                "effect_size_plot": np.nan,
                "effect_size_name": np.nan,
                "modality_x": modality,
            })
    if spacer_rows:
        df = pd.concat([df, pd.DataFrame(spacer_rows)], ignore_index=True)

    x_levels = []
    for modality in modalities:
        xs = df.loc[(df["modality_x"] == modality) & df["x"].astype(str).str.contains(r"\|", regex=True), "x"].drop_duplicates().tolist()
        x_levels.extend(xs)
        if modality != modalities[-1]:
            x_levels.append(f"{modality}{spacer_suffix}")
    x_pos = {}
    cur = 0.0
    for x in x_levels:
        if "|" in str(x):
            cur += 1.0
        else:
            cur += 0.20
        x_pos[x] = cur
    df["x"] = pd.Categorical(df["x"], categories=x_levels, ordered=True)
    df["x_pos"] = df["x"].astype(str).map(x_pos)
    return df.drop(columns=["modality_x"], errors="ignore")


@_with_notebook_context(default_expressions={'dictionary_path': 'DICTIONARY_DIR_DIFF'})
def clinical_analysis_heatmap_variable_map(heat_df, dictionary_path=None):
    """Process clinical analysis heatmap variable map."""
    lookup = clinical_analysis_load_metadata_variable_map(dictionary_path=dictionary_path)
    vars_present = (
        heat_df[["variable", "canonical_variable"]]
        .dropna(subset=["variable"])
        .astype(str)
        .drop_duplicates(subset=["variable"], keep="first")
    )
    mapped = vars_present.merge(lookup, on="variable", how="left", suffixes=("", "_mapped"))
    if "canonical_variable_mapped" in mapped:
        mapped["canonical_variable"] = mapped["canonical_variable_mapped"].fillna(mapped["canonical_variable"])
        mapped = mapped.drop(columns=["canonical_variable_mapped"], errors="ignore")
    mapped["Modality"] = mapped["Modality"].fillna("Other / Unmapped")
    mapped["dict_row"] = mapped["dict_row"].fillna(np.inf)
    mapped["element_short"] = mapped["element_short"].fillna(mapped["canonical_variable"]).fillna(mapped["variable"])
    mapped["element_short"] = mapped["element_short"].map(display_feature_name)
    mapped = mapped.sort_values(["dict_row", "canonical_variable", "variable"]).drop_duplicates("canonical_variable", keep="first")
    return mapped


@_with_notebook_context
def clinical_analysis_heatmap_modality_display_label(modality):
    """Process clinical analysis heatmap modality display label."""
    return str(modality).replace("_", " ")


@_with_notebook_context
def clinical_analysis_combined_panel_data(heat_df, modality_order=None):
    """Process clinical analysis combined panel data."""
    modality_order = modality_order or ["Psychoticism", "Detachment", "Internalising", "Functioning", "Cognition", "Demographics", "study_info"]
    df = heat_df.loc[~heat_df["var_type"].eq("spacer")].copy()
    if df.empty:
        return df, [], [], set()

    var_map = clinical_analysis_heatmap_variable_map(df)
    df = df.merge(var_map, on="variable", how="left")
    mods_in_panel = var_map["Modality"].dropna().drop_duplicates().tolist()
    mods_present = [m for m in modality_order if m in mods_in_panel] + [m for m in mods_in_panel if m not in modality_order and m != "Other / Unmapped"]
    if "Other / Unmapped" in mods_in_panel:
        mods_present.append("Other / Unmapped")

    y_rows = []
    for modality in mods_present:
        vars_m = (
            var_map.loc[var_map["Modality"] == modality]
            .sort_values(["dict_row", "element_short", "variable"])
        )
        if vars_m.empty:
            continue
        y_rows.append({
            "y_id": f"##{modality}",
            "y_label": clinical_analysis_heatmap_modality_display_label(modality),
            "is_header": True,
            "variable": None,
        })
        for _, row in vars_m.iterrows():
            y_rows.append({
                "y_id": f"{modality}__{row['variable']}",
                "y_label": f"  {row['element_short']}",
                "is_header": False,
                "variable": row["variable"],
            })
    y_map = pd.DataFrame(y_rows)
    y_levels = y_map["y_id"].tolist()
    y_pos = {y: len(y_levels) - 1 - i for i, y in enumerate(y_levels)}
    label_map = dict(zip(y_map["y_id"], y_map["y_label"]))
    header_set = set(y_map.loc[y_map["is_header"], "y_id"])

    df["y_id"] = df["Modality"].astype(str) + "__" + df["variable"].astype(str)
    df["y_pos"] = df["y_id"].map(y_pos)
    return df, y_levels, [label_map[y] for y in y_levels], header_set


@_with_notebook_context
def clinical_analysis_heatmap_x_display_label(x):
    """Process clinical analysis heatmap x display label."""
    text = str(x)
    if "|" not in text:
        return text
    comparison, group = [part.strip() for part in text.split("|", 1)]
    return f"{clinical_analysis_subgroup_display_label(comparison)} | {group}"


@_with_notebook_context
def clinical_analysis_plot_mixed_heatmap_split_rmd_style(
    heat_df,
    title="My heatmap ordered by modality",
    filename="heatmap_group_effect_sizes_ordered.svg",
    width=14,
    height=18,
    x_text_size=11,
    y_text_size=11,
    p_text_size=8,
    categorical_grey="#d9d9d9",
    effect_clip_quantile=0.95,
):
    """Process clinical analysis plot mixed heatmap split rmd style."""
    if heat_df.empty:
        print("No heatmap data produced.")
        return None

    x_levels = [x for x in heat_df["x"].cat.categories if "|" in str(x)]
    x_pos_df = heat_df[["x", "x_pos"]].drop_duplicates("x").copy()
    x_pos_df["x_key"] = x_pos_df["x"].astype(str)
    x_pos = x_pos_df.set_index("x_key")["x_pos"].to_dict()
    x_tick_pos = [x_pos[str(x)] for x in x_levels if str(x) in x_pos]
    if not x_tick_pos:
        raise ValueError("No real heatmap x-axis columns were found after adding spacer columns.")

    finite_effects = pd.to_numeric(heat_df["effect_size_plot"], errors="coerce")
    finite_effects = finite_effects[np.isfinite(finite_effects)]
    if len(finite_effects):
        abs_effects = np.abs(finite_effects.to_numpy(dtype=float))
        if effect_clip_quantile is None:
            effect_vmax = float(np.nanmax(abs_effects))
        else:
            effect_vmax = float(np.nanquantile(abs_effects, effect_clip_quantile))
    else:
        effect_vmax = 1.0
    effect_vmax = max(effect_vmax, 1e-6)
    effect_norm = plt.Normalize(-effect_vmax, effect_vmax, clip=True)
    effect_cmap = clinical_analysis_effect_size_cmap()

    dfp, y_levels, y_labels, header_set = clinical_analysis_combined_panel_data(heat_df)
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    ax.set_xlim(min(x_tick_pos) - 0.6, max(x_tick_pos) + 0.6)
    ax.set_ylim(-0.5, max(0, len(y_levels) - 0.5))
    ax.set_yticks(range(len(y_levels)))
    ax.set_yticklabels(list(reversed(y_labels)), fontsize=y_text_size)
    for tick, y_id in zip(ax.get_yticklabels(), reversed(y_levels)):
        if y_id in header_set:
            tick.set_fontweight("bold")
            tick.set_fontsize(13)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right"]].set_visible(False)

    if dfp.empty:
        ax.text(0.5, 0.5, "No variables", ha="center", va="center", transform=ax.transAxes)
    else:
        real = dfp.loc[dfp["y_pos"].notna()].copy()
        for idx, row in real.iterrows():
            val = pd.to_numeric(row["effect_size_plot"], errors="coerce")
            face = "white" if pd.isna(val) else effect_cmap(effect_norm(float(val)))
            ax.add_patch(plt.Rectangle((row["x_pos"] - 0.5, row["y_pos"] - 0.45), 1.0, 0.9, facecolor=face, edgecolor="white", linewidth=0.3))
        sm = plt.cm.ScalarMappable(norm=effect_norm, cmap=effect_cmap)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, extend="both")
        cbar_label = "Effect size" if effect_clip_quantile is None else f"Effect size (clipped at |ES|={effect_vmax:.2g})"
        cbar.set_label(cbar_label, fontsize=12)

        # Rmd-style annotation: stars on first group, FDR text on second group.
        for _, row in real.iterrows():
            label = ""
            if row.get("k") == 1:
                label = clinical_analysis_stars_from_fdr(row.get("fdr_num"))
            elif row.get("k") == 2 and str(row.get("fdr_text", "")):
                label = str(row.get("fdr_text"))
            if label:
                val = pd.to_numeric(row["effect_size_plot"], errors="coerce")
                face = "white" if pd.isna(val) else effect_cmap(effect_norm(float(val)))
                ax.text(row["x_pos"], row["y_pos"], label, ha="center", va="center", color=clinical_analysis_readable_text_color(face), fontsize=p_text_size)

    ax.set_xticks(x_tick_pos)
    ax.set_xticklabels([clinical_analysis_heatmap_x_display_label(x) for x in x_levels], rotation=45, ha="right", fontsize=x_text_size)
    ax.set_xlabel("Modality | group", fontsize=14)
    fig.suptitle(title, fontsize=20, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out = POST_ANALYSIS_DIR / filename
    clinical_analysis_save_figure_png_pdf(fig, out, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", out)
    return fig


@_with_notebook_context
def clinical_analysis_run_effect_size_domain_heatmap(data, difference_summary, sample_label, filename_prefix):
    """Process clinical analysis run effect size domain heatmap."""
    heat_df = clinical_analysis_build_mixed_heatmap_data_rmd_style(
        data,
        difference_summary,
        comparisons=SUBGROUP_VARS,
    )
    heat_df_spaced = clinical_analysis_add_modality_spacers_rmd_style(heat_df)
    heat_df_spaced.to_csv(POST_ANALYSIS_DIR / f"{filename_prefix}_effect_sizes_ordered_data.csv", index=False)

    fig = clinical_analysis_plot_mixed_heatmap_split_rmd_style(
        heat_df_spaced,
        title=f"{sample_label} heatmap ordered by modality",
        filename=f"{filename_prefix}_effect_sizes_ordered.svg",
        width=14,
        height=18,
        p_text_size=8,
        x_text_size=11,
        y_text_size=11,
    )
    return {"figure": fig, "data": heat_df_spaced}


@_with_notebook_context
def clinical_analysis_site_recruitment_overview_rmd_style(merged_data):
    """Match the Rmd site/recruitment section: merged_data -> sites factor + chrrecruit factor."""
    d = merged_data.copy()
    if "Site" not in d.columns or "chrrecruit" not in d.columns:
        print("Site/recruitment columns are not available in merged_data.")
        return None
    d = d.loc[d["Site"].notna() & d["chrrecruit"].notna()].copy()
    d["sites"] = d["Site"].astype(str)
    d["chrrecruit"] = d["chrrecruit"].astype(str)

    tab = pd.crosstab(d["sites"], d["chrrecruit"])
    stat, p, v = clinical_analysis_chi_square_test(tab)
    expected = pd.DataFrame(
        np.outer(tab.sum(axis=1), tab.sum(axis=0)) / tab.to_numpy().sum(),
        index=tab.index,
        columns=tab.columns,
    )
    print("Site by recruitment contingency table")
    display(tab)
    print({"chi_square": stat, "p": p, "cramers_v": v})
    tab.to_csv(POST_ANALYSIS_DIR / "site_by_recruitment_contingency.csv")
    expected.to_csv(POST_ANALYSIS_DIR / "site_by_recruitment_expected_counts.csv")

    plot_df = tab.stack().rename("n").reset_index()
    plot_df["prop"] = plot_df.groupby("sites")["n"].transform(lambda s: s / s.sum())
    plot_df.to_csv(POST_ANALYSIS_DIR / "site_recruitment_plot_data.csv", index=False)

    # Rmd `site_recruitment_dist.pdf`: stacked bars, proportions within site.
    prop = tab.div(tab.sum(axis=1), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(20, 20))
    bottom = np.zeros(len(prop))
    for col in prop.columns:
        ax.bar(prop.index.astype(str), prop[col].values, bottom=bottom, label=str(col))
        bottom += prop[col].values
    ax.set_ylim(0, 1)
    ax.set_xlabel("Site")
    ax.set_ylabel("Proportion within site")
    ax.set_title("Recruitment method distribution by site")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Recruitment method", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    out = POST_ANALYSIS_DIR / "site_recruitment_dist.pdf"
    clinical_analysis_save_figure_png_pdf(fig, out, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", out)

    # Rmd `site_recruitment.pdf`: tile heatmap of within-site proportions.
    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(prop.values, aspect="auto", cmap="viridis", vmin=0, vmax=max(1e-9, np.nanmax(prop.values)))
    ax.set_xticks(range(prop.shape[1]))
    ax.set_xticklabels(prop.columns, rotation=45, ha="right")
    ax.set_yticks(range(prop.shape[0]))
    ax.set_yticklabels(prop.index)
    ax.set_xlabel("Recruitment method")
    ax.set_ylabel("Site")
    ax.set_title("Within-site recruitment proportions")
    fig.colorbar(im, ax=ax, label="Within-site %")
    fig.tight_layout()
    out = POST_ANALYSIS_DIR / "site_recruitment.pdf"
    clinical_analysis_save_figure_png_pdf(fig, out, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", out)

    # Rmd `site_recruitment2.pdf`: ordered tile heatmap. Use hierarchical ordering if scipy is available.
    row_order = list(prop.index)
    col_order = list(prop.columns)
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
        if prop.shape[0] > 1:
            row_order = prop.index[leaves_list(linkage(pdist(prop.values), method="average"))].tolist()
        if prop.shape[1] > 1:
            col_order = prop.columns[leaves_list(linkage(pdist(prop.values.T), method="average"))].tolist()
    except Exception as exc:
        print("Could not compute clustered site/recruitment ordering; using original order:", exc)
    prop_ord = prop.loc[row_order, col_order]
    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(prop_ord.values, aspect="auto", cmap="viridis", vmin=0, vmax=max(1e-9, np.nanmax(prop_ord.values)))
    ax.set_xticks(range(prop_ord.shape[1]))
    ax.set_xticklabels(prop_ord.columns, rotation=45, ha="right")
    ax.set_yticks(range(prop_ord.shape[0]))
    ax.set_yticklabels(prop_ord.index)
    ax.set_xlabel("Recruitment method")
    ax.set_ylabel("Site")
    fig.colorbar(im, ax=ax, label="Within-site %")
    fig.tight_layout()
    out = POST_ANALYSIS_DIR / "site_recruitment2.pdf"
    clinical_analysis_save_figure_png_pdf(fig, out, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", out)
    return tab


@_with_notebook_context(default_expressions={'subgroup_vars': 'SUBGROUP_VARS'})
def clinical_analysis_subgroup_factor_long(df, factor_col, subgroup_vars=None):
    """Process clinical analysis subgroup factor long."""
    keep_subgroups = [c for c in subgroup_vars if c in df.columns]
    if factor_col not in df.columns or not keep_subgroups:
        return pd.DataFrame()
    long = df[[factor_col] + keep_subgroups].copy()
    long[factor_col] = long[factor_col].astype(str)
    out = long.melt(id_vars=[factor_col], value_vars=keep_subgroups, var_name="subgroup_type", value_name="subgroup_label")
    out = out.dropna(subset=[factor_col, "subgroup_label"])
    out = out.loc[~out[factor_col].isin(["nan", "None", ""])]
    out = out.loc[~out["subgroup_label"].astype(str).isin(["nan", "None", ""])]
    return out


@_with_notebook_context
def clinical_analysis_test_subgroup_by_factor(df, factor_col, prefix, factor_label=None):
    """Process clinical analysis test subgroup by factor."""
    factor_label = factor_label or factor_col
    long = clinical_analysis_subgroup_factor_long(df, factor_col)
    if long.empty:
        print(f"No data for {factor_col}")
        return pd.DataFrame(), long

    rows = []
    for subgroup_type, sub in long.groupby("subgroup_type", sort=False):
        tab = pd.crosstab(sub[factor_col].astype(str), sub["subgroup_label"].astype(str))
        stat, p, v = clinical_analysis_chi_square_test(tab)
        rows.append({
            "subgroup_type": subgroup_type,
            "statistic": stat,
            "p_value": p,
            "cramers_v": v,
            "n": int(tab.to_numpy().sum()),
            "n_factor_levels": tab.shape[0],
            "n_labels": tab.shape[1],
        })
    results = pd.DataFrame(rows)
    results["p_adj"] = clinical_analysis_bh_fdr(results["p_value"])
    results = results.sort_values("p_adj", na_position="last")
    results.to_csv(POST_ANALYSIS_DIR / f"{prefix}_subgroup_tests.csv", index=False)
    display(results)

    plot_df = (
        long.groupby(["subgroup_type", factor_col, "subgroup_label"])
        .size()
        .rename("n")
        .reset_index()
    )
    plot_df["prop"] = plot_df.groupby(["subgroup_type", factor_col])["n"].transform(lambda s: s / s.sum())
    plot_df.to_csv(POST_ANALYSIS_DIR / f"{prefix}_subgroup_plot_data.csv", index=False)

    subgroup_order = [s for s in SUBGROUP_VARS if s in plot_df["subgroup_type"].unique()]
    ncols = 2
    nrows = int(np.ceil(len(subgroup_order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(5, 4 * nrows)), squeeze=False)
    for ax, subgroup_type in zip(axes.ravel(), subgroup_order):
        sub = plot_df.loc[plot_df["subgroup_type"] == subgroup_type]
        pivot = sub.pivot_table(index=factor_col, columns="subgroup_label", values="prop", fill_value=0)
        bottom = np.zeros(len(pivot))
        for col in pivot.columns:
            ax.bar(pivot.index.astype(str), pivot[col].values, bottom=bottom, label=str(col))
            bottom += pivot[col].values
        test_row = results.loc[results["subgroup_type"] == subgroup_type]
        if not test_row.empty:
            title = f"{subgroup_type}\nBH p={test_row['p_adj'].iloc[0]:.3g}; V={test_row['cramers_v'].iloc[0]:.2f}"
        else:
            title = subgroup_type
        ax.set_title(title)
        ax.set_ylabel(f"Proportion within {factor_label}")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Subgroup label", fontsize=8)
    for ax in axes.ravel()[len(subgroup_order):]:
        ax.axis("off")
    fig.suptitle(f"Subgroup label distribution by {factor_label}", y=1.01)
    fig.tight_layout()
    out = POST_ANALYSIS_DIR / f"{prefix}_subgroup_distribution.pdf"
    clinical_analysis_save_figure_png_pdf(fig, out, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", out)
    return results, long


@_with_notebook_context
def clinical_analysis_stratified_site_tests_within_recruitment(df):
    """Process clinical analysis stratified site tests within recruitment."""
    required = {"Site", "chrrecruit"}
    if not required.issubset(df.columns):
        print("Site and chrrecruit columns are required for stratified tests.")
        return pd.DataFrame()
    long = df[["Site", "chrrecruit"] + [c for c in SUBGROUP_VARS if c in df.columns]].copy()
    long = long.melt(id_vars=["Site", "chrrecruit"], value_vars=[c for c in SUBGROUP_VARS if c in df.columns], var_name="subgroup_type", value_name="subgroup_label")
    long = long.dropna(subset=["Site", "chrrecruit", "subgroup_label"])
    rows = []
    for (subgroup_type, recruit), sub in long.groupby(["subgroup_type", "chrrecruit"], sort=False):
        tab = pd.crosstab(sub["Site"].astype(str), sub["subgroup_label"].astype(str))
        stat, p, v = clinical_analysis_chi_square_test(tab)
        rows.append({
            "subgroup_type": subgroup_type,
            "chrrecruit": recruit,
            "statistic": stat,
            "p_value": p,
            "cramers_v": v,
            "n": int(tab.to_numpy().sum()),
            "n_sites": tab.shape[0],
            "n_labels": tab.shape[1],
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_adj_within_subgroup"] = out.groupby("subgroup_type")["p_value"].transform(clinical_analysis_bh_fdr)
        out.to_csv(POST_ANALYSIS_DIR / "site_by_subgroup_stratified_by_recruitment_tests.csv", index=False)
        display(out.sort_values(["subgroup_type", "p_adj_within_subgroup"], na_position="last"))
    return out


@_with_notebook_context
def clinical_analysis_optional_multinomial_site_recruitment_models(df):
    """Process clinical analysis optional multinomial site recruitment models."""
    required = {"Site", "chrrecruit"}
    if not required.issubset(df.columns):
        print("Skipping adjusted multinomial site/recruitment models because Site/chrrecruit columns are unavailable.")
        return pd.DataFrame()
    try:
        import statsmodels.api as sm
    except Exception as exc:
        print("Skipping adjusted multinomial site/recruitment models because statsmodels is unavailable:", exc)
        return pd.DataFrame()

    rows = []
    for subgroup_type in [c for c in SUBGROUP_VARS if c in df.columns]:
        d = df[["Site", "chrrecruit", subgroup_type]].dropna().copy()
        if d[subgroup_type].astype(str).nunique() < 2:
            continue
        y = pd.Categorical(d[subgroup_type].astype(str)).codes
        X_full = pd.get_dummies(d[["Site", "chrrecruit"]].astype(str), drop_first=True, dtype=float)
        X_full = sm.add_constant(X_full, has_constant="add")
        try:
            model = sm.MNLogit(y, X_full).fit(method="newton", disp=False, maxiter=200)
            rows.append({
                "subgroup_scheme": subgroup_type,
                "llf": model.llf,
                "aic": model.aic,
                "bic": model.bic,
                "n": int(d.shape[0]),
                "status": "fit",
            })
        except Exception as exc:
            rows.append({
                "subgroup_scheme": subgroup_type,
                "llf": np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "n": int(d.shape[0]),
                "status": f"failed: {exc}",
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(POST_ANALYSIS_DIR / "site_recruitment_adjusted_multinomial_models.csv", index=False)
        display(out)
    return out


@_with_notebook_context
def clinical_analysis_domain_map_path_label_frame(
    labels_by_modality,
    subject_ids,
    final_labels=None,
    sample_name="sample",
    stage_order_for_labels=None,
    final_name="Integrated",
    min_group_n=4,
    include_final_in_path=False,
):
    """Process clinical analysis domain map path label frame."""
    stage_order_for_labels = list(stage_order_for_labels or labels_by_modality.keys())
    subject_ids = pd.Series(subject_ids, name=_subject_id_column).astype(str).reset_index(drop=True)
    n_subjects = len(subject_ids)
    label_frame = pd.DataFrame({_subject_id_column: subject_ids})
    path_stages = []
    for stage in stage_order_for_labels:
        if stage not in labels_by_modality:
            continue
        values = pd.Series(labels_by_modality[stage]).astype(str).reset_index(drop=True)
        if len(values) != n_subjects:
            raise ValueError(
                f"{sample_name}: label length mismatch for {stage}: "
                f"{len(values)} labels for {n_subjects} subject IDs."
            )
        label_frame[stage] = values.str.replace(" ", "_", regex=False)
        path_stages.append(stage)
    if final_labels is not None:
        final_values = pd.Series(final_labels).astype(str).reset_index(drop=True)
        if len(final_values) != n_subjects:
            raise ValueError(
                f"{sample_name}: final label length mismatch: "
                f"{len(final_values)} labels for {n_subjects} subject IDs."
            )
        label_frame[final_name] = final_values.str.replace(" ", "_", regex=False)
        if include_final_in_path:
            path_stages.append(final_name)
    if not path_stages:
        raise ValueError(f"{sample_name}: no domain-map labels were available.")

    # The group is the observed path through the domain-map axes. By default this
    # excludes the final/integrated label so the groups are true domain profiles
    # rather than the two integrated clusters.
    label_frame["label"] = label_frame[path_stages].agg(" -> ".join, axis=1)
    label_frame["domain_map_path"] = label_frame["label"]

    component_cols = [_subject_id_column, "domain_map_path"] + path_stages
    label_frame[component_cols].to_csv(
        os.path.join(_DOMAIN_MAP_GROUPS_DIR, f"{sample_name}_domain_map_path_labels.csv"),
        index=False,
    )

    counts = label_frame["label"].value_counts().rename_axis("domain_map_path").reset_index(name="n")
    counts["meets_min_group_n"] = counts["n"].ge(min_group_n)
    counts.to_csv(os.path.join(_DOMAIN_MAP_GROUPS_DIR, f"{sample_name}_domain_map_path_counts.csv"), index=False)
    print(
        f"{sample_name}: {counts.shape[0]} domain-map paths; "
        f"{int(counts['meets_min_group_n'].sum())} have n >= {min_group_n}."
    )
    display(counts.head(30))
    return label_frame, counts


@_with_notebook_context
def clinical_analysis_longitudinal_results_with_long_df(results):
    """Process clinical analysis longitudinal results with long df."""
    analyses = (results or {}).get("analyses", {})
    if not analyses:
        return False
    for analysis in analyses.values():
        long_df = analysis.get("mixedlm", {}).get("long_df", pd.DataFrame())
        if isinstance(long_df, pd.DataFrame) and not long_df.empty:
            return True
    return False


@_with_notebook_context
def clinical_analysis_get_domain_map_source_longitudinal_results():
    """Process clinical analysis get domain map source longitudinal results."""
    if "longitudinal_results" in globals() and clinical_analysis_longitudinal_results_with_long_df(longitudinal_results):
        print("Using in-memory longitudinal long-format data from longitudinal_results.")
        return longitudinal_results

    print("Rebuilding longitudinal long-format data in the domain-map output folder because cached results do not store long_df.")
    _is_singleclust_result = isinstance(final_metrics.get("data"), pd.DataFrame) if isinstance(final_metrics, dict) else False
    _data_dirs = globals().get("_longitudinal_data_dirs") or [
        'path/to/simpleclust_data'
        if _is_singleclust_result
        else 'path/to/multiclust_data'
    ]
    _prescient_ids = globals().get("_longitudinal_prescient_ids")
    if _prescient_ids is None and "prescient_ids" in globals():
        _prescient_ids = prescient_ids
    elif _prescient_ids is None and "prescient" in globals():
        _prescient_ids = set(prescient[_subject_id_column])

    return Utils.run_longitudinal_multiclust_report(
        final_metrics=final_metrics,
        meta=meta,
        plots_dir=_DOMAIN_MAP_GROUPS_DIR,
        data_dirs=_data_dirs,
        prescient_ids=_prescient_ids,
        vars_to_keep=vars_to_keep if "vars_to_keep" in globals() else None,
        categorical_columns=cat_vars if "cat_vars" in globals() else None,
        validation_domain_labels=globals().get("_validation_domain_labels"),
        validation_final_labels=globals().get("_validation_final_labels"),
        validation_subject_ids=globals().get("_validation_subject_ids"),
        validation_baseline_data=globals().get("_validation_baseline_data"),
        months=(1, 2, 3, 4, 5),
        subject_id_column=_subject_id_column,
        min_features_per_analysis=2,
        min_features_for_cluster_change=1,
        min_followup_timepoints_per_feature=1,
        min_nonmissing_per_timepoint=8,
        followup_col_threshold=0.5,
        followup_row_threshold=0.5,
        min_group_n=4,
        reuse_existing=False,
        skip_model_fits=True,
    )


@_with_notebook_context
def clinical_analysis_run_domain_map_mixed_models_from_long_df(
    long_df,
    labels_df,
    output_dir,
    analysis_name,
    min_group_n=4,
    top_n_plot=12,
    max_plot_groups=None,
):
    """Process clinical analysis run domain map mixed models from long df."""
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.multitest import multipletests
    except Exception as exc:
        raise ImportError("statsmodels is required for domain-map longitudinal mixed models.") from exc

    os.makedirs(output_dir, exist_ok=True)
    labels_use = labels_df[[_subject_id_column, "label"]].dropna().drop_duplicates(_subject_id_column).copy()
    labels_use[_subject_id_column] = labels_use[_subject_id_column].astype(str)

    df = long_df.drop(columns=["label"], errors="ignore").copy()
    df[_subject_id_column] = df[_subject_id_column].astype(str)
    df = df.merge(labels_use, on=_subject_id_column, how="inner").dropna(subset=["value", "label"])
    if df.empty:
        summary = pd.DataFrame([{"analysis_name": analysis_name, "status": "no_overlap_with_domain_map_labels"}])
        summary.to_csv(os.path.join(output_dir, f"{analysis_name}_domain_map_mixedlm_summary.csv"), index=False)
        return {"summary": summary, "mean_drift": {}}

    baseline_group_counts = (
        df.loc[df["time"].astype(str).eq("baseline"), [_subject_id_column, "label"]]
        .drop_duplicates(_subject_id_column)["label"]
        .value_counts()
    )
    eligible_groups = baseline_group_counts.loc[baseline_group_counts.ge(min_group_n)].index.astype(str).tolist()
    group_report = baseline_group_counts.rename_axis("domain_map_path").reset_index(name="baseline_n")
    group_report["included_in_models"] = group_report["domain_map_path"].astype(str).isin(eligible_groups)
    group_report.to_csv(os.path.join(output_dir, f"{analysis_name}_domain_map_model_group_counts.csv"), index=False)
    df = df.loc[df["label"].astype(str).isin(eligible_groups)].copy()
    if len(eligible_groups) < 2 or df.empty:
        summary = pd.DataFrame([{
            "analysis_name": analysis_name,
            "status": "too_few_domain_map_paths_after_min_group_filter",
            "n_eligible_groups": int(len(eligible_groups)),
            "min_group_n": int(min_group_n),
        }])
        summary.to_csv(os.path.join(output_dir, f"{analysis_name}_domain_map_mixedlm_summary.csv"), index=False)
        return {"summary": summary, "mean_drift": {}}

    time_order = ["baseline"] + [f"month{m}" for m in sorted(pd.to_numeric(df.loc[df["month"] > 0, "month"], errors="coerce").dropna().astype(int).unique())]
    df["time"] = pd.Categorical(df["time"].astype(str), categories=time_order, ordered=True)
    features = sorted(df["feature"].dropna().astype(str).unique())
    summary_path = os.path.join(output_dir, f"{analysis_name}_domain_map_mixedlm_summary.csv")
    if os.path.exists(summary_path):
        try:
            cached_summary = pd.read_csv(summary_path)
        except Exception:
            cached_summary = pd.DataFrame()
        if (
            not cached_summary.empty
            and "feature" in cached_summary.columns
            and set(cached_summary["feature"].dropna().astype(str)).issuperset(set(features))
        ):
            print(f"{analysis_name}: using cached domain-map mixed model summary ({len(cached_summary)} rows).", flush=True)
            return {
                "summary": cached_summary,
                "mean_drift": {},
                "mean_drift_summary_path": os.path.join(output_dir, f"{analysis_name}_domain_map_paths_mean_drift_sensitivity.csv"),
                "mean_drift_raw_plot_path": os.path.join(output_dir, f"{analysis_name}_domain_map_paths_mean_drift_raw_trajectories.png"),
                "mean_drift_change_plot_path": os.path.join(output_dir, f"{analysis_name}_domain_map_paths_mean_drift_change_scores.png"),
            }

    print(
        f"{analysis_name}: fitting {len(features)} longitudinal features across "
        f"{len(eligible_groups)} eligible domain-map paths.",
        flush=True,
    )

    def _wald_term_values(result, term_name):
        try:
            term_table = result.wald_test_terms(skip_single=False).table
        except Exception:
            return np.nan, np.nan, np.nan
        if term_name not in term_table.index:
            return np.nan, np.nan, np.nan
        row = term_table.loc[term_name]
        statistic = row.get("statistic", np.nan)
        if hasattr(statistic, "item"):
            statistic = statistic.item()
        p_value = row.get("pvalue", np.nan)
        if hasattr(p_value, "item"):
            p_value = p_value.item()
        df_constraint = row.get("df_constraint", np.nan)
        if hasattr(df_constraint, "item"):
            df_constraint = df_constraint.item()
        return float(statistic), float(p_value), float(df_constraint)

    rows = []
    for feature_idx, feat in enumerate(features, start=1):
        print(f"{analysis_name}: feature {feature_idx}/{len(features)} - {feat}", flush=True)
        tmp = df.loc[df["feature"].astype(str).eq(str(feat)), [_subject_id_column, "label", "time", "month", "value"]].dropna().copy()
        tmp["group"] = tmp["label"].astype(str)
        tmp["time"] = tmp["time"].cat.remove_unused_categories()
        group_counts = tmp.drop_duplicates([_subject_id_column, "group"])["group"].value_counts()
        followup_months = sorted(int(m) for m in pd.to_numeric(tmp.loc[tmp["month"] > 0, "month"], errors="coerce").dropna().unique())
        if tmp["time"].nunique() < 2 or not followup_months:
            rows.append({"analysis_name": analysis_name, "feature": feat, "status": "too_few_timepoints"})
            continue
        if group_counts.shape[0] < 2 or group_counts.min() < min_group_n:
            rows.append({
                "analysis_name": analysis_name,
                "feature": feat,
                "status": "too_few_subjects_or_groups",
                "n_subjects": int(tmp[_subject_id_column].nunique()),
                "n_groups": int(group_counts.shape[0]),
                "min_group_n": int(group_counts.min()) if not group_counts.empty else 0,
            })
            continue
        try:
            full = smf.mixedlm(
                "value ~ C(group) * C(time)",
                tmp,
                groups=tmp[_subject_id_column],
            ).fit(reml=False, method="lbfgs", disp=False, maxiter=200)
            group_wald, group_p, group_df = _wald_term_values(full, "C(group)")
            time_wald, time_p, time_df = _wald_term_values(full, "C(time)")
            interaction_wald, interaction_p, interaction_df = _wald_term_values(full, "C(group):C(time)")
            status = "ok"
        except Exception as exc:
            rows.append({
                "analysis_name": analysis_name,
                "feature": feat,
                "status": f"model_failed: {exc}",
                "n_subjects": int(tmp[_subject_id_column].nunique()),
                "n_groups": int(group_counts.shape[0]),
                "min_group_n": int(group_counts.min()) if not group_counts.empty else 0,
            })
            continue

        means = tmp.groupby(["group", "time"], observed=False)["value"].mean().unstack()
        mean_row = {
            f"Mean {group} {time}": float(value)
            for group, values in means.iterrows()
            for time, value in values.items()
            if pd.notna(value)
        }
        rows.append({
            "analysis_name": analysis_name,
            "feature": feat,
            "status": status,
            "n_subjects": int(tmp[_subject_id_column].nunique()),
            "n_observations": int(tmp.shape[0]),
            "n_groups": int(group_counts.shape[0]),
            "min_group_n": int(group_counts.min()),
            "n_timepoints": int(tmp["time"].nunique()),
            "n_followup_timepoints": int(len(followup_months)),
            "followup_months_in_model": ",".join(map(str, followup_months)),
            "test_type": "mixedlm_wald_terms",
            "group_wald_chi2": group_wald,
            "group_df": group_df,
            "group_p": group_p,
            "time_wald_chi2": time_wald,
            "time_df": time_df,
            "time_p": time_p,
            "interaction_wald_chi2": interaction_wald,
            "interaction_df": interaction_df,
            "interaction_p": interaction_p,
            **mean_row,
        })

    summary = pd.DataFrame(rows)
    print(f"{analysis_name}: finished model fitting; writing summary and plots.", flush=True)
    for col in ["group_p", "time_p", "interaction_p"]:
        summary[f"{col}_fdr"] = np.nan
        ok = summary[col].notna() if col in summary.columns else pd.Series(False, index=summary.index)
        if ok.any():
            summary.loc[ok, f"{col}_fdr"] = multipletests(summary.loc[ok, col], method="fdr_bh")[1]
    sort_cols = [c for c in ["interaction_p_fdr", "interaction_p", "time_p_fdr", "group_p_fdr"] if c in summary.columns]
    if sort_cols:
        summary = summary.sort_values(sort_cols, na_position="last")
    summary.to_csv(summary_path, index=False)

    ok_features = summary.loc[summary["status"].eq("ok"), "feature"].astype(str).head(top_n_plot).tolist() if "status" in summary else []
    if ok_features:
        all_plot_groups = df.drop_duplicates([_subject_id_column, "label"])["label"].value_counts().index.astype(str).tolist()
        plot_group_order = all_plot_groups if max_plot_groups is None else all_plot_groups[:max_plot_groups]
        pd.DataFrame({
            "plot_group": [f"Path {idx + 1}" for idx in range(len(all_plot_groups))],
            "domain_map_path": all_plot_groups,
            "shown_in_top_feature_plot": [group in set(plot_group_order) for group in all_plot_groups],
        }).to_csv(os.path.join(output_dir, f"{analysis_name}_domain_map_plot_path_lookup.csv"), index=False)
        ncols = min(3, len(ok_features))
        nrows = int(np.ceil(len(ok_features) / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.8 * ncols, 4.6 * nrows), squeeze=False)
        for ax, feat in zip(axes.ravel(), ok_features):
            tmp = df.loc[df["feature"].astype(str).eq(str(feat))].dropna(subset=["value", "label"]).copy()
            time_order_feature = [str(t) for t in tmp["time"].cat.categories if (tmp["time"].astype(str) == str(t)).any()]
            group_order = plot_group_order
            palette_base = sns.color_palette("tab20", n_colors=max(20, len(group_order)))
            palette = {group: palette_base[idx % len(palette_base)] for idx, group in enumerate(group_order)}
            x_positions = {time: idx for idx, time in enumerate(time_order_feature)}
            for group in group_order:
                group_tmp = tmp.loc[tmp["label"].astype(str).eq(group)]
                stats = group_tmp.groupby("time", observed=False)["value"].agg(["mean", "count", "std"]).reindex(time_order_feature).dropna(subset=["mean"])
                if stats.empty:
                    continue
                stats["se"] = stats["std"] / np.sqrt(stats["count"].clip(lower=1))
                xs = [x_positions[str(t)] for t in stats.index.astype(str)]
                ax.errorbar(
                    xs,
                    stats["mean"],
                    yerr=stats["se"].fillna(0.0),
                    marker="o",
                    linewidth=1.8,
                    capsize=3,
                    label=f"Path {group_order.index(group) + 1}: {group}",
                    color=palette.get(group, None),
                )
            row = summary.loc[summary["feature"].astype(str).eq(str(feat))].iloc[0]
            ax.set_title(Utils.display_feature_name(feat), pad=24)
            ax.text(
                0.5,
                1.01,
                f"group q={row.get('group_p_fdr', np.nan):.3g}; time q={row.get('time_p_fdr', np.nan):.3g}; group*time q={row.get('interaction_p_fdr', np.nan):.3g}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax.set_xticks(list(range(len(time_order_feature))))
            ax.set_xticklabels(time_order_feature, rotation=35, ha="right")
            ax.set_xlabel("")
            ax.set_ylabel("Mean value")
            ax.grid(axis="y", alpha=0.2)
            if len(group_order) <= 12:
                ax.legend(frameon=False, fontsize=6, title="Domain-map paths", loc="best")
            elif ax is axes.ravel()[0]:
                ax.text(
                    0.01,
                    -0.23,
                    f"Showing {len(group_order)} eligible domain-map paths; see the plot-path lookup CSV for labels.",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                )
            sns.despine(ax=ax)
        for ax in axes.ravel()[len(ok_features):]:
            ax.axis("off")
        fig.tight_layout()
        Utils._save_longitudinal_matplotlib_image(
            fig,
            os.path.join(output_dir, f"{analysis_name}_domain_map_mixedlm_top_features.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    mean_drift = Utils._run_mean_drift_sensitivity(
        long_df=df,
        features=features,
        output_dir=output_dir,
        analysis_name=f"{analysis_name}_domain_map_paths",
        subject_id_column=_subject_id_column,
        features_to_plot=[],
        min_group_n=min_group_n,
        top_n_plot=0,
    )
    return {
        "summary": summary,
        "mean_drift": mean_drift,
        "mean_drift_summary_path": mean_drift.get("summary_path", ""),
        "mean_drift_raw_plot_path": mean_drift.get("raw_plot_path", ""),
        "mean_drift_change_plot_path": mean_drift.get("change_plot_path", ""),
    }


@_with_notebook_context
def clinical_analysis_normalise_longitudinal_feature(analysis_name, feature_name):
    """Process clinical analysis normalise longitudinal feature."""
    feature_name = str(feature_name)
    if "__" in feature_name:
        feature_domain, raw_feature = feature_name.split("__", 1)
        return feature_domain, raw_feature
    return str(analysis_name), feature_name


@_with_notebook_context
def clinical_analysis_read_longitudinal_mixedlm_summary(path, grouping_scheme, sample_name, analysis_name, path_variant=""):
    """Process clinical analysis read longitudinal mixedlm summary."""
    if not os.path.exists(path):
        return []
    summary = pd.read_csv(path)
    rows = []
    for _, row in summary.iterrows():
        feature_domain, raw_feature = clinical_analysis_normalise_longitudinal_feature(
            analysis_name,
            row.get("feature", ""),
        )
        rows.append({
            "grouping_scheme": grouping_scheme,
            "path_variant": path_variant,
            "sample": sample_name,
            "analysis": analysis_name,
            "feature_domain": feature_domain,
            "feature": raw_feature,
            "original_feature": row.get("feature", ""),
            "status": row.get("status", ""),
            "n_subjects": row.get("n_subjects", np.nan),
            "n_groups": row.get("n_groups", np.nan),
            "n_timepoints": row.get("n_timepoints", np.nan),
            "group_q": row.get("group_p_fdr", np.nan),
            "time_q": row.get("time_p_fdr", np.nan),
            "interaction_q": row.get("interaction_p_fdr", np.nan),
            "source_path": path,
        })
    return rows


@_with_notebook_context
def clinical_analysis_load_longitudinal_grouping_scheme_results():
    """Process clinical analysis load longitudinal grouping scheme results."""
    rows = []
    for sample_name in ["discovery", "validation"]:
        for analysis_name in _LONGITUDINAL_ANALYSES:
            standard_path = os.path.join(
                _LONGITUDINAL_ALL_DIR,
                sample_name,
                analysis_name,
                "mixedlm",
                f"{sample_name}_{analysis_name}_mixedlm_summary.csv",
            )
            grouping_scheme = (
                "integrated_subgroups"
                if analysis_name == "integrated"
                else "individual_domain_subgroups"
            )
            rows.extend(clinical_analysis_read_longitudinal_mixedlm_summary(
                standard_path,
                grouping_scheme=grouping_scheme,
                sample_name=sample_name,
                analysis_name=analysis_name,
            ))

            domain_map_path = os.path.join(
                _DOMAIN_MAP_PATH_DIR,
                sample_name,
                analysis_name,
                f"{sample_name}_{analysis_name}_domains_only_domain_map_mixedlm_summary.csv",
            )
            rows.extend(clinical_analysis_read_longitudinal_mixedlm_summary(
                domain_map_path,
                grouping_scheme="domain_map_paths",
                sample_name=sample_name,
                analysis_name=analysis_name,
                path_variant="domains_only",
            ))

    results = pd.DataFrame(rows)
    for col in ["group_q", "time_q", "interaction_q", "n_subjects", "n_groups", "n_timepoints"]:
        results[col] = pd.to_numeric(results[col], errors="coerce")
    results["ok"] = results["status"].eq("ok")
    results["group_sig"] = results["ok"] & results["group_q"].lt(0.05)
    results["time_sig"] = results["ok"] & results["time_q"].lt(0.05)
    results["interaction_sig"] = results["ok"] & results["interaction_q"].lt(0.05)
    return results


@_with_notebook_context
def clinical_analysis_normalize_subject_id(value):
    """Process clinical analysis normalize subject id."""
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    if value in {"", "nan", "none"}:
        return None
    return value


@_with_notebook_context
def clinical_analysis_make_conversion_labels(subject_ids, conversion_subjects):
    """Process clinical analysis make conversion labels."""
    conversion_subjects_normalized = {
        sid for sid in (clinical_analysis_normalize_subject_id(s) for s in conversion_subjects)
        if sid is not None
    }
    subject_ids_normalized = pd.Series(subject_ids).map(clinical_analysis_normalize_subject_id).reset_index(drop=True)
    return np.where(
        subject_ids_normalized.isin(conversion_subjects_normalized),
        "yes",
        "no",
    ).tolist(), conversion_subjects_normalized, set(subject_ids_normalized.dropna())


@_with_notebook_context
def clinical_analysis_plot_conversion_domain_map(sample_name, labels_by_modality, subject_ids, conversion_subjects, save_file_name):
    """Process clinical analysis plot conversion domain map."""
    conversion_labels, conversion_ids, sample_ids = clinical_analysis_make_conversion_labels(subject_ids, conversion_subjects)
    n_subjects = len(subject_ids)

    if not all(len(labels_by_modality[stage]) == n_subjects for stage in stage_order):
        label_lengths = {stage: len(labels_by_modality[stage]) for stage in stage_order}
        raise ValueError(
            f"{sample_name}: label lengths do not match subject IDs. "
            f"subject_ids={n_subjects}, label_lengths={label_lengths}"
        )

    conversion_counts = pd.Series(
        conversion_labels,
        name="converted_to_psychosis",
    ).value_counts()
    print(f"Conversion labels mapped to {sample_name} subjects:")
    print(conversion_counts)
    print(
        f"{sample_name}: {len(conversion_ids & sample_ids)} of "
        f"{len(conversion_ids)} raw conversion IDs are present in this clustered sample."
    )

    fig, _ = domain_map(
        new_labels_by_modality=labels_by_modality,
        final_labels=conversion_labels,
        stage_order=stage_order,
        final_name="Conversion",
        top_token="high_severity",
        bottom_token="low_severity",
        final_top_value="yes",
        final_bottom_value="no",
        color_for_top_final="#B64242",
        color_for_bottom_final="#B3D9D5",
        add_gap_in_final=True,
        gap_weight=20,
        title=f"Domain mapping by psychosis conversion ({sample_name})",
        plots_dir=plots_dir,
        save_file_name=save_file_name,
    )
    return fig, conversion_labels


@_with_notebook_context
def clinical_analysis_build_profile_predictor_df(labels_by_modality, subject_ids, conversion_labels):
    """Process clinical analysis build profile predictor df."""
    df = pd.DataFrame({"src_subject_id": pd.Series(subject_ids).astype(str).reset_index(drop=True)})
    for stage in stage_order:
        df[stage] = pd.Series(labels_by_modality[stage]).astype(str).reset_index(drop=True)
    df["clinical_profile"] = df[stage_order].agg(" | ".join, axis=1)
    df["n_high_severity_domains"] = df[stage_order].eq("high_severity").sum(axis=1)
    df["converted_to_psychosis"] = pd.Series(conversion_labels).astype(str).reset_index(drop=True)
    return df


@_with_notebook_context
def clinical_analysis_encode_profile_predictors(df, reference_columns=None):
    """Process clinical analysis encode profile predictors."""
    encoded = pd.get_dummies(
        df[categorical_predictors],
        prefix=categorical_predictors,
        dtype=float,
    )
    encoded = pd.concat(
        [encoded.reset_index(drop=True), df[numeric_predictors].astype(float).reset_index(drop=True)],
        axis=1,
    )
    if reference_columns is not None:
        encoded = encoded.reindex(columns=reference_columns, fill_value=0.0)
    return encoded


@_with_notebook_context
def clinical_analysis_build_preprocessed_conversion_feature_df(data_by_modality, subject_ids, conversion_labels, sample_name):
    """Merge the saved preprocessed clustering variables into one wide matrix."""
    merged = None
    for modality in stage_order:
        if modality not in data_by_modality:
            raise KeyError(f"{sample_name}: missing modality data for {modality}")

        df_mod = data_by_modality[modality].copy()
        if "src_subject_id" not in df_mod.columns:
            raise KeyError(f"{sample_name} {modality}: src_subject_id column is missing")

        feature_cols = [col for col in df_mod.columns if col != "src_subject_id"]
        tmp = df_mod[["src_subject_id"] + feature_cols].copy()
        tmp["src_subject_id"] = tmp["src_subject_id"].astype(str)
        tmp = tmp.rename(columns={col: f"{modality}__{col}" for col in feature_cols})

        merged = tmp if merged is None else merged.merge(tmp, on="src_subject_id", how="inner")

    label_df = pd.DataFrame({
        "src_subject_id": pd.Series(subject_ids).astype(str).reset_index(drop=True),
        "converted_to_psychosis": pd.Series(conversion_labels).astype(str).reset_index(drop=True),
    })
    out = merged.merge(label_df, on="src_subject_id", how="inner")

    if len(out) != len(label_df):
        missing = sorted(set(label_df["src_subject_id"]) - set(out["src_subject_id"]))[:10]
        print(
            f"WARNING: {sample_name}: retained {len(out)} of {len(label_df)} subjects "
            f"after merging modalities. Missing examples: {missing}"
        )

    return out


# -----------------------------------------------------------------------------
# clinical_paper analysis helpers
# Implementations are preserved from the original notebook cells.
# -----------------------------------------------------------------------------

@_with_notebook_context
def clinical_pipeline_truthy_profile_value(value):
    """Process clinical pipeline truthy profile value."""
    return str(value).strip().strip('"\'').upper() in {"TRUE", "1", "YES", "Y"}


@_with_notebook_context
def clinical_pipeline_parse_profile_exports(profile_path):
    """Process clinical pipeline parse profile exports."""
    exports = {}
    if profile_path is None or not profile_path.exists():
        return exports
    for line in profile_path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[len("export "):].split("=", 1)
        value = value.strip()
        # Keep literal defaults such as ${VAR:-0} as-is; simple quoted values are enough for the flag check.
        exports[key.strip()] = value.strip('"\'')
    return exports


@_with_notebook_context
def clinical_pipeline_infer_notebook_profile():
    """Process clinical pipeline infer notebook profile."""
    notebook_dir = Path.cwd().name
    profile_candidates = []
    if notebook_dir == "clinical_paper":
        profile_candidates.append("clinical_paper")
    elif notebook_dir == "multiclust_extended":
        profile_candidates.append("multiclust_extended")
    elif notebook_dir == "prospect":
        profile_candidates.append("prospect")
    elif notebook_dir == "schizbull_legacy":
        profile_candidates.append("schizbull_legacy")
    profile_candidates.append(os.environ.get("RUN_PROFILE", ""))
    for candidate in profile_candidates:
        if candidate:
            return candidate
    return None


@_with_notebook_context
def clinical_pipeline_find_repo_root(start):
    """Process clinical pipeline find repo root."""
    start = Path(start).resolve()
    for parent in [start] + list(start.parents):
        if (parent / "run_profiles").is_dir() and (parent / "full_pipeline.py").exists():
            return parent
    return None


@_with_notebook_context
def clinical_pipeline_profile_enabled_for_sensitivity(repo_root, profile_name):
    """Process clinical pipeline profile enabled for sensitivity."""
    if not profile_name or repo_root is None:
        return None, None
    profile_path = repo_root / "run_profiles" / f"{profile_name}.sh"
    exports = clinical_pipeline_parse_profile_exports(profile_path)
    value = exports.get("DO_CLUSTER_VALIDATION_SENSITIVITY", "FALSE")
    return clinical_pipeline_truthy_profile_value(value), profile_path


@_with_notebook_context
def clinical_pipeline_display_if_available(obj):
    """Process clinical pipeline display if available."""
    if "display" in globals():
        display(obj)
    else:
        print(obj)


@_with_notebook_context
def clinical_pipeline_get_nested(dct, path, default=np.nan):
    """Process clinical pipeline get nested."""
    cur = dct
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


@_with_notebook_context
def clinical_pipeline_flatten_sensitivity_results(results):
    """Process clinical pipeline flatten sensitivity results."""
    rows = []
    for solution, payload in results.get("solutions", {}).items():
        rows.append({
            "solution": solution,
            "kind": payload.get("kind"),
            "observed_k": clinical_pipeline_get_nested(payload, ["observed_quality", "k"]),
            "observed_n": clinical_pipeline_get_nested(payload, ["observed_quality", "n"]),
            "observed_n_features": clinical_pipeline_get_nested(payload, ["observed_quality", "n_features"]),
            "observed_composite": clinical_pipeline_get_nested(payload, ["observed_quality", "composite"]),
            "observed_pipeline_stability_ari": clinical_pipeline_get_nested(payload, ["covariance_matched_gaussian_null", "observed_pipeline_stability_ari"]),
            "observed_silhouette": clinical_pipeline_get_nested(payload, ["observed_quality", "silhouette"]),
            "observed_calinski_harabasz": clinical_pipeline_get_nested(payload, ["observed_quality", "calinski_harabasz"]),
            "observed_davies_bouldin": clinical_pipeline_get_nested(payload, ["observed_quality", "davies_bouldin"]),
            "k1_composite": clinical_pipeline_get_nested(payload, ["uni_cluster_baseline", "quality", "composite"]),
            "projection_pca_components": clinical_pipeline_get_nested(payload, ["projection_median_split", "pca_components"]),
            "projection_pca_variance_explained": clinical_pipeline_get_nested(payload, ["projection_median_split", "pca_variance_explained"]),
            "projection_median_best_quality_projection": clinical_pipeline_get_nested(payload, ["projection_median_split", "best_quality_projection"]),
            "projection_median_best_quality_composite": clinical_pipeline_get_nested(payload, ["projection_median_split", "best_quality", "composite"]),
            "projection_median_best_quality_ari_with_observed": clinical_pipeline_get_nested(payload, ["projection_median_split", "best_quality_ari_with_observed_labels"]),
            "projection_median_best_ari_projection": clinical_pipeline_get_nested(payload, ["projection_median_split", "best_ari_projection"]),
            "projection_median_best_ari_with_observed": clinical_pipeline_get_nested(payload, ["projection_median_split", "best_ari"]),
            "projection_median_best_ari_composite": clinical_pipeline_get_nested(payload, ["projection_median_split", "best_ari_quality", "composite"]),
            "projection_median_stability_ari": clinical_pipeline_get_nested(payload, ["projection_median_split", "bootstrap_stability", "mean_ari"]),
            "projection_median_stability_sd_ari": clinical_pipeline_get_nested(payload, ["projection_median_split", "bootstrap_stability", "sd_ari"]),
            "dip_available": clinical_pipeline_get_nested(payload, ["dip_test_projections", "available"], default=False),
            "dip_global_projection_p_value": clinical_pipeline_get_nested(payload, ["dip_test_projections", "global_projection_p_value"]),
            "dip_min_projection_p_value": clinical_pipeline_get_nested(payload, ["dip_test_projections", "min_projection_p_value"]),
            "dip_bonferroni_min_p_value": clinical_pipeline_get_nested(payload, ["dip_test_projections", "bonferroni_min_p_value"]),
            "dip_best_projection": clinical_pipeline_get_nested(payload, ["dip_test_projections", "best_projection"]),
            "dip_pc1_p_value": clinical_pipeline_get_nested(payload, ["dip_test_projections", "pc1_p_value"]),
            "gap_selected_k_tibshirani": clinical_pipeline_get_nested(payload, ["gap_statistic", "selected_k_tibshirani_rule"]),
            "gap_selected_k_max_gap": clinical_pipeline_get_nested(payload, ["gap_statistic", "selected_k_max_gap"]),
            "gap_method": clinical_pipeline_get_nested(payload, ["gap_statistic", "method"], default=""),
            "sigclust_available": clinical_pipeline_get_nested(payload, ["sigclust", "available"], default=False),
            "sigclust_cluster_index": clinical_pipeline_get_nested(payload, ["sigclust", "observed_cluster_index"]),
            "sigclust_p_value": clinical_pipeline_get_nested(payload, ["sigclust", "p_value"]),
            "sigclust_p_value_normal_approx": clinical_pipeline_get_nested(payload, ["sigclust", "p_value_normal_approx"]),
            "sigclust_null_mean_cluster_index": clinical_pipeline_get_nested(payload, ["sigclust", "null_mean_cluster_index"]),
            "gaussian_null_p_quality": clinical_pipeline_get_nested(payload, ["covariance_matched_gaussian_null", "p_quality_ge_observed_solution"]),
            "gaussian_null_p_stability": clinical_pipeline_get_nested(payload, ["covariance_matched_gaussian_null", "p_stability_ge_observed_pipeline"]),
            "gaussian_null_mean_stability_ari": clinical_pipeline_get_nested(payload, ["covariance_matched_gaussian_null", "null_stability_mean_ari"]),
            "gaussian_null_mean_quality": clinical_pipeline_get_nested(payload, ["covariance_matched_gaussian_null", "null_quality_mean"]),
        })
    return pd.DataFrame(rows)


@_with_notebook_context
def clinical_pipeline_build_group_palette(modality, group_order, modality_palettes=None, default_palette=None):
    """Process clinical pipeline build group palette."""
    return modality_cluster_palette([str(g) for g in group_order], modality=modality)


@_with_notebook_context
def clinical_pipeline_add_metadata_and_clusters_individual_labels(final_metrics, data_full, mod_num):
    """
    Merge cluster labels into full metadata DataFrame using src_subject_id
    from the actual training data stored in fold_metrics['data'].
    This ensures correct alignment even when not all train_ids were clustered.
    """
    clusters = pd.Series(final_metrics['individual_labels'][mod_num])

    # Get the first modality’s dataframe — all have the same subject order
    modality_dfs = final_metrics.get('data', {})
    if not modality_dfs:
        raise ValueError("final_metrics['data'] is empty; cannot extract subject IDs.")

    # Use the src_subject_id column from the first available modality
    first_modality = list(modality_dfs.keys())[0]
    subj_ids = modality_dfs[first_modality]['src_subject_id'].reset_index(drop=True)

    # Sanity check: should match cluster array length
    if len(subj_ids) != len(clusters):
        print(f"⚠️ Mismatch: {len(subj_ids)} subject IDs vs {len(clusters)} cluster labels.")
        min_len = min(len(subj_ids), len(clusters))
        subj_ids = subj_ids.iloc[:min_len]
        clusters = clusters.iloc[:min_len]
        print(f"Trimmed both to {min_len} entries to align.")

    # Build cluster mapping dataframe
    cluster_df = pd.DataFrame({
        'src_subject_id': subj_ids,
        'Cluster': clusters
    })

    # Merge back into full metadata
    merged = pd.merge(data_full, cluster_df, on='src_subject_id', how='left')

    print(f"✅ Merged clusters for {merged['Cluster'].notna().sum()} subjects (out of {len(merged)} total).")
    return merged


@_with_notebook_context
def clinical_pipeline_chi_square_comparison_individual_labels(df, group_col, label_col, title_prefix):
    """
    Perform chi-square test and plot grouped bar chart.
    """
    # Drop missing values
    df = df.dropna(subset=[group_col, label_col])

    # Make a copy for plotting / stats so we can safely relabel
    df_plot = df.copy()

    # Rename 'HC' -> 'CC' only for phenotype (for plotting and stats)
    if label_col == 'phenotype':
        df_plot[label_col] = df_plot[label_col].replace({'HC': 'CC'})

    # 1. Summarize counts
    summary_df = (
        df_plot.groupby([group_col, label_col])
        .size()
        .reset_index(name='n')
    )

    # 2. Chi-square test
    tbl = pd.crosstab(df_plot[group_col], df_plot[label_col])
    chi2, pval, dof, expected = chi2_contingency(tbl)
    pval = round(pval, 3)

    # 3. Grouped bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=summary_df,
        x=group_col,
        y='n',
        hue=label_col,
        dodge=True
    )
    plt.title(f"{title_prefix}\n(Chi-square p = {pval})")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    sns.despine()
    plt.tight_layout()
    plt.show()


@_with_notebook_context
def clinical_pipeline_add_metadata_and_clusters_final_labels(final_metrics, data_full):
    """
    Merge cluster labels into full metadata DataFrame using src_subject_id
    from the actual training data stored in fold_metrics['data'].
    This ensures correct alignment even when not all train_ids were clustered.
    """
    clusters = pd.Series(final_metrics['final_labels'])

    # Get the first modality’s dataframe — all have the same subject order
    modality_dfs = final_metrics.get('data', {})
    if not modality_dfs:
        raise ValueError("final_metrics['data'] is empty; cannot extract subject IDs.")

    # Use the src_subject_id column from the first available modality
    first_modality = list(modality_dfs.keys())[0]
    subj_ids = modality_dfs[first_modality]['src_subject_id'].reset_index(drop=True)

    # Sanity check: should match cluster array length
    if len(subj_ids) != len(clusters):
        print(f"⚠️ Mismatch: {len(subj_ids)} subject IDs vs {len(clusters)} cluster labels.")
        min_len = min(len(subj_ids), len(clusters))
        subj_ids = subj_ids.iloc[:min_len]
        clusters = clusters.iloc[:min_len]
        print(f"Trimmed both to {min_len} entries to align.")

    # Build cluster mapping dataframe
    cluster_df = pd.DataFrame({
        'src_subject_id': subj_ids,
        'Cluster': clusters
    })

    # Merge back into full metadata
    merged = pd.merge(data_full, cluster_df, on='src_subject_id', how='left')

    print(f"✅ Merged clusters for {merged['Cluster'].notna().sum()} subjects (out of {len(merged)} total).")
    return merged


@_with_notebook_context
def clinical_pipeline_chi_square_comparison_final_labels(df, group_col, label_col, title_prefix, save_path):
    """
    Perform chi-square test and plot grouped bar chart.
    """
    # Drop missing values
    df = df.dropna(subset=[group_col, label_col])

    # Make a copy for plotting / stats so we can safely relabel
    df_plot = df.copy()

    # Rename 'HC' -> 'CC' only for phenotype (for plotting and stats)
    if label_col == 'phenotype':
        df_plot[label_col] = df_plot[label_col].replace({'HC': 'CC'})

    # 1. Summarize counts
    summary_df = (
        df_plot.groupby([group_col, label_col])
        .size()
        .reset_index(name='n')
    )

    # 2. Chi-square test
    tbl = pd.crosstab(df_plot[group_col], df_plot[label_col])
    chi2, pval, dof, expected = chi2_contingency(tbl)
    pval = round(pval, 3)

    # 3. Grouped bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=summary_df,
        x=group_col,
        y='n',
        hue=label_col,
        dodge=True
    )
    plt.title(f"{title_prefix}\n(Chi-square p = {pval})")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)
    plt.show()

    # 4. Cluster summary (including CHR percentage if relevant)
    cluster_summary = (
        df_plot.groupby(group_col)
        .agg(
            Size=(group_col, 'size'),
            CHR_percentage=(
                label_col,
                lambda x: (x.eq('CHR').sum() / len(x)) * 100
                if 'CHR' in x.values else None
            )
        )
        .reset_index()
        .sort_values(group_col)
    )
    print(f"\nCluster summary for {title_prefix}")
    print(cluster_summary)
    print("\n" + "-"*60)


@_with_notebook_context
def clinical_pipeline_alluvial_sankey_force_high_top(
    labels_by_modality: dict,
    final_labels,
    stage_order: list,
    final_name="final",
    high_token="high_severity",
    low_token="low_severity",
    final_order="auto",
    arrangement="snap",
    high_y=0.10,
    low_y=0.90,
    other_y=0.50,
    node_pad=22,
    node_thickness=18,
    width=1400,
    height=650,
    title="All modalities -> final (alluvial Sankey)",
):
    """Process clinical pipeline alluvial sankey force high top."""
    return alluvial_sankey_general(
        labels_by_modality=labels_by_modality,
        final_labels=final_labels,
        stage_order=stage_order,
        final_name=final_name,
        high_token=high_token,
        low_token=low_token,
        final_order=final_order,
        arrangement=arrangement,
        high_y=high_y,
        low_y=low_y,
        other_y=other_y,
        node_pad=node_pad,
        node_thickness=node_thickness,
        width=width,
        height=height,
        title=title,
    )


@_with_notebook_context
def clinical_pipeline_summarize_streams(df_paths, stage_order, top_k=20, sample_ids=10):
    """Process clinical pipeline summarize streams."""
    group_cols = stage_order + ["final"]
    total = len(df_paths)

    g = (df_paths
         .groupby(group_cols, dropna=False)
         .agg(
             n=("src_subject_id", "size"),
             example_ids=("src_subject_id", lambda x: list(x.head(sample_ids))),
         )
         .reset_index()
        )

    g["pct"] = (g["n"] / total * 100).round(2)

    # A readable "stream label"
    def make_stream_label(row):
        parts = [f"{c}={row[c]}" for c in stage_order] + [f"final={row['final']}"]
        return " → ".join(parts)

    g["stream"] = g.apply(make_stream_label, axis=1)

    g = g.sort_values("n", ascending=False).head(top_k)

    # Reorder columns nicely
    cols = ["n", "pct", "stream", "example_ids"] + stage_order + ["final"]
    return g[cols]


@_with_notebook_context
def clinical_pipeline_plot_pred_modality(df, name):
    """
    Plots confidence/uncertainty diagnostics for a single modality DataFrame.
    Expected columns (any subset is ok): p_0, p_1, confidence, entropy, margin
    """
    cols = set(df.columns)

    # Compute missing fields if probabilities exist
    if {"p_0", "p_1"}.issubset(cols):
        p0 = df["p_0"].astype(float).to_numpy()
        p1 = df["p_1"].astype(float).to_numpy()

        if "confidence" not in cols:
            df = df.copy()
            df["confidence"] = np.maximum(p0, p1)

        if "margin" not in cols:
            df = df.copy()
            df["margin"] = np.abs(p1 - p0)

        if "entropy" not in cols:
            df = df.copy()
            eps = 1e-12
            df["entropy"] = -(p0 * np.log(p0 + eps) + p1 * np.log(p1 + eps))

    # Helper to plot a histogram if column exists
    def hist_if_exists(col, bins=30, xlabel=None):
        if col in df.columns:
            plt.figure()
            plt.hist(df[col].dropna().astype(float), bins=bins)
            plt.xlabel(xlabel or col)
            plt.ylabel("Count")
            plt.title(f"{name}: {col} distribution (unlabeled hold-out)")
            plt.show()

    # 1) Histograms
    hist_if_exists("p_1", xlabel="Predicted probability p(class=1)")
    hist_if_exists("confidence", xlabel="Confidence = max(p_0, p_1)")
    hist_if_exists("entropy", xlabel="Entropy (higher = more uncertain)")
    hist_if_exists("margin", xlabel="Margin = |p_1 - p_0| (lower = more uncertain)")

    # 2) Scatter plots that are often informative
    if ("confidence" in df.columns) and ("entropy" in df.columns):
        plt.figure()
        plt.scatter(df["confidence"].astype(float), df["entropy"].astype(float), s=10)
        plt.xlabel("Confidence")
        plt.ylabel("Entropy")
        plt.title(f"{name}: confidence vs entropy")
        plt.show()

    if ("confidence" in df.columns) and ("margin" in df.columns):
        plt.figure()
        plt.scatter(df["confidence"].astype(float), df["margin"].astype(float), s=10)
        plt.xlabel("Confidence")
        plt.ylabel("Margin")
        plt.title(f"{name}: confidence vs margin")
        plt.show()

    # 3) Print “most uncertain” cases (lowest confidence / lowest margin / highest entropy)
    # (useful for manual review)
    print(f"\n=== {name}: most uncertain examples ===")

    if "confidence" in df.columns:
        print("\nLowest confidence:")
        display(df.sort_values("confidence", ascending=True).head(10))

    if "margin" in df.columns:
        print("\nLowest margin:")
        display(df.sort_values("margin", ascending=True).head(10))

    if "entropy" in df.columns:
        print("\nHighest entropy:")
        display(df.sort_values("entropy", ascending=False).head(10))


@_with_notebook_context
def clinical_pipeline_add_metadata_and_clusters_validation_individual_labels(dict_final, data_full, mod_num):
    """
    Merge cluster labels into full metadata DataFrame using src_subject_id
    from the actual training data stored in fold_metrics['data'].
    This ensures correct alignment even when not all train_ids were clustered.
    """
    clusters = pd.Series(labels_test_modalities[mod_num])

    # Get the first modality’s dataframe — all have the same subject order
    modality_dfs = dict_final
    if not modality_dfs:
        raise ValueError("dict_final is empty; cannot extract subject IDs.")

    # Use the src_subject_id column from the first available modality
    first_modality = list(modality_dfs.keys())[0]
    subj_ids = modality_dfs[first_modality]['src_subject_id'].reset_index(drop=True)

    # Sanity check: should match cluster array length
    if len(subj_ids) != len(clusters):
        print(f"⚠️ Mismatch: {len(subj_ids)} subject IDs vs {len(clusters)} cluster labels.")
        min_len = min(len(subj_ids), len(clusters))
        subj_ids = subj_ids.iloc[:min_len]
        clusters = clusters.iloc[:min_len]
        print(f"Trimmed both to {min_len} entries to align.")

    # Build cluster mapping dataframe
    cluster_df = pd.DataFrame({
        'src_subject_id': subj_ids,
        'Cluster': clusters
    })

    # Merge back into full metadata
    merged = pd.merge(data_full, cluster_df, on='src_subject_id', how='left')

    print(f"✅ Merged clusters for {merged['Cluster'].notna().sum()} subjects (out of {len(merged)} total).")
    return merged


@_with_notebook_context
def clinical_pipeline_chi_square_comparison_validation_individual_labels(df, group_col, label_col, title_prefix):
    """
    Perform chi-square test and plot grouped bar chart.
    """
    # Drop missing values
    df = df.dropna(subset=[group_col, label_col])

    # Make a copy for plotting / stats so we can safely relabel
    df_plot = df.copy()

    # Rename 'HC' -> 'CC' only for phenotype (for plotting and stats)
    if label_col == 'phenotype':
        df_plot[label_col] = df_plot[label_col].replace({'HC': 'CC'})

    # 1. Summarize counts
    summary_df = (
        df_plot.groupby([group_col, label_col])
        .size()
        .reset_index(name='n')
    )

    # 2. Chi-square test
    tbl = pd.crosstab(df_plot[group_col], df_plot[label_col])
    chi2, pval, dof, expected = chi2_contingency(tbl)
    pval = round(pval, 3)

    # 3. Grouped bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=summary_df,
        x=group_col,
        y='n',
        hue=label_col,
        dodge=True
    )
    plt.title(f"{title_prefix}\n(Chi-square p = {pval})")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    sns.despine()
    plt.tight_layout()
    plt.show()


@_with_notebook_context
def clinical_pipeline_parse_stream(stream_str):
    """
    Parse stream like:
      "Psychoticism=low_severity → Detachment=high_severity → ... → final=1"
    into ordered list of (domain, label).
    """
    if pd.isna(stream_str):
        return []
    parts = [p.strip() for p in ARROW_PAT.split(str(stream_str).strip()) if p.strip()]
    out = []
    for p in parts:
        if "=" in p:
            dom, lab = p.split("=", 1)
            out.append((dom.strip(), lab.strip()))
        else:
            out.append((p.strip(), "<NA>"))
    return out


@_with_notebook_context
def clinical_pipeline_infer_stage_order(df, stream_col="stream"):
    """
    Infer stage order from the first non-null stream.
    Assumes all streams follow the same domain order.
    """
    s = df[stream_col].dropna().astype(str)
    if s.empty:
        raise ValueError("No streams found to infer stage order.")
    path = clinical_pipeline_parse_stream(s.iloc[0])
    return [d for d, _ in path]


@_with_notebook_context
def clinical_pipeline_normalize(v, eps=1e-12):
    """Process clinical pipeline normalize."""
    v = np.asarray(v, dtype=float)
    s = v.sum()
    if s <= 0:
        return np.ones_like(v) / max(1, len(v))
    return v / (s + eps)


@_with_notebook_context
def clinical_pipeline_build_prefix_next(df, stream_col="stream", n_col="n"):
    """
    Returns:
      prefix_next: dict[prefix_tuple_of_(domain,label)] -> dict[next_token_(domain,label)|<END>] -> weight
      prefix_mass: dict[prefix] -> total weight passing through prefix
    Where prefix is a tuple of (domain,label) tokens, e.g.:
      prefix = (("Psychoticism","low_severity"), ("Detachment","high_severity"))
    and next token is the next (domain,label) or "<END>".
    """
    prefix_next = {}
    prefix_mass = {}

    for _, row in df.iterrows():
        w = float(row.get(n_col, 1.0))
        path = clinical_pipeline_parse_stream(row[stream_col])
        if not path:
            continue

        # For each prefix (including empty prefix), record what comes next
        for i in range(len(path)):
            prefix = tuple(path[:i])  # empty prefix for i=0
            nxt = path[i]             # next token
            prefix_mass[prefix] = prefix_mass.get(prefix, 0.0) + w
            d = prefix_next.setdefault(prefix, {})
            d[nxt] = d.get(nxt, 0.0) + w

        # terminal transition from full path to END
        full_prefix = tuple(path)
        prefix_mass[full_prefix] = prefix_mass.get(full_prefix, 0.0) + w
        d = prefix_next.setdefault(full_prefix, {})
        d["<END>"] = d.get("<END>", 0.0) + w

    return prefix_next, prefix_mass


@_with_notebook_context
def clinical_pipeline_compare_prefix_structure(df_disc, df_test, stream_col="stream", n_col="n", eps=1e-12):
    """
    For each prefix, compare the conditional distribution over next tokens:
        P_disc(next | prefix)  vs  P_test(next | prefix)

    Returns DataFrame with:
      - prefix_str
      - depth
      - js_next (Jensen–Shannon distance on next-step distributions)
      - mass_disc / mass_test (how much data passes through prefix; useful for weighting but not "size-equality")
      - top_next_disc / top_next_test (most likely next step)
      - support_next_overlap (Jaccard on next-token supports)
    """
    pn_d, pm_d = clinical_pipeline_build_prefix_next(df_disc, stream_col, n_col)
    pn_t, pm_t = clinical_pipeline_build_prefix_next(df_test, stream_col, n_col)

    prefixes = set(pn_d.keys()) | set(pn_t.keys())

    rows = []
    for pref in prefixes:
        nd = pn_d.get(pref, {})
        nt = pn_t.get(pref, {})

        keys = set(nd.keys()) | set(nt.keys())
        # Align next-token vectors
        vd = np.array([nd.get(k, 0.0) for k in keys], dtype=float)
        vt = np.array([nt.get(k, 0.0) for k in keys], dtype=float)

        pd_ = clinical_pipeline_normalize(vd, eps=eps)
        pt_ = clinical_pipeline_normalize(vt, eps=eps)

        js = float(jensenshannon(pd_ + eps, pt_ + eps, base=2.0))

        # Next-token support overlap (presence/absence, structure)
        supp_d = {k for k, v in nd.items() if v > 0}
        supp_t = {k for k, v in nt.items() if v > 0}
        supp_j = len(supp_d & supp_t) / max(1, len(supp_d | supp_t))

        # Most likely next token in each
        top_d = max(nd.items(), key=lambda x: x[1])[0] if nd else None
        top_t = max(nt.items(), key=lambda x: x[1])[0] if nt else None

        def tok_str(tok):
            if tok == "<END>":
                return "<END>"
            if tok is None:
                return "<NONE>"
            return f"{tok[0]}={tok[1]}"

        prefix_str = " → ".join([f"{d}={l}" for d, l in pref]) if pref else "<START>"
        rows.append({
            "prefix_str": prefix_str,
            "depth": len(pref),
            "js_next": js,
            "mass_disc": pm_d.get(pref, 0.0),
            "mass_test": pm_t.get(pref, 0.0),
            "top_next_disc": tok_str(top_d),
            "top_next_test": tok_str(top_t),
            "support_next_jaccard": supp_j,
            "prefix_exists_in_disc": pref in pn_d,
            "prefix_exists_in_test": pref in pn_t,
        })

    out = pd.DataFrame(rows)
    # A useful default sorting: prioritize structurally-different AND commonly-used prefixes
    out["mass_min"] = np.minimum(out["mass_disc"], out["mass_test"])
    out = out.sort_values(["mass_min", "js_next"], ascending=[False, False]).reset_index(drop=True)
    return out


@_with_notebook_context
def clinical_pipeline_plot_top_prefix_differences(prefix_report, top_n=20, min_depth=1):
    """
    Barh plot of top prefixes by JS(next) after filtering.
    """
    d = prefix_report[prefix_report["depth"] >= min_depth].copy()
    d = d.sort_values("js_next", ascending=False).head(top_n)

    plt.figure(figsize=(11, max(4, 0.35 * len(d))))
    y = np.arange(len(d))[::-1]
    plt.barh(y, d["js_next"].iloc[::-1].to_numpy())
    plt.yticks(y, d["prefix_str"].iloc[::-1].to_list(), fontsize=9)
    plt.xlabel("Jensen–Shannon distance of P(next | prefix)")
    plt.title("Most structurally different prefixes (next-step rule)")
    plt.tight_layout()
    plt.show()


@_with_notebook_context
def clinical_pipeline_final_mapping_table(df, stream_col="stream", n_col="n", final_domain="final"):
    """
    Builds a table over modality-prefixes (everything up to but excluding final)
    with P(final=label | prefix) computed within each dataset.

    Returns DataFrame:
      prefix_str, final_label, weight
    and also a pivoted table of P(final=...) by prefix.
    """
    rows = []
    for _, row in df.iterrows():
        w = float(row.get(n_col, 1.0))
        path = clinical_pipeline_parse_stream(row[stream_col])
        if not path:
            continue

        # split into prefix (before final) and final token
        final_tokens = [t for t in path if t[0] == final_domain]
        if not final_tokens:
            # If final isn't explicitly present, skip
            continue
        final_tok = final_tokens[-1]  # in case of duplicates, take last
        final_label = final_tok[1]

        # prefix = all tokens before the final token position (first occurrence)
        # more robust: take all tokens except the final-domain token(s)
        prefix = tuple([t for t in path if t[0] != final_domain])

        prefix_str = " → ".join([f"{d}={l}" for d, l in prefix]) if prefix else "<NO_MODALITIES>"
        rows.append({"prefix_str": prefix_str, "final_label": final_label, "w": w})

    long = pd.DataFrame(rows)
    if long.empty:
        return long, pd.DataFrame()

    # Compute conditional probabilities P(final_label | prefix)
    grp = long.groupby(["prefix_str", "final_label"], dropna=False)["w"].sum().reset_index()
    totals = grp.groupby("prefix_str")["w"].sum().reset_index().rename(columns={"w": "w_total"})
    grp = grp.merge(totals, on="prefix_str", how="left")
    grp["p_final_given_prefix"] = grp["w"] / grp["w_total"]

    pivot = grp.pivot_table(index="prefix_str", columns="final_label", values="p_final_given_prefix", fill_value=0.0)
    return grp, pivot


@_with_notebook_context
def clinical_pipeline_compare_final_mapping(df_disc, df_test, stream_col="stream", n_col="n", final_domain="final"):
    """
    Compare P(final | modalities) between discovery and test.
    Returns a table with per-prefix deltas per final label plus summary metrics.
    """
    long_d, piv_d = clinical_pipeline_final_mapping_table(df_disc, stream_col, n_col, final_domain)
    long_t, piv_t = clinical_pipeline_final_mapping_table(df_test, stream_col, n_col, final_domain)

    if piv_d.empty or piv_t.empty:
        return pd.DataFrame(), {"note": "No final mapping found (missing final tokens?)"}

    # align
    idx = sorted(set(piv_d.index) | set(piv_t.index))
    cols = sorted(set(piv_d.columns) | set(piv_t.columns))
    A = piv_d.reindex(index=idx, columns=cols).fillna(0.0)
    B = piv_t.reindex(index=idx, columns=cols).fillna(0.0)

    # Delta per final label
    delta = (B - A)
    out = delta.copy()
    out.columns = [f"delta_final={c}" for c in out.columns]
    out.insert(0, "prefix_str", out.index)

    # A per-prefix summary: JS distance between final distributions for that prefix
    js_list = []
    for i in range(A.shape[0]):
        p = clinical_pipeline_normalize(A.iloc[i].to_numpy())
        q = clinical_pipeline_normalize(B.iloc[i].to_numpy())
        js_list.append(float(jensenshannon(p + 1e-12, q + 1e-12, base=2.0)))
    out["js_final_given_prefix"] = js_list

    # Add weights: how common the prefix is (within each dataset)
    wD = long_d.groupby("prefix_str")["w"].sum() if not long_d.empty else pd.Series(dtype=float)
    wT = long_t.groupby("prefix_str")["w"].sum() if not long_t.empty else pd.Series(dtype=float)
    out["w_disc"] = out["prefix_str"].map(wD).fillna(0.0)
    out["w_test"] = out["prefix_str"].map(wT).fillna(0.0)
    out["w_min"] = np.minimum(out["w_disc"], out["w_test"])

    # Sort by (common prefixes) then by biggest JS shift in final mapping
    out = out.sort_values(["w_min", "js_final_given_prefix"], ascending=[False, False]).reset_index(drop=True)

    # Global summary metric: weighted average JS over prefixes (weights = w_min)
    w = out["w_min"].to_numpy()
    if w.sum() > 0:
        global_js = float(np.sum(out["js_final_given_prefix"].to_numpy() * w) / w.sum())
    else:
        global_js = float(out["js_final_given_prefix"].mean())

    metrics = {
        "num_prefixes_union": len(out),
        "weighted_js_final_given_prefix": global_js,
        "final_labels": cols,
    }
    return out, metrics


@_with_notebook_context
def clinical_pipeline_plot_top_final_mapping_shifts(final_cmp, top_n=20):
    """Process clinical pipeline plot top final mapping shifts."""
    d = final_cmp.head(top_n).copy()
    plt.figure(figsize=(11, max(4, 0.35 * len(d))))
    y = np.arange(len(d))[::-1]
    plt.barh(y, d["js_final_given_prefix"].iloc[::-1].to_numpy())
    plt.yticks(y, d["prefix_str"].iloc[::-1].to_list(), fontsize=9)
    plt.xlabel("JS distance between P(final | prefix) in test vs discovery")
    plt.title("Prefixes with biggest changes in final mapping")
    plt.tight_layout()
    plt.show()


@_with_notebook_context
def clinical_pipeline_stream_presence_and_topk(df_disc, df_test, stream_col="stream", n_col="n", topk=30):
    """Process clinical pipeline stream presence and topk."""
    A = set(df_disc[stream_col].dropna().astype(str))
    B = set(df_test[stream_col].dropna().astype(str))

    presence = {
        "unique_streams_disc": len(A),
        "unique_streams_test": len(B),
        "stream_jaccard_presence": len(A & B) / max(1, len(A | B)),
        "coverage_disc_in_test": len(A & B) / max(1, len(A)),
        "coverage_test_in_disc": len(A & B) / max(1, len(B)),
    }

    pD = df_disc.groupby(stream_col)[n_col].sum().sort_values(ascending=False)
    pT = df_test.groupby(stream_col)[n_col].sum().sort_values(ascending=False)
    pD = pD / pD.sum()
    pT = pT / pT.sum()

    topD = set(pD.index[: min(topk, len(pD))].astype(str))
    topT = set(pT.index[: min(topk, len(pT))].astype(str))

    presence[f"top{topk}_jaccard_by_rank"] = len(topD & topT) / max(1, len(topD | topT))
    presence["top_disc_only"] = sorted(list(topD - topT))[:10]
    presence["top_test_only"] = sorted(list(topT - topD))[:10]
    return presence, pD, pT


@_with_notebook_context
def clinical_pipeline_sankey_from_streams(df, stream_col="stream", n_col="n", max_edges=200):
    """
    Build a Sankey graph from full streams.
    Nodes are stage-specific label tokens: f"{domain}={label}".
    Edges connect consecutive tokens. We prune to max_edges by weight.
    """
    if not _HAS_PLOTLY:
        raise RuntimeError("Plotly not installed; cannot draw sankey.")

    edge_w = {}
    for _, row in df.iterrows():
        w = float(row.get(n_col, 1.0))
        path = clinical_pipeline_parse_stream(row[stream_col])
        toks = [f"{d}={l}" for d, l in path]
        for a, b in zip(toks[:-1], toks[1:]):
            edge_w[(a, b)] = edge_w.get((a, b), 0.0) + w

    # prune
    edges = sorted(edge_w.items(), key=lambda x: -x[1])[:max_edges]
    nodes = {}
    def nid(x):
        if x not in nodes:
            nodes[x] = len(nodes)
        return nodes[x]

    src, tgt, val = [], [], []
    for (a, b), w in edges:
        src.append(nid(a))
        tgt.append(nid(b))
        val.append(w)

    labels = [None] * len(nodes)
    for k, i in nodes.items():
        labels[i] = k

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=12, thickness=12),
        link=dict(source=src, target=tgt, value=val),
    )])
    return fig


@_with_notebook_context
def clinical_pipeline_full_structure_report(stream_summary, stream_summary_test, stream_col="stream", n_col="n", topk=30, final_domain="final"):
    # Presence + top-k overlap
    """Process clinical pipeline full structure report."""
    presence, pD, pT = clinical_pipeline_stream_presence_and_topk(stream_summary, stream_summary_test, stream_col, n_col, topk=topk)

    # Prefix-tree structural differences
    prefix_report = clinical_pipeline_compare_prefix_structure(stream_summary, stream_summary_test, stream_col, n_col)

    # Full mapping to final
    final_cmp, final_metrics = clinical_pipeline_compare_final_mapping(stream_summary, stream_summary_test, stream_col, n_col, final_domain)

    return {
        "presence_metrics": presence,
        "p_stream_disc": pD,
        "p_stream_test": pT,
        "prefix_report": prefix_report,
        "final_mapping_compare": final_cmp,
        "final_mapping_metrics": final_metrics,
    }


@_with_notebook_context
def clinical_pipeline_all_streams_table(stream_summary, stream_summary_test, stream_col="stream", n_col="n"):
    # Aggregate in case there are duplicate stream rows
    """Process clinical pipeline all streams table."""
    disc = (stream_summary
            .groupby(stream_col, dropna=False)[n_col]
            .sum()
            .rename("n_disc")
            .to_frame())

    test = (stream_summary_test
            .groupby(stream_col, dropna=False)[n_col]
            .sum()
            .rename("n_test")
            .to_frame())

    # Outer join gives union of streams
    tbl = disc.join(test, how="outer").fillna(0)

    # Add proportions within each dataset
    N_disc = tbl["n_disc"].sum()
    N_test = tbl["n_test"].sum()

    tbl["p_disc"] = tbl["n_disc"] / N_disc if N_disc > 0 else np.nan
    tbl["p_test"] = tbl["n_test"] / N_test if N_test > 0 else np.nan

    # Helpful comparisons
    tbl["delta_p"] = tbl["p_test"] - tbl["p_disc"]
    tbl["abs_delta_p"] = tbl["delta_p"].abs()
    tbl["log2_fc"] = np.log2((tbl["p_test"] + 1e-12) / (tbl["p_disc"] + 1e-12))

    # Make stream a real column, sort by biggest shift
    tbl = tbl.reset_index().rename(columns={stream_col: "stream"})
    tbl = tbl.sort_values("abs_delta_p", ascending=False).reset_index(drop=True)

    return tbl


# -----------------------------------------------------------------------------
# multiclust_extended analysis helpers
# Implementations are preserved from the original notebook cells.
# -----------------------------------------------------------------------------

@_with_notebook_context
def multiclust_extended_print_remaining_after_full_missing_modality_removal(df, df_name, meta, modalities, subject_id_column="src_subject_id"):
    """Process multiclust extended print remaining after full missing modality removal."""
    modality_to_cols = {
        modality: [
            col for col in meta.loc[meta["Modality"] == modality, "ElementName"]
            if col in df.columns
        ]
        for modality in modalities
    }

    full_missing_mask = pd.Series(False, index=df.index)
    for cols in modality_to_cols.values():
        if cols:
            full_missing_mask |= df[cols].isna().all(axis=1)

    print(
        f"{df_name}: {(~full_missing_mask).sum()} participants remain "
        f"after removing participants with >=1 fully missing modality "
        f"({full_missing_mask.sum()} removed)."
    )


@_with_notebook_context
def multiclust_extended_truthy_profile_value(value):
    """Process multiclust extended truthy profile value."""
    return str(value).strip().strip('"\'').upper() in {"TRUE", "1", "YES", "Y"}


@_with_notebook_context
def multiclust_extended_parse_profile_exports(profile_path):
    """Process multiclust extended parse profile exports."""
    exports = {}
    if profile_path is None or not profile_path.exists():
        return exports
    for line in profile_path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[len("export "):].split("=", 1)
        value = value.strip()
        # Keep literal defaults such as ${VAR:-0} as-is; simple quoted values are enough for the flag check.
        exports[key.strip()] = value.strip('"\'')
    return exports


@_with_notebook_context
def multiclust_extended_infer_notebook_profile():
    """Process multiclust extended infer notebook profile."""
    notebook_dir = Path.cwd().name
    profile_candidates = []
    if notebook_dir == "clinical_paper":
        profile_candidates.append("clinical_paper")
    elif notebook_dir == "multiclust_extended":
        profile_candidates.append("multiclust_extended")
    elif notebook_dir == "prospect":
        profile_candidates.append("prospect")
    elif notebook_dir == "schizbull_legacy":
        profile_candidates.append("schizbull_legacy")
    profile_candidates.append(os.environ.get("RUN_PROFILE", ""))
    for candidate in profile_candidates:
        if candidate:
            return candidate
    return None


@_with_notebook_context
def multiclust_extended_find_repo_root(start):
    """Process multiclust extended find repo root."""
    start = Path(start).resolve()
    for parent in [start] + list(start.parents):
        if (parent / "run_profiles").is_dir() and (parent / "full_pipeline.py").exists():
            return parent
    return None


@_with_notebook_context
def multiclust_extended_profile_enabled_for_sensitivity(repo_root, profile_name):
    """Process multiclust extended profile enabled for sensitivity."""
    if not profile_name or repo_root is None:
        return None, None
    profile_path = repo_root / "run_profiles" / f"{profile_name}.sh"
    exports = multiclust_extended_parse_profile_exports(profile_path)
    value = exports.get("DO_CLUSTER_VALIDATION_SENSITIVITY", "FALSE")
    return multiclust_extended_truthy_profile_value(value), profile_path


@_with_notebook_context
def multiclust_extended_display_if_available(obj):
    """Process multiclust extended display if available."""
    if "display" in globals():
        display(obj)
    else:
        print(obj)


@_with_notebook_context
def multiclust_extended_get_nested(dct, path, default=np.nan):
    """Process multiclust extended get nested."""
    cur = dct
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


@_with_notebook_context
def multiclust_extended_flatten_sensitivity_results(results):
    """Process multiclust extended flatten sensitivity results."""
    rows = []
    for solution, payload in results.get("solutions", {}).items():
        rows.append({
            "solution": solution,
            "kind": payload.get("kind"),
            "observed_k": multiclust_extended_get_nested(payload, ["observed_quality", "k"]),
            "observed_n": multiclust_extended_get_nested(payload, ["observed_quality", "n"]),
            "observed_n_features": multiclust_extended_get_nested(payload, ["observed_quality", "n_features"]),
            "observed_composite": multiclust_extended_get_nested(payload, ["observed_quality", "composite"]),
            "observed_silhouette": multiclust_extended_get_nested(payload, ["observed_quality", "silhouette"]),
            "observed_calinski_harabasz": multiclust_extended_get_nested(payload, ["observed_quality", "calinski_harabasz"]),
            "observed_davies_bouldin": multiclust_extended_get_nested(payload, ["observed_quality", "davies_bouldin"]),
            "k1_composite": multiclust_extended_get_nested(payload, ["uni_cluster_baseline", "quality", "composite"]),
            "pc1_variance_explained": multiclust_extended_get_nested(payload, ["pc1_median_split", "pc1_variance_explained"]),
            "pc1_median_composite": multiclust_extended_get_nested(payload, ["pc1_median_split", "quality", "composite"]),
            "pc1_median_ari_with_observed": multiclust_extended_get_nested(payload, ["pc1_median_split", "ari_with_observed_labels"]),
            "pc1_median_stability_ari": multiclust_extended_get_nested(payload, ["pc1_median_split", "bootstrap_stability", "mean_ari"]),
            "pc1_median_stability_sd_ari": multiclust_extended_get_nested(payload, ["pc1_median_split", "bootstrap_stability", "sd_ari"]),
            "dip_available": multiclust_extended_get_nested(payload, ["dip_test_pc1", "available"], default=False),
            "dip_statistic_pc1": multiclust_extended_get_nested(payload, ["dip_test_pc1", "dip"]),
            "dip_p_value_pc1": multiclust_extended_get_nested(payload, ["dip_test_pc1", "p_value"]),
            "gap_selected_k_tibshirani": multiclust_extended_get_nested(payload, ["gap_statistic", "selected_k_tibshirani_rule"]),
            "gap_selected_k_max_gap": multiclust_extended_get_nested(payload, ["gap_statistic", "selected_k_max_gap"]),
            "sigclust_cluster_index": multiclust_extended_get_nested(payload, ["sigclust_approx", "observed_cluster_index"]),
            "sigclust_p_value": multiclust_extended_get_nested(payload, ["sigclust_approx", "p_value"]),
            "sigclust_null_mean_cluster_index": multiclust_extended_get_nested(payload, ["sigclust_approx", "null_mean_cluster_index"]),
            "gaussian_null_p_quality": multiclust_extended_get_nested(payload, ["covariance_matched_gaussian_null", "p_quality_ge_observed_recluster"]),
            "gaussian_null_p_stability": multiclust_extended_get_nested(payload, ["covariance_matched_gaussian_null", "p_stability_ge_observed_recluster"]),
            "observed_recluster_stability_ari": multiclust_extended_get_nested(payload, ["covariance_matched_gaussian_null", "observed_recluster_stability", "mean_ari"]),
            "gaussian_null_mean_stability_ari": multiclust_extended_get_nested(payload, ["covariance_matched_gaussian_null", "null_stability_mean_ari"]),
            "gaussian_null_mean_quality": multiclust_extended_get_nested(payload, ["covariance_matched_gaussian_null", "null_quality_mean"]),
        })
    return pd.DataFrame(rows)


@_with_notebook_context
def multiclust_extended_build_group_palette(modality, group_order, modality_palettes=None, default_palette=None):
    """Process multiclust extended build group palette."""
    return modality_cluster_palette([str(g) for g in group_order], modality=modality)


@_with_notebook_context
def multiclust_extended_add_metadata_and_clusters_individual_labels(final_metrics, data_full, mod_num):
    """
    Merge cluster labels into full metadata DataFrame using src_subject_id
    from the actual training data stored in fold_metrics['data'].
    This ensures correct alignment even when not all train_ids were clustered.
    """
    clusters = pd.Series(final_metrics['individual_labels'][mod_num])

    # Get the first modality’s dataframe — all have the same subject order
    modality_dfs = final_metrics.get('data', {})
    if not modality_dfs:
        raise ValueError("final_metrics['data'] is empty; cannot extract subject IDs.")

    # Use the src_subject_id column from the first available modality
    first_modality = list(modality_dfs.keys())[0]
    subj_ids = modality_dfs[first_modality]['src_subject_id'].reset_index(drop=True)

    # Sanity check: should match cluster array length
    if len(subj_ids) != len(clusters):
        print(f"⚠️ Mismatch: {len(subj_ids)} subject IDs vs {len(clusters)} cluster labels.")
        min_len = min(len(subj_ids), len(clusters))
        subj_ids = subj_ids.iloc[:min_len]
        clusters = clusters.iloc[:min_len]
        print(f"Trimmed both to {min_len} entries to align.")

    # Build cluster mapping dataframe
    cluster_df = pd.DataFrame({
        'src_subject_id': subj_ids,
        'Cluster': clusters
    })

    # Merge back into full metadata
    merged = pd.merge(data_full, cluster_df, on='src_subject_id', how='left')

    print(f"✅ Merged clusters for {merged['Cluster'].notna().sum()} subjects (out of {len(merged)} total).")
    return merged


@_with_notebook_context
def multiclust_extended_chi_square_comparison_individual_labels(df, group_col, label_col, title_prefix):
    """
    Perform chi-square test and plot grouped bar chart.
    """
    # Drop missing values
    df = df.dropna(subset=[group_col, label_col])

    # Make a copy for plotting / stats so we can safely relabel
    df_plot = df.copy()

    # Rename 'HC' -> 'CC' only for phenotype (for plotting and stats)
    if label_col == 'phenotype':
        df_plot[label_col] = df_plot[label_col].replace({'HC': 'CC'})

    # 1. Summarize counts
    summary_df = (
        df_plot.groupby([group_col, label_col])
        .size()
        .reset_index(name='n')
    )

    # 2. Chi-square test
    tbl = pd.crosstab(df_plot[group_col], df_plot[label_col])
    chi2, pval, dof, expected = chi2_contingency(tbl)
    pval = round(pval, 3)

    # 3. Grouped bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=summary_df,
        x=group_col,
        y='n',
        hue=label_col,
        dodge=True
    )
    plt.title(f"{title_prefix}\n(Chi-square p = {pval})")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    sns.despine()
    plt.tight_layout()
    plt.show()


@_with_notebook_context
def multiclust_extended_add_metadata_and_clusters_final_labels(final_metrics, data_full):
    """
    Merge cluster labels into full metadata DataFrame using src_subject_id
    from the actual training data stored in fold_metrics['data'].
    This ensures correct alignment even when not all train_ids were clustered.
    """
    clusters = pd.Series(final_metrics['final_labels'])

    # Get the first modality’s dataframe — all have the same subject order
    modality_dfs = final_metrics.get('data', {})
    if not modality_dfs:
        raise ValueError("final_metrics['data'] is empty; cannot extract subject IDs.")

    # Use the src_subject_id column from the first available modality
    first_modality = list(modality_dfs.keys())[0]
    subj_ids = modality_dfs[first_modality]['src_subject_id'].reset_index(drop=True)

    # Sanity check: should match cluster array length
    if len(subj_ids) != len(clusters):
        print(f"⚠️ Mismatch: {len(subj_ids)} subject IDs vs {len(clusters)} cluster labels.")
        min_len = min(len(subj_ids), len(clusters))
        subj_ids = subj_ids.iloc[:min_len]
        clusters = clusters.iloc[:min_len]
        print(f"Trimmed both to {min_len} entries to align.")

    # Build cluster mapping dataframe
    cluster_df = pd.DataFrame({
        'src_subject_id': subj_ids,
        'Cluster': clusters
    })

    # Merge back into full metadata
    merged = pd.merge(data_full, cluster_df, on='src_subject_id', how='left')

    print(f"✅ Merged clusters for {merged['Cluster'].notna().sum()} subjects (out of {len(merged)} total).")
    return merged


@_with_notebook_context
def multiclust_extended_chi_square_comparison_final_labels(df, group_col, label_col, title_prefix, save_path):
    """
    Perform chi-square test and plot grouped bar chart.
    """
    # Drop missing values
    df = df.dropna(subset=[group_col, label_col])

    # Make a copy for plotting / stats so we can safely relabel
    df_plot = df.copy()

    # Rename 'HC' -> 'CC' only for phenotype (for plotting and stats)
    if label_col == 'phenotype':
        df_plot[label_col] = df_plot[label_col].replace({'HC': 'CC'})

    # 1. Summarize counts
    summary_df = (
        df_plot.groupby([group_col, label_col])
        .size()
        .reset_index(name='n')
    )

    # 2. Chi-square test
    tbl = pd.crosstab(df_plot[group_col], df_plot[label_col])
    chi2, pval, dof, expected = chi2_contingency(tbl)
    pval = round(pval, 3)

    # 3. Grouped bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=summary_df,
        x=group_col,
        y='n',
        hue=label_col,
        dodge=True
    )
    plt.title(f"{title_prefix}\n(Chi-square p = {pval})")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)
    plt.show()

    # 4. Cluster summary (including CHR percentage if relevant)
    cluster_summary = (
        df_plot.groupby(group_col)
        .agg(
            Size=(group_col, 'size'),
            CHR_percentage=(
                label_col,
                lambda x: (x.eq('CHR').sum() / len(x)) * 100
                if 'CHR' in x.values else None
            )
        )
        .reset_index()
        .sort_values(group_col)
    )
    print(f"\nCluster summary for {title_prefix}")
    print(cluster_summary)
    print("\n" + "-"*60)


@_with_notebook_context
def multiclust_extended_alluvial_sankey_force_high_top(
    labels_by_modality: dict,
    final_labels,
    stage_order: list,
    final_name="final",
    high_token="high_severity",
    low_token="low_severity",
    final_order="auto",
    arrangement="snap",
    high_y=0.10,
    low_y=0.90,
    other_y=0.50,
    node_pad=22,
    node_thickness=18,
    width=1400,
    height=650,
    title="All modalities -> final (alluvial Sankey)",
):
    """Process multiclust extended alluvial sankey force high top."""
    return alluvial_sankey_general(
        labels_by_modality=labels_by_modality,
        final_labels=final_labels,
        stage_order=stage_order,
        final_name=final_name,
        high_token=high_token,
        low_token=low_token,
        final_order=final_order,
        arrangement=arrangement,
        high_y=high_y,
        low_y=low_y,
        other_y=other_y,
        node_pad=node_pad,
        node_thickness=node_thickness,
        width=width,
        height=height,
        title=title,
    )


@_with_notebook_context
def multiclust_extended_summarize_streams(df_paths, stage_order, top_k=20, sample_ids=10):
    """Process multiclust extended summarize streams."""
    group_cols = stage_order + ["final"]
    total = len(df_paths)

    g = (df_paths
         .groupby(group_cols, dropna=False)
         .agg(
             n=("src_subject_id", "size"),
             example_ids=("src_subject_id", lambda x: list(x.head(sample_ids))),
         )
         .reset_index()
        )

    g["pct"] = (g["n"] / total * 100).round(2)

    # A readable "stream label"
    def make_stream_label(row):
        parts = [f"{c}={row[c]}" for c in stage_order] + [f"final={row['final']}"]
        return " → ".join(parts)

    g["stream"] = g.apply(make_stream_label, axis=1)

    g = g.sort_values("n", ascending=False).head(top_k)

    # Reorder columns nicely
    cols = ["n", "pct", "stream", "example_ids"] + stage_order + ["final"]
    return g[cols]


@_with_notebook_context
def multiclust_extended_summarize_streams_clinical(df_paths, stage_order, top_k=20, sample_ids=10):
    """Process multiclust extended summarize streams clinical."""
    group_cols = stage_order
    total = len(df_paths)

    g = (df_paths
         .groupby(group_cols, dropna=False)
         .agg(
             n=("src_subject_id", "size"),
             example_ids=("src_subject_id", lambda x: list(x.head(sample_ids))),
         )
         .reset_index()
        )

    g["pct"] = (g["n"] / total * 100).round(2)

    # A readable "stream label"
    def make_stream_label(row):
        parts = [f"{c}={row[c]}" for c in stage_order]
        return " → ".join(parts)

    g["stream"] = g.apply(make_stream_label, axis=1)

    g = g.sort_values("n", ascending=False).head(top_k)

    # Reorder columns nicely
    cols = ["n", "pct", "stream", "example_ids"] + stage_order
    return g[cols]


@_with_notebook_context
def multiclust_extended_plot_pred_modality(df, name):
    """
    Plots confidence/uncertainty diagnostics for a single modality DataFrame.
    Expected columns (any subset is ok): p_0, p_1, confidence, entropy, margin
    """
    cols = set(df.columns)

    # Compute missing fields if probabilities exist
    if {"p_0", "p_1"}.issubset(cols):
        p0 = df["p_0"].astype(float).to_numpy()
        p1 = df["p_1"].astype(float).to_numpy()

        if "confidence" not in cols:
            df = df.copy()
            df["confidence"] = np.maximum(p0, p1)

        if "margin" not in cols:
            df = df.copy()
            df["margin"] = np.abs(p1 - p0)

        if "entropy" not in cols:
            df = df.copy()
            eps = 1e-12
            df["entropy"] = -(p0 * np.log(p0 + eps) + p1 * np.log(p1 + eps))

    # Helper to plot a histogram if column exists
    def hist_if_exists(col, bins=30, xlabel=None):
        if col in df.columns:
            plt.figure()
            plt.hist(df[col].dropna().astype(float), bins=bins)
            plt.xlabel(xlabel or col)
            plt.ylabel("Count")
            plt.title(f"{name}: {col} distribution (unlabeled hold-out)")
            plt.show()

    # 1) Histograms
    hist_if_exists("p_1", xlabel="Predicted probability p(class=1)")
    hist_if_exists("confidence", xlabel="Confidence = max(p_0, p_1)")
    hist_if_exists("entropy", xlabel="Entropy (higher = more uncertain)")
    hist_if_exists("margin", xlabel="Margin = |p_1 - p_0| (lower = more uncertain)")

    # 2) Scatter plots that are often informative
    if ("confidence" in df.columns) and ("entropy" in df.columns):
        plt.figure()
        plt.scatter(df["confidence"].astype(float), df["entropy"].astype(float), s=10)
        plt.xlabel("Confidence")
        plt.ylabel("Entropy")
        plt.title(f"{name}: confidence vs entropy")
        plt.show()

    if ("confidence" in df.columns) and ("margin" in df.columns):
        plt.figure()
        plt.scatter(df["confidence"].astype(float), df["margin"].astype(float), s=10)
        plt.xlabel("Confidence")
        plt.ylabel("Margin")
        plt.title(f"{name}: confidence vs margin")
        plt.show()

    # 3) Print “most uncertain” cases (lowest confidence / lowest margin / highest entropy)
    # (useful for manual review)
    print(f"\n=== {name}: most uncertain examples ===")

    if "confidence" in df.columns:
        print("\nLowest confidence:")
        display(df.sort_values("confidence", ascending=True).head(10))

    if "margin" in df.columns:
        print("\nLowest margin:")
        display(df.sort_values("margin", ascending=True).head(10))

    if "entropy" in df.columns:
        print("\nHighest entropy:")
        display(df.sort_values("entropy", ascending=False).head(10))


@_with_notebook_context
def multiclust_extended_add_metadata_and_clusters_validation_individual_labels(dict_final, data_full, mod_num):
    """
    Merge cluster labels into full metadata DataFrame using src_subject_id
    from the actual training data stored in fold_metrics['data'].
    This ensures correct alignment even when not all train_ids were clustered.
    """
    clusters = pd.Series(labels_test_modalities[mod_num])

    # Get the first modality’s dataframe — all have the same subject order
    modality_dfs = dict_final
    if not modality_dfs:
        raise ValueError("dict_final is empty; cannot extract subject IDs.")

    # Use the src_subject_id column from the first available modality
    first_modality = list(modality_dfs.keys())[0]
    subj_ids = modality_dfs[first_modality]['src_subject_id'].reset_index(drop=True)

    # Sanity check: should match cluster array length
    if len(subj_ids) != len(clusters):
        print(f"⚠️ Mismatch: {len(subj_ids)} subject IDs vs {len(clusters)} cluster labels.")
        min_len = min(len(subj_ids), len(clusters))
        subj_ids = subj_ids.iloc[:min_len]
        clusters = clusters.iloc[:min_len]
        print(f"Trimmed both to {min_len} entries to align.")

    # Build cluster mapping dataframe
    cluster_df = pd.DataFrame({
        'src_subject_id': subj_ids,
        'Cluster': clusters
    })

    # Merge back into full metadata
    merged = pd.merge(data_full, cluster_df, on='src_subject_id', how='left')

    print(f"✅ Merged clusters for {merged['Cluster'].notna().sum()} subjects (out of {len(merged)} total).")
    return merged


@_with_notebook_context
def multiclust_extended_chi_square_comparison_validation_individual_labels(df, group_col, label_col, title_prefix):
    """
    Perform chi-square test and plot grouped bar chart.
    """
    # Drop missing values
    df = df.dropna(subset=[group_col, label_col])

    # Make a copy for plotting / stats so we can safely relabel
    df_plot = df.copy()

    # Rename 'HC' -> 'CC' only for phenotype (for plotting and stats)
    if label_col == 'phenotype':
        df_plot[label_col] = df_plot[label_col].replace({'HC': 'CC'})

    # 1. Summarize counts
    summary_df = (
        df_plot.groupby([group_col, label_col])
        .size()
        .reset_index(name='n')
    )

    # 2. Chi-square test
    tbl = pd.crosstab(df_plot[group_col], df_plot[label_col])
    chi2, pval, dof, expected = chi2_contingency(tbl)
    pval = round(pval, 3)

    # 3. Grouped bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=summary_df,
        x=group_col,
        y='n',
        hue=label_col,
        dodge=True
    )
    plt.title(f"{title_prefix}\n(Chi-square p = {pval})")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    sns.despine()
    plt.tight_layout()
    plt.show()


@_with_notebook_context
def multiclust_extended_parse_stream(stream_str):
    """
    Parse stream like:
      "Psychoticism=low -> Detachment=high -> ... -> final=1"
    into ordered list of (domain, label).
    """
    if pd.isna(stream_str):
        return []
    parts = [p.strip() for p in ARROW_PAT.split(str(stream_str).strip()) if p.strip()]
    out = []
    for p in parts:
        if "=" in p:
            dom, lab = p.split("=", 1)
            out.append((dom.strip(), lab.strip()))
        else:
            out.append((p.strip(), "<NA>"))
    return out


@_with_notebook_context
def multiclust_extended_infer_stage_order(df, stream_col="stream"):
    """
    Infer stage order from the first non-null stream.
    Assumes all streams follow the same domain order.
    """
    s = df[stream_col].dropna().astype(str)
    if s.empty:
        raise ValueError("No streams found to infer stage order.")
    path = multiclust_extended_parse_stream(s.iloc[0])
    return [d for d, _ in path]


@_with_notebook_context
def multiclust_extended_normalize(v, eps=1e-12):
    """Process multiclust extended normalize."""
    v = np.asarray(v, dtype=float)
    s = v.sum()
    if s <= 0:
        return np.ones_like(v) / max(1, len(v))
    return v / (s + eps)


@_with_notebook_context
def multiclust_extended_build_prefix_next(df, stream_col="stream", n_col="n"):
    """
    Returns:
      prefix_next: dict[prefix_tuple_of_(domain,label)] -> dict[next_token_(domain,label)|<END>] -> weight
      prefix_mass: dict[prefix] -> total weight passing through prefix
    Where prefix is a tuple of (domain,label) tokens, e.g.:
      prefix = (("Psychoticism","low"), ("Detachment","high"))
    and next token is the next (domain,label) or "<END>".
    """
    prefix_next = {}
    prefix_mass = {}

    for _, row in df.iterrows():
        w = float(row.get(n_col, 1.0))
        path = multiclust_extended_parse_stream(row[stream_col])
        if not path:
            continue

        # For each prefix (including empty prefix), record what comes next
        for i in range(len(path)):
            prefix = tuple(path[:i])  # empty prefix for i=0
            nxt = path[i]             # next token
            prefix_mass[prefix] = prefix_mass.get(prefix, 0.0) + w
            d = prefix_next.setdefault(prefix, {})
            d[nxt] = d.get(nxt, 0.0) + w

        # terminal transition from full path to END
        full_prefix = tuple(path)
        prefix_mass[full_prefix] = prefix_mass.get(full_prefix, 0.0) + w
        d = prefix_next.setdefault(full_prefix, {})
        d["<END>"] = d.get("<END>", 0.0) + w

    return prefix_next, prefix_mass


@_with_notebook_context
def multiclust_extended_compare_prefix_structure(df_disc, df_test, stream_col="stream", n_col="n", eps=1e-12):
    """
    For each prefix, compare the conditional distribution over next tokens:
        P_disc(next | prefix)  vs  P_test(next | prefix)

    Returns DataFrame with:
      - prefix_str
      - depth
      - js_next (Jensen–Shannon distance on next-step distributions)
      - mass_disc / mass_test (how much data passes through prefix; useful for weighting but not "size-equality")
      - top_next_disc / top_next_test (most likely next step)
      - support_next_overlap (Jaccard on next-token supports)
    """
    pn_d, pm_d = multiclust_extended_build_prefix_next(df_disc, stream_col, n_col)
    pn_t, pm_t = multiclust_extended_build_prefix_next(df_test, stream_col, n_col)

    prefixes = set(pn_d.keys()) | set(pn_t.keys())

    rows = []
    for pref in prefixes:
        nd = pn_d.get(pref, {})
        nt = pn_t.get(pref, {})

        keys = set(nd.keys()) | set(nt.keys())
        # Align next-token vectors
        vd = np.array([nd.get(k, 0.0) for k in keys], dtype=float)
        vt = np.array([nt.get(k, 0.0) for k in keys], dtype=float)

        pd_ = multiclust_extended_normalize(vd, eps=eps)
        pt_ = multiclust_extended_normalize(vt, eps=eps)

        js = float(jensenshannon(pd_ + eps, pt_ + eps, base=2.0))

        # Next-token support overlap (presence/absence, structure)
        supp_d = {k for k, v in nd.items() if v > 0}
        supp_t = {k for k, v in nt.items() if v > 0}
        supp_j = len(supp_d & supp_t) / max(1, len(supp_d | supp_t))

        # Most likely next token in each
        top_d = max(nd.items(), key=lambda x: x[1])[0] if nd else None
        top_t = max(nt.items(), key=lambda x: x[1])[0] if nt else None

        def tok_str(tok):
            if tok == "<END>":
                return "<END>"
            if tok is None:
                return "<NONE>"
            return f"{tok[0]}={tok[1]}"

        prefix_str = " → ".join([f"{d}={l}" for d, l in pref]) if pref else "<START>"
        rows.append({
            "prefix_str": prefix_str,
            "depth": len(pref),
            "js_next": js,
            "mass_disc": pm_d.get(pref, 0.0),
            "mass_test": pm_t.get(pref, 0.0),
            "top_next_disc": tok_str(top_d),
            "top_next_test": tok_str(top_t),
            "support_next_jaccard": supp_j,
            "prefix_exists_in_disc": pref in pn_d,
            "prefix_exists_in_test": pref in pn_t,
        })

    out = pd.DataFrame(rows)
    # A useful default sorting: prioritize structurally-different AND commonly-used prefixes
    out["mass_min"] = np.minimum(out["mass_disc"], out["mass_test"])
    out = out.sort_values(["mass_min", "js_next"], ascending=[False, False]).reset_index(drop=True)
    return out


@_with_notebook_context
def multiclust_extended_plot_top_prefix_differences(prefix_report, top_n=20, min_depth=1):
    """
    Barh plot of top prefixes by JS(next) after filtering.
    """
    d = prefix_report[prefix_report["depth"] >= min_depth].copy()
    d = d.sort_values("js_next", ascending=False).head(top_n)

    plt.figure(figsize=(11, max(4, 0.35 * len(d))))
    y = np.arange(len(d))[::-1]
    plt.barh(y, d["js_next"].iloc[::-1].to_numpy())
    plt.yticks(y, d["prefix_str"].iloc[::-1].to_list(), fontsize=9)
    plt.xlabel("Jensen–Shannon distance of P(next | prefix)")
    plt.title("Most structurally different prefixes (next-step rule)")
    plt.tight_layout()
    plt.show()


@_with_notebook_context
def multiclust_extended_final_mapping_table(df, stream_col="stream", n_col="n", final_domain="final"):
    """
    Builds a table over modality-prefixes (everything up to but excluding final)
    with P(final=label | prefix) computed within each dataset.

    Returns DataFrame:
      prefix_str, final_label, weight
    and also a pivoted table of P(final=...) by prefix.
    """
    rows = []
    for _, row in df.iterrows():
        w = float(row.get(n_col, 1.0))
        path = multiclust_extended_parse_stream(row[stream_col])
        if not path:
            continue

        # split into prefix (before final) and final token
        final_tokens = [t for t in path if t[0] == final_domain]
        if not final_tokens:
            # If final isn't explicitly present, skip
            continue
        final_tok = final_tokens[-1]  # in case of duplicates, take last
        final_label = final_tok[1]

        # prefix = all tokens before the final token position (first occurrence)
        # more robust: take all tokens except the final-domain token(s)
        prefix = tuple([t for t in path if t[0] != final_domain])

        prefix_str = " → ".join([f"{d}={l}" for d, l in prefix]) if prefix else "<NO_MODALITIES>"
        rows.append({"prefix_str": prefix_str, "final_label": final_label, "w": w})

    long = pd.DataFrame(rows)
    if long.empty:
        return long, pd.DataFrame()

    # Compute conditional probabilities P(final_label | prefix)
    grp = long.groupby(["prefix_str", "final_label"], dropna=False)["w"].sum().reset_index()
    totals = grp.groupby("prefix_str")["w"].sum().reset_index().rename(columns={"w": "w_total"})
    grp = grp.merge(totals, on="prefix_str", how="left")
    grp["p_final_given_prefix"] = grp["w"] / grp["w_total"]

    pivot = grp.pivot_table(index="prefix_str", columns="final_label", values="p_final_given_prefix", fill_value=0.0)
    return grp, pivot


@_with_notebook_context
def multiclust_extended_compare_final_mapping(df_disc, df_test, stream_col="stream", n_col="n", final_domain="final"):
    """
    Compare P(final | modalities) between discovery and test.
    Returns a table with per-prefix deltas per final label plus summary metrics.
    """
    long_d, piv_d = multiclust_extended_final_mapping_table(df_disc, stream_col, n_col, final_domain)
    long_t, piv_t = multiclust_extended_final_mapping_table(df_test, stream_col, n_col, final_domain)

    if piv_d.empty or piv_t.empty:
        return pd.DataFrame(), {"note": "No final mapping found (missing final tokens?)"}

    # align
    idx = sorted(set(piv_d.index) | set(piv_t.index))
    cols = sorted(set(piv_d.columns) | set(piv_t.columns))
    A = piv_d.reindex(index=idx, columns=cols).fillna(0.0)
    B = piv_t.reindex(index=idx, columns=cols).fillna(0.0)

    # Delta per final label
    delta = (B - A)
    out = delta.copy()
    out.columns = [f"delta_final={c}" for c in out.columns]
    out.insert(0, "prefix_str", out.index)

    # A per-prefix summary: JS distance between final distributions for that prefix
    js_list = []
    for i in range(A.shape[0]):
        p = multiclust_extended_normalize(A.iloc[i].to_numpy())
        q = multiclust_extended_normalize(B.iloc[i].to_numpy())
        js_list.append(float(jensenshannon(p + 1e-12, q + 1e-12, base=2.0)))
    out["js_final_given_prefix"] = js_list

    # Add weights: how common the prefix is (within each dataset)
    wD = long_d.groupby("prefix_str")["w"].sum() if not long_d.empty else pd.Series(dtype=float)
    wT = long_t.groupby("prefix_str")["w"].sum() if not long_t.empty else pd.Series(dtype=float)
    out["w_disc"] = out["prefix_str"].map(wD).fillna(0.0)
    out["w_test"] = out["prefix_str"].map(wT).fillna(0.0)
    out["w_min"] = np.minimum(out["w_disc"], out["w_test"])

    # Sort by (common prefixes) then by biggest JS shift in final mapping
    out = out.sort_values(["w_min", "js_final_given_prefix"], ascending=[False, False]).reset_index(drop=True)

    # Global summary metric: weighted average JS over prefixes (weights = w_min)
    w = out["w_min"].to_numpy()
    if w.sum() > 0:
        global_js = float(np.sum(out["js_final_given_prefix"].to_numpy() * w) / w.sum())
    else:
        global_js = float(out["js_final_given_prefix"].mean())

    metrics = {
        "num_prefixes_union": len(out),
        "weighted_js_final_given_prefix": global_js,
        "final_labels": cols,
    }
    return out, metrics


@_with_notebook_context
def multiclust_extended_plot_top_final_mapping_shifts(final_cmp, top_n=20):
    """Process multiclust extended plot top final mapping shifts."""
    d = final_cmp.head(top_n).copy()
    plt.figure(figsize=(11, max(4, 0.35 * len(d))))
    y = np.arange(len(d))[::-1]
    plt.barh(y, d["js_final_given_prefix"].iloc[::-1].to_numpy())
    plt.yticks(y, d["prefix_str"].iloc[::-1].to_list(), fontsize=9)
    plt.xlabel("JS distance between P(final | prefix) in test vs discovery")
    plt.title("Prefixes with biggest changes in final mapping")
    plt.tight_layout()
    plt.show()


@_with_notebook_context
def multiclust_extended_stream_presence_and_topk(df_disc, df_test, stream_col="stream", n_col="n", topk=30):
    """Process multiclust extended stream presence and topk."""
    A = set(df_disc[stream_col].dropna().astype(str))
    B = set(df_test[stream_col].dropna().astype(str))

    presence = {
        "unique_streams_disc": len(A),
        "unique_streams_test": len(B),
        "stream_jaccard_presence": len(A & B) / max(1, len(A | B)),
        "coverage_disc_in_test": len(A & B) / max(1, len(A)),
        "coverage_test_in_disc": len(A & B) / max(1, len(B)),
    }

    pD = df_disc.groupby(stream_col)[n_col].sum().sort_values(ascending=False)
    pT = df_test.groupby(stream_col)[n_col].sum().sort_values(ascending=False)
    pD = pD / pD.sum()
    pT = pT / pT.sum()

    topD = set(pD.index[: min(topk, len(pD))].astype(str))
    topT = set(pT.index[: min(topk, len(pT))].astype(str))

    presence[f"top{topk}_jaccard_by_rank"] = len(topD & topT) / max(1, len(topD | topT))
    presence["top_disc_only"] = sorted(list(topD - topT))[:10]
    presence["top_test_only"] = sorted(list(topT - topD))[:10]
    return presence, pD, pT


@_with_notebook_context
def multiclust_extended_sankey_from_streams(df, stream_col="stream", n_col="n", max_edges=200):
    """
    Build a Sankey graph from full streams.
    Nodes are stage-specific label tokens: f"{domain}={label}".
    Edges connect consecutive tokens. We prune to max_edges by weight.
    """
    if not _HAS_PLOTLY:
        raise RuntimeError("Plotly not installed; cannot draw sankey.")

    edge_w = {}
    for _, row in df.iterrows():
        w = float(row.get(n_col, 1.0))
        path = multiclust_extended_parse_stream(row[stream_col])
        toks = [f"{d}={l}" for d, l in path]
        for a, b in zip(toks[:-1], toks[1:]):
            edge_w[(a, b)] = edge_w.get((a, b), 0.0) + w

    # prune
    edges = sorted(edge_w.items(), key=lambda x: -x[1])[:max_edges]
    nodes = {}
    def nid(x):
        if x not in nodes:
            nodes[x] = len(nodes)
        return nodes[x]

    src, tgt, val = [], [], []
    for (a, b), w in edges:
        src.append(nid(a))
        tgt.append(nid(b))
        val.append(w)

    labels = [None] * len(nodes)
    for k, i in nodes.items():
        labels[i] = k

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=12, thickness=12),
        link=dict(source=src, target=tgt, value=val),
    )])
    return fig


@_with_notebook_context
def multiclust_extended_full_structure_report(stream_summary, stream_summary_test, stream_col="stream", n_col="n", topk=30, final_domain="final"):
    # Presence + top-k overlap
    """Process multiclust extended full structure report."""
    presence, pD, pT = multiclust_extended_stream_presence_and_topk(stream_summary, stream_summary_test, stream_col, n_col, topk=topk)

    # Prefix-tree structural differences
    prefix_report = multiclust_extended_compare_prefix_structure(stream_summary, stream_summary_test, stream_col, n_col)

    # Full mapping to final
    final_cmp, final_metrics = multiclust_extended_compare_final_mapping(stream_summary, stream_summary_test, stream_col, n_col, final_domain)

    return {
        "presence_metrics": presence,
        "p_stream_disc": pD,
        "p_stream_test": pT,
        "prefix_report": prefix_report,
        "final_mapping_compare": final_cmp,
        "final_mapping_metrics": final_metrics,
    }


@_with_notebook_context
def multiclust_extended_all_streams_table(stream_summary, stream_summary_test, stream_col="stream", n_col="n"):
    # Aggregate in case there are duplicate stream rows
    """Process multiclust extended all streams table."""
    disc = (stream_summary
            .groupby(stream_col, dropna=False)[n_col]
            .sum()
            .rename("n_disc")
            .to_frame())

    test = (stream_summary_test
            .groupby(stream_col, dropna=False)[n_col]
            .sum()
            .rename("n_test")
            .to_frame())

    # Outer join gives union of streams
    tbl = disc.join(test, how="outer").fillna(0)

    # Add proportions within each dataset
    N_disc = tbl["n_disc"].sum()
    N_test = tbl["n_test"].sum()

    tbl["p_disc"] = tbl["n_disc"] / N_disc if N_disc > 0 else np.nan
    tbl["p_test"] = tbl["n_test"] / N_test if N_test > 0 else np.nan

    # Helpful comparisons
    tbl["delta_p"] = tbl["p_test"] - tbl["p_disc"]
    tbl["abs_delta_p"] = tbl["delta_p"].abs()
    tbl["log2_fc"] = np.log2((tbl["p_test"] + 1e-12) / (tbl["p_disc"] + 1e-12))

    # Make stream a real column, sort by biggest shift
    tbl = tbl.reset_index().rename(columns={stream_col: "stream"})
    tbl = tbl.sort_values("abs_delta_p", ascending=False).reset_index(drop=True)

    return tbl


# -----------------------------------------------------------------------------
# prospect analysis helpers
# Implementations are preserved from the original notebook cells.
# -----------------------------------------------------------------------------

@_with_notebook_context
def prospect_print_remaining_after_full_missing_modality_removal(df, df_name, meta, modalities, subject_id_column="src_subject_id"):
    """Process prospect print remaining after full missing modality removal."""
    modality_to_cols = {
        modality: [
            col for col in meta.loc[meta["Modality"] == modality, "ElementName"]
            if col in df.columns
        ]
        for modality in modalities
    }

    full_missing_mask = pd.Series(False, index=df.index)
    for cols in modality_to_cols.values():
        if cols:
            full_missing_mask |= df[cols].isna().all(axis=1)

    print(
        f"{df_name}: {(~full_missing_mask).sum()} participants remain "
        f"after removing participants with >=1 fully missing modality "
        f"({full_missing_mask.sum()} removed)."
    )


@_with_notebook_context
def prospect_truthy_profile_value(value):
    """Process prospect truthy profile value."""
    return str(value).strip().strip('"\'').upper() in {"TRUE", "1", "YES", "Y"}


@_with_notebook_context
def prospect_parse_profile_exports(profile_path):
    """Process prospect parse profile exports."""
    exports = {}
    if profile_path is None or not profile_path.exists():
        return exports
    for line in profile_path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[len("export "):].split("=", 1)
        value = value.strip()
        # Keep literal defaults such as ${VAR:-0} as-is; simple quoted values are enough for the flag check.
        exports[key.strip()] = value.strip('"\'')
    return exports


@_with_notebook_context
def prospect_infer_notebook_profile():
    """Process prospect infer notebook profile."""
    notebook_dir = Path.cwd().name
    profile_candidates = []
    if notebook_dir == "clinical_paper":
        profile_candidates.append("clinical_paper")
    elif notebook_dir == "multiclust_extended":
        profile_candidates.append("multiclust_extended")
    elif notebook_dir == "prospect":
        profile_candidates.append("prospect")
    elif notebook_dir == "schizbull_legacy":
        profile_candidates.append("schizbull_legacy")
    profile_candidates.append(os.environ.get("RUN_PROFILE", ""))
    for candidate in profile_candidates:
        if candidate:
            return candidate
    return None


@_with_notebook_context
def prospect_find_repo_root(start):
    """Process prospect find repo root."""
    start = Path(start).resolve()
    for parent in [start] + list(start.parents):
        if (parent / "run_profiles").is_dir() and (parent / "full_pipeline.py").exists():
            return parent
    return None


@_with_notebook_context
def prospect_profile_enabled_for_sensitivity(repo_root, profile_name):
    """Process prospect profile enabled for sensitivity."""
    if not profile_name or repo_root is None:
        return None, None
    profile_path = repo_root / "run_profiles" / f"{profile_name}.sh"
    exports = prospect_parse_profile_exports(profile_path)
    value = exports.get("DO_CLUSTER_VALIDATION_SENSITIVITY", "FALSE")
    return prospect_truthy_profile_value(value), profile_path


@_with_notebook_context
def prospect_display_if_available(obj):
    """Process prospect display if available."""
    if "display" in globals():
        display(obj)
    else:
        print(obj)


@_with_notebook_context
def prospect_get_nested(dct, path, default=np.nan):
    """Process prospect get nested."""
    cur = dct
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


@_with_notebook_context
def prospect_flatten_sensitivity_results(results):
    """Process prospect flatten sensitivity results."""
    rows = []
    for solution, payload in results.get("solutions", {}).items():
        rows.append({
            "solution": solution,
            "kind": payload.get("kind"),
            "observed_k": prospect_get_nested(payload, ["observed_quality", "k"]),
            "observed_n": prospect_get_nested(payload, ["observed_quality", "n"]),
            "observed_n_features": prospect_get_nested(payload, ["observed_quality", "n_features"]),
            "observed_composite": prospect_get_nested(payload, ["observed_quality", "composite"]),
            "observed_silhouette": prospect_get_nested(payload, ["observed_quality", "silhouette"]),
            "observed_calinski_harabasz": prospect_get_nested(payload, ["observed_quality", "calinski_harabasz"]),
            "observed_davies_bouldin": prospect_get_nested(payload, ["observed_quality", "davies_bouldin"]),
            "k1_composite": prospect_get_nested(payload, ["uni_cluster_baseline", "quality", "composite"]),
            "pc1_variance_explained": prospect_get_nested(payload, ["pc1_median_split", "pc1_variance_explained"]),
            "pc1_median_composite": prospect_get_nested(payload, ["pc1_median_split", "quality", "composite"]),
            "pc1_median_ari_with_observed": prospect_get_nested(payload, ["pc1_median_split", "ari_with_observed_labels"]),
            "pc1_median_stability_ari": prospect_get_nested(payload, ["pc1_median_split", "bootstrap_stability", "mean_ari"]),
            "pc1_median_stability_sd_ari": prospect_get_nested(payload, ["pc1_median_split", "bootstrap_stability", "sd_ari"]),
            "dip_available": prospect_get_nested(payload, ["dip_test_pc1", "available"], default=False),
            "dip_statistic_pc1": prospect_get_nested(payload, ["dip_test_pc1", "dip"]),
            "dip_p_value_pc1": prospect_get_nested(payload, ["dip_test_pc1", "p_value"]),
            "gap_selected_k_tibshirani": prospect_get_nested(payload, ["gap_statistic", "selected_k_tibshirani_rule"]),
            "gap_selected_k_max_gap": prospect_get_nested(payload, ["gap_statistic", "selected_k_max_gap"]),
            "sigclust_cluster_index": prospect_get_nested(payload, ["sigclust_approx", "observed_cluster_index"]),
            "sigclust_p_value": prospect_get_nested(payload, ["sigclust_approx", "p_value"]),
            "sigclust_null_mean_cluster_index": prospect_get_nested(payload, ["sigclust_approx", "null_mean_cluster_index"]),
            "gaussian_null_p_quality": prospect_get_nested(payload, ["covariance_matched_gaussian_null", "p_quality_ge_observed_recluster"]),
            "gaussian_null_p_stability": prospect_get_nested(payload, ["covariance_matched_gaussian_null", "p_stability_ge_observed_recluster"]),
            "observed_recluster_stability_ari": prospect_get_nested(payload, ["covariance_matched_gaussian_null", "observed_recluster_stability", "mean_ari"]),
            "gaussian_null_mean_stability_ari": prospect_get_nested(payload, ["covariance_matched_gaussian_null", "null_stability_mean_ari"]),
            "gaussian_null_mean_quality": prospect_get_nested(payload, ["covariance_matched_gaussian_null", "null_quality_mean"]),
        })
    return pd.DataFrame(rows)


@_with_notebook_context
def prospect_categorical_like(series, max_discrete_levels=6):
    """Process prospect categorical like."""
    observed = series.dropna()
    return (not pd.api.types.is_numeric_dtype(series)) or observed.nunique() <= max_discrete_levels


@_with_notebook_context
def prospect_eta_squared_by_category(values, categories):
    """Process prospect eta squared by category."""
    frame = pd.DataFrame({"factor": pd.to_numeric(values, errors="coerce"), "category": categories.astype("string").fillna("<missing>")}).dropna(subset=["factor"])
    if frame.empty or frame["category"].nunique() < 2:
        return np.nan
    grand_mean = frame["factor"].mean()
    total_ss = ((frame["factor"] - grand_mean) ** 2).sum()
    if not np.isfinite(total_ss) or total_ss <= 0:
        return np.nan
    grouped = frame.groupby("category", observed=False)["factor"]
    between_ss = sum(len(group) * (group.mean() - grand_mean) ** 2 for _, group in grouped)
    return float(between_ss / total_ss)


@_with_notebook_context
def prospect_safe_spearman(values, factor_scores):
    """Process prospect safe spearman."""
    numeric = pd.to_numeric(values, errors="coerce")
    aligned = pd.DataFrame({"value": numeric, "factor": factor_scores}).dropna()
    if aligned["value"].nunique() < 2 or aligned["factor"].nunique() < 2:
        return np.nan
    return float(aligned["value"].corr(aligned["factor"], method="spearman"))


@_with_notebook_context
def prospect_infer_cluster_association_feature_type(series, max_discrete_levels=6):
    """Process prospect infer cluster association feature type."""
    observed = series.dropna()
    if observed.empty:
        return "empty"
    if pd.api.types.is_numeric_dtype(series) and observed.nunique() > max_discrete_levels:
        return "continuous"
    return "categorical"


@_with_notebook_context
def prospect_eta_squared_for_feature(values, clusters):
    """Process prospect eta squared for feature."""
    frame = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "cluster": pd.Series(clusters).astype(str)}).dropna()
    if frame.empty or frame["cluster"].nunique() < 2 or frame["value"].nunique() < 2:
        return np.nan
    grand_mean = frame["value"].mean()
    total_ss = ((frame["value"] - grand_mean) ** 2).sum()
    if not np.isfinite(total_ss) or total_ss <= 0:
        return np.nan
    between_ss = sum(len(g) * (g["value"].mean() - grand_mean) ** 2 for _, g in frame.groupby("cluster", observed=False))
    return float(between_ss / total_ss)


@_with_notebook_context
def prospect_categorical_cluster_score(values, clusters):
    """Process prospect categorical cluster score."""
    frame = pd.DataFrame({"value": pd.Series(values).astype("string").fillna("<missing>"), "cluster": pd.Series(clusters).astype(str)}).dropna()
    if frame.empty or frame["cluster"].nunique() < 2 or frame["value"].nunique() < 2:
        return np.nan
    table = pd.crosstab(frame["cluster"], frame["value"])
    try:
        chi2, _, _, _ = chi2_contingency(table)
    except Exception:
        return np.nan
    n = table.to_numpy().sum()
    denom = n * max(1, min(table.shape) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan


@_with_notebook_context
def prospect_add_metadata_and_clusters_individual_labels(final_metrics, data_full, mod_num):
    """
    Merge cluster labels into full metadata DataFrame using src_subject_id
    from the actual training data stored in fold_metrics['data'].
    This ensures correct alignment even when not all train_ids were clustered.
    """
    clusters = pd.Series(final_metrics['individual_labels'][mod_num])

    # Get the first modality’s dataframe — all have the same subject order
    modality_dfs = final_metrics.get('data', {})
    if not modality_dfs:
        raise ValueError("final_metrics['data'] is empty; cannot extract subject IDs.")

    # Use the src_subject_id column from the first available modality
    first_modality = list(modality_dfs.keys())[0]
    subj_ids = modality_dfs[first_modality]['src_subject_id'].reset_index(drop=True)

    # Sanity check: should match cluster array length
    if len(subj_ids) != len(clusters):
        print(f"⚠️ Mismatch: {len(subj_ids)} subject IDs vs {len(clusters)} cluster labels.")
        min_len = min(len(subj_ids), len(clusters))
        subj_ids = subj_ids.iloc[:min_len]
        clusters = clusters.iloc[:min_len]
        print(f"Trimmed both to {min_len} entries to align.")

    # Build cluster mapping dataframe
    cluster_df = pd.DataFrame({
        'src_subject_id': subj_ids,
        'Cluster': clusters
    })

    # Merge back into full metadata
    merged = pd.merge(data_full, cluster_df, on='src_subject_id', how='left')

    print(f"✅ Merged clusters for {merged['Cluster'].notna().sum()} subjects (out of {len(merged)} total).")
    return merged


@_with_notebook_context
def prospect_chi_square_comparison_individual_labels(df, group_col, label_col, title_prefix):
    """
    Perform chi-square test and plot grouped bar chart.
    """
    # Drop missing values
    df = df.dropna(subset=[group_col, label_col])

    # Make a copy for plotting / stats so we can safely relabel
    df_plot = df.copy()

    # Rename 'HC' -> 'CC' only for phenotype (for plotting and stats)
    if label_col == 'phenotype':
        df_plot[label_col] = df_plot[label_col].replace({'HC': 'CC'})

    # 1. Summarize counts
    summary_df = (
        df_plot.groupby([group_col, label_col])
        .size()
        .reset_index(name='n')
    )

    # 2. Chi-square test
    tbl = pd.crosstab(df_plot[group_col], df_plot[label_col])
    chi2, pval, dof, expected = chi2_contingency(tbl)
    pval = round(pval, 3)

    # 3. Grouped bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=summary_df,
        x=group_col,
        y='n',
        hue=label_col,
        dodge=True
    )
    plt.title(f"{title_prefix}\n(Chi-square p = {pval})")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    sns.despine()
    plt.tight_layout()
    plt.show()


@_with_notebook_context
def prospect_add_metadata_and_clusters_final_labels(final_metrics, data_full):
    """
    Merge cluster labels into full metadata DataFrame using src_subject_id
    from the actual training data stored in fold_metrics['data'].
    This ensures correct alignment even when not all train_ids were clustered.
    """
    clusters = pd.Series(final_metrics['final_labels'])

    # Get the first modality’s dataframe — all have the same subject order
    modality_dfs = final_metrics.get('data', {})
    if not modality_dfs:
        raise ValueError("final_metrics['data'] is empty; cannot extract subject IDs.")

    # Use the src_subject_id column from the first available modality
    first_modality = list(modality_dfs.keys())[0]
    subj_ids = modality_dfs[first_modality]['src_subject_id'].reset_index(drop=True)

    # Sanity check: should match cluster array length
    if len(subj_ids) != len(clusters):
        print(f"⚠️ Mismatch: {len(subj_ids)} subject IDs vs {len(clusters)} cluster labels.")
        min_len = min(len(subj_ids), len(clusters))
        subj_ids = subj_ids.iloc[:min_len]
        clusters = clusters.iloc[:min_len]
        print(f"Trimmed both to {min_len} entries to align.")

    # Build cluster mapping dataframe
    cluster_df = pd.DataFrame({
        'src_subject_id': subj_ids,
        'Cluster': clusters
    })

    # Merge back into full metadata
    merged = pd.merge(data_full, cluster_df, on='src_subject_id', how='left')

    print(f"✅ Merged clusters for {merged['Cluster'].notna().sum()} subjects (out of {len(merged)} total).")
    return merged


@_with_notebook_context
def prospect_chi_square_comparison_final_labels(df, group_col, label_col, title_prefix, save_path):
    """
    Perform chi-square test and plot grouped bar chart.
    """
    # Drop missing values
    df = df.dropna(subset=[group_col, label_col])

    # Make a copy for plotting / stats so we can safely relabel
    df_plot = df.copy()

    # Rename 'HC' -> 'CC' only for phenotype (for plotting and stats)
    if label_col == 'phenotype':
        df_plot[label_col] = df_plot[label_col].replace({'HC': 'CC'})

    # 1. Summarize counts
    summary_df = (
        df_plot.groupby([group_col, label_col])
        .size()
        .reset_index(name='n')
    )

    # 2. Chi-square test
    tbl = pd.crosstab(df_plot[group_col], df_plot[label_col])
    chi2, pval, dof, expected = chi2_contingency(tbl)
    pval = round(pval, 3)

    # 3. Grouped bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=summary_df,
        x=group_col,
        y='n',
        hue=label_col,
        dodge=True
    )
    plt.title(f"{title_prefix}\n(Chi-square p = {pval})")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)
    plt.show()

    # 4. Cluster summary (including CHR percentage if relevant)
    cluster_summary = (
        df_plot.groupby(group_col)
        .agg(
            Size=(group_col, 'size'),
            CHR_percentage=(
                label_col,
                lambda x: (x.eq('CHR').sum() / len(x)) * 100
                if 'CHR' in x.values else None
            )
        )
        .reset_index()
        .sort_values(group_col)
    )
    print(f"\nCluster summary for {title_prefix}")
    print(cluster_summary)
    print("\n" + "-"*60)


@_with_notebook_context
def prospect_alluvial_sankey_force_high_top(
    labels_by_modality: dict,
    final_labels,
    stage_order: list,
    final_name="final",
    high_token="high_severity",
    low_token="low_severity",
    final_order="auto",
    arrangement="snap",
    high_y=0.10,
    low_y=0.90,
    other_y=0.50,
    node_pad=22,
    node_thickness=18,
    width=1400,
    height=650,
    title="All modalities -> final (alluvial Sankey)",
):
    """Process prospect alluvial sankey force high top."""
    return alluvial_sankey_general(
        labels_by_modality=labels_by_modality,
        final_labels=final_labels,
        stage_order=stage_order,
        final_name=final_name,
        high_token=high_token,
        low_token=low_token,
        final_order=final_order,
        arrangement=arrangement,
        high_y=high_y,
        low_y=low_y,
        other_y=other_y,
        node_pad=node_pad,
        node_thickness=node_thickness,
        width=width,
        height=height,
        title=title,
    )


@_with_notebook_context
def prospect_summarize_streams(df_paths, stage_order, top_k=20, sample_ids=10):
    """Process prospect summarize streams."""
    group_cols = stage_order + ["final"]
    total = len(df_paths)

    g = (df_paths
         .groupby(group_cols, dropna=False)
         .agg(
             n=("src_subject_id", "size"),
             example_ids=("src_subject_id", lambda x: list(x.head(sample_ids))),
         )
         .reset_index()
        )

    g["pct"] = (g["n"] / total * 100).round(2)

    # A readable "stream label"
    def make_stream_label(row):
        parts = [f"{c}={row[c]}" for c in stage_order] + [f"final={row['final']}"]
        return " → ".join(parts)

    g["stream"] = g.apply(make_stream_label, axis=1)

    g = g.sort_values("n", ascending=False).head(top_k)

    # Reorder columns nicely
    cols = ["n", "pct", "stream", "example_ids"] + stage_order + ["final"]
    return g[cols]


@_with_notebook_context
def prospect_summarize_streams_clinical(df_paths, stage_order, top_k=20, sample_ids=10):
    """Process prospect summarize streams clinical."""
    group_cols = stage_order
    total = len(df_paths)

    g = (df_paths
         .groupby(group_cols, dropna=False)
         .agg(
             n=("src_subject_id", "size"),
             example_ids=("src_subject_id", lambda x: list(x.head(sample_ids))),
         )
         .reset_index()
        )

    g["pct"] = (g["n"] / total * 100).round(2)

    # A readable "stream label"
    def make_stream_label(row):
        parts = [f"{c}={row[c]}" for c in stage_order]
        return " → ".join(parts)

    g["stream"] = g.apply(make_stream_label, axis=1)

    g = g.sort_values("n", ascending=False).head(top_k)

    # Reorder columns nicely
    cols = ["n", "pct", "stream", "example_ids"] + stage_order
    return g[cols]


@_with_notebook_context
def prospect_finite_values(frame, col):
    """Process prospect finite values."""
    vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


@_with_notebook_context
def prospect_hist_if_finite(frame, col, bins=30, xlabel=None, title=None):
    """Process prospect hist if finite."""
    if col not in frame.columns:
        print(f"Skipping {col}: column is missing.")
        return
    vals = prospect_finite_values(frame, col)
    if vals.size == 0:
        print(f"Skipping {col}: no finite values to plot.")
        return
    plt.figure()
    plt.hist(vals, bins=bins)
    plt.xlabel(xlabel or col)
    plt.ylabel("Count")
    plt.title(title or f"{col} distribution (unlabeled hold-out)")
    plt.show()


@_with_notebook_context
def prospect_plot_pred_modality(df, name):
    """
    Plots confidence/uncertainty diagnostics for a single modality DataFrame.
    Expected columns (any subset is ok): p_*, confidence, entropy, margin.
    """
    if df is None:
        print(f"{name}: no predictions to plot.")
        return
    df = df.copy()

    prob_cols = [c for c in df.columns if str(c).startswith("p_")]
    if prob_cols:
        p = df[prob_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if p.shape[1] >= 2:
            p_sorted = np.sort(p, axis=1)
            df["confidence"] = np.nanmax(p, axis=1)
            df["margin"] = p_sorted[:, -1] - p_sorted[:, -2]
            p_clip = np.clip(p, 1e-12, 1.0)
            df["entropy"] = -np.nansum(p_clip * np.log(p_clip), axis=1)

    def finite_values(col):
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        return vals[np.isfinite(vals)]

    def hist_if_exists(col, bins=30, xlabel=None):
        if col not in df.columns:
            print(f"{name}: skipping {col}; column is missing.")
            return
        vals = finite_values(col)
        if vals.size == 0:
            print(f"{name}: skipping {col}; no finite values to plot.")
            return
        plt.figure()
        plt.hist(vals, bins=bins)
        plt.xlabel(xlabel or col)
        plt.ylabel("Count")
        plt.title(f"{name}: {col} distribution (unlabeled hold-out)")
        plt.show()

    # 1) Histograms
    if "p_1" in df.columns:
        hist_if_exists("p_1", xlabel="Predicted probability p(class=1)")
    elif prob_cols:
        hist_if_exists(prob_cols[0], xlabel=f"Predicted probability {prob_cols[0]}")
    hist_if_exists("confidence", xlabel="Confidence = max class probability")
    hist_if_exists("entropy", xlabel="Entropy (higher = more uncertain)")
    hist_if_exists("margin", xlabel="Probability margin: best - second-best (lower = more uncertain)")

    # 2) Scatter plots that are often informative
    if ("confidence" in df.columns) and ("entropy" in df.columns):
        x = pd.to_numeric(df["confidence"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df["entropy"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.any():
            plt.figure()
            plt.scatter(x[ok], y[ok], s=10)
            plt.xlabel("Confidence")
            plt.ylabel("Entropy")
            plt.title(f"{name}: confidence vs entropy")
            plt.show()
        else:
            print(f"{name}: skipping confidence vs entropy; no finite paired values.")

    if ("confidence" in df.columns) and ("margin" in df.columns):
        x = pd.to_numeric(df["confidence"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df["margin"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.any():
            plt.figure()
            plt.scatter(x[ok], y[ok], s=10)
            plt.xlabel("Confidence")
            plt.ylabel("Margin")
            plt.title(f"{name}: confidence vs margin")
            plt.show()
        else:
            print(f"{name}: skipping confidence vs margin; no finite paired values.")

    # 3) Print most uncertain cases.
    print(f"=== {name}: most uncertain examples ===")

    if "confidence" in df.columns:
        print("Lowest confidence:")
        display(df.sort_values("confidence", ascending=True).head(10))

    if "margin" in df.columns:
        print("Lowest margin:")
        display(df.sort_values("margin", ascending=True).head(10))

    if "entropy" in df.columns:
        print("Highest entropy:")
        display(df.sort_values("entropy", ascending=False).head(10))


@_with_notebook_context
def prospect_safe_name(value):
    """Process prospect safe name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unnamed"


@_with_notebook_context
def prospect_ordered_cluster_labels(labels):
    """Process prospect ordered cluster labels."""
    labels = pd.Series(labels).astype(str)
    return sorted(labels.unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))


@_with_notebook_context
def prospect_prepare_feature_table(df, clusters, subject_id_column="src_subject_id"):
    """Process prospect prepare feature table."""
    features = df.drop(columns=[subject_id_column], errors="ignore").reset_index(drop=True).copy()
    clusters = pd.Series(clusters).astype(str).reset_index(drop=True)
    if len(features) != len(clusters):
        raise ValueError(f"feature rows ({len(features)}) != labels ({len(clusters)}).")

    keep_cols = []
    for col in features.columns:
        s = features[col].replace([np.inf, -np.inf], np.nan)
        if s.dropna().nunique() > 1:
            keep_cols.append(col)
    return features[keep_cols], clusters


@_with_notebook_context
def prospect_infer_ranked_feature_type(series, max_categorical_levels=8):
    """Classify columns for plotting/testing. Binary and low-cardinality numeric columns are categorical."""
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan)
    observed = s.dropna()
    if observed.empty or observed.nunique() <= 1:
        return "constant"

    numeric = pd.to_numeric(s, errors="coerce")
    numeric_fraction = numeric.notna().mean()
    numeric_unique = numeric.dropna().nunique()

    if numeric_fraction >= 0.90 and numeric_unique > max_categorical_levels:
        return "continuous"
    return "categorical"


@_with_notebook_context
def prospect_continuous_score(series, clusters, cluster_order):
    """Process prospect continuous score."""
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    groups = [values[clusters == cl].dropna().to_numpy(dtype=float) for cl in cluster_order]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2 or sum(np.nanvar(g) > 0 for g in groups) == 0:
        return 0.0, np.nan, np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_value = f_oneway(*groups)

    if not np.isfinite(stat) or not np.isfinite(p_value):
        return 0.0, np.nan, np.nan
    score = -np.log10(max(float(p_value), np.finfo(float).tiny))
    return score, float(stat), float(p_value)


@_with_notebook_context
def prospect_categorical_values(series, max_levels=10):
    """Process prospect categorical values."""
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan)
    values = s.astype("object").where(s.notna(), "Missing").astype(str)
    counts = values.value_counts(dropna=False)
    if len(counts) > max_levels:
        top_levels = set(counts.head(max_levels - 1).index)
        values = values.where(values.isin(top_levels), "Other")
    return values


@_with_notebook_context
def prospect_categorical_score(series, clusters, cluster_order):
    """Process prospect categorical score."""
    values = prospect_categorical_values(series)
    table = pd.crosstab(clusters, values).reindex(cluster_order, fill_value=0)
    table = table.loc[:, table.sum(axis=0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0, np.nan, np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chi2, p_value, _, _ = chi2_contingency(table.to_numpy())

    if not np.isfinite(chi2) or not np.isfinite(p_value):
        return 0.0, np.nan, np.nan
    score = -np.log10(max(float(p_value), np.finfo(float).tiny))
    return score, float(chi2), float(p_value)


@_with_notebook_context
def prospect_rank_features_by_cluster(features, clusters, top_k=10):
    """Process prospect rank features by cluster."""
    cluster_order = prospect_ordered_cluster_labels(clusters)
    rows = []
    for col in features.columns:
        kind = prospect_infer_ranked_feature_type(features[col])
        if kind == "continuous":
            score, stat, p_value = prospect_continuous_score(features[col], clusters, cluster_order)
            test = "ANOVA"
        elif kind == "categorical":
            score, stat, p_value = prospect_categorical_score(features[col], clusters, cluster_order)
            test = "chi-square"
        else:
            continue

        rows.append({
            "feature": col,
            "kind": kind,
            "test": test,
            "score": score,
            "statistic": stat,
            "p_value": p_value,
        })

    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    return ranked.sort_values(["score", "statistic"], ascending=False).head(top_k).reset_index(drop=True)


@_with_notebook_context
def prospect_plot_continuous_feature(ax, plot_df, feature, cluster_order, palette):
    """Process prospect plot continuous feature."""
    plot_df = plot_df.copy()
    plot_df[feature] = pd.to_numeric(plot_df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    sns.boxplot(
        data=plot_df,
        x="cluster",
        y=feature,
        order=cluster_order,
        hue="cluster",
        hue_order=cluster_order,
        palette=palette,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="cluster",
        y=feature,
        order=cluster_order,
        color="black",
        size=3,
        jitter=True,
        alpha=0.45,
        ax=ax,
    )
    ax.set_ylabel("")


@_with_notebook_context
def prospect_plot_categorical_feature(ax, plot_df, feature, cluster_order):
    """Process prospect plot categorical feature."""
    cats = prospect_categorical_values(plot_df[feature])
    table = pd.crosstab(plot_df["cluster"], cats, normalize="index").reindex(cluster_order).fillna(0.0)
    table = table.reindex(sorted(table.columns, key=str), axis=1)

    bottom = np.zeros(len(table), dtype=float)
    colors = sns.color_palette("tab20", n_colors=max(1, table.shape[1]))
    x = np.arange(len(table.index))
    for i, category in enumerate(table.columns):
        heights = table[category].to_numpy(dtype=float)
        ax.bar(x, heights, bottom=bottom, label=str(category), color=colors[i], edgecolor="white", linewidth=0.4)
        bottom += heights

    ax.set_xticks(x)
    ax.set_xticklabels(table.index)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Proportion")
    if table.shape[1] <= 8:
        ax.legend(title="", fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    else:
        ax.legend_.remove() if ax.legend_ is not None else None


@_with_notebook_context
def prospect_plot_ranked_feature_grid(features, clusters, ranked, modality, output_dir):
    """Process prospect plot ranked feature grid."""
    cluster_order = prospect_ordered_cluster_labels(clusters)
    palette = sns.color_palette(n_colors=len(cluster_order))
    top_features = ranked["feature"].tolist()
    max_cols = 3
    n_rows = int(np.ceil(len(top_features) / max_cols))
    fig, axes = plt.subplots(n_rows, max_cols, figsize=(max_cols * 6, n_rows * 4.8), squeeze=False)
    axes = axes.flatten()

    plot_df = features.copy()
    plot_df["cluster"] = pd.Series(clusters).astype(str).to_numpy()

    for j, feat in enumerate(top_features):
        ax = axes[j]
        kind = ranked.loc[ranked["feature"] == feat, "kind"].iloc[0]
        score = ranked.loc[ranked["feature"] == feat, "score"].iloc[0]
        test = ranked.loc[ranked["feature"] == feat, "test"].iloc[0]
        if kind == "continuous":
            prospect_plot_continuous_feature(ax, plot_df, feat, cluster_order, palette)
        else:
            prospect_plot_categorical_feature(ax, plot_df, feat, cluster_order)
        ax.set_title(f"{feat}\n{test}, -log10(p)={score:.2f}", fontsize=11)
        ax.set_xlabel("Cluster")
        ax.tick_params(axis="both", labelsize=10)

    for j in range(len(top_features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{modality} - Top features by domain-predicted cluster",
        y=1.02,
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"test_{prospect_safe_name(modality)}_domain_top_features.png"), dpi=300, bbox_inches="tight")
    plt.show()


@_with_notebook_context
def prospect_labels_for_modality(modality, mod_index):
    """Process prospect labels for modality."""
    if "labels_test_by_modality" in globals() and labels_test_by_modality.get(modality) is not None:
        return np.asarray(labels_test_by_modality[modality])
    if "labels_test_modalities" in globals() and mod_index < len(labels_test_modalities):
        labels = labels_test_modalities[mod_index]
        return None if labels is None else np.asarray(labels)
    return None


@_with_notebook_context
def prospect_labels_for_modality_local(modality, mod_index):
    """Process prospect labels for modality local."""
    if "labels_test_by_modality" in globals() and labels_test_by_modality.get(modality) is not None:
        return np.asarray(labels_test_by_modality[modality])
    if "labels_test_modalities" in globals() and mod_index < len(labels_test_modalities):
        labels = labels_test_modalities[mod_index]
        return None if labels is None else np.asarray(labels)
    return None


@_with_notebook_context
def prospect_add_metadata_and_clusters_validation_individual_labels(dict_final, data_full, mod_num):
    """
    Merge cluster labels into full metadata DataFrame using src_subject_id
    from the actual training data stored in fold_metrics['data'].
    This ensures correct alignment even when not all train_ids were clustered.
    """
    clusters = pd.Series(labels_test_modalities[mod_num])

    # Get the first modality’s dataframe — all have the same subject order
    modality_dfs = dict_final
    if not modality_dfs:
        raise ValueError("dict_final is empty; cannot extract subject IDs.")

    # Use the src_subject_id column from the first available modality
    first_modality = list(modality_dfs.keys())[0]
    subj_ids = modality_dfs[first_modality]['src_subject_id'].reset_index(drop=True)

    # Sanity check: should match cluster array length
    if len(subj_ids) != len(clusters):
        print(f"⚠️ Mismatch: {len(subj_ids)} subject IDs vs {len(clusters)} cluster labels.")
        min_len = min(len(subj_ids), len(clusters))
        subj_ids = subj_ids.iloc[:min_len]
        clusters = clusters.iloc[:min_len]
        print(f"Trimmed both to {min_len} entries to align.")

    # Build cluster mapping dataframe
    cluster_df = pd.DataFrame({
        'src_subject_id': subj_ids,
        'Cluster': clusters
    })

    # Merge back into full metadata
    merged = pd.merge(data_full, cluster_df, on='src_subject_id', how='left')

    print(f"✅ Merged clusters for {merged['Cluster'].notna().sum()} subjects (out of {len(merged)} total).")
    return merged


@_with_notebook_context
def prospect_chi_square_comparison_validation_individual_labels(df, group_col, label_col, title_prefix):
    """
    Perform chi-square test and plot grouped bar chart.
    """
    # Drop missing values
    df = df.dropna(subset=[group_col, label_col])

    # Make a copy for plotting / stats so we can safely relabel
    df_plot = df.copy()

    # Rename 'HC' -> 'CC' only for phenotype (for plotting and stats)
    if label_col == 'phenotype':
        df_plot[label_col] = df_plot[label_col].replace({'HC': 'CC'})

    # 1. Summarize counts
    summary_df = (
        df_plot.groupby([group_col, label_col])
        .size()
        .reset_index(name='n')
    )

    # 2. Chi-square test
    tbl = pd.crosstab(df_plot[group_col], df_plot[label_col])
    chi2, pval, dof, expected = chi2_contingency(tbl)
    pval = round(pval, 3)

    # 3. Grouped bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=summary_df,
        x=group_col,
        y='n',
        hue=label_col,
        dodge=True
    )
    plt.title(f"{title_prefix}\n(Chi-square p = {pval})")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    sns.despine()
    plt.tight_layout()
    plt.show()


@_with_notebook_context
def prospect_parse_stream(stream_str):
    """
    Parse stream like:
      "Metabolic_Risk=low -> Blood_markers=high -> ... -> final=1"
    into ordered list of (domain, label).
    """
    if pd.isna(stream_str):
        return []
    parts = [p.strip() for p in ARROW_PAT.split(str(stream_str).strip()) if p.strip()]
    out = []
    for p in parts:
        if "=" in p:
            dom, lab = p.split("=", 1)
            out.append((dom.strip(), lab.strip()))
        else:
            out.append((p.strip(), "<NA>"))
    return out


@_with_notebook_context
def prospect_infer_stage_order(df, stream_col="stream"):
    """
    Infer stage order from the first non-null stream.
    Assumes all streams follow the same domain order.
    """
    s = df[stream_col].dropna().astype(str)
    if s.empty:
        raise ValueError("No streams found to infer stage order.")
    path = prospect_parse_stream(s.iloc[0])
    return [d for d, _ in path]


@_with_notebook_context
def prospect_normalize(v, eps=1e-12):
    """Process prospect normalize."""
    v = np.asarray(v, dtype=float)
    s = v.sum()
    if s <= 0:
        return np.ones_like(v) / max(1, len(v))
    return v / (s + eps)


@_with_notebook_context
def prospect_build_prefix_next(df, stream_col="stream", n_col="n"):
    """
    Returns:
      prefix_next: dict[prefix_tuple_of_(domain,label)] -> dict[next_token_(domain,label)|<END>] -> weight
      prefix_mass: dict[prefix] -> total weight passing through prefix
    Where prefix is a tuple of (domain,label) tokens, e.g.:
      prefix = (("Metabolic_Risk","low"), ("Blood_markers","high"))
    and next token is the next (domain,label) or "<END>".
    """
    prefix_next = {}
    prefix_mass = {}

    for _, row in df.iterrows():
        w = float(row.get(n_col, 1.0))
        path = prospect_parse_stream(row[stream_col])
        if not path:
            continue

        # For each prefix (including empty prefix), record what comes next
        for i in range(len(path)):
            prefix = tuple(path[:i])  # empty prefix for i=0
            nxt = path[i]             # next token
            prefix_mass[prefix] = prefix_mass.get(prefix, 0.0) + w
            d = prefix_next.setdefault(prefix, {})
            d[nxt] = d.get(nxt, 0.0) + w

        # terminal transition from full path to END
        full_prefix = tuple(path)
        prefix_mass[full_prefix] = prefix_mass.get(full_prefix, 0.0) + w
        d = prefix_next.setdefault(full_prefix, {})
        d["<END>"] = d.get("<END>", 0.0) + w

    return prefix_next, prefix_mass


@_with_notebook_context
def prospect_compare_prefix_structure(df_disc, df_test, stream_col="stream", n_col="n", eps=1e-12):
    """
    For each prefix, compare the conditional distribution over next tokens:
        P_disc(next | prefix)  vs  P_test(next | prefix)

    Returns DataFrame with:
      - prefix_str
      - depth
      - js_next (Jensen–Shannon distance on next-step distributions)
      - mass_disc / mass_test (how much data passes through prefix; useful for weighting but not "size-equality")
      - top_next_disc / top_next_test (most likely next step)
      - support_next_overlap (Jaccard on next-token supports)
    """
    pn_d, pm_d = prospect_build_prefix_next(df_disc, stream_col, n_col)
    pn_t, pm_t = prospect_build_prefix_next(df_test, stream_col, n_col)

    prefixes = set(pn_d.keys()) | set(pn_t.keys())

    rows = []
    for pref in prefixes:
        nd = pn_d.get(pref, {})
        nt = pn_t.get(pref, {})

        keys = set(nd.keys()) | set(nt.keys())
        # Align next-token vectors
        vd = np.array([nd.get(k, 0.0) for k in keys], dtype=float)
        vt = np.array([nt.get(k, 0.0) for k in keys], dtype=float)

        pd_ = prospect_normalize(vd, eps=eps)
        pt_ = prospect_normalize(vt, eps=eps)

        js = float(jensenshannon(pd_ + eps, pt_ + eps, base=2.0))

        # Next-token support overlap (presence/absence, structure)
        supp_d = {k for k, v in nd.items() if v > 0}
        supp_t = {k for k, v in nt.items() if v > 0}
        supp_j = len(supp_d & supp_t) / max(1, len(supp_d | supp_t))

        # Most likely next token in each
        top_d = max(nd.items(), key=lambda x: x[1])[0] if nd else None
        top_t = max(nt.items(), key=lambda x: x[1])[0] if nt else None

        def tok_str(tok):
            if tok == "<END>":
                return "<END>"
            if tok is None:
                return "<NONE>"
            return f"{tok[0]}={tok[1]}"

        prefix_str = " → ".join([f"{d}={l}" for d, l in pref]) if pref else "<START>"
        rows.append({
            "prefix_str": prefix_str,
            "depth": len(pref),
            "js_next": js,
            "mass_disc": pm_d.get(pref, 0.0),
            "mass_test": pm_t.get(pref, 0.0),
            "top_next_disc": tok_str(top_d),
            "top_next_test": tok_str(top_t),
            "support_next_jaccard": supp_j,
            "prefix_exists_in_disc": pref in pn_d,
            "prefix_exists_in_test": pref in pn_t,
        })

    out = pd.DataFrame(rows)
    # A useful default sorting: prioritize structurally-different AND commonly-used prefixes
    out["mass_min"] = np.minimum(out["mass_disc"], out["mass_test"])
    out = out.sort_values(["mass_min", "js_next"], ascending=[False, False]).reset_index(drop=True)
    return out


@_with_notebook_context
def prospect_plot_top_prefix_differences(prefix_report, top_n=20, min_depth=1):
    """
    Barh plot of top prefixes by JS(next) after filtering.
    """
    d = prefix_report[prefix_report["depth"] >= min_depth].copy()
    d = d.sort_values("js_next", ascending=False).head(top_n)

    plt.figure(figsize=(11, max(4, 0.35 * len(d))))
    y = np.arange(len(d))[::-1]
    plt.barh(y, d["js_next"].iloc[::-1].to_numpy())
    plt.yticks(y, d["prefix_str"].iloc[::-1].to_list(), fontsize=9)
    plt.xlabel("Jensen–Shannon distance of P(next | prefix)")
    plt.title("Most structurally different prefixes (next-step rule)")
    plt.tight_layout()
    plt.show()


@_with_notebook_context
def prospect_final_mapping_table(df, stream_col="stream", n_col="n", final_domain="final"):
    """
    Builds a table over modality-prefixes (everything up to but excluding final)
    with P(final=label | prefix) computed within each dataset.

    Returns DataFrame:
      prefix_str, final_label, weight
    and also a pivoted table of P(final=...) by prefix.
    """
    rows = []
    for _, row in df.iterrows():
        w = float(row.get(n_col, 1.0))
        path = prospect_parse_stream(row[stream_col])
        if not path:
            continue

        # split into prefix (before final) and final token
        final_tokens = [t for t in path if t[0] == final_domain]
        if not final_tokens:
            # If final isn't explicitly present, skip
            continue
        final_tok = final_tokens[-1]  # in case of duplicates, take last
        final_label = final_tok[1]

        # prefix = all tokens before the final token position (first occurrence)
        # more robust: take all tokens except the final-domain token(s)
        prefix = tuple([t for t in path if t[0] != final_domain])

        prefix_str = " → ".join([f"{d}={l}" for d, l in prefix]) if prefix else "<NO_MODALITIES>"
        rows.append({"prefix_str": prefix_str, "final_label": final_label, "w": w})

    long = pd.DataFrame(rows)
    if long.empty:
        return long, pd.DataFrame()

    # Compute conditional probabilities P(final_label | prefix)
    grp = long.groupby(["prefix_str", "final_label"], dropna=False)["w"].sum().reset_index()
    totals = grp.groupby("prefix_str")["w"].sum().reset_index().rename(columns={"w": "w_total"})
    grp = grp.merge(totals, on="prefix_str", how="left")
    grp["p_final_given_prefix"] = grp["w"] / grp["w_total"]

    pivot = grp.pivot_table(index="prefix_str", columns="final_label", values="p_final_given_prefix", fill_value=0.0)
    return grp, pivot


@_with_notebook_context
def prospect_compare_final_mapping(df_disc, df_test, stream_col="stream", n_col="n", final_domain="final"):
    """
    Compare P(final | modalities) between discovery and test.
    Returns a table with per-prefix deltas per final label plus summary metrics.
    """
    long_d, piv_d = prospect_final_mapping_table(df_disc, stream_col, n_col, final_domain)
    long_t, piv_t = prospect_final_mapping_table(df_test, stream_col, n_col, final_domain)

    if piv_d.empty or piv_t.empty:
        return pd.DataFrame(), {"note": "No final mapping found (missing final tokens?)"}

    # align
    idx = sorted(set(piv_d.index) | set(piv_t.index))
    cols = sorted(set(piv_d.columns) | set(piv_t.columns))
    A = piv_d.reindex(index=idx, columns=cols).fillna(0.0)
    B = piv_t.reindex(index=idx, columns=cols).fillna(0.0)

    # Delta per final label
    delta = (B - A)
    out = delta.copy()
    out.columns = [f"delta_final={c}" for c in out.columns]
    out.insert(0, "prefix_str", out.index)

    # A per-prefix summary: JS distance between final distributions for that prefix
    js_list = []
    for i in range(A.shape[0]):
        p = prospect_normalize(A.iloc[i].to_numpy())
        q = prospect_normalize(B.iloc[i].to_numpy())
        js_list.append(float(jensenshannon(p + 1e-12, q + 1e-12, base=2.0)))
    out["js_final_given_prefix"] = js_list

    # Add weights: how common the prefix is (within each dataset)
    wD = long_d.groupby("prefix_str")["w"].sum() if not long_d.empty else pd.Series(dtype=float)
    wT = long_t.groupby("prefix_str")["w"].sum() if not long_t.empty else pd.Series(dtype=float)
    out["w_disc"] = out["prefix_str"].map(wD).fillna(0.0)
    out["w_test"] = out["prefix_str"].map(wT).fillna(0.0)
    out["w_min"] = np.minimum(out["w_disc"], out["w_test"])

    # Sort by (common prefixes) then by biggest JS shift in final mapping
    out = out.sort_values(["w_min", "js_final_given_prefix"], ascending=[False, False]).reset_index(drop=True)

    # Global summary metric: weighted average JS over prefixes (weights = w_min)
    w = out["w_min"].to_numpy()
    if w.sum() > 0:
        global_js = float(np.sum(out["js_final_given_prefix"].to_numpy() * w) / w.sum())
    else:
        global_js = float(out["js_final_given_prefix"].mean())

    metrics = {
        "num_prefixes_union": len(out),
        "weighted_js_final_given_prefix": global_js,
        "final_labels": cols,
    }
    return out, metrics


@_with_notebook_context
def prospect_plot_top_final_mapping_shifts(final_cmp, top_n=20):
    """Process prospect plot top final mapping shifts."""
    d = final_cmp.head(top_n).copy()
    plt.figure(figsize=(11, max(4, 0.35 * len(d))))
    y = np.arange(len(d))[::-1]
    plt.barh(y, d["js_final_given_prefix"].iloc[::-1].to_numpy())
    plt.yticks(y, d["prefix_str"].iloc[::-1].to_list(), fontsize=9)
    plt.xlabel("JS distance between P(final | prefix) in test vs discovery")
    plt.title("Prefixes with biggest changes in final mapping")
    plt.tight_layout()
    plt.show()


@_with_notebook_context
def prospect_stream_presence_and_topk(df_disc, df_test, stream_col="stream", n_col="n", topk=30):
    """Process prospect stream presence and topk."""
    A = set(df_disc[stream_col].dropna().astype(str))
    B = set(df_test[stream_col].dropna().astype(str))

    presence = {
        "unique_streams_disc": len(A),
        "unique_streams_test": len(B),
        "stream_jaccard_presence": len(A & B) / max(1, len(A | B)),
        "coverage_disc_in_test": len(A & B) / max(1, len(A)),
        "coverage_test_in_disc": len(A & B) / max(1, len(B)),
    }

    pD = df_disc.groupby(stream_col)[n_col].sum().sort_values(ascending=False)
    pT = df_test.groupby(stream_col)[n_col].sum().sort_values(ascending=False)
    pD = pD / pD.sum()
    pT = pT / pT.sum()

    topD = set(pD.index[: min(topk, len(pD))].astype(str))
    topT = set(pT.index[: min(topk, len(pT))].astype(str))

    presence[f"top{topk}_jaccard_by_rank"] = len(topD & topT) / max(1, len(topD | topT))
    presence["top_disc_only"] = sorted(list(topD - topT))[:10]
    presence["top_test_only"] = sorted(list(topT - topD))[:10]
    return presence, pD, pT


@_with_notebook_context
def prospect_sankey_from_streams(df, stream_col="stream", n_col="n", max_edges=200):
    """
    Build a Sankey graph from full streams.
    Nodes are stage-specific label tokens: f"{domain}={label}".
    Edges connect consecutive tokens. We prune to max_edges by weight.
    """
    if not _HAS_PLOTLY:
        raise RuntimeError("Plotly not installed; cannot draw sankey.")

    edge_w = {}
    for _, row in df.iterrows():
        w = float(row.get(n_col, 1.0))
        path = prospect_parse_stream(row[stream_col])
        toks = [f"{d}={l}" for d, l in path]
        for a, b in zip(toks[:-1], toks[1:]):
            edge_w[(a, b)] = edge_w.get((a, b), 0.0) + w

    # prune
    edges = sorted(edge_w.items(), key=lambda x: -x[1])[:max_edges]
    nodes = {}
    def nid(x):
        if x not in nodes:
            nodes[x] = len(nodes)
        return nodes[x]

    src, tgt, val = [], [], []
    for (a, b), w in edges:
        src.append(nid(a))
        tgt.append(nid(b))
        val.append(w)

    labels = [None] * len(nodes)
    for k, i in nodes.items():
        labels[i] = k

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=12, thickness=12),
        link=dict(source=src, target=tgt, value=val),
    )])
    return fig


@_with_notebook_context
def prospect_full_structure_report(stream_summary, stream_summary_test, stream_col="stream", n_col="n", topk=30, final_domain="final"):
    # Presence + top-k overlap
    """Process prospect full structure report."""
    presence, pD, pT = prospect_stream_presence_and_topk(stream_summary, stream_summary_test, stream_col, n_col, topk=topk)

    # Prefix-tree structural differences
    prefix_report = prospect_compare_prefix_structure(stream_summary, stream_summary_test, stream_col, n_col)

    # Full mapping to final
    final_cmp, final_metrics = prospect_compare_final_mapping(stream_summary, stream_summary_test, stream_col, n_col, final_domain)

    return {
        "presence_metrics": presence,
        "p_stream_disc": pD,
        "p_stream_test": pT,
        "prefix_report": prefix_report,
        "final_mapping_compare": final_cmp,
        "final_mapping_metrics": final_metrics,
    }


@_with_notebook_context
def prospect_all_streams_table(stream_summary, stream_summary_test, stream_col="stream", n_col="n"):
    # Aggregate in case there are duplicate stream rows
    """Process prospect all streams table."""
    disc = (stream_summary
            .groupby(stream_col, dropna=False)[n_col]
            .sum()
            .rename("n_disc")
            .to_frame())

    test = (stream_summary_test
            .groupby(stream_col, dropna=False)[n_col]
            .sum()
            .rename("n_test")
            .to_frame())

    # Outer join gives union of streams
    tbl = disc.join(test, how="outer").fillna(0)

    # Add proportions within each dataset
    N_disc = tbl["n_disc"].sum()
    N_test = tbl["n_test"].sum()

    tbl["p_disc"] = tbl["n_disc"] / N_disc if N_disc > 0 else np.nan
    tbl["p_test"] = tbl["n_test"] / N_test if N_test > 0 else np.nan

    # Helpful comparisons
    tbl["delta_p"] = tbl["p_test"] - tbl["p_disc"]
    tbl["abs_delta_p"] = tbl["delta_p"].abs()
    tbl["log2_fc"] = np.log2((tbl["p_test"] + 1e-12) / (tbl["p_disc"] + 1e-12))

    # Make stream a real column, sort by biggest shift
    tbl = tbl.reset_index().rename(columns={stream_col: "stream"})
    tbl = tbl.sort_values("abs_delta_p", ascending=False).reset_index(drop=True)

    return tbl
