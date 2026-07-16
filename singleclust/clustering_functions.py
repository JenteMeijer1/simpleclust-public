"""Define candidate generation and scoring for single-view clustering."""

import re
import os
import sys
from dataclasses import dataclass
from typing import List
from collections import defaultdict

import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import pairwise_distances

_PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from parea_classes import EnsembleClusteringPyrea, enforce_min_cluster_size

CLUSTER_METHODS = ['spectral', 'hierarchical', 'gmm', 'kmeans', 'meanshift', 'ensemble']
FUSION_METHODS = ['agreement', 'consensus', 'disagreement']
LINKAGES = ['complete', 'average', 'weighted', 'ward']


class _Clusterer:
    def execute(self, data):
        """Handle execute."""
        raise NotImplementedError


class _HierarchicalClusterer(_Clusterer):
    def __init__(self, n_clusters=2, precomputed=False, linkage_method='average', linkage=None, **kwargs):
        """Initialize the object."""
        self.n_clusters = int(max(1, n_clusters))
        self.precomputed = bool(precomputed)
        self.linkage_method = (linkage_method or linkage or 'average')
        if self.precomputed and self.linkage_method == 'ward':
            self.linkage_method = 'average'

    def execute(self, data):
        """Handle execute."""
        X = np.asarray(data)
        if X.shape[0] <= 1 or self.n_clusters <= 1:
            return np.zeros(X.shape[0], dtype=int)
        if self.precomputed:
            D = np.asarray(X, dtype=float)
            D = np.nan_to_num(D, nan=0.0, posinf=1.0, neginf=0.0)
            D = 0.5 * (D + D.T)
            np.fill_diagonal(D, 0.0)
            y = squareform(D, checks=False)
            Z = hierarchy.linkage(y, method=self.linkage_method)
        else:
            X = np.asarray(X, dtype=float)
            if self.linkage_method == 'ward':
                Z = hierarchy.linkage(X, method='ward')
            else:
                y = pdist(X, metric='euclidean')
                Z = hierarchy.linkage(y, method=self.linkage_method)
        try:
            labels = hierarchy.cut_tree(Z, n_clusters=self.n_clusters).reshape(-1)
        except Exception:
            labels = hierarchy.fcluster(Z, t=self.n_clusters, criterion='maxclust') - 1
        return np.asarray(labels, dtype=int)


class _KMeansClusterer(_Clusterer):
    def __init__(self, n_clusters=2, random_state=42, **kwargs):
        """Initialize the object."""
        self.n_clusters = int(max(1, n_clusters))
        self.random_state = random_state

    def execute(self, data):
        """Handle execute."""
        X = np.asarray(data, dtype=float)
        if X.shape[0] <= 1 or self.n_clusters <= 1:
            return np.zeros(X.shape[0], dtype=int)
        if X.shape[0] < self.n_clusters:
            return np.arange(X.shape[0], dtype=int)
        return KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state).fit_predict(X)


class _GMMClusterer(_Clusterer):
    def __init__(self, n_clusters=2, n_components=None, random_state=42, **kwargs):
        """Initialize the object."""
        self.n_clusters = int(max(1, n_components or n_clusters))
        self.random_state = random_state

    def execute(self, data):
        """Handle execute."""
        X = np.asarray(data, dtype=float)
        if X.shape[0] <= 1 or self.n_clusters <= 1:
            return np.zeros(X.shape[0], dtype=int)
        if X.shape[0] < self.n_clusters:
            return np.arange(X.shape[0], dtype=int)
        return GaussianMixture(n_components=self.n_clusters, random_state=self.random_state).fit_predict(X)


class _SpectralClusterer(_Clusterer):
    def __init__(self, n_clusters=2, precomputed=False, random_state=42, **kwargs):
        """Initialize the object."""
        self.n_clusters = int(max(1, n_clusters))
        self.precomputed = bool(precomputed)
        self.random_state = random_state

    def execute(self, data):
        """Handle execute."""
        X = np.asarray(data, dtype=float)
        n = X.shape[0]
        if n <= 1 or self.n_clusters <= 1:
            return np.zeros(n, dtype=int)
        if n < self.n_clusters:
            return np.arange(n, dtype=int)
        try:
            if self.precomputed:
                A = 1.0 - X
                A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
                A = 0.5 * (A + A.T)
                np.fill_diagonal(A, 1.0)
                return SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed', random_state=self.random_state, n_init=10).fit_predict(A)
            return SpectralClustering(n_clusters=self.n_clusters, affinity='nearest_neighbors', random_state=self.random_state, n_init=10).fit_predict(X)
        except Exception:
            return _KMeansClusterer(self.n_clusters, random_state=self.random_state).execute(X)


class _MeanShiftClusterer(_Clusterer):
    def execute(self, data):
        """Handle execute."""
        X = np.asarray(data, dtype=float)
        if X.shape[0] <= 1:
            return np.zeros(X.shape[0], dtype=int)
        return MeanShift().fit_predict(X)


class _EnsembleClusterer(_Clusterer):
    def __init__(self, n_clusters=2, precomputed=False, linkage_method='average', linkage=None, random_state=42, **kwargs):
        """Initialize the object."""
        self.n_clusters = int(max(1, n_clusters))
        self.precomputed = bool(precomputed)
        self.linkage_method = linkage_method or linkage or 'average'
        self.random_state = random_state

    def execute(self, data):
        """Handle execute."""
        X = np.asarray(data, dtype=float)
        n = X.shape[0]
        if n <= 1 or self.n_clusters <= 1:
            return np.zeros(n, dtype=int)
        if self.precomputed:
            D = np.asarray(X, dtype=float)
            D = np.nan_to_num(D, nan=0.0, posinf=1.0, neginf=0.0)
            D = 0.5 * (D + D.T)
            np.fill_diagonal(D, 0.0)
            labelings = [
                _HierarchicalClusterer(self.n_clusters, precomputed=True, linkage_method=self.linkage_method).execute(D),
                _SpectralClusterer(self.n_clusters, precomputed=True, random_state=self.random_state).execute(D),
            ]
            return _consensus_from_labelings(labelings, self.n_clusters, self.linkage_method)

        labelings = []
        for model in (
            _KMeansClusterer(self.n_clusters, random_state=self.random_state),
            _GMMClusterer(self.n_clusters, random_state=self.random_state),
            _HierarchicalClusterer(self.n_clusters, precomputed=False, linkage_method=self.linkage_method),
            _SpectralClusterer(self.n_clusters, precomputed=False, random_state=self.random_state),
        ):
            try:
                labelings.append(np.asarray(model.execute(X), dtype=int))
            except Exception:
                continue
        if not labelings:
            return np.zeros(n, dtype=int)
        return _consensus_from_labelings(labelings, self.n_clusters, self.linkage_method)


def _consensus_from_labelings(labelings, n_clusters, linkage_method='average'):
    """Handle consensus from labelings."""
    n = len(labelings[0])
    coassoc = np.zeros((n, n), dtype=float)
    for lab in labelings:
        coassoc += (lab[:, None] == lab[None, :]).astype(float)
    coassoc /= float(len(labelings))
    D = 1.0 - coassoc
    D = np.clip(np.nan_to_num(D, nan=1.0), 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    method = 'average' if linkage_method == 'ward' else linkage_method
    return _HierarchicalClusterer(n_clusters=n_clusters, precomputed=True, linkage_method=method).execute(D)


# Public builder API

def build_clusterer(kind: str, precomputed: bool = False, **kwargs):
    """Build clusterer."""
    if kind not in CLUSTER_METHODS:
        raise ValueError(f"Unknown clusterer '{kind}'.")
    if kind == 'hierarchical':
        return _HierarchicalClusterer(precomputed=precomputed, **kwargs)
    if kind == 'kmeans':
        return _KMeansClusterer(**kwargs)
    if kind == 'gmm':
        return _GMMClusterer(**kwargs)
    if kind == 'spectral':
        return _SpectralClusterer(precomputed=precomputed, **kwargs)
    if kind == 'meanshift':
        return _MeanShiftClusterer()
    if kind == 'ensemble':
        return _EnsembleClusterer(precomputed=precomputed, **kwargs)
    raise ValueError(f"Unsupported clusterer '{kind}'.")


class View:
    def __init__(self, data, clusterer):
        """Initialize the object."""
        self.data = np.asarray(data)
        self.clusterer = clusterer
        self.labels = None

    def execute(self):
        """Handle execute."""
        self.labels = np.asarray(self.clusterer.execute(self.data), dtype=int)
        return self.labels


def build_view(data, clusterer):
    """Build view."""
    return View(data, clusterer)


@dataclass
class Fusion:
    name: str


class Agreement(Fusion):
    def __init__(self):
        """Initialize the object."""
        super().__init__('agreement')


class Disagreement(Fusion):
    def __init__(self):
        """Initialize the object."""
        super().__init__('disagreement')


class Consensus(Fusion):
    def __init__(self):
        """Initialize the object."""
        super().__init__('consensus')


def build_fuser(name: str):
    """Build fuser."""
    name = str(name).lower()
    if name == 'agreement':
        return Agreement()
    if name == 'disagreement':
        return Disagreement()
    if name == 'consensus':
        return Consensus()
    raise ValueError(f"Unknown fusion method '{name}'.")


def _coassociation_from_labels(labelings: List[np.ndarray]) -> np.ndarray:
    """Handle coassociation from labels."""
    n = len(labelings[0])
    M = np.zeros((n, n), dtype=float)
    for lab in labelings:
        M += (lab[:, None] == lab[None, :]).astype(float)
    M /= float(len(labelings))
    M = 0.5 * (M + M.T)
    np.fill_diagonal(M, 1.0)
    return M


def consensus(labels: list):
    """Handle consensus."""
    if len(labels) < 1:
        raise ValueError('labels must be non-empty')
    labmat = np.vstack([np.asarray(l) for l in labels]).T
    mapper = {}
    out = np.zeros(labmat.shape[0], dtype=int)
    next_id = 0
    for i, row in enumerate(labmat):
        key = tuple(row.tolist())
        if key not in mapper:
            mapper[key] = next_id
            next_id += 1
        out[i] = mapper[key]
    return out


def run_ensemble_fusion(views: List[View], fuser: Fusion, mincluster: str = 'FALSE', mincluster_n: int = 10, cluster_algorithm='ensemble'):
    """Run ensemble fusion."""
    if not isinstance(views, list) or len(views) == 0:
        raise ValueError('views must be a non-empty list')
    member_labels = [np.asarray(v.execute(), dtype=int) for v in views]
    if any(len(l) != len(member_labels[0]) for l in member_labels):
        raise ValueError('All views must have the same number of samples')
    name = getattr(fuser, 'name', str(fuser)).lower()
    if name == 'consensus':
        lab = consensus(member_labels)
        coassoc = (lab[:, None] == lab[None, :]).astype(float)
        np.fill_diagonal(coassoc, 1.0)
        fusion_distance = 1.0 - coassoc
    else:
        coassoc = _coassociation_from_labels(member_labels)
        if name == 'agreement':
            fusion_distance = 1.0 - coassoc
        elif name == 'disagreement':
            fusion_distance = coassoc.copy()
            np.fill_diagonal(fusion_distance, 0.0)
        else:
            raise ValueError(f"Unknown fusion method '{name}'.")
    fusion_distance = np.asarray(fusion_distance, dtype=float)
    fusion_distance = np.clip(np.nan_to_num(fusion_distance, nan=1.0), 0.0, 1.0)
    np.fill_diagonal(fusion_distance, 0.0)
    return fusion_distance, member_labels


def _normalize_views(data):
    """Normalize views."""
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError('Expected 2D array for clustering data.')
        return [np.asarray(data, dtype=float)]
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError('Data list is empty.')
        out = [np.asarray(x, dtype=float) for x in data]
        n = out[0].shape[0]
        if any(a.ndim != 2 for a in out):
            raise ValueError('Each view must be a 2D array.')
        if any(a.shape[0] != n for a in out):
            raise ValueError('All views must share rows.')
        return out
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 2:
        return [arr]
    raise ValueError('Unsupported data type for run_ensemble_clustering')


def _normalize_per_view_param(value, n_views, default):
    """Normalize per view param."""
    if value is None:
        return [default] * n_views
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = list(value)
        if not vals:
            return [default] * n_views
        if len(vals) < n_views:
            vals += [vals[-1]] * (n_views - len(vals))
        return vals[:n_views]
    return [value] * n_views


def _enforce_min_cluster_size(X, labels, min_size=10):
    """Enforce min cluster size."""
    D = pairwise_distances(np.asarray(X, dtype=float))
    return enforce_min_cluster_size(D, labels, min_size=min_size)


def _safe_quality(X, labels, precomputed=False):
    """Handle safe quality."""
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0 or len(np.unique(labels)) <= 1:
        return 0.0
    try:
        sil = silhouette_score(X, labels, metric='precomputed' if precomputed else 'euclidean')
        sil_n = (sil + 1.0) / 2.0
    except Exception:
        sil_n = 0.0
    try:
        if precomputed:
            D = np.asarray(X, dtype=float)
            n = D.shape[0]
            J = np.eye(n) - np.ones((n, n)) / n
            B = -0.5 * J @ (D ** 2) @ J
            vals, vecs = np.linalg.eigh(B)
            idx = np.argsort(vals)[::-1]
            vals, vecs = vals[idx], vecs[:, idx]
            pos = vals > 1e-9
            if np.any(pos):
                m = min(10, int(np.sum(pos)))
                Xq = vecs[:, pos][:, :m] * np.sqrt(vals[pos][:m])
            else:
                Xq = np.zeros((n, 1))
        else:
            Xq = np.asarray(X, dtype=float)
        ch_n = calinski_harabasz_score(Xq, labels)
        ch_n = ch_n / (ch_n + 1.0)
        db_n = 1.0 / (1.0 + davies_bouldin_score(Xq, labels))
    except Exception:
        ch_n = 0.0
        db_n = 0.0
    return float(np.mean([sil_n, ch_n, db_n]))


def run_ensemble_clustering(
    data,
    k=None,
    linkage='average',
    fitness=False,
    subject_id_list=None,
    inner_jobs: int = 1,
    pre_inner_jobs: int = 1,
    mincluster='FALSE',
    mincluster_n=10,
    internal_ensemble_enabled=False,
    internal_ensemble_bcs=5,
    internal_ensemble_sample_frac=0.8,
    internal_ensemble_feature_frac=1.0,
    internal_ensemble_seed=0,
    **kwargs,
):
    """Run the singleclust candidate clusterer on one numeric feature matrix.

    The SchizBull/simpleclust path passes one view: the preprocessed
    `Include_cluster` feature matrix, or an optional reduced representation of
    it. The candidate controls requested `k` and hierarchical linkage. When the
    internal ensemble is enabled, `EnsembleClusteringPyrea` builds perturbed
    base clusterings and returns the consensus labels used for scoring.
    """
    X = _normalize_views(data)[0]
    n_samples = X.shape[0]

    if k is None:
        k = 2

    k = max(1, int(k))
    linkage = str(linkage or 'average')

    if str(mincluster).upper() == 'TRUE' and n_samples > 0:
        max_k = max(1, n_samples // max(1, int(mincluster_n)))
        k = min(k, max_k)

    labels = EnsembleClusteringPyrea(
        n_clusters=k,
        precomputed=False,
        linkage_method=linkage,
        internal_ensemble_enabled=internal_ensemble_enabled,
        internal_ensemble_bcs=internal_ensemble_bcs,
        internal_ensemble_sample_frac=internal_ensemble_sample_frac,
        internal_ensemble_feature_frac=internal_ensemble_feature_frac,
        internal_ensemble_seed=internal_ensemble_seed,
    ).execute(X)

    labels = np.asarray(labels, dtype=int).reshape(-1)
    if str(mincluster).upper() == 'TRUE':
        labels = _enforce_min_cluster_size(X, labels, int(mincluster_n))

    quality = _safe_quality(X, labels, precomputed=False)

    if fitness and len(np.unique(labels)) <= 1:
        return 0.0
    return labels, quality


def decode_search_candidate(data_len, individual):
    """Decode a single-search candidate for one clustering run (k + linkage)."""
    if not hasattr(individual, 'gene_names'):
        raise AttributeError('Individual missing gene_names metadata.')

    grouped = defaultdict(list)
    for name, val in zip(individual.gene_names, individual):
        grouped[name].append(val)

    if 'k' not in grouped:
        raise KeyError("Search candidate is missing required gene 'k'.")
    if 'linkage' not in grouped:
        raise KeyError("Search candidate is missing required gene 'linkage'.")

    return {
        'k': int(grouped['k'][0]),
        'linkage': str(grouped['linkage'][0]),
    }


__all__ = [
    'CLUSTER_METHODS', 'FUSION_METHODS', 'LINKAGES',
    'View', 'Fusion', 'Agreement', 'Disagreement', 'Consensus',
    'build_clusterer', 'build_view', 'build_fuser',
    'run_ensemble_fusion', 'run_ensemble_clustering', 'decode_search_candidate',
    'consensus',
]
