"""Provide supporting operations for Parea ensemble clustering."""

# Parea functions

import random
from array import array
from cmath import exp
from typing import List, Union
import numpy as np
import warnings
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from operator import itemgetter
# Parallelization imports

from joblib import Parallel, delayed
import pandas as pd
from sklearn.impute import KNNImputer
from operator import itemgetter



# Genetic algorithm imports
from deap import base, creator, tools, algorithms
from parea_classes import *

CLUSTER_METHODS = ['spectral', 'hierarchical', 'gmm', 'kmeans', 'meanshift', 'ensemble']
FUSION_METHODS = ['agreement', 'consensus', 'disagreement']
LINKAGES = ['complete', 'average', 'weighted', 'ward', 'ward2']

def clusterer(clusterer: str, precomputed: bool=False, **kwargs) -> Clusterer:
    #It tests which cluster method is given. For each cluster method there is a seperate function in the structure.py file.
    # Each of these functions are relatively simple so no need to add them here. It calls a function to perform a specific clustering.
    # If we want to add clustering methods, look at these functions on how to do that and make a similar function for the new clustering method. #TODO

    """
    Creates a :class:`~pyrea.structure.Clusterer` object to be used when
    creating a :class:`~pyrea.structure.View` or
    :class:`~pyrea.structure.Ensemble`. Can be one of: :attr:`'spectral'`,
    :attr:`'hierarchical'`, :attr:`'dbscan'`, or :attr:`'optics'`.

    .. code::

        c = pyrea.clusterer('hierarchical', n_clusters=2)

    Then, :attr:`c` can be used when creating a view:

    .. code::

        v = pyrea.view(d, c)

    Where :attr:`d` is a data source.

    .. seealso:: The :func:`~view` function.
    .. seealso:: The :func:`~execute_ensemble` function.

    Each clustering algorithm has a different set of parameters, default values
    are used throughout and can be overridden if required. For example,
    hierarchical and spectral clustering allow you to specify the number of
    clusters to find using :attr:`n_clusters`, while DBSCAN and OPTICS do not.

    Also, hierarchical clustering allows for a :attr:`distance_metric` to be
    set, which can be one of: :attr:`'braycurtis'`, :attr:`'canberra'`,
    :attr:`'chebyshev'`, :attr:`'cityblock'`, :attr:`'correlation'`,
    :attr:`'cosine'`, :attr:`'dice'`, :attr:`'euclidean'`, :attr:`'hamming'`,
    :attr:`'jaccard'`, :attr:`'jensenshannon'`, :attr:`'kulczynski1'`,
    :attr:`'mahalanobis'`, :attr:`'matching'`, :attr:`'minkowski'`,
    :attr:`'rogerstanimoto'`, :attr:`'russellrao'`, :attr:`'seuclidean'`,
    :attr:`'sokalmichener'`, :attr:`'sokalsneath'`, :attr:`'sqeuclidean'`, or
    :attr:`'yule'`.

    Likewise, adjusting the linkage method is possible using hierarchical
    clustering algorithms, this can be one of: :attr:`'single'`,
    :attr:`'complete'`, :attr:`'average'`, :attr:`'weighted'`,
    :attr:`'centroid'`, :attr:`'median'`, or :attr:`'ward'`.

    For complete documentation of each clustering algorithm's parameters see
    the following:

    * Spectral: :class:`~pyrea.structure.SpectralClusteringPyrea`
    * Hierarchical: :class:`~pyrea.structure.HierarchicalClusteringPyrea`
    * DBSCAN: :class:`~pyrea.structure.DBSCANPyrea`
    * OPTICS: :class:`~pyrea.structure.OPTICSPyrea`

    :param clusterer: The type of clusterer to use. Can be one of:
     :attr:`'spectral'`, :attr:`'hierarchical'`, :attr:`'dbscan'`,
     or :attr:`'optics'`.
    :param precomputed: Whether the clusterer should assume the data is a
     distance matrix.
    :param \*\*kwargs: Keyword arguments to be passed to the clusterer.
     See each clustering algorithm's documentation for full details: Spectral:
     :class:`~pyrea.structure.SpectralClusteringPyrea`, Hierarchical:
     :class:`~pyrea.structure.HierarchicalClusteringPyrea`, DBSCAN:
     :class:`~pyrea.structure.DBSCANPyrea`, and OPTICS:
     :class:`~pyrea.structure.OPTICSPyrea`.
    """
    if not isinstance(clusterer, str):
        raise TypeError("Parameter 'clusterer' must be of type string. Choices available are: %s."
                        % ("'" + "', '".join(CLUSTER_METHODS[:-1]) + "', or '" + CLUSTER_METHODS[-1] + "'"))

    if clusterer not in CLUSTER_METHODS:
        raise TypeError("Parameter 'clusterer' must be one of %s: you passed '%s'."
                        % ("'" + "', '".join(CLUSTER_METHODS[:-1]) + "', or '" + CLUSTER_METHODS[-1] + "'", clusterer))

    if clusterer == 'spectral':
        if precomputed:
            kwargs['affinity'] = 'precomputed'
        return SpectralClusteringPyrea(**kwargs)

    elif clusterer == 'hierarchical':

        method = kwargs.get('linkage')
        if method:
            if method not in LINKAGES:
                raise TypeError("Illegal method.")
        else:
            kwargs['linkage'] = 'ward' #If no method is given, it defaults to ward.

        if not kwargs.get('n_clusters'):
            raise TypeError("Error: n_clusters not set and is required for hierarchical clustering.")

        return HierarchicalClusteringPyrea(precomputed=precomputed, **kwargs)

    elif clusterer == 'dbscan':
        if precomputed:
            kwargs['metric']='precomputed'

        return DBSCANPyrea(**kwargs)

    elif clusterer == 'optics':
        if precomputed:
            kwargs['metric']='precomputed'

        return OPTICSPyrea(**kwargs)

    elif clusterer == 'gmm':
        return ModelBasedClusteringPyrea(**kwargs)

    elif clusterer == 'kmeans':
        return KMeansPyrea(**kwargs)

    elif clusterer == 'meanshift':
        return MeanShiftPyrea(**kwargs)

    elif clusterer == 'birch':
        return BIRCHPyrea(**kwargs)

    elif clusterer == 'ensemble':
        return EnsembleClusteringPyrea(precomputed=precomputed, **kwargs)
    else:
        raise ValueError("Unknown clustering method.")


def view(data: array, clusterer: Clusterer) -> View:
    """
    Creates a :class:`View` object that can subsequently used to create an
    :class:`Ensemble`.

    Views are created using some data in the form of a NumPy matrix or 2D array,
    and a clustering algorithm:

    .. code::

        d = numpy.random.rand(100,10)
        v = pyrea.view(d, c)

    Views are used to create ensembles. They consist of some data, :attr:`d`
    above, and a clustering algorimth, :attr:`c` above.
    """
    return View(data, clusterer)


def fuser(fuser: str):
    """
    Creates a :class:`Fusion` object, which is used to fuse the results of
    an arbitrarily long list of clusterings.

    .. code::

        f = pyrea.fuser('agreement')

    :param fuser: The fusion algorithm to use. Must be one of 'agreement',
     'disagreement', 'consensus'.
    """
    if not isinstance(fuser, str):
        raise TypeError("Parameter 'fuser' must be of type string.")

    if fuser == "disagreement":
        return Disagreement()
    elif fuser == "agreement":
        return Agreement()
    elif fuser == "consensus":
        return Consensus()

def execute_ensemble(views: List[View], fuser: Fusion, mincluster: str ="FALSE", mincluster_n: int = 10) -> list:
    """
    Executes an ensemble and returns a new :class:`View` object.

    :param views: The ensemble's views.
    :param fuser: The fusion algorithm used to fuse the clustered data.
    :param clusterers: A clustering algorithm or list of clustering algorithms
     used to cluster the fused matrix created by the fusion algorithm.

    .. code::

        v = pyrea.execute_ensemble([view1, view2, view3], fusion, clusterer)

    Returns a :class:`~pyrea.structure.View` object which can consequently be
    included in a further ensemble.

    .. seealso:: The :func:`~view` function.
    .. seealso:: The :func:`~clusterer` function.

    """
    if not isinstance(views, list):
        raise TypeError("Parameter 'views' must be a list of Views. You provided %s" % type(views))

    return Ensemble(views, fuser, mincluster, mincluster_n).execute()



def get_ensemble(views: List[View], fuser: Fusion, clusterers: List[Clusterer]) -> Ensemble:
    """
    Creates and returns an :class:`~pyrea.structure.Ensemble` object which must
    be executed later to get the ensemble's computed view.
    """
    if not isinstance(views, list):
        raise TypeError("Parameter 'views' must be a list of Views. You provided %s" % type(views))

    return Ensemble(views, fuser, clusterers)

def consensus(labels: list):
    """
    Strict consensus: points share a cluster only if they co-occur in every labeling.
    """
    if len(labels) <= 1:
        raise ValueError("You must provide a list of labellings of length >= 2.")
    # Stack into (C, N) and transpose to (N, C)
    label_mat = np.vstack(labels).T
    # Map each unique tuple to a new cluster ID
    mapper = {}
    cl_cons = np.zeros(label_mat.shape[0], dtype=int)
    next_id = 1
    for i, row in enumerate(label_mat):
        key = tuple(row)
        if key not in mapper:
            mapper[key] = next_id
            next_id += 1
        cl_cons[i] = mapper[key]
    return cl_cons



def parea_2_mv(
    data: list,
    k_s: list,
    k_final=None,
    linkage= 'average',
    pre_linkage= 'average',
    fusion_method='agreement',
    fitness=False,
    subject_id_list=None,
    inner_jobs: int = 1,
    pre_inner_jobs: int = 1,
    mincluster="FALSE",
    mincluster_n=10,
    allow_null_view: bool = True,
    drop_null_views: bool = True,
    null_quality_neutral: float = 0.5,
    internal_ensemble_enabled=False,
    internal_ensemble_bcs=5,
    internal_ensemble_sample_frac=0.8,
    internal_ensemble_feature_frac=1.0,
    internal_ensemble_seed=0
):
    # Sanity checks and linkage broadcasting
    """Handle parea 2 mv."""
    if isinstance(linkage, str):
        linkage = [linkage] * len(data)
    if len(data) != len(k_s) or len(data) != len(linkage):
        raise ValueError("The number of views, k_s and linkages must match.")
    if len(data) == 0:
        raise RuntimeError("No data passed to parea_2_mv! Something went wrong upstream.")

    n_views = len(data)

    # --- Enforce a hard upper bound on k when a minimum cluster size is requested ---
    # If you require clusters of at least `mincluster_n`, you cannot have more than
    # floor(N / mincluster_n) clusters (otherwise at least one cluster must be too small).
    def _mincluster_enabled(x):
        """Handle mincluster enabled."""
        if isinstance(x, str):
            return x.strip().upper() == "TRUE"
        return bool(x)

    if _mincluster_enabled(mincluster):
        # Infer N from the first view (all views should be aligned to the same subjects)
        N = data[0].shape[0] if hasattr(data[0], "shape") else len(data[0])
        max_k_allowed = max(1, int(N // int(mincluster_n)))
        # Clamp per-view k's
        k_s = [min(int(k), max_k_allowed) for k in k_s]
        # Clamp final k
        if k_final is not None:
            k_final = min(int(k_final), max_k_allowed)

    from parea_classes import NullClusteringPyrea

    clustering_algorithms = []
    views = []
    null_mask = []
    for i, d in enumerate(data):
        # Degenerate view skip
        if hasattr(d, "shape") and (d.shape[0] < 3 or np.allclose(np.var(d, axis=0), 0)):
            warnings.warn(f"Skipping degenerate view {i}")
            continue
        if allow_null_view and k_s[i] == 1:
            if drop_null_views:
                null_mask.append(True)
                continue  # skip this view entirely
            else:
                clustering_algorithms.append(NullClusteringPyrea())
        else:
            clustering_algorithms.append(clusterer(
                'ensemble',
                n_clusters=int(k_s[i]),
                precomputed=False,
                linkage_method=linkage[i],
                random_state=None,
                final=False,
                internal_ensemble_enabled=internal_ensemble_enabled,
                internal_ensemble_bcs=internal_ensemble_bcs,
                internal_ensemble_sample_frac=internal_ensemble_sample_frac,
                internal_ensemble_feature_frac=internal_ensemble_feature_frac,
                internal_ensemble_seed=int(internal_ensemble_seed) + 1009 * (i + 1)
            ))
        views.append(view(d, clustering_algorithms[-1]))
        null_mask.append(False)
    #Compute the individual clustering labels from each view (before fusion)
    #individual_labels = [v.execute() for v in views] #TODO I think this is not needed. Does this within the execute_ensemble.

    # ==== Align and impute missing labels across views ====
    # [Imputation and label alignment logic may be required here, but fusion will always use execute_ensemble on views.]
    # Create fusion algorithm
    f = fuser(fusion_method)

    # Compute fusion matrix by executing the ensemble of views directly
    fusion_matrix, individual_labels = execute_ensemble(views, f, mincluster, mincluster_n)
    # --- Align individual_labels back to original number of views ---
    # When drop_null_views=True and some k_s[i]==1, execute_ensemble returns fewer
    # per-view label arrays. We expand back to length n_views, inserting a
    # single-cluster (all zeros) label vector for dropped views so downstream
    # code can index by original modality index without errors.
    aligned_labels = []
    used_view_idx = 0
    for i in range(n_views):
        if drop_null_views and allow_null_view and k_s[i] == 1:
            n_i = data[i].shape[0] if hasattr(data[i], 'shape') else len(data[i])
            aligned_labels.append(np.zeros(n_i, dtype=int))
        else:
            lbl = np.asarray(individual_labels[used_view_idx]).ravel()
            aligned_labels.append(lbl)
            used_view_idx += 1
    individual_labels = aligned_labels


    ## TODO: Check if we can delete this part and just use the fusion_matrix.
    # --- Ensure the fused matrix is a distance matrix for downstream steps ---
    S = np.asarray(fusion_matrix, dtype=float)
    S = np.nan_to_num(S, nan=0.0, posinf=1.0, neginf=0.0)
    diag = np.diag(S)
    looks_like_similarity = np.nanmax(np.abs(diag - 1.0)) < 0.05 and S.max() <= 1.5
    if looks_like_similarity:
        # Clamp to [0,1] then convert similarity -> distance
        S = np.clip(S, 0.0, 1.0)
        v_res = 1.0 - S
        np.fill_diagonal(v_res, 0.0)
    else:
        # Assume it's already a distance matrix
        v_res = S

    # === Second step: use the SAME ensemble method as step 1, with fixed k ===
    # After we have the fused matrix (v_res), apply the ensemble clusterer directly
    # to v_res with a predefined k.

    if not k_final:
        raise ValueError(
            "k_final must be provided for the second-step ensemble. "
            "Pass an explicit number of clusters to apply the predefined ensemble."
        )

    # Final clustering: use ensemble on the fused **distance** matrix
    c_final = clusterer(
        'ensemble',
        precomputed=True,
        n_clusters=int(k_final),
        linkage_method=pre_linkage if pre_linkage else 'average',
        internal_ensemble_enabled=internal_ensemble_enabled,
        internal_ensemble_bcs=internal_ensemble_bcs,
        internal_ensemble_sample_frac=internal_ensemble_sample_frac,
        internal_ensemble_feature_frac=1.0,
        internal_ensemble_seed=int(internal_ensemble_seed) + 7919
    )

    # Validate fused matrix before creating final view
    if not np.all(np.isfinite(v_res)) or np.any(v_res < 0):
        warnings.warn("Invalid distances detected in fused matrix; replacing with zeros.")
        v_res = np.nan_to_num(v_res, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(v_res, 0.0)
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

    # --- Enforce minimum cluster size on final fused labels if requested ---
    if mincluster=="TRUE" or mincluster is True:
        print("Enforcing minimum cluster size of", mincluster_n)
        final_labels = enforce_min_cluster_size(v_res, final_labels, min_size=mincluster_n)


    # --- Quality metrics: composite indices for views and final ---
    # One-cluster solutions are treated as degenerate during GA optimisation.
    def _silhouette_norm(mat, labels, precomputed=False):
        """Handle silhouette norm."""
        labels = np.asarray(labels)
        if len(np.unique(labels)) <= 1:
            result = 0.0
        else:
            try:
                if precomputed:
                    sil = silhouette_score(mat, labels, metric='precomputed')
                else:
                    sil = silhouette_score(mat, labels)
                result = (sil + 1.0) / 2.0  # [-1,1] -> [0,1]
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
                ch = calinski_harabasz_score(X, labels)   # [0, +inf)
                result = ch / (ch + 1.0)             # (0,1)
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
                db = davies_bouldin_score(X, labels)      # lower is better
                result = 1.0 / (1.0 + db)            # (0,1]
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
        # Deterministic classical MDS from distance matrix D (NxN)
        """Handle classical mds."""
        D = np.asarray(D, dtype=float)
        n = D.shape[0]
        if n == 0:
            return np.zeros((0, 0), dtype=float)
        # Double centering: B = -1/2 * J D^2 J
        J = np.eye(n) - np.ones((n, n)) / n
        D2 = D ** 2
        B = -0.5 * J.dot(D2).dot(J)
        # Eigen-decompose; keep top positive components
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

    # Per-view composite qualities and their mean.
    # One-cluster views receive quality 0 during optimisation.
    view_scores_per_view = []
    for i in range(n_views):
        labs = np.asarray(individual_labels[i]).ravel()
        view_scores_per_view.append(float(_composite_view_quality(data[i], labs)))
    view_score = float(np.mean(view_scores_per_view)) if len(view_scores_per_view) else 0.0

    # Final composite quality. One-cluster final solutions receive quality 0
    # during optimisation; no-cluster plausibility is assessed post hoc.
    k_final_unique = len(np.unique(final_labels))
    if k_final_unique <= 1:
        final_score = 0.0
    else:
        sil_final = _silhouette_norm(v_res, final_labels, precomputed=True)
        X_mds = _classical_mds(v_res, p=min(10, v_res.shape[0]-1))
        ch_final = _ch_norm(X_mds, final_labels)
        dbi_final = _db_inv(X_mds, final_labels)
        final_score = float((sil_final + ch_final + dbi_final) / 3.0)

    if fitness:
        if len(np.unique(final_labels)) == 1:
            return 0.0
        else:
            return final_labels, individual_labels, view_scores_per_view, view_score, final_score
    else:
        return final_labels, individual_labels, view_scores_per_view, view_score, final_score




from collections import defaultdict

def convert_to_parameters(data_len, individual):
    """
    Convert a DEAP individual with gene_names metadata into the parameters
    dict required by parea_2_mv.

    Expected genes per view (i = 1..data_len):
      - c_{i}_k           -> number of clusters for view i
      - c_{i}_method      -> linkage for view i (e.g., 'average', 'ward', ...)

    Global genes:
      - pre_method        -> linkage used in the final fused step
      - k_final           -> number of clusters for the final ensemble step
      - fusion_method     -> fusion method string

    Backward compatibility:
      - If per-view linkage genes are missing, we do not fail; we will return
        no `linkage` key and let parea_2_mv handle defaults/broadcasting.
      - If no pre_method (or legacy c_{i}_pre_method) genes exist, `pre_linkage` is omitted.
    """
    if not hasattr(individual, 'gene_names'):
        raise AttributeError(
            "Individual missing gene_names attribute; please initialize your population with gene_names metadata."
        )

    grouped = defaultdict(list)
    for name, val in zip(individual.gene_names, individual):
        grouped[name].append(val)

    # Required per-view k's
    k_s = [int(grouped[f"c_{i+1}_k"][0]) for i in range(data_len)]

    # Optional: per-view linkage methods
    linkage = []
    have_linkage = True
    for i in range(data_len):
        key = f"c_{i+1}_method"
        if key in grouped:
            linkage.append(str(grouped[key][0]))
        else:
            have_linkage = False
            break

    # Optional: choose a single pre_linkage for the final step
    pre_linkage = None
    if "pre_method" in grouped:
        pre_linkage = str(grouped["pre_method"][0])
    else:
        for i in range(data_len):
            key = f"c_{i+1}_pre_method"
            if key in grouped:
                pre_linkage = str(grouped[key][0])
                break  # take the first one deterministically

    # Globals
    k_final = int(grouped["k_final"][0]) if "k_final" in grouped else None
    fusion_method = str(grouped["fusion_method"][0]) if "fusion_method" in grouped else 'agreement'

    params = {
        "k_s": k_s,
        "k_final": k_final,
        "fusion_method": fusion_method,
    }
    if have_linkage and len(linkage) == data_len:
        params["linkage"] = linkage
    if pre_linkage is not None:
        params["pre_linkage"] = pre_linkage

    return params


