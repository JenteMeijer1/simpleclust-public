# Purpose: Run the simpleclust task.
# SchizBull paper profile using the singleclust single-view grid-search runner.
# The executable stage flow is documented in docs/singleclust_pipeline_overview.md.
# This profile contains paper-specific configuration only; orchestration stays in run.sh.
export CLUSTER_PIPELINE="singleclust"
export INPUT_CSV="cleaned_discovery_data_simpleclust.csv"
export META_CSV="merged_meta_simpleclust.csv"
export NOTEBOOK="notebooks/Simpleclust/Main.ipynb"

# Singleclust uses one analysis domain. The shared preprocessing code still
# receives a modality name so feature handling stays aligned with multiclust.
export MODALITIES="Include_cluster"
export DUMMY_CODE_MODALITIES="Include_cluster"
export MIXED_CATEGORICAL_MODALITIES=""

# Primary SchizBull run uses the full preprocessed feature matrix. The override
# block is used by run_simpleclust_dimreduction_sequence.sh for sensitivity runs.
if [ "${ALLOW_PROFILE_DIMREDUCTION_OVERRIDE:-FALSE}" = "TRUE" ]; then
  export DIMREDUCTION="${SIMPLECLUST_DIMREDUCTION_OVERRIDE:-None}"
  export MAXPC="${SIMPLECLUST_MAXPC_OVERRIDE:-20}"
  export SPCA_ALPHA="${SIMPLECLUST_SPCA_ALPHA_OVERRIDE:-${SPCA_ALPHA:-1.0}}"
  export SNMF_ALPHA="${SIMPLECLUST_SNMF_ALPHA_OVERRIDE:-${SNMF_ALPHA:-0.1}}"
  export SNMF_L1_RATIO="${SIMPLECLUST_SNMF_L1_RATIO_OVERRIDE:-${SNMF_L1_RATIO:-1.0}}"
  export SPARSE_L1_LAMBDA="${SIMPLECLUST_SPARSE_L1_OVERRIDE:-${SPARSE_L1_LAMBDA:-1e-3}}"
else
  export DIMREDUCTION="None"
  export MAXPC=20
  export SPCA_ALPHA="${SPCA_ALPHA:-1.0}"
  export SNMF_ALPHA="${SNMF_ALPHA:-0.1}"
  export SNMF_L1_RATIO="${SNMF_L1_RATIO:-1.0}"
fi
export SPCA_RIDGE_ALPHA="${SPCA_RIDGE_ALPHA:-0.01}"
export SPCA_MAX_ITER="${SPCA_MAX_ITER:-1000}"
export SNMF_MAX_ITER="${SNMF_MAX_ITER:-1000}"

export N_FOLDS=5
# In singleclust this is a grid-size cap, not a GA population size. Zero keeps
# the complete linkage x k grid (3 x 9 = 27 candidates with the defaults).
export N_POPULATION=0
# Compatibility with the shared scheduler. The singleclust paper path evaluates
# the grid once rather than evolving candidates across generations.
export N_GENERATIONS=1
export N_BOOTSTRAP=100
export BOOTSTRAP_MODE="subsample"

# Internal ensemble creates perturbed base clusterings inside each candidate
# evaluation. SAMPLE_FRAC=0.8 resamples subjects; FEATURE_FRAC=1.0 keeps all
# features for the SchizBull paper profile.
export INTERNAL_ENSEMBLE_ENABLED="TRUE"
if [ "${ALLOW_PROFILE_INTERNAL_ENSEMBLE_BCS_OVERRIDE:-FALSE}" = "TRUE" ]; then
  export INTERNAL_ENSEMBLE_BCS=${INTERNAL_ENSEMBLE_BCS:-100}
else
  export INTERNAL_ENSEMBLE_BCS=100
fi
export INTERNAL_ENSEMBLE_SAMPLE_FRAC=0.8
export INTERNAL_ENSEMBLE_FEATURE_FRAC=1.0
if [ "${ALLOW_PROFILE_INTERNAL_ENSEMBLE_BCS_OVERRIDE:-FALSE}" = "TRUE" ]; then
  export FINAL_INTERNAL_ENSEMBLE_BCS=${FINAL_INTERNAL_ENSEMBLE_BCS:-100}
else
  export FINAL_INTERNAL_ENSEMBLE_BCS=100
fi
export MAX_CONCURRENT=200

# Keep the five-fold Marvin run below its 300-job submission limit while
# retaining near one-bootstrap-per-job scheduling. Spartan has separate defaults
# in run.sh and can be overridden at submission time.
export MARVIN_BOOTSTRAPS_PER_JOB=2
export MARVIN_BOOTSTRAP_PARALLEL_STEPS=2
export MARVIN_BOOTSTRAP_STEP_CPUS=16
export MARVIN_BOOTSTRAP_JOB_CPUS=32
export MARVIN_BOOTSTRAP_JOB_TIME="01:00:00"
export BOOTSTRAP_JOB_TIME="01:00:00"
export MARVIN_MAX_SUBMITTED_JOBS=300
export MARVIN_SUBMIT_JOB_BUDGET=300

export GA_OBJECTIVES="stab_ari quality"
export SEARCH_OBJECTIVES="${GA_OBJECTIVES}"
export MINCLUSTER="TRUE"
export MINCLUSTER_N=50
# fixed = enforce MINCLUSTER_N against each resample directly. This is the
# paper profile setting and makes the minimum-cluster provenance explicit in
# fold/final metrics.
export MINCLUSTER_RESAMPLE_MODE="fixed"
export USE_EFFECTIVE_K_FOR_FOLD_MERGE="FALSE"
export USE_CROSS_FOLD_EFFECTIVE_K_FOR_FINAL_RUN="FALSE"
export DO_SVM="TRUE"
export RUN_MERGE="TRUE"
export RUN_POSTPROCESS_REPORT="TRUE"
# outside = preprocess/dimensionality-reduce the full data once, then estimate
# final stability by subsampling that fixed representation.
export FINAL_BOOTSTRAP_PREPROCESSING="outside"
export DO_CLUSTER_VALIDATION_SENSITIVITY="TRUE"
