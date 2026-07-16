#!/bin/bash
# Submit only the final merge/post-hoc stage.
#
# Use this after the fold-level pipeline has already completed and
# results/fold*/metrics.pkl files exist. It does not rerun GA generations or
# outer fold bootstraps.

set -euo pipefail

if [ -z "${SERVER:-}" ]; then
  hn=$(hostname -f 2>/dev/null || hostname)
  case "$hn" in
    *marvin*|*hpc.uni-bonn.de*) SERVER="marvin" ;;
    *spartan*|*unimelb*|*melbourne*) SERVER="spartan" ;;
    *) SERVER="marvin" ;;
  esac
fi
export SERVER
if [ "${SERVER}" = "marvin" ]; then
  module load Apptainer 2>/dev/null || module load apptainer 2>/dev/null || true
  if [ -z "${BASE_DIR:-}" ]; then
    case "${RUN_PROFILE:-}" in
      prospect|run_profiles/prospect.sh|*/run_profiles/prospect.sh) BASE_DIR="/home/s45jmeij_hpc/prospect_clust" ;;
      *) BASE_DIR="/home/s45jmeij_hpc/multiclust" ;;
    esac
  fi
  export BASE_DIR
else
  export BASE_DIR=${BASE_DIR:-path/to/multiclust}
fi
cd "${BASE_DIR}"

if [ -n "${RUN_PROFILE:-}" ]; then
  if [ -f "${RUN_PROFILE}" ]; then
    echo "Loading run profile: ${RUN_PROFILE}"
    # shellcheck disable=SC1090
    source "${RUN_PROFILE}"
  elif [ -f "run_profiles/${RUN_PROFILE}.sh" ]; then
    echo "Loading run profile: run_profiles/${RUN_PROFILE}.sh"
    # shellcheck disable=SC1090
    source "run_profiles/${RUN_PROFILE}.sh"
  else
    echo "ERROR: RUN_PROFILE '${RUN_PROFILE}' was not found as a file or run_profiles/${RUN_PROFILE}.sh" >&2
    exit 2
  fi
fi

export SIF=${SIF:-multiview_env.sif}
export INPUT_CSV=${INPUT_CSV:-cleaned_discovery_data.csv}
export META_CSV=${META_CSV:-merged_meta.csv}

export N_FOLDS=${N_FOLDS:-5}
export COL_THRESHOLD=${COL_THRESHOLD:-0.5}
export ROW_THRESHOLD=${ROW_THRESHOLD:-0.5}
export SKEW_THRESHOLD=${SKEW_THRESHOLD:-0.75}
export SCALER_TYPE=${SCALER_TYPE:-robust}
export MODALITIES=${MODALITIES:-"Internalising Functioning Detachment Psychoticism Cognition"}
export DUMMY_CODE_MODALITIES=${DUMMY_CODE_MODALITIES:-}
export MIXED_CATEGORICAL_MODALITIES=${MIXED_CATEGORICAL_MODALITIES:-}
export DIMREDUCTION=${DIMREDUCTION:-None}
export DIM_REDUCTION_BY_MODALITY=${DIM_REDUCTION_BY_MODALITY:-}

export HIDDEN_DIMS=${HIDDEN_DIMS:-"100 250 500 1000"}
export ACTIVATION_FUNCTIONS=${ACTIVATION_FUNCTIONS:-"LeakyReLU selu swish"}
export LEARNING_RATES=${LEARNING_RATES:-"0.001 0.0001"}
export BATCH_SIZES=${BATCH_SIZES:-"32 64 128"}
export LATENT_DIMS=${LATENT_DIMS:-"2 5 10 20"}

export OPTIMISATION=${OPTIMISATION:-multi}
export K_MAX=${K_MAX:-10}
export CLUSTER_LINKAGES=${CLUSTER_LINKAGES:-"average complete weighted"}
export FUSION_METHODS=${FUSION_METHODS:-agreement}
export GA_OBJECTIVES=${GA_OBJECTIVES:-"mean_view_stability_ari mean_view_quality final_stability_ari final_quality"}

export N_BOOTSTRAP=${N_BOOTSTRAP:-100}
export DO_SVM=${DO_SVM:-TRUE}
export MINCLUSTER=${MINCLUSTER:-TRUE}
export MINCLUSTER_N=${MINCLUSTER_N:-50}
export MINCLUSTER_RESAMPLE_MODE=${MINCLUSTER_RESAMPLE_MODE:-fixed}
export USE_EFFECTIVE_K_FOR_FOLD_MERGE=${USE_EFFECTIVE_K_FOR_FOLD_MERGE:-FALSE}
export USE_CROSS_FOLD_EFFECTIVE_K_FOR_FINAL_RUN=${USE_CROSS_FOLD_EFFECTIVE_K_FOR_FINAL_RUN:-FALSE}
export TEST=${TEST:-FALSE}

export FINAL_BOOTSTRAP_JOBS=${FINAL_BOOTSTRAP_JOBS:-}
export FINAL_BOOTSTRAP_PREPROCESSING=${FINAL_BOOTSTRAP_PREPROCESSING:-outside}

export COMPUTE_CLUSTER_PVALUES=${COMPUTE_CLUSTER_PVALUES:-TRUE}
export CLUSTER_PVALUE_MODE=${CLUSTER_PVALUE_MODE:-fast}
export CLUSTER_PVALUE_STAT=${CLUSTER_PVALUE_STAT:-composite}
export CLUSTER_PVALUE_PERMUTATIONS=${CLUSTER_PVALUE_PERMUTATIONS:-1000}
export CLUSTER_PVALUE_PERMUTATIONS_QUALITY=${CLUSTER_PVALUE_PERMUTATIONS_QUALITY:-1000}
export CLUSTER_PVALUE_PERMUTATIONS_ARI=${CLUSTER_PVALUE_PERMUTATIONS_ARI:-500}
export CLUSTER_PVALUE_JOBS=${CLUSTER_PVALUE_JOBS:-0}
export CLUSTER_PVALUE_SEED=${CLUSTER_PVALUE_SEED:-314159}

export DO_CLUSTER_VALIDATION_SENSITIVITY=${DO_CLUSTER_VALIDATION_SENSITIVITY:-FALSE}
export CLUSTER_VALIDATION_GAUSSIAN_NULLS=${CLUSTER_VALIDATION_GAUSSIAN_NULLS:-100}
export CLUSTER_VALIDATION_GAP_REFS=${CLUSTER_VALIDATION_GAP_REFS:-100}
export CLUSTER_VALIDATION_SIGCLUST_SIMS=${CLUSTER_VALIDATION_SIGCLUST_SIMS:-200}
export CLUSTER_VALIDATION_STABILITY_BOOTSTRAPS=${CLUSTER_VALIDATION_STABILITY_BOOTSTRAPS:-30}
export CLUSTER_VALIDATION_JOBS=${CLUSTER_VALIDATION_JOBS:-0}
export SUBJECT_ID_COLUMN=${SUBJECT_ID_COLUMN:-src_subject_id}

mkdir -p logs

if [ ! -f "${INPUT_CSV}" ]; then
  echo "ERROR: input CSV not found: ${BASE_DIR}/${INPUT_CSV}" >&2
  exit 2
fi
if [ ! -f "${META_CSV}" ]; then
  echo "ERROR: meta CSV not found: ${BASE_DIR}/${META_CSV}" >&2
  exit 2
fi
if [ ! -f "${SIF}" ] && [ ! -f "${BASE_DIR}/${SIF}" ]; then
  echo "ERROR: Apptainer image not found: ${SIF}" >&2
  echo "Set SIF to a valid .sif path or place multiview_env.sif in ${BASE_DIR}" >&2
  exit 2
fi

if [ -d "${RESULTS_DIR}" ]; then
  metrics_count=$(find "${RESULTS_DIR}" -maxdepth 2 -path "${RESULTS_DIR}/fold*/metrics.pkl" -type f | wc -l | tr -d ' ')
else
  metrics_count=0
fi
if [ "${metrics_count:-0}" -lt "${N_FOLDS}" ]; then
  echo "ERROR: expected at least ${N_FOLDS} fold metrics files, found ${metrics_count:-0} under ${RESULTS_DIR}/fold*/metrics.pkl" >&2
  exit 2
fi

PARTITION_OPT=""
if [ "${SERVER}" = "marvin" ]; then
  FINAL_MERGE_PARTITION=${FINAL_MERGE_PARTITION:-intelsr_short}
  FINAL_MERGE_TIME=${FINAL_MERGE_TIME:-04:00:00}
  PARTITION_OPT="-p ${FINAL_MERGE_PARTITION} --time=${FINAL_MERGE_TIME}"
fi

sbatch ${PARTITION_OPT} --export=ALL \
  final_merge.sh
