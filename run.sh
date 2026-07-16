#!/bin/bash
# Purpose: Submit the configured end-to-end clustering workflow.
#SBATCH --job-name=run        # run everything          
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00

 # Detect server if not explicitly set in the environment
if [ -z "${SERVER:-}" ]; then
  hn=$(hostname -f 2>/dev/null || hostname)
  case "$hn" in
    *marvin*|*hpc.uni-bonn.de*) SERVER="marvin" ;;
    *spartan*|*unimelb*|*melbourne*) SERVER="spartan" ;;
    *) SERVER="marvin" ;; # fallback
  esac
fi
export SERVER

# Partition selection (Marvin defaults to a 1h devel queue; use short for longer jobs)
export PARTITION_OPT=""
if [ "$SERVER" == "marvin" ]; then
  PARTITION_OPT="-p intelsr_short"
fi
if [ "$SERVER" == "spartan" ]; then
  module load Apptainer
  export BASE_DIR="${BASE_DIR:-path/to/multiclust}"
  cd "${BASE_DIR}"
elif [ "$SERVER" == "marvin" ]; then
  module load Apptainer 2>/dev/null || module load apptainer 2>/dev/null || true
  if [ -z "${BASE_DIR:-}" ]; then
    case "${RUN_PROFILE:-}" in
      prospect|run_profiles/prospect.sh|*/run_profiles/prospect.sh) BASE_DIR="/home/s45jmeij_hpc/prospect_clust" ;;
      *) BASE_DIR="/home/s45jmeij_hpc/multiclust" ;;
    esac
  fi
  export BASE_DIR
  cd "${BASE_DIR}"
else
  export BASE_DIR="${BASE_DIR:-$(pwd)}"
fi
export RESULTS_DIR="${RESULTS_DIR:-${BASE_DIR}/results}"
export INTERMEDIATES_DIR="${INTERMEDIATES_DIR:-${BASE_DIR}/intermediates}"
export LOGS_DIR="${LOGS_DIR:-${BASE_DIR}/logs}"
export PLOTS_DIR="${PLOTS_DIR:-${BASE_DIR}/plots}"
mkdir -p "${LOGS_DIR}"


# Apptainer image with all dependencies
export SIF=${SIF:-multiview_env.sif}
export DEF_FILE=${DEF_FILE:-multiview_env.def}
export REQUIREMENTS_FILE=${REQUIREMENTS_FILE:-requirements_multiview_env.txt}
export CHECK_REQUIREMENTS_SCRIPT=${CHECK_REQUIREMENTS_SCRIPT:-check_container_requirements.py}
export REBUILD_SIF_IF_NEEDED=${REBUILD_SIF_IF_NEEDED:-TRUE}
export APPTAINER_BUILD_ARGS=${APPTAINER_BUILD_ARGS:-}

if [ ! -f "${SIF}" ] && [ -f "${BASE_DIR}/${SIF}" ]; then
  export SIF="${BASE_DIR}/${SIF}"
fi

ensure_sif_requirements() {
  local reason=""

  if [ ! -f "${DEF_FILE}" ]; then
    echo "ERROR: Apptainer definition not found: ${DEF_FILE}" >&2
    exit 2
  fi
  if [ ! -f "${REQUIREMENTS_FILE}" ]; then
    echo "ERROR: requirements file not found: ${REQUIREMENTS_FILE}" >&2
    exit 2
  fi
  if [ ! -f "${CHECK_REQUIREMENTS_SCRIPT}" ]; then
    echo "ERROR: requirements check script not found: ${CHECK_REQUIREMENTS_SCRIPT}" >&2
    exit 2
  fi

  if [ ! -f "${SIF}" ]; then
    reason="image missing"
  else
    echo "Checking Apptainer image requirements: ${SIF}"
    if ! apptainer exec "${SIF}" python "${CHECK_REQUIREMENTS_SCRIPT}" "${REQUIREMENTS_FILE}"; then
      reason="requirements mismatch"
    fi
  fi

  if [ -n "${reason}" ]; then
    if [ "${REBUILD_SIF_IF_NEEDED}" != "TRUE" ]; then
      echo "ERROR: Apptainer image ${SIF} needs rebuild (${reason}), but REBUILD_SIF_IF_NEEDED=${REBUILD_SIF_IF_NEEDED}." >&2
      exit 2
    fi
    echo "Rebuilding Apptainer image ${SIF} from ${DEF_FILE} because: ${reason}"
    # shellcheck disable=SC2086
    apptainer build --force ${APPTAINER_BUILD_ARGS} "${SIF}" "${DEF_FILE}"
    echo "Rechecking rebuilt Apptainer image requirements: ${SIF}"
    apptainer exec "${SIF}" python "${CHECK_REQUIREMENTS_SCRIPT}" "${REQUIREMENTS_FILE}"
  fi
}

ensure_sif_requirements
if [ ! -f "${SIF}" ]; then
  echo "ERROR: Apptainer image not found after rebuild check: ${SIF}" >&2
  exit 2
fi
echo "Using Apptainer image: ${SIF}"
# Import data
#export INPUT_CSV="synthetic_multimodal_spartan.csv" #For synthetic data in testing
export INPUT_CSV="prospect_data.csv" #actual data
#export META_CSV="synthetic_multimodal_spartan_meta.csv" #For synthetic data in testing
export META_CSV="prospect_meta.csv" #Actual data

####### Pipeline parameters ########

# Number of outer CV folds
export N_FOLDS=5
# Threshold for removing columns with too many missing values
export COL_THRESHOLD=0.5 
# Threshold for removing rows with too many missing values
export ROW_THRESHOLD=0.5 
# Threshold for when to log-transform a variable based on its skewness
export SKEW_THRESHOLD=0.75 
# Type of scaler to use: "standard", "minmax", "robust"
export SCALER_TYPE="robust" 
# Actual modalities in the data
export MODALITIES="Metabolic_Risk Blood_Markers Substance_Use Suicidality Injury Physical_health" 
# Modalities whose raw variables should be dummy/ordinal encoded during preprocessing.
# Imaging/EEG modalities are left in their original form.
export DUMMY_CODE_MODALITIES="Metabolic_Risk Blood_Markers Injury Substance_Use"
# Modalities to keep as mixed categorical/binary/numeric during preprocessing.
# Use FAMD, MCA, or MIXED_SVD for these in DIM_REDUCTION_BY_MODALITY.
export MIXED_CATEGORICAL_MODALITIES="Physical_health Suicidality"
#export MODALITIES="m1 m2 m3 m4" # For synthetic test
# Selection of dimensionality reduction method:urrently supported: None, VAE, PCA, AE
export DIMREDUCTION="None" 
# Optional per-modality overrides, e.g. "Internalising=PCA Psychoticism=None". For categorical modalties, must add one of: FAMD, MCA, or MIXED_SVD. 
export DIM_REDUCTION_BY_MODALITY=""
# For PCA modalities, optionally retain the smallest number of PCs reaching this explained-variance fraction.
# Leave empty to use the legacy PCA cap of up to 50 components.
export PCA_VARIANCE_THRESHOLD=""
export HIDDEN_DIMS="100 250 500 1000" 
# Activation functions to try in VAE
export ACTIVATION_FUNCTIONS="LeakyReLU selu swish" 
# Learning rates to try in VAE
export LEARNING_RATES="0.001 0.0001"
# Batch sizes to try in VAE
export BATCH_SIZES="32 64 128" 
# Latent dimensions to try in VAE
export LATENT_DIMS="2 5 10 20" 
# Optimisation objective: "single" or "multi" (multi-objective). Multi = optimising both cluster quality and stability for each modality and final clusters.
export OPTIMISATION="multi" 
# Number of hyperparameter combinations in each generation
export N_POPULATION=100
#Number of GA generations
export N_GENERATIONS=10
# Minimum number of clusters tested (for both individual modalities and final clusters)
export K_MIN=2 
# Maximum number of clusters tested (for both individual modalities and final clusters)
export K_MAX=10 
# Which linkages to use in clustering (options: single, complete, average, ward, weighted, default=average). Multiple are allowed for GA optimisation
export CLUSTER_LINKAGES="average complete weighted"
# Space of clustering. Euclidian is default but for ordinal or categorical data, other metrics can be used. Currently supported: Euclidean, 
# Number of bootstraps per generation in which stability is tested
export N_BOOTSTRAP=100
# Bootstrap modes. Options: 'bootstrap' (with replacement) or 'subsample' (without replacement).
export BOOTSTRAP_MODE='subsample' 
# Internal ensemble variance. FALSE preserves the original fixed five-method ensemble.
# When TRUE, each ensemble is built from balanced perturbed base clusterings:
# methods are rotated evenly, and each base clustering samples subjects/features.
export INTERNAL_ENSEMBLE_ENABLED=${INTERNAL_ENSEMBLE_ENABLED:-"FALSE"}
export INTERNAL_ENSEMBLE_BCS=${INTERNAL_ENSEMBLE_BCS:-100}
export INTERNAL_ENSEMBLE_SAMPLE_FRAC=${INTERNAL_ENSEMBLE_SAMPLE_FRAC:-0.8}
export INTERNAL_ENSEMBLE_FEATURE_FRAC=${INTERNAL_ENSEMBLE_FEATURE_FRAC:-1.0}
# Optional stronger setting for final_merge.sh; defaults to INTERNAL_ENSEMBLE_BCS there.
export FINAL_INTERNAL_ENSEMBLE_BCS=${FINAL_INTERNAL_ENSEMBLE_BCS:-}
# Maximum number of concurrent bootstrap jobs running on server
export MAX_CONCURRENT=200
# Whether to run SVM classification on the final clustering labels in OUTER mode (TRUE/FALSE)
export DO_SVM="TRUE" 
# Whether to enforce a minimum cluster size of 10 in the final clustering step.
export MINCLUSTER="TRUE" 
# Minimum cluster size when enforcing minimum cluster size
export MINCLUSTER_N=50
export MINCLUSTER_RESAMPLE_MODE="${MINCLUSTER_RESAMPLE_MODE:-fixed}"
export USE_EFFECTIVE_K_FOR_FOLD_MERGE="${USE_EFFECTIVE_K_FOR_FOLD_MERGE:-FALSE}"
export USE_CROSS_FOLD_EFFECTIVE_K_FOR_FINAL_RUN="${USE_CROSS_FOLD_EFFECTIVE_K_FOR_FINAL_RUN:-FALSE}"
# Objectives optimised by the GA (order matters). Allowed tokens defined in full_pipeline.py. An example of different options are: min or mean. Example: "avg_view_stability avg_view_quality final_stability final_quality" 
# For the stability measures you need to add which stability metric you want to use. So after stability, you can add: _jaccard, _coassoc, _ccc or _ari
export GA_OBJECTIVES="mean_view_stability_ari mean_view_quality final_stability_ari final_quality"
# Fusion methods allowed during GA search / mutation. Current options: agreement, disagreement, consensus (which is a strict consensus)
export FUSION_METHODS="agreement"
# Set to "TRUE" for testing mode (tests against true labels); "FALSE" for full run
export TEST="FALSE" 
# Pipeline implementation. Use "singleclust" for the SchizBull single-view grid runner.
export CLUSTER_PIPELINE="${CLUSTER_PIPELINE:-multiview}"
#Crossover rate in GA
export GA_CXPB=0.7 
#Mutation rate in GA
export GA_MUTPB=0.3 
#Number of elite individuals to keep in next generation in GA.
export GA_ELITISM=2 
# Wether to run the final merge or not 
export RUN_MERGE="TRUE"
# Whether to create a PDF report from the post-processing cells in the selected project notebook after the final merge.
# The report runner reads NOTEBOOK at runtime, so notebook changes are picked up automatically.
export RUN_POSTPROCESS_REPORT="TRUE"
export REPORT_PDF="${REPORT_PDF:-${PLOTS_DIR}/postprocess_report.pdf}"
export NOTEBOOK="${NOTEBOOK:-notebooks/multiclust_extended/Main.ipynb}"
# Compute permutation-based p-values for modality and integrated cluster solutions in merge mode
export COMPUTE_CLUSTER_PVALUES="TRUE"
# Permutation mode for p-values: "fast" keeps labels fixed, "full" reclusters each permutation
export CLUSTER_PVALUE_MODE="fast"
# Statistic used in permutation test: "composite" or "silhouette"
export CLUSTER_PVALUE_STAT="composite"
# Number of permutations for p-value estimation
export CLUSTER_PVALUE_PERMUTATIONS=1000
# Optional: separate permutation counts (defaults to CLUSTER_PVALUE_PERMUTATIONS when unset)
export CLUSTER_PVALUE_PERMUTATIONS_QUALITY=1000
export CLUSTER_PVALUE_PERMUTATIONS_ARI=500
# Parallel workers for permutation test (0 uses --n_jobs from merge script)
export CLUSTER_PVALUE_JOBS=0
# Base RNG seed for permutation test
export CLUSTER_PVALUE_SEED=314159
# Final stability bootstrap preprocessing mode.
# Options:
#   outside = preprocess/dimensionality-reduce once on the full data, then bootstrap fixed representations
#   inside  = rerun preprocessing and dimensionality reduction inside each final bootstrap
#   both    = report both outside and inside stability scores; inside is used as the primary final assessment
export FINAL_BOOTSTRAP_PREPROCESSING="outside"
# Optional cap on concurrent final stability bootstraps. Empty uses --n_jobs from final_merge.sh.
export FINAL_BOOTSTRAP_JOBS=""
# Optional post-hoc no-cluster / continuum validation sensitivity analyses.
export DO_CLUSTER_VALIDATION_SENSITIVITY="FALSE"
export CLUSTER_VALIDATION_GAUSSIAN_NULLS=100
export CLUSTER_VALIDATION_GAP_REFS=100
export CLUSTER_VALIDATION_SIGCLUST_SIMS=200
export CLUSTER_VALIDATION_STABILITY_BOOTSTRAPS=30
export CLUSTER_VALIDATION_JOBS=${CLUSTER_VALIDATION_JOBS:-0}

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

case "${CLUSTER_PIPELINE}" in
  multiview|singleclust) ;;
  *)
    echo "ERROR: unsupported CLUSTER_PIPELINE '${CLUSTER_PIPELINE}'. Use 'multiview' or 'singleclust'." >&2
    exit 2
    ;;
esac

# Optional resume controls (0-indexed fold indices). Define them before Marvin
# submission estimates so fold-sliced submissions are budgeted correctly.
RESUME_FROM_FOLD=${RESUME_FROM_FOLD:-0}
RESUME_TO_FOLD=${RESUME_TO_FOLD:-$((N_FOLDS-1))}
export RESUME_FROM_FOLD RESUME_TO_FOLD
SUBMIT_FOLD_COUNT=$((RESUME_TO_FOLD - RESUME_FROM_FOLD + 1))
if [ "${SUBMIT_FOLD_COUNT}" -lt 0 ]; then
  SUBMIT_FOLD_COUNT=0
fi

bootstrap_minutes_for_bcs() {
  local bcs="${1:-5}"
  if [ "${bcs}" -le 15 ]; then
    echo 10
  elif [ "${bcs}" -le 50 ]; then
    echo 15
  elif [ "${bcs}" -le 100 ]; then
    echo 25
  else
    echo $(( (bcs + 3) / 4 ))
  fi
}

uses_slow_dimreduction() {
  local dim_tokens="${DIMREDUCTION:-} ${DIM_REDUCTION_BY_MODALITY:-}"
  case " ${dim_tokens} " in
    *" VAE "*|*" AE "*|*" AUTOENCODER "*|*" AutoEncoder "*|*" autoencoder "*) return 0 ;;
    *) return 1 ;;
  esac
}

format_minutes_as_timelimit() {
  local minutes="$1"
  printf "%02d:%02d:00" $((minutes / 60)) $((minutes % 60))
}

bootstrap_time_for_request() {
  local bcs="${1:-5}"
  local bootstraps_per_job="${2:-1}"
  local parallel_steps="${3:-1}"
  local per_bootstrap_minutes waves total_minutes
  per_bootstrap_minutes=$(bootstrap_minutes_for_bcs "${bcs}")
  if uses_slow_dimreduction; then
    per_bootstrap_minutes=$((per_bootstrap_minutes + 30))
  fi
  if [ "${parallel_steps}" -lt 1 ]; then
    parallel_steps=1
  fi
  waves=$(( (bootstraps_per_job + parallel_steps - 1) / parallel_steps ))
  total_minutes=$(( per_bootstrap_minutes * waves ))
  if [ "${waves}" -gt 1 ]; then
    total_minutes=$(( total_minutes + 5 ))
  fi
  format_minutes_as_timelimit "${total_minutes}"
}

if [ "${SERVER}" == "marvin" ]; then
  export MAX_CONCURRENT=${MARVIN_MAX_CONCURRENT:-${MAX_CONCURRENT:-200}}
  if [ "${MAX_CONCURRENT}" -gt "${MARVIN_MAX_CONCURRENT_LIMIT:-200}" ]; then
    echo "Marvin safety cap: reducing MAX_CONCURRENT from ${MAX_CONCURRENT} to ${MARVIN_MAX_CONCURRENT_LIMIT:-200}."
    export MAX_CONCURRENT=${MARVIN_MAX_CONCURRENT_LIMIT:-200}
  fi
  export MARVIN_MAX_SUBMITTED_JOBS=${MARVIN_MAX_SUBMITTED_JOBS:-300}
  export MARVIN_SUBMIT_POLL_SECONDS=${MARVIN_SUBMIT_POLL_SECONDS:-30}
  export MARVIN_SERIALIZE_FOLDS=${MARVIN_SERIALIZE_FOLDS:-FALSE}
  export MARVIN_BOOTSTRAP_STAGGER_SECONDS=${MARVIN_BOOTSTRAP_STAGGER_SECONDS:-2}
  export MARVIN_BOOTSTRAP_STAGGER_SLOTS=${MARVIN_BOOTSTRAP_STAGGER_SLOTS:-10}
  export MARVIN_BOOTSTRAPS_PER_JOB=${MARVIN_BOOTSTRAPS_PER_JOB:-25}
  export MARVIN_BOOTSTRAP_PARALLEL_STEPS=${MARVIN_BOOTSTRAP_PARALLEL_STEPS:-4}
  export MARVIN_BOOTSTRAP_STEP_CPUS=${MARVIN_BOOTSTRAP_STEP_CPUS:-16}
  export MARVIN_BOOTSTRAP_JOB_CPUS=${MARVIN_BOOTSTRAP_JOB_CPUS:-$((MARVIN_BOOTSTRAP_PARALLEL_STEPS * MARVIN_BOOTSTRAP_STEP_CPUS))}
  export MARVIN_SUBMIT_JOB_BUDGET=${MARVIN_SUBMIT_JOB_BUDGET:-300}
  export BOOTSTRAPS_PER_JOB=${BOOTSTRAPS_PER_JOB:-${MARVIN_BOOTSTRAPS_PER_JOB}}
  export BOOTSTRAP_PARALLEL_STEPS=${BOOTSTRAP_PARALLEL_STEPS:-${MARVIN_BOOTSTRAP_PARALLEL_STEPS}}
  export BOOTSTRAP_STEP_CPUS_DEFAULT=${BOOTSTRAP_STEP_CPUS_DEFAULT:-${MARVIN_BOOTSTRAP_STEP_CPUS}}
  export BOOTSTRAP_JOB_CPUS=${BOOTSTRAP_JOB_CPUS:-${MARVIN_BOOTSTRAP_JOB_CPUS}}
  export MARVIN_BOOTSTRAP_JOB_TIME=${MARVIN_BOOTSTRAP_JOB_TIME:-$(bootstrap_time_for_request "${INTERNAL_ENSEMBLE_BCS:-5}" "${BOOTSTRAPS_PER_JOB}" "${BOOTSTRAP_PARALLEL_STEPS}")}
  export BOOTSTRAP_JOB_TIME=${BOOTSTRAP_JOB_TIME:-${MARVIN_BOOTSTRAP_JOB_TIME}}
  marvin_bootstrap_array_jobs=$(( (N_BOOTSTRAP + MARVIN_BOOTSTRAPS_PER_JOB - 1) / MARVIN_BOOTSTRAPS_PER_JOB ))
  marvin_submit_jobs_per_fold=$(( N_GENERATIONS * (marvin_bootstrap_array_jobs + 1) + 1 ))
  marvin_submit_jobs_final=0
  if [ "${RUN_MERGE}" == "TRUE" ]; then
    marvin_submit_jobs_final=1
  fi
  marvin_submit_jobs_report=0
  if [ "${RUN_POSTPROCESS_REPORT}" == "TRUE" ]; then
    marvin_submit_jobs_report=1
  fi
  marvin_submit_jobs_estimate=$(( SUBMIT_FOLD_COUNT * marvin_submit_jobs_per_fold + marvin_submit_jobs_final + marvin_submit_jobs_report ))
  echo "Marvin safety caps: MAX_CONCURRENT=${MAX_CONCURRENT}, max queued/running jobs=${MARVIN_MAX_SUBMITTED_JOBS}, serialize folds=${MARVIN_SERIALIZE_FOLDS}, bootstrap startup stagger=${MARVIN_BOOTSTRAP_STAGGER_SECONDS}s x ${MARVIN_BOOTSTRAP_STAGGER_SLOTS} slots, bootstrap batching=${MARVIN_BOOTSTRAPS_PER_JOB}/job with ${MARVIN_BOOTSTRAP_PARALLEL_STEPS} parallel step(s) at ${MARVIN_BOOTSTRAP_STEP_CPUS} CPU(s)/step, bootstrap time=${BOOTSTRAP_JOB_TIME}."
  echo "Marvin submission estimate: ${marvin_submit_jobs_estimate} submitted job record(s) for this run versus configured budget ${MARVIN_SUBMIT_JOB_BUDGET}."
  if [ "${marvin_submit_jobs_estimate}" -gt "${MARVIN_SUBMIT_JOB_BUDGET}" ]; then
    echo "ERROR: estimated Marvin submission footprint exceeds MARVIN_SUBMIT_JOB_BUDGET=${MARVIN_SUBMIT_JOB_BUDGET}." >&2
    echo "Use fewer folds per submission with RESUME_FROM_FOLD/RESUME_TO_FOLD, increase MARVIN_BOOTSTRAPS_PER_JOB if bootstrap array tasks dominate, or lower N_FOLDS/N_GENERATIONS." >&2
    exit 2
  fi
elif [ "${SERVER}" == "spartan" ]; then
  export SPARTAN_BOOTSTRAPS_PER_JOB=${SPARTAN_BOOTSTRAPS_PER_JOB:-1}
  export SPARTAN_BOOTSTRAP_PARALLEL_STEPS=${SPARTAN_BOOTSTRAP_PARALLEL_STEPS:-1}
  export SPARTAN_BOOTSTRAP_STEP_CPUS=${SPARTAN_BOOTSTRAP_STEP_CPUS:-16}
  export SPARTAN_BOOTSTRAP_JOB_CPUS=${SPARTAN_BOOTSTRAP_JOB_CPUS:-$((SPARTAN_BOOTSTRAP_PARALLEL_STEPS * SPARTAN_BOOTSTRAP_STEP_CPUS))}
  export SPARTAN_SERIALIZE_FOLDS=${SPARTAN_SERIALIZE_FOLDS:-FALSE}
  export BOOTSTRAPS_PER_JOB=${BOOTSTRAPS_PER_JOB:-${SPARTAN_BOOTSTRAPS_PER_JOB}}
  export BOOTSTRAP_PARALLEL_STEPS=${BOOTSTRAP_PARALLEL_STEPS:-${SPARTAN_BOOTSTRAP_PARALLEL_STEPS}}
  export BOOTSTRAP_STEP_CPUS_DEFAULT=${BOOTSTRAP_STEP_CPUS_DEFAULT:-${SPARTAN_BOOTSTRAP_STEP_CPUS}}
  export BOOTSTRAP_JOB_CPUS=${BOOTSTRAP_JOB_CPUS:-${SPARTAN_BOOTSTRAP_JOB_CPUS}}
  export SPARTAN_BOOTSTRAP_JOB_TIME=${SPARTAN_BOOTSTRAP_JOB_TIME:-$(bootstrap_time_for_request "${INTERNAL_ENSEMBLE_BCS:-5}" "${BOOTSTRAPS_PER_JOB}" "${BOOTSTRAP_PARALLEL_STEPS}")}
  export BOOTSTRAP_JOB_TIME=${BOOTSTRAP_JOB_TIME:-${SPARTAN_BOOTSTRAP_JOB_TIME}}
  spartan_bootstrap_array_jobs=$(( (N_BOOTSTRAP + BOOTSTRAPS_PER_JOB - 1) / BOOTSTRAPS_PER_JOB ))
  spartan_submit_jobs_per_fold=$(( N_GENERATIONS * (spartan_bootstrap_array_jobs + 1) + 1 ))
  spartan_submit_jobs_final=0
  if [ "${RUN_MERGE}" == "TRUE" ]; then
    spartan_submit_jobs_final=1
  fi
  spartan_submit_jobs_report=0
  if [ "${RUN_POSTPROCESS_REPORT}" == "TRUE" ]; then
    spartan_submit_jobs_report=1
  fi
  spartan_submit_jobs_estimate=$(( SUBMIT_FOLD_COUNT * spartan_submit_jobs_per_fold + spartan_submit_jobs_final + spartan_submit_jobs_report ))
  echo "Spartan submission: ${BOOTSTRAPS_PER_JOB} bootstrap(s)/job, ${BOOTSTRAP_PARALLEL_STEPS} parallel step(s), ${BOOTSTRAP_STEP_CPUS_DEFAULT} CPU(s)/step, bootstrap time=${BOOTSTRAP_JOB_TIME}; serialize folds=${SPARTAN_SERIALIZE_FOLDS}; estimated submitted job record(s): ${spartan_submit_jobs_estimate}. No Marvin-style submit cap/throttle is applied."
fi

TEST_phase=${TEST_phase:-0} # Set to 0 for full run or no testing

# Ensure we’ve created the initial GA population
N_POPULATION=${N_POPULATION}
K_MIN=${K_MIN}
K_MAX=${K_MAX}
MODALITIES=${MODALITIES}

DUMMY_CODE_ARGS=()
if [ -n "${DUMMY_CODE_MODALITIES:-}" ]; then
  # shellcheck disable=SC2206
  DUMMY_CODE_MODALITY_LIST=( ${DUMMY_CODE_MODALITIES} )
  DUMMY_CODE_ARGS+=(--dummy_code_modalities "${DUMMY_CODE_MODALITY_LIST[@]}")
fi
MIXED_CATEGORICAL_ARGS=()
if [ -n "${MIXED_CATEGORICAL_MODALITIES:-}" ]; then
  # shellcheck disable=SC2206
  MIXED_CATEGORICAL_MODALITY_LIST=( ${MIXED_CATEGORICAL_MODALITIES} )
  MIXED_CATEGORICAL_ARGS+=(--mixed_categorical_modalities "${MIXED_CATEGORICAL_MODALITY_LIST[@]}")
fi
DIM_RED_ARGS=(--dim_reduction "${DIMREDUCTION}")
if [ -n "${DIM_REDUCTION_BY_MODALITY:-}" ]; then
  # shellcheck disable=SC2206
  DIM_RED_OVERRIDES=( ${DIM_REDUCTION_BY_MODALITY} )
  DIM_RED_ARGS+=(--dim_reduction_by_modality "${DIM_RED_OVERRIDES[@]}")
fi
if [ -n "${PCA_VARIANCE_THRESHOLD:-}" ]; then
  DIM_RED_ARGS+=(--pca_variance_threshold "${PCA_VARIANCE_THRESHOLD}")
fi
DIM_RED_ARGS+=(
  --snmf_n_components "${SNMF_N_COMPONENTS:-${MAXPC:-20}}"
  --snmf_alpha "${SNMF_ALPHA:-0.1}"
  --snmf_l1_ratio "${SNMF_L1_RATIO:-1.0}"
  --snmf_max_iter "${SNMF_MAX_ITER:-1000}"
)

wait_for_submit_slot() {
  if [ "${SERVER}" != "marvin" ]; then
    return 0
  fi
  local max_jobs="${MARVIN_MAX_SUBMITTED_JOBS:-180}"
  local poll_seconds="${MARVIN_SUBMIT_POLL_SECONDS:-30}"
  local n_jobs
  while true; do
    n_jobs=$(squeue -u "${USER}" -h 2>/dev/null | wc -l | tr -d ' ')
    n_jobs=${n_jobs:-0}
    if [ "${n_jobs}" -lt "${max_jobs}" ]; then
      return 0
    fi
    echo "Marvin safety throttle: ${n_jobs} queued/running jobs >= ${max_jobs}; waiting ${poll_seconds}s before submitting more." >&2
    sleep "${poll_seconds}"
  done
}

submit_sbatch() {
  wait_for_submit_slot
  if [ -n "${LOGS_DIR:-}" ]; then
    mkdir -p "${LOGS_DIR}"
    sbatch --output="${LOGS_DIR}/slurm_%x_%A_%a.out" "$@"
  else
    sbatch "$@"
  fi
}

schedule_chained_run() {
  local terminal_job_id="$1"
  if [ -z "${CHAIN_NEXT_BASE_DIR:-}" ]; then
    return 0
  fi
  if [ -z "${terminal_job_id:-}" ]; then
    echo "ERROR: CHAIN_NEXT_BASE_DIR was set, but no terminal dependency job id is available." >&2
    return 1
  fi

  echo "Scheduling chained pipeline after job ${terminal_job_id}: ${CHAIN_NEXT_BASE_DIR}"
  BASE_DIR="${CHAIN_NEXT_BASE_DIR}" \
  RESULTS_DIR="${CHAIN_NEXT_RESULTS_DIR:-}" \
  INTERMEDIATES_DIR="${CHAIN_NEXT_INTERMEDIATES_DIR:-}" \
  LOGS_DIR="${CHAIN_NEXT_LOGS_DIR:-}" \
  PLOTS_DIR="${CHAIN_NEXT_PLOTS_DIR:-}" \
  REPORT_PDF="" \
  SIF="${SIF:-}" \
  RUN_PROFILE="${CHAIN_NEXT_RUN_PROFILE:-${RUN_PROFILE:-}}" \
  INTERNAL_ENSEMBLE_ENABLED="${CHAIN_NEXT_INTERNAL_ENSEMBLE_ENABLED:-${INTERNAL_ENSEMBLE_ENABLED:-TRUE}}" \
  INTERNAL_ENSEMBLE_BCS="${CHAIN_NEXT_INTERNAL_ENSEMBLE_BCS:-${INTERNAL_ENSEMBLE_BCS:-}}" \
  FINAL_INTERNAL_ENSEMBLE_BCS="${CHAIN_NEXT_FINAL_INTERNAL_ENSEMBLE_BCS:-${FINAL_INTERNAL_ENSEMBLE_BCS:-}}" \
  CHAIN_NEXT_BASE_DIR="" \
  submit_sbatch ${PARTITION_OPT} --parsable --dependency=afterok:${terminal_job_id} \
    --export=ALL \
    run.sh
}

schedule_next_simpleclust_dimreduction() {
  local merge_job_id="$1"
  local report_job_id="${2:-}"
  local remaining="${SIMPLECLUST_DIMREDUCTION_REMAINING:-}"
  if [ -z "${remaining}" ]; then
    return 1
  fi

  local specs=()
  local next_spec next_method next_maxpc next_penalty next_spca_alpha next_snmf_alpha next_sparse_l1 rest label
  # shellcheck disable=SC2206
  specs=( ${remaining} )
  next_spec="${specs[0]}"
  rest="${specs[*]:1}"
  next_method="${next_spec%%:*}"
  next_maxpc="${SIMPLECLUST_SEQUENCE_DEFAULT_MAXPC:-20}"
  next_penalty=""
  next_spca_alpha="${SIMPLECLUST_SEQUENCE_DEFAULT_SPCA_ALPHA:-1.0}"
  next_snmf_alpha="${SIMPLECLUST_SEQUENCE_DEFAULT_SNMF_ALPHA:-0.1}"
  next_sparse_l1="${SIMPLECLUST_SEQUENCE_DEFAULT_SPARSE_L1:-1e-3}"
  if [ "${next_spec}" != "${next_method}" ]; then
    local spec_remainder="${next_spec#*:}"
    next_maxpc="${spec_remainder%%:*}"
    if [ "${spec_remainder}" != "${next_maxpc}" ]; then
      next_penalty="${spec_remainder#*:}"
    fi
  fi

  case "${next_method}" in
    None|none) next_method="None"; label="none" ;;
    PCA|pca) next_method="PCA"; label="pca_${next_maxpc}" ;;
    SparsePCA|sparsepca|SPCA|spca)
      next_method="SparsePCA"
      next_spca_alpha="${next_penalty:-${next_spca_alpha}}"
      label="sparsepca_${next_maxpc}_alpha_${next_spca_alpha//./p}"
      ;;
    SparseNMF|sparsenmf|Sparse_NMF|sparse_nmf|SNMF|snmf)
      next_method="SparseNMF"
      next_snmf_alpha="${next_penalty:-${next_snmf_alpha}}"
      label="sparsenmf_${next_maxpc}_alpha_${next_snmf_alpha//./p}"
      ;;
    AE|ae) next_method="AE"; label="ae" ;;
    SparseAE|sparseae)
      next_method="SparseAE"
      next_sparse_l1="${next_penalty:-${next_sparse_l1}}"
      label="sparseae_l1_${next_sparse_l1//./p}"
      ;;
    VAE|vae) next_method="VAE"; label="vae" ;;
    SparseVAE|sparsevae)
      next_method="SparseVAE"
      next_sparse_l1="${next_penalty:-${next_sparse_l1}}"
      label="sparsevae_l1_${next_sparse_l1//./p}"
      ;;
    *)
      echo "ERROR: unsupported simpleclust dimensionality specification '${next_spec}'." >&2
      return 2
      ;;
  esac

  local source_dir="${SIMPLECLUST_SEQUENCE_SOURCE_DIR:-${BASE_DIR}}"
  local prefix="${SIMPLECLUST_SEQUENCE_OUTPUT_PREFIX:-simpleclust_dimred}"
  local results_dir="${source_dir}/results_${prefix}_${label}"
  local intermediates_dir="${source_dir}/intermediates_${prefix}_${label}"
  local logs_dir="${source_dir}/logs_${prefix}_${label}"
  local plots_dir="${source_dir}/plots_${prefix}_${label}"
  local dependency="afterok:${merge_job_id}"
  if [ -n "${report_job_id}" ]; then
    dependency="${dependency},afterany:${report_job_id}"
  fi

  echo "Scheduling next simpleclust dimensionality run '${next_spec}' after ${dependency}."
  wait_for_submit_slot
  BASE_DIR="${source_dir}" \
  RESULTS_DIR="${results_dir}" \
  INTERMEDIATES_DIR="${intermediates_dir}" \
  LOGS_DIR="${logs_dir}" \
  PLOTS_DIR="${plots_dir}" \
  REPORT_PDF="" \
  SIF="${SIF:-}" \
  RUN_PROFILE="${RUN_PROFILE:-simpleclust}" \
  ALLOW_PROFILE_DIMREDUCTION_OVERRIDE="TRUE" \
  SIMPLECLUST_DIMREDUCTION_OVERRIDE="${next_method}" \
  SIMPLECLUST_MAXPC_OVERRIDE="${next_maxpc}" \
  SIMPLECLUST_SPCA_ALPHA_OVERRIDE="${next_spca_alpha}" \
  SIMPLECLUST_SNMF_ALPHA_OVERRIDE="${next_snmf_alpha}" \
  SIMPLECLUST_SPARSE_L1_OVERRIDE="${next_sparse_l1}" \
  SIMPLECLUST_DIMREDUCTION_REMAINING="${rest}" \
  SIMPLECLUST_SEQUENCE_SOURCE_DIR="${source_dir}" \
  SIMPLECLUST_SEQUENCE_OUTPUT_PREFIX="${prefix}" \
  SIMPLECLUST_SEQUENCE_DEFAULT_MAXPC="${SIMPLECLUST_SEQUENCE_DEFAULT_MAXPC:-20}" \
  SIMPLECLUST_SEQUENCE_DEFAULT_SPCA_ALPHA="${SIMPLECLUST_SEQUENCE_DEFAULT_SPCA_ALPHA:-1.0}" \
  SIMPLECLUST_SEQUENCE_DEFAULT_SNMF_ALPHA="${SIMPLECLUST_SEQUENCE_DEFAULT_SNMF_ALPHA:-0.1}" \
  SIMPLECLUST_SEQUENCE_DEFAULT_SPARSE_L1="${SIMPLECLUST_SEQUENCE_DEFAULT_SPARSE_L1:-1e-3}" \
  sbatch ${PARTITION_OPT} --parsable --dependency="${dependency}" \
    --output="${logs_dir}/run_%j.log" --export=ALL run.sh
  return 0
}


########################################
# 1) Schedule outer CV folds
########################################

post_ids=() #Store the job ID of postprocess for each fold
prev_fold_final_id=""

for OUTER_FOLD in $(seq 1 $N_FOLDS); do

  export OUTER_FOLD
  FOLD_INDEX=$((OUTER_FOLD-1))
  export FOLD_INDEX
  if [ ${FOLD_INDEX} -lt ${RESUME_FROM_FOLD} ] || [ ${FOLD_INDEX} -gt ${RESUME_TO_FOLD} ]; then
    echo "[Fold ${FOLD_INDEX}] Skipping (resume range ${RESUME_FROM_FOLD}-${RESUME_TO_FOLD})"
    continue
  fi
  GA_ROOT="${INTERMEDIATES_DIR}/fold${FOLD_INDEX}/ga"
  INIT_POP="${GA_ROOT}/population_init_fold${FOLD_INDEX}.pkl"
  POP_INIT=${INIT_POP}
  export GA_ROOT
  export POP_INIT

  ########################################
  # --- Test phases that do not need full pipeline ---
  # Expect: TEST is "TRUE" or "FALSE"; TEST_phase is 0..4
  if [[ "$TEST" == "TRUE" ]]; then
    case "$TEST_phase" in
      1)
        echo "=== Test phase 1: only running single method (KMeans) ==="
        apptainer exec "$SIF" python full_pipeline.py --mode test1 \
            --input_csv          "${INPUT_CSV}" \
            --meta_csv           "${META_CSV}" \
            --fold_index         "${FOLD_INDEX}" \
            --n_folds            "${N_FOLDS}" \
            --col_threshold      "${COL_THRESHOLD}" \
            --row_threshold      "${ROW_THRESHOLD}" \
            --skew_threshold     "${SKEW_THRESHOLD}" \
            --scaler_type        "${SCALER_TYPE}" \
            --modalities         ${MODALITIES} \
            "${DUMMY_CODE_ARGS[@]}" \
            "${MIXED_CATEGORICAL_ARGS[@]}" \
            --base_dir           "${BASE_DIR}" \
            --TEST               "${TEST}" 
        exit 0
        ;;
      2)
        echo "=== Test phase 2: only running single method (Spectral) ==="
        apptainer exec "$SIF" python full_pipeline.py --mode test2 \
            --input_csv          "${INPUT_CSV}" \
            --meta_csv           "${META_CSV}" \
            --fold_index         "${FOLD_INDEX}" \
            --n_folds            "${N_FOLDS}" \
            --col_threshold      "${COL_THRESHOLD}" \
            --row_threshold      "${ROW_THRESHOLD}" \
            --skew_threshold     "${SKEW_THRESHOLD}" \
            --scaler_type        "${SCALER_TYPE}" \
            --modalities         ${MODALITIES} \
            "${DUMMY_CODE_ARGS[@]}" \
            "${MIXED_CATEGORICAL_ARGS[@]}" \
            --base_dir           "${BASE_DIR}" \
            --TEST               "${TEST}" 
        exit 0
        ;;
      3)
        echo "=== Test phase 3: Verify fusion matrices and individual ensemble cluster ==="
        apptainer exec "$SIF" python full_pipeline.py --mode test3 \
            --input_csv          "${INPUT_CSV}" \
            --meta_csv           "${META_CSV}" \
            --fold_index         "${FOLD_INDEX}" \
            --n_folds            "${N_FOLDS}" \
            --col_threshold      "${COL_THRESHOLD}" \
            --row_threshold      "${ROW_THRESHOLD}" \
            --skew_threshold     "${SKEW_THRESHOLD}" \
            --scaler_type        "${SCALER_TYPE}" \
            --modalities         ${MODALITIES} \
            "${DUMMY_CODE_ARGS[@]}" \
            "${MIXED_CATEGORICAL_ARGS[@]}" \
            --base_dir           "${BASE_DIR}" \
            --TEST               "${TEST}" 
        exit 0
        ;;
      4)
        echo "=== Test phase 4: Final clustering correctness on fusion matrix ==="
        apptainer exec "$SIF" python full_pipeline.py --mode test4 \
            --input_csv          "${INPUT_CSV}" \
            --meta_csv           "${META_CSV}" \
            --fold_index         "${FOLD_INDEX}" \
            --n_folds            "${N_FOLDS}" \
            --col_threshold      "${COL_THRESHOLD}" \
            --row_threshold      "${ROW_THRESHOLD}" \
            --skew_threshold     "${SKEW_THRESHOLD}" \
            --scaler_type        "${SCALER_TYPE}" \
            --modalities         ${MODALITIES} \
            "${DUMMY_CODE_ARGS[@]}" \
            "${MIXED_CATEGORICAL_ARGS[@]}" \
            --base_dir           "${BASE_DIR}" \
            --TEST               "${TEST}" 
        exit 0
        ;;
    esac
  fi


  ########################################

  echo "[Fold ${FOLD_INDEX}] Initializing GA population…"
  mkdir -p "${GA_ROOT}" "${LOGS_DIR}"
  if [ "${CLUSTER_PIPELINE}" = "singleclust" ]; then
    init_cmd=(
      apptainer exec "${SIF}"
      python -u singleclust/full_pipeline_singleclust.py --mode init
      --population_file "${INIT_POP}"
      --fold_index "${FOLD_INDEX}"
      --n_population ${N_POPULATION}
      --k_min ${K_MIN} --k_max ${K_MAX}
      --linkages ${CLUSTER_LINKAGES}
      --search_objectives ${SEARCH_OBJECTIVES:-${GA_OBJECTIVES}}
      --optimisation "${OPTIMISATION}"
      --dim_reduction "${DIMREDUCTION}"
      --maxPC "${MAXPC:-20}"
      --snmf_alpha "${SNMF_ALPHA:-0.1}"
      --snmf_l1_ratio "${SNMF_L1_RATIO:-1.0}"
      --snmf_max_iter "${SNMF_MAX_ITER:-1000}"
      --base_dir "${BASE_DIR}"
      --seed ${OUTER_FOLD}
    )
  else
    init_cmd=(
      apptainer exec "${SIF}"
      python full_pipeline.py --mode init
      --population_file "${INIT_POP}"
      --n_population ${N_POPULATION}
      --k_min ${K_MIN} --k_max ${K_MAX}
      --linkages ${CLUSTER_LINKAGES}
      --ga_objectives ${GA_OBJECTIVES}
      --fusion_methods ${FUSION_METHODS}
      --modalities ${MODALITIES}
      "${DUMMY_CODE_ARGS[@]}"
      "${MIXED_CATEGORICAL_ARGS[@]}"
      "${DIM_RED_ARGS[@]}"
      --base_dir "${BASE_DIR}"
      --seed ${OUTER_FOLD}
    )
  fi
  if ! "${init_cmd[@]}" 2>&1 | tee "${LOGS_DIR}/fold${FOLD_INDEX}_init.log"; then
    echo "ERROR: failed to initialize GA population for fold ${FOLD_INDEX}. See ${LOGS_DIR}/fold${FOLD_INDEX}_init.log" >&2
    exit 1
  fi

  if [ ! -s "${INIT_POP}" ]; then
    echo "ERROR: expected initial population was not created: ${INIT_POP}" >&2
    echo "See ${LOGS_DIR}/fold${FOLD_INDEX}_init.log and the main Slurm output for the root cause." >&2
    exit 1
  fi

  # Chain GA generations so each bootstrap waits on the previous gather
  prev_gather_id=""
  fold_dependency_args=()
  serialize_folds="FALSE"
  if [ "${SERVER}" == "marvin" ] && [ "${MARVIN_SERIALIZE_FOLDS:-TRUE}" == "TRUE" ]; then
    serialize_folds="TRUE"
  elif [ "${SERVER}" == "spartan" ] && [ "${SPARTAN_SERIALIZE_FOLDS:-FALSE}" == "TRUE" ]; then
    serialize_folds="TRUE"
  fi
  if [ "${serialize_folds}" == "TRUE" ] && [ -n "${prev_fold_final_id}" ]; then
    fold_dependency_args+=(--dependency=afterok:${prev_fold_final_id})
    echo "[Fold ${FOLD_INDEX}] ${SERVER} fold serialization: first bootstrap waits for job ${prev_fold_final_id}."
  fi

  # ————————————————————————————————————————

  echo "=== Starting outer fold ${FOLD_INDEX} ==="

  # GA settings
  N_GENERATION=${N_GENERATIONS}
  N_BOOTSTRAPS=${N_BOOTSTRAP}
  MAX_CONCURRENT=${MAX_CONCURRENT}

  # initial population (you must generate this once, e.g. with a small helper script;
  # it should be a pickled list of DEAP Individuals)
  POP_INIT=${INIT_POP}
  export POP_INIT


  ########################################
  # 1) Nested‐GA
  ########################################
  for GEN in $(seq 1 $N_GENERATIONS); do
    # Export current generation for child jobs
    export GEN
    echo "[Fold $FOLD_INDEX] Generation $GEN: launching bootstrap array..."
    bootstrap_array_spec="1-${N_BOOTSTRAPS}%${MAX_CONCURRENT}"
    bootstrap_sbatch_resource_args=(--cpus-per-task="${BOOTSTRAP_STEP_CPUS_DEFAULT}")
    if [ -n "${BOOTSTRAP_JOB_TIME:-}" ]; then
      bootstrap_sbatch_resource_args+=(--time="${BOOTSTRAP_JOB_TIME}")
    fi
    if [ "${BOOTSTRAPS_PER_JOB:-1}" -gt 1 ]; then
      grouped_bootstrap_array_jobs=$(( (N_BOOTSTRAPS + BOOTSTRAPS_PER_JOB - 1) / BOOTSTRAPS_PER_JOB ))
      grouped_bootstrap_job_concurrency=$(( (MAX_CONCURRENT + BOOTSTRAP_PARALLEL_STEPS - 1) / BOOTSTRAP_PARALLEL_STEPS ))
      if [ "${grouped_bootstrap_job_concurrency}" -lt 1 ]; then
        grouped_bootstrap_job_concurrency=1
      fi
      bootstrap_array_spec="1-${grouped_bootstrap_array_jobs}%${grouped_bootstrap_job_concurrency}"
      bootstrap_sbatch_resource_args=(--cpus-per-task="${BOOTSTRAP_JOB_CPUS}")
      if [ -n "${BOOTSTRAP_JOB_TIME:-}" ]; then
        bootstrap_sbatch_resource_args+=(--time="${BOOTSTRAP_JOB_TIME}")
      fi
      echo "[Fold $FOLD_INDEX] Generation $GEN: ${SERVER} batches ${N_BOOTSTRAPS} bootstrap(s) into ${grouped_bootstrap_array_jobs} array job(s), up to ${grouped_bootstrap_job_concurrency} grouped job(s) active."
    fi
    if [ -z "$prev_gather_id" ]; then
      array_id=$(submit_sbatch ${PARTITION_OPT} --parsable \
        "${fold_dependency_args[@]}" \
        "${bootstrap_sbatch_resource_args[@]}" \
        --export=ALL,FOLD_INDEX,GEN,POP_INIT,DIMREDUCTION,DIM_REDUCTION_BY_MODALITY,DUMMY_CODE_MODALITIES,MIXED_CATEGORICAL_MODALITIES,MINCLUSTER,N_BOOTSTRAP,BOOTSTRAP_MODE,GA_CXPB,GA_MUTPB,GA_ELITISM,BOOTSTRAPS_PER_JOB,BOOTSTRAP_PARALLEL_STEPS,BOOTSTRAP_STEP_CPUS_DEFAULT,BOOTSTRAP_JOB_CPUS,BOOTSTRAP_JOB_TIME,MARVIN_BOOTSTRAPS_PER_JOB,MARVIN_BOOTSTRAP_PARALLEL_STEPS,MARVIN_BOOTSTRAP_STEP_CPUS,MARVIN_BOOTSTRAP_JOB_CPUS,MARVIN_BOOTSTRAP_STAGGER_SECONDS,MARVIN_BOOTSTRAP_STAGGER_SLOTS,SPARTAN_BOOTSTRAPS_PER_JOB,SPARTAN_BOOTSTRAP_PARALLEL_STEPS,SPARTAN_BOOTSTRAP_STEP_CPUS,SPARTAN_BOOTSTRAP_JOB_CPUS,SPARTAN_BOOTSTRAP_JOB_TIME \
        --array="${bootstrap_array_spec}" \
        bootstrap_generation.sh)
    else
      array_id=$(submit_sbatch ${PARTITION_OPT} --parsable \
        --dependency=afterok:${prev_gather_id} \
        "${bootstrap_sbatch_resource_args[@]}" \
        --export=ALL,FOLD_INDEX,GEN,POP_INIT,DIMREDUCTION,DIM_REDUCTION_BY_MODALITY,DUMMY_CODE_MODALITIES,MIXED_CATEGORICAL_MODALITIES,MINCLUSTER,N_BOOTSTRAP,BOOTSTRAP_MODE,GA_CXPB,GA_MUTPB,GA_ELITISM,BOOTSTRAPS_PER_JOB,BOOTSTRAP_PARALLEL_STEPS,BOOTSTRAP_STEP_CPUS_DEFAULT,BOOTSTRAP_JOB_CPUS,BOOTSTRAP_JOB_TIME,MARVIN_BOOTSTRAPS_PER_JOB,MARVIN_BOOTSTRAP_PARALLEL_STEPS,MARVIN_BOOTSTRAP_STEP_CPUS,MARVIN_BOOTSTRAP_JOB_CPUS,MARVIN_BOOTSTRAP_STAGGER_SECONDS,MARVIN_BOOTSTRAP_STAGGER_SLOTS,SPARTAN_BOOTSTRAPS_PER_JOB,SPARTAN_BOOTSTRAP_PARALLEL_STEPS,SPARTAN_BOOTSTRAP_STEP_CPUS,SPARTAN_BOOTSTRAP_JOB_CPUS,SPARTAN_BOOTSTRAP_JOB_TIME \
        --array="${bootstrap_array_spec}" \
        bootstrap_generation.sh)
    fi
    echo "  -> array job ID: $array_id"
    if [ -z "$array_id" ]; then
      echo "ERROR: bootstrap array submission failed for fold ${FOLD_INDEX} gen ${GEN}. Aborting to avoid broken dependencies." >&2
      exit 1
    fi

    echo "[Fold $FOLD_INDEX] Generation $GEN: launching gather job..."
    gather_id=$(submit_sbatch ${PARTITION_OPT} --parsable \
      --dependency=afterok:${array_id} \
      --export=ALL,FOLD_INDEX,GEN,POP_INIT,DIMREDUCTION,DIM_REDUCTION_BY_MODALITY,DUMMY_CODE_MODALITIES,MIXED_CATEGORICAL_MODALITIES,MINCLUSTER,N_BOOTSTRAP,BOOTSTRAP_MODE,GA_CXPB,GA_MUTPB,GA_ELITISM \
      gather_generation.sh)
    echo "  -> gather job ID: $gather_id"
    if [ -z "$gather_id" ]; then
      echo "ERROR: gather job submission failed for fold ${FOLD_INDEX} gen ${GEN}. Aborting to avoid broken dependencies." >&2
      exit 1
    fi

    prev_gather_id=$gather_id

    # now point to the newly‐evolved population for the next iteration
    POP_IN=${GA_ROOT}/population_fold${FOLD_INDEX}_gen$((GEN+1)).pkl
    export POP_IN
  done

  ########################################
  # 2) When generation $N_GENERATIONS finishes, run post‐processing (AE + Parea)
  ########################################

  echo "[Fold $FOLD_INDEX] Scheduling final post‐processing..."
  outer_id=$(submit_sbatch ${PARTITION_OPT} --parsable --dependency=afterok:${gather_id} \
    --export=ALL,FOLD_INDEX,POP_IN,DIMREDUCTION,DIM_REDUCTION_BY_MODALITY,DUMMY_CODE_MODALITIES,MIXED_CATEGORICAL_MODALITIES,ASSEMBLE_HYBRID,N_BOOTSTRAP,DO_SVM,MINCLUSTER \
    postprocess_outer.sh)

  post_ids+=("${outer_id}")
  prev_fold_final_id="${outer_id}"
done



########################################
# 2) When outer fold finishes, run final merge and SVM classification
########################################
dep_string=$(IFS=:; echo "${post_ids[*]}")
merge_dep_args=()
if [ -n "${dep_string}" ]; then
  merge_dep_args+=(--dependency=afterok:${dep_string})
fi

if [ "${RUN_MERGE}" == "FALSE" ]; then
  echo "Final merge disabled; skipping scheduling of final merge job."
  if [ "${RUN_POSTPROCESS_REPORT}" == "TRUE" ]; then
    report_dep_args=()
    if [ -n "${dep_string}" ]; then
      report_dep_args+=(--dependency=afterany:${dep_string})
    fi
    echo "Scheduling post-processing PDF report job..."
    submit_sbatch ${PARTITION_OPT} "${report_dep_args[@]}" \
      --export=ALL,REPORT_PDF \
      postprocess_report.sh
  fi
  exit 0

elif [ "${RUN_MERGE}" == "TRUE" ]; then
  echo "Scheduling final merge and SVM classification job..."
  merge_id=$(submit_sbatch ${PARTITION_OPT} "${merge_dep_args[@]}" --parsable \
    --export=ALL,DO_SVM,N_BOOTSTRAP,MINCLUSTER,DUMMY_CODE_MODALITIES,MIXED_CATEGORICAL_MODALITIES,DIM_REDUCTION_BY_MODALITY,FINAL_BOOTSTRAP_PREPROCESSING,FINAL_BOOTSTRAP_JOBS,COMPUTE_CLUSTER_PVALUES,CLUSTER_PVALUE_MODE,CLUSTER_PVALUE_STAT,CLUSTER_PVALUE_PERMUTATIONS,CLUSTER_PVALUE_PERMUTATIONS_QUALITY,CLUSTER_PVALUE_PERMUTATIONS_ARI,CLUSTER_PVALUE_JOBS,CLUSTER_PVALUE_SEED,DO_CLUSTER_VALIDATION_SENSITIVITY,CLUSTER_VALIDATION_GAUSSIAN_NULLS,CLUSTER_VALIDATION_GAP_REFS,CLUSTER_VALIDATION_SIGCLUST_SIMS,CLUSTER_VALIDATION_STABILITY_BOOTSTRAPS,CLUSTER_VALIDATION_JOBS \
    final_merge.sh)
  echo "  -> final merge job ID: ${merge_id}"
  if [ -z "${merge_id}" ]; then
    echo "ERROR: final merge job submission failed. Aborting to avoid broken report dependency." >&2
    exit 1
  fi

  report_id=""
  if [ "${RUN_POSTPROCESS_REPORT}" == "TRUE" ]; then
    echo "Scheduling post-processing PDF report job..."
    report_id=$(submit_sbatch ${PARTITION_OPT} --parsable --dependency=afterany:${merge_id} \
      --export=ALL,REPORT_PDF \
      postprocess_report.sh)
    echo "  -> post-processing report job ID: ${report_id}"
    if [ -z "${report_id}" ]; then
      echo "ERROR: post-processing report job submission failed." >&2
      exit 1
    fi
  fi
  if [ -n "${SIMPLECLUST_DIMREDUCTION_REMAINING:-}" ]; then
    schedule_next_simpleclust_dimreduction "${merge_id}" "${report_id}" || exit $?
  else
    schedule_chained_run "${merge_id}"
  fi
  exit 0
fi


#sbatch --export=ALL,DO_SVM,N_BOOTSTRAP,MINCLUSTER final_merge.sh
#sbatch -p intelsr_short --export=ALL,DO_SVM,N_BOOTSTRAP,MINCLUSTER final_merge.sh
#sbatch postprocess_report.sh

n

  
