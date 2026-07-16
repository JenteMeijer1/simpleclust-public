#!/bin/bash
# Purpose: Merge completed folds and create the final clustering outputs.
#SBATCH --job-name=merge
#SBATCH --output=logs/merge.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=62
#SBATCH --mem=64G
#SBATCH --time=8:00:00

if [ "$SERVER" == "spartan" ]; then
  module load Apptainer
  BASE_DIR=${BASE_DIR:-path/to/multiclust}
  cd "${BASE_DIR}"
elif [ "$SERVER" == "marvin" ]; then
  module load Apptainer 2>/dev/null || module load apptainer 2>/dev/null || true
  if [ -z "${BASE_DIR:-}" ]; then
    case "${RUN_PROFILE:-}" in
      prospect|run_profiles/prospect.sh|*/run_profiles/prospect.sh) BASE_DIR="/home/s45jmeij_hpc/prospect_clust" ;;
      *) BASE_DIR="/home/s45jmeij_hpc/multiclust" ;;
    esac
  fi
  cd "${BASE_DIR}"
fi

BASE_DIR=${BASE_DIR:-$(pwd)}
export BASE_DIR
export RESULTS_DIR="${RESULTS_DIR:-${BASE_DIR}/results}"
export INTERMEDIATES_DIR="${INTERMEDIATES_DIR:-${BASE_DIR}/intermediates}"
export LOGS_DIR="${LOGS_DIR:-${BASE_DIR}/logs}"
export PLOTS_DIR="${PLOTS_DIR:-${BASE_DIR}/plots}"
mkdir -p "${LOGS_DIR}"

SIF=${SIF:-multiview_env.sif}
if [ ! -f "${SIF}" ] && [ -f "${BASE_DIR}/${SIF}" ]; then
  SIF="${BASE_DIR}/${SIF}"
fi
if [ ! -f "${SIF}" ]; then
  echo "ERROR: Apptainer image not found: ${SIF}" >&2
  echo "Set SIF to a valid .sif path or place multiview_env.sif in ${BASE_DIR}" >&2
  exit 2
fi
export SIF
echo "Using Apptainer image: ${SIF}"

FINAL_RESULTS_DIR=${RESULTS_DIR}/final/
mkdir -p ${FINAL_RESULTS_DIR}

COMPUTE_CLUSTER_PVALUES=${COMPUTE_CLUSTER_PVALUES:-FALSE}
CLUSTER_PVALUE_MODE=${CLUSTER_PVALUE_MODE:-fast}
CLUSTER_PVALUE_STAT=${CLUSTER_PVALUE_STAT:-composite}
CLUSTER_PVALUE_PERMUTATIONS=${CLUSTER_PVALUE_PERMUTATIONS:-200}
CLUSTER_PVALUE_PERMUTATIONS_QUALITY=${CLUSTER_PVALUE_PERMUTATIONS_QUALITY:-}
CLUSTER_PVALUE_PERMUTATIONS_ARI=${CLUSTER_PVALUE_PERMUTATIONS_ARI:-}
CLUSTER_PVALUE_JOBS=${CLUSTER_PVALUE_JOBS:-0}
CLUSTER_PVALUE_SEED=${CLUSTER_PVALUE_SEED:-314159}
FINAL_BOOTSTRAP_PREPROCESSING=${FINAL_BOOTSTRAP_PREPROCESSING:-outside}
FINAL_BOOTSTRAP_JOBS=${FINAL_BOOTSTRAP_JOBS:-}
N_JOBS=${SLURM_CPUS_PER_TASK:-1}

PVAL_EXTRA_ARGS=()
if [ -n "${CLUSTER_PVALUE_PERMUTATIONS_QUALITY}" ]; then
  PVAL_EXTRA_ARGS+=(--cluster_pvalue_permutations_quality "${CLUSTER_PVALUE_PERMUTATIONS_QUALITY}")
fi
if [ -n "${CLUSTER_PVALUE_PERMUTATIONS_ARI}" ]; then
  PVAL_EXTRA_ARGS+=(--cluster_pvalue_permutations_ari "${CLUSTER_PVALUE_PERMUTATIONS_ARI}")
fi
BOOTSTRAP_EXTRA_ARGS=()
if [ -n "${FINAL_BOOTSTRAP_JOBS}" ]; then
  BOOTSTRAP_EXTRA_ARGS+=(--bootstrap_jobs "${FINAL_BOOTSTRAP_JOBS}")
fi
FINAL_INTERNAL_ENSEMBLE_BCS=${FINAL_INTERNAL_ENSEMBLE_BCS:-${INTERNAL_ENSEMBLE_BCS:-5}}
echo "Final merge internal ensemble: enabled=${INTERNAL_ENSEMBLE_ENABLED:-FALSE} bcs=${FINAL_INTERNAL_ENSEMBLE_BCS} sample_frac=${INTERNAL_ENSEMBLE_SAMPLE_FRAC:-0.8} feature_frac=${INTERNAL_ENSEMBLE_FEATURE_FRAC:-1.0}"
INTERNAL_ENSEMBLE_ARGS=(
  --internal_ensemble_enabled "${INTERNAL_ENSEMBLE_ENABLED:-FALSE}"
  --internal_ensemble_bcs "${FINAL_INTERNAL_ENSEMBLE_BCS}"
  --internal_ensemble_sample_frac "${INTERNAL_ENSEMBLE_SAMPLE_FRAC:-0.8}"
  --internal_ensemble_feature_frac "${INTERNAL_ENSEMBLE_FEATURE_FRAC:-1.0}"
)

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

if [ "${CLUSTER_PIPELINE:-multiview}" = "singleclust" ]; then
  cmd=(
    apptainer exec "${SIF}"
    python -u singleclust/full_pipeline_singleclust.py
    --mode merge
    --base_dir "${BASE_DIR}"
    --n_folds "${N_FOLDS}"
    --n_jobs "${N_JOBS}"
    --n_bootstrap "${N_BOOTSTRAP}"
    --input_csv "${INPUT_CSV}"
    --meta_csv "${META_CSV}"
    --col_threshold "${COL_THRESHOLD}"
    --row_threshold "${ROW_THRESHOLD}"
    --skew_threshold "${SKEW_THRESHOLD}"
    --scaler_type "${SCALER_TYPE}"
    --modalities ${MODALITIES}
    "${DUMMY_CODE_ARGS[@]}"
    "${MIXED_CATEGORICAL_ARGS[@]}"
    --dim_reduction "${DIMREDUCTION}"
    --maxPC "${MAXPC:-20}"
    --spca_alpha "${SPCA_ALPHA:-1.0}"
    --spca_ridge_alpha "${SPCA_RIDGE_ALPHA:-0.01}"
    --spca_max_iter "${SPCA_MAX_ITER:-1000}"
    --snmf_alpha "${SNMF_ALPHA:-0.1}"
    --snmf_l1_ratio "${SNMF_L1_RATIO:-1.0}"
    --snmf_max_iter "${SNMF_MAX_ITER:-1000}"
    --sparse_l1_lambda "${SPARSE_L1_LAMBDA:-0.001}"
    --hidden_dims ${HIDDEN_DIMS}
    --activation_functions ${ACTIVATION_FUNCTIONS}
    --learning_rates ${LEARNING_RATES}
    --batch_sizes ${BATCH_SIZES}
    --latent_dims ${LATENT_DIMS}
    --optimisation "${OPTIMISATION}"
    --search_objectives ${SEARCH_OBJECTIVES:-${GA_OBJECTIVES}}
    --linkages ${CLUSTER_LINKAGES}
    --mincluster ${MINCLUSTER}
    --mincluster_n ${MINCLUSTER_N}
    --mincluster_resample_mode "${MINCLUSTER_RESAMPLE_MODE:-fixed}"
    --use_effective_k_for_fold_merge "${USE_EFFECTIVE_K_FOR_FOLD_MERGE:-FALSE}"
    --use_cross_fold_effective_k_for_final_run "${USE_CROSS_FOLD_EFFECTIVE_K_FOR_FINAL_RUN:-FALSE}"
    --output_final_metrics "${FINAL_RESULTS_DIR}/final_metrics.pkl"
    --final_bootstrap_preprocessing "${FINAL_BOOTSTRAP_PREPROCESSING}"
    --n_permutations_pvalue "${CLUSTER_PVALUE_PERMUTATIONS:-200}"
    "${INTERNAL_ENSEMBLE_ARGS[@]}"
    --TEST "${TEST}"
    --DO_SVM "${DO_SVM}"
  )
else
  cmd=(
    apptainer exec "${SIF}"
    python -u full_pipeline.py
    --mode merge
    --base_dir "${BASE_DIR}"
    --n_folds "${N_FOLDS}"
    --n_jobs "${N_JOBS}"
    --n_bootstrap "${N_BOOTSTRAP}"
    --input_csv "${INPUT_CSV}"
    --meta_csv "${META_CSV}"
    --col_threshold "${COL_THRESHOLD}"
    --row_threshold "${ROW_THRESHOLD}"
    --skew_threshold "${SKEW_THRESHOLD}"
    --scaler_type "${SCALER_TYPE}"
    --modalities ${MODALITIES}
    "${DUMMY_CODE_ARGS[@]}"
    "${MIXED_CATEGORICAL_ARGS[@]}"
    "${DIM_RED_ARGS[@]}"
    --hidden_dims ${HIDDEN_DIMS}
    --activation_functions ${ACTIVATION_FUNCTIONS}
    --learning_rates ${LEARNING_RATES}
    --batch_sizes ${BATCH_SIZES}
    --latent_dims ${LATENT_DIMS}
    --optimisation "${OPTIMISATION}"
    --ga_objectives ${GA_OBJECTIVES}
    --linkages ${CLUSTER_LINKAGES}
    --fusion_methods ${FUSION_METHODS}
    --mincluster ${MINCLUSTER}
    --mincluster_n ${MINCLUSTER_N}
    --mincluster_resample_mode "${MINCLUSTER_RESAMPLE_MODE:-fixed}"
    --use_effective_k_for_fold_merge "${USE_EFFECTIVE_K_FOR_FOLD_MERGE:-FALSE}"
    --use_cross_fold_effective_k_for_final_run "${USE_CROSS_FOLD_EFFECTIVE_K_FOR_FINAL_RUN:-FALSE}"
    --output_final_metrics "${FINAL_RESULTS_DIR}/final_metrics.pkl"
    --final_bootstrap_preprocessing "${FINAL_BOOTSTRAP_PREPROCESSING}"
    "${BOOTSTRAP_EXTRA_ARGS[@]}"
    "${INTERNAL_ENSEMBLE_ARGS[@]}"
    --TEST "${TEST}"
    --DO_SVM "${DO_SVM}"
    --compute_cluster_pvalues "${COMPUTE_CLUSTER_PVALUES}"
    --cluster_pvalue_mode "${CLUSTER_PVALUE_MODE}"
    --cluster_pvalue_stat "${CLUSTER_PVALUE_STAT}"
    --cluster_pvalue_permutations "${CLUSTER_PVALUE_PERMUTATIONS}"
    "${PVAL_EXTRA_ARGS[@]}"
    --cluster_pvalue_jobs "${CLUSTER_PVALUE_JOBS}"
    --cluster_pvalue_seed "${CLUSTER_PVALUE_SEED}"
  )
fi

"${cmd[@]}"
merge_status=${PIPESTATUS[0]}
if [ ${merge_status} -ne 0 ]; then
  exit ${merge_status}
fi

DO_CLUSTER_VALIDATION_SENSITIVITY=${DO_CLUSTER_VALIDATION_SENSITIVITY:-FALSE}
CLUSTER_VALIDATION_GAUSSIAN_NULLS=${CLUSTER_VALIDATION_GAUSSIAN_NULLS:-100}
CLUSTER_VALIDATION_GAP_REFS=${CLUSTER_VALIDATION_GAP_REFS:-100}
CLUSTER_VALIDATION_SIGCLUST_SIMS=${CLUSTER_VALIDATION_SIGCLUST_SIMS:-200}
CLUSTER_VALIDATION_STABILITY_BOOTSTRAPS=${CLUSTER_VALIDATION_STABILITY_BOOTSTRAPS:-30}
CLUSTER_VALIDATION_PROJECTION_DIRECTIONS=${CLUSTER_VALIDATION_PROJECTION_DIRECTIONS:-200}
if [ -z "${CLUSTER_VALIDATION_JOBS:-}" ] || [ "${CLUSTER_VALIDATION_JOBS}" = "0" ]; then
  CLUSTER_VALIDATION_JOBS=${SLURM_CPUS_PER_TASK:-1}
fi

if [ "${DO_CLUSTER_VALIDATION_SENSITIVITY}" == "TRUE" ]; then
  echo "Running additional no-cluster / continuum validation sensitivity analyses..."
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
  export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
  export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
  export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
  export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}

  CLUSTER_VALIDATION_APPTAINER_ARGS=()
  if apptainer exec "${SIF}" python -c "import diptest" >/dev/null 2>&1; then
    echo "diptest is already available in ${SIF}."
  else
    DIPTEST_OVERLAY=${DIPTEST_OVERLAY:-diptest_overlay.img}
    PIP_CACHE_DIR=${PIP_CACHE_DIR:-${PWD}/.pip_cache}
    PYTHONUSERBASE=${PYTHONUSERBASE:-${PWD}/.apptainer_python_userbase}
    export PIP_CACHE_DIR
    export PYTHONUSERBASE

    if [ -f "${DIPTEST_OVERLAY}" ] && apptainer exec --overlay "${DIPTEST_OVERLAY}" \
        --env PYTHONUSERBASE="${PYTHONUSERBASE}" \
        "${SIF}" python -c "import diptest" >/dev/null 2>&1; then
      echo "Using existing overlay with diptest: ${DIPTEST_OVERLAY}"
      CLUSTER_VALIDATION_APPTAINER_ARGS=(--overlay "${DIPTEST_OVERLAY}" --env PYTHONUSERBASE="${PYTHONUSERBASE}")
    else
      echo "diptest not found. Creating/updating Apptainer overlay: ${DIPTEST_OVERLAY}"
      if [ ! -f "${DIPTEST_OVERLAY}" ]; then
        apptainer overlay create --size 1024 "${DIPTEST_OVERLAY}"
      fi

      mkdir -p "${PIP_CACHE_DIR}" "${PYTHONUSERBASE}"
      apptainer exec --overlay "${DIPTEST_OVERLAY}" \
        --env PYTHONUSERBASE="${PYTHONUSERBASE}",PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
        "${SIF}" \
        python -m pip install --user diptest

      if apptainer exec --overlay "${DIPTEST_OVERLAY}" \
          --env PYTHONUSERBASE="${PYTHONUSERBASE}" \
          "${SIF}" python -c "import diptest" >/dev/null 2>&1; then
        echo "diptest installed and importable through overlay/user site."
        CLUSTER_VALIDATION_APPTAINER_ARGS=(--overlay "${DIPTEST_OVERLAY}" --env PYTHONUSERBASE="${PYTHONUSERBASE}")
      else
        echo "WARNING: diptest could not be imported after installation. Validation will continue without Hartigan dip-test." >&2
        CLUSTER_VALIDATION_APPTAINER_ARGS=()
      fi
    fi
  fi

  apptainer exec "${CLUSTER_VALIDATION_APPTAINER_ARGS[@]}" "${SIF}" \
    python -u cluster_validation_sensitivity.py \
      --metrics_pkl "${FINAL_RESULTS_DIR}/final_metrics.pkl" \
      --output_dir "${FINAL_RESULTS_DIR}/cluster_validation_sensitivity" \
      --subject_id_column "${SUBJECT_ID_COLUMN:-src_subject_id}" \
      --modalities ${MODALITIES} \
      --k_max "${K_MAX:-10}" \
      --gap_reference_datasets "${CLUSTER_VALIDATION_GAP_REFS}" \
      --sigclust_simulations "${CLUSTER_VALIDATION_SIGCLUST_SIMS}" \
      --gaussian_null_datasets "${CLUSTER_VALIDATION_GAUSSIAN_NULLS}" \
      --null_stability_bootstraps "${CLUSTER_VALIDATION_STABILITY_BOOTSTRAPS}" \
      --projection_directions "${CLUSTER_VALIDATION_PROJECTION_DIRECTIONS}" \
      --n_jobs "${CLUSTER_VALIDATION_JOBS}" \
      --seed "${CLUSTER_PVALUE_SEED}"
  validation_status=$?
  if [ ${validation_status} -ne 0 ]; then
    exit ${validation_status}
  fi
fi

exit 0
