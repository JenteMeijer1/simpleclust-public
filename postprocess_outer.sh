#!/bin/bash
# Purpose: Post-process one completed outer fold.
#SBATCH --job-name=postprocess
#SBATCH --output=logs/fold${FOLD_INDEX}_postprocess.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=10G
#SBATCH --time=02:00:00

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

GA_ROOT=${GA_ROOT:-"${INTERMEDIATES_DIR}/fold${FOLD_INDEX}/ga"}
FOLD_RESULTS_DIR=${RESULTS_DIR}/fold${FOLD_INDEX}
mkdir -p ${FOLD_RESULTS_DIR}

POP_FIN=${GA_ROOT}/population_fold${FOLD_INDEX}_gen${N_GENERATIONS}.pkl

DIM_RED_ARGS=(--dim_reduction "${DIMREDUCTION}")
if [ -n "${DIM_REDUCTION_BY_MODALITY:-}" ]; then
  # shellcheck disable=SC2206
  DIM_RED_OVERRIDES=( ${DIM_REDUCTION_BY_MODALITY} )
  DIM_RED_ARGS+=(--dim_reduction_by_modality "${DIM_RED_OVERRIDES[@]}")
fi

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

INTERNAL_ENSEMBLE_ARGS=(
  --internal_ensemble_enabled "${INTERNAL_ENSEMBLE_ENABLED:-FALSE}"
  --internal_ensemble_bcs "${INTERNAL_ENSEMBLE_BCS:-5}"
  --internal_ensemble_sample_frac "${INTERNAL_ENSEMBLE_SAMPLE_FRAC:-0.8}"
  --internal_ensemble_feature_frac "${INTERNAL_ENSEMBLE_FEATURE_FRAC:-1.0}"
)

if [ "${CLUSTER_PIPELINE:-multiview}" = "singleclust" ]; then
  cmd=(
    apptainer exec "${SIF}"
    python -u singleclust/full_pipeline_singleclust.py
    --mode outer
    --fold_index "${FOLD_INDEX}"
    --n_folds "${N_FOLDS}"
    --n_bootstrap "${N_BOOTSTRAP}"
    --population_file "${POP_FIN}"
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
    --sparse_l1_lambda "${SPARSE_L1_LAMBDA:-0.001}"
    --hidden_dims ${HIDDEN_DIMS}
    --activation_functions ${ACTIVATION_FUNCTIONS}
    --learning_rates ${LEARNING_RATES}
    --batch_sizes ${BATCH_SIZES}
    --latent_dims ${LATENT_DIMS}
    --optimisation "${OPTIMISATION}"
    --search_objectives ${SEARCH_OBJECTIVES:-${GA_OBJECTIVES}}
    --output_metrics "${FOLD_RESULTS_DIR}/metrics.pkl"
    --mincluster ${MINCLUSTER}
    --mincluster_n ${MINCLUSTER_N}
    --mincluster_resample_mode "${MINCLUSTER_RESAMPLE_MODE:-fixed}"
    --use_effective_k_for_fold_merge "${USE_EFFECTIVE_K_FOR_FOLD_MERGE:-FALSE}"
    --use_cross_fold_effective_k_for_final_run "${USE_CROSS_FOLD_EFFECTIVE_K_FOR_FINAL_RUN:-FALSE}"
    "${INTERNAL_ENSEMBLE_ARGS[@]}"
    --TEST "${TEST}"
    --DO_SVM "${DO_SVM}"
    --base_dir "${BASE_DIR}"
  )
else
  cmd=(
    apptainer exec "${SIF}"
    python -u full_pipeline.py
    --mode outer
    --fold_index "${FOLD_INDEX}"
    --n_folds "${N_FOLDS}"
    --n_bootstrap "${N_BOOTSTRAP}"
    --population_file "${POP_FIN}"
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
    --fusion_methods ${FUSION_METHODS}
    --output_metrics "${FOLD_RESULTS_DIR}/metrics.pkl"
    --mincluster ${MINCLUSTER}
    --mincluster_n ${MINCLUSTER_N}
    --mincluster_resample_mode "${MINCLUSTER_RESAMPLE_MODE:-fixed}"
    --use_effective_k_for_fold_merge "${USE_EFFECTIVE_K_FOR_FOLD_MERGE:-FALSE}"
    --use_cross_fold_effective_k_for_final_run "${USE_CROSS_FOLD_EFFECTIVE_K_FOR_FINAL_RUN:-FALSE}"
    "${INTERNAL_ENSEMBLE_ARGS[@]}"
    --TEST "${TEST}"
    --DO_SVM "${DO_SVM}"
    --base_dir "${BASE_DIR}"
  )
fi

"${cmd[@]}" | tee -a "${LOGS_DIR}/fold${FOLD_INDEX}_postprocess.log"

exit ${PIPESTATUS[0]}
