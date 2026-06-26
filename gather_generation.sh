#!/bin/bash
#SBATCH --job-name=ga_gather
#SBATCH --output=logs/fold${FOLD_INDEX}_gen${GEN}_gather.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00

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

GA_ROOT=${GA_ROOT:-"${INTERMEDIATES_DIR}/fold${FOLD_INDEX}/ga"}
BOOT_DIR=${GA_ROOT}/gen${GEN}

POP_IN=${GA_ROOT}/population_fold${FOLD_INDEX}_gen${GEN}.pkl
POP_OUT=${GA_ROOT}/population_fold${FOLD_INDEX}_gen$((GEN+1)).pkl
POP_INIT=${POP_INIT:-"${GA_ROOT}/population_init_fold${FOLD_INDEX}.pkl"}

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

# Use all allocated CPUs for parallelisation
N_JOBS=${SLURM_CPUS_PER_TASK:-1}

if [ "${CLUSTER_PIPELINE:-multiview}" = "singleclust" ]; then
  cmd=(
    apptainer exec "${SIF}"
    python -u singleclust/full_pipeline_singleclust.py
    --mode gather
    --input_csv "${INPUT_CSV}"
    --meta_csv "${META_CSV}"
    --fold_index "${FOLD_INDEX}"
    --generation "${GEN}"
    --bootstrap_dir "${BOOT_DIR}"
    --population_dir "${GA_ROOT}/gen${GEN}"
    --population_file "${POP_IN}"
    --population_initial_file "${POP_INIT}"
    --n_folds "${N_FOLDS}"
    --n_bootstrap "${N_BOOTSTRAP}"
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
    --output_population "${POP_OUT}"
    --n_jobs "${N_JOBS}"
    --mincluster ${MINCLUSTER}
    --mincluster_n ${MINCLUSTER_N}
    --mincluster_resample_mode "${MINCLUSTER_RESAMPLE_MODE:-fixed}"
    --use_effective_k_for_fold_merge "${USE_EFFECTIVE_K_FOR_FOLD_MERGE:-FALSE}"
    --use_cross_fold_effective_k_for_final_run "${USE_CROSS_FOLD_EFFECTIVE_K_FOR_FINAL_RUN:-FALSE}"
    --TEST "${TEST}"
    --ga_cxpb "${GA_CXPB}"
    --ga_mutpb "${GA_MUTPB}"
    --ga_elitism "${GA_ELITISM}"
    --base_dir "${BASE_DIR}"
  )
else
  cmd=(
    apptainer exec "${SIF}"
    python -u full_pipeline.py
    --mode gather
    --input_csv "${INPUT_CSV}"
    --meta_csv "${META_CSV}"
    --fold_index "${FOLD_INDEX}"
    --generation "${GEN}"
    --bootstrap_dir "${BOOT_DIR}"
    --population_dir "${GA_ROOT}/gen${GEN}"
    --population_file "${POP_IN}"
    --population_initial_file "${POP_INIT}"
    --n_folds "${N_FOLDS}"
    --n_bootstrap "${N_BOOTSTRAP}"
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
    --output_population "${POP_OUT}"
    --n_jobs "${N_JOBS}"
    --mincluster ${MINCLUSTER}
    --mincluster_n ${MINCLUSTER_N}
    --mincluster_resample_mode "${MINCLUSTER_RESAMPLE_MODE:-fixed}"
    --use_effective_k_for_fold_merge "${USE_EFFECTIVE_K_FOR_FOLD_MERGE:-FALSE}"
    --use_cross_fold_effective_k_for_final_run "${USE_CROSS_FOLD_EFFECTIVE_K_FOR_FINAL_RUN:-FALSE}"
    --TEST "${TEST}"
    --ga_cxpb "${GA_CXPB}"
    --ga_mutpb "${GA_MUTPB}"
    --ga_elitism "${GA_ELITISM}"
    --base_dir "${BASE_DIR}"
  )
fi

"${cmd[@]}" | tee -a "${LOGS_DIR}/fold${FOLD_INDEX}_gen${GEN}_gather.log"


exit ${PIPESTATUS[0]}
