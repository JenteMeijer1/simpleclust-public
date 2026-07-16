#!/bin/bash
# Purpose: Run one generation of bootstrap candidate evaluation.
#SBATCH --job-name=ga_bootstrap
#SBATCH --output=logs/fold${FOLD_INDEX}_gen${GEN}_boot%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=15G
#SBATCH --time=00:45:00

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

echo ">>> bootstrap_generation.sh: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-<unset>}  FOLD_INDEX=${FOLD_INDEX:-<unset>}  GEN=${GEN:-<unset>}  BOOTSTRAP_STEP_INDEX=${BOOTSTRAP_STEP_INDEX:-<batch>}"
echo ">>> Internal ensemble: enabled=${INTERNAL_ENSEMBLE_ENABLED:-FALSE} bcs=${INTERNAL_ENSEMBLE_BCS:-5} sample_frac=${INTERNAL_ENSEMBLE_SAMPLE_FRAC:-0.8} feature_frac=${INTERNAL_ENSEMBLE_FEATURE_FRAC:-1.0}"

BOOTSTRAPS_PER_JOB=${BOOTSTRAPS_PER_JOB:-${MARVIN_BOOTSTRAPS_PER_JOB:-1}}
BOOTSTRAP_PARALLEL_STEPS=${BOOTSTRAP_PARALLEL_STEPS:-${MARVIN_BOOTSTRAP_PARALLEL_STEPS:-1}}
BOOTSTRAP_STEP_CPUS_DEFAULT=${BOOTSTRAP_STEP_CPUS_DEFAULT:-${MARVIN_BOOTSTRAP_STEP_CPUS:-${SLURM_CPUS_PER_TASK:-1}}}
BOOTSTRAP_JOB_CPUS=${BOOTSTRAP_JOB_CPUS:-${MARVIN_BOOTSTRAP_JOB_CPUS:-${SLURM_CPUS_PER_TASK:-1}}}
export BOOTSTRAPS_PER_JOB BOOTSTRAP_PARALLEL_STEPS BOOTSTRAP_STEP_CPUS_DEFAULT BOOTSTRAP_JOB_CPUS

if [ "${SERVER}" == "marvin" ] && [ -z "${BOOTSTRAP_STEP_INDEX:-}" ]; then
  MARVIN_BOOTSTRAP_STAGGER_SECONDS=${MARVIN_BOOTSTRAP_STAGGER_SECONDS:-2}
  MARVIN_BOOTSTRAP_STAGGER_SLOTS=${MARVIN_BOOTSTRAP_STAGGER_SLOTS:-10}
  if [ "${MARVIN_BOOTSTRAP_STAGGER_SECONDS}" -gt 0 ] && [ "${MARVIN_BOOTSTRAP_STAGGER_SLOTS}" -gt 0 ]; then
    stagger_slot=$(( (SLURM_ARRAY_TASK_ID - 1) % MARVIN_BOOTSTRAP_STAGGER_SLOTS ))
    startup_delay=$(( stagger_slot * MARVIN_BOOTSTRAP_STAGGER_SECONDS ))
    if [ "${startup_delay}" -gt 0 ]; then
      echo "Marvin startup stagger: grouped bootstrap job ${SLURM_ARRAY_TASK_ID} sleeping ${startup_delay}s before launching steps."
      sleep "${startup_delay}"
    fi
  fi
fi

GA_ROOT=${GA_ROOT:-"${INTERMEDIATES_DIR}/fold${FOLD_INDEX}/ga"}
POP_IN=${GA_ROOT}/population_fold${FOLD_INDEX}_gen${GEN}.pkl
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

INTERNAL_ENSEMBLE_ARGS=(
  --internal_ensemble_enabled "${INTERNAL_ENSEMBLE_ENABLED:-FALSE}"
  --internal_ensemble_bcs "${INTERNAL_ENSEMBLE_BCS:-5}"
  --internal_ensemble_sample_frac "${INTERNAL_ENSEMBLE_SAMPLE_FRAC:-0.8}"
  --internal_ensemble_feature_frac "${INTERNAL_ENSEMBLE_FEATURE_FRAC:-1.0}"
)

run_bootstrap() {
  local BIDX="$1"
  local OUTDIR="${GA_ROOT}/gen${GEN}/bootstrap_${BIDX}"
  local LAB_OUT="${OUTDIR}/labels_${BIDX}.pkl"
  local START_TIME END_TIME ELAPSED
  local MAX_RETRIES=3
  local RETRY_COUNT=0
  local EXIT_CODE=1
  local N_JOBS=${BOOTSTRAP_STEP_CPUS:-${SLURM_CPUS_PER_TASK:-1}}

  mkdir -p "${OUTDIR}"
  START_TIME=$(date +%s)
  echo "=== [Fold ${FOLD_INDEX}] Bootstrap ${BIDX} started at $(date -Is) with ${N_JOBS} CPU worker(s) ==="
  echo "=== [Fold ${FOLD_INDEX}] Bootstrap ${BIDX} internal ensemble: enabled=${INTERNAL_ENSEMBLE_ENABLED:-FALSE}, bcs=${INTERNAL_ENSEMBLE_BCS:-5}, sample_frac=${INTERNAL_ENSEMBLE_SAMPLE_FRAC:-0.8}, feature_frac=${INTERNAL_ENSEMBLE_FEATURE_FRAC:-1.0} ==="

  while [ "${RETRY_COUNT}" -lt "${MAX_RETRIES}" ] && [ "${EXIT_CODE}" -ne 0 ]; do
    ((RETRY_COUNT++))
    echo "=== [Fold ${FOLD_INDEX}] Bootstrap ${BIDX} attempt ${RETRY_COUNT} at $(date -Is) ==="
    if [ "${CLUSTER_PIPELINE:-multiview}" = "singleclust" ]; then
      cmd=(
        apptainer exec "${SIF}"
        python -u singleclust/full_pipeline_singleclust.py
        --mode bootstrap
        --input_csv "${INPUT_CSV}"
        --meta_csv "${META_CSV}"
        --fold_index "${FOLD_INDEX}"
        --generation ${GEN}
        --population_dir "${GA_ROOT}/gen${GEN}"
        --population_file "${POP_IN}"
        --population_initial_file "${POP_INIT}"
        --bootstrap_index ${BIDX}
        --n_bootstrap ${N_BOOTSTRAP}
        --bootstrap_mode ${BOOTSTRAP_MODE}
        --n_folds "${N_FOLDS}"
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
        --n_jobs "${N_JOBS}"
        --output_labels "${LAB_OUT}"
        --mincluster ${MINCLUSTER}
        --mincluster_n ${MINCLUSTER_N}
        --mincluster_resample_mode "${MINCLUSTER_RESAMPLE_MODE:-fixed}"
        --use_effective_k_for_fold_merge "${USE_EFFECTIVE_K_FOR_FOLD_MERGE:-FALSE}"
        --use_cross_fold_effective_k_for_final_run "${USE_CROSS_FOLD_EFFECTIVE_K_FOR_FINAL_RUN:-FALSE}"
        "${INTERNAL_ENSEMBLE_ARGS[@]}"
        --TEST "${TEST}"
        --base_dir "${BASE_DIR}"
      )
    else
      cmd=(
        apptainer exec "${SIF}"
        python -u full_pipeline.py
        --mode bootstrap
        --input_csv "${INPUT_CSV}"
        --meta_csv "${META_CSV}"
        --fold_index "${FOLD_INDEX}"
        --generation ${GEN}
        --population_dir "${GA_ROOT}/gen${GEN}"
        --population_file "${POP_IN}"
        --population_initial_file "${POP_INIT}"
        --bootstrap_index ${BIDX}
        --n_bootstrap ${N_BOOTSTRAP}
        --bootstrap_mode ${BOOTSTRAP_MODE}
        --n_folds "${N_FOLDS}"
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
        --n_jobs "${N_JOBS}"
        --output_labels "${LAB_OUT}"
        --mincluster ${MINCLUSTER}
        --mincluster_n ${MINCLUSTER_N}
        --mincluster_resample_mode "${MINCLUSTER_RESAMPLE_MODE:-fixed}"
        --use_effective_k_for_fold_merge "${USE_EFFECTIVE_K_FOR_FOLD_MERGE:-FALSE}"
        --use_cross_fold_effective_k_for_final_run "${USE_CROSS_FOLD_EFFECTIVE_K_FOR_FINAL_RUN:-FALSE}"
        "${INTERNAL_ENSEMBLE_ARGS[@]}"
        --TEST "${TEST}"
        --base_dir "${BASE_DIR}"
      )
    fi
    "${cmd[@]}" | tee "${LOGS_DIR}/fold${FOLD_INDEX}_gen${GEN}_boot${BIDX}.log"
    EXIT_CODE=${PIPESTATUS[0]}
    if [ "${EXIT_CODE}" -ne 0 ] && [ "${RETRY_COUNT}" -lt "${MAX_RETRIES}" ]; then
      echo "!!! Bootstrap ${BIDX} failed (exit ${EXIT_CODE}), retrying ..."
    fi
  done

  if [ "${EXIT_CODE}" -ne 0 ]; then
    echo "!!! [Fold ${FOLD_INDEX}] Bootstrap ${BIDX} failed after ${MAX_RETRIES} attempts at $(date -Is) !!!"
  fi

  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))
  echo "=== [Fold ${FOLD_INDEX}] Bootstrap ${BIDX} finished at $(date -Is) (Duration: ${ELAPSED} seconds) ==="
  return "${EXIT_CODE}"
}

if [ -n "${BOOTSTRAP_STEP_INDEX:-}" ]; then
  run_bootstrap "${BOOTSTRAP_STEP_INDEX}"
  exit $?
fi

if [ "${BOOTSTRAPS_PER_JOB:-1}" -gt 1 ]; then
  allocated_bootstrap_cpus=${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-${BOOTSTRAP_JOB_CPUS:-1}}}
  max_parallel_steps_by_allocation=$(( allocated_bootstrap_cpus / BOOTSTRAP_STEP_CPUS_DEFAULT ))
  if [ "${max_parallel_steps_by_allocation}" -lt 1 ]; then
    max_parallel_steps_by_allocation=1
  fi
  if [ "${BOOTSTRAP_PARALLEL_STEPS}" -gt "${max_parallel_steps_by_allocation}" ]; then
    echo "${SERVER} grouped bootstrap job ${SLURM_ARRAY_TASK_ID}: capping parallel steps from ${BOOTSTRAP_PARALLEL_STEPS} to ${max_parallel_steps_by_allocation} for ${allocated_bootstrap_cpus} allocated CPU(s)."
    BOOTSTRAP_PARALLEL_STEPS="${max_parallel_steps_by_allocation}"
  fi
  batch_start=$(( (SLURM_ARRAY_TASK_ID - 1) * BOOTSTRAPS_PER_JOB + 1 ))
  batch_end=$(( batch_start + BOOTSTRAPS_PER_JOB - 1 ))
  if [ "${batch_end}" -gt "${N_BOOTSTRAP}" ]; then
    batch_end="${N_BOOTSTRAP}"
  fi
  echo "${SERVER} grouped bootstrap job ${SLURM_ARRAY_TASK_ID}: indices ${batch_start}-${batch_end}, up to ${BOOTSTRAP_PARALLEL_STEPS} parallel Slurm step(s), ${BOOTSTRAP_STEP_CPUS_DEFAULT} CPU(s) per step."

  step_pids=()
  for BIDX in $(seq "${batch_start}" "${batch_end}"); do
    BOOTSTRAP_STEP_CPUS="${BOOTSTRAP_STEP_CPUS_DEFAULT}" \
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="${BOOTSTRAP_STEP_CPUS_DEFAULT}" \
      --export=ALL,BOOTSTRAP_STEP_INDEX="${BIDX}",BOOTSTRAP_STEP_CPUS="${BOOTSTRAP_STEP_CPUS_DEFAULT}" \
      bash "$0" &
    step_pids+=("$!")
    if [ "${#step_pids[@]}" -ge "${BOOTSTRAP_PARALLEL_STEPS}" ]; then
      wait_status=0
      for step_pid in "${step_pids[@]}"; do
        if ! wait "${step_pid}"; then
          wait_status=1
        fi
      done
      if [ "${wait_status}" -ne 0 ]; then
        exit 1
      fi
      step_pids=()
    fi
  done

  EXIT_CODE=0
  for step_pid in "${step_pids[@]}"; do
    if ! wait "${step_pid}"; then
      EXIT_CODE=1
    fi
  done
  exit "${EXIT_CODE}"
fi

run_bootstrap "${SLURM_ARRAY_TASK_ID}"
exit $?
