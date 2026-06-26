#!/bin/bash
#SBATCH --job-name=grid_bootstrap
#SBATCH --output=logs/bootstrap_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=15G
#SBATCH --time=4:00:00

if [ "$SERVER" == "spartan" ]; then
  module load Apptainer
  BASE_DIR=${BASE_DIR:-path/to/multiclust}
  cd "${BASE_DIR}"
elif [ "$SERVER" == "marvin" ]; then
  BASE_DIR=${BASE_DIR:-/home/s45jmeij_hpc/multiclust}
  cd "${BASE_DIR}"
fi

BASE_DIR=${BASE_DIR:-$(pwd)}
export BASE_DIR
mkdir -p "${BASE_DIR}/logs"

parse_dimred_spec() {
  local spec="$1"
  local default_comps="${2:-}"
  local default_l1="${3:-}"
  local method comps sparse_l1 remainder
  method="${spec%%:*}"
  if [ "${spec}" = "${method}" ]; then
    comps="${default_comps}"
    sparse_l1="${default_l1}"
  else
    remainder="${spec#*:}"
    if [ "${remainder}" = "${remainder#*:}" ]; then
      comps="${remainder:-${default_comps}}"
      sparse_l1="${default_l1}"
    else
      comps="${remainder%%:*}"
      sparse_l1="${remainder#*:}"
      comps="${comps:-${default_comps}}"
      sparse_l1="${sparse_l1:-${default_l1}}"
    fi
  fi
  printf "%s\t%s\t%s\n" "${method}" "${comps}" "${sparse_l1}"
}

dimred_label() {
  local method="$1"
  local comps="${2:-}"
  local sparse_l1="${3:-}"
  local sparse_l1_label="${sparse_l1//./p}"
  case "${method}" in
    PCA|pca) echo "pca_${comps}" ;;
    SparsePCA|sparsepca|SPCA|spca) echo "sparsepca_${comps}" ;;
    SparseNMF|sparsenmf|Sparse_NMF|sparse_nmf|SNMF|snmf) echo "sparsenmf_${comps}" ;;
    AE|ae) echo "ae" ;;
    SparseAE|sparseae) echo "sparseae_l1_${sparse_l1_label}" ;;
    VAE|vae) echo "vae" ;;
    SparseVAE|sparsevae) echo "sparsevae_l1_${sparse_l1_label}" ;;
    None|none) echo "none" ;;
    *) echo "${method,,}${comps:+_${comps}}" ;;
  esac
}

dimred_run_base() {
  local method="$1"
  local comps="${2:-}"
  local sparse_l1="${3:-}"
  local root="${DIMRED_SWEEP_ROOT:-${BASE_DIR}/dimred_runs}"
  echo "${root}/$(dimred_label "${method}" "${comps}" "${sparse_l1}")"
}

echo ">>> bootstrap_candidates.sh: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}  FOLD_INDEX=${FOLD_INDEX:-<unset>}"

BIDX=${SLURM_ARRAY_TASK_ID}
# Record start time
START_TIME=$(date +%s)
echo "=== [Fold ${FOLD_INDEX}] Bootstrap ${BIDX} started at $(date -Is) ==="

# Use all allocated CPUs for parallel clustering
export N_JOBS=${SLURM_CPUS_PER_TASK:-1}

run_bootstrap_for_spec() {
  local method="$1"
  local comps="$2"
  local sparse_l1="$3"
  local run_base="$4"
  local search_root="${run_base}/intermediates/fold${FOLD_INDEX}/search"
  local bootstrap_outdir="${search_root}/bootstrap_${BIDX}"
  local labels_file="${bootstrap_outdir}/labels_${BIDX}.pkl"
  local current_file="${search_root}/candidates_scored_fold${FOLD_INDEX}.pkl"
  local init_file="${search_root}/population_init_fold${FOLD_INDEX}.pkl"
  local log_file="${run_base}/logs/fold${FOLD_INDEX}_boot${BIDX}.log"

  mkdir -p "${bootstrap_outdir}" "${run_base}/logs"

  MAX_RETRIES=3
  RETRY_COUNT=0
  EXIT_CODE=1
  while [ $RETRY_COUNT -lt $MAX_RETRIES ] && [ $EXIT_CODE -ne 0 ]; do
    ((RETRY_COUNT++))
    echo "=== [Fold ${FOLD_INDEX}] Bootstrap ${BIDX} ${method} ${comps} attempt $RETRY_COUNT at $(date -Is) ==="
    apptainer exec ${SIF} \
      python -u singleclust/full_pipeline_singleclust.py \
        --mode bootstrap \
        --input_csv "${INPUT_CSV}" \
        --meta_csv "${META_CSV}" \
        --fold_index "${FOLD_INDEX}" \
        --population_dir "${search_root}" \
        --population_file "${current_file}" \
        --population_initial_file "${init_file}" \
        --bootstrap_index ${BIDX} \
        --n_bootstrap ${N_BOOTSTRAP} \
        --bootstrap_mode ${BOOTSTRAP_MODE} \
        --n_folds "${N_FOLDS}" \
        --col_threshold "${COL_THRESHOLD}" \
        --row_threshold "${ROW_THRESHOLD}" \
        --skew_threshold "${SKEW_THRESHOLD}" \
        --scaler_type "${SCALER_TYPE}" \
        --dim_reduction "${method}" \
        --maxPC "${comps}" \
        --spca_alpha "${SPCA_ALPHA}" \
        --spca_ridge_alpha "${SPCA_RIDGE_ALPHA}" \
        --spca_max_iter "${SPCA_MAX_ITER}" \
        --snmf_alpha "${SNMF_ALPHA:-0.1}" \
        --snmf_l1_ratio "${SNMF_L1_RATIO:-1.0}" \
        --snmf_max_iter "${SNMF_MAX_ITER:-1000}" \
        --sparse_l1_lambda "${sparse_l1}" \
        --hidden_dims ${HIDDEN_DIMS} \
        --activation_functions ${ACTIVATION_FUNCTIONS} \
        --learning_rates ${LEARNING_RATES} \
        --batch_sizes ${BATCH_SIZES} \
        --latent_dims ${LATENT_DIMS} \
        --optimisation "${OPTIMISATION}" \
        --search_objectives ${SEARCH_OBJECTIVES} \
        --n_jobs "${N_JOBS}" \
        --output_labels "${labels_file}" \
        --mincluster ${MINCLUSTER} \
        --mincluster_n ${MINCLUSTER_N} \
        --mincluster_resample_mode "${MINCLUSTER_RESAMPLE_MODE:-fixed}" \
        --use_effective_k_for_fold_merge "${USE_EFFECTIVE_K_FOR_FOLD_MERGE:-FALSE}" \
        --use_cross_fold_effective_k_for_final_run "${USE_CROSS_FOLD_EFFECTIVE_K_FOR_FINAL_RUN:-FALSE}" \
        --internal_ensemble_enabled "${INTERNAL_ENSEMBLE_ENABLED:-FALSE}" \
        --internal_ensemble_bcs "${INTERNAL_ENSEMBLE_BCS:-5}" \
        --internal_ensemble_sample_frac "${INTERNAL_ENSEMBLE_SAMPLE_FRAC:-0.8}" \
        --internal_ensemble_feature_frac "${INTERNAL_ENSEMBLE_FEATURE_FRAC:-1.0}" \
        --TEST "${TEST}" \
        --base_dir "${run_base}" \
      | tee "${log_file}"
    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ] && [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
      echo "!!! Bootstrap ${BIDX} ${method} ${comps} failed (exit $EXIT_CODE), retrying ..."
    fi
  done

  if [ $EXIT_CODE -ne 0 ]; then
    echo "!!! [Fold ${FOLD_INDEX}] Bootstrap ${BIDX} ${method} ${comps} failed after ${MAX_RETRIES} attempts at $(date -Is) !!!"
  fi
  return $EXIT_CODE
}

EXIT_CODE=0
if [ "${RUN_DIMRED_SWEEP:-FALSE}" == "TRUE" ]; then
  for spec in ${DIMRED_SWEEP_SPECS}; do
    IFS=$'\t' read -r method comps sparse_l1 <<< "$(parse_dimred_spec "${spec}" "${MAXPC}" "${SPARSE_L1_LAMBDA}")"
    run_base=$(dimred_run_base "${method}" "${comps}" "${sparse_l1}")
    run_bootstrap_for_spec "${method}" "${comps}" "${sparse_l1}" "${run_base}" || EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
      break
    fi
  done
else
  run_bootstrap_for_spec "${DIMREDUCTION}" "${MAXPC}" "${SPARSE_L1_LAMBDA}" "${BASE_DIR}" || EXIT_CODE=$?
fi


# Record end time and compute duration
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=== [Fold ${FOLD_INDEX}] Bootstrap ${BIDX} finished at $(date -Is) (Duration: ${ELAPSED} seconds) ==="

exit $EXIT_CODE
