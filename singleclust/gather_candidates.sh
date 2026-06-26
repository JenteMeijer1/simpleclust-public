#!/bin/bash
#SBATCH --job-name=grid_gather
#SBATCH --output=logs/gather_%A.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00

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

# Use all allocated CPUs for parallelisation
N_JOBS=${SLURM_CPUS_PER_TASK:-1}

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

run_gather_for_spec() {
  local method="$1"
  local comps="$2"
  local sparse_l1="$3"
  local run_base="$4"
  local search_root="${run_base}/intermediates/fold${FOLD_INDEX}/search"
  local current_file="${search_root}/candidates_scored_fold${FOLD_INDEX}.pkl"
  local next_file="${search_root}/candidates_scored_fold${FOLD_INDEX}.pkl"
  local init_file="${search_root}/population_init_fold${FOLD_INDEX}.pkl"
  local log_file="${run_base}/logs/fold${FOLD_INDEX}_gather.log"

  mkdir -p "${run_base}/logs"

  apptainer exec ${SIF} \
    python -u singleclust/full_pipeline_singleclust.py \
      --mode gather \
      --input_csv          "${INPUT_CSV}" \
      --meta_csv           "${META_CSV}" \
      --fold_index         "${FOLD_INDEX}" \
      --bootstrap_dir      "${search_root}" \
      --population_dir     "${search_root}" \
      --population_file    "${current_file}" \
      --population_initial_file "${init_file}" \
      --n_folds            "${N_FOLDS}" \
      --n_bootstrap        "${N_BOOTSTRAP}" \
      --col_threshold      "${COL_THRESHOLD}" \
      --row_threshold      "${ROW_THRESHOLD}" \
      --skew_threshold     "${SKEW_THRESHOLD}" \
      --scaler_type        "${SCALER_TYPE}" \
      --dim_reduction      "${method}" \
      --maxPC              "${comps}" \
      --spca_alpha         "${SPCA_ALPHA:-1.0}" \
      --spca_ridge_alpha   "${SPCA_RIDGE_ALPHA:-0.01}" \
      --spca_max_iter      "${SPCA_MAX_ITER:-1000}" \
      --snmf_alpha         "${SNMF_ALPHA:-0.1}" \
      --snmf_l1_ratio      "${SNMF_L1_RATIO:-1.0}" \
      --snmf_max_iter      "${SNMF_MAX_ITER:-1000}" \
      --sparse_l1_lambda   "${sparse_l1}" \
      --hidden_dims        ${HIDDEN_DIMS} \
      --activation_functions ${ACTIVATION_FUNCTIONS} \
      --learning_rates     ${LEARNING_RATES} \
      --batch_sizes        ${BATCH_SIZES} \
      --latent_dims        ${LATENT_DIMS} \
      --optimisation       "${OPTIMISATION}" \
      --search_objectives  ${SEARCH_OBJECTIVES} \
      --output_population  "${next_file}" \
      --n_jobs             "${N_JOBS}" \
      --mincluster         ${MINCLUSTER} \
      --mincluster_n       ${MINCLUSTER_N} \
      --mincluster_resample_mode "${MINCLUSTER_RESAMPLE_MODE:-fixed}" \
      --use_effective_k_for_fold_merge "${USE_EFFECTIVE_K_FOR_FOLD_MERGE:-FALSE}" \
      --use_cross_fold_effective_k_for_final_run "${USE_CROSS_FOLD_EFFECTIVE_K_FOR_FINAL_RUN:-FALSE}" \
      --TEST               "${TEST}" \
      --base_dir           "${run_base}" \
    | tee -a "${log_file}"
  return ${PIPESTATUS[0]}
}

EXIT_CODE=0
if [ "${RUN_DIMRED_SWEEP:-FALSE}" == "TRUE" ]; then
  for spec in ${DIMRED_SWEEP_SPECS}; do
    IFS=$'\t' read -r method comps sparse_l1 <<< "$(parse_dimred_spec "${spec}" "${MAXPC}" "${SPARSE_L1_LAMBDA}")"
    run_base=$(dimred_run_base "${method}" "${comps}" "${sparse_l1}")
    run_gather_for_spec "${method}" "${comps}" "${sparse_l1}" "${run_base}" || EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
      break
    fi
  done
else
  run_gather_for_spec "${DIMREDUCTION}" "${MAXPC}" "${SPARSE_L1_LAMBDA}" "${BASE_DIR}" || EXIT_CODE=$?
fi

exit ${EXIT_CODE}
