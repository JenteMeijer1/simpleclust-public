#!/bin/bash
#SBATCH --job-name=report
#SBATCH --output=logs/postprocess_report.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00

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
mkdir -p "${LOGS_DIR}" "${PLOTS_DIR}"

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

REPORT_PDF=${REPORT_PDF:-"${PLOTS_DIR}/postprocess_report.pdf"}
NOTEBOOK=${NOTEBOOK:-"notebooks/multiclust_extended/Main.ipynb"}
N_JOBS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${N_JOBS}}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-${N_JOBS}}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-${N_JOBS}}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-${N_JOBS}}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-${N_JOBS}}

echo "Using Apptainer image: ${SIF}"
echo "Creating post-processing report: ${REPORT_PDF}"
echo "Using ${N_JOBS} CPU thread(s) for numerical libraries"

apptainer exec "${SIF}" \
  python -u postprocess_report.py \
    --notebook "${NOTEBOOK}" \
    --base_dir "${BASE_DIR}" \
    --output_pdf "${REPORT_PDF}" \
    --start_heading "# Full pipeline results"

exit ${PIPESTATUS[0]}
