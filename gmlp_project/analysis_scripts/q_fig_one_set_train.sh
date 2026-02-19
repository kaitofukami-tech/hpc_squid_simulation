#!/bin/bash
#------- qsub option -----------
#PBS -q SQUID-H
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=01:00:00
#PBS -l cpunum_job=8
#PBS -l gpunum_job=1
#PBS -o ~/q2_single_analysis_log.out
#PBS -e ~/q2_single_analysis_log.err
#PBS -r n
#PBS -V

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONO_ROOT=""

# 1) Honor REPO_ROOT if provided (can point to repo root or gmlp_project).
if [ -n "${REPO_ROOT:-}" ]; then
  if [ -d "$REPO_ROOT/.git" ] || [ -d "$REPO_ROOT/gmlp_project" ]; then
    MONO_ROOT="$REPO_ROOT"
  elif [ -d "$REPO_ROOT/scripts" ] && [ -d "$REPO_ROOT/data" ]; then
    MONO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
  fi
fi

# 2) Prefer submit directory if available (PBS copies this script into a jobfile dir).
if [ -z "$MONO_ROOT" ]; then
  START_DIR="${PBS_O_WORKDIR:-$SCRIPT_DIR}"
  dir="$START_DIR"
  while [ "$dir" != "/" ]; do
    if [ -d "$dir/.git" ] || [ -d "$dir/gmlp_project" ]; then
      MONO_ROOT="$dir"
      break
    fi
    if [ -d "$dir/scripts" ] && [ -d "$dir/data" ]; then
      MONO_ROOT="$(cd "$dir/.." && pwd)"
      break
    fi
    dir="$(dirname "$dir")"
  done
fi

# 3) Fallback to script dir (jobfile dir); warn if repo not found.
if [ -z "$MONO_ROOT" ]; then
  MONO_ROOT="$SCRIPT_DIR"
fi

if [ ! -d "$MONO_ROOT/gmlp_project" ]; then
  echo "❌ Repo root not found. MONO_ROOT=$MONO_ROOT"
  echo "   Set REPO_ROOT to the repository root (the directory containing gmlp_project)."
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-$MONO_ROOT}"
set -euo pipefail

echo "🚀 Starting single q_inv figure job"
JOB_ID="${PBS_JOBID:-manual}"
echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "Time: $(date)"

module purge
module load BaseGPU/2025
module load BasePy/2025
module load python3/3.11
source ${MONO_ROOT}/torch-env/bin/activate

REPO_ROOT="${MONO_ROOT}"
SCRATCH_BASE="/sqfs/ssd/cm9029/${USER_ID}"
SCRATCH_JOB_DIR="${SCRATCH_BASE}/qfig_one_set_${JOB_ID}_$$"
mkdir -p "$SCRATCH_JOB_DIR"
cleanup() {
  rm -rf "$SCRATCH_JOB_DIR"
}
trap cleanup EXIT
cd "$SCRATCH_JOB_DIR"
echo "📁 Scratch dir: $(pwd)"

# Optional environment overrides for flexibility
ALL_LAYERS="${ALL_LAYERS:-1}"   # default: all layers
LAYER_INDEX="${LAYER_INDEX:-0}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/thesis/gmlp_randomnet}"
TITLE="${TITLE:-q_inv (gMLP random)}"

if [ -n "$ALL_LAYERS" ] && [ "$ALL_LAYERS" != "0" ]; then
  LAYER_ARGS=(--all-layers)
else
  LAYER_ARGS=(--layer-index "$LAYER_INDEX")
fi

CMD=(python "${REPO_ROOT}/analysis_scripts/qinv_calc_fig_one_set.py" \
  --spin_file_a ${MONO_ROOT}/gmlp_output/random/gmlp_diff_model_p4_mnist_input_fashion/gmlp_spinA_train_D256_F1536_L10_M1000_seedA123_recomputed_A.pkl \
  --spin_file_b ${MONO_ROOT}/gmlp_output/random/gmlp_diff_model_p4_mnist_input_fashion/gmlp_spinB_train_D256_F1536_L10_M1000_seedB456_recomputed_B.pkl \
  ${METRICS_A:+--metrics_a "$METRICS_A"} \
  ${METRICS_B:+--metrics_b "$METRICS_B"} \
  --label        GMLP_p4_random \
  "${LAYER_ARGS[@]}" \
  --title        "${TITLE}" \
  --output-dir   "${OUTDIR}")

echo "🧮 Command: ${CMD[*]}"
"${CMD[@]}"

exit_code=$?
echo "🏁 Exit code: $exit_code"
echo "Done at: $(date)"
exit $exit_code
