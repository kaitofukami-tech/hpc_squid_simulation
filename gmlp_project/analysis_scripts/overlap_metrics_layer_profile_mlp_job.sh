#!/bin/bash
#------- qsub option -----------
#PBS -q DBG
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=00:10:00
#PBS -l cpunum_job=16
#PBS -o ~/mlp_overlap_layer_profile.out
#PBS -e ~/mlp_overlap_layer_profile.err
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

echo "🚀 Starting MLP overlap layer-profile job"
echo "Job ID : ${PBS_JOBID:-manual}"
echo "Host   : $(hostname)"
echo "Time   : $(date)"
echo ""

module purge
module load BaseGPU/2025
module load BasePy/2025
module load python3/3.11
source ${MONO_ROOT}/torch-env/bin/activate

REPO_ROOT="${MONO_ROOT}"
SCRATCH_BASE="/sqfs/ssd/cm9029/${USER_ID}"
SCRATCH_JOB_DIR="${SCRATCH_BASE}/mlp_overlap_layer_${PBS_JOBID:-manual_$$}"
mkdir -p "$SCRATCH_JOB_DIR"
cleanup() {
  rm -rf "$SCRATCH_JOB_DIR"
}
trap cleanup EXIT
cd "$SCRATCH_JOB_DIR"
echo "📁 Scratch dir: $(pwd)"

# Default paths (PLEASE SET THESE ENV VARS BEFORE RUNNING)
# Example: 
# export SPIN_A1="/path/to/mlp_spinA.pkl"
# ...
SPIN_A1="${SPIN_A1:-}"
SPIN_B1="${SPIN_B1:-}"
SPIN_A2="${SPIN_A2:-}"
SPIN_B2="${SPIN_B2:-}"

if [[ -z "$SPIN_A1" || -z "$SPIN_B1" || -z "$SPIN_A2" || -z "$SPIN_B2" ]]; then
  echo "Error: SPIN_A1, SPIN_B1, SPIN_A2, SPIN_B2 environment variables must be set."
  exit 1
fi

EPOCHS="${EPOCHS:-1 10 50}"
DEFAULT_OUTPUT="$REPO_ROOT/mlp_qinv_out/overlap_layer_profile.png"
OUTPUT="${OUTPUT:-$DEFAULT_OUTPUT}"
if [[ "$OUTPUT" != /* ]]; then
  OUTPUT="$REPO_ROOT/$OUTPUT"
fi
mkdir -p "$(dirname "$OUTPUT")"
TITLE="${TITLE:-MLP Overlap metrics vs layer}"
LABEL1="${LABEL1:-mlp_set1}"
LABEL2="${LABEL2:-mlp_set2}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(python "$REPO_ROOT/analysis_scripts/overlap_metrics_layer_profile_mlp.py"
    --spin_file_a1 "$SPIN_A1"
    --spin_file_b1 "$SPIN_B1"
    --spin_file_a2 "$SPIN_A2"
    --spin_file_b2 "$SPIN_B2"
    --label1 "$LABEL1"
    --label2 "$LABEL2"
    --epochs $EPOCHS
    --output "$OUTPUT"
    --title "$TITLE"
)

if [ -n "$EXTRA_ARGS" ]; then
  CMD+=($EXTRA_ARGS)
fi

echo "🧮 Command: ${CMD[*]}"
"${CMD[@]}"

status=$?
echo ""
echo "Job finished with status $status at $(date)"
exit $status
