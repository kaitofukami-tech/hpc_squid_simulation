#!/bin/bash
#------- qsub option -----------
#PBS -q DBG
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=00:05:00
#PBS -l cpunum_job=2
#PBS -o ../logs/show_mnist_npz.out
#PBS -e ../logs/show_mnist_npz.err
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
#------- Program execution -----------

echo "🚀 Starting MNIST inspect job"
echo "======================================"
echo "Job ID: $PBS_JOBID"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo ""

echo "📦 Loading Python modules..."
module purge
module load BasePy/2025
module load python3/3.11

source ${MONO_ROOT}/torch-env/bin/activate

echo "🔍 Python version:"
which python
python --version

REPO_ROOT="${MONO_ROOT}/gmlp_project"
SCRATCH_BASE="/sqfs/ssd/cm9029/${USER_ID}"
SCRATCH_JOB_DIR="${SCRATCH_BASE}/show_mnist_${PBS_JOBID:-manual_$$}"
mkdir -p "$SCRATCH_JOB_DIR"
cleanup() {
    rm -rf "$SCRATCH_JOB_DIR"
}
trap cleanup EXIT
cd "$SCRATCH_JOB_DIR"
echo "📁 Scratch dir: $(pwd)"

INPUT="${INPUT:-${MONO_ROOT}/gmlp_project/data/denoise/fashion_mnist_lambda10.npz}"
SHOW_SAMPLES="${SHOW_SAMPLES:-0}"
LABEL="${LABEL:-}"
LABEL_SAMPLES="${LABEL_SAMPLES:-10}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "📥 Input : $INPUT"
if [ "$SHOW_SAMPLES" = "1" ]; then
    EXTRA_ARGS="--show-samples ${EXTRA_ARGS}"
fi
if [ -n "$LABEL" ]; then
    EXTRA_ARGS="--label $LABEL --label-samples $LABEL_SAMPLES ${EXTRA_ARGS}"
fi

CMD=(python "$REPO_ROOT/scripts/show_mnist_npz.py" --input "$INPUT")
if [ -n "$EXTRA_ARGS" ]; then
    CMD+=($EXTRA_ARGS)
fi

echo "🧮 Command: ${CMD[*]}"
"${CMD[@]}"

exit_code=$?
echo "🏁 Exit code: $exit_code"
echo "Done at: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ MNIST inspect job completed successfully"
else
    echo "❌ MNIST inspect job failed"
    tail -20 ../logs/show_mnist_npz.err 2>/dev/null || true
fi
