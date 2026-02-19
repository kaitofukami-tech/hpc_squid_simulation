#!/bin/bash
#------- qsub option -----------
#PBS -q DBG
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=00:10:00
#PBS -l cpunum_job=4
#PBS -o ../logs/split_mnist_by_label.out
#PBS -e ../logs/split_mnist_by_label.err
#PBS -r n

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONO_ROOT=""

find_repo_root() {
  local start_dir="$1"
  local dir="$start_dir"
  while [ "$dir" != "/" ]; do
    if [ -d "$dir/gmlp_project" ]; then
      echo "$dir"
      return 0
    fi
    if [ -d "$dir/scripts" ] && [ -d "$dir/data" ]; then
      echo "$(cd "$dir/.." && pwd)"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}

if [ -n "${REPO_ROOT:-}" ]; then
  if found="$(find_repo_root "$REPO_ROOT")"; then
    MONO_ROOT="$found"
  fi
fi

if [ -z "$MONO_ROOT" ]; then
  START_DIR="${PBS_O_WORKDIR:-$SCRIPT_DIR}"
  if found="$(find_repo_root "$START_DIR")"; then
    MONO_ROOT="$found"
  fi
fi

if [ -z "$MONO_ROOT" ]; then
  if found="$(find_repo_root "$SCRIPT_DIR")"; then
    MONO_ROOT="$found"
  fi
fi

if [ -z "$MONO_ROOT" ]; then
  echo "Repo root not found from REPO_ROOT/PBS_O_WORKDIR/SCRIPT_DIR"
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-$MONO_ROOT}"
export PYTHONPATH="${MONO_ROOT}/gmlp_project/src:${MONO_ROOT}:${PYTHONPATH:-}"
#------- Program execution -----------

echo "🚀 Starting MNIST split job"
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
SCRATCH_JOB_DIR="${SCRATCH_BASE}/split_mnist_${PBS_JOBID:-manual_$$}"
mkdir -p "$SCRATCH_JOB_DIR"
cleanup() {
  rm -rf "$SCRATCH_JOB_DIR"
}
trap cleanup EXIT
cd "$SCRATCH_JOB_DIR"
echo "📁 Scratch dir: $(pwd)"

echo "Running script: scripts/split_mnist_by_label.py"

INPUT="${INPUT:-${MONO_ROOT}/gmlp_project/data/mnist.npz}"
OUTDIR="${OUTDIR:-${MONO_ROOT}/gmlp_project/data/mnist_by_label}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "📥 Input : $INPUT"
echo "📤 Output: $OUTDIR"

CMD=(python "$REPO_ROOT/scripts/split_mnist_by_label.py" --input "$INPUT" --output_dir "$OUTDIR")
if [ -n "$EXTRA_ARGS" ]; then
    CMD+=($EXTRA_ARGS)
fi

echo "🧮 Command: ${CMD[*]}"
"${CMD[@]}"

exit_code=$?
echo "🏁 Exit code: $exit_code"
echo "Done at: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ MNIST split job completed successfully"
else
    echo "❌ MNIST split job failed"
    tail -20 ../logs/split_mnist_by_label.err 2>/dev/null || true
fi
