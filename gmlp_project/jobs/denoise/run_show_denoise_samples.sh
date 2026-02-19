#!/bin/bash
#------- qsub option -----------
#PBS -q DBG
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=00:05:00
#PBS -l cpunum_job=2
#PBS -o ../logs/show_denoise_samples.out
#PBS -e ../logs/show_denoise_samples.err
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
set -euo pipefail

echo "🚀 Starting denoise sample viewer"
echo "======================================"
echo "Job ID: ${PBS_JOBID:-manual}"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo ""

module purge
module load BasePy/2025
module load python3/3.11

source ${MONO_ROOT}/torch-env/bin/activate

REPO_ROOT="${MONO_ROOT}/gmlp_project"
SCRATCH_BASE="/sqfs/ssd/cm9029/${USER_ID}"
SCRATCH_JOB_DIR="${SCRATCH_BASE}/show_denoise_${PBS_JOBID:-manual_$$}"
mkdir -p "$SCRATCH_JOB_DIR"
cleanup() {
  rm -rf "$SCRATCH_JOB_DIR"
}
trap cleanup EXIT
cd "$SCRATCH_JOB_DIR"
echo "📁 Scratch dir: $(pwd)"

INPUT="${INPUT:-${MONO_ROOT}/gmlp_project/data/denoise/mnist_lambda150.npz}"
SPLIT="${SPLIT:-test}"
SAMPLES="${SAMPLES:-8}"
INDICES="${INDICES:-}"
AS_JSON="${AS_JSON:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${MONO_ROOT}/output/denoise_samples}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(python "$REPO_ROOT/scripts/show_denoise_samples.py" --input "$INPUT" --split "$SPLIT" --output-dir "$OUTPUT_DIR")
if [ -n "$INDICES" ]; then
  CMD+=(--indices $INDICES)
else
  CMD+=(--samples "$SAMPLES")
fi
if [ "$AS_JSON" = "1" ]; then
  CMD+=(--as-json)
fi
if [ -n "$EXTRA_ARGS" ]; then
  CMD+=($EXTRA_ARGS)
fi

echo "🧮 Command: ${CMD[*]}"
"${CMD[@]}"

status=$?
echo ""
echo "Job finished with status $status at $(date)"
exit $status
