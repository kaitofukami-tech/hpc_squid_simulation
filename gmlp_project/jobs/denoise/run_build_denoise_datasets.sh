#!/bin/bash
#------- qsub option -----------
#PBS -q DBG
#PBS --group=cm9029
#PBS -m abe
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=00:10:00
#PBS -l cpunum_job=8
#PBS -o ../logs/build_denoise_datasets.out
#PBS -e ../logs/build_denoise_datasets.err
#PBS -r n

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONO_ROOT=""
dir="$SCRIPT_DIR"
while [ "$dir" != "/" ]; do
  if [ -d "$dir/.git" ]; then
    MONO_ROOT="$dir"
    break
  fi
  dir="$(dirname "$dir")"
done
if [ -z "$MONO_ROOT" ]; then
  MONO_ROOT="$SCRIPT_DIR"
fi
REPO_ROOT="${REPO_ROOT:-$MONO_ROOT}"
set -euo pipefail

echo "🚀 Starting denoise dataset builder"
echo "Job ID : ${PBS_JOBID:-manual}"
echo "Host   : $(hostname)"
echo "Time   : $(date)"
echo ""

module purge
module load BasePy/2025
module load python3/3.11

source ${MONO_ROOT}/torch-env/bin/activate

REPO_ROOT="${MONO_ROOT}/gmlp_project"
SCRATCH_BASE="/sqfs/ssd/cm9029/${USER_ID}"
SCRATCH_JOB_DIR="${SCRATCH_BASE}/build_denoise_${PBS_JOBID:-manual_$$}"
mkdir -p "$SCRATCH_JOB_DIR"
cleanup() {
    rm -rf "$SCRATCH_JOB_DIR"
}
trap cleanup EXIT
cd "$SCRATCH_JOB_DIR"
echo "📁 Scratch dir: $(pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-${MONO_ROOT}/gmlp_project/data/denoise}"
LAMBDAS="${LAMBDAS:-150 200 500}"
DATASETS="${DATASETS:-mnist fashion_mnist}"
SOURCE_DIR="${SOURCE_DIR:-${MONO_ROOT}/gmlp_project/data}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-1}"

CMD=(python "$REPO_ROOT/scripts/build_denoise_datasets.py"
    --source-dir "$SOURCE_DIR"
    --output-root "$OUTPUT_ROOT"
    --seed "$SEED"
    --datasets $DATASETS
    --lambdas $LAMBDAS
)

if [[ "$OVERWRITE" -eq 1 ]]; then
    CMD+=(--overwrite)
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    CMD+=(--dry-run)
fi

echo "🧮 Command: ${CMD[*]}"
"${CMD[@]}"

status=$?
echo ""
echo "Job finished with status $status at $(date)"
exit $status
