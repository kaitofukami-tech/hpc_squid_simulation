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

source /sqfs/work/cm9029/${USER_ID}/torch-env/bin/activate

REPO_ROOT="/sqfs/work/cm9029/${USER_ID}/gmlp_project"
SCRATCH_BASE="/sqfs/ssd/cm9029/${USER_ID}"
SCRATCH_JOB_DIR="${SCRATCH_BASE}/show_denoise_${PBS_JOBID:-manual_$$}"
mkdir -p "$SCRATCH_JOB_DIR"
cleanup() {
  rm -rf "$SCRATCH_JOB_DIR"
}
trap cleanup EXIT
cd "$SCRATCH_JOB_DIR"
echo "📁 Scratch dir: $(pwd)"

INPUT="${INPUT:-/sqfs/work/cm9029/${USER_ID}/gmlp_project/data/denoise/mnist_lambda150.npz}"
SPLIT="${SPLIT:-test}"
SAMPLES="${SAMPLES:-8}"
INDICES="${INDICES:-}"
AS_JSON="${AS_JSON:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/sqfs/work/cm9029/${USER_ID}/output/denoise_samples}"
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
