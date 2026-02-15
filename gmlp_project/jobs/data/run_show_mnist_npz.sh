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

source /sqfs/work/cm9029/${USER_ID}/torch-env/bin/activate

echo "🔍 Python version:"
which python
python --version

REPO_ROOT="/sqfs/work/cm9029/${USER_ID}/gmlp_project"
SCRATCH_BASE="/sqfs/ssd/cm9029/${USER_ID}"
SCRATCH_JOB_DIR="${SCRATCH_BASE}/show_mnist_${PBS_JOBID:-manual_$$}"
mkdir -p "$SCRATCH_JOB_DIR"
cleanup() {
    rm -rf "$SCRATCH_JOB_DIR"
}
trap cleanup EXIT
cd "$SCRATCH_JOB_DIR"
echo "📁 Scratch dir: $(pwd)"

INPUT="${INPUT:-/sqfs/work/cm9029/${USER_ID}/gmlp_project/data/denoise/fashion_mnist_lambda10.npz}"
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
