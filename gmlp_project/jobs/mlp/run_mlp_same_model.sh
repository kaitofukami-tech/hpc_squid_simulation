#!/bin/bash
#------- qsub option -----------
#PBS -q SQUID
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=01:30:00
#PBS -l gpunum_job=4
#PBS -o ../../logs/mlp_same_model.out
#PBS -e ../../logs/mlp_same_model.err
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
export PYTHONPATH="${MONO_ROOT}:${PYTHONPATH:-}"
#------- Program execution -----------

echo "🚀 Starting MLP dual-model job (same)"
echo "======================================"
echo "Job ID: $PBS_JOBID"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo ""

# === モジュール環境のセットアップ ===
echo "📦 Loading Python & GPU modules..."
module purge
module load BaseGPU/2025
module load BasePy/2025
module load python3/3.11
module load cudnncd

# === 仮想環境をアクティベート ===
source ${MONO_ROOT}/torch-env/bin/activate

which python
python --version

# === CUDA 環境確認 ===
echo "🎯 CUDA info:"
nvcc --version || echo "nvcc not found"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# === 実行パラメータ（環境変数で上書き可能） ===
EPOCHS="${EPOCHS:-1000}"
DMODEL="${DMODEL:-256}"
DFFN="${DFFN:-1024}"
BLOCKS="${BLOCKS:-10}"
BATCH="${BATCH:-256}"
LR="${LR:-1e-3}"
INPUT="${INPUT:-${MONO_ROOT}/gmlp_project/data/mnist.npz}"
OUTDIR="${OUTDIR:-${MONO_ROOT}/mlp_output/same_model}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# === 作業ディレクトリ (SSD) 設定 ===
REPO_ROOT="${MONO_ROOT}/gmlp_project"
SCRATCH_BASE="/sqfs/ssd/cm9029/${USER_ID}"
JOB_LABEL="mlp_same_${PBS_JOBID:-manual_$$}"
SCRATCH_JOB_DIR="${SCRATCH_BASE}/${JOB_LABEL}"
SCRATCH_INPUT="${SCRATCH_JOB_DIR}/input"
SCRATCH_OUTPUT="${SCRATCH_JOB_DIR}/output"
FINAL_OUTDIR="$OUTDIR"
mkdir -p "$SCRATCH_JOB_DIR" "$SCRATCH_INPUT" "$SCRATCH_OUTPUT" "$FINAL_OUTDIR"
cd "$SCRATCH_JOB_DIR"
echo "📁 Scratch dir: $(pwd)"

cleanup() {
    local exit_code=$1
    if [ "${STAGING_OK:-1}" -eq 1 ]; then
        echo "📦 Syncing outputs from ${SCRATCH_OUTPUT} -> ${FINAL_OUTDIR}"
        rsync -a "${SCRATCH_OUTPUT}/" "${FINAL_OUTDIR}/" 2>/dev/null || true
    fi
    echo "🧹 Cleaning scratch dir ${SCRATCH_JOB_DIR}"
    rm -rf "$SCRATCH_JOB_DIR"
}
trap 'cleanup "$?"' EXIT

LOCAL_INPUT="${SCRATCH_INPUT}/$(basename "$INPUT")"
LOCAL_OUT="${SCRATCH_OUTPUT}"
STAGING_OK=1

echo "💾 Scratch dir : $SCRATCH_JOB_DIR"
df -h "$SCRATCH_BASE" || true
echo "📥 Staging input to scratch..."
if ! rsync -a "$INPUT" "$SCRATCH_INPUT/"; then
    echo "⚠️ Scratch staging failed, fallback to main storage."
    STAGING_OK=0
    LOCAL_INPUT="$INPUT"
    LOCAL_OUT="$FINAL_OUTDIR"
fi

echo "🧪 Params: epochs=$EPOCHS, d_model=$DMODEL, d_ffn=$DFFN, blocks=$BLOCKS"
echo "📥 Input : $INPUT"
echo "📤 Output: $OUTDIR (final destination)"
echo "📥 Local input : $LOCAL_INPUT"
echo "📤 Local output: $LOCAL_OUT"

# --- Run 1: Measure TRAIN (GPU 0) ---
echo "🚀 Launching TRAIN measurement on device 0..."
CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/scripts/mlp_same_model.py" \
    --epochs "$EPOCHS" \
    --d_model "$DMODEL" \
    --d_ffn "$DFFN" \
    --num_blocks "$BLOCKS" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --input "$LOCAL_INPUT" \
    --output_dir "$LOCAL_OUT" \
    --measure_data "train" \
    ${EXTRA_ARGS} > "${LOCAL_OUT}/train_log.out" 2>&1 &
PID_TRAIN=$!

# --- Run 2: Measure VAL (GPU 1) ---
echo "🚀 Launching VAL measurement on device 1..."
CUDA_VISIBLE_DEVICES=1 python "$REPO_ROOT/scripts/mlp_same_model.py" \
    --epochs "$EPOCHS" \
    --d_model "$DMODEL" \
    --d_ffn "$DFFN" \
    --num_blocks "$BLOCKS" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --input "$LOCAL_INPUT" \
    --output_dir "$LOCAL_OUT" \
    --measure_data "val" \
    ${EXTRA_ARGS} > "${LOCAL_OUT}/val_log.out" 2>&1 &
PID_VAL=$!

echo "⏳ Waiting for PIDs: $PID_TRAIN (train), $PID_VAL (val)"
wait $PID_TRAIN
EXIT_TRAIN=$?
wait $PID_VAL
EXIT_VAL=$?

echo "🏁 Exit codes: Train=$EXIT_TRAIN, Val=$EXIT_VAL"
echo "Done at: $(date)"

if [ $EXIT_TRAIN -eq 0 ] && [ $EXIT_VAL -eq 0 ]; then
    echo "✅ MLP same-model job completed successfully (Both Train & Val)"
    exit_code=0
else
    echo "❌ MLP same-model job failed (Train=$EXIT_TRAIN, Val=$EXIT_VAL)"
    echo "--- Train Log tail ---"
    tail -10 "${LOCAL_OUT}/train_log.out" 2>/dev/null || true
    echo "--- Val Log tail ---"
    tail -10 "${LOCAL_OUT}/val_log.out" 2>/dev/null || true
    exit_code=1
fi

exit "$exit_code"
