#!/bin/bash
#------- qsub option -----------
#PBS -q SQUID-H
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=17:10:00
#PBS -l gpunum_job=2
#PBS -o ../gmlp_logs/gmlp_diff_model.out
#PBS -e ../gmlp_logs/gmlp_diff_model.err
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

echo "🚀 Starting gMLP dual-model job"
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
PATCH="${PATCH:-4}"
DMODEL="${DMODEL:-256}"
DFFN="${DFFN:-1536}"
BATCH="${BATCH:-256}"
LR="${LR:-1e-3}"
INPUT="${INPUT:-${MONO_ROOT}/gmlp_project/data/mnist.npz}"
OUTDIR="${OUTDIR:-${MONO_ROOT}/gmlp_output/diff_model}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# === 作業ディレクトリ (SSD) 設定 ===
REPO_ROOT="${MONO_ROOT}/gmlp_project"
SCRATCH_BASE="/sqfs/ssd/cm9029/${USER_ID}"
JOB_LABEL="gmlp_diff_${PBS_JOBID:-manual_$$}"
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

echo "🧪 Params: epochs=$EPOCHS, patch=$PATCH, d_model=$DMODEL, d_ffn=$DFFN"
echo "📥 Input : $INPUT"
echo "📤 Output: $OUTDIR (final destination)"
echo "📥 Local input : $LOCAL_INPUT"
echo "📤 Local output: $LOCAL_OUT"

# === GPU Resources Splitting ===
IFS=',' read -ra COMMAS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#COMMAS[@]}
HALF=$((NUM_GPUS / 2))
if [ "$HALF" -lt 1 ]; then
    HALF=1 # Fallback for single GPU case (run sequentially or error? defaulting to 1 might overlap)
fi

# Arrays for GPU sets
GPUS_TRAIN=("${COMMAS[@]:0:$HALF}")
GPUS_VAL=("${COMMAS[@]:$HALF}")

# Convert back to comma-separated strings
STR_GPUS_TRAIN=$(IFS=,; echo "${GPUS_TRAIN[*]}")
STR_GPUS_VAL=$(IFS=,; echo "${GPUS_VAL[*]}")

echo "🚀 Parallel Execution Strategy:"
echo "   Train process GPUs : [${STR_GPUS_TRAIN}]"
echo "   Val process GPUs   : [${STR_GPUS_VAL}]"

# Train Process
echo "▶️ Launching TRAIN process..."
CUDA_VISIBLE_DEVICES="$STR_GPUS_TRAIN" python "$REPO_ROOT/scripts/gmlp_diff_model.py" \
    --epochs "$EPOCHS" \
    --patch "$PATCH" \
    --d_model "$DMODEL" \
    --d_ffn "$DFFN" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --input "$LOCAL_INPUT" \
    --output_dir "$LOCAL_OUT" \
    --measure_data "train" \
    ${EXTRA_ARGS} > "${SCRATCH_JOB_DIR}/train.log" 2>&1 &
PID_TRAIN=$!

# Val Process
echo "▶️ Launching VAL process..."
CUDA_VISIBLE_DEVICES="$STR_GPUS_VAL" python "$REPO_ROOT/scripts/gmlp_diff_model.py" \
    --epochs "$EPOCHS" \
    --patch "$PATCH" \
    --d_model "$DMODEL" \
    --d_ffn "$DFFN" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --input "$LOCAL_INPUT" \
    --output_dir "$LOCAL_OUT" \
    --measure_data "val" \
    ${EXTRA_ARGS} > "${SCRATCH_JOB_DIR}/val.log" 2>&1 &
PID_VAL=$!

echo "⏳ Waiting for processes (Train: $PID_TRAIN, Val: $PID_VAL)..."
wait $PID_TRAIN
RC_TRAIN=$?
wait $PID_VAL
RC_VAL=$?

echo "📄 Train Log Tail:"
tail -n 5 "${SCRATCH_JOB_DIR}/train.log"
echo "📄 Val Log Tail:"
tail -n 5 "${SCRATCH_JOB_DIR}/val.log"

if [ $RC_TRAIN -eq 0 ] && [ $RC_VAL -eq 0 ]; then
    echo "✅ gMLP diff-model job completed successfully (Both Train & Val)"
    exit_code=0
else
    echo "❌ gMLP diff-model job failed"
    echo "   Train RC: $RC_TRAIN"
    echo "   Val RC  : $RC_VAL"
    exit_code=1
fi

exit "$exit_code"
