#!/bin/bash
#------- qsub option -----------
#PBS -q SQUID-H
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=17:10:00
#PBS -l gpunum_job=2
#PBS -o ../gmlp_logs/gmlp_denoise_diff_model.out
#PBS -e ../gmlp_logs/gmlp_denoise_diff_model.err
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
#------- Program execution -----------

echo "🚀 Starting gMLP denoiser dual-model job"
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
if module avail cudnncd &>/dev/null; then
    module load cudnncd
else
    echo "⚠️ cudnncd module not available; continuing without it"
fi

# === 仮想環境をアクティベート ===
source ${MONO_ROOT}/torch-env/bin/activate

which python
python --version

# === CUDA 環境確認 ===
echo "🎯 CUDA info:"
nvcc --version || echo "nvcc not found"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# === プロジェクトディレクトリへ移動 ===
PROJECT_ROOT="${MONO_ROOT}/gmlp_project"
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "❌ Project root ${PROJECT_ROOT} が見つかりません"
    exit 1
fi
cd "$PROJECT_ROOT"
echo "📁 Current directory: $(pwd)"

echo "Running script: scripts/gmlp_denoise_diff_model.py"

# === 実行パラメータ（環境変数で上書き可能） ===
EPOCHS="${EPOCHS:-1000}"
PATCH="${PATCH:-4}"
DMODEL="${DMODEL:-256}"
DFFN="${DFFN:-1536}"
BATCH="${BATCH:-256}"
LR="${LR:-1e-3}"
INPUT="${INPUT:-${MONO_ROOT}/gmlp_project/data/denoise/mnist_lambda100.npz}"
OUTDIR="${OUTDIR:-${MONO_ROOT}/gmlp_output/denoise_diff_model/lambda100}"
INIT_SEEDA="${INIT_SEEDA:-123}"
INIT_SEEDB="${INIT_SEEDB:-456}"
TRAIN_SEED="${TRAIN_SEED:-2025}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

SCRATCH_BASE="/sqfs/ssd/cm9029/${USER_ID}"
JOB_LABEL="gmlp_denoise_${PBS_JOBID:-manual_$$}"
SCRATCH_JOB_DIR="${SCRATCH_BASE}/${JOB_LABEL}"
SCRATCH_INPUT="${SCRATCH_JOB_DIR}/input"
SCRATCH_OUTPUT="${SCRATCH_JOB_DIR}/output"
FINAL_OUTDIR="$OUTDIR"

if [ ! -d "$SCRATCH_BASE" ]; then
    echo "❌ Scratch base ${SCRATCH_BASE} が見つかりません"
    exit 1
fi

mkdir -p "$SCRATCH_INPUT" "$SCRATCH_OUTPUT" "$FINAL_OUTDIR"

cleanup() {
    local exit_code=$1
    echo "📦 Syncing outputs from ${SCRATCH_OUTPUT} -> ${FINAL_OUTDIR}"
    rsync -a "${SCRATCH_OUTPUT}/" "${FINAL_OUTDIR}/" 2>/dev/null || true
    echo "🧹 Cleaning scratch dir ${SCRATCH_JOB_DIR}"
    rm -rf "$SCRATCH_JOB_DIR"
}
trap 'cleanup "$?"' EXIT

LOCAL_INPUT="${SCRATCH_INPUT}/$(basename "$INPUT")"
LOCAL_OUT="${SCRATCH_OUTPUT}"

echo "💾 Scratch dir : $SCRATCH_JOB_DIR"
df -h "$SCRATCH_BASE" || true
echo "📥 Staging input to scratch..."
STAGING_OK=1
if ! rsync -a "$INPUT" "$SCRATCH_INPUT/"; then
    echo "⚠️ Scratch へのコピーに失敗しました (たとえば quota 超過)。ローカル入力を直接使用します。"
    STAGING_OK=0
fi

echo "🧪 Params: epochs=$EPOCHS, patch=$PATCH, d_model=$DMODEL, d_ffn=$DFFN"
echo "📥 Input : $INPUT"
echo "📤 Output: $OUTDIR (final destination)"
echo "📥 Local input : $LOCAL_INPUT"
echo "📤 Local output: $LOCAL_OUT"

if [ "$STAGING_OK" -eq 0 ]; then
    LOCAL_INPUT="$INPUT"
    LOCAL_OUT="$FINAL_OUTDIR"
fi

# === GPU Resources Splitting ===
IFS=',' read -ra COMMAS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#COMMAS[@]}
HALF=$((NUM_GPUS / 2))
if [ "$HALF" -lt 1 ]; then
    HALF=1
fi

GPUS_TRAIN=("${COMMAS[@]:0:$HALF}")
GPUS_VAL=("${COMMAS[@]:$HALF}")
STR_GPUS_TRAIN=$(IFS=,; echo "${GPUS_TRAIN[*]}")
STR_GPUS_VAL=$(IFS=,; echo "${GPUS_VAL[*]}")

echo "🚀 Parallel Execution Strategy:"
echo "   Train process GPUs : [${STR_GPUS_TRAIN}]"
echo "   Val process GPUs   : [${STR_GPUS_VAL}]"

# Train Process
echo "▶️ Launching TRAIN process..."
CUDA_VISIBLE_DEVICES="$STR_GPUS_TRAIN" python scripts/gmlp_denoise_diff_model.py \
    --epochs "$EPOCHS" \
    --patch "$PATCH" \
    --d_model "$DMODEL" \
    --d_ffn "$DFFN" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --input "$LOCAL_INPUT" \
    --output_dir "$LOCAL_OUT" \
    --init_seedA "$INIT_SEEDA" \
    --init_seedB "$INIT_SEEDB" \
    --train_seed "$TRAIN_SEED" \
    --measure_data "train" \
    ${EXTRA_ARGS} > "${SCRATCH_JOB_DIR}/train.log" 2>&1 &
PID_TRAIN=$!

# Val Process
echo "▶️ Launching VAL process..."
CUDA_VISIBLE_DEVICES="$STR_GPUS_VAL" python scripts/gmlp_denoise_diff_model.py \
    --epochs "$EPOCHS" \
    --patch "$PATCH" \
    --d_model "$DMODEL" \
    --d_ffn "$DFFN" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --input "$LOCAL_INPUT" \
    --output_dir "$LOCAL_OUT" \
    --init_seedA "$INIT_SEEDA" \
    --init_seedB "$INIT_SEEDB" \
    --train_seed "$TRAIN_SEED" \
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
    echo "✅ gMLP denoise diff-model job completed successfully (Both Train & Val)"
    exit_code=0
else
    echo "❌ gMLP denoise diff-model job failed"
    echo "   Train RC: $RC_TRAIN"
    echo "   Val RC  : $RC_VAL"
    exit_code=1
fi

exit "$exit_code"
