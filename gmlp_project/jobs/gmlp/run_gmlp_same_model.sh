#!/bin/bash
#------- qsub option -----------
#PBS -q SQUID-H
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=17:10:00
#PBS -l gpunum_job=2
#PBS -o ../logs/gmlp_same_model.out
#PBS -e ../logs/gmlp_same_model.err
#PBS -r n

#------- Program execution -----------

echo "🚀 Starting gMLP dual-model job (same init)"
echo "======================================"
echo "Job ID: $PBS_JOBID"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo ""

echo "📦 Loading Python & GPU modules..."
module purge
module load BaseGPU/2025
module load BasePy/2025
module load python3/3.11
module load cudnncd

source /sqfs/work/cm9029/${USER_ID}/torch-env/bin/activate

which python
python --version

echo "🎯 CUDA info:"
nvcc --version || echo "nvcc not found"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

EPOCHS="${EPOCHS:-1000}"
PATCH="${PATCH:-4}"
DMODEL="${DMODEL:-256}"
DFFN="${DFFN:-1536}"
BATCH="${BATCH:-256}"
LR="${LR:-1e-3}"
INPUT="${INPUT:-/sqfs/work/cm9029/${USER_ID}/gmlp_project/data/mnist.npz}"
OUTDIR="${OUTDIR:-/sqfs/work/cm9029/${USER_ID}/gmlp_output/same_model}"
INIT_SEED="${INIT_SEED:-123}"
TRAIN_SEED_A="${TRAIN_SEED_A:-2025}"
TRAIN_SEED_B="${TRAIN_SEED_B:-4242}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# === 作業ディレクトリ (SSD) 設定 ===
REPO_ROOT="/sqfs/work/cm9029/${USER_ID}/gmlp_project"
SCRATCH_BASE="/sqfs/ssd/cm9029/${USER_ID}"
JOB_LABEL="gmlp_same_${PBS_JOBID:-manual_$$}"
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
    echo "⚠️ Scratch staging failed, fallbackを使用します。"
    STAGING_OK=0
    LOCAL_INPUT="$INPUT"
    LOCAL_OUT="$FINAL_OUTDIR"
fi

echo "🧪 Params: epochs=$EPOCHS, patch=$PATCH, d_model=$DMODEL, d_ffn=$DFFN"
echo "🧪 Seeds : init=$INIT_SEED, trainA=$TRAIN_SEED_A, trainB=$TRAIN_SEED_B"
echo "📥 Input : $INPUT"
echo "📤 Output: $OUTDIR (final destination)"
echo "📥 Local input : $LOCAL_INPUT"
echo "📤 Local output: $LOCAL_OUT"

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
CUDA_VISIBLE_DEVICES="$STR_GPUS_TRAIN" python "$REPO_ROOT/scripts/gmlp_same_model.py" \
    --epochs "$EPOCHS" \
    --patch "$PATCH" \
    --d_model "$DMODEL" \
    --d_ffn "$DFFN" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --input "$LOCAL_INPUT" \
    --output_dir "$LOCAL_OUT" \
    --init_seed "$INIT_SEED" \
    --train_seedA "$TRAIN_SEED_A" \
    --train_seedB "$TRAIN_SEED_B" \
    --measure_data "train" \
    ${EXTRA_ARGS} > "${SCRATCH_JOB_DIR}/train.log" 2>&1 &
PID_TRAIN=$!

# Val Process
echo "▶️ Launching VAL process..."
CUDA_VISIBLE_DEVICES="$STR_GPUS_VAL" python "$REPO_ROOT/scripts/gmlp_same_model.py" \
    --epochs "$EPOCHS" \
    --patch "$PATCH" \
    --d_model "$DMODEL" \
    --d_ffn "$DFFN" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --input "$LOCAL_INPUT" \
    --output_dir "$LOCAL_OUT" \
    --init_seed "$INIT_SEED" \
    --train_seedA "$TRAIN_SEED_A" \
    --train_seedB "$TRAIN_SEED_B" \
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
    echo "✅ gMLP same-model job completed successfully (Both Train & Val)"
    exit_code=0
else
    echo "❌ gMLP same-model job failed"
    echo "   Train RC: $RC_TRAIN"
    echo "   Val RC  : $RC_VAL"
    exit_code=1
fi

exit "$exit_code"
