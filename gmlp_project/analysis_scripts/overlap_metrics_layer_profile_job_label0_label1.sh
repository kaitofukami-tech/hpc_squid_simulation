#!/bin/bash
#------- qsub option -----------
#PBS -q SQUID-H
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=01:00:00
#PBS -l cpunum_job=1
#PBS -o /sqfs/work/cm9029/${USER_ID}/gmlp_logs/overlap_layer_profile_label0_label1.out
#PBS -e /sqfs/work/cm9029/${USER_ID}/gmlp_logs/overlap_layer_profile_label0_label1.err
#PBS -r n

set -euo pipefail

echo "🚀 Starting overlap layer-profile job (Label 0 vs Label 1)"
echo "Job ID : ${PBS_JOBID:-manual}"
echo "Host   : $(hostname)"
echo "Time   : $(date)"
echo ""

module purge
module load BaseGPU/2025
module load BasePy/2025
module load python3/3.11
source /sqfs/work/cm9029/${USER_ID}/torch-env/bin/activate

REPO_ROOT="/sqfs/work/cm9029/${USER_ID}"
SCRATCH_BASE="/sqfs/ssd/cm9029/${USER_ID}"
SCRATCH_JOB_DIR="${SCRATCH_BASE}/overlap_layer_${PBS_JOBID:-manual_$$}"
mkdir -p "$SCRATCH_JOB_DIR"
cleanup() {
  rm -rf "$SCRATCH_JOB_DIR"
}
trap cleanup EXIT
cd "$SCRATCH_JOB_DIR"
echo "📁 Scratch dir: $(pwd)"

# Pointing to the outputs of the recompute job
RECOMPUTE_BASE="/sqfs/work/cm9029/${USER_ID}/gmlp_output/recompute/Label0_Label1_Diff"

SPIN_A1="${SPIN_A1:-$RECOMPUTE_BASE/on_Label0/spin_A.pkl}"
SPIN_B1="${SPIN_B1:-$RECOMPUTE_BASE/on_Label0/spin_B.pkl}"
SPIN_A2="${SPIN_A2:-$RECOMPUTE_BASE/on_Label1/spin_A.pkl}"
SPIN_B2="${SPIN_B2:-$RECOMPUTE_BASE/on_Label1/spin_B.pkl}"

# Recompute spins if missing
PROJECT_ROOT="$REPO_ROOT/gmlp_project"
MODEL_CHECKPOINT_DIR="${MODEL_CHECKPOINT_DIR:-/sqfs/work/cm9029/${USER_ID}/gmlp_output/diff_model/checkpoints/run_gmlp_20260101-020554_p4_train}"
CHECKPOINT_ROOT_A="${CHECKPOINT_ROOT_A:-$MODEL_CHECKPOINT_DIR/A}"
CHECKPOINT_ROOT_B="${CHECKPOINT_ROOT_B:-$MODEL_CHECKPOINT_DIR/B}"
SPIN_SRC_A="${SPIN_SRC_A:-/sqfs/work/cm9029/${USER_ID}/gmlp_output/diff_model/run_gmlp_20260101-020554_p4_train/gmlp_spinA_train_D256_F1536_L10_M1000_seedA123.pkl}"
SPIN_SRC_B="${SPIN_SRC_B:-/sqfs/work/cm9029/${USER_ID}/gmlp_output/diff_model/run_gmlp_20260101-020554_p4_train/gmlp_spinB_train_D256_F1536_L10_M1000_seedB456.pkl}"
DATA_LABEL0="${DATA_LABEL0:-$PROJECT_ROOT/data/mnist_by_label/mnist_label0.npz}"
DATA_LABEL1="${DATA_LABEL1:-$PROJECT_ROOT/data/mnist_by_label/mnist_label1.npz}"
MEASURE_DATA="${MEASURE_DATA:-train}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
RUN_RECOMPUTE="${RUN_RECOMPUTE:-0}"

recompute_spins_if_missing() {
  if [[ -f "$SPIN_A1" && -f "$SPIN_B1" && -f "$SPIN_A2" && -f "$SPIN_B2" ]]; then
    echo "✅ Recompute outputs already exist. Skipping recompute."
    return 0
  fi

  echo "⚠️ Missing recompute outputs. Recomputing spins now..."
  mkdir -p "$RECOMPUTE_BASE/on_Label0" "$RECOMPUTE_BASE/on_Label1"
  cd "$PROJECT_ROOT"

  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
    NUM_DEVICES=${#DEVICES[@]}
  else
    NUM_DEVICES=0
  fi

  if [[ "$NUM_DEVICES" -ge 4 ]]; then
    echo "▶️ Parallel recompute using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    CUDA_VISIBLE_DEVICES="${DEVICES[0]}" python scripts/recompute_spins_from_checkpoints.py       --project-root "$PROJECT_ROOT"       --spin-pkl "$SPIN_SRC_A"       --checkpoint-root "$CHECKPOINT_ROOT_A"       --dataset "$DATA_LABEL0"       --output "$RECOMPUTE_BASE/on_Label0/spin_A.pkl"       --tag A       --measure-data "$MEASURE_DATA"       --sample-size "$SAMPLE_SIZE"       > "$RECOMPUTE_BASE/log_A_L0.txt" 2>&1 &
    PID1=$!

    CUDA_VISIBLE_DEVICES="${DEVICES[1]}" python scripts/recompute_spins_from_checkpoints.py       --project-root "$PROJECT_ROOT"       --spin-pkl "$SPIN_SRC_B"       --checkpoint-root "$CHECKPOINT_ROOT_B"       --dataset "$DATA_LABEL0"       --output "$RECOMPUTE_BASE/on_Label0/spin_B.pkl"       --tag B       --measure-data "$MEASURE_DATA"       --sample-size "$SAMPLE_SIZE"       > "$RECOMPUTE_BASE/log_B_L0.txt" 2>&1 &
    PID2=$!

    CUDA_VISIBLE_DEVICES="${DEVICES[2]}" python scripts/recompute_spins_from_checkpoints.py       --project-root "$PROJECT_ROOT"       --spin-pkl "$SPIN_SRC_A"       --checkpoint-root "$CHECKPOINT_ROOT_A"       --dataset "$DATA_LABEL1"       --output "$RECOMPUTE_BASE/on_Label1/spin_A.pkl"       --tag A       --measure-data "$MEASURE_DATA"       --sample-size "$SAMPLE_SIZE"       > "$RECOMPUTE_BASE/log_A_L1.txt" 2>&1 &
    PID3=$!

    CUDA_VISIBLE_DEVICES="${DEVICES[3]}" python scripts/recompute_spins_from_checkpoints.py       --project-root "$PROJECT_ROOT"       --spin-pkl "$SPIN_SRC_B"       --checkpoint-root "$CHECKPOINT_ROOT_B"       --dataset "$DATA_LABEL1"       --output "$RECOMPUTE_BASE/on_Label1/spin_B.pkl"       --tag B       --measure-data "$MEASURE_DATA"       --sample-size "$SAMPLE_SIZE"       > "$RECOMPUTE_BASE/log_B_L1.txt" 2>&1 &
    PID4=$!

    wait $PID1; RC1=$?
    wait $PID2; RC2=$?
    wait $PID3; RC3=$?
    wait $PID4; RC4=$?
    if [ $((RC1+RC2+RC3+RC4)) -ne 0 ]; then
      echo "❌ Recompute failed. RCs: $RC1, $RC2, $RC3, $RC4"
      exit 1
    fi
  else
    echo "▶️ Sequential recompute (no CUDA_VISIBLE_DEVICES with >=4 GPUs set)"
    python scripts/recompute_spins_from_checkpoints.py       --project-root "$PROJECT_ROOT"       --spin-pkl "$SPIN_SRC_A"       --checkpoint-root "$CHECKPOINT_ROOT_A"       --dataset "$DATA_LABEL0"       --output "$RECOMPUTE_BASE/on_Label0/spin_A.pkl"       --tag A       --measure-data "$MEASURE_DATA"       --sample-size "$SAMPLE_SIZE"
    python scripts/recompute_spins_from_checkpoints.py       --project-root "$PROJECT_ROOT"       --spin-pkl "$SPIN_SRC_B"       --checkpoint-root "$CHECKPOINT_ROOT_B"       --dataset "$DATA_LABEL0"       --output "$RECOMPUTE_BASE/on_Label0/spin_B.pkl"       --tag B       --measure-data "$MEASURE_DATA"       --sample-size "$SAMPLE_SIZE"
    python scripts/recompute_spins_from_checkpoints.py       --project-root "$PROJECT_ROOT"       --spin-pkl "$SPIN_SRC_A"       --checkpoint-root "$CHECKPOINT_ROOT_A"       --dataset "$DATA_LABEL1"       --output "$RECOMPUTE_BASE/on_Label1/spin_A.pkl"       --tag A       --measure-data "$MEASURE_DATA"       --sample-size "$SAMPLE_SIZE"
    python scripts/recompute_spins_from_checkpoints.py       --project-root "$PROJECT_ROOT"       --spin-pkl "$SPIN_SRC_B"       --checkpoint-root "$CHECKPOINT_ROOT_B"       --dataset "$DATA_LABEL1"       --output "$RECOMPUTE_BASE/on_Label1/spin_B.pkl"       --tag B       --measure-data "$MEASURE_DATA"       --sample-size "$SAMPLE_SIZE"
  fi
}

if [[ "$RUN_RECOMPUTE" == "1" ]]; then
  recompute_spins_if_missing
else
  echo "ℹ️ Skipping recompute; using existing spin files."
fi

cd "$SCRATCH_JOB_DIR"

# Assuming standard epochs for this run
EPOCHS="${EPOCHS:-1 492 1000}" 

DEFAULT_OUTPUT="$REPO_ROOT/gmlp_output/thesis_analysis/overlap_layer_profile_L0_vs_L1.png"
OUTPUT="${OUTPUT:-$DEFAULT_OUTPUT}"
if [[ "$OUTPUT" != /* ]]; then
  OUTPUT="$REPO_ROOT/$OUTPUT"
fi
mkdir -p "$(dirname "$OUTPUT")"

TITLE="${TITLE:-Overlap: Diff Model on Label 0 vs Label 1}"
LABEL1="${LABEL1:-Label_0}"
LABEL2="${LABEL2:-Label_1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(python "$REPO_ROOT/analysis_scripts/overlap_metrics_layer_profile.py"
    --spin_file_a1 "$SPIN_A1"
    --spin_file_b1 "$SPIN_B1"
    --spin_file_a2 "$SPIN_A2"
    --spin_file_b2 "$SPIN_B2"
    --label1 "$LABEL1"
    --label2 "$LABEL2"
    --epochs $EPOCHS
    --output "$OUTPUT"
    --title "$TITLE"
)

if [ -n "$EXTRA_ARGS" ]; then
  CMD+=($EXTRA_ARGS)
fi

echo "🧮 Command: ${CMD[*]}"
"${CMD[@]}"

status=$?
echo ""
echo "Job finished with status $status at $(date)"
exit $status
