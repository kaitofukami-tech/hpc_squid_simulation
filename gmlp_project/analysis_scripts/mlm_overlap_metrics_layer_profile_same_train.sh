#!/bin/bash
#------- qsub option -----------
#PBS -q SQUID-H
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=04:00:00
#PBS -l cpunum_job=16
#PBS --enable-cloud-bursting=yes   
#PBS -U cloud_wait_limit=01:00:00 
#PBS -o ~/mlm_overlap_layer_profile_same_train.out
#PBS -e ~/mlm_overlap_layer_profile_same_train.err
#PBS -r n

set -euo pipefail

echo "🚀 Starting overlap layer-profile job"
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

SPIN_A1="${SPIN_A1:-/sqfs/work/cm9029/${USER_ID}/gmlp_output/mlm_100sent/manifold100/recompute_run_20260122-215517_same_seq128_train_1sent/first_sent_train_spinA.pkl}"
SPIN_B1="${SPIN_B1:-/sqfs/work/cm9029/${USER_ID}/gmlp_output/mlm_100sent/manifold100/recompute_run_20260122-215517_same_seq128_train_1sent/first_sent_train_spinB.pkl}"
SPIN_A2="${SPIN_A2:-/sqfs/work/cm9029/${USER_ID}/gmlp_output/mlm_100sent/manifold100/recompute_run_20260122-215517_same_seq128_train_100sent/100_sent_train_spinA.pkl}"
SPIN_B2="${SPIN_B2:-/sqfs/work/cm9029/${USER_ID}/gmlp_output/mlm_100sent/manifold100/recompute_run_20260122-215517_same_seq128_train_100sent/100_sent_train_spinB.pkl}"
EPOCHS="${EPOCHS:-1 94 484 727 1000}"
DEFAULT_OUTPUT="$REPO_ROOT/batchnorm/gmlp_mlm_qinv_out/manifold100/same_train_overlap_layer_profile1_100.png"
OUTPUT="${OUTPUT:-$DEFAULT_OUTPUT}"
DEFAULT_OUTPUT_CSV="$REPO_ROOT/batchnorm/gmlp_mlm_qinv_out/manifold100/same_train_overlap_layer_profile1_100.csv"
OUTPUT_CSV="${OUTPUT_CSV:-$DEFAULT_OUTPUT_CSV}"
if [[ "$OUTPUT" != /* ]]; then
  OUTPUT="$REPO_ROOT/$OUTPUT"
fi
if [[ "$OUTPUT_CSV" != /* ]]; then
  OUTPUT_CSV="$REPO_ROOT/$OUTPUT_CSV"
fi
mkdir -p "$(dirname "$OUTPUT")"
mkdir -p "$(dirname "$OUTPUT_CSV")"
TITLE="${TITLE:-Overlap metrics vs layer}"
LABEL1="${LABEL1:-SentenceA}"
LABEL2="${LABEL2:-SentenceB}"
Q_BLOCK_SIZE="${Q_BLOCK_SIZE:-128}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [[ "$EXTRA_ARGS" == *"--patch-mean"* ]]; then
  echo "ERROR: --patch-mean is not allowed for this job."
  exit 2
fi

CMD=(python "$REPO_ROOT/analysis_scripts/mlm_overlap_metrics_layer_profile.py"
    --spin_file_a1 "$SPIN_A1"
    --spin_file_b1 "$SPIN_B1"
    --spin_file_a2 "$SPIN_A2"
    --spin_file_b2 "$SPIN_B2"
    --label1 "$LABEL1"
    --label2 "$LABEL2"
    --epochs $EPOCHS
    --flatten
    --q-block-size "$Q_BLOCK_SIZE"
    --output "$OUTPUT"
    --output-csv "$OUTPUT_CSV"
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
