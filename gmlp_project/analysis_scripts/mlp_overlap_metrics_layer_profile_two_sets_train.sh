#!/bin/bash
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
#------- qsub option -----------
#PBS -q SQUID-H
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=04:00:00
#PBS -l cpunum_job=16
#PBS --enable-cloud-bursting=yes   #クラウドバースティングすることを許可します。
#PBS -U cloud_wait_limit=01:00:00
#PBS -o ~/mlp_overlap_layer_profile_two_sets_train.out
#PBS -e ~/mlp_overlap_layer_profile_two_sets_train.err
#PBS -r n

set -euo pipefail

echo "🚀 Starting MLP MLM overlap layer-profile job (diff vs same train)"
echo "Job ID : ${PBS_JOBID:-manual}"
echo "Host   : $(hostname)"
echo "Time   : $(date)"
echo ""

module purge
module load BaseGPU/2025
module load BasePy/2025
module load python3/3.11
source ${MONO_ROOT}/torch-env/bin/activate

REPO_ROOT="${MONO_ROOT}"
SCRATCH_BASE="/sqfs/ssd/cm9029/${USER_ID}"
SCRATCH_JOB_DIR="${SCRATCH_BASE}/mlp_overlap_layer_${PBS_JOBID:-manual_$$}"
mkdir -p "$SCRATCH_JOB_DIR"
cleanup() {
  rm -rf "$SCRATCH_JOB_DIR"
}
trap cleanup EXIT
cd "$SCRATCH_JOB_DIR"
echo "📁 Scratch dir: $(pwd)"

# Two-set comparison (diff vs same) using global spin pickles
SPIN_A1="${SPIN_A1:-${MONO_ROOT}/mlp_output/mlp_mlm/recompute_run_20260126-031304_diff_seq128_train/spin_A_Global.pkl}"
SPIN_B1="${SPIN_B1:-${MONO_ROOT}/mlp_output/mlp_mlm/recompute_run_20260126-031304_diff_seq128_train/spin_B_Global.pkl}"
SPIN_A2="${SPIN_A2:-${MONO_ROOT}/mlp_output/mlp_mlm/recompute_run_20260126-135707_same_seq128_train/spin_A_Global.pkl}"
SPIN_B2="${SPIN_B2:-${MONO_ROOT}/mlp_output/mlp_mlm/recompute_run_20260126-135707_same_seq128_train/spin_B_Global.pkl}"
EPOCHS="${EPOCHS:-1 94 484 727 1000}"

DEFAULT_OUTPUT="$REPO_ROOT/thesis/mlp_qinv_out/mlp_mlm/train/overlap_layer_profile_diff_vs_same.png"
OUTPUT="${OUTPUT:-$DEFAULT_OUTPUT}"
DEFAULT_OUTPUT_CSV="$REPO_ROOT/thesis/mlp_qinv_out/mlp_mlm/train/overlap_layer_profile_diff_vs_same.csv"
OUTPUT_CSV="${OUTPUT_CSV:-$DEFAULT_OUTPUT_CSV}"
if [[ "$OUTPUT" != /* ]]; then
  OUTPUT="$REPO_ROOT/$OUTPUT"
fi
if [[ "$OUTPUT_CSV" != /* ]]; then
  OUTPUT_CSV="$REPO_ROOT/$OUTPUT_CSV"
fi
mkdir -p "$(dirname "$OUTPUT")"
mkdir -p "$(dirname "$OUTPUT_CSV")"

TITLE="${TITLE:-Overlap metrics vs layer (MLP MLM train: diff vs same)}"
LABEL1="${LABEL1:-MLP_MLM_DIFF_train}"
LABEL2="${LABEL2:-MLP_MLM_SAME_train}"
Q_BLOCK_SIZE="${Q_BLOCK_SIZE:-128}"
LAYERS="${LAYERS:-}"
CENTER="${CENTER:-0}"
PATCH_MEAN="${PATCH_MEAN:-0}"
if [[ -z "${NUM_WORKERS:-}" ]]; then
  if [[ -n "${PJM_NODE_CORES:-}" ]]; then
    NUM_WORKERS="${PJM_NODE_CORES}"
  elif [[ -n "${PBS_NUM_PPN:-}" ]]; then
    NUM_WORKERS="${PBS_NUM_PPN}"
  elif [[ -n "${PBS_NCPUS:-}" ]]; then
    NUM_WORKERS="${PBS_NCPUS}"
  elif [[ -n "${OMP_NUM_THREADS:-}" ]]; then
    NUM_WORKERS="${OMP_NUM_THREADS}"
  else
    NUM_WORKERS="0"
  fi
else
  NUM_WORKERS="${NUM_WORKERS}"
fi
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [[ "$EXTRA_ARGS" == *"--patch-mean"* ]] && [[ "$PATCH_MEAN" != "1" ]]; then
  echo "ERROR: --patch-mean is not allowed for this job."
  exit 2
fi

CMD=(python "$REPO_ROOT/analysis_scripts/mlp_mlm_overlap_metrics_layer_profile.py"
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

if [[ -n "$LAYERS" ]]; then
  CMD+=(--layers $LAYERS)
fi
if [[ "$CENTER" == "1" ]]; then
  CMD+=(--center)
fi
if [[ "$PATCH_MEAN" == "1" ]]; then
  CMD+=(--patch-mean)
fi
if [[ "$NUM_WORKERS" != "0" ]]; then
  CMD+=(--num-workers "$NUM_WORKERS")
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
