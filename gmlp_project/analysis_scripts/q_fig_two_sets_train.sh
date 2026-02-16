#!/bin/bash
#------- qsub option -----------
#PBS -q SQUID-H
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=01:00:00
#PBS --enable-cloud-bursting=yes   #クラウドバースティングすることを許可します。
#PBS -U cloud_wait_limit=01:00:00  
#PBS -l cpunum_job=8
#PBS -o ~/q2_dual_analysis_log.out
#PBS -e ~/q2_dual_analysis_log.err
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

echo "🚀 Starting dual q_inv figure job"
JOB_ID="${PBS_JOBID:-manual}"
echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "Time: $(date)"

module purge
module load BaseGPU/2025
module load BasePy/2025
module load python3/3.11
source ${MONO_ROOT}/torch-env/bin/activate

REPO_ROOT="${MONO_ROOT}"
SCRATCH_BASE="/sqfs/ssd/cm9029/${USER_ID}"
SCRATCH_JOB_DIR="${SCRATCH_BASE}/qfig_two_sets_${JOB_ID}_$$"
mkdir -p "$SCRATCH_JOB_DIR"
cleanup() {
  rm -rf "$SCRATCH_JOB_DIR"
}
trap cleanup EXIT
cd "$SCRATCH_JOB_DIR"
echo "📁 Scratch dir: $(pwd)"

# Optional environment overrides for flexibility
ALL_LAYERS="${ALL_LAYERS:-1}"   # default: all layers
LAYER_INDEX="${LAYER_INDEX:-0}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/qinv/class_diff}"
TITLE="${TITLE:-q_inv comparison (gMLP)}"

# Build layer args
if [ -n "$ALL_LAYERS" ] && [ "$ALL_LAYERS" != "0" ]; then
  LAYER_ARGS=(--all-layers)
else
  LAYER_ARGS=(--layer-index "$LAYER_INDEX")
fi

CMD=(python "${REPO_ROOT}/analysis_scripts/qinv_calc_fig_two_sets.py" \
  --spin_file_a1 ${MONO_ROOT}/archive/outputs_root/outputs/gmlp_output/diff_model/run_gmlp_20260101-020554_p4_train/gmlp_spinA_train_D256_F1536_L10_M1000_seedA123.pkl\
  --spin_file_b1 ${MONO_ROOT}/archive/outputs_root/outputs/gmlp_output/diff_model/run_gmlp_20260101-020554_p4_train/gmlp_spinB_train_D256_F1536_L10_M1000_seedB456.pkl\
  ${METRICS_A1:+--metrics_a1 "$METRICS_A1"} \
  ${METRICS_B1:+--metrics_b1 "$METRICS_B1"} \
  --label1       GMLP_p4_diff \
  --spin_file_a2  ${MONO_ROOT}/archive/outputs_root/outputs/gmlp_output/same_model/run_gmlp_same_20260101-020554_p4_train/gmlp_same_model_spinA_train_D256_F1536_L10_M1000_seed123_trA2025.pkl\
  --spin_file_b2  ${MONO_ROOT}/archive/outputs_root/outputs/gmlp_output/same_model/run_gmlp_same_20260101-020554_p4_train/gmlp_same_model_spinB_train_D256_F1536_L10_M1000_seed123_trB4242.pkl\
  ${METRICS_A2:+--metrics_a2 "$METRICS_A2"} \
  ${METRICS_B2:+--metrics_b2 "$METRICS_B2"} \
  --label2        GMLP_p4_same \
  "${LAYER_ARGS[@]}" \
  --title        "${TITLE}" \
  --output-dir   "${OUTDIR}")

echo "🧮 Command: ${CMD[*]}"
"${CMD[@]}"

exit_code=$?
echo "🏁 Exit code: $exit_code"
echo "Done at: $(date)"
exit $exit_code
