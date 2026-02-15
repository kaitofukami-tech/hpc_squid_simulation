#!/bin/bash
#------- qsub option -----------
#PBS -q SQUID-H
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=05:00:00
#PBS -l cpunum_job=8
#PBS --enable-cloud-bursting=yes   #クラウドバースティングすることを許可します。
#PBS -U cloud_wait_limit=01:00:00   #待ち時間が指定時間を超える場合、バースティング対象ジョブとなり、クラウドで実行されることがあります。待ち時間=4時間の例
#PBS -o ~/q2_dual_analysis_log_train10mani.out
#PBS -e ~/q2_dual_analysis_log_train10mani.err
#PBS -r n

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
source /sqfs/work/cm9029/${USER_ID}/torch-env/bin/activate

REPO_ROOT="/sqfs/work/cm9029/${USER_ID}"
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
OUTDIR="${OUTDIR:-${REPO_ROOT}/thesis/gmlp_mlm_qinv_out/non_mani_random}"
TITLE="${TITLE:-q_inv comparison (gMLP MLM random)}"

# Build layer args
if [ -n "$ALL_LAYERS" ] && [ "$ALL_LAYERS" != "0" ]; then
  LAYER_ARGS=(--all-layers)
else
  LAYER_ARGS=(--layer-index "$LAYER_INDEX")
fi

CMD=(python "${REPO_ROOT}/analysis_scripts/mlm_qinv_calc_fig_two_sets.py" \
  --spin_file_a1 /sqfs/work/cm9029/${USER_ID}/gmlp_output/non_mani/random/diff_random/spin_A_Global.pkl\
  --spin_file_b1 /sqfs/work/cm9029/${USER_ID}/gmlp_output/non_mani/random/diff_random/spin_B_Global.pkl\
  ${METRICS_A1:+--metrics_a1 "$METRICS_A1"} \
  ${METRICS_B1:+--metrics_b1 "$METRICS_B1"} \
  --label1       GMLP_MLM_DIFF_randomdata \
  --spin_file_a2  /sqfs/work/cm9029/${USER_ID}/gmlp_output/non_mani/random/diff_random/spin_A_Global.pkl\
  --spin_file_b2  /sqfs/work/cm9029/${USER_ID}/gmlp_output/non_mani/random/diff_random2/spin_A_Global.pkl\
  ${METRICS_A2:+--metrics_a2 "$METRICS_A2"} \
  ${METRICS_B2:+--metrics_b2 "$METRICS_B2"} \
  --label2       GMLP_MLM_SAME_random_random \
  "${LAYER_ARGS[@]}" \
  --title        "${TITLE}" \
  --output-dir   "${OUTDIR}")

echo "🧮 Command: ${CMD[*]}"
"${CMD[@]}"

exit_code=$?
echo "🏁 Exit code: $exit_code"
echo "Done at: $(date)"
exit $exit_code
