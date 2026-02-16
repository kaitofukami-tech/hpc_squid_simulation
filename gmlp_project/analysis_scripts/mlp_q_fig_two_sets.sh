#!/bin/bash
#------- qsub option -----------
#PBS -q DBG
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=00:10:00
#PBS -l cpunum_job=8
#PBS -l gpunum_job=0
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
echo "🚀 Starting MLP dual q_inv figure job"
echo "Job ID: $PBS_JOBID"
echo "Host: $(hostname)"
echo "Time: $(date)"

module purge
module load BaseGPU/2025
module load BasePy/2025
module load python3/3.11
source ${MONO_ROOT}/torch-env/bin/activate
python3 -m pip install -q joblib


# User-editable params
ALL_LAYERS="${ALL_LAYERS:-1}"       # default: all layers
LAYER_INDEX="${LAYER_INDEX:-0}"     # 0..4 for MLP layers (out_bn excluded)
OUTDIR="${OUTDIR:-${MONO_ROOT}/thesis/mlp_qinv_out/mlp_mlm/test}"
TITLE="${TITLE:-q_inv comparison (MLP MLM val)}"

cd ${MONO_ROOT}

# Build layer args
if [ -n "$ALL_LAYERS" ] && [ "$ALL_LAYERS" != "0" ]; then
  LAYER_ARGS=(--all-layers)
else
  LAYER_ARGS=(--layer-index "$LAYER_INDEX")
fi

python analysis_scripts/mlp_qinv_calc_fig_two_sets.py \
  --spin_file_a1 ${MONO_ROOT}/mlp_output/mlp_mlm/recompute_run_20260126-031304_diff_seq128_val/spin_A_Global.pkl \
  --spin_file_b1 ${MONO_ROOT}/mlp_output/mlp_mlm/recompute_run_20260126-031304_diff_seq128_val/spin_B_Global.pkl\
  ${METRICS_A1:+--metrics_a1 "$METRICS_A1"} \
  ${METRICS_B1:+--metrics_b1 "$METRICS_B1"} \
  --label1       MLP_MLM_DIFF \
  --spin_file_a2  ${MONO_ROOT}/mlp_output/mlp_mlm/recompute_run_20260126-135707_same_seq128_val/spin_A_Global.pkl\
  --spin_file_b2  ${MONO_ROOT}/mlp_output/mlp_mlm/recompute_run_20260126-135707_same_seq128_val/spin_B_Global.pkl\
  ${METRICS_A2:+--metrics_a2 "$METRICS_A2"} \
  ${METRICS_B2:+--metrics_b2 "$METRICS_B2"} \
  --label2       MLP_MLM_SAME \
  "${LAYER_ARGS[@]}" \
  --title        "${TITLE}" \
  --output-dir   "${OUTDIR}"

exit_code=$?
echo "🏁 Exit code: $exit_code"
echo "Done at: $(date)"
exit $exit_code
