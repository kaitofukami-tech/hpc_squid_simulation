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
#PBS -q DBG
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=00:10:00
#PBS -l cpunum_job=8
#PBS -l gpunum_job=0
#PBS -o ~/q2_single_mlp_analysis_log.out
#PBS -e ~/q2_single_mlp_analysis_log.err
#PBS -r n

set -euo pipefail

echo "🚀 Starting MLP single q_inv figure job"
JOB_ID="${PBS_JOBID:-manual}"
echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "Time: $(date)"

module purge
module load BaseGPU/2025
module load BasePy/2025
module load python3/3.11
source ${MONO_ROOT}/torch-env/bin/activate

# User-editable params
ALL_LAYERS="${ALL_LAYERS:-1}"       # default: all layers
LAYER_INDEX="${LAYER_INDEX:-0}"     # 0..9 for MLP layers
OUTDIR="${OUTDIR:-./thesis/mlp_qinv_out/random}"
TITLE="${TITLE:-q_inv (MLP random)}"

cd ${MONO_ROOT}

if [ -n "$ALL_LAYERS" ] && [ "$ALL_LAYERS" != "0" ]; then
  LAYER_ARGS=(--all-layers)
else
  LAYER_ARGS=(--layer-index "$LAYER_INDEX")
fi

python analysis_scripts/mlp_qinv_calc_fig_one_set.py \
  --spin_file_a ${MONO_ROOT}/mlp_output/random/mlp_spinA_train_D256_F1024_L10_M1000_seedA123_recomputed_A.pkl \
  --spin_file_b ${MONO_ROOT}/mlp_output/random/mlp_spinB_train_D256_F1024_L10_M1000_seedB456_recomputed_B.pkl \
  ${METRICS_A:+--metrics_a "$METRICS_A"} \
  ${METRICS_B:+--metrics_b "$METRICS_B"} \
  --label        MLP_random \
  "${LAYER_ARGS[@]}" \
  --title        "${TITLE}" \
  --output-dir   "${OUTDIR}"

exit_code=$?
echo "🏁 Exit code: $exit_code"
echo "Done at: $(date)"
exit $exit_code
