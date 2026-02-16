#!/bin/bash
#------- qsub option -----------
#PBS -q SQUID-H
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=12:00:00
#PBS -l gpunum_job=8
#PBS -l cpunum_job=8
#PBS --enable-cloud-bursting=yes
#PBS -U cloud_wait_limit=01:00:00
#PBS -o ~/qinv_autocorr_log.out
#PBS -e ~/qinv_autocorr_log.err
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

echo "🚀 Starting q_inv autocorr job (all pairs)"
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
SCRATCH_JOB_DIR="${SCRATCH_BASE}/qinv_autocorr_${JOB_ID}_$$"
mkdir -p "$SCRATCH_JOB_DIR"
cleanup() {
  rm -rf "$SCRATCH_JOB_DIR"
}
trap cleanup EXIT
cd "$SCRATCH_JOB_DIR"
echo "📁 Scratch dir: $(pwd)"

# Required (default provided; override via environment)
SPIN_FILE="${SPIN_FILE:-${MONO_ROOT}/gmlp_output/non_mani/diff_train/spin_A_Global.pkl}"

# Optional overrides
ALL_LAYERS="${ALL_LAYERS:-1}"
LAYER_INDEX="${LAYER_INDEX:-0}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/thesis/gmlp_mlm_qinv_out/autocorr}"
TITLE="${TITLE:-q_inv autocorr (t1 vs t2)}"
LABEL="${LABEL:-}"
PROFILE_EPOCHS="${PROFILE_EPOCHS:-}"
BLOCK_SIZE="${BLOCK_SIZE:-2000}"
DEVICE="${DEVICE:-auto}"
N_JOBS="${N_JOBS:-4}"

if [ -n "$ALL_LAYERS" ] && [ "$ALL_LAYERS" != "0" ]; then
  LAYER_ARGS=(--all-layers)
else
  LAYER_ARGS=(--layer-index "$LAYER_INDEX")
fi

CMD=(python "${REPO_ROOT}/analysis_scripts/mlm_qinv_autocorr.py" \
  --spin-file "${SPIN_FILE}" \
  ${LABEL:+--label "$LABEL"} \
  ${PROFILE_EPOCHS:+--profile-epochs "$PROFILE_EPOCHS"} \
  ${BLOCK_SIZE:+--block-size "$BLOCK_SIZE"} \
  ${DEVICE:+--device "$DEVICE"} \
  --n-jobs "${N_JOBS}" \
  "${LAYER_ARGS[@]}" \
  --title "${TITLE}" \
  --output-dir "${OUTDIR}")

echo "🧮 Command: ${CMD[*]}"
"${CMD[@]}"

exit_code=$?
echo "🏁 Exit code: $exit_code"
echo "Done at: $(date)"
exit $exit_code
