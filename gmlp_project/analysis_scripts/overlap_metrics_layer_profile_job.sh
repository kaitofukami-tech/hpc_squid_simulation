#!/bin/bash
#------- qsub option -----------
#PBS -q DBG
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=00:10:00
#PBS -l cpunum_job=16
#PBS -o ~/overlap_layer_profile.out
#PBS -e ~/overlap_layer_profile.err
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

echo "🚀 Starting overlap layer-profile job"
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
SCRATCH_JOB_DIR="${SCRATCH_BASE}/overlap_layer_${PBS_JOBID:-manual_$$}"
mkdir -p "$SCRATCH_JOB_DIR"
cleanup() {
  rm -rf "$SCRATCH_JOB_DIR"
}
trap cleanup EXIT
cd "$SCRATCH_JOB_DIR"
echo "📁 Scratch dir: $(pwd)"

SPIN_A1="${SPIN_A1:-${MONO_ROOT}/gmlp_output_diff_denoise_model/run_20251112-093642_p4/gmlp_spinA_D256_F1536_L10_M1000_seedA123.pkl}"
SPIN_B1="${SPIN_B1:-${MONO_ROOT}/gmlp_output_diff_denoise_model/run_20251112-093642_p4/gmlp_spinB_D256_F1536_L10_M1000_seedB456.pkl}"
SPIN_A2="${SPIN_A2:-${MONO_ROOT}/gmlp_output_same_denoise_model/run_20251112-113648_p4/gmlp_same_spinA_D256_F1536_L10_M1000_seed123_trA2025.pkl}"
SPIN_B2="${SPIN_B2:-${MONO_ROOT}/gmlp_output_same_denoise_model/run_20251112-113648_p4/gmlp_same_spinB_D256_F1536_L10_M1000_seed123_trB4242.pkl}"
EPOCHS="${EPOCHS:-1 484 1000}"
DEFAULT_OUTPUT="$REPO_ROOT/gmlp_mlm_qinv_out/dualsentenceAB-trainAB/overlap_layer_profile.png"
OUTPUT="${OUTPUT:-$DEFAULT_OUTPUT}"
if [[ "$OUTPUT" != /* ]]; then
  OUTPUT="$REPO_ROOT/$OUTPUT"
fi
mkdir -p "$(dirname "$OUTPUT")"
TITLE="${TITLE:-Overlap metrics vs layer}"
LABEL1="${LABEL1:-mnist_label0}"
LABEL2="${LABEL2:-mnist_label1}"
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
