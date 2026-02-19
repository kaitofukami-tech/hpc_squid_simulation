#!/bin/bash
#------- qsub option -----------
#PBS -q DBG
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=00:10:00
#PBS -l cpunum_job=8
#PBS -o ~/qinv_layer_profile.out
#PBS -e ~/qinv_layer_profile.err
#PBS -r n
#PBS -V

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONO_ROOT=""

# 1) Honor REPO_ROOT if provided (can point to repo root or gmlp_project).
if [ -n "${REPO_ROOT:-}" ]; then
  if [ -d "$REPO_ROOT/.git" ] || [ -d "$REPO_ROOT/gmlp_project" ]; then
    MONO_ROOT="$REPO_ROOT"
  elif [ -d "$REPO_ROOT/scripts" ] && [ -d "$REPO_ROOT/data" ]; then
    MONO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
  fi
fi

# 2) Prefer submit directory if available (PBS copies this script into a jobfile dir).
if [ -z "$MONO_ROOT" ]; then
  START_DIR="${PBS_O_WORKDIR:-$SCRIPT_DIR}"
  dir="$START_DIR"
  while [ "$dir" != "/" ]; do
    if [ -d "$dir/.git" ] || [ -d "$dir/gmlp_project" ]; then
      MONO_ROOT="$dir"
      break
    fi
    if [ -d "$dir/scripts" ] && [ -d "$dir/data" ]; then
      MONO_ROOT="$(cd "$dir/.." && pwd)"
      break
    fi
    dir="$(dirname "$dir")"
  done
fi

# 3) Fallback to script dir (jobfile dir); warn if repo not found.
if [ -z "$MONO_ROOT" ]; then
  MONO_ROOT="$SCRIPT_DIR"
fi

if [ ! -d "$MONO_ROOT/gmlp_project" ]; then
  echo "❌ Repo root not found. MONO_ROOT=$MONO_ROOT"
  echo "   Set REPO_ROOT to the repository root (the directory containing gmlp_project)."
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-$MONO_ROOT}"
set -euo pipefail

echo "🚀 Starting q_inv layer-profile job"
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
SCRATCH_JOB_DIR="${SCRATCH_BASE}/qinv_layer_${PBS_JOBID:-manual_$$}"
mkdir -p "$SCRATCH_JOB_DIR"
cleanup() {
  rm -rf "$SCRATCH_JOB_DIR"
}
trap cleanup EXIT
cd "$SCRATCH_JOB_DIR"
echo "📁 Scratch dir: $(pwd)"

SPIN_A="${SPIN_A:-${MONO_ROOT}/gmlp_output_diff_model/run_20251024-042441_p4/gmlp_spinA_D256_F1536_L10_M1000_seedA123.pkl}"
SPIN_B="${SPIN_B:-${MONO_ROOT}/gmlp_output_diff_model/run_20251024-042441_p4/gmlp_spinB_D256_F1536_L10_M1000_seedB456.pkl}"
SPIN_A2="${SPIN_A2:-${MONO_ROOT}/gmlp_output_same_model/run_20251102-083236_p4/gmlp_same_model_spinA_D256_F1536_L10_M1000_seed123_trA2025.pkl}"
SPIN_B2="${SPIN_B2:-${MONO_ROOT}/gmlp_output_same_model/run_20251102-083236_p4/gmlp_same_model_spinB_D256_F1536_L10_M1000_seed123_trB4242.pkl}"
LABEL1="${LABEL1:-diff}"
LABEL2="${LABEL2:-same}"
EPOCHS="${EPOCHS:-1 492 1000}"
OUTPUT="${OUTPUT:-./gmlp_qinv_out/layer_profile/mnist_fulllabel/qinv_layer_profile.png}"
TITLE="${TITLE:-q_inv vs layer index}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ -z "$SPIN_A" ] || [ -z "$SPIN_B" ]; then
  echo "❌ SPIN_A と SPIN_B を設定してください (Replica A/B pickle)。"
  exit 2
fi

CMD=(python "$REPO_ROOT/analysis_scripts/qinv_layer_profile.py"
    --spin_file_a "$SPIN_A"
    --spin_file_b "$SPIN_B"
    --label1 "$LABEL1"
    --output "$OUTPUT"
    --title "$TITLE"
    --epochs $EPOCHS
)
if [ -n "$SPIN_A2" ] && [ -n "$SPIN_B2" ]; then
  CMD+=(--spin_file_a2 "$SPIN_A2" --spin_file_b2 "$SPIN_B2" --label2 "$LABEL2")
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
