#!/bin/bash
#------- qsub option -----------
#PBS -q SQUID
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=00:30:00
#PBS -l cpunum_job=4
#PBS -o ../gmlp_logs/overlap_layer_profile_gmlp.out
#PBS -e ../gmlp_logs/overlap_layer_profile_gmlp.err
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

# Pointing to the outputs of the recompute job
RECOMPUTE_BASE="/sqfs/work/cm9029/${USER_ID}/gmlp_output/recompute/L200_Diff"

SPIN_A1="${SPIN_A1:-$RECOMPUTE_BASE/on_L200/spin_A.pkl}"
SPIN_B1="${SPIN_B1:-$RECOMPUTE_BASE/on_L200/spin_B.pkl}"
SPIN_A2="${SPIN_A2:-$RECOMPUTE_BASE/on_L500/spin_A.pkl}"
SPIN_B2="${SPIN_B2:-$RECOMPUTE_BASE/on_L500/spin_B.pkl}"

# Use all available epochs if possible, or select key ones.
# The user script overlap_metrics_layer_profile.py takes specific epochs.
# Let's pick a few representative ones if we don't automatedly find them.
# Or better, let's parse the pkl to find common epochs? No, shell script.
# Let's assume common ones like 1, 10, 100, 1000 exist.
EPOCHS="${EPOCHS:-1 12 94 492 1000}" 

DEFAULT_OUTPUT="$REPO_ROOT/gmlp_output/denoise_analysis/overlap_layer_profile_L200_vs_L500.png"
OUTPUT="${OUTPUT:-$DEFAULT_OUTPUT}"
if [[ "$OUTPUT" != /* ]]; then
  OUTPUT="$REPO_ROOT/$OUTPUT"
fi
mkdir -p "$(dirname "$OUTPUT")"

TITLE="${TITLE:-Overlap: Diff Model (L200) on Data L200 vs Data L500}"
LABEL1="${LABEL1:-Model_on_L200}"
LABEL2="${LABEL2:-Model_on_L500}"
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
