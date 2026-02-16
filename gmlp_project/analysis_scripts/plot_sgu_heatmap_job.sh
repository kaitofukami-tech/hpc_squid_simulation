#!/bin/bash
#------- qsub option -----------
#PBS -q DBG
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=00:10:00
#PBS -l cpunum_job=72
#PBS -l gpunum_job=2
#PBS -o ~/plot_sgu_heatmap.log
#PBS -e ~/plot_sgu_heatmap.err
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

echo "🚀 Starting SGU heatmap job"
echo "Job ID : ${PBS_JOBID:-manual}"
echo "Host   : $(hostname)"
echo "Time   : $(date)"
echo ""

# ------- Module & env setup -------
module purge
module load BasePy/2025
module load python3/3.11

source ${MONO_ROOT}/torch-env/bin/activate

REPO_ROOT="${MONO_ROOT}"
SCRATCH_BASE="/sqfs/ssd/cm9029/${USER_ID}"
SCRATCH_JOB_DIR="${SCRATCH_BASE}/plot_sgu_${PBS_JOBID:-manual_$$}"
mkdir -p "$SCRATCH_JOB_DIR"
cleanup() {
  rm -rf "$SCRATCH_JOB_DIR"
}
trap cleanup EXIT
cd "$SCRATCH_JOB_DIR"
echo "📁 Scratch dir: $(pwd)"

# ------- User settings -------
CHECKPOINTS=(
  "${MONO_ROOT}/gmlp_output_diff_model/checkpoints/run_20251024-042441_p4/A/epoch0001.pt"
  "${MONO_ROOT}/gmlp_output_diff_model/checkpoints/run_20251024-042441_p4/A/epoch0542.pt"
  "${MONO_ROOT}/gmlp_output_diff_model/checkpoints/run_20251024-042441_p4/A/epoch1000.pt"
)

OUTPUT_DIR="${MONO_ROOT}/sgu_heatmaps/run_20251023-075727_p4"
BLOCK_INDICES=(0 1 2 3 4 5 6 7 8 9)
CLAMP_RANGE=("-2.0" "2.0")    # leave empty to use auto min/max, e.g., CLAMP_RANGE=()
DPI=220
C_MAP="coolwarm"
SHOW_BIAS=1                  # set 0 to suppress bias stats
PLOT_KIND="both"             # matrix | tokens | both
TOKEN_GRID=(7 7)           # leave empty to let script infer

# ------- Build command -------
mkdir -p "$OUTPUT_DIR"

CMD=(python "$REPO_ROOT/analysis_scripts/plot_sgu_heatmap.py" --output-dir "$OUTPUT_DIR" --cmap "$C_MAP" --dpi "$DPI")
CMD+=(--checkpoints "${CHECKPOINTS[@]}")

if [[ ${#BLOCK_INDICES[@]} -gt 0 ]]; then
  CMD+=(--block-indices "${BLOCK_INDICES[@]}")
fi

if [[ ${#CLAMP_RANGE[@]} -eq 2 ]]; then
  CMD+=(--clamp "${CLAMP_RANGE[@]}")
fi

if [[ -n "$PLOT_KIND" ]]; then
  CMD+=(--plot-kind "$PLOT_KIND")
fi

if [[ ${#TOKEN_GRID[@]} -eq 2 ]]; then
  CMD+=(--token-grid "${TOKEN_GRID[@]}")
fi

if [[ "$SHOW_BIAS" -eq 1 ]]; then
  CMD+=(--show-bias)
fi

echo "🧮 Command: ${CMD[*]}"
"${CMD[@]}"
status=$?

echo ""
echo "Job finished with status $status at $(date)"
exit $status
