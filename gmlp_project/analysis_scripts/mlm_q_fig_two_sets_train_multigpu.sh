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
#PBS -l elapstim_req=07:00:00
#PBS -l cpunum_job=8
#PBS -l gpunum_job=8
#PBS --enable-cloud-bursting=yes
#PBS -U cloud_wait_limit=01:00:00
#PBS -o ~/q2_dual_analysis_log_train_multigpu.out
#PBS -e ~/q2_dual_analysis_log_train_multigpu.err
#PBS -r n

set -euo pipefail

echo "🚀 Starting dual q_inv figure job (multi-GPU by layer)"
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

# Layer list and GPU list (override via env)
LAYER_LIST="${LAYER_LIST:-0,1,2,3,4,5,6,7,8,9}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
BLOCK_SIZE="${BLOCK_SIZE:-2048}"
DEVICE="${DEVICE:-cuda}"

OUTDIR="${OUTDIR:-${REPO_ROOT}/thesis/gmlp_mlm_qinv_out/non_mani/train}"
TITLE="${TITLE:-q_inv comparison (gMLP MLM train)}"

IFS=',' read -r -a LAYERS <<< "$LAYER_LIST"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

gpu_count=${#GPUS[@]}
if [ "$gpu_count" -eq 0 ]; then
  echo "No GPUs in GPU_LIST."
  exit 1
fi

echo "Layers: ${LAYERS[*]}"
echo "GPUs:   ${GPUS[*]}"
echo "BLOCK_SIZE: $BLOCK_SIZE"
echo "DEVICE: $DEVICE"

run_layer() {
  local layer="$1"
  local gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" \
  ALL_LAYERS=0 \
  LAYER_INDEX="$layer" \
  DEVICE="$DEVICE" \
  BLOCK_SIZE="$BLOCK_SIZE" \
  OUTDIR="$OUTDIR" \
  TITLE="$TITLE (L=$layer)" \
  "${REPO_ROOT}/analysis_scripts/mlm_q_fig_two_sets_train.sh"
}

active=0
for i in "${!LAYERS[@]}"; do
  layer="${LAYERS[$i]}"
  gpu="${GPUS[$((i % gpu_count))]}"
  echo "▶ Launch L=$layer on GPU $gpu"
  run_layer "$layer" "$gpu" &
  active=$((active+1))
  if [ "$active" -ge "$gpu_count" ]; then
    wait -n
    active=$((active-1))
  fi
done

wait
echo "Done at: $(date)"
