#!/bin/bash
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
#PBS -o ~/q2_dual_analysis_log_mlp_mlm_train_sup_multigpu.out
#PBS -e ~/q2_dual_analysis_log_mlp_mlm_train_sup_multigpu.err
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
export PYTHONPATH="${MONO_ROOT}:${PYTHONPATH:-}"
set -euo pipefail

echo "🚀 Starting MLP MLM dual q_inv figure job (train/val, multi-GPU by layer)"
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

OUTDIR="${OUTDIR:-${REPO_ROOT}/thesis/mlp_qinv_out/mlp_mlm/train-val/sup}"
TITLE="${TITLE:-q_inv comparison (MLP MLM train-val)}"

IFS=',' read -r -a LAYERS <<< "$LAYER_LIST"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

gpu_count=${#GPUS[@]}
if [ "$gpu_count" -eq 0 ]; then
  echo "No GPUs in GPU_LIST."
  exit 1
fi

echo "Layers: ${LAYERS[*]}"
echo "GPUs:   ${GPUS[*]}"

action() {
  local layer="$1"
  local gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" \
  ALL_LAYERS=0 \
  LAYER_INDEX="$layer" \
  OUTDIR="$OUTDIR" \
  TITLE="$TITLE (L=$layer)" \
  "${REPO_ROOT}/analysis_scripts/mlp_q_fig_two_sets_train_sup.sh"
}

active=0
for i in "${!LAYERS[@]}"; do
  layer="${LAYERS[$i]}"
  gpu="${GPUS[$((i % gpu_count))]}"
  echo "▶ Launch L=$layer on GPU $gpu"
  action "$layer" "$gpu" &
  active=$((active+1))
  if [ "$active" -ge "$gpu_count" ]; then
    wait -n
    active=$((active-1))
  fi
done

wait
echo "Done at: $(date)"
