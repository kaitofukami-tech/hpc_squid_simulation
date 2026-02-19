#!/bin/bash
#------- qsub option -----------
#PBS -q SQUID-H
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=17:10:00
#PBS -l gpunum_job=2
#PBS -o ../../gmlp_logs/gmlp_diff_model.out
#PBS -e ../../gmlp_logs/gmlp_diff_model.err
#PBS -r n

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONO_ROOT=""

find_repo_root() {
  local start_dir="$1"
  local dir="$start_dir"
  while [ "$dir" != "/" ]; do
    if [ -d "$dir/gmlp_project" ]; then
      echo "$dir"
      return 0
    fi
    if [ -d "$dir/scripts" ] && [ -d "$dir/data" ]; then
      echo "$(cd "$dir/.." && pwd)"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}

if [ -n "${REPO_ROOT:-}" ]; then
  if found="$(find_repo_root "$REPO_ROOT")"; then
    MONO_ROOT="$found"
  fi
fi

if [ -z "$MONO_ROOT" ]; then
  START_DIR="${PBS_O_WORKDIR:-$SCRIPT_DIR}"
  if found="$(find_repo_root "$START_DIR")"; then
    MONO_ROOT="$found"
  fi
fi

if [ -z "$MONO_ROOT" ]; then
  if found="$(find_repo_root "$SCRIPT_DIR")"; then
    MONO_ROOT="$found"
  fi
fi

if [ -z "$MONO_ROOT" ]; then
  echo "Repo root not found from REPO_ROOT/PBS_O_WORKDIR/SCRIPT_DIR"
  exit 2
fi

REPO_ROOT="$MONO_ROOT/gmlp_project"
export PYTHONPATH="${MONO_ROOT}/gmlp_project/src:${MONO_ROOT}:${PYTHONPATH:-}"

echo "Starting gMLP diff-model job"
echo "Job ID: ${PBS_JOBID:-manual}"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "MONO_ROOT: $MONO_ROOT"

module purge
module load BaseGPU/2025
module load BasePy/2025
module load python3/3.11
if module avail cudnncd &>/dev/null; then
  module load cudnncd
fi

if [ -f "${MONO_ROOT}/torch-env/bin/activate" ]; then
  source "${MONO_ROOT}/torch-env/bin/activate"
fi

EPOCHS="${EPOCHS:-1000}"
PATCH="${PATCH:-4}"
DMODEL="${DMODEL:-256}"
DFFN="${DFFN:-1536}"
BATCH="${BATCH:-256}"
LR="${LR:-1e-3}"
INPUT="${INPUT:-${MONO_ROOT}/gmlp_project/data/mnist.npz}"
OUTDIR="${OUTDIR:-${MONO_ROOT}/gmlp_output}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

SCRATCH_BASE="/sqfs/ssd/cm9029/${USER_ID:-$USER}"
JOB_LABEL="gmlp_diff_${PBS_JOBID:-manual_$$}"
SCRATCH_JOB_DIR="${SCRATCH_BASE}/${JOB_LABEL}"
SCRATCH_INPUT="${SCRATCH_JOB_DIR}/input"
SCRATCH_OUTPUT="${SCRATCH_JOB_DIR}/output"
FINAL_OUTDIR="$OUTDIR"
mkdir -p "$SCRATCH_JOB_DIR" "$SCRATCH_INPUT" "$SCRATCH_OUTPUT" "$FINAL_OUTDIR"
cd "$SCRATCH_JOB_DIR"

cleanup() {
  if [ "${STAGING_OK:-1}" -eq 1 ]; then
    rsync -a "${SCRATCH_OUTPUT}/" "${FINAL_OUTDIR}/" 2>/dev/null || true
  fi
  rm -rf "$SCRATCH_JOB_DIR"
}
trap cleanup EXIT

LOCAL_INPUT="${SCRATCH_INPUT}/$(basename "$INPUT")"
LOCAL_OUT="$SCRATCH_OUTPUT"
STAGING_OK=1
if ! rsync -a "$INPUT" "$SCRATCH_INPUT/"; then
  STAGING_OK=0
  LOCAL_INPUT="$INPUT"
  LOCAL_OUT="$FINAL_OUTDIR"
fi

IFS=',' read -ra COMMAS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS=${#COMMAS[@]}
HALF=$((NUM_GPUS / 2))
if [ "$HALF" -lt 1 ]; then
  HALF=1
fi
GPUS_TRAIN=("${COMMAS[@]:0:$HALF}")
GPUS_VAL=("${COMMAS[@]:$HALF}")
STR_GPUS_TRAIN=$(IFS=,; echo "${GPUS_TRAIN[*]}")
STR_GPUS_VAL=$(IFS=,; echo "${GPUS_VAL[*]}")

CUDA_VISIBLE_DEVICES="$STR_GPUS_TRAIN" python "$REPO_ROOT/scripts/gmlp_diff_model.py" \
  --epochs "$EPOCHS" \
  --patch "$PATCH" \
  --d_model "$DMODEL" \
  --d_ffn "$DFFN" \
  --batch_size "$BATCH" \
  --lr "$LR" \
  --input "$LOCAL_INPUT" \
  --output_dir "$LOCAL_OUT" \
  --measure_data train \
  ${EXTRA_ARGS} > "${SCRATCH_JOB_DIR}/train.log" 2>&1 &
PID_TRAIN=$!

CUDA_VISIBLE_DEVICES="$STR_GPUS_VAL" python "$REPO_ROOT/scripts/gmlp_diff_model.py" \
  --epochs "$EPOCHS" \
  --patch "$PATCH" \
  --d_model "$DMODEL" \
  --d_ffn "$DFFN" \
  --batch_size "$BATCH" \
  --lr "$LR" \
  --input "$LOCAL_INPUT" \
  --output_dir "$LOCAL_OUT" \
  --measure_data val \
  ${EXTRA_ARGS} > "${SCRATCH_JOB_DIR}/val.log" 2>&1 &
PID_VAL=$!

wait "$PID_TRAIN"
RC_TRAIN=$?
wait "$PID_VAL"
RC_VAL=$?

tail -n 20 "${SCRATCH_JOB_DIR}/train.log" || true
tail -n 20 "${SCRATCH_JOB_DIR}/val.log" || true

if [ "$RC_TRAIN" -ne 0 ] || [ "$RC_VAL" -ne 0 ]; then
  echo "gMLP diff-model job failed: train=$RC_TRAIN val=$RC_VAL"
  exit 1
fi

echo "gMLP diff-model job completed"
