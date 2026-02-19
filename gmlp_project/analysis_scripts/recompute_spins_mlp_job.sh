#!/bin/bash
#------- qsub option -----------
#PBS -q DBG
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=00:10:00
#PBS -l cpunum_job=36
#PBS -l gpunum_job=1
#PBS -o ~/recompute_spins_mlp_log.out
#PBS -e ~/recompute_spins_mlp_log.err
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
set -o pipefail

#------- Program execution -----------

echo "🚀 Starting MLP recompute_spins_from_checkpoints job"
echo "======================================"
echo "Job ID: $PBS_JOBID"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo ""

# === モジュール環境のセットアップ ===
echo "📦 Loading Python & GPU modules..."
module purge
module load BaseGPU/2025
module load BasePy/2025
module load python3/3.11
module load cudnn

# === 仮想環境をアクティベート ===
source ${MONO_ROOT}/torch-env/bin/activate

echo "🔍 Python version:"
which python
python --version

# === CUDA 環境確認 ===
echo "🎯 CUDA info:"
nvcc --version || echo "nvcc not found"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

REPO_ROOT="${MONO_ROOT}"
SCRATCH_BASE="/sqfs/ssd/cm9029/${USER_ID}"
SCRATCH_JOB_DIR="${SCRATCH_BASE}/recompute_spins_mlp_${PBS_JOBID:-manual_$$}"
mkdir -p "$SCRATCH_JOB_DIR"
cleanup_scratch() {
  rm -rf "$SCRATCH_JOB_DIR"
}
trap cleanup_scratch EXIT
cd "$SCRATCH_JOB_DIR"
echo "📁 Scratch dir: $(pwd)"

# ------------------------------
# 🛠 設定（必要に応じて書き換えてください）
# ------------------------------
# Example: RUN_ID="run_mlp_20251112-..."
RUN_ID="YOUR_MLP_RUN_ID_HERE"
RUN_ROOT="${MONO_ROOT}/mlp_output/${RUN_ID}"
CHECKPOINT_ROOT_BASE="${MONO_ROOT}/mlp_output/checkpoints/${RUN_ID}"
TAGS=(A B)

DATASET="${MONO_ROOT}/gmlp_project/data/mnist_by_label/mnist_label1.npz"
PROJECT_ROOT="${MONO_ROOT}/gmlp_project"
OUTPUT_ROOT="${MONO_ROOT}/mlp_output/recomputed_spins"
# 任意の出力ディレクトリ名（未指定なら RUN_ID を使用）
OUTPUT_DIR_NAME="mlp_recomputed_${RUN_ID}"

# スペース区切りで列挙（空文字で全エポック対象）
EPOCHS=""

BATCH_SIZE=256
SAMPLE_SIZE=1000
SAMPLE_SEED=2025

# ------------------------------
# 🔎 入力チェック
# ------------------------------
if [[ ! -f "$DATASET" ]]; then
  echo "❌ dataset npz not found: $DATASET" >&2
  # exit 11
  echo "Warning: Continuing blindly, please ensure dataset path is correct before production use."
fi
# (Keeping project root check liberal as it might vary)

if [[ ! -d "$RUN_ROOT" ]]; then
  echo "❌ run root not found (Please set RUN_ID correctly): $RUN_ROOT" >&2
  # exit 13
fi

mkdir -p "$OUTPUT_ROOT"
RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_DIR_NAME}"
mkdir -p "$RUN_OUTPUT_DIR"

find_spin_pkl() {
  local tag="$1"
  local pattern match
  for pattern in \
    "mlp_spin${tag}_*.pkl" \
    "mlp_same_spin${tag}_*.pkl" \
    "*spin${tag}_*.pkl"
  do
    match=$(find "$RUN_ROOT" -maxdepth 1 -type f -name "$pattern" | sort | head -n 1)
    if [[ -n "$match" ]]; then
      echo "$match"
      return 0
    fi
  done
  return 1
}

build_epoch_args() {
  local -n ref=$1
  ref=()
  if [[ -n "$EPOCHS" ]]; then
    read -r -a _epochs <<< "$EPOCHS"
    ref=(--epochs "${_epochs[@]}")
  fi
}

build_sample_args() {
  local -n ref=$1
  ref=()
  if [[ -n "$SAMPLE_SIZE" ]]; then
    ref+=(--sample-size "$SAMPLE_SIZE")
  fi
  if [[ -n "$SAMPLE_SEED" ]]; then
    ref+=(--sample-seed "$SAMPLE_SEED")
  fi
}

PROJECT_ARGS=()
if [[ -n "$PROJECT_ROOT" ]]; then
  PROJECT_ARGS=(--project-root "$PROJECT_ROOT")
fi

overall_status=0

for TAG in "${TAGS[@]}"; do
  echo ""
  echo "====== Processing TAG=$TAG ======"

  SPIN_PKL=$(find_spin_pkl "$TAG")
  if [[ -z "$SPIN_PKL" ]]; then
    echo "❌ No spin pickle found for tag $TAG under $RUN_ROOT" >&2
    # overall_status=20
    # continue
    echo "Warning: Skipping TAG $TAG because pickle was not found"
    continue
  fi
  
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT_BASE}/${TAG}"
  if [[ ! -d "$CHECKPOINT_ROOT" ]]; then
    echo "❌ checkpoint directory not found for tag $TAG: $CHECKPOINT_ROOT" >&2
    # overall_status=22
    # continue
  fi

  SPIN_BASENAME=$(basename "$SPIN_PKL")
  OUTPUT_PATH="${RUN_OUTPUT_DIR}/${SPIN_BASENAME%.pkl}_recomputed_${TAG}.pkl"
  LOG_PATH="${OUTPUT_PATH%.pkl}.log"

  build_epoch_args EPOCH_ARGS
  build_sample_args SAMPLE_ARGS

  CMD=(python "$REPO_ROOT/analysis_scripts/recompute_spins_from_checkpoints_mlp.py"
    --spin-pkl "$SPIN_PKL"
    --dataset "$DATASET"
    --output "$OUTPUT_PATH"
    --batch-size "$BATCH_SIZE"
    --checkpoint-root "$CHECKPOINT_ROOT"
    --tag "$TAG"
    --log-file "$LOG_PATH"
  )
  CMD+=("${PROJECT_ARGS[@]}")
  CMD+=("${EPOCH_ARGS[@]}")
  CMD+=("${SAMPLE_ARGS[@]}")

  echo "🧮 Executing: ${CMD[*]}"
  echo "🧾 Logs will also be written to: $LOG_PATH"
  "${CMD[@]}"
  exit_code=$?
  echo "TAG=$TAG finished with exit code $exit_code"
  if [[ $exit_code -ne 0 ]]; then
    overall_status=$exit_code
  fi
done

echo ""
echo "🏁 Overall exit code: $overall_status"
echo "Done at: $(date)"
exit $overall_status
