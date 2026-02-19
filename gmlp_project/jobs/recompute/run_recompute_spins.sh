#!/bin/bash
#------- qsub option -----------
#PBS -q DBG
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=00:10:00
#PBS -l gpunum_job=1
#PBS -o ../logs/recompute_spins.out
#PBS -e ../logs/recompute_spins.err
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
#------- Program execution -----------

echo "🚀 Starting gMLP spin recomputation job"
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
module load cudnncd 2>/dev/null || true

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
SCRATCH_JOB_DIR="${SCRATCH_BASE}/run_recompute_${PBS_JOBID:-manual_$$}"
mkdir -p "$SCRATCH_JOB_DIR"
cleanup() {
    rm -rf "$SCRATCH_JOB_DIR"
}
trap cleanup EXIT
cd "$SCRATCH_JOB_DIR"
echo "📁 Scratch dir: $(pwd)"

# === 各種パラメータ（環境変数で指定） ===
SCRIPT_PATH="${SCRIPT_PATH:-$REPO_ROOT/analysis_scripts/recompute_spins_from_checkpoints.py}"
SPIN_PKL="${SPIN_PKL:?Set SPIN_PKL to the original spin pickle (e.g. gmlp_spinA_*.pkl)}"
DATASET="${DATASET:?Set DATASET to the Fashion-MNIST style .npz used for measurements}"
OUTPUT="${OUTPUT:?Set OUTPUT to the destination pickle path for recomputed spins}"
PROJECT_ROOT="${PROJECT_ROOT:-}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
TAG="${TAG:-}"
EPOCHS="${EPOCHS:-}"
GAMMA="${GAMMA:-}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"
SAMPLE_SEED="${SAMPLE_SEED:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "📜 Script : $SCRIPT_PATH"
echo "🧾 Spin PKL : $SPIN_PKL"
echo "🧾 Dataset  : $DATASET"
echo "📤 Output   : $OUTPUT"
if [ -n "$PROJECT_ROOT" ]; then echo "📁 Project root override: $PROJECT_ROOT"; fi
if [ -n "$CHECKPOINT_ROOT" ]; then echo "📁 Checkpoint root override: $CHECKPOINT_ROOT"; fi
if [ -n "$TAG" ]; then echo "🔖 Tag override: $TAG"; fi
if [ -n "$EPOCHS" ]; then echo "🗓 Epoch subset: $EPOCHS"; fi
if [ -n "$GAMMA" ]; then echo "γ override: $GAMMA"; fi
if [ -n "$SAMPLE_SIZE" ]; then echo "📊 Sample size: $SAMPLE_SIZE (seed=$SAMPLE_SEED)"; fi
if [ -n "$BATCH_SIZE" ]; then echo "📦 Batch size: $BATCH_SIZE"; fi

# === コマンド組み立て ===
CMD=(python "$SCRIPT_PATH"
    --spin-pkl "$SPIN_PKL"
    --dataset "$DATASET"
    --output "$OUTPUT"
)

if [ -n "$PROJECT_ROOT" ]; then
    CMD+=(--project-root "$PROJECT_ROOT")
fi
if [ -n "$CHECKPOINT_ROOT" ]; then
    CMD+=(--checkpoint-root "$CHECKPOINT_ROOT")
fi
if [ -n "$TAG" ]; then
    CMD+=(--tag "$TAG")
fi
if [ -n "$EPOCHS" ]; then
    # allow comma or space separated
    EPOCH_LIST=$(echo "$EPOCHS" | tr ',' ' ')
    CMD+=(--epochs $EPOCH_LIST)
fi
if [ -n "$GAMMA" ]; then
    CMD+=(--gamma "$GAMMA")
fi
if [ -n "$SAMPLE_SIZE" ]; then
    CMD+=(--sample-size "$SAMPLE_SIZE")
fi
if [ -n "$SAMPLE_SEED" ]; then
    CMD+=(--sample-seed "$SAMPLE_SEED")
fi
if [ -n "$BATCH_SIZE" ]; then
    CMD+=(--batch-size "$BATCH_SIZE")
fi
if [ -n "$EXTRA_ARGS" ]; then
    CMD+=($EXTRA_ARGS)
fi

echo ""
echo "🛠 Running command:"
printf ' %q' "${CMD[@]}"
echo ""
echo ""

"${CMD[@]}"
exit_code=$?

echo ""
echo "🏁 Exit code: $exit_code"
echo "Done at: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ Spin recomputation completed successfully"
else
    echo "❌ Spin recomputation failed"
    tail -50 ../logs/recompute_spins.err 2>/dev/null || true
fi

exit $exit_code
