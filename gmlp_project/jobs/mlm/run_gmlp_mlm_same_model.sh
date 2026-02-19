#!/bin/bash
#------- qsub option -----------
#PBS -q SQUID-H
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=48:00:00
#PBS -l gpunum_job=4
#PBS -o ../logs/gmlp_mlm_same_model.out
#PBS -e ../logs/gmlp_mlm_same_model.err
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
echo "🚀 Starting gMLP MLM dual-model job (same init)"
echo "==============================================="
echo "Job ID: ${PBS_JOBID:-manual}"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo ""

# === モジュール環境のセットアップ ===
echo "📦 Loading Python & GPU modules..."
module purge
module load BaseGPU/2025
module load BasePy/2025
module load python3/3.11
if module avail cudnncd &>/dev/null; then
    module load cudnncd
else
    echo "⚠️ cudnncd module not available; continuing without it"
fi

# === 仮想環境をアクティベート ===
source ${MONO_ROOT}/torch-env/bin/activate

which python
python --version

# === CUDA 環境確認 ===
echo "🎯 CUDA info:"
nvcc --version || echo "nvcc not found"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# === プロジェクトディレクトリへ移動 ===
PROJECT_ROOT="${MONO_ROOT}/gmlp_project"
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "❌ Project root ${PROJECT_ROOT} が見つかりません"
    exit 1
fi
cd "$PROJECT_ROOT"
echo "📁 Current directory: $(pwd)"

echo "Running script: scripts/gmlp_mlm_diff_model.py (mode=same)"

# === 実行パラメータ（環境変数で上書き可能） ===
EPOCHS="${EPOCHS:-5}"
BATCH="${BATCH:-64}"
LR="${LR:-1e-3}"
SEQ_LEN="${SEQ_LEN:-128}"
DMODEL="${DMODEL:-256}"
DFFN="${DFFN:-1536}"
NUM_BLOCKS="${NUM_BLOCKS:-10}"
DROPOUT="${DROPOUT:-0.1}"
MLM_PROB="${MLM_PROB:-0.15}"
LOADER_WORKERS="${LOADER_WORKERS:-0}"

DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-2-raw-v1}"
TOKENIZER_NAME="${TOKENIZER_NAME:-bert-base-uncased}"
CACHE_DIR="${CACHE_DIR:-}"

OUTDIR="${OUTDIR:-${MONO_ROOT}/gmlp_mlm_output/same}"

# seeds (same: 初期化をコピー、ミニバッチ/マスク乱数のみ別)
INIT_SEED="${INIT_SEED:-123}"
TRAIN_SEEDA="${TRAIN_SEEDA:-2025}"
TRAIN_SEEDB="${TRAIN_SEEDB:-4242}"
DATA_SEED="${DATA_SEED:-4244}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ -n "$CACHE_DIR" ]; then
    export HF_HOME="$CACHE_DIR"
    export TRANSFORMERS_CACHE="$CACHE_DIR"
    export HF_DATASETS_CACHE="$CACHE_DIR"
fi

mkdir -p "$OUTDIR"

echo "🧪 Params:"
echo "  mode=same epochs=$EPOCHS batch=$BATCH lr=$LR seq_len=$SEQ_LEN d_model=$DMODEL d_ffn=$DFFN blocks=$NUM_BLOCKS dropout=$DROPOUT"
echo "  dataset=${DATASET_NAME}/${DATASET_CONFIG} tokenizer=${TOKENIZER_NAME} mlm_prob=${MLM_PROB}"
echo "  output_dir=$OUTDIR cache=${CACHE_DIR:-<default>}"
echo "  seeds: initSame=$INIT_SEED trainSeedA=$TRAIN_SEEDA trainSeedB=$TRAIN_SEEDB dataSeed=$DATA_SEED"

CMD=(
  python scripts/gmlp_mlm_diff_model.py
    --mode same
    --epochs "$EPOCHS"
    --batch_size "$BATCH"
    --lr "$LR"
    --seq_len "$SEQ_LEN"
    --d_model "$DMODEL"
    --d_ffn "$DFFN"
    --num_blocks "$NUM_BLOCKS"
    --dropout "$DROPOUT"
    --mlm_probability "$MLM_PROB"
    --dataset_name "$DATASET_NAME"
    --dataset_config "$DATASET_CONFIG"
    --tokenizer_name "$TOKENIZER_NAME"
    --output_dir "$OUTDIR"
    --loader_workers "$LOADER_WORKERS"
    --init_seed "$INIT_SEED"
    --train_seedA "$TRAIN_SEEDA"
    --train_seedB "$TRAIN_SEEDB"
    --data_seed "$DATA_SEED"
)

if [ -n "$CACHE_DIR" ]; then
  CMD+=(--cache_dir "$CACHE_DIR")
fi

if [ -n "$EXTRA_ARGS" ]; then
  CMD+=($EXTRA_ARGS)
fi

echo "🔧 Command:"
printf '  %q' "${CMD[@]}"
echo ""

"${CMD[@]}"
exit_code=$?

echo "🏁 Exit code: $exit_code"
echo "Done at: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ gMLP MLM same job completed successfully"
else
    echo "❌ gMLP MLM same job failed"
    tail -20 logs/gmlp_mlm_same_model.err 2>/dev/null || true
fi

exit "$exit_code"
