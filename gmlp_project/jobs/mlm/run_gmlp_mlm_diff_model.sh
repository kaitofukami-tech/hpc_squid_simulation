#!/bin/bash
#------- qsub option -----------
#PBS -q SQUID-H
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=48:00:00
#PBS -l gpunum_job=4
#PBS -o ../logs/gmlp_mlm_diff_model.out
#PBS -e ../logs/gmlp_mlm_diff_model.err
#PBS -r n

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

REPO_ROOT="${REPO_ROOT:-$MONO_ROOT}"
export PYTHONPATH="${MONO_ROOT}/gmlp_project/src:${MONO_ROOT}:${PYTHONPATH:-}"
echo "🚀 Starting gMLP MLM dual-model job (diff init)"
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

echo "Running script: scripts/gmlp_mlm_diff_model.py (mode=diff)"

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

OUTDIR="${OUTDIR:-${MONO_ROOT}/gmlp_mlm_output/diff}"

# seeds (diff: 異なる初期化、同一ミニバッチ乱数)
INIT_SEEDA="${INIT_SEEDA:-123}"
INIT_SEEDB="${INIT_SEEDB:-456}"
TRAIN_SEED="${TRAIN_SEED:-2025}"
DATA_SEED="${DATA_SEED:-4244}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ -n "$CACHE_DIR" ]; then
    export HF_HOME="$CACHE_DIR"
    export TRANSFORMERS_CACHE="$CACHE_DIR"
    export HF_DATASETS_CACHE="$CACHE_DIR"
fi

mkdir -p "$OUTDIR"

echo "🧪 Params:"
echo "  mode=diff epochs=$EPOCHS batch=$BATCH lr=$LR seq_len=$SEQ_LEN d_model=$DMODEL d_ffn=$DFFN blocks=$NUM_BLOCKS dropout=$DROPOUT"
echo "  dataset=${DATASET_NAME}/${DATASET_CONFIG} tokenizer=${TOKENIZER_NAME} mlm_prob=${MLM_PROB}"
echo "  output_dir=$OUTDIR cache=${CACHE_DIR:-<default>}"
echo "  seeds: initA=$INIT_SEEDA initB=$INIT_SEEDB trainSeed=$TRAIN_SEED dataSeed=$DATA_SEED"

CMD=(
  python scripts/gmlp_mlm_diff_model.py
    --mode diff
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
    --init_seedA "$INIT_SEEDA"
    --init_seedB "$INIT_SEEDB"
    --train_seed "$TRAIN_SEED"
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
    echo "✅ gMLP MLM diff job completed successfully"
else
    echo "❌ gMLP MLM diff job failed"
    tail -20 logs/gmlp_mlm_diff_model.err 2>/dev/null || true
fi

exit "$exit_code"
