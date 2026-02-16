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
export CACHE_DIR=${MONO_ROOT}/gmlp_project/data/mlm  # 変更可
export OFFLINE=1
# 必要ならローカルの展開先を指定
# export DATASET_DIR=/path/to/wikitext-2-raw-v1
# export TOKENIZER_PATH=/path/to/bert-base-uncased
qsub gmlp_project/jobs/run_gmlp_mlm_diff_same_parallel.sh
