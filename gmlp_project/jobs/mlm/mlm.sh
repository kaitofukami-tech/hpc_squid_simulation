export CACHE_DIR=/sqfs/work/cm9029/${USER_ID}/gmlp_project/data/mlm  # 変更可
export OFFLINE=1
# 必要ならローカルの展開先を指定
# export DATASET_DIR=/path/to/wikitext-2-raw-v1
# export TOKENIZER_PATH=/path/to/bert-base-uncased
qsub gmlp_project/jobs/run_gmlp_mlm_diff_same_parallel.sh
