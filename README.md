# hpc_squid_simulation 運用ガイド

このリポジトリは、SQUID 環境で gMLP / MLP 系の学習ジョブと解析を回すための実行コードです。

## 前提
- 環境: SQUID (qsub が使える環境)
- クローン先の例: `/sqfs/work/cm9029/${USER_ID}/hpc_squid_simulation`
- Python: `python3.11` を使用

## ディレクトリ構成
- `gmlp_project/`: 学習・評価コード本体
- `gmlp_project/jobs/`: ジョブスクリプト
- `setup_torch_env.sh`: 仮想環境セットアップ
- `torch-env/`: 仮想環境（セットアップ後に作成）
- `gmlp_output/`: 学習出力（ジョブ実行で作成）

## 初回セットアップ
1. リポジトリへ移動
```bash
cd /sqfs/work/cm9029/${USER_ID}/hpc_squid_simulation
```

2. 仮想環境作成
```bash
bash setup_torch_env.sh
```

3. アクティベート
```bash
source /sqfs2/cmc/1/work/cm9029/${USER_ID}/hpc_squid_simulation/torch-env/bin/activate
```

補足:
- この環境では `/sqfs/work/...` が実体 `/sqfs2/cmc/1/work/...` を指すため、`setup_torch_env.sh` は実体パス側で `torch-env` を作ります。
- `python3.11` が見つからない場合は以下を実行:
```bash
USE_MODULES=1 bash setup_torch_env.sh
```

## データ配置
- 既定入力データ:
  - `gmlp_project/data/mnist.npz`
- 必要ならジョブ投入時に `INPUT` 環境変数で上書き可能

## 最初に動かすジョブ（gMLP diff/same）
`qsub` は次のディレクトリから実行できます:
- `/sqfs/work/cm9029/${USER_ID}/hpc_squid_simulation/gmlp_project/jobs/gmlp`

実行例:
```bash
cd /sqfs/work/cm9029/${USER_ID}/hpc_squid_simulation/gmlp_project/jobs/gmlp
qsub run_gmlp_diff_model.sh
qsub run_gmlp_same_model.sh
```

## ログと出力先
### gMLP diff/same ジョブ
- qsub ログ:
  - `gmlp_project/jobs/gmlp_logs/gmlp_diff_model.out`
  - `gmlp_project/jobs/gmlp_logs/gmlp_diff_model.err`
  - `gmlp_project/jobs/gmlp_logs/gmlp_same_model.out`
  - `gmlp_project/jobs/gmlp_logs/gmlp_same_model.err`

- 学習出力:
  - diff 既定: `gmlp_output/`
  - same 既定: `gmlp_output/same_model/`

### 重要
- ジョブは scratch (`/sqfs/ssd/...`) を使って計算し、終了時に成果物を最終出力先へ同期して scratch を削除します。

## ジョブスクリプトの共通仕様
`gmlp_project/jobs` 配下の PBS ジョブは以下を共通化しています。
- リポジトリルート自動探索 (`REPO_ROOT` / `PBS_O_WORKDIR` / `SCRIPT_DIR`)
- `PYTHONPATH` に `gmlp_project/src` を追加

そのため、`gmlp_project/jobs` 以下であればどの場所から `qsub` しても動作するようにしてあります。

## よくあるエラー
### `ERROR: Python 3.8+ is required ... Found 3.6.8`
- `setup_torch_env.sh` 実行時の Python が古い
- 対処:
```bash
PYTHON_BIN=python3.11 bash setup_torch_env.sh
```

### `ModuleNotFoundError: No module named 'gmlp_project'`
- ジョブの `PYTHONPATH` が通っていないか、古いスクリプトを使っている
- まず最新版の `gmlp_project/jobs/*.sh` を利用する

### `torch-env/bin/activate` が見つからない
- `setup_torch_env.sh` 未実行、または別パスで作られている
- 実体パス側 (`/sqfs2/cmc/1/work/...`) に `torch-env` があるか確認
