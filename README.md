# /sqfs/work/cm9029/${USER_ID}（日本語ガイド）

このリポジトリには gMLP/MLP の実験コード、解析スクリプト、関連資料が入っています。

## 1. 全体構成（トップレベル）
- `gmlp_project/` 実験の本体（学習・評価・ジョブスクリプトなど）
- `gmlp_project/analysis_scripts/` 解析・可視化スクリプト
- `docs/` ドキュメント・資料
- `batchnorm/`, `thesis/` 研究用の補助資料
- `archive/` 過去の出力や不要になったファイルの退避場所

※ 以前の output 系ディレクトリはすべて `archive/` に移動済みです。

## 2. 環境構築（最低限）

### (A) 既存の仮想環境を使う場合
この環境では `torch-env/` に既存の仮想環境があります。

```bash
source /sqfs/work/cm9029/${USER_ID}/torch-env/bin/activate
```

### (B) gmlp_project を編集可能インストールする
`gmlp_project/` では `pip install -e . --user` を使う運用になっています。

```bash
cd /sqfs/work/cm9029/${USER_ID}/gmlp_project
python3 -m pip install -e . --user
```

これで `gmlp_project.*` を import できます。


## 2.1 ユーザーIDの環境変数

この環境では `USER_ID` を使ってパスを組み立てます。
以下のように `~/.bashrc` に設定します。

```bash
export USER_ID=u6c398
```

`~/.bashrc` の場所は次の通りです：
`/sqfs/home/u6c398/.bashrc`

設定後は次を実行すると反映されます。

```bash
source ~/.bashrc
```


### (C) 自動セットアップスクリプト
以下のスクリプトで `torch-env/` を作成できます。

```bash
cd /sqfs/work/cm9029/${USER_ID}
./setup_torch_env.sh
```

GPU版を入れたい場合は例のように指定します。

```bash
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121" ./setup_torch_env.sh
```

## 3. 実行例

### 解析スクリプトを直接実行
```bash
python /sqfs/work/cm9029/${USER_ID}/gmlp_project/analysis_scripts/mlp_qinv_calc_fig_two_sets.py --help
```

### ジョブを投げる（qsub）
```bash
qsub /sqfs/work/cm9029/${USER_ID}/gmlp_project/jobs/<job_file>.sh
```

## 4. データ
- データは `gmlp_project/data/` に置きます。
- 大容量データは Git には入れません。

## 5. 出力
- 学習結果や図などの出力は `archive/` に退避しています。
- 新しい出力先を作り直したい場合は相談してください。

