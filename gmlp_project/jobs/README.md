# jobs/ 構成

ジョブスクリプトは種類ごとにサブフォルダに分けています。
パラメータは `configs/` で管理する方針です。

```
jobs/
  gmlp/       # gMLP本体の学習・再現
  mlm/        # MLM関連
  denoise/    # デノイズ課題
  recompute/  # スピン再計算など
  data/       # データ前処理や可視化
  mlp/        # MLP系
  misc/       # テスト・一時
```

## 使い方（例）

```bash
qsub jobs/gmlp/run_gmlp_diff_model.sh
```

