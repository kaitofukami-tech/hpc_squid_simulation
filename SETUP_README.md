# setup_torch_env.sh 使い方

このスクリプトは `torch-env/` を作成し、必要な Python パッケージを入れるためのものです。

## 1. 実行方法

```bash
cd /sqfs/work/cm9029/${USER_ID}/hpc_squid_simulation
bash setup_torch_env.sh
```

完了後、次で仮想環境を有効化できます。

```bash
source /sqfs/work/cm9029/${USER_ID}/hpc_squid_simulation/torch-env/bin/activate
```

## 2. GPU 版を入れたい場合

CUDA 対応版を入れるには、`TORCH_INDEX_URL` を指定します。

```bash
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121" bash setup_torch_env.sh
```

## 3. CPU 版に固定したい場合

CPU 版を入れたい場合は指定不要（デフォルトが CPU 版）です。

```bash
bash setup_torch_env.sh
```

## 4. パッケージ一覧
- torch / torchvision / torchaudio
- numpy / pandas / matplotlib / joblib
- scipy / scikit-learn / tqdm / pyyaml
- gmlp_project（editable install）

## 5. よくあるトラブル
- `pip` が古い場合は自動で更新します。
- GPU があるのに `torch.cuda.is_available()` が `False` の場合は、
  CUDA対応ホイールが入っているか確認してください。

