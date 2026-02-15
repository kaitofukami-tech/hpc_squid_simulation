#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gMLP (with SGU) dual-model spin recorder (A & B) + metrics recorder
- A/B: 完全に同一の初期化（A→Bへ state をコピー）、同一学習データ・独立ミニバッチシャッフル
- スピン観測: gMLP 各ブロックの TokenBN (:pre/:post) と out_bn
- 記録エポック: {1,3}+プログレッシブ。loss/acc も同じタイミングで Train/Val を評価して保存
"""

import os, math, time, json, argparse, logging, hashlib, tempfile, shutil, gc, csv
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# ----------------- argparse -----------------
def build_args():
    p = argparse.ArgumentParser("gMLP+SGU dual-model spin recorder (same init, different minibatch order)")
    p.add_argument("--input", type=str, default="/sqfs/home/${USER_ID}/workspace/gmlp_project/data/mnist.npz",
                   help="Fashion-MNIST .npz (x_train,y_train など)")
    p.add_argument("--output_dir", type=str, default="/sqfs/work/cm9029/${USER_ID}/gmlp_output/same_model")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)

    # gMLP dims
    p.add_argument("--patch", type=int, default=4)          # 28x28, patch=4 → S=49
    p.add_argument("--d_model", type=int, default=256)      # D
    p.add_argument("--d_ffn", type=int, default=1536)       # 拡大後（SGU前）サイズ
    p.add_argument("--num_blocks", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.1)

    # 学習/測定
    p.add_argument("--init_seed", type=int, default=123, help="A/B 共通の初期化シード（同一重みで開始）")
    p.add_argument("--train_seedA", type=int, default=2025, help="A の学習シャッフル用シード")
    p.add_argument("--train_seedB", type=int, default=4242, help="B の学習シャッフル用シード")
    p.add_argument("--data_seed", type=int, default=4244, help="計測サンプル抽出用シード")
    p.add_argument("--M", type=int, default=1000, help="スピン計測サンプル数")
    p.add_argument("--record_layers", type=str, default=":post", choices=[":pre",":post"],
                   help="TokenBN や out_bn を :pre/:post でフック")
    p.add_argument("--no-epoch0", action="store_true", default=True, help="epoch0（未学習）での計測を行わない")
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])

    # Progressive recording params（1, 3 固定、その後は間隔を広げる）
    p.add_argument("--record-start", type=int, default=2, help="最初の可変間隔（3の次に足す最初の間隔）")
    p.add_argument("--record-mult", type=float, default=1.5, help="記録間隔の倍率")
    p.add_argument("--record-max", type=int, default=50, help="記録間隔の上限")
    p.add_argument("--always-include-last", action="store_true", default=True, help="最終epochを必ず含める")

    # ★ 追加: 検証分割
    p.add_argument("--val_ratio", type=float, default=0.1, help="検証のホールドアウト比率")
    p.add_argument("--val_seed", type=int, default=None, help="検証分割用シード（未指定なら train_seedA を使用）")
    p.add_argument("--measure_data", type=str, default="train", choices=["train", "val"], help="スピン計測対象 (train/val)")
    return p.parse_args()

# ----------------- logging -----------------
def setup_logger(outdir, level="INFO"):
    os.makedirs(outdir, exist_ok=True)
    logger = logging.getLogger("gMLP_SGU_Dual_SAMEINIT")
    logger.setLevel(getattr(logging, level))
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(outdir, "train.log")); fh.setFormatter(fmt)
    logger.handlers.clear(); logger.addHandler(sh); logger.addHandler(fh)
    return logger

# ----------------- utils -----------------
def dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed_all(seed:int):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_npz_fashion(path, logger):
    with np.load(path) as f:
        keys = list(f.keys()); logger.info(f"NPZ keys: {keys}")
        if "x_train" in f and "y_train" in f:
            x, y = f["x_train"], f["y_train"]
        elif "X_train" in f and "y_train" in f:
            x, y = f["X_train"], f["y_train"]
        elif "train_images" in f and "train_labels" in f:
            x, y = f["train_images"], f["train_labels"]
        else:
            raise KeyError(f"Unsupported keys: {keys}")
    x_img = torch.tensor(x, dtype=torch.float32).view(x.shape[0], 1, 28, 28) / 255.0
    y_t   = torch.tensor(y, dtype=torch.long)
    logger.info(f"Loaded X={tuple(x_img.shape)} y={tuple(y_t.shape)}")
    return x_img, y_t

def make_loader(dataset, bs, seed, logger, *, shuffle=True):
    g = torch.Generator()
    g.manual_seed(seed)
    dl = DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=2, pin_memory=True, generator=g if shuffle else None)
    if shuffle:
        logger.info(f"steps/epoch ≈ {math.ceil(len(dataset)/bs)} (bs={bs}, N={len(dataset)})")
    return dl

def atomic_save_pickle(obj, final_path, logger):
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    tmpdir = os.getenv("SLURM_TMPDIR", "/tmp")
    with tempfile.NamedTemporaryFile(dir=tmpdir, delete=False, suffix=".pkl") as tmp:
        import pickle
        pickle.dump(obj, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.flush(); os.fsync(tmp.fileno())
        tmp_path = tmp.name
    shutil.move(tmp_path, final_path)
    h = hashlib.sha256(open(final_path, "rb").read()).hexdigest()
    with open(final_path + ".sha256","w") as f: f.write(h+"\n")
    sz = os.path.getsize(final_path)/(1024**2)
    logger.info(f"Saved: {final_path} ({sz:.2f} MB) sha256={h[:12]}")

def atomic_save_state_dict(model: nn.Module, final_path: str, logger):
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    tmpdir = os.getenv("SLURM_TMPDIR", "/tmp")
    with tempfile.NamedTemporaryFile(dir=tmpdir, delete=False, suffix=".pt") as tmp:
        torch.save(model.state_dict(), tmp)
        tmp.flush(); os.fsync(tmp.fileno())
        tmp_path = tmp.name
    shutil.move(tmp_path, final_path)
    h = hashlib.sha256(open(final_path, "rb").read()).hexdigest()
    with open(final_path + ".sha256","w") as f: f.write(h+"\n")
    sz = os.path.getsize(final_path)/(1024**2)
    logger.info(f"Saved checkpoint: {final_path} ({sz:.2f} MB) sha256={h[:12]}")

def save_metrics_csv(path, rows, logger):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch","split","loss","acc"])
        writer.writeheader()
        for r in rows: writer.writerow(r)
    logger.info(f"Saved metrics CSV: {path} (rows={len(rows)})")

def model_state_hash(model):
    with torch.no_grad():
        vec = torch.cat([p.detach().float().flatten().cpu() for p in model.parameters()])
    return hashlib.sha256(vec.numpy().tobytes()).hexdigest()

# ----------------- Token-wise BatchNorm -----------------
class TokenBN(nn.Module):
    def __init__(self, d_model, momentum=0.1, eps=1e-5, affine=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model, momentum=momentum, eps=eps, affine=affine)
    def forward(self, x):  # x:(B,S,D)
        B,S,D = x.shape
        y = self.bn(x.reshape(B*S, D))
        return y.reshape(B,S,D)

# ----------------- SGU -----------------
class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        dim_half = d_ffn // 2
        self.norm = nn.LayerNorm(dim_half)
        self.proj = nn.Conv1d(dim_half, dim_half, kernel_size=1, groups=dim_half, bias=True)
        nn.init.constant_(self.proj.bias, 1.0)
    def forward(self, x):  # (B,S,d_ffn)
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = v.transpose(1, 2)       # (B,dim_half,S)
        v = self.proj(v)
        v = v.transpose(1, 2)       # (B,S,dim_half)
        return u*v



# ----------------- SGU (トークンゲート + γブレンド) -----------------
class SpatialGatingUnitGamma(nn.Module):
    """
    分割あり SGU（トークン間相互作用あり）:
      v = LN(v) → （B,dim_half,S）へ転置 → S×S 線形射影（nn.Linear） → （B,S,dim_half）へ戻す
      gate = g
      出力 = gate（gate を u に掛ける）
    """
    def __init__(self, dim_half: int, seq_len: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim_half)
        # トークン次元S上の線形射影（チャネル・バッチに共有）
        self.proj = nn.Linear(seq_len, seq_len, bias=True)
        nn.init.constant_(self.proj.bias, 1.0)  # g≈1 で初期は恒等ゲートに近い

    def forward(self, v):  # v:(B,S,dim_half)
        g = self.norm(v)            # (B,S,dim_half)
        g = g.transpose(1, 2)       # (B,dim_half,S)
        g = self.proj(g)            # (B,dim_half,S) ← S方向の線形射影でトークン混合
        g = g.transpose(1, 2)       # (B,S,dim_half)
        return g

# ----------------- gMLP Block（チャネルプロジェクション + γゲート） -----------------
class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len, dropout=0.1):
        super().__init__()
        assert d_ffn % 2 == 0, "d_ffn must be even to split for gating."
        self.bn = TokenBN(d_model)               # 観測対象
        self.channel_proj = nn.Linear(d_model, d_ffn)
        self.act = nn.GELU()
        dim_half = d_ffn // 2
        self.sgu = SpatialGatingUnitGamma(dim_half, seq_len)
        self.channel_proj_out = nn.Linear(dim_half, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # (B,S,D)
        residual = x
        x = self.bn(x)                     # gmlp_blocks.i.bn :post
        x = self.channel_proj(x)
        x = self.act(x)
        u, v = x.chunk(2, dim=-1)
        gate = self.sgu(v)                 # トークン混合ゲート
        x = u * gate
        x = self.channel_proj_out(x)
        x = self.drop(x)
        return x + residual

# ----------------- gMLP Model -----------------
class gMLP(nn.Module):
    def __init__(self, image_size=28, patch_size=4, d_model=256, d_ffn=256,
                 num_blocks=5, num_classes=10, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0
        if d_ffn % 2 != 0:
            raise ValueError("d_ffn は 2 で割り切れる必要があります。")
        self.seq_len = (image_size // patch_size) ** 2  # 49
        patch_dim = patch_size * patch_size             # 16
        self.patch = patch_size

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.gmlp_blocks = nn.ModuleList([
            gMLPBlock(d_model, d_ffn, self.seq_len, dropout)
            for _ in range(num_blocks)
        ])
        self.out_bn = TokenBN(d_model)      # 追加の観測ポイント
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x_img):  # (B,1,28,28)
        B, C, H, W = x_img.shape
        p = self.patch
        x = x_img.unfold(2, p, p).unfold(3, p, p).contiguous().view(B, -1, p*p)  # (B,S,p^2)
        x = self.patch_embed(x)                                                  # (B,S,D)
        for blk in self.gmlp_blocks:
            x = blk(x)                                                           # (B,S,D)
        x = self.out_bn(x)                                                       # out_bn :post
        x = x.mean(dim=1)                                                        # token 平均
        return self.classifier(x)


# ----------------- 初期化（重み~N(0,1), バイアス=0.1） -----------------
def init_random_normal_and_bias_const(model: nn.Module, *, mean=0.0, std=1.0, bias_const=0.1):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if m.weight is not None:
                nn.init.normal_(m.weight, mean=mean, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, bias_const)

# ----------------- spin recording（pre/post 対応） -----------------
def _parse_layer_flag(name: str) -> Tuple[str, bool]:
    return name.replace(':pre','').replace(':post',''), name.endswith(':pre')

@torch.no_grad()
def measure_spins(model: nn.Module, layer_names: List[str], X_meas: torch.Tensor, batch: int, logger):
    model.eval()
    store: Dict[str, list] = {ln: [] for ln in layer_names}
    name_map = {}
    for ln in layer_names:
        base, is_pre = _parse_layer_flag(ln)
        name_map.setdefault(base, []).append((ln, is_pre))

    handles = []
    for n, m in model.named_modules():
        if n in name_map:
            for out_key, is_pre in name_map[n]:
                if is_pre:
                    def _pre_hook_factory(key):
                        def _pre(_m, inputs):
                            x = inputs[0]
                            store[key].append(x.detach().to("cpu", dtype=torch.float16))
                        return _pre
                    handles.append(m.register_forward_pre_hook(_pre_hook_factory(out_key)))
                else:
                    def _hook_factory(key):
                        def _hook(_m, _in, out):
                            store[key].append(out.detach().to("cpu", dtype=torch.float16))
                        return _hook
                    handles.append(m.register_forward_hook(_hook_factory(out_key)))

    dl = DataLoader(TensorDataset(X_meas), batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    d = dev()
    for (xb,) in dl:
        _ = model(xb.to(d, non_blocking=True))
    for h in handles: h.remove()

    for k, v in store.items():
        store[k] = torch.cat(v, dim=0)   # (M,S,D) あるいは (M,D)
        logger.info(f"[spin] {k}: {tuple(store[k].shape)} dtype={store[k].dtype}")
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return store

# ----------------- Progressive recording scheduler -----------------
def build_epochs_1_3_then_progressive(total_epochs: int,
                                      start: int,
                                      mult: float,
                                      max_interval: int,
                                      include_last: bool) -> List[int]:
    fixed = [e for e in (1, 3) if 1 <= e <= total_epochs]
    epochs = list(sorted(set(fixed)))
    cur = 3 if 3 <= total_epochs else (1 if 1 <= total_epochs else 0)
    interval = max(1, start)
    i = 0
    while True:
        cur = cur + interval
        if cur > total_epochs:
            break
        if cur not in epochs:
            epochs.append(cur)
        interval = min(max_interval, max(1, int(round(interval * mult))))
        i += 1
        if i > 10_000:
            break
    if include_last and total_epochs >= 1 and total_epochs not in epochs:
        epochs.append(total_epochs)
    epochs = sorted(set([e for e in epochs if 1 <= e <= total_epochs]))
    return epochs

# ----------------- train / eval -----------------
def train_one_epoch(model, loader, opt, loss_fn, logger, tag):
    model.train(); d = dev()
    total, n, correct = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(d, non_blocking=True), yb.to(d, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward(); opt.step()
        total += float(loss) * xb.size(0); n += xb.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum())
        del xb, yb, logits, loss, pred
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    acc = correct / max(n,1)
    logger.info(f"[{tag}] train_loss(batch-avg)={total/max(n,1):.4f} acc={acc:.4f}")
    return total / max(n, 1), acc

@torch.no_grad()
def eval_full(model, loader, loss_fn):
    model.eval(); d = dev()
    total, n, correct = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(d, non_blocking=True), yb.to(d, non_blocking=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        total += float(loss) * xb.size(0); n += xb.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum())
    return (total / max(n,1)), (correct / max(n,1))

# ----------------- main -----------------
def main():
    args = build_args()
    ts = time.strftime("%Y%m%d-%H%M%S")
    def _fmt(x):
        s = f"{x:.6g}" if isinstance(x, float) else str(x)
        return s.replace('.', 'p').replace('-', 'm')
    run_name = f"run_gmlp_same_{ts}_p{args.patch}_{args.measure_data}"
    outdir = os.path.join(args.output_dir, run_name)
    logger = setup_logger(outdir, args.log_level)
    logger.info(f"Args: {json.dumps(vars(args), ensure_ascii=False)}")
    ckpt_root = os.path.join(args.output_dir, "checkpoints", run_name)
    ckpt_dir_A = os.path.join(ckpt_root, "A")
    ckpt_dir_B = os.path.join(ckpt_root, "B")

    # GPU 情報
    logger.info(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"cuda_device={torch.cuda.current_device()} name={torch.cuda.get_device_name(0)}")

    # data
    X, y = load_npz_fashion(args.input, logger)

    # 検証分割（A/B 共通で固定）
    val_seed = args.val_seed if args.val_seed is not None else args.train_seedA
    N = X.shape[0]
    n_val = int(round(N * args.val_ratio))
    n_train = N - n_val
    gsplit = torch.Generator().manual_seed(val_seed)
    train_ds, val_ds = random_split(TensorDataset(X, y), [n_train, n_val], generator=gsplit)
    logger.info(f"Split: train={len(train_ds)} val={len(val_ds)} (val_ratio={args.val_ratio})")

    # 計測用サンプル（A/B共通）
    rng = np.random.RandomState(args.data_seed)
    if args.measure_data == "train":
        source_indices = train_ds.indices
        source_data = X[source_indices]
        logger.info(f"Measure Target: TRAIN (from split, N={len(source_data)})")
    else:
        # val_ds is Subset
        val_indices = val_ds.indices
        source_data = X[val_indices]
        logger.info(f"Measure Target: VAL (from split, N={len(source_data)})")

    idx_meas = rng.choice(len(source_data), size=args.M, replace=False)
    X_meas = source_data[idx_meas].clone()
    logger.info(f"measure idx head: {idx_meas[:10].tolist()}")

    # 学習データローダ（A/B で独立したシャッフル）
    loaderA = make_loader(train_ds, args.batch_size, seed=args.train_seedA, logger=logger, shuffle=True)
    loaderB = make_loader(train_ds, args.batch_size, seed=args.train_seedB, logger=logger, shuffle=True)

    # 評価ローダ（shuffle=False で決定的）
    train_eval_loader = make_loader(train_ds, args.batch_size, seed=0, logger=logger, shuffle=False)
    val_loader        = make_loader(val_ds,   args.batch_size, seed=0, logger=logger, shuffle=False)

    # 観測ターゲット（TokenBN :pre/:post）
    if args.record_layers not in [":pre", ":post"]:
        raise ValueError("--record_layers must be ':pre' or ':post'")
    layer_list = [f"gmlp_blocks.{i}.bn{args.record_layers}" for i in range(len(train_eval_loader.dataset.dataset[0][0].view(1,1,28,28))) if False]  # dummy to keep editor happy
    layer_list = [f"gmlp_blocks.{i}.bn{args.record_layers}" for i in range(args.num_blocks)]
    #layer_list += [f"out_bn{args.record_layers}"]
    logger.info(f"record layers: {layer_list}")

    # 記録するエポック集合（1,3固定＋プログレッシブ）
    rec_epochs = build_epochs_1_3_then_progressive(
        total_epochs=args.epochs,
        start=args.record_start,
        mult=args.record_mult,
        max_interval=args.record_max,
        include_last=args.always_include_last
    )
    logger.info(f"Record epochs ({len(rec_epochs)} points): {rec_epochs}")

    # 共通
    loss_fn = nn.CrossEntropyLoss()
    d = dev()

    # ====== モデル A/B（同一初期化） ======
    set_seed_all(args.init_seed)
    modelA = gMLP(28, args.patch, args.d_model, args.d_ffn,
              args.num_blocks, 10, args.dropout).to(d)
    init_random_normal_and_bias_const(modelA, mean=0.0, std=1.0, bias_const=0.1)
    hashA0 = model_state_hash(modelA)
    logger.info(f"init hash A: {hashA0[:16]}")

    modelB = gMLP(28, args.patch, args.d_model, args.d_ffn,
              args.num_blocks, 10, args.dropout).to(d)
    modelB.load_state_dict(modelA.state_dict())
    hashB0 = model_state_hash(modelB)
    logger.info(f"init hash B: {hashB0[:16]} (A==B? {hashA0 == hashB0})")
    if hashA0 != hashB0:
        logger.warning("⚠️ A/B init hash mismatch — expected identical weights.")

    # 記録バッファ（spin & metrics）
    spinsA = {ln: [] for ln in layer_list}; timeA = []
    spinsB = {ln: [] for ln in layer_list}; timeB = []
    metrics_rows_A = []  # dict(epoch, split, loss, acc)
    metrics_rows_B = []
    ckpt_records_A = []
    ckpt_records_B = []

    # epoch0 記録（オプション）
    if not args.no_epoch0:
        sA0 = measure_spins(modelA, layer_list, X_meas, batch=min(256, args.batch_size), logger=logger)
        for ln in layer_list: spinsA[ln].append(sA0[ln])
        timeA.append(0); del sA0
        sB0 = measure_spins(modelB, layer_list, X_meas, batch=min(256, args.batch_size), logger=logger)
        for ln in layer_list: spinsB[ln].append(sB0[ln])
        timeB.append(0); del sB0
        pathA0 = os.path.join(ckpt_dir_A, "epoch0000.pt")
        atomic_save_state_dict(modelA, pathA0, logger)
        ckpt_records_A.append({"epoch": 0, "path": pathA0})
        pathB0 = os.path.join(ckpt_dir_B, "epoch0000.pt")
        atomic_save_state_dict(modelB, pathB0, logger)
        ckpt_records_B.append({"epoch": 0, "path": pathB0})
        # metrics も epoch0 で評価したい場合はここで eval_full を呼ぶ（必要なら解除）
        gc.collect()

    # オプティマイザ
    optA = torch.optim.Adam(modelA.parameters(), lr=args.lr)
    optB = torch.optim.Adam(modelB.parameters(), lr=args.lr)

    # ====== 学習 & 記録 ======
    for ep in range(1, args.epochs+1):
        _ = train_one_epoch(modelA, loaderA, optA, loss_fn, logger, tag=f"A/ep{ep}")
        _ = train_one_epoch(modelB, loaderB, optB, loss_fn, logger, tag=f"B/ep{ep}")

        if ep in rec_epochs:
            # ---- spins ----
            sA = measure_spins(modelA, layer_list, X_meas, batch=min(256, args.batch_size), logger=logger)
            for ln in layer_list: spinsA[ln].append(sA[ln])
            timeA.append(ep); del sA

            sB = measure_spins(modelB, layer_list, X_meas, batch=min(256, args.batch_size), logger=logger)
            for ln in layer_list: spinsB[ln].append(sB[ln])
            timeB.append(ep); del sB

            # ---- metrics (train/val) ----
            tr_loss_A, tr_acc_A = eval_full(modelA, train_eval_loader, loss_fn)
            va_loss_A, va_acc_A = eval_full(modelA, val_loader,         loss_fn)
            metrics_rows_A += [
                {"epoch": ep, "split": "train", "loss": f"{tr_loss_A:.6f}", "acc": f"{tr_acc_A:.6f}"},
                {"epoch": ep, "split": "val",   "loss": f"{va_loss_A:.6f}", "acc": f"{va_acc_A:.6f}"},
            ]
            tr_loss_B, tr_acc_B = eval_full(modelB, train_eval_loader, loss_fn)
            va_loss_B, va_acc_B = eval_full(modelB, val_loader,         loss_fn)
            metrics_rows_B += [
                {"epoch": ep, "split": "train", "loss": f"{tr_loss_B:.6f}", "acc": f"{tr_acc_B:.6f}"},
                {"epoch": ep, "split": "val",   "loss": f"{va_loss_B:.6f}", "acc": f"{va_acc_B:.6f}"},
            ]
            logger.info(f"[A/ep{ep}] eval train: loss={tr_loss_A:.4f} acc={tr_acc_A:.4f} | val: loss={va_loss_A:.4f} acc={va_acc_A:.4f}")
            logger.info(f"[B/ep{ep}] eval train: loss={tr_loss_B:.4f} acc={tr_acc_B:.4f} | val: loss={va_loss_B:.4f} acc={va_acc_B:.4f}")

            gc.collect()
            pathAe = os.path.join(ckpt_dir_A, f"epoch{ep:04d}.pt")
            atomic_save_state_dict(modelA, pathAe, logger)
            ckpt_records_A.append({"epoch": ep, "path": pathAe})
            pathBe = os.path.join(ckpt_dir_B, f"epoch{ep:04d}.pt")
            atomic_save_state_dict(modelB, pathBe, logger)
            ckpt_records_B.append({"epoch": ep, "path": pathBe})

    # ====== 保存 ======
    meta_common = dict(
        data_name="Fashion-MNIST",
        patch=args.patch, d_model=args.d_model, d_ffn=args.d_ffn, num_blocks=args.num_blocks,
        dropout=args.dropout, record_at=args.record_layers, M=args.M, epochs=args.epochs,
        record_epochs=rec_epochs,
        val_ratio=args.val_ratio,
        train_seedA=args.train_seedA,
        train_seedB=args.train_seedB
    )

    # spins + meta(A)
    dumpA = {"meta": {**meta_common, "tag":"A", "init_seed":args.init_seed,
                      "train_seed":args.train_seedA, "time": timeA, "record_layers": layer_list,
                      "metrics_epochs": rec_epochs, "checkpoints": ckpt_records_A}}
    for ln in layer_list:
        dumpA[ln] = np.stack([t.numpy() for t in spinsA[ln]], axis=0)  # (T, M, S, D)
    pathA = os.path.join(outdir, f"gmlp_same_model_spinA_{args.measure_data}_D{args.d_model}_F{args.d_ffn}_L{args.num_blocks}_M{args.M}_seed{args.init_seed}_trA{args.train_seedA}.pkl")
    atomic_save_pickle(dumpA, pathA, logger)

    # spins + meta(B)
    dumpB = {"meta": {**meta_common, "tag":"B", "init_seed":args.init_seed,
                      "train_seed":args.train_seedB, "time": timeB, "record_layers": layer_list,
                      "metrics_epochs": rec_epochs, "checkpoints": ckpt_records_B}}
    for ln in layer_list:
        dumpB[ln] = np.stack([t.numpy() for t in spinsB[ln]], axis=0)
    pathB = os.path.join(outdir, f"gmlp_same_model_spinB_{args.measure_data}_D{args.d_model}_F{args.d_ffn}_L{args.num_blocks}_M{args.M}_seed{args.init_seed}_trB{args.train_seedB}.pkl")
    atomic_save_pickle(dumpB, pathB, logger)

    # metrics CSV
    save_metrics_csv(os.path.join(outdir, "metrics_A.csv"), metrics_rows_A, logger)
    save_metrics_csv(os.path.join(outdir, "metrics_B.csv"), metrics_rows_B, logger)

    logger.info(f"🎉 Finished. Out dir: {outdir}")
    logger.info(f"A spins: {pathA}")
    logger.info(f"B spins: {pathB}")
    logger.info(f"Metrics: {os.path.join(outdir,'metrics_A.csv')} | {os.path.join(outdir,'metrics_B.csv')}")
    if ckpt_records_A:
        logger.info(f"A checkpoints dir: {ckpt_dir_A} (saved {len(ckpt_records_A)} states)")
    if ckpt_records_B:
        logger.info(f"B checkpoints dir: {ckpt_dir_B} (saved {len(ckpt_records_B)} states)")

if __name__ == "__main__":
    main()
