#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fully-connected MLP (+BatchNorm) + pre/post spin recorder + Progressive recording (epoch1 start)
- 観測: Linear の直後に BatchNorm1d を入れ、その :post（=BN 出力）を観測（:pre も可）
- 記録: epoch0 は記録しない。epoch1,2,3 は必ず記録し、以降は間隔を広げて記録
- 最終 epoch を常に含める (--always-include-last)
- A/B は同一の測定サンプル M、同一の記録 epoch セットで記録
- スピンは CPU float16 に退避、原子的保存＋sha256
"""

import os, math, time, json, argparse, logging, hashlib, tempfile, shutil, gc
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------- argparse -----------------
def build_args():
    p = argparse.ArgumentParser("MLP+BN spin recorder with progressive recording (start at epoch 1)")
    p.add_argument("--input", type=str, default="/sqfs/home/${USER_ID}/workspace/gmlp_project/data/fashion_mnist.npz",
                   help="Fashion-MNIST .npz (x_train,y_train など)")
    p.add_argument("--output_dir", type=str, default="/sqfs/work/cm9029/${USER_ID}/mlp_output_diff")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--activation", type=str, default="relu", choices=["relu"])
    p.add_argument("--seedA", type=int, default=123)
    p.add_argument("--seedB", type=int, default=456)
    p.add_argument("--data_seed", type=int, default=4244)
    p.add_argument("--M", type=int, default=1000)
    p.add_argument("--record_layers", type=str, default=":post", help="':pre' or ':post'（BNの前/後）")
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    # Progressive recording params（epoch1,2,3は必ず記録）
    p.add_argument("--record-start", type=int, default=2, help="最初の可変間隔（3の次に足す最初の間隔）")
    p.add_argument("--record-mult", type=float, default=1.5, help="記録間隔の倍率")
    p.add_argument("--record-max", type=int, default=50, help="記録間隔の上限")
    p.add_argument("--always-include-last", action="store_true", default=True, help="最終epochを必ず含める")
    return p.parse_args()

# ----------------- logging -----------------
def setup_logger(outdir, level="INFO"):
    os.makedirs(outdir, exist_ok=True)
    logger = logging.getLogger("MLP_BN_Progressive")
    logger.setLevel(getattr(logging, level))
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(outdir, "train.log")); fh.setFormatter(fmt)
    logger.handlers.clear(); logger.addHandler(sh); logger.addHandler(fh)
    return logger

# ----------------- utils -----------------
def dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed:int):
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
    x = torch.tensor(x, dtype=torch.float32).view(x.shape[0], -1) / 255.0
    y = torch.tensor(y, dtype=torch.long)
    logger.info(f"Loaded X={tuple(x.shape)} y={tuple(y.shape)}")
    return x, y

def make_loader(X, y, bs, logger):
    dl = DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
    logger.info(f"Train steps/epoch ≈ {math.ceil(len(X)/bs)} (bs={bs})")
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

def model_state_hash(model):
    with torch.no_grad():
        vec = torch.cat([p.detach().float().flatten().cpu() for p in model.parameters()])
    return hashlib.sha256(vec.numpy().tobytes()).hexdigest()

# ----------------- 初期化（重み~N(0,1), バイアス=0.1） -----------------
def init_random_normal_and_bias_const(model: nn.Module, *, mean=0.0, std=1.0, bias_const=0.1):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if m.weight is not None:
                nn.init.normal_(m.weight, mean=mean, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, bias_const)

# ----------------- モデル（Linear -> BN -> ReLU） -----------------
class FullyConnectedBlockBN(nn.Module):
    def __init__(self, in_dim, out_dim, act="relu"):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        self.bn = nn.BatchNorm1d(out_dim, affine=True, momentum=0.1, eps=1e-5)
        self.act = nn.ReLU() if act == "relu" else nn.ReLU()
    def forward(self, x):
        x = self.fc(x)   # features.i.fc :post（:pre はこの入力）
        x = self.bn(x)   # features.i.bn :post（ここを観測）
        x = self.act(x)
        return x

class FullyConnectedMLPBN(nn.Module):
    def __init__(self, in_dim=28*28, hidden=512, num_layers=10, num_classes=10, act="relu"):
        super().__init__()
        blocks = []
        d = in_dim
        for _ in range(num_layers):
            blocks += [FullyConnectedBlockBN(d, hidden, act=act)]
            d = hidden
        self.features = nn.Sequential(*blocks)
        self.out_fc = nn.Linear(hidden, num_classes)
        self.out_bn = nn.BatchNorm1d(num_classes, affine=True, momentum=0.1, eps=1e-5)
    def forward(self, x):
        x = self.features(x)
        x = self.out_fc(x)     # out_fc:post
        x = self.out_bn(x)     # out_bn:post（観測可能）
        return x

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
        store[k] = torch.cat(v, dim=0)
        logger.info(f"[spin] {k}: {tuple(store[k].shape)} dtype={store[k].dtype}")
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return store

# ----------------- Progressive recording scheduler -----------------
def build_progressive_epochs(total_epochs: int,
                             start: int,
                             mult: float,
                             max_interval: int,
                             include_last: bool) -> List[int]:
    """
    仕様:
      - epoch0 は記録しない
      - epoch1,2,3 は必ず記録（total_epochs が小さければ入る範囲だけ）
      - 以降は interval を広げながら追加（start, mult, max_interval を使用）
      - 最終 epoch（total_epochs）を include_last=True のとき必ず含める
    """
    epochs = [e for e in (1, 2, 3) if 1 <= e <= total_epochs]

    cur = epochs[-1] if len(epochs) else 0
    interval = max(1, start)
    i = 0
    while True:
        cur = cur + interval
        if cur > total_epochs:
            break
        if cur not in epochs and cur >= 1:
            epochs.append(cur)
        interval = min(max_interval, max(1, int(round(interval * mult))))
        i += 1
        if i > 10_000:
            break

    if include_last and total_epochs >= 1 and total_epochs not in epochs:
        epochs.append(total_epochs)

    epochs = sorted(set([e for e in epochs if 1 <= e <= total_epochs]))
    return epochs

# ----------------- train -----------------
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
    logger.info(f"[{tag}] train_loss={total/max(n,1):.4f} acc={acc:.4f}")
    return total / max(n, 1), acc

# ----------------- main -----------------
def main():
    args = build_args()
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(args.output_dir, f"run_{ts}")
    logger = setup_logger(outdir, args.log_level)
    logger.info(f"Args: {json.dumps(vars(args), ensure_ascii=False)}")

    # data & measure subset
    X, y = load_npz_fashion(args.input, logger)
    rng = np.random.RandomState(args.data_seed)
    idx = rng.choice(X.shape[0], size=args.M, replace=False)
    logger.info(f"measure idx head: {idx[:10].tolist()}")
    X_meas = X[idx].clone()

    # record targets（BN 後を例: ':post'）
    if args.record_layers not in [":pre", ":post"]:
        raise ValueError("--record_layers must be ':pre' or ':post'")
    layer_list = [f"features.{i}.bn{args.record_layers}" for i in range(args.num_layers)]
    layer_list += [f"out_bn{args.record_layers}"]
    logger.info(f"record layers: {layer_list}")

    # Progressive epoch set（A/B 共通, epoch1スタート）
    rec_epochs = build_progressive_epochs(
        total_epochs=args.epochs,
        start=args.record_start,
        mult=args.record_mult,
        max_interval=args.record_max,
        include_last=args.always_include_last
    )
    logger.info(f"Progressive record epochs ({len(rec_epochs)} points): {rec_epochs}")

    # common
    loss_fn = nn.CrossEntropyLoss()
    loader = make_loader(X, y, args.batch_size, logger)
    d = dev()

    # --------- A ----------
    set_seed(args.seedA)
    modelA = FullyConnectedMLPBN(
        in_dim=28*28, hidden=args.N, num_layers=args.num_layers,
        num_classes=args.num_classes, act=args.activation
    ).to(d)
    init_random_normal_and_bias_const(modelA, mean=0.0, std=1.0, bias_const=0.1)
    logger.info(f"A init hash: {model_state_hash(modelA)[:16]}")
    spinsA = {ln: [] for ln in layer_list}; timeA = []

    optA = torch.optim.Adam(modelA.parameters(), lr=args.lr)
    for ep in range(1, args.epochs+1):
        _, acc = train_one_epoch(modelA, loader, optA, loss_fn, logger, tag=f"A/ep{ep}")
        if ep in rec_epochs:
            s = measure_spins(modelA, layer_list, X_meas, batch=min(256,args.batch_size), logger=logger)
            for ln in layer_list: spinsA[ln].append(s[ln])
            timeA.append(ep)
            del s; gc.collect()

    dumpA = {"meta": {
                "tag":"A","epochs":args.epochs,"M":args.M,"layers":layer_list,
                "record_at": args.record_layers, "seed": args.seedA, "N": args.N,
                "num_layers": args.num_layers, "init":"normal(0,1)+bias0.1",
                "time": timeA, "record_epochs": rec_epochs
             }}
    for ln in layer_list:
        dumpA[ln] = np.stack([t.numpy() for t in spinsA[ln]], axis=0)  # (len(timeA), M, dim)
    pathA = os.path.join(outdir, f"mlp_spinA_prog_N{args.N}_L{args.num_layers}_M{args.M}_seed{args.seedA}.pkl")
    atomic_save_pickle(dumpA, pathA, logger)

    # --------- B ----------
    set_seed(args.seedB)
    modelB = FullyConnectedMLPBN(
        in_dim=28*28, hidden=args.N, num_layers=args.num_layers,
        num_classes=args.num_classes, act=args.activation
    ).to(d)
    init_random_normal_and_bias_const(modelB, mean=0.0, std=1.0, bias_const=0.1)
    logger.info(f"B init hash: {model_state_hash(modelB)[:16]}")
    spinsB = {ln: [] for ln in layer_list}; timeB = []

    optB = torch.optim.Adam(modelB.parameters(), lr=args.lr)
    for ep in range(1, args.epochs+1):
        _, acc = train_one_epoch(modelB, loader, optB, loss_fn, logger, tag=f"B/ep{ep}")
        if ep in rec_epochs:
            s = measure_spins(modelB, layer_list, X_meas, batch=min(256,args.batch_size), logger=logger)
            for ln in layer_list: spinsB[ln].append(s[ln])
            timeB.append(ep)
            del s; gc.collect()

    dumpB = {"meta": {
                "tag":"B","epochs":args.epochs,"M":args.M,"layers":layer_list,
                "record_at": args.record_layers, "seed": args.seedB, "N": args.N,
                "num_layers": args.num_layers, "init":"normal(0,1)+bias0.1",
                "time": timeB, "record_epochs": rec_epochs
             }}
    for ln in layer_list:
        dumpB[ln] = np.stack([t.numpy() for t in spinsB[ln]], axis=0)
    pathB = os.path.join(outdir, f"mlp_spinB_prog_N{args.N}_L{args.num_layers}_M{args.M}_seed{args.seedB}.pkl")
    atomic_save_pickle(dumpB, pathB, logger)

    logger.info(f"Done. Out: {outdir}")

if __name__ == "__main__":
    main()

