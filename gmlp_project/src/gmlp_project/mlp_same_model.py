#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MLP (ResMLP-like) dual-model spin recorder (A & B) + metrics recorder
- Based on gmlp_same_model.py, adapted for MLP architecture.
- A/B: SAME initialization (A copied to B), SAME training data, DIFFERENT shuffle order.
- Spin observation: MLP blocks' BN (:pre/:post) and out_bn.
- Recording epochs: {1,3} + progressive.
"""

import os, math, time, json, argparse, logging, hashlib, tempfile, shutil, gc, csv
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# ----------------- argparse -----------------
def build_args():
    p = argparse.ArgumentParser("MLP dual-model spin recorder (same init, different minibatch order)")
    p.add_argument("--input", type=str, default="/sqfs/work/cm9029/${USER_ID}/gmlp_project/data/fashion_mnist.npz",
                   help="Fashion-MNIST .npz")
    p.add_argument("--output_dir", type=str, default="/sqfs/work/cm9029/${USER_ID}/True_mlp_output_same_model")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)

    # MLP dims
    p.add_argument("--d_model", type=int, default=256)      # D
    p.add_argument("--d_ffn", type=int, default=1024)       # Expansion dim
    p.add_argument("--num_blocks", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training/Measurement
    p.add_argument("--init_seed", type=int, default=123, help="Shared Init seed for A/B")
    p.add_argument("--train_seedA", type=int, default=2025, help="Shuffle seed for A")
    p.add_argument("--train_seedB", type=int, default=4242, help="Shuffle seed for B")
    p.add_argument("--data_seed", type=int, default=4244, help="Measurement sample selection seed")
    p.add_argument("--M", type=int, default=1000, help="Number of spin measurement samples")
    p.add_argument("--record_layers", type=str, default=":post", choices=[":pre",":post"],
                   help="Record BN layers at :pre or :post")
    p.add_argument("--measure_data", type=str, default="train", choices=["train", "val"],
                   help="Which split to measure spins on: 'train' or 'val'")
    p.add_argument("--no-epoch0", action="store_true", default=True, help="Skip measurement at epoch 0")
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])

    # Progressive recording params
    p.add_argument("--record-start", type=int, default=2)
    p.add_argument("--record-mult", type=float, default=1.5)
    p.add_argument("--record-max", type=int, default=50)
    p.add_argument("--always-include-last", action="store_true", default=True, help="Always include last epoch")

    # Validation
    p.add_argument("--val_ratio", type=float, default=0.1, help="Validation holdout ratio")
    p.add_argument("--val_seed", type=int, default=None, help="Validation split seed (train_seedA if None)")
    
    # Legacy/Compatibility args
    p.add_argument("--patch", type=int, default=4, help="Ignored in MLP")

    return p.parse_args()

# ----------------- logging -----------------
def setup_logger(outdir, level="INFO"):
    os.makedirs(outdir, exist_ok=True)
    logger = logging.getLogger("MLP_Dual_SAME")
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

# ----------------- MLP Block (Residual) -----------------
class MLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model) # Observation point
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x): # (B, D)
        residual = x
        x = self.bn(x)      # blocks.i.bn :post
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x + residual

# ----------------- MLP Model -----------------
class MLP(nn.Module):
    def __init__(self, input_dim=784, d_model=256, d_ffn=1024,
                 num_blocks=10, num_classes=10, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.blocks = nn.ModuleList([
            MLPBlock(d_model, d_ffn, dropout)
            for _ in range(num_blocks)
        ])
        
        self.out_bn = nn.BatchNorm1d(d_model) 
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x_img): # (B, 1, 28, 28)
        B = x_img.size(0)
        x = x_img.view(B, -1) # (B, 784)
        x = self.input_proj(x)
        
        for blk in self.blocks:
            x = blk(x)
            
        x = self.out_bn(x)
        return self.classifier(x)

# ----------------- Init -----------------
def init_random_normal_and_bias_const(model: nn.Module, *, mean=0.0, std=1.0, bias_const=0.1):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if m.weight is not None:
                nn.init.normal_(m.weight, mean=mean, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, bias_const)

# ----------------- spin recording -----------------
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
        store[k] = torch.cat(v, dim=0)   # (M, D)
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
    run_name = f"run_mlp_same_{ts}_{args.measure_data}"
    outdir = os.path.join(args.output_dir, run_name)
    logger = setup_logger(outdir, args.log_level)
    logger.info(f"Args: {json.dumps(vars(args), ensure_ascii=False)}")
    ckpt_root = os.path.join(args.output_dir, "checkpoints", run_name)
    ckpt_dir_A = os.path.join(ckpt_root, "A")
    ckpt_dir_B = os.path.join(ckpt_root, "B")

    # GPU
    logger.info(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"cuda_device={torch.cuda.current_device()} name={torch.cuda.get_device_name(0)}")

    # Data
    X, y = load_npz_fashion(args.input, logger)

    # Validation Split
    val_seed = args.val_seed if args.val_seed is not None else args.train_seedA
    N = X.shape[0]
    n_val = int(round(N * args.val_ratio))
    n_train = N - n_val
    gsplit = torch.Generator().manual_seed(val_seed)
    train_ds, val_ds = random_split(TensorDataset(X, y), [n_train, n_val], generator=gsplit)
    logger.info(f"Split: train={len(train_ds)} val={len(val_ds)} (val_ratio={args.val_ratio})")

    # Measurement Samples
    # Measurement Samples
    # Choose source indices based on args.measure_data
    if args.measure_data == "train":
        source_indices = train_ds.indices
        logger.info(f"Measurement target: TRAIN (pool size={len(source_indices)})")
    else:
        source_indices = val_ds.indices
        logger.info(f"Measurement target: VAL (pool size={len(source_indices)})")

    rng = np.random.RandomState(args.data_seed)
    # Sample M indices from the available source indices
    if len(source_indices) < args.M:
        logger.warning(f"Pool size {len(source_indices)} < M={args.M}. Using all available.")
        idx_meas = np.array(source_indices)
    else:
        idx_meas = rng.choice(source_indices, size=args.M, replace=False)

    X_meas = X[idx_meas].clone()
    logger.info(f"measure idx head: {idx_meas[:10].tolist()}")

    # Loaders (Different shuffles)
    loaderA = make_loader(train_ds, args.batch_size, seed=args.train_seedA, logger=logger, shuffle=True)
    loaderB = make_loader(train_ds, args.batch_size, seed=args.train_seedB, logger=logger, shuffle=True)
    train_eval_loader = make_loader(train_ds, args.batch_size, seed=0, logger=logger, shuffle=False)
    val_loader        = make_loader(val_ds,   args.batch_size, seed=0, logger=logger, shuffle=False)

    # Recording Targets
    if args.record_layers not in [":pre", ":post"]:
        raise ValueError("--record_layers must be ':pre' or ':post'")
    
    layer_list = [f"blocks.{i}.bn{args.record_layers}" for i in range(args.num_blocks)]
    logger.info(f"record layers: {layer_list}")

    # Recording Strategy
    rec_epochs = build_epochs_1_3_then_progressive(
        total_epochs=args.epochs,
        start=args.record_start,
        mult=args.record_mult,
        max_interval=args.record_max,
        include_last=args.always_include_last
    )
    logger.info(f"Record epochs ({len(rec_epochs)} points): {rec_epochs}")

    loss_fn = nn.CrossEntropyLoss()
    d = dev()

    # ====== Model A/B (Same Initialization) ======
    set_seed_all(args.init_seed)
    
    # Model A
    modelA = MLP(d_model=args.d_model, d_ffn=args.d_ffn, num_blocks=args.num_blocks, dropout=args.dropout).to(d)
    init_random_normal_and_bias_const(modelA, mean=0.0, std=1.0, bias_const=0.1)
    hashA0 = model_state_hash(modelA)
    logger.info(f"init hash A: {hashA0[:16]}")

    # Model B (Copy A)
    modelB = MLP(d_model=args.d_model, d_ffn=args.d_ffn, num_blocks=args.num_blocks, dropout=args.dropout).to(d)
    modelB.load_state_dict(modelA.state_dict())
    hashB0 = model_state_hash(modelB)
    logger.info(f"init hash B: {hashB0[:16]} (A==B? {hashA0 == hashB0})")
    if hashA0 != hashB0:
        logger.warning("⚠️ A/B init hash mismatch — expected identical weights.")

    # Buffers
    spinsA = {ln: [] for ln in layer_list}; timeA = []
    spinsB = {ln: [] for ln in layer_list}; timeB = []
    metrics_rows_A = []
    metrics_rows_B = []
    ckpt_records_A = []
    ckpt_records_B = []

    # Epoch 0
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
        gc.collect()

    # Optimizer
    optA = torch.optim.Adam(modelA.parameters(), lr=args.lr)
    optB = torch.optim.Adam(modelB.parameters(), lr=args.lr)

    # Train
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

            # ---- metrics ----
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

    # Save
    meta_common = dict(
        data_name="Fashion-MNIST",
        d_model=args.d_model, d_ffn=args.d_ffn, num_blocks=args.num_blocks,
        dropout=args.dropout, record_at=args.record_layers, M=args.M, epochs=args.epochs,
        record_epochs=rec_epochs,
        val_ratio=args.val_ratio,
        train_seedA=args.train_seedA,
        train_seedB=args.train_seedB,
        measure_data=args.measure_data,
        arch="MLP_SAME"
    )

    # Dump A
    dumpA = {"meta": {**meta_common, "tag":"A", "init_seed":args.init_seed,
                      "train_seed":args.train_seedA, "time": timeA, "record_layers": layer_list,
                      "metrics_epochs": rec_epochs, "checkpoints": ckpt_records_A}}
    for ln in layer_list:
        dumpA[ln] = np.stack([t.numpy() for t in spinsA[ln]], axis=0)
    pathA = os.path.join(outdir, f"mlp_same_model_spinA_{args.measure_data}_D{args.d_model}_F{args.d_ffn}_L{args.num_blocks}_M{args.M}_seed{args.init_seed}_trA{args.train_seedA}.pkl")
    atomic_save_pickle(dumpA, pathA, logger)

    # Dump B
    dumpB = {"meta": {**meta_common, "tag":"B", "init_seed":args.init_seed,
                      "train_seed":args.train_seedB, "time": timeB, "record_layers": layer_list,
                      "metrics_epochs": rec_epochs, "checkpoints": ckpt_records_B}}
    for ln in layer_list:
        dumpB[ln] = np.stack([t.numpy() for t in spinsB[ln]], axis=0)
    pathB = os.path.join(outdir, f"mlp_same_model_spinB_{args.measure_data}_D{args.d_model}_F{args.d_ffn}_L{args.num_blocks}_M{args.M}_seed{args.init_seed}_trB{args.train_seedB}.pkl")
    atomic_save_pickle(dumpB, pathB, logger)

    # Metrics CSV
    save_metrics_csv(os.path.join(outdir, "metrics_A.csv"), metrics_rows_A, logger)
    save_metrics_csv(os.path.join(outdir, "metrics_B.csv"), metrics_rows_B, logger)

    logger.info(f"🎉 Finished. Out dir: {outdir}")
    logger.info(f"A spins: {pathA}")
    logger.info(f"B spins: {pathB}")

if __name__ == "__main__":
    main()
