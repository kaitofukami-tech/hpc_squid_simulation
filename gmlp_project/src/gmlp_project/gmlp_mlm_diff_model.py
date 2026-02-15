#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gMLP (SGU) dual-model spin recorder for Masked Language Modeling (MLM).
- mode=diff : different initializations, same minibatch order & masking RNG
- mode=same : same initialization copied to B, different minibatch order & masking RNG
The gMLP block body is unchanged from the vision experiments; only the
input/output heads are swapped for token/position embeddings and vocab logits.
"""

import os, math, time, json, argparse, logging, hashlib, tempfile, shutil, gc, csv
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mlm_dataset import (
    build_wikitext_dataset,
    MLMDataCollator,
)

# ----------------- argparse -----------------
def build_args():
    p = argparse.ArgumentParser("gMLP+SGU dual-model (MLM) spin recorder")
    # data/tokenizer
    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--dataset_dir", type=str, default=None, help="ローカルに展開済みのHFデータセットディレクトリを指定する場合")
    p.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    p.add_argument("--tokenizer_path", type=str, default=None, help="ローカルに展開済みのトークナイザーを指定する場合")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--offline", action="store_true", help="ネットアクセスせずキャッシュ/ローカルのみを使う")

    # output
    p.add_argument("--output_dir", type=str, default="./gmlp_mlm_output")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loader_workers", type=int, default=0)

    # gMLP dims
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--d_ffn", type=int, default=1536)
    p.add_argument("--num_blocks", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.1)

    # replica setup
    p.add_argument("--mode", type=str, choices=["diff", "same"], default="diff")
    p.add_argument("--init_seedA", type=int, default=123)
    p.add_argument("--init_seedB", type=int, default=456)
    p.add_argument("--init_seed", type=int, default=123, help="used when mode=same")
    p.add_argument("--train_seed", type=int, default=2025, help="used when mode=diff")
    p.add_argument("--train_seedA", type=int, default=2025, help="used when mode=same")
    p.add_argument("--train_seedB", type=int, default=4242, help="used when mode=same")
    p.add_argument("--data_seed", type=int, default=4244, help="measurement sample & mask seed")
    p.add_argument("--M", type=int, default=256, help="spin measurement samples")
    p.add_argument("--record_layers", type=str, default=":post", choices=[":pre", ":post"])
    p.add_argument("--no-epoch0", action="store_true", default=True)
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])

    # Progressive recoding params
    p.add_argument("--record-start", type=int, default=2)
    p.add_argument("--record-mult", type=float, default=1.5)
    p.add_argument("--record-max", type=int, default=50)
    p.add_argument("--always-include-last", action="store_true", default=True)
    return p.parse_args()


# ----------------- logging -----------------
def setup_logger(outdir, level="INFO"):
    os.makedirs(outdir, exist_ok=True)
    logger = logging.getLogger("gMLP_SGU_MLM_Dual")
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


def make_loader(dataset, bs, seed, logger, *, collate_fn, shuffle=True, num_workers=0):
    g = torch.Generator()
    g.manual_seed(seed)
    dl = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=g if shuffle else None,
        collate_fn=collate_fn,
    )
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


# ----------------- SGU (token mixing gate) -----------------
class SpatialGatingUnitGamma(nn.Module):
    def __init__(self, dim_half: int, seq_len: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim_half)
        self.proj = nn.Linear(seq_len, seq_len, bias=True)
        nn.init.constant_(self.proj.bias, 1.0)

    def forward(self, v):  # v:(B,S,dim_half)
        g = self.norm(v)            # (B,S,dim_half)
        g = g.transpose(1, 2)       # (B,dim_half,S)
        g = self.proj(g)            # (B,dim_half,S)
        g = g.transpose(1, 2)       # (B,S,dim_half)
        return g


# ----------------- gMLP Block -----------------
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
        x = self.bn(x)
        x = self.channel_proj(x)
        x = self.act(x)
        u, v = x.chunk(2, dim=-1)
        gate = self.sgu(v)
        x = u * gate
        x = self.channel_proj_out(x)
        x = self.drop(x)
        return x + residual


# ----------------- gMLP Model for MLM -----------------
class gMLPForMLM(nn.Module):
    def __init__(self, vocab_size:int, seq_len:int, d_model:int, d_ffn:int,
                 num_blocks:int, dropout:float, pad_token_id:int):
        super().__init__()
        if d_ffn % 2 != 0:
            raise ValueError("d_ffn must be divisible by 2.")
        self.seq_len = seq_len
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.gmlp_blocks = nn.ModuleList([
            gMLPBlock(d_model, d_ffn, seq_len, dropout)
            for _ in range(num_blocks)
        ])
        self.out_bn = TokenBN(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):  # input_ids:(B,S)
        B, S = input_ids.shape
        if S != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {S}.")
        device = input_ids.device
        pos = torch.arange(self.seq_len, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).type_as(x)
            x = x * mask
        for blk in self.gmlp_blocks:
            x = blk(x)
        x = self.out_bn(x)
        return self.lm_head(x)


# ----------------- init -----------------
def init_random_normal_and_bias_const(model: nn.Module, *, mean=0.0, std=1.0, bias_const=0.1):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if m.weight is not None:
                nn.init.normal_(m.weight, mean=mean, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, bias_const)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=mean, std=std)
            if getattr(m, "padding_idx", None) is not None and m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0.0)


# ----------------- spin recording -----------------
def _parse_layer_flag(name: str) -> Tuple[str, bool]:
    return name.replace(':pre','').replace(':post',''), name.endswith(':pre')


@torch.no_grad()
def measure_spins(model: nn.Module, layer_names: List[str], dataset: TensorDataset, batch: int, logger):
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

    dl = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    d = dev()
    for xb, attn in dl:
        _ = model(xb.to(d, non_blocking=True), attn.to(d, non_blocking=True))
    for h in handles: h.remove()

    for k, v in store.items():
        store[k] = torch.cat(v, dim=0)   # (M,S,D)
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
def _masked_accuracy(logits, labels):
    with torch.no_grad():
        mask = labels != -100
        if mask.sum() == 0:
            return 0.0
        preds = logits.argmax(dim=-1)
        correct = (preds[mask] == labels[mask]).sum().item()
        return correct / mask.sum().item()


def train_one_epoch(model, loader, opt, loss_fn, logger, tag):
    model.train(); d = dev()
    total_loss, total_masked, total_correct = 0.0, 0, 0
    for batch in loader:
        xb = batch["input_ids"].to(d, non_blocking=True)
        attn = batch["attention_mask"].to(d, non_blocking=True)
        labels = batch["labels"].to(d, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(xb, attn)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward(); opt.step()
        mask = labels != -100
        mask_count = int(mask.sum())
        total_masked += mask_count
        total_loss += float(loss) * max(mask_count, 1)
        if total_masked > 0:
            preds = logits.argmax(dim=-1)
            total_correct += int((preds[mask] == labels[mask]).sum())
        del xb, attn, labels, logits, loss, mask
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    acc = (total_correct / total_masked) if total_masked > 0 else 0.0
    avg_loss = total_loss / max(total_masked, 1)
    logger.info(f"[{tag}] train_loss(masked-avg)={avg_loss:.4f} acc_masked={acc:.4f}")
    return avg_loss, acc


@torch.no_grad()
def eval_full(model, loader, loss_fn):
    model.eval(); d = dev()
    total_loss, total_masked, total_correct = 0.0, 0, 0
    for batch in loader:
        xb = batch["input_ids"].to(d, non_blocking=True)
        attn = batch["attention_mask"].to(d, non_blocking=True)
        labels = batch["labels"].to(d, non_blocking=True)
        logits = model(xb, attn)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        mask = labels != -100
        mask_count = int(mask.sum())
        total_masked += mask_count
        total_loss += float(loss) * max(mask_count, 1)
        if total_masked > 0:
            preds = logits.argmax(dim=-1)
            total_correct += int((preds[mask] == labels[mask]).sum())
    acc = (total_correct / total_masked) if total_masked > 0 else 0.0
    avg_loss = total_loss / max(total_masked, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, acc, ppl


# ----------------- main -----------------
def main():
    args = build_args()
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{ts}_{args.mode}_seq{args.seq_len}"
    outdir = os.path.join(args.output_dir, run_name)
    logger = setup_logger(outdir, args.log_level)
    logger.info(f"Args: {json.dumps(vars(args), ensure_ascii=False)}")
    ckpt_root = os.path.join(args.output_dir, "checkpoints", run_name)
    ckpt_dir_A = os.path.join(ckpt_root, "A")
    ckpt_dir_B = os.path.join(ckpt_root, "B")

    logger.info(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"cuda_device={torch.cuda.current_device()} name={torch.cuda.get_device_name(0)}")

    # data/tokenizer
    logger.info(f"Loading dataset {args.dataset_name}/{args.dataset_config} (offline={args.offline})...")
    train_ds, val_ds, tokenizer = build_wikitext_dataset(
        tokenizer_name=args.tokenizer_name,
        seq_len=args.seq_len,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        cache_dir=args.cache_dir,
        dataset_dir=args.dataset_dir,
        tokenizer_path=args.tokenizer_path,
        offline=args.offline,
    )
    vocab_size = len(tokenizer)
    logger.info(f"Dataset sizes: train={len(train_ds)} val={len(val_ds)} vocab={vocab_size}")

    # collators
    if args.mode == "diff":
        collatorA = MLMDataCollator(tokenizer, mlm_probability=args.mlm_probability, seed=args.train_seed)
        collatorB = MLMDataCollator(tokenizer, mlm_probability=args.mlm_probability, seed=args.train_seed)
    else:
        collatorA = MLMDataCollator(tokenizer, mlm_probability=args.mlm_probability, seed=args.train_seedA)
        collatorB = MLMDataCollator(tokenizer, mlm_probability=args.mlm_probability, seed=args.train_seedB)
    eval_collator = MLMDataCollator(tokenizer, mlm_probability=args.mlm_probability, seed=args.data_seed)

    # measurement samples (fixed mask)
    rng = np.random.RandomState(args.data_seed)
    idx_meas = rng.choice(len(train_ds), size=min(args.M, len(train_ds)), replace=False)
    meas_batch = [train_ds[int(i)] for i in idx_meas]
    meas_batch = [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in meas_batch]
    meas_masked = eval_collator(meas_batch)
    X_meas = TensorDataset(meas_masked["input_ids"], meas_masked["attention_mask"])
    logger.info(f"measure idx head: {idx_meas[:10].tolist()}")

    # loaders
    if args.mode == "diff":
        loaderA = make_loader(train_ds, args.batch_size, seed=args.train_seed, logger=logger, collate_fn=collatorA, shuffle=True, num_workers=args.loader_workers)
        loaderB = make_loader(train_ds, args.batch_size, seed=args.train_seed, logger=logger, collate_fn=collatorB, shuffle=True, num_workers=args.loader_workers)
    else:
        loaderA = make_loader(train_ds, args.batch_size, seed=args.train_seedA, logger=logger, collate_fn=collatorA, shuffle=True, num_workers=args.loader_workers)
        loaderB = make_loader(train_ds, args.batch_size, seed=args.train_seedB, logger=logger, collate_fn=collatorB, shuffle=True, num_workers=args.loader_workers)

    train_eval_loader = make_loader(train_ds, args.batch_size, seed=0, logger=logger, collate_fn=eval_collator, shuffle=False, num_workers=args.loader_workers)
    val_loader        = make_loader(val_ds,   args.batch_size, seed=0, logger=logger, collate_fn=eval_collator, shuffle=False, num_workers=args.loader_workers)

    # record targets
    layer_list = [f"gmlp_blocks.{i}.bn{args.record_layers}" for i in range(args.num_blocks)]
    logger.info(f"record layers: {layer_list}")

    # record epochs
    rec_epochs = build_epochs_1_3_then_progressive(
        total_epochs=args.epochs,
        start=args.record_start,
        mult=args.record_mult,
        max_interval=args.record_max,
        include_last=args.always_include_last
    )
    logger.info(f"Record epochs ({len(rec_epochs)} points): {rec_epochs}")

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    d = dev()

    # models
    if args.mode == "diff":
        set_seed_all(args.init_seedA)
        modelA = gMLPForMLM(vocab_size, args.seq_len, args.d_model, args.d_ffn, args.num_blocks, args.dropout, tokenizer.pad_token_id).to(d)
        init_random_normal_and_bias_const(modelA, mean=0.0, std=1.0, bias_const=0.1)
        hashA0 = model_state_hash(modelA)

        set_seed_all(args.init_seedB)
        modelB = gMLPForMLM(vocab_size, args.seq_len, args.d_model, args.d_ffn, args.num_blocks, args.dropout, tokenizer.pad_token_id).to(d)
        init_random_normal_and_bias_const(modelB, mean=0.0, std=1.0, bias_const=0.1)
        hashB0 = model_state_hash(modelB)
    else:
        set_seed_all(args.init_seed)
        modelA = gMLPForMLM(vocab_size, args.seq_len, args.d_model, args.d_ffn, args.num_blocks, args.dropout, tokenizer.pad_token_id).to(d)
        init_random_normal_and_bias_const(modelA, mean=0.0, std=1.0, bias_const=0.1)
        hashA0 = model_state_hash(modelA)

        modelB = gMLPForMLM(vocab_size, args.seq_len, args.d_model, args.d_ffn, args.num_blocks, args.dropout, tokenizer.pad_token_id).to(d)
        modelB.load_state_dict(modelA.state_dict())
        hashB0 = model_state_hash(modelB)

    logger.info(f"init hash A: {hashA0[:16]}")
    logger.info(f"init hash B: {hashB0[:16]}")

    # buffers
    spinsA = {ln: [] for ln in layer_list}; timeA = []
    spinsB = {ln: [] for ln in layer_list}; timeB = []
    metrics_rows_A = []
    metrics_rows_B = []
    ckpt_records_A = []
    ckpt_records_B = []

    # epoch0
    if not args.no_epoch0:
        sA0 = measure_spins(modelA, layer_list, X_meas, batch=min(256, args.batch_size), logger=logger)
        for ln in layer_list: spinsA[ln].append(sA0[ln])
        timeA.append(0); del sA0
        sB0 = measure_spins(modelB, layer_list, X_meas, batch=min(256, args.batch_size), logger=logger)
        for ln in layer_list: spinsB[ln].append(sB0[ln])
        timeB.append(0); del sB0
        pathA0 = os.path.join(ckpt_dir_A, "epoch0000.pt")
        atomic_save_state_dict(modelA, pathA0, logger); ckpt_records_A.append({"epoch": 0, "path": pathA0})
        pathB0 = os.path.join(ckpt_dir_B, "epoch0000.pt")
        atomic_save_state_dict(modelB, pathB0, logger); ckpt_records_B.append({"epoch": 0, "path": pathB0})
        gc.collect()

    optA = torch.optim.Adam(modelA.parameters(), lr=args.lr)
    optB = torch.optim.Adam(modelB.parameters(), lr=args.lr)

    # train
    for ep in range(1, args.epochs+1):
        _ = train_one_epoch(modelA, loaderA, optA, loss_fn, logger, tag=f"A/ep{ep}")
        _ = train_one_epoch(modelB, loaderB, optB, loss_fn, logger, tag=f"B/ep{ep}")

        if ep in rec_epochs:
            sA = measure_spins(modelA, layer_list, X_meas, batch=min(256, args.batch_size), logger=logger)
            for ln in layer_list: spinsA[ln].append(sA[ln])
            timeA.append(ep); del sA

            sB = measure_spins(modelB, layer_list, X_meas, batch=min(256, args.batch_size), logger=logger)
            for ln in layer_list: spinsB[ln].append(sB[ln])
            timeB.append(ep); del sB

            tr_loss_A, tr_acc_A, tr_ppl_A = eval_full(modelA, train_eval_loader, loss_fn)
            va_loss_A, va_acc_A, va_ppl_A = eval_full(modelA, val_loader,         loss_fn)
            metrics_rows_A += [
                {"epoch": ep, "split": "train", "loss": f"{tr_loss_A:.6f}", "acc": f"{tr_acc_A:.6f}"},
                {"epoch": ep, "split": "val",   "loss": f"{va_loss_A:.6f}", "acc": f"{va_acc_A:.6f}"},
            ]
            tr_loss_B, tr_acc_B, tr_ppl_B = eval_full(modelB, train_eval_loader, loss_fn)
            va_loss_B, va_acc_B, va_ppl_B = eval_full(modelB, val_loader,         loss_fn)
            metrics_rows_B += [
                {"epoch": ep, "split": "train", "loss": f"{tr_loss_B:.6f}", "acc": f"{tr_acc_B:.6f}"},
                {"epoch": ep, "split": "val",   "loss": f"{va_loss_B:.6f}", "acc": f"{va_acc_B:.6f}"},
            ]
            logger.info(f"[A/ep{ep}] eval train: loss={tr_loss_A:.4f} acc={tr_acc_A:.4f} ppl≈{tr_ppl_A:.2f} | val: loss={va_loss_A:.4f} acc={va_acc_A:.4f} ppl≈{va_ppl_A:.2f}")
            logger.info(f"[B/ep{ep}] eval train: loss={tr_loss_B:.4f} acc={tr_acc_B:.4f} ppl≈{tr_ppl_B:.2f} | val: loss={va_loss_B:.4f} acc={va_acc_B:.4f} ppl≈{va_ppl_B:.2f}")

            gc.collect()
            pathAe = os.path.join(ckpt_dir_A, f"epoch{ep:04d}.pt")
            atomic_save_state_dict(modelA, pathAe, logger)
            ckpt_records_A.append({"epoch": ep, "path": pathAe})
            pathBe = os.path.join(ckpt_dir_B, f"epoch{ep:04d}.pt")
            atomic_save_state_dict(modelB, pathBe, logger)
            ckpt_records_B.append({"epoch": ep, "path": pathBe})

    # save spins/metrics
    meta_common = dict(
        data_name=f"{args.dataset_name}-{args.dataset_config}",
        seq_len=args.seq_len, d_model=args.d_model, d_ffn=args.d_ffn, num_blocks=args.num_blocks,
        dropout=args.dropout, record_at=args.record_layers, M=args.M, epochs=args.epochs,
        record_epochs=rec_epochs, mlm_probability=args.mlm_probability,
        tokenizer=args.tokenizer_name
    )

    dumpA = {"meta": {**meta_common, "tag":"A",
                      "init_seed": args.init_seedA if args.mode=="diff" else args.init_seed,
                      "train_seed": args.train_seed if args.mode=="diff" else args.train_seedA,
                      "time": timeA, "record_layers": layer_list,
                      "metrics_epochs": rec_epochs, "checkpoints": ckpt_records_A}}
    for ln in layer_list:
        dumpA[ln] = np.stack([t.numpy() for t in spinsA[ln]], axis=0)
    pathA = os.path.join(outdir, f"gmlp_mlm_spinA_D{args.d_model}_F{args.d_ffn}_L{args.num_blocks}_M{args.M}_seedA{args.init_seedA if args.mode=='diff' else args.init_seed}.pkl")
    atomic_save_pickle(dumpA, pathA, logger)

    dumpB = {"meta": {**meta_common, "tag":"B",
                      "init_seed": args.init_seedB if args.mode=="diff" else args.init_seed,
                      "train_seed": args.train_seed if args.mode=="diff" else args.train_seedB,
                      "time": timeB, "record_layers": layer_list,
                      "metrics_epochs": rec_epochs, "checkpoints": ckpt_records_B}}
    for ln in layer_list:
        dumpB[ln] = np.stack([t.numpy() for t in spinsB[ln]], axis=0)
    pathB = os.path.join(outdir, f"gmlp_mlm_spinB_D{args.d_model}_F{args.d_ffn}_L{args.num_blocks}_M{args.M}_seedB{args.init_seedB if args.mode=='diff' else args.init_seed}.pkl")
    atomic_save_pickle(dumpB, pathB, logger)

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
