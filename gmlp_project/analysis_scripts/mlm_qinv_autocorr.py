#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import re
import shutil

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


def build_args():
    p = argparse.ArgumentParser("q_inv autocorrelation q(t1, t2) for a single spin set")
    p.add_argument("--spin-file", type=str, required=True, help="Spin pickle (single model)")
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--layer-index", type=int, default=0,
                   help="Target index for blocks.[i].bn:post (ignored if --all-layers)")
    p.add_argument("--all-layers", action="store_true", default=True,
                   help="Process and plot all layers")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="./gmlp_qinv_out/autocorr")
    p.add_argument("--block-size", type=int, default=2000,
                   help="Block size for feature-dimension chunking to avoid OOM")
    # Options
    p.add_argument("--apply-bn", default=True, action="store_true")
    p.add_argument("--use-nm-correction", default=True, action="store_true")
    p.add_argument("--combine-patches", default=False, action="store_true")
    p.add_argument("--patch-mean", default=False, action="store_true")
    p.add_argument("--flatten-patches-as-features", default=True, action="store_true")
    p.add_argument("--skip-epoch0", default=True, action="store_true")
    p.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs (default: -1 for all CPUs)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                   help="Compute backend for q_inv (auto=use cuda if available)")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args, _unknown = p.parse_known_args()
    return args


def setup_logger(level="INFO"):
    logging.basicConfig(level=getattr(logging, level),
                        format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger("qinv_autocorr")


def batch_norm_np(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    if x.ndim == 2:
        mean = x.mean(axis=0, keepdims=True)
        var = x.var(axis=0, keepdims=True)
        return (x - mean) / np.sqrt(var + eps + 1e-12)
    if x.ndim == 3:
        M, S, D = x.shape
        x2 = x.reshape(M, S * D)
        mean = x2.mean(axis=0, keepdims=True)
        var = x2.var(axis=0, keepdims=True)
        x2 = (x2 - mean) / np.sqrt(var + eps + 1e-12)
        return x2.reshape(M, S, D)
    raise ValueError(f"Unsupported ndim for BN: {x.shape}")


def calc_q2_blockwise(A: np.ndarray, B: np.ndarray, chunk_size: int = 2048) -> float:
    """
    Computes (sum((A.T @ B)**2)) / (M^2 * N) using block-wise processing.
    Casts chunks to float64 on the fly to avoid OOM with large N.
    """
    M, N = A.shape
    total_sq = 0.0
    is_symmetric = (A is B)
    
    # Outer loop over chunks of columns of A
    for i in range(0, N, chunk_size):
        i_end = min(i + chunk_size, N)
        # Cast only the current chunk to float64
        A_chunk = A[:, i:i_end].astype(np.float64)
        
        # Inner loop over chunks of columns of B
        start_j = i if is_symmetric else 0
        
        for j in range(start_j, N, chunk_size):
            j_end = min(j + chunk_size, N)
            
            if is_symmetric and i == j:
                # Diagonal block
                P = A_chunk.T @ A_chunk
                total_sq += float((P ** 2).sum())
                del P
            elif is_symmetric:
                # Off-diagonal block (upper triangle)
                B_chunk = A[:, j:j_end].astype(np.float64)
                P = A_chunk.T @ B_chunk
                total_sq += 2.0 * float((P ** 2).sum())
                del P, B_chunk
            else:
                # Asymmetric case
                B_chunk = B[:, j:j_end].astype(np.float64)
                P = A_chunk.T @ B_chunk
                total_sq += float((P ** 2).sum())
                del P, B_chunk
        
        del A_chunk

    return float(total_sq / (float(M)**2 * float(N)))


def _calc_q2_blockwise_torch(A: torch.Tensor, B: torch.Tensor, chunk_size: int = 2048) -> float:
    M, N = A.shape
    total_sq = 0.0
    is_symmetric = A.data_ptr() == B.data_ptr()
    for i in range(0, N, chunk_size):
        i_end = min(i + chunk_size, N)
        A_chunk = A[:, i:i_end]
        start_j = i if is_symmetric else 0
        for j in range(start_j, N, chunk_size):
            j_end = min(j + chunk_size, N)
            if is_symmetric and i == j:
                P = A_chunk.T @ A_chunk
                total_sq += float((P ** 2).sum().item())
            elif is_symmetric:
                B_chunk = A[:, j:j_end]
                P = A_chunk.T @ B_chunk
                total_sq += 2.0 * float((P ** 2).sum().item())
            else:
                B_chunk = B[:, j:j_end]
                P = A_chunk.T @ B_chunk
                total_sq += float((P ** 2).sum().item())
    return float(total_sq / (float(M)**2 * float(N)))


def calc_qinv_epoch(A: np.ndarray, B: np.ndarray, *, use_nm: bool, combine_patches: bool,
                    block_size: int = 2048, device: str = "cpu") -> Tuple[float, float, float, float, bool]:
    # Do NOT cast to float64 globally to save memory.
    A = np.asarray(A)
    B = np.asarray(B)
    
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: A{A.shape} vs B{B.shape}")
    if A.ndim == 2:
        A2, B2 = A, B
    elif A.ndim == 3:
        if combine_patches:
            M, S, D = A.shape
            A2 = A.reshape(M * S, D)
            B2 = B.reshape(M * S, D)
        else:
            # Recursive call for uncombined patches
            qinv_list, q2ab_list, q2aa_list, q2bb_list = [], [], [], []
            nm_flags = []
            for s in range(A.shape[1]):
                qinv, q2ab, q2aa, q2bb, use_nm_applied = calc_qinv_epoch(
                    A[:, s, :], B[:, s, :], use_nm=use_nm, combine_patches=False, block_size=block_size
                )
                qinv_list.append(qinv)
                q2ab_list.append(q2ab)
                q2aa_list.append(q2aa)
                q2bb_list.append(q2bb)
                nm_flags.append(bool(use_nm_applied))
            use_nm_applied_all = all(nm_flags) if nm_flags else False
            return (float(np.mean(qinv_list)), float(np.mean(q2ab_list)),
                    float(np.mean(q2aa_list)), float(np.mean(q2bb_list)), use_nm_applied_all)
    else:
        raise ValueError(f"Unsupported ndim for q_inv: {A.shape}")

    M, N = A2.shape
    
    # Use blockwise computation automatically if needed
    use_chunking = False
    if N > 2048 or (block_size > 0 and block_size < N):
        use_chunking = True
        
    use_cuda = (device == "cuda")
    if use_cuda:
        A2_t = torch.as_tensor(A2, device="cuda", dtype=torch.float32)
        B2_t = torch.as_tensor(B2, device="cuda", dtype=torch.float32)
        if use_chunking:
            bs = block_size if block_size > 0 else 2048
            q2_ab_raw = _calc_q2_blockwise_torch(A2_t, B2_t, chunk_size=bs)
            q2_aa_raw = _calc_q2_blockwise_torch(A2_t, A2_t, chunk_size=bs)
            q2_bb_raw = _calc_q2_blockwise_torch(B2_t, B2_t, chunk_size=bs)
        else:
            q_ab = (A2_t.T @ B2_t) / M
            qaa = (A2_t.T @ A2_t) / M
            qbb = (B2_t.T @ B2_t) / M
            q2_ab_raw = float((q_ab ** 2).sum().item() / N)
            q2_aa_raw = float((qaa ** 2).sum().item() / N)
            q2_bb_raw = float((qbb ** 2).sum().item() / N)
    else:
        if use_chunking:
            bs = block_size if block_size > 0 else 2048
            q2_ab_raw = calc_q2_blockwise(A2, B2, chunk_size=bs)
            q2_aa_raw = calc_q2_blockwise(A2, A2, chunk_size=bs)
            q2_bb_raw = calc_q2_blockwise(B2, B2, chunk_size=bs)
        else:
            # Small enough to do directly
            A2_f = A2.astype(np.float64)
            B2_f = B2.astype(np.float64)
            q_ab = (A2_f.T @ B2_f) / M
            qaa = (A2_f.T @ A2_f) / M
            qbb = (B2_f.T @ B2_f) / M
            q2_ab_raw = float((q_ab ** 2).sum() / N)
            q2_aa_raw = float((qaa ** 2).sum() / N)
            q2_bb_raw = float((qbb ** 2).sum() / N)
    
    use_nm_applied = False
    if use_nm:
        q2_ab_nm = q2_ab_raw - (N / M)
        q2_aa_nm = q2_aa_raw - (N / M)
        q2_bb_nm = q2_bb_raw - (N / M)
        if q2_ab_nm < 0.0 or q2_aa_nm < 0.0 or q2_bb_nm < 0.0:
            q2_ab, q2_aa, q2_bb = q2_ab_raw, q2_aa_raw, q2_bb_raw
        else:
            q2_ab, q2_aa, q2_bb = q2_ab_nm, q2_aa_nm, q2_bb_nm
            use_nm_applied = True
    else:
        q2_ab, q2_aa, q2_bb = q2_ab_raw, q2_aa_raw, q2_bb_raw

    q2_aa = max(q2_aa, 0.0)
    q2_bb = max(q2_bb, 0.0)
    denom_val = np.sqrt(q2_aa) * np.sqrt(q2_bb)
    if not np.isfinite(denom_val) or denom_val < 1e-12:
        denom_val = 1e-12
    q_inv = q2_ab / denom_val
    return float(q_inv), float(q2_ab), float(q2_aa), float(q2_bb), use_nm_applied


def load_spin_pkl(path: str) -> Dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid pkl structure: {path}")
    return data


def _is_records_format(data: Dict) -> bool:
    return isinstance(data, dict) and isinstance(data.get("records"), list)


def _coerce_int_seq(seq: Sequence) -> List[int]:
    out = []
    for v in seq:
        if isinstance(v, (int, np.integer)):
            out.append(int(v))
        else:
            out.append(int(round(float(v))))
    return out


def get_epoch_labels_for(data: Dict, example_layer: str) -> List[int]:
    if _is_records_format(data):
        epochs = []
        for rec in data.get("records", []):
            if isinstance(rec, dict) and "epoch" in rec:
                try:
                    epochs.append(int(rec["epoch"]))
                except Exception:
                    continue
        return epochs
    meta = data.get("meta", {})
    E = data[example_layer].shape[0]
    for key in ["saved_epochs", "epoch_ticks", "epochs_saved"]:
        if key in meta and isinstance(meta[key], (list, tuple, np.ndarray)):
            arr = list(meta[key])
            if len(arr) == E:
                return _coerce_int_seq(arr)
    if "time" in meta and isinstance(meta["time"], (list, tuple, np.ndarray)):
        arr = list(meta["time"])
        if len(arr) == E:
            try:
                return _coerce_int_seq(arr)
            except Exception:
                pass
    return list(range(E))


def layer_key_for_index(idx: int, prefix: str = "gmlp_blocks") -> str:
    return f"{prefix}.{idx}.bn:post"


def layer_idx_from_key(layer_key: str) -> int:
    m = re.search(r"(?:gmlp_blocks|mlp_blocks)\.(\d+)\.bn:post", layer_key)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0


def _layer_keys_from_data(data: Dict) -> List[str]:
    if _is_records_format(data):
        records = data.get("records", [])
        if not records:
            return []
        layers = records[0].get("layers", {})
        keys = list(layers.keys()) if isinstance(layers, dict) else []
    else:
        keys = list(data.keys())
    out = []
    for k in keys:
        if k == "out_bn:post":
            continue
        if re.fullmatch(r"(?:gmlp_blocks|mlp_blocks)\.\d+\.bn:post", str(k)):
            out.append(k)
    return sorted(out, key=layer_idx_from_key)


def _detect_layer_prefix(data: Dict) -> str:
    keys = _layer_keys_from_data(data)
    for k in keys:
        if str(k).startswith("gmlp_blocks."):
            return "gmlp_blocks"
        if str(k).startswith("mlp_blocks."):
            return "mlp_blocks"
    return "gmlp_blocks"


def _epoch_layer_map(data: Dict, layer_key: str) -> Dict[int, np.ndarray]:
    if _is_records_format(data):
        out = {}
        for rec in data.get("records", []):
            if not isinstance(rec, dict):
                continue
            epoch = rec.get("epoch")
            layers = rec.get("layers", {})
            if epoch is None or not isinstance(layers, dict):
                continue
            if layer_key not in layers:
                continue
            try:
                epoch_i = int(epoch)
            except Exception:
                continue
            out[epoch_i] = layers[layer_key]
        return out
    if layer_key not in data:
        return {}
    labels = get_epoch_labels_for(data, layer_key)
    arr_all = data[layer_key]
    return {int(lab): arr_all[i] for i, lab in enumerate(labels)}


def _ensure_spin_data(spin_or_path) -> Dict:
    if isinstance(spin_or_path, (str, Path)):
        return load_spin_pkl(str(spin_or_path))
    return spin_or_path


def get_processed_data_for_layer(
    spin_data: Dict, layer_key: str, *,
    apply_bn: bool, combine_patches: bool, patch_mean: bool, flatten: bool
) -> Dict[int, np.ndarray]:
    """Pre-process all epochs for a layer to avoid repeating work in the loop."""
    raw_map = _epoch_layer_map(spin_data, layer_key)
    processed = {}
    for ep, arr in raw_map.items():
        A = np.asarray(arr)
        if flatten and A.ndim == 3:
            Mx, Sx, Dx = A.shape
            feat = Sx * Dx
            A = A.reshape(Mx, feat) # view
        if (not flatten) and patch_mean and A.ndim == 3:
            A = A.mean(axis=1) # copy
        if apply_bn:
            A = batch_norm_np(A) # copy
        processed[ep] = A
    return processed


def compute_all_pairs_autocorr_for_layer(
    spin_data: Dict, layer_key: str, *,
    apply_bn: bool, use_nm: bool, combine_patches: bool,
    patch_mean: bool, flatten: bool, skip_epoch0: bool,
    block_size: int, n_jobs: int, device: str, logger: logging.Logger
) -> pd.DataFrame:
    
    emap = get_processed_data_for_layer(
        spin_data, layer_key, 
        apply_bn=apply_bn, combine_patches=combine_patches, 
        patch_mean=patch_mean, flatten=flatten
    )
    epochs = sorted(emap.keys())
    if len(epochs) < 2:
        return pd.DataFrame()

    from joblib import Parallel, delayed

    def _process_t1(i: int, t1: int):
        local_rows = []
        if skip_epoch0 and t1 == 0:
            return local_rows
        
        # A is fixed for this t1
        A = emap[t1]
        
        # Inner loop t2 > t1
        for t2 in epochs[i+1:]:
            B = emap[t2]
            
            # shape check
            Mm = min(A.shape[0], B.shape[0])
            Ae = A[:Mm]
            Be = B[:Mm]
            
            q_inv, q2_ab, q2_aa, q2_bb, use_nm_applied = calc_qinv_epoch(
                Ae, Be, use_nm=use_nm, combine_patches=combine_patches, block_size=block_size, device=device
            )
            
            local_rows.append({
                "q_inv_self_epoch_prev": t1,
                "q_inv_self_epoch": t2,
                "dt": t2 - t1,
                "q_inv": q_inv,
                "use_nm_applied": use_nm_applied,
                "M": Mm
            })
        return local_rows

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_t1)(i, t1) for i, t1 in enumerate(epochs)
    )
    
    # Flatten results
    rows = [r for sublist in results for r in sublist]
    return pd.DataFrame(rows)


def default_label_from_path(path: str) -> str:
    p = Path(path)
    return p.parent.name or p.stem


def plot_autocorr_t1(fig_out: Path, df: pd.DataFrame, t1: int, label: str, title: Optional[str]):
    """Plot q_inv vs t2 for a specific t1."""
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = sorted(df["layer"].unique(), key=layer_idx_from_key)
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10)) 
    
    for i, lk in enumerate(layers):
        sub = df[df["layer"] == lk].sort_values("q_inv_self_epoch")
        if sub.empty: continue
        
        x = sub["q_inv_self_epoch"].to_numpy()
        y = sub["q_inv"].to_numpy()
        
        c = colors[i % 10]
        ax.plot(x, y, marker="o", markersize=4, label=f"L={layer_idx_from_key(lk)}", color=c)

    ax.set_xlabel("Epoch t2 (recorded)")
    ax.set_ylabel(f"Autocorrelation q_inv(t1={t1}, t2)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    if not sub.empty and sub["q_inv_self_epoch"].max() > 50:
        ax.set_xscale("log")
    
    ax.set_title(title or f"q_inv autocorr (t1={t1}) - {label}")
    ax.legend(loc="best", ncol=min(3, max(1, len(layers))))
    
    fig.tight_layout()
    fig.savefig(fig_out, dpi=150)
    plt.close(fig)


def main() -> int:
    args = build_args()
    logger = setup_logger(args.log_level)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    label = args.label or default_label_from_path(args.spin_file)
    logger.info(f"Loading data: {args.spin_file}")
    
    try:
        data = _ensure_spin_data(args.spin_file)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    prefix = _detect_layer_prefix(data)
    layer_keys = _layer_keys_from_data(data)
    
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA requested but not available. Aborting.")
        return 2
    logger.info(f"Compute device: {device}")

    if args.all_layers:
        target_layers = layer_keys
    else:
        lk = layer_key_for_index(args.layer_index, prefix=prefix)
        if lk in layer_keys:
            target_layers = [lk]
        else:
            logger.error(f"Layer {lk} not found.")
            return 1

    # Keep track of all dfs
    all_rows = []
    
    # Process layers one by one (to save memory if not all at once?)
    # But we want to aggregate by t1 later. It's better to process all layers then concat.
    
    for i, lk in enumerate(target_layers):
        logger.info(f"Computing {lk} ({i+1}/{len(target_layers)})")
        df_layer = compute_all_pairs_autocorr_for_layer(
            data, lk,
            apply_bn=args.apply_bn, use_nm=args.use_nm_correction,
            combine_patches=args.combine_patches, patch_mean=args.patch_mean,
            flatten=args.flatten_patches_as_features, skip_epoch0=args.skip_epoch0,
            block_size=args.block_size, n_jobs=args.n_jobs, device=device, logger=logger
        )
        if not df_layer.empty:
            df_layer["layer"] = lk
            all_rows.append(df_layer)
    
    if not all_rows:
        logger.warning("No results to plot.")
        return 0
        
    df_all = pd.concat(all_rows, ignore_index=True)
    
    # Save raw data
    csv_path = out_dir / f"qinv_autocorr_{label}_all_pairs.csv"
    df_all.to_csv(csv_path, index=False)
    logger.info(f"Saved (all pairs): {csv_path}")
    
    # Generate Plots per t1
    plots_dir = out_dir / f"plots_{label}"
    if plots_dir.exists():
        shutil.rmtree(plots_dir)
    plots_dir.mkdir()
    
    t1_values = sorted(df_all["q_inv_self_epoch_prev"].unique())
    logger.info(f"Generating plots for {len(t1_values)} start epochs (t1)...")
    
    for t1 in t1_values:
        sub = df_all[df_all["q_inv_self_epoch_prev"] == t1]
        fig_out = plots_dir / f"autocorr_t1_{int(t1):06d}.png"
        plot_autocorr_t1(fig_out, sub, int(t1), label, args.title)
        
    logger.info(f"Saved {len(t1_values)} plots to {plots_dir}")

    # Also save joblib for portability
    joblib.dump({
        "config": vars(args),
        "layers": target_layers,
        "label": label,
        "table": df_all,
    }, out_dir / f"qinv_autocorr_{label}_all_pairs.joblib")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
