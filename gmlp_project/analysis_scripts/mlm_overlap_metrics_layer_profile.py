#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Layer-wise overlap metrics (intra/inter) for specified epochs.
"""

import argparse
import csv
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from multiprocessing.pool import ThreadPool

from overlap_metrics import overlap_metrics


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Layer-wise overlap metrics for replica pairs")
    p.add_argument("--spin_file_a1", required=True)
    p.add_argument("--spin_file_b1", required=True)
    p.add_argument("--spin_file_a2", required=True)
    p.add_argument("--spin_file_b2", required=True)
    p.add_argument("--label1", default="set1")
    p.add_argument("--label2", default="set2")
    p.add_argument("--epochs", type=int, nargs="+", required=True)
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--flatten", action="store_true", default=True)
    p.add_argument("--patch-mean", action="store_true", default=False)
    p.add_argument("--center", action="store_true", default=False,
                   help="(deprecated) no-op; kept for backward compatibility")
    p.add_argument("--apply-bn", action="store_true", default=True)
    p.add_argument("--num-workers", type=int, default=1,
                   help="Number of parallel workers for epoch-level computation")
    p.add_argument("--q-block-size", type=int, default=0,
                   help="If >0, compute q2_* blockwise over feature dims (slower, lower memory).")
    p.add_argument("--output", default="./gmlp_qinv_out/manifold/overlap_layer_profile.png")
    p.add_argument("--output-csv", default=None)
    p.add_argument("--title", default=None)
    return p.parse_args()


def load_spin(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid pickle: {path}")
    return data

def _is_records_format(data: Dict[str, Any]) -> bool:
    return isinstance(data, dict) and isinstance(data.get("records"), list)


def get_epochs(data: Dict[str, Any], layer_key: str) -> List[int]:
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
    total = data[layer_key].shape[0]
    for key in ["saved_epochs", "epoch_ticks", "epochs_saved", "epochs", "time"]:
        arr = meta.get(key)
        if isinstance(arr, (list, tuple, np.ndarray)) and len(arr) == total:
            return [int(round(float(x))) for x in arr]
    return list(range(total))


def resolve_layers(data: Dict[str, Any], user_layers: Optional[Sequence[int]]) -> List[str]:
    if user_layers:
        return [f"gmlp_blocks.{i}.bn:post" for i in user_layers]
    pat = re.compile(r"gmlp_blocks\.(\d+)\.bn:post")
    if _is_records_format(data):
        records = data.get("records", [])
        if records and isinstance(records[0], dict):
            layer_map = records[0].get("layers", {})
            keys = list(layer_map.keys()) if isinstance(layer_map, dict) else []
        else:
            keys = []
    else:
        keys = list(data.keys())
    layers = [k for k in keys if pat.fullmatch(str(k)) and k != "out_bn:post"]
    layers.sort(key=lambda k: int(pat.fullmatch(k).group(1)))
    return layers

def intersect_layers(*layer_lists: List[str]) -> List[str]:
    if not layer_lists:
        return []
    shared = set(layer_lists[0])
    for lst in layer_lists[1:]:
        shared &= set(lst)
    pat = re.compile(r"gmlp_blocks\.(\d+)\.bn:post")
    return sorted(shared, key=lambda k: int(pat.fullmatch(k).group(1)))

def preprocess(arr: np.ndarray, flatten: bool, patch_mean: bool, *, allow_large_flatten: bool) -> np.ndarray:
    if arr.ndim == 3:
        if flatten:
            M, S, D = arr.shape
            feat = S * D
            qab_bytes = feat * feat * 8
            if not allow_large_flatten and qab_bytes > 512 * 1024 * 1024:
                print(f"[warn] flatten would create {feat} features (q_ab ~ {qab_bytes/1024/1024:.1f} MB); using patch-mean.")
                return arr.mean(axis=1)
            return arr.reshape(M, feat)
        if patch_mean:
            return arr.mean(axis=1)
        print("[warn] 3D input without flatten/patch-mean; using patch-mean.")
        return arr.mean(axis=1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unsupported array shape {arr.shape}")

def batch_norm_np(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    if x.ndim == 2:
        mean = x.mean(axis=0, keepdims=True)
        var = x.var(axis=0, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)
    if x.ndim == 3:
        M, S, D = x.shape
        x2 = x.reshape(M, S * D)
        mean = x2.mean(axis=0, keepdims=True)
        var = x2.var(axis=0, keepdims=True)
        x2 = (x2 - mean) / np.sqrt(var + eps)
        return x2.reshape(M, S, D)
    raise ValueError(f"Unsupported ndim for BN: {x.shape}")

def _records_epoch_map(data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for rec in data.get("records", []):
        if not isinstance(rec, dict):
            continue
        epoch = rec.get("epoch")
        layers = rec.get("layers")
        if epoch is None or not isinstance(layers, dict):
            continue
        try:
            epoch_i = int(epoch)
        except Exception:
            continue
        out[epoch_i] = layers
    return out


def extract_epoch(
    data: Dict[str, Any],
    layer_key: str,
    epoch: int,
    epochs_list: List[int],
) -> Tuple[Optional[np.ndarray], bool]:
    if _is_records_format(data):
        layer_map = data.get("_epoch_layers", {})
        layers = layer_map.get(epoch)
        if layers is None or layer_key not in layers:
            return None, False
        return np.asarray(layers[layer_key]), True
    idx = {lab: i for i, lab in enumerate(epochs_list)}
    if epoch not in idx:
        return None, False
    arr = np.asarray(data[layer_key][idx[epoch]])
    return arr, True


def align_samples(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = min(A.shape[0], B.shape[0])
    if A.shape[0] != M:
        A = A[:M]
    if B.shape[0] != M:
        B = B[:M]
    return A, B


def _ensure_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == np.float64:
        return x
    return x.astype(np.float32, copy=False)


def _center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def _accum_q2_blockwise(
    X1: np.ndarray,
    Y1: np.ndarray,
    X2: np.ndarray,
    Y2: np.ndarray,
    *,
    block_size: int,
) -> float:
    M, N = X1.shape
    bs = max(1, min(int(block_size), N))
    total = 0.0
    for i in range(0, N, bs):
        X1i = X1[:, i:i + bs]
        X2i = X2[:, i:i + bs]
        for j in range(0, N, bs):
            Y1j = Y1[:, j:j + bs]
            Y2j = Y2[:, j:j + bs]
            Q1 = (X1i.T @ Y1j) / float(M)
            Q2 = (X2i.T @ Y2j) / float(M)
            total += float((Q1 * Q2).sum())
    return total / float(N)


def overlap_metrics_inter_blockwise(
    Sa1: np.ndarray,
    Sb1: np.ndarray,
    Sa2: np.ndarray,
    Sb2: np.ndarray,
    *,
    block_size: int,
    eps: float = 1e-12,
) -> Dict[str, float]:
    Sa1 = _ensure_float(Sa1)
    Sb1 = _ensure_float(Sb1)
    Sa2 = _ensure_float(Sa2)
    Sb2 = _ensure_float(Sb2)
    if Sa1.shape != Sb1.shape or Sa2.shape != Sb2.shape or Sa1.shape != Sa2.shape:
        raise ValueError("Sa1/Sb1/Sa2/Sb2 must have matching shapes (M,N)")
    if Sa1.ndim != 2:
        raise ValueError("Blockwise q_inv expects 2D arrays (M,N)")
    q2_ab = _accum_q2_blockwise(Sa1, Sb1, Sa2, Sb2, block_size=block_size)
    q2_aa = _accum_q2_blockwise(Sa1, Sa1, Sa2, Sa2, block_size=block_size)
    q2_bb = _accum_q2_blockwise(Sb1, Sb1, Sb2, Sb2, block_size=block_size)
    denom = (max(q2_aa, 0.0) * max(q2_bb, 0.0)) ** 0.5 + eps
    return {
        "q_inv": q2_ab / denom,
        "q2_ab": q2_ab,
        "q2_aa": q2_aa,
        "q2_bb": q2_bb,
        "use_nm_applied": False,
    }

def overlap_metrics_intra_blockwise(
    Sa: np.ndarray,
    Sb: np.ndarray,
    *,
    block_size: int,
    use_nm: bool = True,
    eps: float = 1e-12,
) -> Dict[str, float]:
    Sa = _ensure_float(Sa)
    Sb = _ensure_float(Sb)
    if Sa.shape != Sb.shape:
        raise ValueError("Sa/Sb must have matching shapes (M,N)")
    if Sa.ndim != 2:
        raise ValueError("Blockwise q_inv expects 2D arrays (M,N)")
    M, N = Sa.shape
    q2_ab_raw = _accum_q2_blockwise(Sa, Sb, Sa, Sb, block_size=block_size)
    q2_aa_raw = _accum_q2_blockwise(Sa, Sa, Sa, Sa, block_size=block_size)
    q2_bb_raw = _accum_q2_blockwise(Sb, Sb, Sb, Sb, block_size=block_size)
    use_nm_applied = False
    if use_nm:
        corr = float(N) / float(M)
        q2_ab_nm = q2_ab_raw - corr
        q2_aa_nm = q2_aa_raw - corr
        q2_bb_nm = q2_bb_raw - corr
        if q2_ab_nm >= 0.0 and q2_aa_nm >= 0.0 and q2_bb_nm >= 0.0:
            q2_ab, q2_aa, q2_bb = q2_ab_nm, q2_aa_nm, q2_bb_nm
            use_nm_applied = True
        else:
            q2_ab, q2_aa, q2_bb = q2_ab_raw, q2_aa_raw, q2_bb_raw
    else:
        q2_ab, q2_aa, q2_bb = q2_ab_raw, q2_aa_raw, q2_bb_raw
    denom = (max(q2_aa, 0.0) * max(q2_bb, 0.0)) ** 0.5 + eps
    return {
        "q_inv": q2_ab / denom,
        "q2_ab": q2_ab,
        "q2_aa": q2_aa,
        "q2_bb": q2_bb,
        "use_nm_applied": use_nm_applied,
    }


def compute_profiles(
    layers: List[str],
    epochs: Sequence[int],
    dataA1: Dict[str, Any],
    dataB1: Dict[str, Any],
    dataA2: Dict[str, Any],
    dataB2: Dict[str, Any],
    *,
    flatten: bool,
    patch_mean: bool,
    apply_bn: bool,
    q_block_size: int,
    num_workers: int,
) -> Dict[int, Dict[int, Dict[str, Any]]]:
    profiles: Dict[int, Dict[int, Dict[str, Any]]] = {}
    if _is_records_format(dataA1):
        dataA1["_epoch_layers"] = _records_epoch_map(dataA1)
    if _is_records_format(dataB1):
        dataB1["_epoch_layers"] = _records_epoch_map(dataB1)
    if _is_records_format(dataA2):
        dataA2["_epoch_layers"] = _records_epoch_map(dataA2)
    if _is_records_format(dataB2):
        dataB2["_epoch_layers"] = _records_epoch_map(dataB2)
    shared_epochs1 = {layer: get_epochs(dataA1, layer) for layer in layers}
    shared_epochs2 = {layer: get_epochs(dataA2, layer) for layer in layers}
    def compute_epoch(epoch: int) -> Tuple[int, Dict[int, Dict[str, Any]]]:
        epoch_map: Dict[int, Dict[str, Any]] = {}
        for layer in layers:
            layer_idx = int(re.search(r"gmlp_blocks\.(\d+)", layer).group(1))
            Ae1, okA1 = extract_epoch(dataA1, layer, epoch, shared_epochs1[layer])
            Be1, okB1 = extract_epoch(dataB1, layer, epoch, shared_epochs1[layer])
            Ae2, okA2 = extract_epoch(dataA2, layer, epoch, shared_epochs2[layer])
            Be2, okB2 = extract_epoch(dataB2, layer, epoch, shared_epochs2[layer])
            if not (okA1 and okB1 and okA2 and okB2):
                print(f"[warn] epoch {epoch} missing for layer {layer}; skipping.")
                continue
            Ae1, Be1 = align_samples(Ae1, Be1)
            allow_large_flatten = q_block_size > 0
            Sa1 = preprocess(Ae1, flatten, patch_mean, allow_large_flatten=allow_large_flatten)
            Sb1 = preprocess(Be1, flatten, patch_mean, allow_large_flatten=allow_large_flatten)
            Ae2, Be2 = align_samples(Ae2, Be2)
            Sa2 = preprocess(Ae2, flatten, patch_mean, allow_large_flatten=allow_large_flatten)
            Sb2 = preprocess(Be2, flatten, patch_mean, allow_large_flatten=allow_large_flatten)
            if apply_bn:
                Sa1 = batch_norm_np(Sa1)
                Sb1 = batch_norm_np(Sb1)
                Sa2 = batch_norm_np(Sa2)
                Sb2 = batch_norm_np(Sb2)
            if q_block_size > 0:
                metrics_inter = overlap_metrics_inter_blockwise(
                    Sa1, Sb1, Sa2, Sb2, block_size=q_block_size
                )
            else:
                metrics_inter = overlap_metrics(
                    "inter", Sa1, Sb1, Sa2=Sa2, Sb2=Sb2, center=False, backend="numpy"
                )
            epoch_map[layer_idx] = {
                "q_inv": metrics_inter["q_inv"],
                "q2_ab": metrics_inter.get("q2_ab"),
                "q2_aa": metrics_inter.get("q2_aa"),
                "q2_bb": metrics_inter.get("q2_bb"),
                "use_nm_applied": bool(metrics_inter.get("use_nm_applied", False)),
            }
        return epoch, epoch_map

    if num_workers > 1 and len(epochs) > 1:
        with ThreadPool(processes=num_workers) as pool:
            results = pool.map(compute_epoch, epochs)
    else:
        results = [compute_epoch(epoch) for epoch in epochs]

    for epoch, epoch_map in results:
        if epoch_map:
            profiles[epoch] = epoch_map
    return profiles

def compute_intra_profiles(
    layers: List[str],
    epochs: Sequence[int],
    dataA: Dict[str, Any],
    dataB: Dict[str, Any],
    *,
    flatten: bool,
    patch_mean: bool,
    apply_bn: bool,
    q_block_size: int,
    num_workers: int,
) -> Dict[int, Dict[int, Dict[str, Any]]]:
    profiles: Dict[int, Dict[int, Dict[str, Any]]] = {}
    if _is_records_format(dataA):
        dataA["_epoch_layers"] = _records_epoch_map(dataA)
    if _is_records_format(dataB):
        dataB["_epoch_layers"] = _records_epoch_map(dataB)
    shared_epochs = {layer: get_epochs(dataA, layer) for layer in layers}
    def compute_epoch(epoch: int) -> Tuple[int, Dict[int, Dict[str, Any]]]:
        epoch_map: Dict[int, Dict[str, Any]] = {}
        for layer in layers:
            layer_idx = int(re.search(r"gmlp_blocks\.(\d+)", layer).group(1))
            Ae, okA = extract_epoch(dataA, layer, epoch, shared_epochs[layer])
            Be, okB = extract_epoch(dataB, layer, epoch, shared_epochs[layer])
            if not (okA and okB):
                print(f"[warn] epoch {epoch} missing for layer {layer}; skipping.")
                continue
            Ae, Be = align_samples(Ae, Be)
            allow_large_flatten = q_block_size > 0
            Sa = preprocess(Ae, flatten, patch_mean, allow_large_flatten=allow_large_flatten)
            Sb = preprocess(Be, flatten, patch_mean, allow_large_flatten=allow_large_flatten)
            if apply_bn:
                Sa = batch_norm_np(Sa)
                Sb = batch_norm_np(Sb)
            if q_block_size > 0:
                metrics_intra = overlap_metrics_intra_blockwise(
                    Sa, Sb, block_size=q_block_size
                )
            else:
                metrics_intra = overlap_metrics(
                    "intra", Sa, Sb, center=False, backend="numpy"
                )
            epoch_map[layer_idx] = {
                "q_inv": metrics_intra["q_inv"],
                "q2_ab": metrics_intra.get("q2_ab"),
                "q2_aa": metrics_intra.get("q2_aa"),
                "q2_bb": metrics_intra.get("q2_bb"),
                "use_nm_applied": bool(metrics_intra.get("use_nm_applied", False)),
            }
        return epoch, epoch_map

    if num_workers > 1 and len(epochs) > 1:
        with ThreadPool(processes=num_workers) as pool:
            results = pool.map(compute_epoch, epochs)
    else:
        results = [compute_epoch(epoch) for epoch in epochs]

    for epoch, epoch_map in results:
        if epoch_map:
            profiles[epoch] = epoch_map
    return profiles


def plot_profiles(
    profiles: Dict[int, Dict[int, Dict[str, Any]]],
    epochs: Sequence[int],
    output: Path,
    *,
    title: Optional[str],
    label1: str,
    label2: str,
):
    layer_indices = sorted({idx for epoch_map in profiles.values() for idx in epoch_map.keys()})
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd"]
    fig, ax = plt.subplots(figsize=(9, 5))
    epoch_colors = {}
    for epi_idx, epoch in enumerate(epochs):
        color = colors[epi_idx % len(colors)]
        epoch_colors[epoch] = color
        epoch_map = profiles.get(epoch)
        if epoch_map is None:
            continue
        y = [epoch_map.get(idx, {}).get("q_inv", np.nan) for idx in layer_indices]
        if all(np.isnan(y)):
            continue
        ax.plot(
            layer_indices,
            y,
            color=color,
            linestyle="solid",
            linewidth=2.0,
            marker="o",
            label=f"{label1} vs {label2} epoch {epoch}",
        )
    ax.set_xlabel("Layer index L")
    ax.set_ylabel("q_inv (overlap)")
    ax.set_xticks(layer_indices)
    ax.set_ylim(0.98, 1.0)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_title(title or "Inter-dataset q_inv vs layer")
    color_handles = [
        Line2D([0], [0], color=color, linestyle="solid", linewidth=2.0, marker="o", label=f"epoch {epoch}")
        for epoch, color in epoch_colors.items()
    ]
    if color_handles:
        ax.legend(color_handles, [h.get_label() for h in color_handles], title="Epoch", loc="lower left", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> int:
    args = build_args()
    dataA1 = load_spin(args.spin_file_a1)
    dataB1 = load_spin(args.spin_file_b1)
    dataA2 = load_spin(args.spin_file_a2)
    dataB2 = load_spin(args.spin_file_b2)
    if args.layers:
        layer_keys = resolve_layers(dataA1, args.layers)
    else:
        layers1 = resolve_layers(dataA1, None)
        layers2 = resolve_layers(dataB1, None)
        layers3 = resolve_layers(dataA2, None)
        layers4 = resolve_layers(dataB2, None)
        layer_keys = intersect_layers(layers1, layers2, layers3, layers4)
        if not layer_keys:
            raise ValueError("No gmlp_blocks.*.bn:post layers found in all spin files.")
    profiles = compute_profiles(
        layer_keys,
        args.epochs,
        dataA1,
        dataB1,
        dataA2,
        dataB2,
        flatten=args.flatten,
        patch_mean=args.patch_mean,
        apply_bn=args.apply_bn,
        q_block_size=args.q_block_size,
        num_workers=args.num_workers,
    )
    intra_profiles_1 = compute_intra_profiles(
        layer_keys,
        args.epochs,
        dataA1,
        dataB1,
        flatten=args.flatten,
        patch_mean=args.patch_mean,
        apply_bn=args.apply_bn,
        q_block_size=args.q_block_size,
        num_workers=args.num_workers,
    )
    intra_profiles_2 = compute_intra_profiles(
        layer_keys,
        args.epochs,
        dataA2,
        dataB2,
        flatten=args.flatten,
        patch_mean=args.patch_mean,
        apply_bn=args.apply_bn,
        q_block_size=args.q_block_size,
        num_workers=args.num_workers,
    )
    output = Path(args.output).resolve()
    plot_profiles(profiles, args.epochs, output, title=args.title, label1=args.label1, label2=args.label2)
    if args.output_csv:
        csv_path = Path(args.output_csv).resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        def _append_rows(mode: str, set_label: str, prof: Dict[int, Dict[int, Dict[str, Any]]]):
            for epoch, layer_map in sorted(prof.items()):
                for layer_idx, entry in sorted(layer_map.items()):
                    q2_ab = float(entry.get("q2_ab")) if entry.get("q2_ab") is not None else float("nan")
                    q2_aa = float(entry.get("q2_aa")) if entry.get("q2_aa") is not None else float("nan")
                    q2_bb = float(entry.get("q2_bb")) if entry.get("q2_bb") is not None else float("nan")
                    rows.append(
                        {
                            "mode": mode,
                            "set": set_label,
                            "epoch": int(epoch),
                            "layer": int(layer_idx),
                            "q_inv": float(entry.get("q_inv")),
                            "q2_ab": q2_ab,
                            "q2_aa": q2_aa,
                            "q2_bb": q2_bb,
                            "q_ab": q2_ab,
                            "q_aa": q2_aa,
                            "q_bb": q2_bb,
                            "use_nm_applied": bool(entry.get("use_nm_applied", False)),
                        }
                    )
        _append_rows("inter", f"{args.label1}_vs_{args.label2}", profiles)
        _append_rows("intra", args.label1, intra_profiles_1)
        _append_rows("intra", args.label2, intra_profiles_2)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "mode",
                    "set",
                    "epoch",
                    "layer",
                    "q_inv",
                    "q2_ab",
                    "q2_aa",
                    "q2_bb",
                    "q_ab",
                    "q_aa",
                    "q_bb",
                    "use_nm_applied",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        q_only_rows = [
            {
                "mode": r["mode"],
                "set": r["set"],
                "epoch": r["epoch"],
                "layer": r["layer"],
                "q_ab": r["q_ab"],
                "q_aa": r["q_aa"],
                "q_bb": r["q_bb"],
            }
            for r in rows
        ]
        q_only_path = csv_path.with_name(f"{csv_path.stem}_q_only{csv_path.suffix}")
        with open(q_only_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["mode", "set", "epoch", "layer", "q_ab", "q_aa", "q_bb"],
            )
            writer.writeheader()
            writer.writerows(q_only_rows)
    print(json.dumps({"saved": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
