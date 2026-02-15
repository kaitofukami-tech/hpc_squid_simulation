#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot q_inv as a function of layer index for selected epochs.
"""

import argparse
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("q_inv vs layer index for selected epochs (supports up to two sets)")
    p.add_argument("--spin_file_a", type=str, required=True, help="Set1 Replica A spin pickle")
    p.add_argument("--spin_file_b", type=str, required=True, help="Set1 Replica B spin pickle")
    p.add_argument("--label1", type=str, default="set1", help="Legend label for set1")
    p.add_argument("--spin_file_a2", type=str, help="Optional Set2 Replica A spin pickle")
    p.add_argument("--spin_file_b2", type=str, help="Optional Set2 Replica B spin pickle")
    p.add_argument("--label2", type=str, default="set2", help="Legend label for set2")
    p.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        required=True,
        help="Epoch numbers to plot as separate curves (e.g., 1 3 100)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="./gmlp_qinv_out/layer_profile/qinv_layer_profile.png",
        help="Output PNG path",
    )
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--apply-bn", default=True, action="store_true")
    p.add_argument("--use-nm-correction", default=False, action="store_true")
    p.add_argument("--combine-patches", default=False, action="store_true")
    p.add_argument("--patch-mean", default=False, action="store_true")
    p.add_argument("--flatten-patches-as-features", default=True, action="store_true")
    p.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of layer indices (default: detect from pickle, fallback 0-9)",
    )
    p.add_argument("--point-style", type=str, default="o", help="Matplotlib marker for the curves")
    return p.parse_args()


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger("qinv_layer")


def load_spin_pkl(path: Path) -> Dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected data structure in {path}")
    return data


def get_epoch_labels(data: Dict, layer_key: str) -> List[int]:
    meta = data.get("meta", {})
    total = data[layer_key].shape[0]
    for key in ["saved_epochs", "epoch_ticks", "epochs_saved", "time"]:
        arr = meta.get(key)
        if isinstance(arr, (list, tuple, np.ndarray)) and len(arr) == total:
            return [int(round(float(v))) for v in arr]
    return list(range(total))


def layer_keys_from_data(data: Dict) -> List[str]:
    pat = re.compile(r"gmlp_blocks\.(\d+)\.bn:post")
    keys = []
    for k in data.keys():
        if pat.fullmatch(k):
            keys.append(k)
    keys.sort(key=lambda k: int(pat.fullmatch(k).group(1)) if pat.fullmatch(k) else 0)
    return keys


def batch_norm_np(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    if x.ndim == 2:
        mean = x.mean(axis=0, keepdims=True)
        var = x.var(axis=0, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)
    elif x.ndim == 3:
        M, S, D = x.shape
        x2 = x.reshape(M, S * D)
        mean = x2.mean(axis=0, keepdims=True)
        var = x2.var(axis=0, keepdims=True)
        x2 = (x2 - mean) / np.sqrt(var + eps)
        return x2.reshape(M, S, D)
    raise ValueError(f"Unsupported ndim {x.shape}")


def calc_qinv_epoch(A: np.ndarray, B: np.ndarray, *, use_nm: bool, combine_patches: bool) -> float:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
    if A.ndim == 3:
        if combine_patches:
            M, S, D = A.shape
            A = A.reshape(M * S, D)
            B = B.reshape(M * S, D)
        else:
            q_list = []
            for s in range(A.shape[1]):
                q_list.append(calc_qinv_epoch(A[:, s, :], B[:, s, :], use_nm=use_nm, combine_patches=False))
            return float(np.mean(q_list))
    M, N = A.shape
    q_ab = (A.T @ B) / M
    qaa = (A.T @ A) / M
    qbb = (B.T @ B) / M
    q2_ab = float((q_ab ** 2).sum() / N)
    q2_aa = float((qaa ** 2).sum() / N)
    q2_bb = float((qbb ** 2).sum() / N)
    if use_nm:
        adj = (N / M)
        q2_ab_nm = q2_ab - adj
        q2_aa_nm = q2_aa - adj
        q2_bb_nm = q2_bb - adj
        if q2_ab_nm >= 0.0 and q2_aa_nm >= 0.0 and q2_bb_nm >= 0.0:
            q2_ab, q2_aa, q2_bb = q2_ab_nm, q2_aa_nm, q2_bb_nm
    q2_aa = max(q2_aa, 0.0)
    q2_bb = max(q2_bb, 0.0)
    denom = np.sqrt(q2_aa) * np.sqrt(q2_bb)
    if denom < 1e-12 or not np.isfinite(denom):
        denom = 1e-12
    return float(q2_ab / denom)


def preprocess_epoch_array(arr: np.ndarray, *, flatten: bool, patch_mean: bool) -> np.ndarray:
    if arr.ndim == 3 and flatten:
        M, S, D = arr.shape
        return arr.reshape(M, S * D)
    if arr.ndim == 3 and patch_mean:
        return arr.mean(axis=1)
    return arr


def compute_layer_profiles(
    dataA: Dict,
    dataB: Dict,
    layer_keys: Sequence[str],
    epochs: Sequence[int],
    *,
    apply_bn: bool,
    use_nm: bool,
    combine_patches: bool,
    patch_mean: bool,
    flatten: bool,
    logger: logging.Logger,
) -> Dict[int, Dict[int, float]]:
    profiles: Dict[int, Dict[int, float]] = {}
    for layer_key in layer_keys:
        idx_match = re.search(r"gmlp_blocks\.(\d+)\.bn:post", layer_key)
        if not idx_match:
            continue
        layer_idx = int(idx_match.group(1))
        arrA = np.array(dataA[layer_key], copy=False)
        arrB = np.array(dataB[layer_key], copy=False)
        labelsA = get_epoch_labels(dataA, layer_key)
        labelsB = get_epoch_labels(dataB, layer_key)
        label_set = sorted(set(labelsA) & set(labelsB))
        idxA = {lab: i for i, lab in enumerate(labelsA)}
        idxB = {lab: i for i, lab in enumerate(labelsB)}
        layer_map: Dict[int, float] = {}
        for epoch in epochs:
            if epoch not in label_set:
                logger.warning("Epoch %s missing for layer %s; skipping.", epoch, layer_key)
                continue
            Ae = arrA[idxA[epoch]]
            Be = arrB[idxB[epoch]]
            M = min(Ae.shape[0], Be.shape[0])
            if Ae.shape[0] != M:
                Ae = Ae[:M]
            if Be.shape[0] != M:
                Be = Be[:M]
            Ae = preprocess_epoch_array(Ae, flatten=flatten, patch_mean=patch_mean)
            Be = preprocess_epoch_array(Be, flatten=flatten, patch_mean=patch_mean)
            if apply_bn:
                Ae = batch_norm_np(Ae)
                Be = batch_norm_np(Be)
            q_inv = calc_qinv_epoch(Ae, Be, use_nm=use_nm, combine_patches=combine_patches)
            layer_map[epoch] = q_inv
        profiles[layer_idx] = layer_map
    return profiles


def plot_profiles(
    out_path: Path,
    sets: Sequence[tuple],
    epochs: Sequence[int],
    *,
    title: Optional[str],
    marker: str,
):
    if not sets:
        raise ValueError("No q_inv data computed for any layer.")
    # union of layer indices from first set (assume consistent)
    layer_set = set()
    for profiles, _, _ in sets:
        layer_set.update(profiles.keys())
    layer_indices = sorted(layer_set)
    fig, ax = plt.subplots(figsize=(9, 5))
    base_colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd"]
    colors = [base_colors[i] if i < len(base_colors) else plt.cm.tab10(i % 10) for i in range(len(epochs))]
    plotted = []
    for epoch_idx, epoch in enumerate(epochs):
        color = colors[epoch_idx]
        for profiles, label, style in sets:
            y = [profiles.get(layer, {}).get(epoch, np.nan) for layer in layer_indices]
            if all(np.isnan(val) for val in y):
                continue
            ax.plot(
                layer_indices,
                y,
                marker=marker,
                linewidth=2.0,
                color=color,
                linestyle=style,
            )
            plotted.append((epoch, label))
    ax.set_xlabel("Layer index L")
    ax.set_ylabel("q_inv")
    ax.set_xticks(layer_indices)
    ax.set_ylim(0.8, 1.0)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D

    epoch_handles = [
        Line2D([0], [0], color=colors[idx], linestyle="solid", linewidth=2.0, label=f"epoch {epoch}")
        for idx, epoch in enumerate(epochs)
    ]
    seen_labels = {}
    for _, label, style in sets:
        if label not in seen_labels:
            seen_labels[label] = style
    set_handles = [
        Line2D([0], [0], color="black", linestyle=style, linewidth=2.0, label=label)
        for label, style in seen_labels.items()
    ]
    leg1 = ax.legend(epoch_handles, [h.get_label() for h in epoch_handles], title="Epoch", loc="lower left")
    ax.add_artist(leg1)
    ax.legend(set_handles, [h.get_label() for h in set_handles], title="Set", loc="lower right")
    ax.set_title(title or "q_inv vs layer index")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(json.dumps({"saved": str(out_path)}, ensure_ascii=False))


def main() -> int:
    args = build_args()
    logger = setup_logger(args.log_level)
    spinA = Path(args.spin_file_a).expanduser().resolve()
    spinB = Path(args.spin_file_b).expanduser().resolve()
    dataA = load_spin_pkl(spinA)
    dataB = load_spin_pkl(spinB)
    if args.layers:
        layer_keys = [f"gmlp_blocks.{i}.bn:post" for i in args.layers]
    else:
        layer_keys = layer_keys_from_data(dataA)
        if not layer_keys:
            layer_keys = [f"gmlp_blocks.{i}.bn:post" for i in range(10)]
    layer_ids = []
    for k in layer_keys:
        m = re.search(r"gmlp_blocks\.(\d+)\.bn:post", k)
        layer_ids.append(m.group(1) if m else k)
    logger.info("Using layers: %s", layer_ids)
    epochs = sorted(set(int(e) for e in args.epochs))
    profiles1 = compute_layer_profiles(
        dataA,
        dataB,
        layer_keys,
        epochs,
        apply_bn=args.apply_bn,
        use_nm=args.use_nm_correction,
        combine_patches=args.combine_patches,
        patch_mean=args.patch_mean,
        flatten=args.flatten_patches_as_features,
        logger=logger,
    )
    profiles_sets = [(profiles1, args.label1, "solid")]
    if args.spin_file_a2 and args.spin_file_b2:
        dataA2 = load_spin_pkl(Path(args.spin_file_a2).expanduser().resolve())
        dataB2 = load_spin_pkl(Path(args.spin_file_b2).expanduser().resolve())
        profiles2 = compute_layer_profiles(
            dataA2,
            dataB2,
            layer_keys,
            epochs,
            apply_bn=args.apply_bn,
            use_nm=args.use_nm_correction,
            combine_patches=args.combine_patches,
            patch_mean=args.patch_mean,
            flatten=args.flatten_patches_as_features,
            logger=logger,
        )
        profiles_sets.append((profiles2, args.label2, "dashed"))
    plot_profiles(Path(args.output), profiles_sets, epochs, title=args.title, marker=args.point_style)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
