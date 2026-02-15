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
    p.add_argument("--center", action="store_true", default=False)
    p.add_argument("--output", default="./gmlp_qinv_out/manifold/overlap_layer_profile.png")
    p.add_argument("--title", default=None)
    return p.parse_args()


def load_spin(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid pickle: {path}")
    return data


def get_epochs(data: Dict[str, Any], layer_key: str) -> List[int]:
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
    layers = [k for k in data.keys() if pat.fullmatch(k)]
    layers.sort(key=lambda k: int(pat.fullmatch(k).group(1)))
    if not layers:
        layers = [f"gmlp_blocks.{i}.bn:post" for i in range(10)]
    return layers


def preprocess(arr: np.ndarray, flatten: bool, patch_mean: bool) -> np.ndarray:
    if arr.ndim == 3:
        if flatten:
            M, S, D = arr.shape
            return arr.reshape(M, S * D)
        if patch_mean:
            return arr.mean(axis=1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unsupported array shape {arr.shape}")


def extract_epoch(
    data: Dict[str, Any],
    layer_key: str,
    epoch: int,
    epochs_list: List[int],
) -> Tuple[Optional[np.ndarray], bool]:
    idx = {lab: i for i, lab in enumerate(epochs_list)}
    if epoch not in idx:
        return None, False
    arr = np.array(data[layer_key][idx[epoch]], copy=False)
    return arr, True


def align_samples(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = min(A.shape[0], B.shape[0])
    if A.shape[0] != M:
        A = A[:M]
    if B.shape[0] != M:
        B = B[:M]
    return A, B


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
    center: bool,
) -> Dict[str, Dict[int, Dict[int, Dict[str, Any]]]]:
    profiles: Dict[str, Dict[int, Dict[int, Dict[str, Any]]]] = {"set1": {}, "set2": {}, "inter": {}}
    shared_epochs1 = {layer: get_epochs(dataA1, layer) for layer in layers}
    shared_epochs2 = {layer: get_epochs(dataA2, layer) for layer in layers}
    for layer in layers:
        layer_idx = int(re.search(r"gmlp_blocks\.(\d+)", layer).group(1))
        for epoch in epochs:
            Ae1, okA1 = extract_epoch(dataA1, layer, epoch, shared_epochs1[layer])
            Be1, okB1 = extract_epoch(dataB1, layer, epoch, shared_epochs1[layer])
            Ae2, okA2 = extract_epoch(dataA2, layer, epoch, shared_epochs2[layer])
            Be2, okB2 = extract_epoch(dataB2, layer, epoch, shared_epochs2[layer])
            if not (okA1 and okB1 and okA2 and okB2):
                print(f"[warn] epoch {epoch} missing for layer {layer}; skipping.")
                continue
            Ae1, Be1 = align_samples(Ae1, Be1)
            Sa1 = preprocess(Ae1, flatten, patch_mean)
            Sb1 = preprocess(Be1, flatten, patch_mean)
            Ae2, Be2 = align_samples(Ae2, Be2)
            Sa2 = preprocess(Ae2, flatten, patch_mean)
            Sb2 = preprocess(Be2, flatten, patch_mean)
            metrics1 = overlap_metrics("intra", Sa1, Sb1, center=center, backend="numpy")
            metrics2 = overlap_metrics("intra", Sa2, Sb2, center=center, backend="numpy")
            metrics_inter = overlap_metrics(
                "inter", Sa1, Sb1, Sa2=Sa2, Sb2=Sb2, center=center, backend="numpy"
            )
            profiles["set1"].setdefault(epoch, {})[layer_idx] = {
                "q_inv": metrics1["q_inv"],
                "use_nm_applied": bool(metrics1.get("use_nm_applied", False)),
            }
            profiles["set2"].setdefault(epoch, {})[layer_idx] = {
                "q_inv": metrics2["q_inv"],
                "use_nm_applied": bool(metrics2.get("use_nm_applied", False)),
            }
            profiles["inter"].setdefault(epoch, {})[layer_idx] = {
                "q_inv": metrics_inter["q_inv"],
                "use_nm_applied": bool(metrics_inter.get("use_nm_applied", False)),
            }
    return profiles


def plot_profiles(
    profiles: Dict[str, Dict[int, Dict[int, Dict[str, Any]]]],
    epochs: Sequence[int],
    output: Path,
    *,
    title: Optional[str],
    label1: str,
    label2: str,
):
    layer_indices = sorted(
        {idx for epoch_map in profiles["set1"].values() for idx in epoch_map.keys()}
    )
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd"]
    fig, ax = plt.subplots(figsize=(9, 5))
    style_map = {"set1": "solid", "set2": "dashed", "inter": "dotted"}
    label_map = {"set1": label1, "set2": label2, "inter": "inter"}
    epoch_colors = {}
    for epi_idx, epoch in enumerate(epochs):
        color = colors[epi_idx % len(colors)]
        epoch_colors[epoch] = color
        for key in ["set1", "set2", "inter"]:
            epoch_map = profiles[key].get(epoch)
            if epoch_map is None:
                continue
            y = [epoch_map.get(idx, {}).get("q_inv", np.nan) for idx in layer_indices]
            if all(np.isnan(y)):
                continue
            ax.plot(
                layer_indices,
                y,
                color=color,
                linestyle=style_map[key],
                linewidth=2.0,
                marker="o",
                label=f"{label_map[key]} epoch {epoch}",
            )
    ax.set_xlabel("Layer index L")
    ax.set_ylabel("q_inv (overlap)")
    ax.set_xticks(layer_indices)
    ax.set_ylim(0.5, 1.0)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_title(title or "Overlap metrics vs layer")
    style_handles = [
        Line2D([0], [0], color="black", linestyle=style_map[key], linewidth=2.0, label=label_map[key])
        for key in ["set1", "set2", "inter"]
    ]
    legend_style = ax.legend(style_handles, [h.get_label() for h in style_handles], title="Replica set", loc="upper left")
    ax.add_artist(legend_style)
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



def save_profiles_csv(
    profiles: Dict[str, Dict[int, Dict[int, Dict[str, Any]]]],
    output_path: Path,
    label1: str,
    label2: str,
):
    """
    Save profile data to a CSV file.
    """
    csv_path = output_path.with_suffix(".csv")
    
    # Flatten the data structure for CSV
    # columns: epoch, layer_idx, type, q_inv
    rows = []
    
    set_mapping = {
        "set1": label1,
        "set2": label2,
        "inter": "inter"
    }

    # Helper function to extract rows from nested dict
    def extract_rows(type_key, type_label):
        data = profiles.get(type_key, {})
        for epoch, layer_data in data.items():
            for layer_idx, q_inv in layer_data.items():
                rows.append({
                    "epoch": epoch,
                    "layer_idx": layer_idx,
                    "type": type_label,
                    "q_inv": q_inv.get("q_inv"),
                    "use_nm_applied": q_inv.get("use_nm_applied", False),
                })

    for key, label in set_mapping.items():
        extract_rows(key, label)
        
    # Sort rows for better readability (optional)
    rows.sort(key=lambda x: (x["epoch"], x["layer_idx"], x["type"]))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "layer_idx", "type", "q_inv", "use_nm_applied"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to: {csv_path}")


def main() -> int:
    args = build_args()
    dataA1 = load_spin(args.spin_file_a1)
    dataB1 = load_spin(args.spin_file_b1)
    dataA2 = load_spin(args.spin_file_a2)
    dataB2 = load_spin(args.spin_file_b2)
    layer_keys = resolve_layers(dataA1, args.layers)
    profiles = compute_profiles(
        layer_keys,
        args.epochs,
        dataA1,
        dataB1,
        dataA2,
        dataB2,
        flatten=args.flatten,
        patch_mean=args.patch_mean,
        center=args.center,
    )
    output = Path(args.output).resolve()
    plot_profiles(profiles, args.epochs, output, title=args.title, label1=args.label1, label2=args.label2)
    save_profiles_csv(profiles, output, label1=args.label1, label2=args.label2)
    print(json.dumps({"saved": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
