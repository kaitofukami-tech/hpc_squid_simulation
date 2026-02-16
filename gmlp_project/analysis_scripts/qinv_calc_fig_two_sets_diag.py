#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import re

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def build_args():
    p = argparse.ArgumentParser("q_inv (diag-only) over epochs for two A/B sets on one figure")
    # Set 1
    p.add_argument("--spin_file_a1", type=str, required=True)
    p.add_argument("--spin_file_b1", type=str, required=True)
    p.add_argument("--metrics_a1", type=str, default=None)
    p.add_argument("--metrics_b1", type=str, default=None)
    p.add_argument("--label1", type=str, default=None)
    # Set 2
    p.add_argument("--spin_file_a2", type=str, required=True)
    p.add_argument("--spin_file_b2", type=str, required=True)
    p.add_argument("--metrics_a2", type=str, default=None)
    p.add_argument("--metrics_b2", type=str, default=None)
    p.add_argument("--label2", type=str, default=None)
    # Layer and plotting
    # Layer selection
    p.add_argument("--layer-index", type=int, choices=[0,1,2,3,4,5,6,7,8,9], default=0,
                   help="Target index for gmlp_blocks.[i].bn:post (ignored if --all-layers)")
    p.add_argument("--all-layers", action="store_true", default=True,
                   help="Process and plot all layers (0..9) with set1 solid and set2 dashed")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="./gmlp_qinv_out/dual")
    # Options
    p.add_argument("--apply-bn", default=True, action="store_true")
    p.add_argument("--use-nm-correction", default=True, action="store_true")
    p.add_argument("--combine-patches", default=False, action="store_true")
    p.add_argument("--patch-mean", default=False, action="store_true")
    p.add_argument("--flatten-patches-as-features", default=True, action="store_true")
    p.add_argument("--skip-epoch0", default=True, action="store_true")
    p.add_argument("--q-block-size", type=int, default=0,
                   help="Unused in diag-only version (kept for CLI compatibility).")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    # Allow ignoring stray args (e.g., accidental positional params from qsub invocations)
    args, _unknown = p.parse_known_args()
    return args

def setup_logger(level="INFO"):
    logging.basicConfig(level=getattr(logging, level),
                        format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger("qinv2_diag")

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
    else:
        raise ValueError(f"Unsupported ndim for BN: {x.shape}")

def _q2_blockwise(X: np.ndarray, Y: np.ndarray, block_size: int) -> float:
    M, N = X.shape
    bs = max(1, min(int(block_size), N))
    total = 0.0
    for i in range(0, N, bs):
        Xi = X[:, i:i + bs]
        for j in range(0, N, bs):
            Yj = Y[:, j:j + bs]
            Qij = (Xi.T @ Yj) / float(M)
            total += float((Qij * Qij).sum())
    return total / float(N)


def calc_qinv_epoch(A: np.ndarray, B: np.ndarray, *, use_nm: bool, combine_patches: bool,
                    q_block_size: int) -> Tuple[float, float, float, float, bool]:
    # Use float64 for numerical stability
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
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
            qinv_list, q2ab_list, q2aa_list, q2bb_list = [], [], [], []
            nm_flags = []
            for s in range(A.shape[1]):
                qinv, q2ab, q2aa, q2bb, use_nm_applied = calc_qinv_epoch(
                    A[:, s, :], B[:, s, :], use_nm=use_nm, combine_patches=False, q_block_size=q_block_size
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
    # diag-only version: use only i=j terms (delta_ij)
    # diag(q_ab) = sum_m A[m,i] * B[m,i] / M
    ab_diag = (A2 * B2).sum(axis=0) / M
    aa_diag = (A2 * A2).sum(axis=0) / M
    bb_diag = (B2 * B2).sum(axis=0) / M
    q2_ab_raw = float((ab_diag ** 2).mean())
    q2_aa_raw = float((aa_diag ** 2).mean())
    q2_bb_raw = float((bb_diag ** 2).mean())
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
    # Clip to avoid negative due to numerical error
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
    if not isinstance(data, dict) or "meta" not in data:
        raise ValueError(f"Invalid pkl structure: {path}")
    return data

def _coerce_int_seq(seq: Sequence) -> List[int]:
    out = []
    for v in seq:
        if isinstance(v, (int, np.integer)):
            out.append(int(v))
        else:
            out.append(int(round(float(v))))
    return out

def get_epoch_labels_for(data: Dict, example_layer: str) -> List[int]:
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

def layer_key_for_index(idx: int) -> str:
    return f"gmlp_blocks.{idx}.bn:post"

def all_layer_keys() -> List[str]:
    return [layer_key_for_index(i) for i in range(10)]

_FIXED_COLORS = [
    "#1f77b4",  # L=0 blue
    "#ff7f0e",  # L=1 orange
    "#2ca02c",  # L=2 green
    "#d62728",  # L=3 red
    "#9467bd",  # L=4 purple
    "#8c564b",  # L=5 brown
    "#e377c2",  # L=6 pink
    "#7f7f7f",  # L=7 gray
    "#bcbd22",  # L=8 olive
    "#17becf",  # L=9 teal
]

def layer_idx_from_key_gmlp(layer_key: str) -> int:
    m = re.search(r"gmlp_blocks\.(\d+)\.bn:post", layer_key)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0

def compute_qinv_df_for_layer(spinA_path: str, spinB_path: str, layer_key: str, *,
                              apply_bn: bool, use_nm: bool, combine_patches: bool,
                              patch_mean: bool, flatten: bool, q_block_size: int,
                              logger: logging.Logger) -> pd.DataFrame:
    A = load_spin_pkl(spinA_path)
    B = load_spin_pkl(spinB_path)
    if layer_key not in A or layer_key not in B:
        raise KeyError(f"Layer '{layer_key}' not found in both spin files.")
    labels_A = get_epoch_labels_for(A, layer_key)
    labels_B = get_epoch_labels_for(B, layer_key)
    matched = sorted(set(labels_A) & set(labels_B))
    if not matched:
        raise ValueError("No overlapping epoch labels between A and B.")
    idxA = {lab: i for i, lab in enumerate(labels_A)}
    idxB = {lab: i for i, lab in enumerate(labels_B)}
    rows = []
    arrA_all = A[layer_key]
    arrB_all = B[layer_key]
    for lab in matched:
        eA = idxA[lab]
        eB = idxB[lab]
        Ae = np.array(arrA_all[eA], copy=False)
        Be = np.array(arrB_all[eB], copy=False)
        Mm = min(Ae.shape[0], Be.shape[0])
        if Ae.shape[0] != Mm: Ae = Ae[:Mm]
        if Be.shape[0] != Mm: Be = Be[:Mm]
        if flatten and Ae.ndim == 3:
            Mx, Sx, Dx = Ae.shape
            Ae = Ae.reshape(Mx, Sx * Dx)
            Be = Be.reshape(Mx, Sx * Dx)
        if (not flatten) and patch_mean and Ae.ndim == 3:
            Ae = Ae.mean(axis=1)
            Be = Be.mean(axis=1)
        if apply_bn:
            if Ae.ndim == 2:
                Ae = batch_norm_np(Ae); Be = batch_norm_np(Be)
            elif Ae.ndim == 3:
                Ae = batch_norm_np(Ae); Be = batch_norm_np(Be)
        q_inv, q2_ab, q2_aa, q2_bb, use_nm_applied = calc_qinv_epoch(
            Ae, Be, use_nm=use_nm, combine_patches=combine_patches, q_block_size=q_block_size
        )
        rows.append({
            "epoch": lab,
            "q_inv": q_inv,
            "M": Ae.shape[0] if Ae.ndim == 2 else Ae.shape[0] * (Ae.shape[1] if combine_patches else 1),
            "use_nm_applied": bool(use_nm_applied),
        })
    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    return df

def load_metrics_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path: return None
    df = pd.read_csv(path)
    required = {"epoch", "split", "loss", "acc"}
    if not required.issubset(df.columns): return None
    df = df.copy()
    df["epoch"] = df["epoch"].astype(int)
    df["split"] = df["split"].astype(str)
    df["loss"]  = pd.to_numeric(df["loss"], errors="coerce")
    df["acc"]   = pd.to_numeric(df["acc"], errors="coerce")
    return df.dropna(subset=["loss", "acc"])

def default_label_from_path(path: str) -> str:
    p = Path(path)
    return p.parent.name or p.stem

def plot_dual_single(fig_out: Path, layer_key: str, *,
                     df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str,
                     skip_epoch0: bool, title: Optional[str], metrics_a1: Optional[pd.DataFrame],
                     metrics_b1: Optional[pd.DataFrame], metrics_a2: Optional[pd.DataFrame], metrics_b2: Optional[pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    def filt(x: np.ndarray, y: np.ndarray):
        if skip_epoch0:
            m = x != 0
            return x[m], y[m]
        return x, y
    # q_inv lines
    x1 = df1["epoch"].to_numpy(); y1 = df1["q_inv"].to_numpy(); x1, y1 = filt(x1, y1)
    x2 = df2["epoch"].to_numpy(); y2 = df2["q_inv"].to_numpy(); x2, y2 = filt(x2, y2)
    h_q1, = ax.plot(x1, y1, marker="o", linewidth=2.5, label=f"qinv-{label1}")
    h_q2, = ax.plot(x2, y2, marker="s", linewidth=2.5, label=f"qinv-{label2}")
    # Acc overlays
    acc_handles, acc_labels = [], []
    def plot_acc(dfm: Optional[pd.DataFrame], prefix: str):
        if dfm is None or dfm.empty: return
        for split, ls in [("train","--"), ("val","--")]:
            sub = dfm[dfm["split"] == split].sort_values("epoch")
            if sub.empty: continue
            x = sub["epoch"].to_numpy(); y = sub["acc"].to_numpy(); x, y = filt(x, y)
            h, = ax.plot(x, y, linestyle=ls, linewidth=1.8, alpha=0.9, label=f"{prefix}-acc-{split}")
            acc_handles.append(h); acc_labels.append(f"{prefix}-acc-{split}")
    plot_acc(metrics_a1, f"A-{label1}"); plot_acc(metrics_b1, f"B-{label1}")
    plot_acc(metrics_a2, f"A-{label2}"); plot_acc(metrics_b2, f"B-{label2}")
    # Loss overlays
    ax2 = ax.twinx()
    loss_handles, loss_labels = [], []
    def plot_loss(dfm: Optional[pd.DataFrame], prefix: str):
        if dfm is None or dfm.empty: return
        for split in ["train", "val"]:
            sub = dfm[dfm["split"] == split].sort_values("epoch")
            if sub.empty: continue
            x = sub["epoch"].to_numpy(); y = sub["loss"].to_numpy(); x, y = filt(x, y)
            y = np.clip(y, 1e-12, None)
            h, = ax2.plot(x, y, linewidth=1.8, linestyle=":", alpha=0.9, label=f"{prefix}-loss-{split}")
            loss_handles.append(h); loss_labels.append(f"{prefix}-loss-{split}")
    plot_loss(metrics_a1, f"A-{label1}"); plot_loss(metrics_b1, f"B-{label1}")
    plot_loss(metrics_a2, f"A-{label2}"); plot_loss(metrics_b2, f"B-{label2}")
    # Axes
    ax.set_xlabel("Epoch"); ax.set_ylabel("q_inv / acc"); ax.set_ylim(0, 1); ax.set_xscale("log"); ax.grid(True, alpha=0.3)
    ax2.set_ylabel("loss (log)"); ax2.set_yscale("log")
    ax.set_title(title or f"q_inv over epochs at {layer_key}")
    # Legends outside
    leg_q = ax.legend(handles=[h_q1, h_q2], bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=10, title="q_inv")
    ax.add_artist(leg_q)
    metric_handles = acc_handles + loss_handles
    metric_labels = acc_labels + loss_labels
    if metric_handles:
        leg_metrics = ax.legend(metric_handles, metric_labels, bbox_to_anchor=(0.5, -0.22), loc="upper center", ncol=4, fontsize=9, title="metrics")
    else:
        leg_metrics = None
    fig.subplots_adjust(right=0.72, bottom=0.30)
    extra_artists = [leg_q]
    if leg_metrics is not None:
        extra_artists.append(leg_metrics)
    fig.savefig(fig_out, dpi=150, bbox_inches="tight", bbox_extra_artists=extra_artists)
    plt.close(fig)

def plot_dual_multi(fig_out: Path, layer_keys: List[str], *,
                    df1_all: pd.DataFrame, df2_all: pd.DataFrame,
                    label1: str, label2: str, skip_epoch0: bool, title: Optional[str]) -> None:
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(12, 7))

    proxies = []
    layer_keys_sorted = sorted(layer_keys, key=layer_idx_from_key_gmlp)
    for lk in layer_keys_sorted:
        idx = layer_idx_from_key_gmlp(lk)
        color = _FIXED_COLORS[idx % len(_FIXED_COLORS)]
        sub1 = df1_all[df1_all["layer"] == lk].sort_values("epoch")
        sub2 = df2_all[df2_all["layer"] == lk].sort_values("epoch")
        if sub1.empty and sub2.empty:
            continue
        x1 = sub1["epoch"].to_numpy(); y1 = sub1["q_inv"].to_numpy()
        x2 = sub2["epoch"].to_numpy(); y2 = sub2["q_inv"].to_numpy()
        if skip_epoch0:
            if len(x1):
                m1 = x1 != 0; x1, y1 = x1[m1], y1[m1]
            if len(x2):
                m2 = x2 != 0; x2, y2 = x2[m2], y2[m2]
        if len(x1):
            ax.plot(x1, y1, color=color, linestyle='-', marker='o', linewidth=2.0, label=f"{lk} ({label1})")
        if len(x2):
            ax.plot(x2, y2, color=color, linestyle='--', marker='s', linewidth=2.0, label=f"{lk} ({label2})")
        proxies.append(Line2D([0], [0], color=color, lw=2, label=f"L={idx}"))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("q_inv")
    ax.set_ylim(0, 1)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_title(title or "q_inv over epochs (all layers)")

    # Legends: styles by set (top), colors by layer (bottom, multi-column)
    style_proxies = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label=label1),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label=label2),
    ]
    leg_style = ax.legend(style_proxies, [label1, label2], bbox_to_anchor=(0.5, -0.10), loc='upper center', ncol=2, fontsize=9, title='sets')

    ncol = min(5, max(1, len(layer_keys_sorted)))
    leg_layers = ax.legend(handles=proxies, bbox_to_anchor=(0.5, -0.26), loc="upper center", fontsize=9, title="layers", ncol=ncol)
    ax.add_artist(leg_layers)

    fig.subplots_adjust(bottom=0.32)
    fig.savefig(fig_out, dpi=150, bbox_inches='tight', bbox_extra_artists=[leg_style, leg_layers])
    plt.close(fig)

def main() -> int:
    args = build_args()
    logger = setup_logger(args.log_level)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    label1 = args.label1 or default_label_from_path(args.spin_file_a1)
    label2 = args.label2 or default_label_from_path(args.spin_file_a2)
    if args.all_layers:
        lkeys = all_layer_keys()
        logger.info(f"Target layers: {lkeys}")
        df1_list, df2_list = [], []
        for i, lk in enumerate(lkeys):
            logger.info(f"Computing {lk} ({i+1}/{len(lkeys)})")
            d1 = compute_qinv_df_for_layer(
                args.spin_file_a1, args.spin_file_b1, lk,
                apply_bn=args.apply_bn, use_nm=args.use_nm_correction,
                combine_patches=args.combine_patches, patch_mean=args.patch_mean,
                flatten=args.flatten_patches_as_features, q_block_size=args.q_block_size, logger=logger,
            )
            d1["layer"] = lk
            df1_list.append(d1)
            d2 = compute_qinv_df_for_layer(
                args.spin_file_a2, args.spin_file_b2, lk,
                apply_bn=args.apply_bn, use_nm=args.use_nm_correction,
                combine_patches=args.combine_patches, patch_mean=args.patch_mean,
                flatten=args.flatten_patches_as_features, q_block_size=args.q_block_size, logger=logger,
            )
            d2["layer"] = lk
            df2_list.append(d2)
        df1_all = pd.concat(df1_list, ignore_index=True)
        df2_all = pd.concat(df2_list, ignore_index=True)
        # Save CSVs (per set and combined)
        df1_all.assign(set=label1).to_csv(out_dir / f"qinv_{label1}_all_layers.csv", index=False)
        df2_all.assign(set=label2).to_csv(out_dir / f"qinv_{label2}_all_layers.csv", index=False)
        pd.concat([df1_all.assign(set=label1), df2_all.assign(set=label2)], ignore_index=True).to_csv(
            out_dir / "qinv_dual_all_layers.csv", index=False
        )
        # Plot
        fig_out = out_dir / "qinv_dual_all_layers.png"
        plot_dual_multi(fig_out, lkeys, df1_all=df1_all, df2_all=df2_all,
                        label1=label1, label2=label2, skip_epoch0=args.skip_epoch0, title=args.title)
        joblib.dump({
            "config": vars(args),
            "layers": lkeys,
            "set1": {"label": label1, "table": df1_all},
            "set2": {"label": label2, "table": df2_all},
        }, out_dir / "qinv_dual_all_layers.joblib")
        logger.info(f"Saved: {fig_out}")
    else:
        layer_key = layer_key_for_index(args.layer_index)
        logger.info(f"Target layer: {layer_key}")
        df1 = compute_qinv_df_for_layer(
            args.spin_file_a1, args.spin_file_b1, layer_key,
            apply_bn=args.apply_bn, use_nm=args.use_nm_correction,
            combine_patches=args.combine_patches, patch_mean=args.patch_mean,
            flatten=args.flatten_patches_as_features, q_block_size=args.q_block_size, logger=logger,
        )
        df2 = compute_qinv_df_for_layer(
            args.spin_file_a2, args.spin_file_b2, layer_key,
            apply_bn=args.apply_bn, use_nm=args.use_nm_correction,
            combine_patches=args.combine_patches, patch_mean=args.patch_mean,
            flatten=args.flatten_patches_as_features, q_block_size=args.q_block_size, logger=logger,
        )
        metrics_a1 = load_metrics_csv(args.metrics_a1)
        metrics_b1 = load_metrics_csv(args.metrics_b1)
        metrics_a2 = load_metrics_csv(args.metrics_a2)
        metrics_b2 = load_metrics_csv(args.metrics_b2)
        df1.assign(layer=layer_key, set=label1).to_csv(out_dir / f"qinv_{label1}_layer{args.layer_index}.csv", index=False)
        df2.assign(layer=layer_key, set=label2).to_csv(out_dir / f"qinv_{label2}_layer{args.layer_index}.csv", index=False)
        joblib.dump({
            "config": vars(args),
            "layer": layer_key,
            "set1": {"label": label1, "table": df1},
            "set2": {"label": label2, "table": df2},
        }, out_dir / f"qinv_dual_layer{args.layer_index}.joblib")
        fig_out = out_dir / f"qinv_dual_layer{args.layer_index}.png"
        plot_dual_single(fig_out, layer_key,
                         df1=df1, df2=df2, label1=label1, label2=label2,
                         skip_epoch0=args.skip_epoch0, title=args.title,
                         metrics_a1=metrics_a1, metrics_b1=metrics_b1,
                         metrics_a2=metrics_a2, metrics_b2=metrics_b2)
        logger.info(f"Saved: {fig_out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
