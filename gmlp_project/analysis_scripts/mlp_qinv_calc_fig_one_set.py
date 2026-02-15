#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


MLP_RECORD_LAYERS: List[str] = [
    f"blocks.{i}.bn:post" for i in range(10)
]


def build_args():
    p = argparse.ArgumentParser("q_inv over epochs for one A/B set on one figure (MLP layers)")
    p.add_argument("--spin_file_a", type=str, required=True)
    p.add_argument("--spin_file_b", type=str, required=True)
    p.add_argument("--metrics_a", type=str, default=None)
    p.add_argument("--metrics_b", type=str, default=None)
    p.add_argument("--label", type=str, default=None)
    p.add_argument(
        "--layer-index",
        type=int,
        choices=list(range(len(MLP_RECORD_LAYERS))),
        default=0,
        help="Target index for MLP layers (ignored if --all-layers): "
             + ", ".join(f"{i}:{k}" for i, k in enumerate(MLP_RECORD_LAYERS)),
    )
    p.add_argument("--all-layers", action="store_true", default=False,
                   help="Process and plot all layers")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="./mlp_qinv_out/single")
    p.add_argument("--apply-bn", default=True, action="store_true")
    p.add_argument("--use-nm-correction", default=True, action="store_true")
    p.add_argument("--combine-patches", default=False, action="store_true")
    p.add_argument("--patch-mean", default=False, action="store_true")
    p.add_argument("--flatten-patches-as-features", default=False, action="store_true")
    p.add_argument("--skip-epoch0", default=True, action="store_true")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args, _unknown = p.parse_known_args()
    return args


def setup_logger(level="INFO"):
    logging.basicConfig(level=getattr(logging, level),
                        format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger("qinv1_mlp")


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


def calc_qinv_epoch(A: np.ndarray, B: np.ndarray, *, use_nm: bool, combine_patches: bool) -> Tuple[float, float, float, float, bool]:
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
                    A[:, s, :], B[:, s, :], use_nm=use_nm, combine_patches=False
                )
                qinv_list.append(qinv)
                q2ab_list.append(q2ab)
                q2aa_list.append(q2aa)
                q2bb_list.append(q2bb)
                nm_flags.append(bool(use_nm_applied))
            use_nm_applied_all = all(nm_flags) if nm_flags else False
            return (
                float(np.mean(qinv_list)),
                float(np.mean(q2ab_list)),
                float(np.mean(q2aa_list)),
                float(np.mean(q2bb_list)),
                use_nm_applied_all,
            )
    else:
        raise ValueError(f"Unsupported ndim for q_inv: {A.shape}")
    M, N = A2.shape
    q_ab = (A2.T @ B2) / M
    qaa = (A2.T @ A2) / M
    qbb = (B2.T @ B2) / M
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
    q2_aa = max(0.0, float(q2_aa))
    q2_bb = max(0.0, float(q2_bb))
    denom = np.sqrt(q2_aa) * np.sqrt(q2_bb)
    if not np.isfinite(denom) or denom < 1e-8:
        q_inv = 0.0
    else:
        q_inv = float(q2_ab / denom)
    q_inv = float(np.clip(q_inv, 0.0, 1.0))
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
    return MLP_RECORD_LAYERS[idx]


def all_layer_keys() -> List[str]:
    return list(MLP_RECORD_LAYERS)


_FIXED_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
]


def layer_idx_from_key_mlp(layer_key: str) -> int:
    try:
        return MLP_RECORD_LAYERS.index(layer_key)
    except ValueError:
        m = re.search(r"blocks\.(\d+)\.bn:post", layer_key)
        if m:
            return int(m.group(1))
        return 0


def compute_qinv_df_for_layer(
    spinA_path: str,
    spinB_path: str,
    layer_key: str,
    *,
    apply_bn: bool,
    use_nm: bool,
    combine_patches: bool,
    patch_mean: bool,
    flatten: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
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
        if Ae.shape[0] != Mm:
            Ae = Ae[:Mm]
        if Be.shape[0] != Mm:
            Be = Be[:Mm]
        if flatten and Ae.ndim == 3:
            Mx, Sx, Dx = Ae.shape
            Ae = Ae.reshape(Mx, Sx * Dx)
            Be = Be.reshape(Mx, Sx * Dx)
        if (not flatten) and patch_mean and Ae.ndim == 3:
            Ae = Ae.mean(axis=1)
            Be = Be.mean(axis=1)
        if apply_bn:
            Ae = batch_norm_np(Ae)
            Be = batch_norm_np(Be)
        q_inv, _q2_ab, _q2_aa, _q2_bb, use_nm_applied = calc_qinv_epoch(
            Ae, Be, use_nm=use_nm, combine_patches=combine_patches
        )
        rows.append(
            {
                "epoch": lab,
                "q_inv": q_inv,
                "M": Ae.shape[0] if Ae.ndim == 2 else Ae.shape[0] * (Ae.shape[1] if combine_patches else 1),
                "use_nm_applied": bool(use_nm_applied),
            }
        )
    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    return df


def load_metrics_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    required = {"epoch", "split", "loss", "acc"}
    if not required.issubset(df.columns):
        return None
    df = df.copy()
    df["epoch"] = df["epoch"].astype(int)
    df["split"] = df["split"].astype(str)
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df["acc"] = pd.to_numeric(df["acc"], errors="coerce")
    return df.dropna(subset=["loss", "acc"])


def default_label_from_path(path: str) -> str:
    p = Path(path)
    return p.parent.name or p.stem


def plot_single(
    fig_out: Path,
    layer_key: str,
    *,
    df: pd.DataFrame,
    label: str,
    skip_epoch0: bool,
    title: Optional[str],
    metrics_a: Optional[pd.DataFrame],
    metrics_b: Optional[pd.DataFrame],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    def filt(x: np.ndarray, y: np.ndarray):
        if skip_epoch0:
            m = x != 0
            return x[m], y[m]
        return x, y

    x = df["epoch"].to_numpy()
    y = df["q_inv"].to_numpy()
    x, y = filt(x, y)
    h_q, = ax.plot(x, y, marker="o", linewidth=2.5, label=f"qinv-{label}")

    acc_handles, acc_labels = [], []

    def plot_acc(dfm: Optional[pd.DataFrame], prefix: str):
        if dfm is None or dfm.empty:
            return
        for split, ls in [("train", "--"), ("val", "--")]:
            sub = dfm[dfm["split"] == split].sort_values("epoch")
            if sub.empty:
                continue
            x2 = sub["epoch"].to_numpy()
            y2 = sub["acc"].to_numpy()
            x2, y2 = filt(x2, y2)
            h, = ax.plot(x2, y2, linestyle=ls, linewidth=1.8, alpha=0.9, label=f"{prefix}-acc-{split}")
            acc_handles.append(h)
            acc_labels.append(f"{prefix}-acc-{split}")

    plot_acc(metrics_a, f"A-{label}")
    plot_acc(metrics_b, f"B-{label}")

    ax2 = ax.twinx()
    loss_handles, loss_labels = [], []

    def plot_loss(dfm: Optional[pd.DataFrame], prefix: str):
        if dfm is None or dfm.empty:
            return
        for split in ["train", "val"]:
            sub = dfm[dfm["split"] == split].sort_values("epoch")
            if sub.empty:
                continue
            x2 = sub["epoch"].to_numpy()
            y2 = sub["loss"].to_numpy()
            x2, y2 = filt(x2, y2)
            y2 = np.clip(y2, 1e-12, None)
            h, = ax2.plot(x2, y2, linewidth=1.8, linestyle=":", alpha=0.9, label=f"{prefix}-loss-{split}")
            loss_handles.append(h)
            loss_labels.append(f"{prefix}-loss-{split}")

    plot_loss(metrics_a, f"A-{label}")
    plot_loss(metrics_b, f"B-{label}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("q_inv / acc")
    ax.set_ylim(0, 1)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax2.set_ylabel("loss (log)")
    ax2.set_yscale("log")
    ax.set_title(title or f"q_inv over epochs at {layer_key}")

    leg_q = ax.legend(handles=[h_q], bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=10, title="q_inv")
    ax.add_artist(leg_q)
    metric_handles = acc_handles + loss_handles
    metric_labels = acc_labels + loss_labels
    if metric_handles:
        leg_metrics = ax.legend(metric_handles, metric_labels, bbox_to_anchor=(0.5, -0.22),
                                loc="upper center", ncol=4, fontsize=9, title="metrics")
    else:
        leg_metrics = None
    fig.subplots_adjust(right=0.72, bottom=0.30)
    extra_artists = [leg_q]
    if leg_metrics is not None:
        extra_artists.append(leg_metrics)
    fig.savefig(fig_out, dpi=150, bbox_inches="tight", bbox_extra_artists=extra_artists)
    plt.close(fig)


def plot_multi(fig_out: Path, layer_keys: List[str], *,
               df_all: pd.DataFrame, label: str, skip_epoch0: bool, title: Optional[str]) -> None:
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(12, 7))

    proxies = []
    layer_keys_sorted = sorted(layer_keys, key=layer_idx_from_key_mlp)
    for lk in layer_keys_sorted:
        idx = layer_idx_from_key_mlp(lk)
        color = _FIXED_COLORS[idx % len(_FIXED_COLORS)]
        sub = df_all[df_all["layer"] == lk].sort_values("epoch")
        if sub.empty:
            continue
        x = sub["epoch"].to_numpy()
        y = sub["q_inv"].to_numpy()
        if skip_epoch0 and len(x):
            m = x != 0
            x, y = x[m], y[m]
        ax.plot(x, y, color=color, linestyle='-', marker='o', linewidth=2.0, label=f"{lk} ({label})")
        proxies.append(Line2D([0], [0], color=color, lw=2, label=f"L={idx}"))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("q_inv")
    ax.set_ylim(0, 1)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_title(title or "q_inv over epochs (all layers)")

    style_proxies = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label=label),
    ]
    leg_style = ax.legend(style_proxies, [label], bbox_to_anchor=(0.5, -0.10), loc='upper center',
                          ncol=1, fontsize=9, title='set')

    ncol = min(5, max(1, len(layer_keys_sorted)))
    leg_layers = ax.legend(handles=proxies, bbox_to_anchor=(0.5, -0.26), loc="upper center",
                           fontsize=9, title="layers", ncol=ncol)
    ax.add_artist(leg_layers)

    fig.subplots_adjust(bottom=0.32)
    fig.savefig(fig_out, dpi=150, bbox_inches='tight', bbox_extra_artists=[leg_style, leg_layers])
    plt.close(fig)


def main() -> int:
    args = build_args()
    logger = setup_logger(args.log_level)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or default_label_from_path(args.spin_file_a)

    if args.all_layers:
        lkeys = all_layer_keys()
        logger.info(f"Target layers (MLP): {lkeys}")
        df_list = []
        for i, lk in enumerate(lkeys):
            logger.info(f"Computing {lk} ({i + 1}/{len(lkeys)})")
            d1 = compute_qinv_df_for_layer(
                args.spin_file_a, args.spin_file_b, lk,
                apply_bn=args.apply_bn, use_nm=args.use_nm_correction,
                combine_patches=args.combine_patches, patch_mean=args.patch_mean,
                flatten=args.flatten_patches_as_features, logger=logger,
            )
            d1["layer"] = lk
            df_list.append(d1)
        df_all = pd.concat(df_list, ignore_index=True)
        df_all.assign(set=label).to_csv(out_dir / f"qinv_{label}_all_layers.csv", index=False)
        fig_out = out_dir / "qinv_single_all_layers.png"
        plot_multi(fig_out, lkeys, df_all=df_all, label=label, skip_epoch0=args.skip_epoch0, title=args.title)
        joblib.dump({
            "config": vars(args),
            "layers": lkeys,
            "set": {"label": label, "table": df_all},
        }, out_dir / "qinv_single_all_layers.joblib")
        logger.info(f"Saved: {fig_out}")
    else:
        layer_key = layer_key_for_index(args.layer_index)
        logger.info(f"Target layer (MLP): {layer_key}")
        df = compute_qinv_df_for_layer(
            args.spin_file_a, args.spin_file_b, layer_key,
            apply_bn=args.apply_bn, use_nm=args.use_nm_correction,
            combine_patches=args.combine_patches, patch_mean=args.patch_mean,
            flatten=args.flatten_patches_as_features, logger=logger,
        )
        metrics_a = load_metrics_csv(args.metrics_a)
        metrics_b = load_metrics_csv(args.metrics_b)
        df.assign(layer=layer_key, set=label).to_csv(out_dir / f"qinv_{label}_layer{args.layer_index}.csv", index=False)
        joblib.dump({
            "config": vars(args),
            "layer": layer_key,
            "set": {"label": label, "table": df},
        }, out_dir / f"qinv_single_layer{args.layer_index}.joblib")
        fig_out = out_dir / f"qinv_single_layer{args.layer_index}.png"
        plot_single(fig_out, layer_key, df=df, label=label,
                    skip_epoch0=args.skip_epoch0, title=args.title,
                    metrics_a=metrics_a, metrics_b=metrics_b)
        logger.info(f"Saved: {fig_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
