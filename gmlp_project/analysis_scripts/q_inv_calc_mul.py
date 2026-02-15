#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, logging, pickle, re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- CLI ----------------
def build_args():
    p = argparse.ArgumentParser("q_inv multi-condition comparator (up to 4 conditions)")
    p.add_argument(
        "--cond", action="append", required=True,
        help="Condition spec formatted as 'LABEL|/path/A.pkl|/path/B.pkl'. Repeat up to 4 times."
    )
    p.add_argument("--output-dir", type=str, default="./qinv_multi_out")
    p.add_argument("--apply-bn", action="store_true", default=True,
                   help="各エポックのスピンを M 方向でゼロ平均・単位分散化")
    p.add_argument("--use-nm-correction", action="store_true", default=True,
                   help="-N/M 補正を有効化")
    p.add_argument("--skip-epoch0", action="store_true", default=True,
                   help="エポック0をプロットから除外（CSVは保存）")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()

def setup_logger(level="INFO"):
    logging.basicConfig(level=getattr(logging, level),
                        format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger("qinv-multi")

# ------------- math -------------
def batch_norm_np(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    if x.ndim == 2:
        mean = x.mean(axis=0, keepdims=True)
        var  = x.var(axis=0, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)
    elif x.ndim == 3:
        mean = x.mean(axis=0, keepdims=True)
        var  = x.var(axis=0, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)
    elif x.ndim == 4:
        M,S,D1,D2 = x.shape
        x = x.reshape(M,S,D1*D2)
        mean = x.mean(axis=0, keepdims=True)
        var  = x.var(axis=0, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)
    else:
        raise ValueError(f"Unsupported shape for BN: {x.shape}")

def calc_qinv_epoch(A_e: np.ndarray, B_e: np.ndarray, use_nm: bool) -> Tuple[float, float, float, float]:
    if A_e.shape != B_e.shape:
        raise ValueError(f"Shape mismatch: {A_e.shape} vs {B_e.shape}")
    if A_e.ndim == 2:
        A_e = A_e[:, None, :]
        B_e = B_e[:, None, :]
    M, S, N = A_e.shape
    qinv, q2ab, q2aa, q2bb = [], [], [], []
    for s in range(S):
        A = A_e[:, s, :].astype(np.float64, copy=False)
        B = B_e[:, s, :].astype(np.float64, copy=False)
        qab = (A.T @ B) / M
        q2_ab_raw = (qab**2).sum() / N
        qaa = (A.T @ A) / M
        qbb = (B.T @ B) / M
        q2_aa_raw = (qaa**2).sum() / N
        q2_bb_raw = (qbb**2).sum() / N
        if use_nm:
            q2_ab = q2_ab_raw - (N/M)
            q2_aa = q2_aa_raw - (N/M)
            q2_bb = q2_bb_raw - (N/M)
        else:
            q2_ab, q2_aa, q2_bb = q2_ab_raw, q2_aa_raw, q2_bb_raw
        denom = np.sqrt(abs(q2_aa)) * np.sqrt(abs(q2_bb))
        qinv.append((q2_ab / denom) if denom > 1e-12 else 0.0)
        q2ab.append(q2_ab); q2aa.append(q2_aa); q2bb.append(q2_bb)
    return float(np.mean(qinv)), float(np.mean(q2ab)), float(np.mean(q2aa)), float(np.mean(q2bb))

# ------------- IO helpers -------------
def load_spin_pkl(path: str) -> Dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "meta" not in data:
        raise ValueError(f"Invalid pkl structure: {path}")
    return data

def detect_layers(data: Dict) -> List[str]:
    return [k for k in data.keys() if k != "meta"]

def _coerce_int_seq(seq: Sequence) -> List[int]:
    out = []
    for v in seq:
        try:
            out.append(int(v))
        except Exception:
            out.append(int(round(float(v))))
    return out

def get_epoch_labels_for(data: Dict, example_layer: str) -> List[int]:
    meta = data.get("meta", {})
    E = data[example_layer].shape[0]
    for key in ["saved_epochs", "epoch_ticks", "epochs_saved", "time"]:
        if key in meta and isinstance(meta[key], (list, tuple, np.ndarray)):
            arr = list(meta[key])
            if len(arr) == E:
                return _coerce_int_seq(arr)
    return list(range(E))

# ------------- plotting helpers -------------
def _natural_layer_order(layers: List[str]) -> List[str]:
    def key(ly: str):
        m1 = re.search(r"gmlp_blocks\.(\d+)", ly)
        m2 = re.search(r"features\.(\d+)", ly)
        if m1:
            return (0, int(m1.group(1)), ly)
        if m2:
            return (0, int(m2.group(1)), ly)
        if "out" in ly:
            return (2, 10**9, ly)
        return (1, 10**9, ly)
    return sorted(layers, key=key)

def plot_epochs_per_layer_allconds(df_all: pd.DataFrame, out_dir: Path, skip_epoch0: bool) -> List[Path]:
    """
    レイヤーごとに1枚の図：x=epoch, y=q_inv。条件（cond）を重ね書き。
    """
    out_paths = []
    for layer, sub in df_all.groupby("layer"):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for cond, g in sub.groupby("cond"):
            g = g.sort_values("epoch")
            x = g["epoch"].to_numpy()
            y = g["q_inv"].to_numpy()
            if skip_epoch0:
                mask = x != 0
                x, y = x[mask], y[mask]
            if len(x) == 0: 
                continue
            ax.plot(x, y, marker="o", linewidth=2, label=cond)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("q_inv")
        ax.set_title(f"q_inv over epochs — {layer}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = out_dir / f"qinv_epochs_layer-{layer.replace('.', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(path)
    return out_paths

def plot_by_layer_epoch1_vs_final_allconds(df_all: pd.DataFrame, out_dir: Path) -> Path:
    """
    全レイヤー1枚：x=layer, y=q_inv。各条件の 'epoch=1' と 'final' を重ね描き。
    凡例は 'COND (ep1)', 'COND (final)' 形式。
    """
    layers = _natural_layer_order(sorted(set(df_all["layer"])))
    fig, ax = plt.subplots(figsize=(11, 6))
    for cond, sub in df_all.groupby("cond"):
        last_df = sub.sort_values(["layer","epoch"]).groupby("layer").tail(1)
        ep1_df  = sub[sub["epoch"] == 1]
        y1, yN = [], []
        for ly in layers:
            v1 = ep1_df.loc[ep1_df["layer"] == ly, "q_inv"]
            vN = last_df.loc[last_df["layer"] == ly, "q_inv"]
            y1.append(float(v1.iloc[0]) if len(v1) else np.nan)
            yN.append(float(vN.iloc[0]) if len(vN) else np.nan)
        x = np.arange(len(layers))
        ax.plot(x, y1, marker="o", linewidth=2, label=f"{cond} (ep1)")
        ax.plot(x, yN, marker="s", linewidth=2, label=f"{cond} (final)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("q_inv")
    ax.set_title("q_inv by layer — epoch 1 vs final (all conditions)")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    path = out_dir / "qinv_by_layer_ep1_vs_final_allconds.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

# ------------- per-condition q_inv calc -------------
def compute_qinv_table_for_condition(label: str, pathA: str, pathB: str,
                                     apply_bn: bool, use_nm: bool, logger) -> pd.DataFrame:
    A = load_spin_pkl(pathA)
    B = load_spin_pkl(pathB)
    layers_A = detect_layers(A)
    layers_B = detect_layers(B)
    common_layers = sorted(set(layers_A) & set(layers_B))
    if not common_layers:
        raise RuntimeError(f"[{label}] no common layers between A and B")
    sample_key = common_layers[0]
    labels_A = get_epoch_labels_for(A, sample_key)
    labels_B = get_epoch_labels_for(B, sample_key)
    setA, setB = set(labels_A), set(labels_B)
    matched_labels = sorted(setA & setB)
    if not matched_labels:
        raise RuntimeError(f"[{label}] no overlapping epoch labels between A and B")
    idxA = {lab: i for i, lab in enumerate(labels_A)}
    idxB = {lab: i for i, lab in enumerate(labels_B)}
    # baseline M
    EA, MA = A[sample_key].shape[0], A[sample_key].shape[1]
    EB, MB = B[sample_key].shape[0], B[sample_key].shape[1]
    M0 = min(MA, MB)
    logger.info(f"[{label}] layers={len(common_layers)} aligned_epochs={len(matched_labels)} example M≈{M0}")
    rows = []
    for layer in common_layers:
        arrA = A[layer]; arrB = B[layer]
        for lab in matched_labels:
            eA = idxA[lab]; eB = idxB[lab]
            Ae = np.array(arrA[eA], copy=False)
            Be = np.array(arrB[eB], copy=False)
            Mm = min(Ae.shape[0], Be.shape[0], M0)
            if Ae.shape[0] != Mm: Ae = Ae[:Mm]
            if Be.shape[0] != Mm: Be = Be[:Mm]
            if apply_bn:
                Ae = batch_norm_np(Ae)
                Be = batch_norm_np(Be)
            q_inv, q2_ab, q2_aa, q2_bb = calc_qinv_epoch(Ae, Be, use_nm=use_nm)
            rows.append({
                "cond": label,
                "layer": layer,
                "epoch": lab,
                "M": Mm,
                "q_inv": q_inv,
                "q2_ab": q2_ab,
                "q2_aa": q2_aa,
                "q2_bb": q2_bb,
            })
    df = pd.DataFrame(rows).sort_values(["cond","layer","epoch"]).reset_index(drop=True)
    return df

# ------------- main -------------
def main():
    args = build_args()
    logger = setup_logger(args.log_level)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # parse conditions
    cond_specs = []
    for s in args.cond:
        parts = s.split("|")
        if len(parts) != 3:
            raise SystemExit(f"--cond must be 'LABEL|/path/A.pkl|/path/B.pkl' but got: {s}")
        label, a, b = parts[0].strip(), parts[1].strip(), parts[2].strip()
        cond_specs.append((label, a, b))
    if len(cond_specs) > 4:
        raise SystemExit("At most 4 conditions are supported.")
    # compute
    all_tables = []
    for label, pa, pb in cond_specs:
        logger.info(f"=== condition: {label} ===")
        logger.info(f"A: {pa}")
        logger.info(f"B: {pb}")
        df = compute_qinv_table_for_condition(label, pa, pb, args.apply_bn, args.use_nm_correction, logger)
        all_tables.append(df)
    df_all = pd.concat(all_tables, ignore_index=True)
    # save tables
    df_all.to_csv(out_dir / "qinv_all_conditions.csv", index=False)
    joblib.dump({"config": vars(args), "table": df_all}, out_dir / "qinv_all_conditions.joblib")
    # plots
    _ = plot_epochs_per_layer_allconds(df_all, out_dir, skip_epoch0=args.skip_epoch0)
    both_path = plot_by_layer_epoch1_vs_final_allconds(df_all, out_dir)
    logger.info(f"Saved plots into: {out_dir}")
    logger.info(f"Summary fig: {both_path}")
    logger.info("DONE")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
