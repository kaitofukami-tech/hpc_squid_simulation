#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recompute intermediate spins for saved MLP checkpoints on a new dataset.

The script reloads the model hyper-parameters from the original spin
metadata (produced by mlp_diff_model.py), iterates over the stored
checkpoints, and measures the requested intermediate features against
an arbitrary Fashion-MNIST compatible .npz dataset.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch


def _default_project_roots(script_path: Path) -> List[Path]:
    candidates = [
        script_path.parent.parent / "gmlp_project",
        Path("/sqfs/home/${USER_ID}/workspace/gmlp_project"),
    ]
    seen = set()
    ordered: List[Path] = []
    for p in candidates:
        if p not in seen and p.exists():
            ordered.append(p)
            seen.add(p)
    return ordered


def _resolve_project_root(project_root: str | None, script_path: Path) -> Path:
    if project_root:
        root = Path(project_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"--project-root {root} does not exist")
        return root
    candidates = _default_project_roots(script_path)
    if not candidates:
        raise FileNotFoundError(
            "Could not locate gmlp_project. Please pass --project-root explicitly."
        )
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Recompute MLP intermediate spins from saved checkpoints"
    )
    parser.add_argument(
        "--spin-pkl",
        required=True,
        help="Path to the original spin pickle (e.g. mlp_spinA_*.pkl)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to Fashion-MNIST compatible .npz containing x_train/y_train style arrays",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination pickle to store newly measured spins",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Override path to gmlp_project (default: auto-detect)",
    )
    parser.add_argument(
        "--tag",
        choices=["A", "B"],
        default=None,
        help="Explicitly choose model tag (defaults to tag stored in spin metadata)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="*",
        default=None,
        help="Subset of epochs to process (default: all epochs in metadata)",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help="Optional directory overriding checkpoint file location. "
             "Useful if checkpoint paths have moved. "
             "Files are resolved by basename within this directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size when feeding measurement samples.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples to measure (default: use full dataset).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=2025,
        help="Random seed when subsampling measurement samples.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to also write logs (overwrites each run).",
    )
    return parser.parse_args()


def setup_logger(level: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("recompute_spins_mlp")
    logger.setLevel(getattr(logging, level))
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # Reset existing handlers to avoid duplicate logs when reusing the logger.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def _load_spin_meta(path: Path) -> Dict:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if "meta" not in payload:
        raise KeyError(f"{path} does not contain 'meta'.")
    return payload["meta"]


def _resolve_checkpoints(
    checkpoints: Sequence[Dict], override_root: Path | None
) -> List[Dict]:
    resolved = []
    for item in checkpoints:
        c_path = Path(item["path"])
        if c_path.exists():
            resolved.append({"epoch": item["epoch"], "path": c_path})
            continue
        if override_root:
            alt = override_root / c_path.name
            if alt.exists():
                resolved.append({"epoch": item["epoch"], "path": alt})
                continue
        raise FileNotFoundError(
            f"Checkpoint for epoch {item['epoch']} not found at {c_path}."
        )
    return resolved


def _select_epochs(
    checkpoints: Sequence[Dict], epochs: Sequence[int] | None
) -> List[Dict]:
    if not epochs:
        return list(checkpoints)
    target = set(epochs)
    selected = [c for c in checkpoints if c["epoch"] in target]
    missing = target - {c["epoch"] for c in selected}
    if missing:
        missing_str = ", ".join(str(m) for m in sorted(missing))
        raise ValueError(f"Requested epochs not found in metadata: {missing_str}")
    return selected


def _prepare_measurement_samples(
    X: torch.Tensor,
    sample_size: int | None,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray]:
    total = X.shape[0]
    if sample_size is None or sample_size >= total:
        indices = np.arange(total)
        return X.clone(), indices
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive.")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(total, size=sample_size, replace=False))
    return X[indices].clone(), indices


def _extract_state_dict(raw: object) -> Mapping[str, torch.Tensor]:
    """
    Normalize torch.load outputs into a state-dict-like mapping.

    Supports plain OrderedDict checkpoints as well as dict containers that wrap
    the state dict under common keys such as 'state_dict' or 'model_state_dict'.
    """
    if isinstance(raw, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            candidate = raw.get(key)  # type: ignore[arg-type]
            if isinstance(candidate, Mapping):
                return candidate  # type: ignore[return-value]
        return raw  # type: ignore[return-value]
    raise TypeError(f"Unsupported checkpoint format: {type(raw)!r}")


def _infer_model_config(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, int]:
    """
    Infer MLP architectural parameters from a checkpoint state dict.
    """
    try:
        input_proj_weight = state_dict["input_proj.weight"]
    except KeyError as exc:
        raise KeyError("Checkpoint missing 'input_proj.weight' needed to infer d_model.") from exc
    
    d_model = int(input_proj_weight.shape[0])
    input_dim = int(input_proj_weight.shape[1])
    
    # Infer num_blocks
    block_indices: List[int] = []
    for key in state_dict.keys():
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.append(int(parts[1]))
    if not block_indices:
        raise ValueError("No blocks.* entries found in checkpoint; cannot infer num_blocks.")
    first_block_idx = min(block_indices)
    num_blocks = max(block_indices) + 1

    # Infer d_ffn from the first block
    fc1_weight_key = f"blocks.{first_block_idx}.fc1.weight"
    try:
        fc1_weight = state_dict[fc1_weight_key]
    except KeyError as exc:
        raise KeyError(f"Checkpoint missing '{fc1_weight_key}' needed to infer d_ffn.") from exc
    d_ffn = int(fc1_weight.shape[0])

    classifier_weight = state_dict.get("classifier.weight")
    num_classes = int(classifier_weight.shape[0]) if classifier_weight is not None else 10

    return {
        "input_dim": input_dim,
        "d_model": d_model,
        "d_ffn": d_ffn,
        "num_blocks": num_blocks,
        "num_classes": num_classes,
    }


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_file).expanduser().resolve() if args.log_file else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    logger = setup_logger(args.log_level, str(log_path) if log_path else None)

    project_root = _resolve_project_root(args.project_root, script_path)
    scripts_dir = project_root / "scripts"
    if not scripts_dir.exists():
        raise FileNotFoundError(f"Scripts directory not found: {scripts_dir}")
    sys.path.insert(0, str(scripts_dir))

    # Lazy imports after path update.
    from mlp_diff_model import (  # type: ignore
        MLP,
        load_npz_fashion,
        measure_spins,
        atomic_save_pickle,
    )

    spin_meta = _load_spin_meta(Path(args.spin_pkl))
    tag = args.tag or spin_meta.get("tag")
    if tag not in ("A", "B"):
        raise ValueError("Unable to determine model tag (A/B). Use --tag.")

    layer_list: List[str] = list(spin_meta.get("record_layers", []))
    if not layer_list:
        raise ValueError("record_layers missing in spin metadata.")

    checkpoints_meta = spin_meta.get("checkpoints")
    if not checkpoints_meta:
        raise ValueError("Checkpoint metadata not found in spin pickle.")

    override_root = Path(args.checkpoint_root).resolve() if args.checkpoint_root else None
    if override_root and not override_root.exists():
        raise FileNotFoundError(f"--checkpoint-root {override_root} does not exist.")

    checkpoints_filtered = _select_epochs(checkpoints_meta, args.epochs)
    checkpoints = _resolve_checkpoints(checkpoints_filtered, override_root)
    checkpoints = sorted(checkpoints, key=lambda c: c["epoch"])
    if not checkpoints:
        raise RuntimeError("No checkpoints available after filtering.")

    logger.debug(f"Loading checkpoint metadata from {checkpoints[0]['path']}")
    config_raw = torch.load(checkpoints[0]["path"], map_location="cpu")
    config_state = _extract_state_dict(config_raw)
    ckpt_config = _infer_model_config(config_state)
    del config_raw

    def _warn_mismatch(field: str, derived: int) -> None:
        meta_val = spin_meta.get(field)
        if meta_val is None:
            return
        try:
            meta_int = int(meta_val)
        except (TypeError, ValueError):
            return
        if meta_int != derived:
            logger.warning(
                "Spin metadata %s=%s differs from checkpoint-derived value %s. Using checkpoint value.",
                field,
                meta_val,
                derived,
            )

    d_model = int(ckpt_config["d_model"])
    d_ffn = int(ckpt_config["d_ffn"])
    num_blocks = int(ckpt_config["num_blocks"])
    num_classes = int(ckpt_config.get("num_classes", spin_meta.get("num_classes", 10)))
    input_dim = int(ckpt_config.get("input_dim", 784))
    dropout = float(spin_meta.get("dropout", 0.1))

    for field, value in (
        ("d_model", d_model),
        ("d_ffn", d_ffn),
        ("num_blocks", num_blocks),
        ("num_classes", num_classes),
    ):
        _warn_mismatch(field, value)

    logger.info(
        "Model configuration inferred from checkpoint: d_model=%d d_ffn=%d num_blocks=%d num_classes=%d "
        "(dropout=%.3f from metadata)",
        d_model,
        d_ffn,
        num_blocks,
        num_classes,
        dropout,
    )

    logger.info(f"Using project root: {project_root}")
    logger.info(f"Loaded spin meta (tag={tag}) with {len(checkpoints)} checkpoints.")

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with np.load(dataset_path) as f:
        keys = set(f.keys())
    if "noisy_train" in keys:
        from mlp_denoise_diff_model import load_denoise_data  # type: ignore
        train_ds, _val_ds = load_denoise_data(str(dataset_path), logger)
        X_all = train_ds.tensors[0]
        logger.info("Detected denoise NPZ; using noisy_train for spin measurement.")
    else:
        X_all, _ = load_npz_fashion(str(dataset_path), logger)
    X_meas, meas_indices = _prepare_measurement_samples(
        X_all, args.sample_size, args.sample_seed
    )
    logger.info(
        f"Measurement tensor: shape={tuple(X_meas.shape)} from indices "
        f"{meas_indices[:10].tolist()}..."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(
        input_dim=input_dim,
        d_model=d_model,
        d_ffn=d_ffn,
        num_blocks=num_blocks,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    layer_spins: Dict[str, List[np.ndarray]] = {ln: [] for ln in layer_list}
    used_epochs: List[int] = []
    used_paths: List[str] = []

    for item in checkpoints:
        epoch = int(item["epoch"])
        ckpt_path = Path(item["path"])
        logger.info(f"Processing epoch {epoch} checkpoint: {ckpt_path}")
        raw_state = torch.load(ckpt_path, map_location="cpu")
        state_dict = _extract_state_dict(raw_state)
        model.load_state_dict(state_dict, strict=True)
        spins_dict = measure_spins(
            model, layer_list, X_meas, batch=min(args.batch_size, X_meas.shape[0]), logger=logger
        )
        for ln, tensor in spins_dict.items():
            layer_spins[ln].append(tensor.numpy())
        used_epochs.append(epoch)
        used_paths.append(str(ckpt_path))
        del raw_state, state_dict, spins_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not used_epochs:
        raise RuntimeError("No checkpoints processed. Nothing to save.")

    out_meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_spin": str(Path(args.spin_pkl).resolve()),
        "tag": tag,
        "epochs": used_epochs,
        "checkpoint_paths": used_paths,
        "layer_list": layer_list,
        "d_model": int(d_model),
        "d_ffn": int(d_ffn),
        "num_blocks": int(num_blocks),
        "num_classes": int(num_classes),
        "dropout": float(dropout),
        "batch_size": int(args.batch_size),
        "model_config_source": "checkpoint_state_dict",
        "dataset": {
            "path": str(dataset_path),
            "sample_size": int(X_meas.shape[0]),
            "indices": meas_indices.tolist(),
        },
        "original_meta": spin_meta,
    }

    out_payload: Dict[str, object] = {"meta": out_meta}
    for ln, tensors in layer_spins.items():
        out_payload[ln] = np.stack(tensors, axis=0)

    out_path = Path(args.output).resolve()
    os.makedirs(out_path.parent, exist_ok=True)
    atomic_save_pickle(out_payload, str(out_path), logger)
    logger.info(f"Saved recomputed spins to {out_path}")


if __name__ == "__main__":
    main()
