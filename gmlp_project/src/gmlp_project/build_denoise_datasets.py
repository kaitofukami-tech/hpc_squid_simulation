#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Create noisy-clean pairs for MNIST/Fashion-MNIST denoising experiments."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/sqfs/work/cm9029/${USER_ID}/gmlp_project/data"),
        help="Directory containing mnist.npz and fashion_mnist.npz",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=("mnist", "fashion_mnist"),
        help="Dataset basenames (without extension) to process.",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=(1.0, 3.0, 5.0, 10.0),
        help="Noise amplitudes to generate.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/sqfs/work/cm9029/${USER_ID}/gmlp_project/data/denoise"),
        help="Directory where noisy datasets will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed to keep noise generation reproducible.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing dataset file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs without writing files.",
    )
    return parser.parse_args()


def _first_available(data: np.lib.npyio.NpzFile, keys: Tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


def load_npz_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        x_train = _first_available(data, ("x_train", "X_train", "train_images"))
        y_train = _first_available(data, ("y_train", "train_labels"))
        x_test = _first_available(data, ("x_test", "X_test", "test_images"))
        y_test = _first_available(data, ("y_test", "test_labels"))
        if x_train is None or y_train is None:
            raise KeyError(f"{path} is missing train image/label arrays")
        if x_test is None or y_test is None:
            raise KeyError(f"{path} is missing test image/label arrays")
    def prep(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        if img.ndim == 3:
            img = img[:, None, :, :]
        return img
    return prep(x_train), prep(x_test), y_train.astype(np.int64), y_test.astype(np.int64)


def add_gaussian_noise(clean: np.ndarray, lam: float, rng: np.random.Generator) -> np.ndarray:
    sigma = lam / 255.0
    noise = rng.normal(loc=0.0, scale=sigma, size=clean.shape).astype(np.float32)
    noisy = np.clip(clean + noise, 0.0, 1.0)
    return noisy


def summarize(out_path: Path, clean_shape: Tuple[int, ...], lamb: float) -> Dict:
    return {
        "path": str(out_path),
        "lambda": lamb,
        "clean_shape": clean_shape,
    }


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict] = []

    for dataset_name in args.datasets:
        source_path = args.source_dir / f"{dataset_name}.npz"
        x_train, x_test, y_train, y_test = load_npz_dataset(source_path)
        clean_stats = dict(
            dataset=dataset_name,
            train_shape=x_train.shape,
            test_shape=x_test.shape,
        )
        print(f"[info] Loaded {dataset_name}: {clean_stats}")

        for lamb in args.lambdas:
            rng_seed = args.seed + hash((dataset_name, lamb)) % (2**31 - 1)
            rng = np.random.default_rng(rng_seed)
            noisy_train = add_gaussian_noise(x_train, lamb, rng)
            noisy_test = add_gaussian_noise(x_test, lamb, rng)

            out_path = args.output_root / f"{dataset_name}_lambda{int(lamb)}.npz"
            if out_path.exists() and not args.overwrite:
                raise FileExistsError(f"{out_path} exists (use --overwrite to replace)")
            summaries.append(summarize(out_path, x_train.shape, lamb))

            if args.dry_run:
                print(f"[dry-run] Would write {out_path}")
                continue

            np.savez_compressed(
                out_path,
                noisy_train=noisy_train,
                clean_train=x_train,
                noisy_test=noisy_test,
                clean_test=x_test,
                labels_train=y_train,
                labels_test=y_test,
                lambda_value=np.float32(lamb),
                source=str(source_path),
            )
            print(f"[ok] Saved {out_path} (lambda={lamb})")

    print("[summary]")
    for item in summaries:
        print(json.dumps(item))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
