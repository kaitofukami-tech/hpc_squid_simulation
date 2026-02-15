#!/usr/bin/env python3

import argparse
import json
import os
from typing import Dict, List

import numpy as np


def summarize_array(name: str, arr: np.ndarray) -> Dict:
    return {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def summarize_labels(labels: np.ndarray) -> List[Dict]:
    unique, counts = np.unique(labels, return_counts=True)
    out = []
    total = labels.size
    for label, count in zip(unique, counts):
        out.append(
            {
                "label": int(label),
                "count": int(count),
                "ratio": float(count / total),
            }
        )
    return out


def describe_label_subset(label: int, images: np.ndarray, labels: np.ndarray, max_samples: int):
    idx = np.where(labels == label)[0]
    if idx.size == 0:
        print(f"Label {label} not found.")
        return
    selected = images[idx]
    print(f"🔸 Label {label}: samples={idx.size}")
    print(
        json.dumps(
            {
                "pixels_min": float(selected.min()),
                "pixels_max": float(selected.max()),
                "pixels_mean": float(selected.mean()),
                "pixels_std": float(selected.std()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    head = idx[:max_samples].tolist()
    print(f"  first_indices: {head}")


def main():
    parser = argparse.ArgumentParser("Inspect MNIST npz contents.")
    parser.add_argument(
        "--input",
        type=str,
        default="/sqfs/home/${USER_ID}/workspace/gmlp_project/data/mnist.npz",
        help="Path to the MNIST npz file.",
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Print a small sample of label indices (train/test).",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=None,
        help="Inspect a specific label (0-9).",
    )
    parser.add_argument(
        "--label-samples",
        type=int,
        default=10,
        help="Number of sample indices to show when --label is set.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    with np.load(args.input) as data:
        keys = sorted(data.keys())
        print("📁 Keys:", keys)
        summaries = []
        for k in keys:
            arr = data[k]
            summaries.append(summarize_array(k, arr))
        print("🔍 Array summary:")
        print(json.dumps(summaries, ensure_ascii=False, indent=2))

        if "y_train" in data:
            print("📊 Train label distribution:")
            print(json.dumps(summarize_labels(data["y_train"]), ensure_ascii=False, indent=2))
        if "y_test" in data:
            print("📊 Test label distribution:")
            print(json.dumps(summarize_labels(data["y_test"]), ensure_ascii=False, indent=2))

        if args.show_samples:
            if "y_train" in data:
                print("y_train sample (first 20):", data["y_train"][:20].tolist())
            if "y_test" in data:
                print("y_test sample (first 20):", data["y_test"][:20].tolist())

        if args.label is not None:
            if not (0 <= args.label <= 9):
                raise ValueError("--label must be in [0,9].")
            if "x_train" in data and "y_train" in data:
                describe_label_subset(args.label, data["x_train"], data["y_train"], args.label_samples)
            if "x_test" in data and "y_test" in data:
                describe_label_subset(args.label, data["x_test"], data["y_test"], args.label_samples)


if __name__ == "__main__":
    main()
