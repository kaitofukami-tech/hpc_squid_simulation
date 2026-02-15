#!/usr/bin/env python3

import argparse
import os

import numpy as np


def load_mnist(path: str):
    with np.load(path) as f:
        x_train = f["x_train"]
        y_train = f["y_train"]
        x_test = f["x_test"]
        y_test = f["y_test"]
    return x_train, y_train, x_test, y_test


def save_label_subset(out_dir: str, label: int, x_train, y_train, x_test, y_test):
    idx_train = np.where(y_train == label)[0]
    idx_test = np.where(y_test == label)[0]
    x_train_sub = x_train[idx_train]
    y_train_sub = y_train[idx_train]
    x_test_sub = x_test[idx_test]
    y_test_sub = y_test[idx_test]
    out_path = os.path.join(out_dir, f"mnist_label{label}.npz")
    np.savez_compressed(
        out_path,
        x_train=x_train_sub,
        y_train=y_train_sub,
        x_test=x_test_sub,
        y_test=y_test_sub,
    )
    return out_path, len(idx_train), len(idx_test)


def main():
    parser = argparse.ArgumentParser("Split MNIST npz into per-label datasets (0-9).")
    parser.add_argument(
        "--input",
        type=str,
        default="/sqfs/home/${USER_ID}/workspace/gmlp_project/data/mnist.npz",
        help="Path to the source MNIST npz.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/sqfs/home/${USER_ID}/workspace/gmlp_project/data/mnist_by_label",
        help="Directory to write per-label npz files.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    x_train, y_train, x_test, y_test = load_mnist(args.input)

    for label in range(10):
        out_path, n_train, n_test = save_label_subset(
            args.output_dir, label, x_train, y_train, x_test, y_test
        )
        print(
            f"label={label}: train={n_train} test={n_test} -> {out_path}"
        )


if __name__ == "__main__":
    main()
