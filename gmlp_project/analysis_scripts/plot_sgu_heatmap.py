#!/usr/bin/env python3

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Render SGU token-mixing weights from a gMLP checkpoint as heatmaps"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a single *.pt checkpoint file (deprecated; prefer --checkpoints)",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="One or more checkpoint files to process",
    )
    parser.add_argument(
        "--block-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of block indices (e.g., 0 3 7). Defaults to all blocks found.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./sgu_heatmaps",
        help="Directory to save the heatmap PNGs",
    )
    parser.add_argument("--cmap", type=str, default="coolwarm", help="Matplotlib colormap")
    parser.add_argument(
        "--clamp",
        type=float,
        nargs=2,
        default=None,
        metavar=("VMIN", "VMAX"),
        help="Optional vmin/vmax for imshow. Defaults to data range.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution for the saved figures",
    )
    parser.add_argument(
        "--plot-kind",
        choices=("matrix", "tokens", "both"),
        default="matrix",
        help="Select between a single matrix heatmap, per-token spatial maps, or both",
    )
    parser.add_argument(
        "--token-grid",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Explicit spatial grid of tokens (e.g., 14 14). Defaults to square root of token count.",
    )
    parser.add_argument(
        "--show-bias",
        action="store_true",
        help="Print the associated SGU bias vector statistics for reference",
    )
    return parser.parse_args()


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model", "module"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)}")


def find_sgu_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    pat = re.compile(r"gmlp_blocks\.(\d+)\.sgu\.proj\.weight")
    weights: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        m = pat.fullmatch(key)
        if not m:
            continue
        idx = int(m.group(1))
        bias_key = f"gmlp_blocks.{idx}.sgu.proj.bias"
        bias = state_dict.get(bias_key)
        weights[idx] = (tensor.detach().cpu(), bias.detach().cpu() if bias is not None else None)
    if not weights:
        raise KeyError("No SGU projection weights found in checkpoint.")
    return weights


def choose_blocks(all_blocks: Iterable[int], subset: Optional[Iterable[int]]) -> List[int]:
    available = sorted(set(all_blocks))
    if subset is None:
        return available
    subset_list = list(subset)
    if len(subset_list) == 0:
        return available
    selected = sorted(set(subset_list))
    missing = [i for i in selected if i not in available]
    if missing:
        raise ValueError(f"Requested block(s) {missing} not found in checkpoint (available: {available})")
    return selected


def render_heatmap(
    matrix: np.ndarray,
    *,
    out_path: Path,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    title: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap=cmap, aspect="equal", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_xlabel("input token index")
    ax.set_ylabel("output token index")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|gate weight|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _auto_token_grid(num_tokens: int) -> Tuple[int, int]:
    root = int(round(num_tokens**0.5))
    if root * root == num_tokens:
        return root, root
    # Fall back to closest factor pair
    factors: List[Tuple[int, int]] = []
    for i in range(1, int(num_tokens**0.5) + 1):
        if num_tokens % i == 0:
            factors.append((i, num_tokens // i))
    if factors:
        return min(factors, key=lambda pair: abs(pair[0] - pair[1]))
    return (1, num_tokens)


def infer_token_grid(num_tokens: int, override: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if override:
        h, w = override
        if h <= 0 or w <= 0:
            raise ValueError("--token-grid must be positive integers.")
        if h * w != num_tokens:
            auto_h, auto_w = _auto_token_grid(num_tokens)
            print(
                f"[warn] --token-grid {h}x{w} mismatch for {num_tokens} tokens; "
                f"falling back to inferred grid {auto_h}x{auto_w}."
            )
            return auto_h, auto_w
        return h, w
    return _auto_token_grid(num_tokens)


def render_token_dependency_grid(
    matrix: np.ndarray,
    *,
    out_path: Path,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    title: str,
    dpi: int,
    token_grid: Tuple[int, int],
) -> None:
    num_out, num_in = matrix.shape
    grid_h, grid_w = token_grid
    expected_tokens = grid_h * grid_w
    if num_out != expected_tokens or num_in != expected_tokens:
        raise ValueError(
            f"Token grid {grid_h}x{grid_w} expects {expected_tokens} tokens, "
            f"but got weight matrix of shape {matrix.shape}."
        )
    fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w, grid_h))
    axes = np.atleast_2d(axes)
    first_im = None
    for out_idx in range(num_out):
        ax = axes[out_idx // grid_w, out_idx % grid_w]
        spatial_map = matrix[out_idx].reshape(grid_h, grid_w)
        im = ax.imshow(spatial_map, cmap=cmap, aspect="equal", origin="lower", vmin=vmin, vmax=vmax)
        if first_im is None:
            first_im = im
        ax.axis("off")
    if first_im is not None:
        cbar = fig.colorbar(
            first_im,
            ax=axes.ravel().tolist(),
            fraction=0.02,
            pad=0.01,
        )
        cbar.set_label("|gate weight|")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def collect_checkpoint_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[str] = []
    if args.checkpoint:
        paths.append(args.checkpoint)
    if args.checkpoints:
        paths.extend(args.checkpoints)
    if not paths:
        raise ValueError("Specify at least one checkpoint via --checkpoint or --checkpoints.")
    unique: List[str] = []
    for p in paths:
        if p not in unique:
            unique.append(p)
    return [Path(p).expanduser().resolve() for p in unique]


def derive_label(path: Path) -> str:
    parent = path.parent.name
    stem = path.stem
    if parent:
        return f"{parent}_{stem}"
    return stem


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = collect_checkpoint_paths(args)
    for ckpt_path in checkpoint_paths:
        print(f"=== Processing checkpoint: {ckpt_path} ===")
        state_dict = load_state_dict(str(ckpt_path))
        sgu_weights = find_sgu_weights(state_dict)
        block_indices = choose_blocks(sgu_weights.keys(), args.block_indices)
        ckpt_label = derive_label(ckpt_path)
        ckpt_out_dir = out_dir / ckpt_label
        ckpt_out_dir.mkdir(parents=True, exist_ok=True)
        for idx in block_indices:
            weight, bias = sgu_weights[idx]
            mat = weight.numpy()
            mat_abs = np.abs(mat)  # color by magnitude, ignoring sign
            if mat.shape[0] != mat.shape[1]:
                print(f"[warn] block {idx}: weight is {mat.shape}, plotting anyway.")
            if args.clamp:
                vmin, vmax = (abs(args.clamp[0]), abs(args.clamp[1]))
                if vmin > vmax:
                    vmin, vmax = vmax, vmin
            else:
                vmin, vmax = (mat_abs.min(), mat_abs.max())
            if args.plot_kind in ("matrix", "both"):
                out_path = ckpt_out_dir / f"sgu_block{idx:02d}_heatmap.png"
                render_heatmap(
                    mat_abs,
                    out_path=out_path,
                    cmap=args.cmap,
                    vmin=vmin,
                    vmax=vmax,
                    title=f"{ckpt_label} | SGU Block {idx}",
                    dpi=args.dpi,
                )
                print(
                    f"Saved {out_path} (abs range [{mat_abs.min():.4f}, {mat_abs.max():.4f}]; "
                    f"raw range [{mat.min():.4f}, {mat.max():.4f}])"
                )
            if args.plot_kind in ("tokens", "both"):
                token_grid = infer_token_grid(mat.shape[0], tuple(args.token_grid) if args.token_grid else None)
                out_path = ckpt_out_dir / f"sgu_block{idx:02d}_token_grid.png"
                render_token_dependency_grid(
                    mat_abs,
                    out_path=out_path,
                    cmap=args.cmap,
                    vmin=vmin,
                    vmax=vmax,
                    title=f"{ckpt_label} | SGU Block {idx} (per-token)",
                    dpi=args.dpi,
                    token_grid=token_grid,
                )
                print(f"Saved {out_path} (token grid {token_grid[0]}x{token_grid[1]})")
            if args.show_bias and bias is not None:
                b = bias.numpy()
                print(f"  bias stats -> mean: {b.mean():.4f}, std: {b.std():.4f}, min: {b.min():.4f}, max: {b.max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
