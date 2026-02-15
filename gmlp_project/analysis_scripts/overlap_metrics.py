#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replica correlation metrics (SquareOverlap) for intra/inter datasets.
Supports NumPy and PyTorch backends.
"""

import argparse
from typing import Any, Dict, Optional

try:
    from typing import Literal
except ImportError:  # python<3.8
    try:
        from typing_extensions import Literal  # type: ignore
    except ImportError:  # fallback stub
        class _LiteralStub:
            def __getitem__(self, _item):
                return Any

        Literal = _LiteralStub()  # type: ignore

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch optional
    torch = None

Mode = Literal["intra", "inter"]
Backend = Literal["numpy", "torch"]


def _detect_backend(x: Any) -> Optional[Backend]:
    if torch is not None and torch.is_tensor(x):
        return "torch"
    if isinstance(x, np.ndarray):
        return "numpy"
    return None


def _as_backend(x: Any, backend: Backend):
    if backend == "numpy":
        if torch is not None and torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    if backend == "torch":
        if torch is None:
            raise RuntimeError("PyTorch backend requested but torch is not installed")
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    raise ValueError(f"Unsupported backend {backend}")


def _ensure_float(x, backend: Backend):
    if backend == "numpy":
        return np.asarray(x, dtype=np.float64 if x.dtype == np.float64 else np.float32)
    dtype = torch.float64 if torch is not None and x.dtype == torch.float64 else torch.float32
    return x.to(dtype)


def _center(X, backend: Backend):
    if backend == "numpy":
        return X - X.mean(axis=0, keepdims=True)
    return X - X.mean(dim=0, keepdim=True)


def _matmul(X, Y, backend: Backend):
    if backend == "numpy":
        return X.T @ Y
    return X.transpose(0, 1).matmul(Y)


def _sum_prod(A, B, backend: Backend) -> float:
    if backend == "numpy":
        return float(np.sum(A * B))
    return float((A * B).sum().item())


def overlap_metrics(
    mode: Mode,
    Sa,
    Sb,
    Sa2=None,
    Sb2=None,
    *,
    center: bool = False,
    backend: Optional[Backend] = None,
    use_nm: bool = True,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Compute intra/inter replica overlap metrics.
    """
    if backend is None:
        backend = (
            _detect_backend(Sa)
            or _detect_backend(Sb)
            or _detect_backend(Sa2)
            or "numpy"
        )

    Sa = _ensure_float(_as_backend(Sa, backend), backend)
    Sb = _ensure_float(_as_backend(Sb, backend), backend)

    if Sa.shape != Sb.shape or Sa.ndim != 2:
        raise ValueError("Sa and Sb must have shape (M,N)")
    M, N = Sa.shape

    if center:
        Sa = _center(Sa, backend)
        Sb = _center(Sb, backend)

    if mode == "inter":
        if Sa2 is None or Sb2 is None:
            raise ValueError("mode='inter' requires Sa2/Sb2")
        Sa2 = _ensure_float(_as_backend(Sa2, backend), backend)
        Sb2 = _ensure_float(_as_backend(Sb2, backend), backend)
        if Sa2.shape != (M, N) or Sb2.shape != (M, N):
            raise ValueError("Sa2/Sb2 must match Sa/Sb shapes")
        if center:
            Sa2 = _center(Sa2, backend)
            Sb2 = _center(Sb2, backend)

    def _Q(X, Y):
        return _matmul(X, Y, backend) / float(M)

    res: Dict[str, Any] = {}
    if mode == "intra":
        Q_ab = _Q(Sa, Sb)
        Q_aa = _Q(Sa, Sa)
        Q_bb = _Q(Sb, Sb)

        def _sqnorm(Q):
            return _sum_prod(Q, Q, backend) / float(N)

        corr = float(N) / float(M)
        q2_ab_raw = _sqnorm(Q_ab)
        q2_aa_raw = _sqnorm(Q_aa)
        q2_bb_raw = _sqnorm(Q_bb)
        use_nm_applied = False
        if use_nm:
            q2_ab_nm = q2_ab_raw - corr
            q2_aa_nm = q2_aa_raw - corr
            q2_bb_nm = q2_bb_raw - corr
            if q2_ab_nm >= 0.0 and q2_aa_nm >= 0.0 and q2_bb_nm >= 0.0:
                q2_ab, q2_aa, q2_bb = q2_ab_nm, q2_aa_nm, q2_bb_nm
                use_nm_applied = True
            else:
                q2_ab, q2_aa, q2_bb = q2_ab_raw, q2_aa_raw, q2_bb_raw
        else:
            q2_ab, q2_aa, q2_bb = q2_ab_raw, q2_aa_raw, q2_bb_raw
        denom = (max(q2_aa, 0.0) * max(q2_bb, 0.0)) ** 0.5 + eps
        res.update(
            {
                "Q_ab": Q_ab,
                "q2_ab": q2_ab,
                "q2_aa": q2_aa,
                "q2_bb": q2_bb,
                "q_inv": q2_ab / denom,
                "use_nm_applied": use_nm_applied,
            }
        )
    elif mode == "inter":
        Q_ab1 = _Q(Sa, Sb)
        Q_ab2 = _Q(Sa2, Sb2)
        Q_aa1 = _Q(Sa, Sa)
        Q_aa2 = _Q(Sa2, Sa2)
        Q_bb1 = _Q(Sb, Sb)
        Q_bb2 = _Q(Sb2, Sb2)

        def _sum_norm(A, B):
            return _sum_prod(A, B, backend) / float(N)

        q2_ab = _sum_norm(Q_ab1, Q_ab2)
        q2_aa = _sum_norm(Q_aa1, Q_aa2)
        q2_bb = _sum_norm(Q_bb1, Q_bb2)
        denom = (max(q2_aa, 0.0) * max(q2_bb, 0.0)) ** 0.5 + eps
        res.update(
            {
                "Q_ab": Q_ab1,
                "Q_ab2": Q_ab2,
                "q2_ab": q2_ab,
                "q2_aa": q2_aa,
                "q2_bb": q2_bb,
                "q_inv": q2_ab / denom,
                "use_nm_applied": False,
            }
        )
    else:
        raise ValueError("mode must be 'intra' or 'inter'")

    return res


def overlap_metrics_numpy(*args, **kwargs):
    kwargs["backend"] = "numpy"
    return overlap_metrics(*args, **kwargs)


def overlap_metrics_torch(*args, **kwargs):
    if torch is None:
        raise RuntimeError("torch backend requested but torch is not available")
    kwargs["backend"] = "torch"
    return overlap_metrics(*args, **kwargs)


def _tiny_tests():
    np.random.seed(0)
    M, N = 2048, 64
    Sa = np.random.randn(M, N).astype(np.float32)
    Sb = Sa + 0.1 * np.random.randn(M, N).astype(np.float32)
    out_intra = overlap_metrics("intra", Sa, Sb, backend="numpy")
    assert 0.0 <= out_intra["q_inv"] <= 1.01

    Sa2 = Sa.copy()
    Sb2 = Sb.copy()
    out_inter = overlap_metrics("inter", Sa, Sb, Sa2, Sb2, backend="numpy")
    assert out_inter["q2_ab"] >= out_intra["q2_ab"]


def _demo():
    M, N = 4096, 256
    Sa1 = np.random.randn(M, N).astype(np.float32)
    Sb1 = Sa1 + 0.2 * np.random.randn(M, N).astype(np.float32)
    Sa2 = np.random.randn(M, N).astype(np.float32)
    Sb2 = Sa2 + 0.2 * np.random.randn(M, N).astype(np.float32)

    out_intra = overlap_metrics("intra", Sa1, Sb1, center=False, backend="numpy")
    print("intra q_inv:", out_intra["q_inv"], "q2_ab:", out_intra["q2_ab"])

    out_inter = overlap_metrics("inter", Sa1, Sb1, Sa2, Sb2, center=False, backend="numpy")
    print("inter q_inv:", out_inter["q_inv"], "q2_ab:", out_inter["q2_ab"])


def main():
    parser = argparse.ArgumentParser("Overlap metrics demo")
    parser.add_argument("--demo", action="store_true", help="Run random demo")
    args = parser.parse_args()

    if args.demo:
        _demo()
    else:
        _tiny_tests()
        print("tiny tests passed")


if __name__ == "__main__":
    main()
