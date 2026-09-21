"""Synthetic SQNR benchmark for the quant R&D prototypes.

Generates weight tensors with LLM-like statistics (Gaussian body, sparse
large outliers, optional skew), runs every scheme in SCHEMES, and reports
mean SQNR (dB) alongside effective bpw.

Run:  python3 -m src.quant_rnd.bench [--seed N] [--groups G]
"""
import argparse

import numpy as np

from .schemes import SCHEMES, GROUP_SIZE


def synthetic_weights(rng: np.random.Generator, n_groups: int,
                      group_size: int = GROUP_SIZE,
                      outlier_frac: float = 0.001,
                      outlier_scale: float = 8.0,
                      skew: float = 0.0) -> np.ndarray:
    """LLM-flavored synthetic weights.

    Body: N(0, 1). Outliers: outlier_frac of entries scaled by
    outlier_scale. skew shifts the body mean (tests asymmetric schemes).
    """
    n = n_groups * group_size
    w = rng.standard_normal(n).astype(np.float32)
    if outlier_frac > 0:
        m = max(1, int(n * outlier_frac))
        idx = rng.choice(n, size=m, replace=False)
        w[idx] *= outlier_scale
    if skew:
        w = w + skew
    return w


def sqnr_db(orig: np.ndarray, recon: np.ndarray) -> float:
    """Signal-to-quantization-noise ratio in dB (higher = better)."""
    orig = orig.astype(np.float64)
    recon = recon.astype(np.float64)
    signal = np.var(orig)
    noise = np.mean((orig - recon) ** 2)
    if noise <= 0:
        return float("inf")
    if signal <= 0:
        return float("-inf")
    return float(10.0 * np.log10(signal / noise))


def run_bench(seed: int = 7, n_groups: int = 64,
              outlier_frac: float = 0.001, outlier_scale: float = 8.0,
              skew: float = 0.0):
    """Quantize one synthetic tensor per scheme; return sorted results."""
    rng = np.random.default_rng(seed)
    w = synthetic_weights(rng, n_groups, GROUP_SIZE,
                          outlier_frac, outlier_scale, skew)
    results = []
    for name, fn in SCHEMES.items():
        q = fn(w)
        s = sqnr_db(w, q.reconstruct())
        results.append({"scheme": name, "sqnr_db": s, "bpw": q.bpw})
    results.sort(key=lambda r: r["sqnr_db"], reverse=True)
    return results


def print_table(results) -> None:
    print(f"{'scheme':<22}{'SQNR (dB)':>10}{'bpw':>8}")
    print("-" * 42)
    for r in results:
        print(f"{r['scheme']:<22}{r['sqnr_db']:>10.2f}{r['bpw']:>8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="quant_rnd synthetic benchmark")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--groups", type=int, default=64)
    ap.add_argument("--skew", type=float, default=0.0)
    args = ap.parse_args()
    print_table(run_bench(seed=args.seed, n_groups=args.groups, skew=args.skew))


if __name__ == "__main__":
    main()
