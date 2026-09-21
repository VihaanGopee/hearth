"""Pareto sweep: SQNR vs bpw over outlier / group-size hyperparameters.

Answers the backlog question "sweep outlier_frac / n_outliers for the
Pareto frontier (SQNR vs bpw) before picking a kernel target" and the
pivot sub-question: can any of our candidates beat Lloyd k-means below
~2.2 bpw on synthetic SQNR?

Swept configs:
- ternary_outlier (candidate B): n_outliers/group in {1, 2, 4, 8}
- int2_outlier_retain (baseline): outlier_frac in {0.001, 0.005, 0.01, 0.02}
- int2_kmeans / int2_kmeans_q8 (classical baselines): group_size in
  {64, 128, 256} (smaller groups = more codebook overhead = higher bpw)
- dual_scale_ternary (candidate A) and ternary_uniform: single points
  (no sweepable hyperparameter).

Run: python3 -m src.quant_rnd.sweep [--seed N] [--groups G]
"""
import argparse

import numpy as np

from .bench import sqnr_db, synthetic_weights
from .schemes import (GROUP_SIZE, quantize_dual_scale_ternary,
                      quantize_int2_kmeans, quantize_int2_kmeans_q8,
                      quantize_int2_outlier_retain, quantize_ternary_outlier,
                      quantize_ternary_uniform)

# (label, weight-array -> QuantResult) configs, in sweep order.
CONFIGS = []
for _n_out in (1, 2, 4, 8):
    CONFIGS.append((
        f"ternary_outlier n={_n_out}",
        lambda w, n=_n_out: quantize_ternary_outlier(w, n_outliers=n),
    ))
for _frac in (0.001, 0.005, 0.01, 0.02):
    CONFIGS.append((
        f"int2_outlier_retain f={_frac}",
        lambda w, f=_frac: quantize_int2_outlier_retain(w, outlier_frac=f),
    ))
for _g in (64, 128, 256):
    CONFIGS.append((
        f"int2_kmeans_q8 g={_g}",
        lambda w, g=_g: quantize_int2_kmeans_q8(w, group_size=g),
    ))
    CONFIGS.append((
        f"int2_kmeans g={_g}",
        lambda w, g=_g: quantize_int2_kmeans(w, group_size=g),
    ))
CONFIGS.append(("dual_scale_ternary", quantize_dual_scale_ternary))
CONFIGS.append(("ternary_uniform", quantize_ternary_uniform))


def pareto_frontier(points):
    """Return the Pareto-optimal subset of dicts with bpw/sqnr_db keys.

    Sorted by ascending bpw; a point is on the frontier if its SQNR beats
    every point at lower (or equal) bitrate. Pure function, unit-tested.
    """
    ordered = sorted(points, key=lambda p: (p["bpw"], -p["sqnr_db"]))
    frontier = []
    best = float("-inf")
    for p in ordered:
        if p["sqnr_db"] > best:
            frontier.append(p)
            best = p["sqnr_db"]
    return frontier


def run_sweep(seed: int = 7, n_groups: int = 64,
              outlier_frac: float = 0.001, outlier_scale: float = 8.0,
              skew: float = 0.0):
    """Run every swept config on one synthetic tensor; mark the frontier.

    Returns a list of dicts {label, sqnr_db, bpw, on_frontier}, sorted by
    ascending bpw.
    """
    rng = np.random.default_rng(seed)
    w = synthetic_weights(rng, n_groups, GROUP_SIZE,
                          outlier_frac, outlier_scale, skew)
    results = []
    for label, fn in CONFIGS:
        q = fn(w)
        results.append({
            "label": label,
            "sqnr_db": sqnr_db(w, q.reconstruct()),
            "bpw": q.bpw,
        })
    results.sort(key=lambda r: (r["bpw"], -r["sqnr_db"]))
    frontier_labels = {id(p) for p in pareto_frontier(results)}
    for r in results:
        r["on_frontier"] = id(r) in frontier_labels
    return results


def print_table(results) -> None:
    print(f"{'config':<30}{'SQNR (dB)':>10}{'bpw':>8}{'':>3}")
    print("-" * 53)
    for r in results:
        star = " *" if r["on_frontier"] else ""
        print(f"{r['label']:<30}{r['sqnr_db']:>10.2f}{r['bpw']:>8.3f}{star}")


def main() -> None:
    ap = argparse.ArgumentParser(description="quant_rnd Pareto sweep")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--groups", type=int, default=64)
    ap.add_argument("--skew", type=float, default=0.0)
    args = ap.parse_args()
    results = run_sweep(seed=args.seed, n_groups=args.groups, skew=args.skew)
    print_table(results)
    print("\n* = Pareto frontier point (best SQNR at or below its bitrate)")


if __name__ == "__main__":
    main()
