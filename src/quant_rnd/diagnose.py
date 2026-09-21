"""Decompose the +1.4 dB Lloyd-fit ternary gain (diagnostic).

Compares symmetric ternary_uniform (absmean heuristic, one shot) against
ternary_lloyd (iterative fit) at the SAME 1.710 bpw, and splits the total
gain into two interpretable pieces:

  1. scale refit: keep the heuristic assignment (thresholds +/-absmean/2)
     but refit the scale L2-optimally for it == Lloyd iteration 1;
  2. threshold adaptation: later iterations re-derive the thresholds from
     the refit scale == iteration 1 -> convergence.

A threshold-grid ablation (scale fixed at absmean, threshold grid-searched
independently) cross-checks the threshold-adaptation component: if the
only thing that matters is where the thresholds sit, the grid should
recover most of part 2 with the heuristic scale untouched.

Run:  python3 -m src.quant_rnd.diagnose [--seed N] [--skew S]
"""
import argparse

import numpy as np

from .bench import synthetic_weights, sqnr_db
from .schemes import (
    GROUP_SIZE,
    QuantResult,
    TERNARY_PAYLOAD_BPW,
    _groups,
    _scale_overhead,
    _ternary_lloyd_fit,
    quantize_ternary_lloyd,
    quantize_ternary_uniform,
)


def lloyd_step_sqnr(w: np.ndarray, group_size: int = GROUP_SIZE,
                    n_iter: int = 20) -> list:
    """Tensor SQNR after each Lloyd-fit step (step 0 = heuristic init).

    Step 1 is exactly the "scale refit only" ablation: the assignment is
    still made with the heuristic absmean thresholds, the scale is the
    L2-optimal refit for that assignment. Step 1 -> last is the
    threshold-adaptation gain.
    """
    wp, n_groups, n = _groups(w, group_size)
    histories = []
    for gi in range(n_groups):
        h: list = []
        _ternary_lloyd_fit(wp[gi], dual=False, n_iter=n_iter, history=h)
        histories.append(h)
    n_steps = max(len(h) for h in histories)
    bpw = TERNARY_PAYLOAD_BPW + _scale_overhead(1, group_size)
    sqnrs = []
    for step in range(n_steps):
        scales = np.zeros((n_groups, 1), dtype=np.float32)
        flat_codes = []
        for gi, h in enumerate(histories):
            s_pos, _s_neg = h[min(step, len(h) - 1)]
            s = max(s_pos, 1e-12)  # same guard as the scheme
            scales[gi, 0] = np.float32(s)
            g = wp[gi]
            seg = np.zeros(g.shape[0], dtype=np.int8)
            seg[g > 0.5 * s] = 1
            seg[g < -0.5 * s] = -1
            flat_codes.append(seg)
        codes = np.concatenate(flat_codes)[:n]
        res = QuantResult("ternary_lloyd", codes, scales, bpw,
                          group_size=group_size)
        sqnrs.append(sqnr_db(w, res.reconstruct()))
    return sqnrs


def threshold_grid_ablation(w: np.ndarray, group_size: int = GROUP_SIZE,
                            n_grid: int = 97) -> tuple:
    """Fix the scale at absmean; grid-search the threshold per tensor.

    Threshold t = alpha * absmean_group; the grid includes alpha=0.5, so
    the best result is provably >= ternary_uniform. Returns
    (best_sqnr_db, best_alpha).
    """
    wp, n_groups, n = _groups(w, group_size)
    s0 = np.maximum(np.mean(np.abs(wp), axis=1), 1e-12).astype(np.float64)
    bpw = TERNARY_PAYLOAD_BPW + _scale_overhead(1, group_size)
    alphas = np.logspace(-2.0, 0.7, n_grid)
    alphas[np.argmin(np.abs(alphas - 0.5))] = 0.5  # include uniform's point
    best = (-np.inf, None)
    for alpha in alphas:
        flat_codes = []
        for gi in range(n_groups):
            g = wp[gi]
            t = alpha * s0[gi]
            seg = np.zeros(g.shape[0], dtype=np.int8)
            seg[g > t] = 1
            seg[g < -t] = -1
            flat_codes.append(seg)
        codes = np.concatenate(flat_codes)[:n]
        res = QuantResult("ternary_lloyd", codes,
                          s0.astype(np.float32).reshape(n_groups, 1), bpw,
                          group_size=group_size)
        s = sqnr_db(w, res.reconstruct())
        if s > best[0]:
            best = (s, float(alpha))
    return best


def diagnostic_report(seed: int = 7, n_groups: int = 64,
                      skew: float = 0.0, n_iter: int = 20) -> dict:
    """Full decomposition on one synthetic tensor; returns the numbers."""
    rng = np.random.default_rng(seed)
    w = synthetic_weights(rng, n_groups, GROUP_SIZE, skew=skew)
    uniform = sqnr_db(w, quantize_ternary_uniform(w).reconstruct())
    lloyd = sqnr_db(w, quantize_ternary_lloyd(w, n_iter=n_iter).reconstruct())
    steps = lloyd_step_sqnr(w, n_iter=n_iter)
    grid_sqnr, grid_alpha = threshold_grid_ablation(w)
    return {
        "seed": seed,
        "skew": skew,
        "uniform_db": uniform,
        "step_sqnr_db": steps,
        "lloyd_db": lloyd,
        "scale_refit_gain_db": steps[1] - steps[0] if len(steps) > 1 else 0.0,
        "threshold_adapt_gain_db": steps[-1] - steps[1] if len(steps) > 1 else 0.0,
        "total_gain_db": steps[-1] - steps[0],
        "grid_db": grid_sqnr,
        "grid_alpha": grid_alpha,
        "n_steps": len(steps),
    }


def print_report(r: dict) -> None:
    steps = r["step_sqnr_db"]
    print(f"Lloyd ternary-fit decomposition (seed {r['seed']}, "
          f"skew {r['skew']}, 1.710 bpw, {r['n_steps']} recorded steps)")
    print(f"  ternary_uniform (heuristic absmean)      {r['uniform_db']:7.2f} dB")
    print(f"  step 0 = heuristic init                  {steps[0]:7.2f} dB")
    if len(steps) > 1:
        print(f"  step 1 = scale refit only                {steps[1]:7.2f} dB"
              f"   (+{r['scale_refit_gain_db']:.2f})")
        print(f"  converged = + threshold adaptation       {steps[-1]:7.2f} dB"
              f"   (+{r['threshold_adapt_gain_db']:.2f})")
    print(f"  ternary_lloyd (scheme, cross-check)      {r['lloyd_db']:7.2f} dB")
    print(f"  total Lloyd gain over heuristic: +{r['total_gain_db']:.2f} dB")
    print(f"  threshold-grid ablation (scale=absmean): {r['grid_db']:7.2f} dB"
          f" @ alpha={r['grid_alpha']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Lloyd ternary gain diagnostic")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--groups", type=int, default=64)
    ap.add_argument("--skew", type=float, default=0.0)
    args = ap.parse_args()
    print_report(diagnostic_report(seed=args.seed, n_groups=args.groups,
                                   skew=args.skew))


if __name__ == "__main__":
    main()
