"""Op-count + roofline model for the ternary-vs-codebook pivot (avenue H).

Part (b) of the pivot verdict: since Lloyd k-means owns the ~2.2 bpw
fidelity Pareto point (see sweep.py), the case for ternary schemes must
rest on *inference cost*, not SQNR. This module quantifies that case in
two layers:

1. DECODE ROOFLINE (the dominant effect on Apple Silicon). Decode is
   memory-bandwidth-bound: tok/s ~= bandwidth / bytes_per_token. Weight
   bytes are set by bpw, so a ternary scheme at 1.71-1.84 bpw moves
   ~17-22% fewer bytes per token than k-means-q8 at 2.188 bpw and gets a
   proportionally higher decode ceiling -- independent of op counts.

2. PER-WEIGHT OP COUNTS (second-order for decode, first-order in
   compute-bound regimes like prefill/large-batch). Ternary MAC =
   signed add (or skip when the code is zero) + one scale multiply per
   group. int2 codebook MAC = index lookup + fp multiply-accumulate, or
   the histogram trick real kernels use (one add per weight into a
   per-centroid bin, then a few ops per group).

Honest limits, stated up front:
- This is a MODEL, not a benchmark. It predicts ceilings and op ratios;
  real kernels add dequant overhead, thread sync, and paging effects, and
  achieve a fraction of roofline. Nothing here is a tok/s claim until it
  is measured on the Mac (roadmap backlog).
- The energy proxy (fp mul ~= 3.5x an fp add) is a textbook ALU ratio,
  not measured on M1. It is used only to collapse adds/muls/lookups into
  one comparable number, never as a joules claim.
"""

import numpy as np

# --- machine model ----------------------------------------------------------
# M1 Pro unified-memory bandwidth is Apple-published (200 GB/s). No peak-
# FLOPS constant is provided: is_bandwidth_bound() takes it as an explicit
# argument so no dubious number is baked in.
M1_PRO_MEM_BW_GBS = 200.0

# Rough energy proxy: one fp multiply costs ~3.5 fp adds on a typical ALU.
ALU_MUL_COST_IN_ADDS = 3.5

# --- target model -----------------------------------------------------------
N_PARAMS_70B = 70e9  # dense 70B-class target


# --- memory model -----------------------------------------------------------

def weight_bytes(n_params: float, bpw: float) -> float:
    """Weight footprint in bytes for n_params at bpw bits/param."""
    return n_params * bpw / 8.0


def kv_cache_bytes(n_layers: int = 80, n_kv_heads: int = 8,
                   head_dim: int = 128, ctx: int = 4096,
                   bytes_per_elem: int = 2) -> float:
    """KV-cache bytes. Defaults are Llama-70B-shaped (80 layers, GQA-8)
    at 4k context in fp16 -> ~1.34 GB, matching the roadmap's RAM math."""
    return 2.0 * n_layers * n_kv_heads * head_dim * ctx * bytes_per_elem


def decode_roofline_tps(bandwidth_gbs: float, weight_B: float,
                        kv_B: float = 0.0, act_B: float = 0.0,
                        efficiency: float = 1.0) -> float:
    """Optimistic decode ceiling in tok/s.

    Assumes every byte is streamed exactly once per token and the kernel
    sustains `efficiency` of peak bandwidth. Real kernels land below this;
    use efficiency < 1 (e.g. 0.5-0.7, typical for llama.cpp decode) for a
    planning number.
    """
    total_B = weight_B + kv_B + act_B
    if total_B <= 0:
        raise ValueError("bytes per token must be positive")
    if not 0.0 < efficiency <= 1.0:
        raise ValueError("efficiency must be in (0, 1]")
    return efficiency * bandwidth_gbs * 1e9 / total_B


def is_bandwidth_bound(flops_per_byte: float, peak_flops: float,
                       bandwidth_gbs: float) -> bool:
    """Roofline test: True when arithmetic intensity is below the machine
    balance (peak_flops / bandwidth), i.e. memory bandwidth -- not ALU
    throughput -- sets the speed limit."""
    return flops_per_byte < peak_flops / (bandwidth_gbs * 1e9)


# --- per-weight op models ---------------------------------------------------

def ternary_opcount(sparsity: float, n_scales: int = 1,
                    group_size: int = 128, n_outliers: int = 0) -> dict:
    """Per-weight op model for ternary matmul: y += s * sum(+-x).

    - Nonzero ternary weights cost one signed add each; exact zeros are
      skipped (sparse gather), so adds/weight = 1 - sparsity.
    - The group scale(s) cost n_scales multiplies per group, amortized.
    - Retained outliers (fp16) cost a full MAC each, amortized.
    Returns dict with adds / muls / lookups per weight.
    """
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must be in [0, 1]")
    adds = (1.0 - sparsity) + n_outliers / group_size
    muls = n_scales / group_size + n_outliers / group_size
    return {"adds": adds, "muls": muls, "lookups": 0.0}


def codebook_opcount(group_size: int = 128, n_centroids: int = 4,
                     n_scales: int = 1, method: str = "histogram") -> dict:
    """Per-weight op model for int2 codebook matmul (int2_kmeans family).

    naive:     per weight: index lookup -> fp centroid -> 1 mul + 1 add.
    histogram: what real kernels do. Per weight one add into a per-centroid
               bin counter; per group, n_centroids (centroid x count) muls +
               n_centroids adds + scale muls. Lloyd centroids are never
               exactly zero, so nothing is skipped.
    """
    if method == "naive":
        overhead = (n_scales + 1) / group_size  # scale + codebook reads
        return {"adds": 1.0 + overhead, "muls": 1.0 + overhead,
                "lookups": 1.0}
    if method == "histogram":
        return {"adds": 1.0 + n_centroids / group_size,
                "muls": (n_centroids + n_scales) / group_size,
                "lookups": 1.0}
    raise ValueError(f"unknown method {method!r}")


def equiv_adds(opcount: dict) -> float:
    """Collapse an op-count dict to one energy-proportional number per
    weight: adds + 3.5*muls + lookups (lookups counted as ~1 add each)."""
    return (opcount["adds"]
            + ALU_MUL_COST_IN_ADDS * opcount["muls"]
            + opcount["lookups"])


def measured_sparsity(result) -> float:
    """Fraction of exact-zero codes in a ternary-family QuantResult -- the
    fraction of weights a sparse ternary kernel skips. (For ternary_outlier
    the outlier positions hold code 0 and are counted here; their real MAC
    cost is modeled separately via n_outliers.)"""
    return float(np.mean(result.codes == 0))


def side_fractions(result) -> dict:
    """Per-code fractions of a ternary-family QuantResult.

    A dual-scale sparse kernel accumulates the two nonzero sides under
    separate scales (y += s_pos*sum(x|code=+1) + s_neg*sum(x|code=-1)), so
    the honest op input is the per-side profile, not just the total zero
    rate. Returns {"pos", "neg", "zero"} fractions that sum to 1.0; on a
    skewed tensor the pos/neg split visibly differs (the single-scale
    case keeps it symmetric), which is exactly the op-profile shift the
    dual opcount re-run needs to measure.
    """
    codes = np.asarray(result.codes)
    if not set(np.unique(codes)) <= {-1, 0, 1}:
        raise ValueError("side_fractions needs ternary codes in {-1, 0, 1}")
    n = codes.size
    return {"pos": float(np.sum(codes == 1) / n),
            "neg": float(np.sum(codes == -1) / n),
            "zero": float(np.sum(codes == 0) / n)}


# --- scheme-level report ----------------------------------------------------

def scheme_report(name: str, *, bpw: float, group_size: int = 128,
                  kind: str = "ternary", sparsity: float = 0.0,
                  n_scales: int = 1, n_outliers: int = 0,
                  method: str = "histogram",
                  n_params: float = N_PARAMS_70B,
                  kv_B: float = None,
                  bandwidth_gbs: float = M1_PRO_MEM_BW_GBS,
                  efficiency: float = 1.0) -> dict:
    """One-row quantitative summary for a quant scheme at 70B scale.

    kind="ternary" uses ternary_opcount (needs measured sparsity);
    kind="codebook" uses codebook_opcount. Returns weight footprint,
    roofline decode ceiling, op counts, and energy-proxy cost.
    """
    if kv_B is None:
        kv_B = kv_cache_bytes()
    w_B = weight_bytes(n_params, bpw)
    if kind == "ternary":
        oc = ternary_opcount(sparsity, n_scales, group_size, n_outliers)
    elif kind == "codebook":
        oc = codebook_opcount(group_size, n_centroids=4,
                              n_scales=n_scales, method=method)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    flops_pw = oc["adds"] + oc["muls"]  # lookups are not FLOPs
    return {
        "scheme": name,
        "bpw": bpw,
        "weight_GB": w_B / 1e9,
        "roofline_tps": decode_roofline_tps(bandwidth_gbs, w_B, kv_B,
                                            efficiency=efficiency),
        "adds_per_w": oc["adds"],
        "muls_per_w": oc["muls"],
        "lookups_per_w": oc["lookups"],
        "equiv_adds_per_w": equiv_adds(oc),
        "flops_per_byte": flops_pw / (bpw / 8.0),
    }


def print_report(rows: list) -> None:
    """Pretty-print a list of scheme_report() dicts."""
    hdr = (f"{'scheme':<22}{'bpw':>7}{'GB':>8}{'t/s ceil':>10}"
           f"{'adds/w':>8}{'muls/w':>8}{'eq.adds/w':>10}")
    print(hdr)
    for r in rows:
        print(f"{r['scheme']:<22}{r['bpw']:>7.3f}{r['weight_GB']:>8.2f}"
              f"{r['roofline_tps']:>10.1f}{r['adds_per_w']:>8.3f}"
              f"{r['muls_per_w']:>8.4f}{r['equiv_adds_per_w']:>10.3f}")
