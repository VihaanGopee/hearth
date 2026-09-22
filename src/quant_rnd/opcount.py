"""Op-count + roofline model for the ternary-vs-codebook pivot (avenue H).

Part (b) of the pivot verdict: since Lloyd k-means owns the ~2.2 bpw
fidelity Pareto point (see sweep.py), the case for ternary schemes must
rest on *inference cost*, not SQNR. This module quantifies that case in
two layers:

1. DECODE ROOFLINE (the dominant effect on Apple Silicon). Decode is
   memory-bandwidth-bound: tok/s ~= bandwidth / bytes_per_token. Weight
   bytes are set by the *packable* storage rate, NOT the entropy bpw:
   a ternary scheme at entropy 1.710 bpw packs to 2 bits/code + fp16
   scale per group = 2.125 storage bpw at g128, so it moves ~3% (not
   ~22%) fewer bytes per token than k-means-q8 at 2.375 bpw. The
   entropy bpw stays the right rate for SQNR/perplexity
   matched-bitrate comparisons; scheme_report takes an optional
   storage_bpw (defaults to bpw) so both rates travel together.

2. PER-WEIGHT OP COUNTS (second-order for decode, first-order in
   compute-bound regimes like prefill/large-batch). Ternary MAC =
   signed add (or skip when the code is zero) + one scale multiply per
   group. int2 codebook MAC = index lookup + fp multiply-accumulate, or
   the histogram trick real kernels use (one add per weight into a
   per-centroid bin, then a few ops per group).

3. PREFILL ROOFLINE (prefill_roofline_tps). Prefill reuses every weight
   `prompt_len` times, so arithmetic intensity grows with prompt length
   and the regime flips from bandwidth-bound (decode) to compute-bound.
   In that regime the ternary op-count advantage stops being second-order
   and sets the ceiling ratio directly.

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


# --- prefill (compute-bound) roofline ---------------------------------------
# Prefill streams the weights ONCE for the whole prompt while every weight
# is reused prompt_len times, so arithmetic intensity ~= prompt_len times
# the decode intensity and the regime flips to compute-bound. The attention
# O(L^2) term IS modeled (4 * n_layers * n_q_heads * L^2 * head_dim FLOPs,
# MAC counted as 2 FLOPs) -- at 32k+ context it dominates the matmuls.
# Pass include_attention=False to recover the matmul-only model.

# Activation traffic per token per layer, in units of d_model elements:
# read input + write output + FFN intermediate traffic. This is an
# ESTIMATE (real kernels fuse and re-tile); it only needs to be roughly
# right because the regime verdict at 4k is compute-bound by 100x+.
_PREFILL_ACT_TRAFFIC_ELEMS = 6.0


def prefill_roofline_tps(n_params: float, bpw: float, opcount: dict,
                         prompt_len: int, bandwidth_gbs: float,
                         peak_flops: float, efficiency: float = 1.0,
                         n_layers: int = 80, d_model: int = 8192,
                         n_q_heads: int = 64, n_kv_heads: int = 8,
                         head_dim: int = 128,
                         bytes_per_elem: int = 2,
                         include_attention: bool = True) -> dict:
    """Optimistic prefill ceiling in tok/s for one prompt of prompt_len.

    Model: weights are streamed once (weight_bytes), activations move an
    estimated 6*d_model elements per token per layer, and the KV write for
    the prompt is kv_cache_bytes(ctx=prompt_len). FLOPs are counted as
    adds + muls per weight (same convention as scheme_report; lookups are
    not FLOPs). Attention adds 4 * n_layers * n_q_heads * prompt_len^2 *
    head_dim FLOPs (Q@K^T + attn@V, MAC = 2 FLOPs), plus a KV read equal
    in size to the KV write; the scores-matrix traffic itself is assumed
    fused (not modeled). Ceiling = prompt_len / max(bytes/bandwidth,
    flops/peak). Pass include_attention=False to get the matmul-only
    model (useful for isolating the quant-scheme ceiling ratio).

    peak_flops is an explicit argument (no baked-in constant): pass the
    published peak for the engine you model, e.g. the Apple-published
    5.2 TFLOPS FP32 for the M1 Pro GPU.

    Honest caveats: adds are counted as 1 FLOP against an FMA-counted
    peak, so this is CONSERVATIVE for add-dominated ternary kernels --
    on FMA hardware an add-only kernel can approach 2x this ceiling, but
    that headroom is stated, not folded in. Real kernels land below the
    ceiling (efficiency < 1); this is a ratio tool, not a tok/s claim.
    """
    if not isinstance(prompt_len, (int, np.integer)) or prompt_len < 1:
        raise ValueError("prompt_len must be a positive integer")
    if bandwidth_gbs <= 0:
        raise ValueError("bandwidth_gbs must be positive")
    if peak_flops <= 0:
        raise ValueError("peak_flops must be positive")
    if not 0.0 < efficiency <= 1.0:
        raise ValueError("efficiency must be in (0, 1]")
    if n_q_heads < 1:
        raise ValueError("n_q_heads must be a positive integer")
    flops_per_token = n_params * (opcount["adds"] + opcount["muls"])
    matmul_flops = flops_per_token * prompt_len
    attention_flops = 0.0
    kv_read_B = 0.0
    if include_attention:
        attention_flops = (4.0 * n_layers * n_q_heads
                           * prompt_len * prompt_len * head_dim)
        kv_read_B = kv_cache_bytes(n_layers=n_layers, n_kv_heads=n_kv_heads,
                                   head_dim=head_dim, ctx=prompt_len,
                                   bytes_per_elem=bytes_per_elem)
    total_flops = matmul_flops + attention_flops
    weight_B = weight_bytes(n_params, bpw)
    act_B = (prompt_len * n_layers * d_model * bytes_per_elem
             * _PREFILL_ACT_TRAFFIC_ELEMS)
    kv_write_B = kv_cache_bytes(n_layers=n_layers, n_kv_heads=n_kv_heads,
                                head_dim=head_dim, ctx=prompt_len,
                                bytes_per_elem=bytes_per_elem)
    total_B = weight_B + act_B + kv_write_B + kv_read_B
    arith_intensity = total_flops / total_B
    t_compute = total_flops / peak_flops
    t_bandwidth = total_B / (bandwidth_gbs * 1e9)
    return {
        "total_flops": total_flops,
        "attention_flops": attention_flops,
        "bytes_moved": total_B,
        "arith_intensity_flops_per_byte": arith_intensity,
        "bandwidth_bound": is_bandwidth_bound(arith_intensity, peak_flops,
                                             bandwidth_gbs),
        "ceil_tps": efficiency * prompt_len / max(t_compute, t_bandwidth),
    }


# --- per-weight op models ---------------------------------------------------

def ternary_storage_bpw(group_size: int = 128, n_scales: int = 1) -> float:
    """Packable storage rate for a fitted-ternary scheme, bits/param.

    log2(3) ~ 1.585 is the *entropy* rate -- the right denominator for
    SQNR/perplexity matched-bitrate comparisons -- but no byte-aligned
    packing stores fractional bits. The packable layout (2-bit codes,
    4 per byte, + one fp16 scale per group; see
    research/ternary_kernel_design.md) is 2.0 + 16*n_scales/group_size
    bits/param: 2.125 at g128 single-scale, 2.25 dual-scale.
    Pass the result as scheme_report's storage_bpw so decode ceilings
    use real bytes while "bpw" keeps the entropy comparison rate.
    """
    if group_size < 1:
        raise ValueError("group_size must be a positive integer")
    if n_scales < 1:
        raise ValueError("n_scales must be a positive integer")
    return 2.0 + (16.0 * n_scales) / group_size


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
                  storage_bpw: float = None,
                  n_params: float = N_PARAMS_70B,
                  kv_B: float = None,
                  bandwidth_gbs: float = M1_PRO_MEM_BW_GBS,
                  efficiency: float = 1.0) -> dict:
    """One-row quantitative summary for a quant scheme at 70B scale.

    kind="ternary" uses ternary_opcount (needs measured sparsity);
    kind="codebook" uses codebook_opcount. Returns weight footprint,
    roofline decode ceiling, op counts, and energy-proxy cost.

    bpw is the entropy/comparison rate (used for SQNR/perplexity
    matched-bitrate comparisons). storage_bpw is the packable on-disk
    rate that actually sets weight_GB and the decode ceiling; it
    defaults to bpw (correct for codebook schemes, whose bpw already
    counts the stored codebook). For the ternary family pass
    ternary_storage_bpw(group_size, n_scales) -- entropy bpw overstates
    the decode edge by ~0.25x if used as the byte rate.
    """
    if kv_B is None:
        kv_B = kv_cache_bytes()
    sbpw = bpw if storage_bpw is None else storage_bpw
    w_B = weight_bytes(n_params, sbpw)
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
        "storage_bpw": sbpw,
        "weight_GB": w_B / 1e9,
        "roofline_tps": decode_roofline_tps(bandwidth_gbs, w_B, kv_B,
                                            efficiency=efficiency),
        "adds_per_w": oc["adds"],
        "muls_per_w": oc["muls"],
        "lookups_per_w": oc["lookups"],
        "equiv_adds_per_w": equiv_adds(oc),
        "flops_per_byte": flops_pw / (sbpw / 8.0),
    }


def print_report(rows: list) -> None:
    """Pretty-print a list of scheme_report() dicts."""
    hdr = (f"{'scheme':<22}{'bpw':>7}{'stor':>6}{'GB':>8}{'t/s ceil':>10}"
           f"{'adds/w':>8}{'muls/w':>8}{'eq.adds/w':>10}")
    print(hdr)
    for r in rows:
        print(f"{r['scheme']:<22}{r['bpw']:>7.3f}{r['storage_bpw']:>6.3f}"
              f"{r['weight_GB']:>8.2f}"
              f"{r['roofline_tps']:>10.1f}{r['adds_per_w']:>8.3f}"
              f"{r['muls_per_w']:>8.4f}{r['equiv_adds_per_w']:>10.3f}")


def prefill_crossover_L(op_a: dict, bpw_a: float, op_b: dict, bpw_b: float,
                        *, prompt_lens=(4096, 8192, 16384, 32768, 65536,
                                        131072),
                        threshold: float = 1.2,
                        **roofline_kwargs) -> dict:
    """Sweep prefill ceilings over prompt lengths and pin where scheme A's
    advantage over scheme B compresses below `threshold`.

    Motivation: in the compute-bound prefill regime the ceiling ratio of
    two quant schemes is the FLOP-per-weight ratio (peak and efficiency
    cancel). The scheme-INDEPENDENT attention O(L^2) term swamps that
    difference as L grows, so the ratio falls toward 1. This function
    answers "at what prompt length does the prefill compute case for A
    over B compress below <threshold>x?".

    op_a / bpw_a are scheme A's per-weight op-count dict and bits/param
    (A is the scheme expected to be faster, so ratios sit above 1);
    op_b / bpw_b likewise for B. roofline_kwargs are forwarded to
    prefill_roofline_tps (must include bandwidth_gbs and peak_flops).

    Returns a dict with the swept lens, per-L ceilings and ratios, and
    "crossover_L": the first prompt length where the ratio drops below
    threshold, or None if it never does within the swept range (plus
    "below_at_start" True when it is already below at the first length).
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    lens = list(prompt_lens)
    if not lens:
        raise ValueError("prompt_lens must be non-empty")
    if any(not isinstance(L, (int, np.integer)) or L < 1 for L in lens):
        raise ValueError("prompt_lens must be positive integers")
    if any(b <= a for a, b in zip(lens, lens[1:])):
        raise ValueError("prompt_lens must be strictly increasing")
    ratios, ceil_a, ceil_b = [], [], []
    for L in lens:
        ca = prefill_roofline_tps(prompt_len=int(L), opcount=op_a,
                                  bpw=bpw_a, **roofline_kwargs)["ceil_tps"]
        cb = prefill_roofline_tps(prompt_len=int(L), opcount=op_b,
                                  bpw=bpw_b, **roofline_kwargs)["ceil_tps"]
        ceil_a.append(ca)
        ceil_b.append(cb)
        ratios.append(ca / cb)
    below_at_start = ratios[0] < threshold
    crossover_L = None
    for L, r in zip(lens, ratios):
        if r < threshold:
            crossover_L = int(L)
            break
    return {
        "prompt_lens": [int(L) for L in lens],
        "ceil_a_tps": ceil_a,
        "ceil_b_tps": ceil_b,
        "ratios": ratios,
        "threshold": threshold,
        "below_at_start": below_at_start,
        "crossover_L": crossover_L,
    }
