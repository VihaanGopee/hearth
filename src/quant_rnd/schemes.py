"""Quantizer prototypes: honest baselines + two candidate schemes.

All quantizers are group-wise (default 128 weights/group), operate on a
flat float32 array, and return a QuantResult that can reconstruct() an
approximation of the input.

Baselines (approximate llama.cpp-style behavior, not bit-exact):
- ternary_uniform: T1_0-like symmetric ternary, absmean scale.
- int2_symmetric: plain 2-bit symmetric.
- int2_outlier_retain: 2-bit + top-k% magnitudes kept in fp16
  (mixed-precision flavor of the oQ/JANG idea).
- int2_kmeans: classical 1-D Lloyd 2-bit (4 fitted centroids per group) -
  the fair comparison naive symmetric int2 lacks.
- int2_kmeans_q8: same Lloyd fit, but the 4 centroids are stored in 8-bit
  (one fp16 scale per group) - the codebook trick real IQ quants use.
  Brings the classical baseline down to 2.375 bpw, near our candidates'
  ~2.06-2.24 bpw, for a (roughly) matched-bitrate comparison.

Candidates (ours, to be validated):
- dual_scale_ternary ("DST"): asymmetric ternary. Real weight tensors are
  often skewed (one-sided outliers); one symmetric scale wastes dynamic
  range. DST keeps the 1.58-bit ternary payload but uses separate positive
  and negative scales. Costs one extra fp16 scale per group.
- ternary_outlier ("T1+out"): ternary base + the n largest-magnitude
  weights per group stored exactly in fp16 (with 8-bit indices). Tests
  whether ternary + sparse outliers beats plain int2 at matched bitrate.
- ternary_lloyd / ternary_lloyd_ds ("Lloyd-fit ternary"): the sweep showed
  Lloyd *fitting* is doing the heavy lifting, not the codebook size, so
  this runs Lloyd with the codebook constrained to ternary {-s,0,+s}
  (symmetric) or dual-scale ternary {-s_neg,0,+s_pos}. Same 1.710 /
  1.835 bpw as ternary_uniform / DST, so the SQNR comparison against them
  is at exactly matched bitrate and isolates the value of fitting.
- ternary_1step ("1-step Lloyd"): the diagnostic (diagnose.py) showed the
  first Lloyd iteration (heuristic assignment -> one L2-optimal scale
  refit -> reassignment at the refit thresholds) captures ~85% of the
  ternary_lloyd win. This scheme is exactly that one iteration
  (_ternary_lloyd_fit with n_iter=1): O(1) extra cost, no convergence
  loop, identical 1.710 bpw. The practical encoder for fitted ternary if
  the ~85% capture holds.
- ternary_1step_ds ("few-step Lloyd", dual): the dual-scale twin of
  ternary_1step - dual=True, default n_iter=1 - identical 1.835 bpw to
  ternary_lloyd_ds. Tests whether few fit iterations also capture the
  dual-Lloyd win on skewed tensors.
- ternary_1step_sp ("sparse 1-step"): ternary_1step with a widened zero
  bin (thresh_factor=1.2, threshold-biased Lloyd): raises the measured
  zero-rate from ~0.41 to ~0.51 on real GPT-2 weights at matched
  1.710 bpw. The compute-side candidate: more zeros = more add-skips in
  a ternary kernel. VERDICT 2026-09-21: NEGATIVE - ppl 45242 vs 2764
  for the 1-step baseline (16x blowup) at only -0.05 dB SQNR. The
  fidelity cliff is razor-sharp; sparsity that SQNR can't see still
  kills the model. Kept as the cost-of-0.5-sparsity reference.
"""
from dataclasses import dataclass, field
from functools import partial

import numpy as np

GROUP_SIZE = 128
SCALE_BITS = 16  # fp16 scale(s) stored per group
INDEX_BITS = 8  # enough to index a 128-weight group
TERNARY_PAYLOAD_BPW = float(np.log2(3))  # 1.585, information-theoretic


@dataclass
class QuantResult:
    """Outcome of quantizing one flat weight array."""

    name: str
    codes: np.ndarray  # int8 codes, flat, same length as input
    scales: np.ndarray  # float32 per-group scale(s); shape (n_groups, n_scales)
    bpw: float  # effective bits/weight incl. all overhead
    group_size: int = GROUP_SIZE  # group size used at quantize time
    _outlier_vals: np.ndarray = field(default=None, repr=False)
    _outlier_idx: np.ndarray = field(default=None, repr=False)

    def reconstruct(self) -> np.ndarray:
        """Dequantize back to float32, same shape as the input.

        Vectorized over groups (fancy indexing on a group_id array):
        the naive per-group boolean-mask loop is O(n^2/group_size) and
        takes ~25 s per 2M params -- unusable at model scale. This is
        O(n) with identical values.
        """
        n = self.codes.shape[0]
        g = self.group_size
        # group_id max is (n-1)//g, always a valid row of self.scales.
        group_id = np.arange(n) // g
        c = self.codes.astype(np.float32)
        s = self.scales
        if self.name in _DUAL_SCALE_SCHEMES:
            rec = np.where(c > 0, s[group_id, 0],
                           np.where(c < 0, -s[group_id, 1], 0.0))
        elif self.name in _CODEBOOK_SCHEMES:
            # s holds the effective per-group codebook (fp16-fitted or
            # 8-bit-rounded centroids); codes index into it.
            rec = s[group_id, c.astype(np.int64)]
        else:
            rec = c * s[group_id, 0]
        out = np.ascontiguousarray(rec, dtype=np.float32)
        if self._outlier_vals is not None:
            out[self._outlier_idx] = self._outlier_vals
        return out


# Schemes whose `scales` hold an effective per-group codebook that the
# integer `codes` index into (rather than multiplicative scales).
_CODEBOOK_SCHEMES = frozenset({"int2_kmeans", "int2_kmeans_q8",
                               "int4_kmeans_q8"})

# Schemes whose `scales` hold (s_pos, s_neg): code +1 decodes as s_pos,
# code -1 as -s_neg.
_DUAL_SCALE_SCHEMES = frozenset({"dual_scale_ternary", "ternary_lloyd_ds",
                                 "ternary_1step_ds"})


def _groups(w: np.ndarray, group_size: int = GROUP_SIZE):
    n = w.shape[0]
    n_groups = (n + group_size - 1) // group_size
    pad = n_groups * group_size - n
    wp = np.pad(w.astype(np.float32), (0, pad)) if pad else w.astype(np.float32)
    return wp.reshape(n_groups, group_size), n_groups, n


def _scale_overhead(n_scales: int, group_size: int = GROUP_SIZE) -> float:
    return n_scales * SCALE_BITS / group_size


def quantize_ternary_uniform(w: np.ndarray, group_size: int = GROUP_SIZE) -> QuantResult:
    """T1_0-like: symmetric ternary, absmean scale (BitNet style)."""
    wp, n_groups, n = _groups(w, group_size)
    scales = np.mean(np.abs(wp), axis=1, keepdims=True).astype(np.float32)
    scales = np.maximum(scales, 1e-12)  # all-zero group guard
    codes = np.clip(np.round(wp / scales), -1, 1).astype(np.int8).ravel()[:n]
    bpw = TERNARY_PAYLOAD_BPW + _scale_overhead(1, group_size)
    return QuantResult("ternary_uniform", codes,
                       scales.reshape(n_groups, 1), bpw,
                       group_size=group_size)


def quantize_int2_symmetric(w: np.ndarray, group_size: int = GROUP_SIZE) -> QuantResult:
    """Plain 2-bit symmetric: 4 levels {-1.5,-0.5,0.5,1.5} * s."""
    wp, n_groups, n = _groups(w, group_size)
    amax = np.max(np.abs(wp), axis=1, keepdims=True).astype(np.float32)
    amax = np.maximum(amax, 1e-12)
    s = amax / 1.5
    codes = np.clip(np.round(wp / s + 0.5) - 0.5, -1.5, 1.5)
    # store doubled so codes are integers in {-3,-1,1,3}
    codes = (codes * 2).astype(np.int8).ravel()[:n]
    bpw = 2.0 + _scale_overhead(1, group_size)
    res = QuantResult("int2_symmetric", codes, (s / 2).reshape(n_groups, 1),
                      bpw, group_size=group_size)
    return res


def quantize_int8_uniform(w: np.ndarray, group_size: int = GROUP_SIZE) -> QuantResult:
    """Symmetric uniform int8: 255 levels in [-127, 127] * absmax/127.

    The q8 reference for the embedding-table probe (roadmap: is the
    2-bit embedding collapse specific to wte, and does an 8-bit table
    survive?). One fp16 scale per group; decode is the plain
    multiplicative path (name is registered in neither the codebook nor
    the dual-scale sets). Deterministic, O(n).
    """
    wp, n_groups, n = _groups(w, group_size)
    amax = np.max(np.abs(wp), axis=1, keepdims=True).astype(np.float32)
    amax = np.maximum(amax, 1e-12)  # all-zero group guard
    s = amax / 127.0
    codes = np.clip(np.round(wp / s), -127, 127).astype(np.int8).ravel()[:n]
    bpw = 8.0 + _scale_overhead(1, group_size)
    return QuantResult("int8_uniform", codes, s.reshape(n_groups, 1),
                       bpw, group_size=group_size)


def quantize_int4_uniform(w: np.ndarray, group_size: int = GROUP_SIZE) -> QuantResult:
    """Symmetric uniform int4: 15 levels in [-7, 7] * absmax/7.

    The mid probe for the embedding-table question (roadmap 2026-09-22):
    q8 on wte survives (ppl 54.76 vs 53.50 fp32) and 2-bit collapses
    (ppl inf) — is there a survivable bitrate between 2-bit and q8 for
    the tied head? One fp16 scale per group -> 4 + 16/group_size bpw.
    Decode is the plain multiplicative path (registered in neither the
    codebook nor the dual-scale sets). Deterministic, O(n).
    """
    wp, n_groups, n = _groups(w, group_size)
    amax = np.max(np.abs(wp), axis=1, keepdims=True).astype(np.float32)
    amax = np.maximum(amax, 1e-12)  # all-zero group guard
    s = amax / 7.0
    codes = np.clip(np.round(wp / s), -7, 7).astype(np.int8).ravel()[:n]
    bpw = 4.0 + _scale_overhead(1, group_size)
    return QuantResult("int4_uniform", codes, s.reshape(n_groups, 1),
                       bpw, group_size=group_size)


def quantize_int2_outlier_retain(w: np.ndarray, group_size: int = GROUP_SIZE,
                                 outlier_frac: float = 0.005) -> QuantResult:
    """2-bit + top outlier_frac magnitudes kept exactly in fp16."""
    wf = w.astype(np.float32)
    n = wf.shape[0]
    m = max(1, int(round(n * outlier_frac)))
    idx = np.argpartition(np.abs(wf), -m)[-m:]
    rest = wf.copy()
    rest[idx] = 0.0
    base = quantize_int2_symmetric(rest, group_size)
    k = m / n
    bpw = ((1 - k) * 2.0 + k * SCALE_BITS + k * INDEX_BITS
           + _scale_overhead(1, group_size))
    base.name = "int2_outlier_retain"
    base.bpw = bpw
    base._outlier_vals = wf[idx]
    base._outlier_idx = idx
    return base


def quantize_dual_scale_ternary(w: np.ndarray,
                                group_size: int = GROUP_SIZE) -> QuantResult:
    """Candidate A (ours): asymmetric ternary.

    Separate scales for the positive and negative sides, each fit to its
    own side's mean magnitude. On skewed groups (one-sided outliers) this
    recovers dynamic range that a symmetric scale wastes. Payload is
    still 1.58-bit ternary; overhead is one extra fp16 scale per group.
    """
    wp, n_groups, n = _groups(w, group_size)
    pos = np.where(wp > 0, wp, 0.0)
    neg = np.where(wp < 0, -wp, 0.0)
    n_pos = np.maximum((wp > 0).sum(axis=1, keepdims=True), 1)
    n_neg = np.maximum((wp < 0).sum(axis=1, keepdims=True), 1)
    s_pos = (pos.sum(axis=1, keepdims=True) / n_pos).astype(np.float32)
    s_neg = (neg.sum(axis=1, keepdims=True) / n_neg).astype(np.float32)
    s_pos = np.maximum(s_pos, 1e-12)
    s_neg = np.maximum(s_neg, 1e-12)
    codes = np.zeros_like(wp, dtype=np.int8)
    codes[wp > 0.5 * s_pos] = 1
    codes[wp < -0.5 * s_neg] = -1
    codes = codes.ravel()[:n]
    bpw = TERNARY_PAYLOAD_BPW + _scale_overhead(2, group_size)
    scales = np.concatenate([s_pos, s_neg], axis=1)
    return QuantResult("dual_scale_ternary", codes, scales, bpw,
                       group_size=group_size)


def quantize_ternary_outlier(w: np.ndarray, group_size: int = GROUP_SIZE,
                             n_outliers: int = 2) -> QuantResult:
    """Candidate B (ours): ternary + n largest magnitudes per group in fp16.

    Asks whether ternary + sparse exact outliers beats plain int2 at a
    matched bitrate (default n=2 -> ~2.09 bpw vs int2's 2.125 bpw).
    """
    wf = w.astype(np.float32)
    n = wf.shape[0]
    n_groups = (n + group_size - 1) // group_size
    idx_all, val_all = [], []
    for gi in range(n_groups):
        seg = wf[gi * group_size:(gi + 1) * group_size]
        if seg.size == 0:
            continue
        k = min(n_outliers, seg.size)
        local = np.argpartition(np.abs(seg), -k)[-k:]
        idx_all.append(gi * group_size + local)
        val_all.append(seg[local])
    idx = np.concatenate(idx_all)
    vals = np.concatenate(val_all)
    rest = wf.copy()
    rest[idx] = 0.0
    base = quantize_ternary_uniform(rest, group_size)
    k_frac = idx.shape[0] / n
    bpw = ((1 - k_frac) * TERNARY_PAYLOAD_BPW + k_frac * SCALE_BITS
           + k_frac * INDEX_BITS + _scale_overhead(1, group_size))
    base.name = "ternary_outlier"
    base.bpw = bpw
    base._outlier_vals = vals
    base._outlier_idx = idx
    return base


def _lloyd_1d(x: np.ndarray, k: int = 4, n_iter: int = 20,
              w: np.ndarray | None = None) -> np.ndarray:
    """Deterministic 1-D Lloyd's algorithm; returns k centroids.

    Quantile initialization keeps it deterministic (no random restarts);
    empty clusters keep their previous centroid.

    If `w` (per-element, non-negative) is given, assignment minimizes
    w_i * (x_i - c_j)^2 and the refit is the weighted mean -- the
    diagonal-Hessian reweighting used by the OBQ first slice (fisher.py):
    weights in high-energy input channels pull the codebook toward
    themselves. w=None runs the original unweighted code path verbatim
    (bit-identical to pre-fisher versions, pinned by test, so all
    previously published anchor numbers reproduce exactly).
    """
    if w is None:
        # Original unweighted path, kept verbatim: float32 .mean() refit
        # differs from a float64 weighted mean in the last ulp, which can
        # flip code assignments at boundaries.
        qs = (np.arange(k) + 0.5) / k
        cent = np.quantile(x.astype(np.float64), qs)
        for _ in range(n_iter):
            assign = np.abs(x[:, None] - cent[None, :]).argmin(axis=1)
            new = cent.copy()
            for j in range(k):
                m = assign == j
                if m.any():
                    new[j] = x[m].mean()
            if np.allclose(new, cent):
                break
            cent = new
        return cent.astype(np.float32)
    x = x.astype(np.float64)
    w = np.asarray(w, dtype=np.float64)
    if w.shape != x.shape:
        raise ValueError(f"weight shape {w.shape} != data shape {x.shape}")
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    qs = (np.arange(k) + 0.5) / k
    cent = np.quantile(x, qs)
    for _ in range(n_iter):
        assign = (w[:, None] * (x[:, None] - cent[None, :]) ** 2).argmin(axis=1)
        new = cent.copy()
        for j in range(k):
            m = assign == j
            sw = w[m].sum()
            if sw > 0:
                new[j] = (w[m] * x[m]).sum() / sw
        if np.allclose(new, cent):
            break
        cent = new
    return cent.astype(np.float32)


def quantize_int2_kmeans(w: np.ndarray, group_size: int = GROUP_SIZE,
                         n_iter: int = 20) -> QuantResult:
    """Baseline (classical): per-group Lloyd 2-bit, 4 fitted centroids.

    The fair classical comparison naive int2_symmetric lacks: instead of a
    fixed symmetric codebook stretched by the group amax (which outliers
    destroy), Lloyd fits 4 centroids to each group's actual distribution.
    Honest cost: 4 fp16 centroids per group -> 2.5 bpw, heavier than the
    ~2.06 bpw candidates. The bench prints bpw beside SQNR so comparisons
    stay honest about the bitrate gap.
    """
    wp, n_groups, n = _groups(w, group_size)
    centroids = np.stack([_lloyd_1d(g, n_iter=n_iter) for g in wp])
    codes = np.abs(wp[:, :, None]
                   - centroids[:, None, :]).argmin(axis=2)
    codes = codes.astype(np.int8).ravel()[:n]
    bpw = 2.0 + _scale_overhead(4, group_size)
    return QuantResult("int2_kmeans", codes, centroids, bpw,
                       group_size=group_size)


def _quantize_kmeans_q8(w: np.ndarray, group_size: int, n_iter: int,
                       sample_weight: np.ndarray | None, n_bits: int,
                       name: str, chunk_groups: int = 16384) -> QuantResult:
    """Lloyd k-means with an 8-bit-stored codebook, generalized over bit width.

    Per-group Lloyd fit with k = 2**n_bits centroids, stored in 8-bit
    (symmetric, one fp16 codebook scale per group) - the storage trick real
    codebook quants (llama.cpp IQ) use. Code assignment is against the
    stored (rounded) codebook, as an honest encoder would; the returned
    scales hold the DEQUANTIZED centroids so reconstruct() measures exactly
    what a real decoder sees. n_bits is 2 or 4 (the widths with published
    anchor numbers); anything else raises ValueError.

    `chunk_groups`: the code-assignment broadcast is (chunk, group, k)
    float32 - at k=16 on a 38.6M-param embedding table the unchunked
    (301k, 128, 16) temp is ~2.5 GB and OOMs this VM (2026-09-22). The
    per-group argmin is row-independent, so chunking is bit-identical;
    a test pins chunked == unchunked.
    """
    if n_bits not in (2, 4):
        raise ValueError(f"n_bits must be 2 or 4, got {n_bits}")
    k = 1 << n_bits
    wp, n_groups, n = _groups(w, group_size)
    if sample_weight is None:
        centroids = np.stack([_lloyd_1d(g, k=k, n_iter=n_iter) for g in wp])
    else:
        sw = np.asarray(sample_weight).ravel()
        if sw.shape[0] != n:
            raise ValueError(f"sample_weight length {sw.shape[0]} != "
                             f"weights length {n}")
        swp, _, _ = _groups(sw, group_size)
        centroids = np.stack([_lloyd_1d(g, k=k, n_iter=n_iter, w=sg)
                              for g, sg in zip(wp, swp)])
    # Symmetric 8-bit quantization of the per-group codebook.
    cmax = np.max(np.abs(centroids), axis=1, keepdims=True).astype(np.float32)
    cmax = np.maximum(cmax, 1e-12)  # all-zero group guard
    q8 = np.round(centroids / cmax * 127.0).astype(np.int8)
    deq = (q8.astype(np.float32) / 127.0 * cmax)
    code_chunks = []
    for s in range(0, n_groups, chunk_groups):
        e = min(s + chunk_groups, n_groups)
        code_chunks.append(
            np.abs(wp[s:e, :, None] - deq[s:e, None, :]).argmin(axis=2))
    codes = np.concatenate(code_chunks).astype(np.int8).ravel()[:n]
    bpw = float(n_bits) + k * 8 / group_size + _scale_overhead(1, group_size)
    return QuantResult(name, codes, deq, bpw, group_size=group_size)


def quantize_int2_kmeans_q8(w: np.ndarray, group_size: int = GROUP_SIZE,
                            n_iter: int = 20,
                            sample_weight: np.ndarray | None = None) -> QuantResult:
    """Matched-bitrate k-means: Lloyd 2-bit with an 8-bit codebook.

    Same per-group Lloyd fit as int2_kmeans, but the 4 fitted centroids are
    stored in 8-bit (symmetric, one fp16 scale per group) - the storage
    trick real codebook quants (llama.cpp IQ) use. This brings the
    classical baseline from 2.5 bpw down to 2.375 bpw, near our candidates'
    ~2.06-2.24 bpw, so the SQNR comparison is at (roughly) matched bitrate.

    The returned scales hold the DEQUANTIZED centroids, so reconstruct()
    measures exactly what a real decoder would see, including the 8-bit
    codebook rounding; code assignment is also done against the stored
    (rounded) codebook, as an honest encoder would.

    `sample_weight`: optional per-weight non-negative importance, same
    length as w (see fisher.per_weight_importance). When given, each
    group's Lloyd fit is reweighted by it (diagonal-Hessian weighting,
    the OBQ first slice). sample_weight=None reproduces the unweighted
    fit exactly.
    """
    return _quantize_kmeans_q8(w, group_size, n_iter, sample_weight,
                               n_bits=2, name="int2_kmeans_q8")


def quantize_int4_kmeans_q8(w: np.ndarray, group_size: int = GROUP_SIZE,
                            n_iter: int = 20) -> QuantResult:
    """Fitted 4-bit: per-group Lloyd with a 16-centroid 8-bit codebook.

    The remaining refinement after the 2026-09-22 naive-q4 embedding probe
    (int4_uniform collapsed on the tied wte head: ppl 7730.81 vs 53.50
    fp32): does a FITTED 4-bit codebook (Lloyd 16-centroid, not naive
    uniform) survive on the tied head? Same 8-bit codebook storage as
    int2_kmeans_q8 -> 4 + 16*8/128 + 16/128 = 5.125 bpw @ g128. Kept
    separate from the uniform int4 scheme name-wise so SQNR/ppl
    comparisons stay honest about fitted-vs-naive.
    """
    return _quantize_kmeans_q8(w, group_size, n_iter, None,
                               n_bits=4, name="int4_kmeans_q8")


def _ternary_lloyd_fit_batch(wp: np.ndarray, dual: bool,
                             n_iter: int = 20,
                             thresh_factor: float = 1.0,
                             record_states: bool = False) -> tuple:
    """Vectorized constrained 1-D Lloyd ternary fit, over all groups.

    wp is the (n_groups, group_size) float32 padded array as returned by
    _groups(). Runs exactly the _ternary_lloyd_fit alternating
    assignment/refit per group, in float64, but as whole-array numpy ops
    instead of a per-group Python loop (measured 2026-09-22: 1-step
    24.7 vs 2.8 Mparams/s, n_iter=20 3.3 vs 0.8 Mparams/s).

    Bit-identical to calling _ternary_lloyd_fit once per group (pinned
    by test against a frozen copy of the old per-group loop): per-row
    np.mean/np.sum reduce identically to their 1-D forms on this numpy
    build, and the zero-filled masked sums match the masked-1-D means
    because the extra zero terms are exact. If a future numpy changes
    its reduction order, the pin test fails loudly.

    Returns (s_pos, s_neg, codes, states): per-group float64 scales
    (guarded at 1e-12 exactly like the per-group path), flat int8 codes
    over all groups (truncate to the unpadded length yourself), and -
    when record_states=True - the list of (s_pos, s_neg) snapshots
    (heuristic init + one per applied update) used to rebuild the
    single-group `history` diagnostic.
    """
    wf = np.asarray(wp, dtype=np.float64)
    n_groups = wf.shape[0]
    t = 0.5 * thresh_factor
    if dual:
        pos = wf > 0
        neg = wf < 0
        # Zero-filled masked sums in row order: identical to the
        # masked-1-D mean of the per-group path (test-pinned); the
        # np.maximum(..., 1) denominator only fires where np.where
        # discards the result anyway.
        s_pos = np.where(pos.any(axis=1),
                         (wf * pos).sum(axis=1) / np.maximum(pos.sum(axis=1), 1),
                         0.0)
        s_neg = np.where(neg.any(axis=1),
                         (-wf * neg).sum(axis=1) / np.maximum(neg.sum(axis=1), 1),
                         0.0)
    else:
        s = np.mean(np.abs(wf), axis=1)
        s_pos = s.copy()
        s_neg = s.copy()
    states = [(s_pos.copy(), s_neg.copy())] if record_states else None
    for _ in range(n_iter):
        a_pos = wf > t * s_pos[:, None]
        a_neg = wf < -t * s_neg[:, None]
        cnt_p = a_pos.sum(axis=1)
        cnt_n = a_neg.sum(axis=1)
        new_pos = np.where(cnt_p > 0,
                           (wf * a_pos).sum(axis=1) / np.maximum(cnt_p, 1),
                           s_pos)
        new_neg = np.where(cnt_n > 0,
                           (-wf * a_neg).sum(axis=1) / np.maximum(cnt_n, 1),
                           s_neg)
        if not dual:
            n_assigned = cnt_p + cnt_n
            # Elementwise int64*float64, same op order as the per-group
            # path; the guarded denominator is only used where np.where
            # keeps it (np.errstate silences the discarded 0/0).
            with np.errstate(invalid="ignore", divide="ignore"):
                new = ((cnt_p * new_pos + cnt_n * new_neg)
                       / np.maximum(n_assigned, 1))
            new = np.where(n_assigned > 0, new, s_pos)
            new_pos = new
            new_neg = new
        # Per-group early exit, same condition as the per-group `break`:
        # a group stops updating exactly when its scales stop changing.
        changed = (new_pos != s_pos) | (new_neg != s_neg)
        if not changed.any():
            break
        s_pos = np.where(changed, new_pos, s_pos)
        s_neg = np.where(changed, new_neg, s_neg)
        if record_states:
            states.append((s_pos.copy(), s_neg.copy()))
    s_pos = np.maximum(s_pos, 1e-12)  # all-zero / one-sided group guards
    s_neg = np.maximum(s_neg, 1e-12)
    codes = np.zeros(wf.shape, dtype=np.int8)
    codes[wf > t * s_pos[:, None]] = 1
    codes[wf < -t * s_neg[:, None]] = -1
    return s_pos, s_neg, codes.reshape(-1), states


def _ternary_lloyd_fit(g: np.ndarray, dual: bool,
                       n_iter: int = 20,
                       thresh_factor: float = 1.0,
                       history: list | None = None) -> tuple:
    """Constrained 1-D Lloyd for a ternary codebook (single group).

    Thin wrapper over _ternary_lloyd_fit_batch for one group: identical
    signature, identical return (s_pos, s_neg, codes), and the same
    `history` semantics (heuristic init appended, then one entry per
    applied scale update) used by diagnose.py. Kept as the per-group
    entry point so existing callers and tests are untouched; the batch
    core is what the scheme functions use.
    """
    wf = np.asarray(g, dtype=np.float64).reshape(1, -1)
    s_pos, s_neg, codes, states = _ternary_lloyd_fit_batch(
        wf, dual=dual, n_iter=n_iter, thresh_factor=thresh_factor,
        record_states=history is not None)
    if history is not None:
        prev = None
        for vsp, vsn in states:
            cur = (float(vsp[0]), float(vsn[0]))
            if cur != prev:  # belt-and-braces; snapshots only append on change
                history.append(cur)
            prev = cur
    return float(s_pos[0]), float(s_neg[0]), codes


def quantize_ternary_lloyd(w: np.ndarray, group_size: int = GROUP_SIZE,
                           n_iter: int = 20) -> QuantResult:
    """Candidate C (ours), symmetric: Lloyd-fit ternary {-s,0,+s}.

    Same payload and storage as ternary_uniform (1.710 bpw at group 128),
    but the per-group scale is Lloyd-fitted to the group's actual
    distribution instead of fixed at the absmean. The SQNR comparison
    against ternary_uniform at identical bitrate isolates whether Lloyd
    *fitting* rescues ternary on the fidelity axis.

    Vectorized over groups via _ternary_lloyd_fit_batch (bit-identical
    to the former per-group loop).
    """
    wp, n_groups, n = _groups(w, group_size)
    s_pos, _s_neg, codes, _states = _ternary_lloyd_fit_batch(
        wp, dual=False, n_iter=n_iter)
    scales = s_pos.astype(np.float32).reshape(n_groups, 1)
    codes = codes[:n]
    bpw = TERNARY_PAYLOAD_BPW + _scale_overhead(1, group_size)
    return QuantResult("ternary_lloyd", codes, scales, bpw,
                       group_size=group_size)


def quantize_ternary_lloyd_ds(w: np.ndarray, group_size: int = GROUP_SIZE,
                              n_iter: int = 20) -> QuantResult:
    """Candidate C (ours), dual-scale: Lloyd-fit {-s_neg,0,+s_pos}.

    The asymmetric twin of ternary_lloyd: separate Lloyd-fitted scales per
    side, 1.835 bpw at group 128 - identical bitrate to dual_scale_ternary,
    so the comparison isolates fitting on skewed distributions.

    Vectorized over groups via _ternary_lloyd_fit_batch (bit-identical
    to the former per-group loop).
    """
    wp, n_groups, n = _groups(w, group_size)
    s_pos, s_neg, codes, _states = _ternary_lloyd_fit_batch(
        wp, dual=True, n_iter=n_iter)
    scales = np.stack([s_pos, s_neg], axis=1).astype(np.float32)
    codes = codes[:n]
    bpw = TERNARY_PAYLOAD_BPW + _scale_overhead(2, group_size)
    return QuantResult("ternary_lloyd_ds", codes, scales, bpw,
                       group_size=group_size)


def quantize_ternary_1step(w: np.ndarray, group_size: int = GROUP_SIZE,
                           thresh_factor: float = 1.0) -> QuantResult:
    """Candidate D (ours): "1-step Lloyd" ternary.

    The diagnose.py decomposition showed the +1.4 dB ternary_lloyd win
    over ternary_uniform is ~85% captured by the FIRST Lloyd iteration
    (heuristic assignment -> one L2-optimal scale refit -> reassignment
    at the refit thresholds) and ~15% by further iterations. This scheme
    is exactly that one iteration - implemented by reusing
    _ternary_lloyd_fit with n_iter=1 - so it costs O(1) extra work over
    ternary_uniform (two passes per group, no convergence loop) and
    stores a single fp16 scale per group: identical 1.710 bpw at
    group 128. The SQNR comparison against ternary_uniform at matched
    bitrate is then a direct test of whether one fit step suffices.

    (A stricter reading - refit the scale but keep the heuristic codes -
    was tried and scores worse: 6.45 vs 6.72 dB on the seed-7 clean
    tensor, because decoding codes chosen for s0 at the larger refit
    scale is inconsistent. Reassigning at the refit thresholds is the
    honest encoder and is what this scheme does.)

    thresh_factor (default 1.0) widens the zero bin for the sparsity
    probe (threshold-biased Lloyd - the fit stays L2-optimal under the
    widened thresholds); it changes zero-rate but not bitrate, so
    widened variants compare at matched bpw.

    Vectorized over groups via _ternary_lloyd_fit_batch (bit-identical
    to the former per-group loop).
    """
    wp, n_groups, n = _groups(w, group_size)
    s_pos, _s_neg, codes, _states = _ternary_lloyd_fit_batch(
        wp, dual=False, n_iter=1, thresh_factor=thresh_factor)
    scales = s_pos.astype(np.float32).reshape(n_groups, 1)
    codes = codes[:n]
    bpw = TERNARY_PAYLOAD_BPW + _scale_overhead(1, group_size)
    return QuantResult("ternary_1step", codes, scales, bpw,
                       group_size=group_size)


def quantize_ternary_1step_ds(w: np.ndarray, group_size: int = GROUP_SIZE,
                              n_iter: int = 1) -> QuantResult:
    """Candidate D (ours), dual-scale: "few-step Lloyd" ternary.

    The dual-scale twin of ternary_1step: heuristic dual thresholds
    (conditional means per side) -> one per-side L2-optimal scale refit ->
    reassignment at the refit thresholds, via _ternary_lloyd_fit with
    dual=True and (default) n_iter=1. O(1) extra work over
    dual_scale_ternary, two fp16 scales per group: identical 1.835 bpw
    at group 128 to ternary_lloyd_ds, so the comparison isolates whether
    few fit iterations capture the full dual-Lloyd win (they largely do
    for the symmetric case - see ternary_1step - but the dual case is
    tested here on skewed tensors). n_iter is exposed for the
    convergence check (2-3 iterations are still O(1)).

    Vectorized over groups via _ternary_lloyd_fit_batch (bit-identical
    to the former per-group loop).
    """
    wp, n_groups, n = _groups(w, group_size)
    s_pos, s_neg, codes, _states = _ternary_lloyd_fit_batch(
        wp, dual=True, n_iter=n_iter)
    scales = np.stack([s_pos, s_neg], axis=1).astype(np.float32)
    codes = codes[:n]
    bpw = TERNARY_PAYLOAD_BPW + _scale_overhead(2, group_size)
    return QuantResult("ternary_1step_ds", codes, scales, bpw,
                       group_size=group_size)


SCHEMES = {
    "ternary_uniform": quantize_ternary_uniform,
    "ternary_lloyd": quantize_ternary_lloyd,
    "ternary_lloyd_ds": quantize_ternary_lloyd_ds,
    "ternary_1step": quantize_ternary_1step,
    # Sparse twin of ternary_1step: threshold-biased Lloyd (1-step fit
    # under widened zero-bin thresholds) pushes the measured zero-rate
    # from ~0.41 to ~0.51 on GPT-2 weights at matched 1.710 bpw
    # (2026-09-21 sparsity probe: factor 1.2 costs only -0.05 dB SQNR).
    # VERDICT (negative): the ppl cliff is razor-sharp - ppl 45242 on
    # eval_text1 vs 2764 for ternary_1step (16x blowup) at that -0.05 dB
    # delta. Zero-rate 0.5 is NOT free; sparsity gains that SQNR can't
    # see still kill the model. Kept as the reference for what 0.5
    # zero-rate costs, and for the energy-ratio table (3.51x -> 4.22x
    # energy-proxy advantage over int2_kmeans_q8 - unrealizable at this
    # fidelity). The decode is the plain symmetric ternary path.
    "ternary_1step_sp": partial(quantize_ternary_1step, thresh_factor=1.2),
    "ternary_1step_ds": quantize_ternary_1step_ds,
    "int2_symmetric": quantize_int2_symmetric,
    "int8_uniform": quantize_int8_uniform,
    "int4_uniform": quantize_int4_uniform,
    "int2_kmeans": quantize_int2_kmeans,
    "int2_kmeans_q8": quantize_int2_kmeans_q8,
    "int4_kmeans_q8": quantize_int4_kmeans_q8,
    "int2_outlier_retain": quantize_int2_outlier_retain,
    "dual_scale_ternary": quantize_dual_scale_ternary,
    "ternary_outlier": quantize_ternary_outlier,
}
