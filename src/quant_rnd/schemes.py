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
"""
from dataclasses import dataclass, field

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
        """Dequantize back to float32, same shape as the input."""
        n = self.codes.shape[0]
        g = self.group_size
        n_groups = (n + g - 1) // g
        out = np.zeros(n, dtype=np.float32)
        group_id = np.arange(n) // g
        for gi in range(n_groups):
            mask = group_id == gi
            c = self.codes[mask].astype(np.float32)
            s = self.scales[gi]
            if self.name == "dual_scale_ternary":
                rec = np.where(c > 0, s[0], np.where(c < 0, -s[1], 0.0))
            elif self.name in _CODEBOOK_SCHEMES:
                # s holds the effective per-group codebook (fp16-fitted or
                # 8-bit-rounded centroids); codes index into it.
                rec = s[c.astype(np.int64)]
            else:
                rec = c * s[0]
            out[mask] = rec
        if self._outlier_vals is not None:
            out[self._outlier_idx] = self._outlier_vals
        return out


# Schemes whose `scales` hold an effective per-group codebook that the
# integer `codes` index into (rather than multiplicative scales).
_CODEBOOK_SCHEMES = frozenset({"int2_kmeans", "int2_kmeans_q8"})


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


def _lloyd_1d(x: np.ndarray, k: int = 4, n_iter: int = 20) -> np.ndarray:
    """Deterministic 1-D Lloyd's algorithm; returns k centroids.

    Quantile initialization keeps it deterministic (no random restarts);
    empty clusters keep their previous centroid.
    """
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


def quantize_int2_kmeans_q8(w: np.ndarray, group_size: int = GROUP_SIZE,
                            n_iter: int = 20) -> QuantResult:
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
    """
    wp, n_groups, n = _groups(w, group_size)
    centroids = np.stack([_lloyd_1d(g, n_iter=n_iter) for g in wp])
    # Symmetric 8-bit quantization of the per-group codebook.
    cmax = np.max(np.abs(centroids), axis=1, keepdims=True).astype(np.float32)
    cmax = np.maximum(cmax, 1e-12)  # all-zero group guard
    q8 = np.round(centroids / cmax * 127.0).astype(np.int8)
    deq = (q8.astype(np.float32) / 127.0 * cmax)
    codes = np.abs(wp[:, :, None] - deq[:, None, :]).argmin(axis=2)
    codes = codes.astype(np.int8).ravel()[:n]
    bpw = 2.0 + 4 * 8 / group_size + _scale_overhead(1, group_size)
    return QuantResult("int2_kmeans_q8", codes, deq, bpw,
                       group_size=group_size)


SCHEMES = {
    "ternary_uniform": quantize_ternary_uniform,
    "int2_symmetric": quantize_int2_symmetric,
    "int2_kmeans": quantize_int2_kmeans,
    "int2_kmeans_q8": quantize_int2_kmeans_q8,
    "int2_outlier_retain": quantize_int2_outlier_retain,
    "dual_scale_ternary": quantize_dual_scale_ternary,
    "ternary_outlier": quantize_ternary_outlier,
}
