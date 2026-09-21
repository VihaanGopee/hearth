"""GPTQ/OBQ-style per-weight error compensation for the quant R&D track.

Second slice of the OBQ backlog item: the diagonal-Fisher reweighting
(fisher.py) moved perplexity ~0, so this implements the full per-weight
error-compensation step. Input channels are quantized in blocks; after
each block, the not-yet-quantized channels are updated with the
Hessian-optimal correction

    W[rest, :] -= H^{-1}[rest, b] @ solve(H^{-1}[b, b], W[b, :] - Q[b, :])

(the GPTQ block update, which is the exact greedy minimizer of the
quadratic (W - Q)^T H (W - Q) given the quantized block).

Conventions match gpt2_forward: a linear is y = x @ W with W of shape
(d_in, d_out), so the OBS/OBQ Hessian H = X^T X is over input channels =
rows of W. Blocks iterate over rows; each block holds an integer number
of per-output-column quantization groups (block_size a multiple of
group_size, the standard GPTQ group-aligned configuration), each fitted
with the same 4-centroid Lloyd codebook + 8-bit codebook rounding as
int2_kmeans_q8, so OBQ-vs-naive comparisons are at matched bitrate
(2.375 bpw @ g128).

This slice handles ONE layer at a time via quantize_layer_obq (per the
backlog item's slicing); quantize_layers_obq applies it over a whole
model (or a named subset) with activations from a single shared forward
pass — each layer still gets its own layer-local Hessian, the GPTQ
convention. The caller supplies the layer's fp32 input activations X
(see fisher.capture_linear_inputs).

Honest caveats:
- Calibration: H is built from a single forward pass over the eval text
  (T=406 tokens < d_in=768 for GPT-2 124M), so H is rank-deficient and the
  damping (default 1% of mean diag(H), the GPTQ convention) is
  load-bearing, not a formality.
- The greedy block order (0..d_in) is not the optimal OBS order; GPTQ
  accepts the same approximation.
"""
import numpy as np

from .schemes import _scale_overhead, quantize_int2_kmeans_q8


def inverse_hessian(X: np.ndarray, damp_frac: float = 0.01) -> np.ndarray:
    """Damped inverse of H = X^T X, float64.

    X is (T, d_in). Damping is damp_frac * mean(diag(H)) on the diagonal
    (the GPTQ convention); damp_frac must be positive because the
    calibration activations on this VM are rank-deficient (T < d_in).
    Returns the (d_in, d_in) inverse via Cholesky.
    """
    if damp_frac <= 0:
        raise ValueError(f"damp_frac must be positive, got {damp_frac}")
    Xd = np.asarray(X, dtype=np.float64)
    if Xd.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {Xd.shape}")
    H = Xd.T @ Xd
    damp = damp_frac * np.mean(np.diag(H))
    # Guard the degenerate all-zero-activation case (damp would be 0).
    damp = max(damp, 1e-12)
    H += damp * np.eye(H.shape[0])
    L = np.linalg.cholesky(H)
    return np.linalg.solve(L.T, np.linalg.solve(L, np.eye(H.shape[0])))


def quantize_layer_obq(W: np.ndarray, X: np.ndarray,
                       group_size: int = 128,
                       block_size: int | None = None,
                       damp_frac: float = 0.01,
                       n_iter: int = 20) -> tuple:
    """Quantize one (d_in, d_out) weight matrix with OBQ error compensation.

    Returns (Wq float32, bpw). The input W is not modified. bpw uses the
    identical formula as int2_kmeans_q8 (2-bit payload + 4 int8 codebook
    entries + one fp16 scale per group), so the naive anchor and OBQ
    share a bitrate by construction.
    """
    if block_size is None:
        block_size = group_size
    if block_size % group_size != 0:
        raise ValueError("this slice requires block_size to be a multiple "
                         f"of group_size (got {block_size} vs {group_size}): "
                         "each block must hold whole per-column groups")
    W = np.asarray(W, dtype=np.float32)
    if W.ndim != 2:
        raise ValueError(f"W must be 2-D, got shape {W.shape}")
    d_in, d_out = W.shape
    if d_in % block_size != 0:
        raise ValueError(f"this slice requires block_size to divide d_in "
                         f"(got block {block_size}, d_in {d_in})")
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != d_in:
        raise ValueError(f"X shape {X.shape} incompatible with W {W.shape}")
    Hinv = inverse_hessian(X, damp_frac=damp_frac)

    Wc = W.astype(np.float64)          # working copy, gets compensated
    Wq = np.zeros_like(Wc)             # quantized output
    n_sub = block_size // group_size
    for start in range(0, d_in, block_size):
        rows = slice(start, start + block_size)
        seg = Wc[rows, :]             # (block_size, d_out), compensated
        # Per-output-column codebook fit: identical math to int2_kmeans_q8
        # groups (4 fitted centroids, 8-bit codebook), n_sub groups/column.
        qcols = []
        for k in range(d_out):
            col = seg[:, k]
            parts = [quantize_int2_kmeans_q8(
                        col[s:s + group_size], group_size=group_size,
                        n_iter=n_iter).reconstruct()
                     for s in range(0, block_size, group_size)]
            qcols.append(np.concatenate(parts))
        Q = np.stack(qcols, axis=1).astype(np.float64)
        Wq[rows, :] = Q
        err = seg - Q                 # (block_size, d_out)
        E = np.linalg.solve(Hinv[rows, rows], err)
        rest = slice(rows.stop, d_in)
        if rest.stop > rest.start:
            Wc[rest, :] -= Hinv[rest, rows] @ E
    bpw = 2.0 + 4 * 8 / group_size + _scale_overhead(1, group_size)
    return Wq.astype(np.float32), bpw


def quantize_layers_obq(linear: dict, inputs: dict,
                        group_size: int = 128,
                        damp_frac: float = 0.01,
                        n_iter: int = 20,
                        only_names: set | None = None) -> tuple:
    """Quantize several (d_in, d_out) weight matrices with OBQ.

    `linear`: {tensor name: fp32 weight}; `inputs`: {tensor name:
    (T, d_in) input activations} from ONE shared forward pass (see
    fisher.capture_linear_inputs) — each layer is compensated against
    its OWN layer Hessian, the GPTQ convention. `only_names` restricts
    to a subset of names (KeyError on unknown names). Returns
    ({name: quantized float32}, bpw); bpw is identical for every layer
    by construction (2.375 @ g128, the int2_kmeans_q8 bookkeeping).
    The input dicts are not modified.
    """
    if only_names is not None:
        unknown = set(only_names) - set(linear)
        if unknown:
            raise KeyError(f"only_names has unknown tensors: "
                           f"{sorted(unknown)}")
        targets = sorted(set(only_names))
    else:
        targets = sorted(linear)
    if not targets:
        raise ValueError("no layers selected")
    out = {}
    bpw = None
    for name in targets:
        if name not in inputs:
            raise KeyError(f"no captured activations for {name!r}")
        Wq, bpw = quantize_layer_obq(linear[name], inputs[name],
                                     group_size=group_size,
                                     damp_frac=damp_frac, n_iter=n_iter)
        out[name] = Wq
    return out, bpw


def hessian_weighted_sse(W: np.ndarray, Wq: np.ndarray,
                         H: np.ndarray) -> float:
    """Sum over output columns of (w - q)^T H (w - q): the OBQ objective.

    Test/diagnostic helper: the quantity the block update greedily
    minimizes. H is the (d_in, d_in) Hessian (need not be inverted).
    """
    D = np.asarray(W, dtype=np.float64) - np.asarray(Wq, dtype=np.float64)
    return float(np.einsum("ik,ij,jk->", D, np.asarray(H, dtype=np.float64), D))
