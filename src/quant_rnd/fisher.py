"""Empirical-Fisher (diagonal) importance weights for the quant R&D track.

First slice of the OBQ-style second-order correction backlog item: before a
full GPTQ-style per-weight error-compensation pass, reweight only the Lloyd
*centroid fit* by the per-input-channel activation energy collected from fp32
forward passes. For a linear y = xW the OBS/OBQ Hessian is H = 2 X^T X; the
diagonal proxy used here is d_j = mean_t(x_{t,j}^2) per input channel j, so a
weight's importance is the energy of the channel it reads from.

Honest caveats:
- Diagonal only: cross-channel terms of the Hessian are ignored, so this is
  a strict subset of what GPTQ/OBQ do. If the ppl delta is ~0, the follow-up
  is full per-weight error compensation (separate backlog item).
- Calibration text: with only two short hand-composed eval texts on this VM
  there is no spare held-out corpus, so the Fisher statistics are collected
  on the eval text itself. The ppl delta is directionally indicative, not a
  held-out measurement; the shuffled-block eval protocol (backlog) will let
  future slices calibrate and evaluate on disjoint blocks.
"""
import numpy as np

from .gpt2_forward import GPT2
from .realweights import layer_index_of, linear_weight_tensors


def capture_linear_inputs(tensors: dict, token_ids) -> dict:
    """{linear tensor name: (T, in_dim) input activation} from one fp32 pass."""
    model = GPT2(tensors)
    cap: dict = {}
    model.forward(token_ids, capture=cap)
    return cap


def diag_fisher_weights(inputs: dict, linear_names) -> dict:
    """Per-input-channel mean squared activation: {name: (in_dim,)}.

    d_j = mean_t(x_{t,j}^2); non-negative by construction.
    """
    out = {}
    for name in linear_names:
        a = inputs[name].astype(np.float64)
        out[name] = (a * a).mean(axis=0).astype(np.float32)
    return out


def per_weight_importance(tensors: dict, token_ids) -> dict:
    """{linear tensor name: flat per-weight importance} aligned with .ravel().

    Weight W[j, k] (input channel j, output k) gets importance d_j, so the
    flat array is np.repeat(channel_energy, out_dim). Consumed by
    quantize_model(..., sample_weights=...) in ppl.py; only schemes whose
    encoder accepts a `sample_weight` kwarg use it.
    """
    linear = linear_weight_tensors(tensors)
    names = list(linear)
    inputs = capture_linear_inputs(tensors, token_ids)
    energies = diag_fisher_weights(inputs, names)
    out = {}
    for name in names:
        w = linear[name]
        e = energies[name]
        if e.shape[0] != w.shape[0]:
            raise ValueError(f"channel mismatch for {name}: energy "
                             f"{e.shape[0]} vs in_dim {w.shape[0]}")
        out[name] = np.repeat(e, w.shape[1]).astype(np.float32)
    return out


def layer_fisher_trace(tensors: dict, token_ids) -> dict:
    """Per-block Fisher sensitivity: {layer_idx: trace}.

    Trace = sum of the diag-Fisher per-weight importance over the
    block's linear weights (the h.N tensors). Total, not mean: a
    block's contribution to the Hessian-weighted error is the sum over
    its weights, so this is the quantity a bit-budget reallocation
    should follow — the most sensitive blocks get the most bits. The
    calibration caveat from per_weight_importance applies unchanged:
    with no held-out corpus on this VM the statistics are collected on
    the eval text itself, so the ranking is directional, not held-out.
    Blocks with no linear weights raise KeyError; linear weights
    without a block index do too — a silent fp32 pass-through would
    corrupt the bpw accounting the selector relies on.
    """
    importance = per_weight_importance(tensors, token_ids)
    linear = set(linear_weight_tensors(tensors))
    trace: dict = {}
    for name in importance:
        if name not in linear:
            continue
        idx = layer_index_of(name)
        if idx is None:
            raise KeyError(f"linear weight {name!r} has no block index; "
                           "cannot assign a per-layer scheme")
        trace[idx] = trace.get(idx, 0.0) + float(importance[name].sum())
    if not trace:
        raise KeyError("no linear weights found for layer sensitivity")
    return trace
