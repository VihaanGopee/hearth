"""Per-scheme quantized perplexity on GPT-2 124M (NumPy only).

Step 2b of the roadmap's "validate candidates on a real tiny model" item:
take the fp32 GPT-2 checkpoint, quantize every linear weight matrix
group-wise with each scheme, reconstruct in fp32, and measure perplexity
on research/data/eval_text.txt against the 53.50 fp32 reference
(see research/log/2026-09-21.md for the 2a session).

Scope notes (honest):
- By default, embeddings (wte, which is also the tied lm_head), position
  embeddings, biases, and LayerNorm weights/biases stay fp32. This is the
  standard weight-only quantization protocol; the `--quantize-embeddings`
  flag runs the ablation that also quantizes the embedding tables
  (biases/LN still stay fp32).
- Every scheme is evaluated at group_size=128 with its default kwargs
  (e.g. ternary_1step_ds keeps n_iter=1), matching the bpw bookkeeping
  of the SQNR probes. Schemes whose quantization is too slow for a
  session's budget (the Lloyd k-means family: ~7 min/full-model here)
  are left out of the default sweep and get their own run.
- Quantization is deterministic (no random init anywhere in the
  encoders), so results are reproducible bit-for-bit.
- `--eval-texts A.txt,B.txt` overrides `--eval-text` and evaluates every
  scheme on every text, reporting per-text ppl plus mean/std — the
  text-robustness protocol the eval_text2 probe motivated (single-text
  ppl is text-sensitive: ~+30% absolute shift text1 -> text2 for the
  same scheme). `fp32` is accepted as a scheme name for a same-table
  unquantized reference (bpw reported as 32.0).

Usage: python3 -m src.quant_rnd.ppl <model.safetensors> [--schemes ...]
"""
import argparse
import inspect
import os
import time

import numpy as np

from .gpt2_forward import GPT2
from .gpt2_tokenizer import GPT2Tokenizer
from .realweights import layer_index_of, read_safetensors, linear_weight_tensors
from .schemes import GROUP_SIZE, SCHEMES

# Mirrors the exclusion list in realweights.linear_weight_tensors: tensors
# whose name matches one of these markers are the embedding tables that
# the weight-only protocol normally leaves in fp32.
EMBEDDING_MARKERS = ("embed", "wte", "wpe")


# Naive int2_kmeans_q8 g128 full-model perplexity on eval_text.txt (the
# text-conditioned anchor the checkpointed OBQ full-model measurement is
# judged against; see roadmap — < ~400 keeps the fidelity path alive).
NAIVE_ANCHOR_PPL_TEXT1 = 795.96

# Pseudo-scheme: unquantized float32 reference. Accepted in --schemes so
# the reference sits in the same table; bpw is reported as 32.0 (the
# checkpoint is float32).
FP32_SCHEME = "fp32"


def eval_text_paths(args: argparse.Namespace) -> list:
    """Ordered eval-text paths: --eval-texts (comma-separated) overrides
    --eval-text. Factored for testability."""
    if args.eval_texts:
        return [p for p in args.eval_texts.split(",") if p]
    return [args.eval_text]


def is_embedding(name: str) -> bool:
    low = name.lower()
    return any(m in low for m in EMBEDDING_MARKERS)


# Schemes cheap enough to run end-to-end inside one session on this VM
# (full-model quantization + one ~40 s forward each). The Lloyd k-means
# family is excluded here for speed (see module docstring) and runs as a
# follow-up slice.
DEFAULT_SWEEP = [
    "int2_symmetric",      # naive floor, 2.125 bpw
    "ternary_uniform",     # naive floor, 1.710 bpw
    "ternary_1step",       # practical ternary ref, 1.710 bpw
    "ternary_1step_ds",    # practical dual ternary, 1.835 bpw
    "ternary_lloyd",       # full-fitted ternary, 1.710 bpw
    "ternary_outlier",     # ternary + exact outliers, ~2.06 bpw
    "dual_scale_ternary",  # naive dual ternary, 1.835 bpw
]


def quantize_model(tensors: dict, scheme_name: str,
                   group_size: int = GROUP_SIZE,
                   sample_weights: dict | None = None,
                   quantize_embeddings: bool = False,
                   only_names: set | None = None) -> tuple:
    """Quantize + reconstruct every linear weight with `scheme_name`.

    Returns (weights, bpw): a new {name: float32 ndarray} dict with the
    same keys and shapes as the input; linear matrices (the same set
    linear_weight_tensors selects) are quantized group-wise over the flat
    array and reconstructed, everything else passes through as a float32
    copy. The input dict is not modified.

    `quantize_embeddings`: also quantize the embedding tables (wte, wpe,
    anything matching EMBEDDING_MARKERS) with the same scheme. Biases and
    LayerNorm scales stay fp32 regardless — the ablation targets the
    weight tables only, and per the roadmap item this is the protocol
    variant being compared against the fp32-embedding reference. bpw is
    unchanged by the flag (same scheme, same group size).

    `only_names`: restrict quantization targets to this subset of tensor
    names (must be linear/embedding names as applicable); everything else
    passes through fp32. None (default) keeps the original behavior of
    quantizing every linear weight.

    `sample_weights`: optional {tensor_name: flat per-weight importance}
    (see fisher.per_weight_importance). It is passed as `sample_weight=`
    only to scheme encoders that accept that kwarg (currently
    int2_kmeans_q8); other schemes silently ignore it, so a --fisher run
    over a mixed sweep only reweights the schemes that support it.

    The pseudo-scheme "fp32" skips quantization entirely: every tensor
    passes through as a float32 copy (the same unquantized reference the
    log quotes by hand), and bpw is reported as 32.0.
    """
    if scheme_name == FP32_SCHEME:
        out = {n: np.ascontiguousarray(t, dtype=np.float32)
               for n, t in tensors.items()}
        return out, 32.0
    if scheme_name not in SCHEMES:
        raise KeyError(f"unknown scheme {scheme_name!r}; "
                       f"have {sorted(SCHEMES)}")
    fn = SCHEMES[scheme_name]
    takes_weight = "sample_weight" in inspect.signature(fn).parameters
    linear = set(linear_weight_tensors(tensors))
    targets = set(linear)
    if quantize_embeddings:
        targets |= {n for n in tensors if is_embedding(n)}
    if only_names is not None:
        unknown = set(only_names) - set(tensors)
        if unknown:
            raise KeyError(f"only_names has unknown tensors: {sorted(unknown)}")
        targets &= set(only_names)
    out = {}
    bpw = None
    for name, t in tensors.items():
        t32 = np.ascontiguousarray(t, dtype=np.float32)
        if name in targets:
            kw = {}
            if (sample_weights is not None and takes_weight
                    and name in sample_weights):
                kw["sample_weight"] = sample_weights[name]
            q = fn(t32.ravel(), group_size=group_size, **kw)
            out[name] = q.reconstruct().reshape(t.shape)
            bpw = q.bpw
        else:
            out[name] = t32
    return out, bpw


def quantize_model_obq(tensors: dict, token_ids,
                       group_size: int = GROUP_SIZE,
                       damp_frac: float = 0.01,
                       only_names: set | None = None) -> tuple:
    """OBQ error compensation over all (or a subset of) linear layers.

    Returns (weights, bpw) like quantize_model: a new {name: float32
    ndarray} dict with the same keys and shapes as the input. One shared
    fp32 forward pass captures every linear layer's input activations
    (fisher.capture_linear_inputs); each target layer is then quantized
    with obq.quantize_layers_obq (GPTQ-style block update with the
    int2_kmeans_q8-equivalent per-column codebook, same 2.375 bpw @ g128
    as the naive anchor, so the perplexity delta is apples-to-apples).
    Non-target tensors pass through as float32 copies. The input dict is
    not modified. `only_names` restricts quantization to a subset of
    linear names (KeyError on unknown/non-linear names); None (default)
    quantizes every linear matrix.
    """
    from .fisher import capture_linear_inputs
    from .obq import quantize_layers_obq
    linear = linear_weight_tensors(tensors)
    if only_names is not None:
        unknown = set(only_names) - set(tensors)
        if unknown:
            raise KeyError(f"only_names has unknown tensors: "
                           f"{sorted(unknown)}")
        non_linear = set(only_names) - set(linear)
        if non_linear:
            raise KeyError(f"only_names are not linear weight matrices: "
                           f"{sorted(non_linear)}")
    inputs = capture_linear_inputs(tensors, token_ids)
    qobq, bpw = quantize_layers_obq(linear, inputs, group_size=group_size,
                                    damp_frac=damp_frac,
                                    only_names=only_names)
    out = {n: np.ascontiguousarray(t, dtype=np.float32)
           for n, t in tensors.items()}
    for name, Wq in qobq.items():
        out[name] = Wq.reshape(tensors[name].shape)
    return out, bpw


def quantize_one_layer_obq(tensors: dict, token_ids,
                           layer_name: str,
                           group_size: int = GROUP_SIZE,
                           damp_frac: float = 0.01) -> tuple:
    """Quantize ONE linear layer with OBQ error compensation; rest fp32.

    Returns (weights, bpw) like quantize_model. Delegates to
    quantize_model_obq with only_names={layer_name}: the layer's fp32
    input activations come from the shared capture forward pass, and the
    Hessian math is obq.quantize_layer_obq's (GPTQ-style block update,
    int2_kmeans_q8-equivalent per-column codebook, 2.375 bpw @ g128).
    """
    return quantize_model_obq(tensors, token_ids, group_size=group_size,
                              damp_frac=damp_frac,
                              only_names={layer_name})


def parse_layer_schemes(spec: str, n_layers: int) -> dict:
    """Parse a per-layer scheme spec into {block_idx: scheme_name}.

    Format: comma-separated "RANGE:scheme" items, where RANGE is "i",
    "i-j" (inclusive), e.g. "0-5:int8_uniform,6-11:ternary_1step".
    Whitespace around items is ignored. Validation (ValueError): every
    block 0..n_layers-1 must be assigned EXACTLY once (no gaps, no
    overlaps), indices must be in range, and scheme names must exist in
    SCHEMES. The strict coverage rule is what makes the reported
    average bpw honest: no block is silently left at fp32.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    assignment: dict = {}
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"bad layer-scheme item {item!r}: "
                             "expected RANGE:scheme")
        rng, scheme = item.split(":", 1)
        scheme = scheme.strip()
        rng = rng.strip()
        if scheme not in SCHEMES:
            raise ValueError(f"unknown scheme {scheme!r} in {item!r}; "
                             f"have {sorted(SCHEMES)}")
        if "-" in rng:
            lo_s, hi_s = rng.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"inverted range {rng!r} in {item!r}")
            idxs = range(lo, hi + 1)
        else:
            idxs = (int(rng),)
        for i in idxs:
            if not 0 <= i < n_layers:
                raise ValueError(f"block index {i} out of range "
                                 f"[0, {n_layers}) in {item!r}")
            if i in assignment:
                raise ValueError(f"block {i} assigned twice in {spec!r}")
            assignment[i] = scheme
    missing = [i for i in range(n_layers) if i not in assignment]
    if missing:
        raise ValueError(f"blocks {missing} have no scheme in {spec!r}; "
                         "every block 0..n_layers-1 must be assigned")
    return assignment


def sensitive_top_k(trace: dict, k: int) -> list:
    """Top-k block indices by Fisher trace, descending; ties break by
    lower index (deterministic). ValueError unless 1 <= k <= len(trace).
    """
    if not isinstance(k, int) or not 1 <= k <= len(trace):
        raise ValueError(f"k must satisfy 1 <= k <= {len(trace)}; "
                         f"got {k!r}")
    return sorted(trace, key=lambda i: (-trace[i], i))[:k]


def block_linear_names(tensors: dict) -> dict:
    """{block_idx: set of linear tensor names} over the model's blocks.

    KeyError if any linear weight lacks a block index (the same
    strictness as count_blocks — a silent fp32 pass-through would
    corrupt the per-block damage comparison the mechanism probe
    relies on).
    """
    groups: dict = {}
    for name in linear_weight_tensors(tensors):
        idx = layer_index_of(name)
        if idx is None:
            raise KeyError(f"linear weight {name!r} has no block index; "
                           "per-block sensitivity probes are unsupported "
                           "for this model")
        groups.setdefault(idx, set()).add(name)
    if not groups:
        raise KeyError("no linear weights found; per-block sensitivity "
                       "probes are unsupported for this model")
    return groups


def ordered_desc(values: dict) -> list:
    """Block indices sorted by value descending; ties break by lower
    index (deterministic). The ranking comparator for the mechanism
    probe: most sensitive / most damaged first."""
    return sorted(values, key=lambda i: (-values[i], i))


def spearman_rho(order_a: list, order_b: list) -> float:
    """Spearman rank correlation between two orderings of the same set.

    Position rank = index in the ordering list; rho is the Pearson
    correlation of the two position vectors. 1.0 = identical order,
    -1.0 = exactly reversed. ValueError unless both orders cover the
    same >= 2 elements.
    """
    if set(order_a) != set(order_b):
        raise ValueError("orderings cover different elements")
    n = len(order_a)
    if n < 2:
        raise ValueError("need at least 2 elements for a rank "
                         "correlation")
    pos_b = {v: i for i, v in enumerate(order_b)}
    ranks_b = [pos_b[v] for v in order_a]
    ranks_a = list(range(n))
    ma = sum(ranks_a) / n
    mb = sum(ranks_b) / n
    cov = sum((a - ma) * (b - mb)
              for a, b in zip(ranks_a, ranks_b))
    va = sum((a - ma) ** 2 for a in ranks_a)
    vb = sum((b - mb) ** 2 for b in ranks_b)
    if va == 0 or vb == 0:
        raise ValueError("degenerate ordering (no rank variance)")
    return cov / (va ** 0.5 * vb ** 0.5)


def quantize_model_per_layer(tensors: dict, layer_schemes: dict,
                             group_size: int = GROUP_SIZE) -> tuple:
    """Quantize each linear weight with its block's scheme; return
    (weights, avg_bpw).

    `layer_schemes` maps block index -> scheme name (see
    parse_layer_schemes). Only linear weights (the
    linear_weight_tensors set, embeddings excluded, biases/LN fp32)
    are quantized — same targeting as quantize_model. A linear weight
    whose block index has no entry in layer_schemes raises KeyError,
    as does a linear weight with no block index: the strictness keeps
    the average bpw honest.

    avg_bpw is the parameter-weighted mean of the per-tensor bpw over
    the quantized targets (bits = params * bpw summed, divided by total
    params) — this is the matched-bitrate denominator the
    sensitivity-adaptive experiments are judged against.
    """
    linear = set(linear_weight_tensors(tensors))
    out = {}
    total_bits = 0.0
    total_params = 0
    for name, t in tensors.items():
        t32 = np.ascontiguousarray(t, dtype=np.float32)
        if name in linear:
            idx = layer_index_of(name)
            if idx is None:
                raise KeyError(f"linear weight {name!r} has no block "
                               "index; cannot assign a per-layer scheme")
            if idx not in layer_schemes:
                raise KeyError(f"block {idx} (tensor {name!r}) has no "
                               "scheme in layer_schemes")
            scheme = layer_schemes[idx]
            if scheme not in SCHEMES:
                raise KeyError(f"unknown scheme {scheme!r}; "
                               f"have {sorted(SCHEMES)}")
            q = SCHEMES[scheme](t32.ravel(), group_size=group_size)
            out[name] = q.reconstruct().reshape(t.shape)
            n = t32.size
            total_bits += n * q.bpw
            total_params += n
        else:
            out[name] = t32
    avg_bpw = total_bits / total_params if total_params else 0.0
    return out, avg_bpw


def mixed_scheme_label(layer_schemes: dict) -> str:
    """Compact sweep-table label: 'mix[0-2:int8_uniform,3-11:ternary_1step]'."""
    n = max(layer_schemes) + 1
    runs = []
    start = prev = 0
    for i in range(1, n):
        if layer_schemes.get(i) != layer_schemes[prev]:
            runs.append((start, prev, layer_schemes[prev]))
            start = i
        prev = i
    runs.append((start, prev, layer_schemes[prev]))
    parts = [f"{a}" if a == b else f"{a}-{b}" for a, b, _ in runs]
    return ("mix[" + ",".join(f"{p}:{s}"
                              for p, (_, _, s) in zip(parts, runs)) + "]")


def main_obq_ckpt(args: argparse.Namespace, tensors: dict, token_ids,
                  text_label: str) -> None:
    """Checkpointed full-model OBQ run (--obq-all --obq-ckpt-dir).

    Quantizes up to --obq-max-layers new linear layers (each
    checkpointed to --obq-ckpt-dir as it finishes), skipping layers
    already in the manifest. When the last layer lands, the full weight
    dict is assembled and the final perplexity is measured on the eval
    text — the verdict against NAIVE_ANCHOR_PPL_TEXT1.
    """
    from . import obq_ckpt
    from .fisher import capture_linear_inputs
    linear = linear_weight_tensors(tensors)
    t0 = time.time()
    print(f"capturing fp32 activations for {len(linear)} layers "
          f"(one forward)...", flush=True)
    inputs = capture_linear_inputs(tensors, token_ids)
    params = obq_ckpt.run_params(args.group_size, args.obq_damp,
                                 token_ids)
    status = obq_ckpt.quantize_missing(linear, inputs, args.obq_ckpt_dir,
                                       params,
                                       max_layers=args.obq_max_layers)
    print(f"OBQ checkpoint run on [{text_label}]: {status['done']}/"
          f"{status['total']} layers checkpointed "
          f"({status['new']} new this run); "
          f"resume point: {status['next'] or 'COMPLETE'}")
    if status["next"] is not None:
        dt = time.time() - t0
        print(f"({dt:.1f} s; re-run to continue from the resume point)")
        return
    qw = obq_ckpt.assemble_full_model(tensors, args.obq_ckpt_dir)
    ppl = perplexity_of(qw, token_ids)
    dt = time.time() - t0
    from .schemes import _scale_overhead
    gs = args.group_size
    bpw = 2.0 + 4 * 8 / gs + _scale_overhead(1, gs)  # matches obq.py
    print(f"\n{'int2_kmeans_q8+obq-all (checkpointed)':<36} "
          f"ppl {ppl:8.2f}  bpw {bpw:5.3f}  ({dt:5.1f} s)")
    print(f"naive anchor (int2_kmeans_q8 g128, {text_label}): "
          f"{NAIVE_ANCHOR_PPL_TEXT1:.2f}")
    if ppl < 400:
        print("verdict: BELOW the ~400 line — the fidelity path is alive; "
              "follow up on the roadmap's next OBQ items")
    elif ppl < NAIVE_ANCHOR_PPL_TEXT1:
        print("verdict: beats the naive anchor but still deep in collapse "
              "territory — directional only, do not over-read")
    else:
        print("verdict: AT or ABOVE the naive anchor — the Hessian "
              "second-order story is exhausted on GPT-2 124M at this "
              "scale; log the honest negative per the roadmap")


def perplexity_of(tensors: dict, token_ids) -> float:
    """Perplexity of a float32 weight dict on a token id sequence."""
    return GPT2(tensors).perplexity(token_ids)


def run_sweep(tensors: dict, token_ids, scheme_names: list,
              group_size: int = GROUP_SIZE,
              sample_weights: dict | None = None,
              quantize_embeddings: bool = False,
              only_names: set | None = None) -> list:
    """Quantize + forward per scheme; results sorted by perplexity asc."""
    results = []
    for name in scheme_names:
        t0 = time.time()
        qw, bpw = quantize_model(tensors, name, group_size,
                                 sample_weights=sample_weights,
                                 quantize_embeddings=quantize_embeddings,
                                 only_names=only_names)
        ppl = perplexity_of(qw, token_ids)
        dt = time.time() - t0
        results.append({"scheme": name, "ppl": ppl, "bpw": bpw,
                        "secs": dt})
        print(f"{name:<22} ppl {ppl:8.2f}  bpw {bpw:5.3f}  ({dt:5.1f} s)",
              flush=True)
    results.sort(key=lambda r: r["ppl"])
    return results


def print_report(results: list) -> None:
    print(f"\n{'scheme':<22}{'ppl':>10}{'bpw':>8}")
    print("-" * 42)
    for r in results:
        print(f"{r['scheme']:<22}{r['ppl']:>10.2f}{r['bpw']:>8.3f}")


def run_multitext_sweep(tensors: dict, texts: list, scheme_names: list,
                        group_size: int = GROUP_SIZE,
                        use_fisher: bool = False,
                        quantize_embeddings: bool = False,
                        only_names: set | None = None) -> dict:
    """Quantize once per scheme, evaluate on every (label, ids) text.

    Returns {scheme: {"bpw": float, "ppls": {label: ppl}, "secs": float}}.
    Quantization does not depend on the eval text, so one quantized
    weight dict serves all texts. With use_fisher the importance weights
    are captured on the FIRST text only (documented choice; per-text
    capture would re-quantize per text).
    """
    results = {}
    for name in scheme_names:
        t0 = time.time()
        sample_weights = None
        if use_fisher:
            from .fisher import per_weight_importance
            print(f"collecting diagonal Fisher weights on "
                  f"{texts[0][0]} (first eval text only)...", flush=True)
            sample_weights = per_weight_importance(tensors, texts[0][1])
        qw, bpw = quantize_model(tensors, name, group_size,
                                 sample_weights=sample_weights,
                                 quantize_embeddings=quantize_embeddings,
                                 only_names=only_names)
        ppls = {}
        for label, ids in texts:
            ppl = perplexity_of(qw, ids)
            ppls[label] = ppl
            print(f"{name:<22} [{label}] ppl {ppl:8.2f}", flush=True)
        dt = time.time() - t0
        results[name] = {"bpw": bpw, "ppls": ppls, "secs": dt}
    return results


def aggregate_multitext(results: dict) -> list:
    """Per-scheme mean/std of ppl over the eval texts, sorted by mean
    ascending. Adds x_fp32 = mean / fp32-mean when the fp32 reference is
    in the results (the methodology item's discrimination denominator);
    None otherwise. std is the population std (2-3 texts, ddof=0)."""
    rows = []
    for scheme, r in results.items():
        ppls = np.array(list(r["ppls"].values()), dtype=np.float64)
        rows.append({"scheme": scheme, "bpw": r["bpw"],
                     "ppls": dict(r["ppls"]),
                     "mean": float(np.mean(ppls)),
                     "std": float(np.std(ppls))})
    fp32 = next((r for r in rows if r["scheme"] == FP32_SCHEME), None)
    for r in rows:
        r["x_fp32"] = (r["mean"] / fp32["mean"]) if fp32 else None
    rows.sort(key=lambda r: r["mean"])
    return rows


def print_multitext_report(rows: list, labels: list) -> None:
    show_x = bool(rows) and rows[0]["x_fp32"] is not None
    head = (f"\n{'scheme':<22}{'bpw':>8}"
            + "".join(f"{l:>12}" for l in labels)
            + f"{'mean':>10}{'std':>8}")
    if show_x:
        head += f"{'x_fp32':>8}"
    print(head)
    print("-" * len(head))
    for r in rows:
        line = (f"{r['scheme']:<22}{r['bpw']:>8.3f}"
                + "".join(f"{r['ppls'][l]:>12.2f}" for l in labels)
                + f"{r['mean']:>10.2f}{r['std']:>8.2f}")
        if show_x:
            line += f"{r['x_fp32']:>8.2f}"
        print(line)


def parse_args(argv: list | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="quantized perplexity probe")
    ap.add_argument("safetensors", help="path to model.safetensors")
    ap.add_argument("--eval-text", default="research/data/eval_text.txt",
                    help="single eval text (overridden by --eval-texts)")
    ap.add_argument("--eval-texts", default=None, metavar="A.TXT,B.TXT",
                    help="comma-separated eval texts; overrides --eval-text. "
                         "Each scheme is evaluated on every text and the "
                         "report shows per-text ppl plus mean/std (the "
                         "text-robustness protocol)")
    ap.add_argument("--tokenizer-dir", default="research/data/tokenizer")
    ap.add_argument("--schemes", default=",".join(DEFAULT_SWEEP),
                    help="comma-separated scheme names")
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--fisher", action="store_true",
                    help="reweight Lloyd centroid fits by the diagonal "
                         "empirical Fisher (per-input-channel activation "
                         "energy from one fp32 forward on the eval text); "
                         "only schemes whose encoder accepts sample_weight "
                         "(currently int2_kmeans_q8) are affected")
    ap.add_argument("--quantize-embeddings", action="store_true",
                    help="also quantize the embedding tables (wte, wpe) "
                         "with the same scheme; biases and LayerNorm stay "
                         "fp32. Ablation vs the default weight-only "
                         "protocol (see roadmap: is the 795 anchor limited "
                         "by the fp32 passthrough parts?)")
    ap.add_argument("--one-layer", default=None, metavar="NAME",
                    help="quantize only the named linear tensor (rest "
                         "fp32); for isolating one layer's contribution, "
                         "e.g. h.0.attn.c_attn.weight")
    ap.add_argument("--only-names", default=None, metavar="A,B",
                    help="comma-separated tensor names: restrict "
                         "quantization to exactly these tensors (rest "
                         "fp32); with --quantize-embeddings this isolates "
                         "one embedding table, e.g. --only-names "
                         "wte.weight. Mutually exclusive with --one-layer")
    ap.add_argument("--per-layer-schemes", default=None, metavar="SPEC",
                    help="per-block scheme assignment, e.g. "
                         "'0-5:int8_uniform,6-11:ternary_1step' (ranges "
                         "inclusive; every block must be assigned exactly "
                         "once). Each linear weight is quantized with its "
                         "block's scheme and the reported bpw is the "
                         "parameter-weighted average — the "
                         "sensitivity-adaptive bit-allocation harness "
                         "(see roadmap fidelity-retrospective item). "
                         "Mutually exclusive with --one-layer/--only-names/"
                         "--obq/--obq-all and with --sensitive-layers")
    ap.add_argument("--sensitive-layers", type=int, default=None, metavar="K",
                    help="K most Fisher-sensitive blocks (diag-Fisher "
                         "trace, captured on the first eval text) are "
                         "quantized with --sensitive-scheme, the rest with "
                         "--base-scheme. The reported bpw is the "
                         "parameter-weighted average; pick the schemes so "
                         "it matches the anchor bitrate for an "
                         "apples-to-apples comparison (e.g. K=5, "
                         "int8_uniform + ternary_1step -> ~2.378 vs the "
                         "2.375 int2_kmeans_q8 anchor). Same mutual "
                         "exclusions as --per-layer-schemes")
    ap.add_argument("--sensitive-scheme", default=None, metavar="NAME",
                    help="scheme for the top-K sensitive blocks "
                         "(requires --sensitive-layers)")
    ap.add_argument("--base-scheme", default=None, metavar="NAME",
                    help="scheme for the remaining blocks "
                         "(requires --sensitive-layers)")
    ap.add_argument("--sensitivity-mechanism", default=None, metavar="NAME",
                    help="run the activation-drift mechanism probe with "
                         "the named scheme (e.g. ternary_1step_ds): "
                         "fp32-activation Fisher trace vs leave-one-out "
                         "per-block damage vs trace recomputed on "
                         "quantized activations, with Spearman rank "
                         "correlations — the roadmap follow-up to the "
                         "sensitivity-adaptive negative. Standalone mode; "
                         "mutually exclusive with --schemes-driven runs, "
                         "--one-layer/--only-names, --fisher, "
                         "--quantize-embeddings, --obq/--obq-all and the "
                         "per-layer mixed-precision flags")
    ap.add_argument("--obq", action="store_true",
                    help="quantize the --one-layer tensor with OBQ "
                         "error compensation (GPTQ-style block update, "
                         "int2_kmeans_q8-equivalent codebook) instead of "
                         "the flat-array scheme; requires --one-layer")
    ap.add_argument("--obq-all", action="store_true",
                    help="run OBQ error compensation over ALL linear "
                         "layers (the full-model slice of the OBQ backlog "
                         "item; one shared capture forward, then per-layer "
                         "GPTQ-style block update); ignores --schemes; "
                         "mutually exclusive with --one-layer/--obq")
    ap.add_argument("--obq-damp", type=float, default=0.01,
                    help="Hessian damping fraction for --obq/--obq-all "
                         "(default 0.01, the GPTQ convention)")
    ap.add_argument("--obq-ckpt-dir", default=None, metavar="DIR",
                    help="checkpointed full-model OBQ protocol (requires "
                         "--obq-all): each layer's quantized weights are "
                         "written to DIR as it finishes, and later runs "
                         "skip already-done layers and resume from the "
                         "manifest — the session-timeout unblocker")
    ap.add_argument("--obq-max-layers", type=int, default=None, metavar="N",
                    help="with --obq-ckpt-dir, quantize at most N new "
                         "layers this run so a chunk fits the session "
                         "budget; 0 = status probe only; default: no cap")
    return ap.parse_args(argv)


def check_obq_args(args: argparse.Namespace) -> None:
    """Validate the --obq/--obq-all/--one-layer combination; SystemExit
    on misuse. Factored for testability."""
    if args.obq and not args.one_layer:
        raise SystemExit("--obq requires --one-layer NAME")
    if args.obq_all and (args.obq or args.one_layer):
        raise SystemExit("--obq-all is mutually exclusive with "
                         "--one-layer/--obq")
    if args.obq_ckpt_dir and not args.obq_all:
        raise SystemExit("--obq-ckpt-dir requires --obq-all")
    if args.obq_max_layers is not None and not args.obq_ckpt_dir:
        raise SystemExit("--obq-max-layers requires --obq-ckpt-dir")
    if args.obq_max_layers is not None and args.obq_max_layers < 0:
        raise SystemExit("--obq-max-layers must be >= 0")


def count_blocks(tensors: dict) -> int:
    """Number of transformer blocks = max block index + 1 over the
    linear weights. KeyError if any linear weight lacks a block index."""
    idxs = [layer_index_of(n) for n in linear_weight_tensors(tensors)]
    if any(i is None for i in idxs):
        raise KeyError("a linear weight has no block index; per-layer "
                       "schemes are unsupported for this model")
    if not idxs:
        raise KeyError("no linear weights found; per-layer schemes "
                       "are unsupported for this model")
    return max(idxs) + 1


def check_mixed_args(args: argparse.Namespace) -> None:
    """Validate the --per-layer-schemes/--sensitive-layers combination;
    SystemExit on misuse. Factored for testability."""
    pls, sl = args.per_layer_schemes, args.sensitive_layers
    if pls and sl is not None:
        raise SystemExit("--per-layer-schemes and --sensitive-layers are "
                         "mutually exclusive")
    if (args.one_layer or args.only_names or args.obq or args.obq_all) \
            and (pls or sl is not None):
        raise SystemExit("--per-layer-schemes/--sensitive-layers are "
                         "mutually exclusive with "
                         "--one-layer/--only-names/--obq/--obq-all")
    if sl is not None:
        if not args.sensitive_scheme or not args.base_scheme:
            raise SystemExit("--sensitive-layers requires "
                             "--sensitive-scheme and --base-scheme")
        for flag, val in (("--sensitive-scheme", args.sensitive_scheme),
                          ("--base-scheme", args.base_scheme)):
            if val not in SCHEMES:
                raise ValueError(f"unknown {flag} {val!r}; "
                                 f"have {sorted(SCHEMES)}")
    if (args.sensitive_scheme or args.base_scheme) and sl is None:
        raise SystemExit("--sensitive-scheme/--base-scheme require "
                         "--sensitive-layers")
    if args.fisher and (pls or sl is not None):
        raise SystemExit("--fisher is not supported with per-layer "
                         "schemes (sensitivity selection runs its own "
                         "Fisher capture)")
    if args.quantize_embeddings and (pls or sl is not None):
        raise SystemExit("--quantize-embeddings is not supported with "
                         "per-layer schemes (the embedding probe stays "
                         "a separate protocol)")


def check_mechanism_args(args: argparse.Namespace) -> None:
    """Validate --sensitivity-mechanism; SystemExit on misuse. Factored
    for testability."""
    mech = args.sensitivity_mechanism
    if mech is None:
        return
    if mech not in SCHEMES:
        raise ValueError(f"unknown --sensitivity-mechanism scheme "
                         f"{mech!r}; have {sorted(SCHEMES)}")
    if (args.one_layer or args.only_names or args.obq or args.obq_all
            or args.per_layer_schemes or args.sensitive_layers
            is not None or args.fisher or args.quantize_embeddings):
        raise SystemExit("--sensitivity-mechanism is a standalone mode; "
                         "it is mutually exclusive with --one-layer/"
                         "--only-names/--fisher/--quantize-embeddings/"
                         "--obq/--obq-all/--per-layer-schemes/"
                         "--sensitive-layers")


def resolve_layer_schemes(args: argparse.Namespace, tensors: dict,
                          token_ids, text_label: str) -> tuple:
    """Resolve the mixed-precision plan into ({block: scheme}, label).

    --per-layer-schemes parses directly. --sensitive-layers captures the
    diag-Fisher trace on `token_ids` (the first eval text — the same
    calibration-on-eval caveat as --fisher) and assigns the K most
    sensitive blocks to --sensitive-scheme, the rest to --base-scheme.
    """
    n = count_blocks(tensors)
    if args.per_layer_schemes:
        assignment = parse_layer_schemes(args.per_layer_schemes, n)
        return assignment, "mix[" + args.per_layer_schemes.strip() + "]"
    from .fisher import layer_fisher_trace
    print(f"capturing diag-Fisher trace on {text_label} (one fp32 "
          "forward)...", flush=True)
    trace = layer_fisher_trace(tensors, token_ids)
    try:
        top = sensitive_top_k(trace, args.sensitive_layers)
    except ValueError as e:
        raise SystemExit(f"--sensitive-layers: {e}")
    ranked = sorted(trace, key=lambda i: (-trace[i], i))
    print("block sensitivity ranking (trace desc; * = top-K):",
          flush=True)
    for i in ranked:
        mark = "*" if i in top else " "
        print(f"  {mark} block {i:2d}: trace {trace[i]:.6g}", flush=True)
    top_set = set(top)
    assignment = {i: (args.sensitive_scheme if i in top_set
                      else args.base_scheme)
                  for i in range(n)}
    label = (f"mix[sensitive-top{args.sensitive_layers}"
             f"[{args.sensitive_scheme}]+rest[{args.base_scheme}]]")
    return assignment, label


def run_mixed(tensors: dict, texts: list, assignment: dict, label: str,
              group_size: int = GROUP_SIZE) -> tuple:
    """Quantize once with the per-layer assignment, evaluate perplexity
    on every text. Returns (label, entry) where entry matches the
    run_multitext_sweep value shape {"bpw", "ppls", "secs"}."""
    t0 = time.time()
    qw, avg_bpw = quantize_model_per_layer(tensors, assignment,
                                           group_size)
    print(f"{label}: avg bpw {avg_bpw:.3f}", flush=True)
    ppls = {}
    for text_label, ids in texts:
        ppl = perplexity_of(qw, ids)
        ppls[text_label] = ppl
        print(f"{label:<22} [{text_label}] ppl {ppl:8.2f}", flush=True)
    return label, {"bpw": avg_bpw, "ppls": ppls,
                   "secs": time.time() - t0}


def leave_one_out_ppl(tensors: dict, token_ids, scheme_name: str,
                      group_size: int = GROUP_SIZE,
                      blocks: list | None = None) -> list:
    """Per-block damage probe: quantize ONLY block b's linears with
    `scheme_name` (rest fp32), measure perplexity.

    Returns [{"block": b, "ppl": float, "bpw": float}] for the requested
    blocks (default: all blocks ascending). bpw is the scheme's own
    bitrate — the same scheme quantizes every probed block, so the
    blocks are comparable by construction and no matched-bitrate
    adjustment is needed. Uses quantize_model's only_names targeting,
    so the weight-only protocol (embeddings fp32) is unchanged.
    """
    if scheme_name not in SCHEMES:
        raise KeyError(f"unknown scheme {scheme_name!r}; "
                       f"have {sorted(SCHEMES)}")
    groups = block_linear_names(tensors)
    want = sorted(groups) if blocks is None else list(blocks)
    unknown = [b for b in want if b not in groups]
    if unknown:
        raise ValueError(f"unknown blocks {unknown}; have "
                         f"{sorted(groups)}")
    rows = []
    for b in want:
        t0 = time.time()
        qw, bpw = quantize_model(tensors, scheme_name, group_size,
                                 only_names=groups[b])
        ppl = perplexity_of(qw, token_ids)
        rows.append({"block": b, "ppl": ppl, "bpw": bpw,
                     "secs": time.time() - t0})
        print(f"leave-one-out block {b:2d}: ppl {ppl:8.2f}  "
              f"({rows[-1]['secs']:5.1f} s)", flush=True)
    return rows


def trace_on_quantized(tensors: dict, token_ids, scheme_name: str,
                       group_size: int = GROUP_SIZE) -> dict:
    """Diag-Fisher trace captured on a FULLY-quantized model's
    activations (every linear weight at `scheme_name`).

    The direct test of the activation-drift hypothesis: if this ranking
    differs from the fp32-activation ranking, the Fisher trace used for
    bit allocation no longer describes the model it is allocating bits
    for once quantization is applied. Lazily imports fisher to match
    the module's existing import pattern.
    """
    if scheme_name not in SCHEMES:
        raise KeyError(f"unknown scheme {scheme_name!r}; "
                       f"have {sorted(SCHEMES)}")
    from .fisher import layer_fisher_trace
    qw, _ = quantize_model(tensors, scheme_name, group_size)
    return layer_fisher_trace(qw, token_ids)


def run_sensitivity_mechanism(tensors: dict, token_ids, scheme_name: str,
                              group_size: int = GROUP_SIZE,
                              blocks: list | None = None) -> dict:
    """The activation-drift mechanism probe (roadmap follow-up to the
    sensitivity-adaptive negative).

    Three measurements on the same eval text:
      1. fp32 baseline perplexity + fp32-activation Fisher trace (the
         ranking the failed experiment allocated bits by).
      2. leave-one-out ppl damage per block (`scheme_name` on one
         block, rest fp32) — the ground-truth block-damage ranking.
      3. Fisher trace recomputed on the fully-quantized model's
         activations — the drift test.
    Returns {"fp32_ppl", "trace_fp32", "leave_one_out" (rows),
    "trace_quantized", "rho_damage", "rho_drift"} where
      rho_damage = spearman(fp32-trace order, leave-one-out damage order)
      rho_drift  = spearman(fp32-trace order, quantized-trace order).
    A low rho_damage means the fp32 ranking never predicted block-wise
    damage; a low rho_drift means quantization shifts the activation
    distribution enough to invalidate the ranking used for allocation.
    `blocks` restricts the leave-one-out sweep (None = all blocks); it
    exists so the gated suite test can probe cheaply without running
    all 12 blocks.
    """
    from .fisher import layer_fisher_trace
    print("sensitivity-mechanism probe: fp32 baseline (one forward)...",
          flush=True)
    fp32_ppl = perplexity_of(tensors, token_ids)
    print(f"fp32 baseline ppl {fp32_ppl:.2f}", flush=True)
    print("capturing fp32-activation diag-Fisher trace...", flush=True)
    trace_fp32 = layer_fisher_trace(tensors, token_ids)
    print(f"leave-one-out damage sweep with {scheme_name} "
          f"(one block at a time, rest fp32)...", flush=True)
    loo_rows = leave_one_out_ppl(tensors, token_ids, scheme_name,
                                 group_size, blocks=blocks)
    print(f"capturing diag-Fisher trace on {scheme_name}-quantized "
          "activations (full-model quantize + one forward)...",
          flush=True)
    trace_q = trace_on_quantized(tensors, token_ids, scheme_name,
                                 group_size)
    order_fp32 = ordered_desc(trace_fp32)
    probed = [r["block"] for r in loo_rows]
    order_damage = ordered_desc({r["block"]: r["ppl"]
                                 for r in loo_rows})
    order_drift = ordered_desc(trace_q)
    # The correlations compare the probed blocks only: with a restricted
    # `blocks` subset the fp32/quantized orders must be filtered to the
    # same set or spearman_rho would compare different element sets.
    fp32_sub = [i for i in order_fp32 if i in set(probed)]
    drift_sub = [i for i in order_drift if i in set(probed)]
    return {
        "fp32_ppl": fp32_ppl,
        "trace_fp32": trace_fp32,
        "leave_one_out": loo_rows,
        "trace_quantized": trace_q,
        "order_fp32": order_fp32,
        "order_damage": order_damage,
        "order_drift": order_drift,
        "rho_damage": spearman_rho(fp32_sub, order_damage),
        "rho_drift": spearman_rho(fp32_sub, drift_sub),
    }


def _qualify_rho(rho: float) -> str:
    if rho >= 0.7:
        return "tracks"
    if rho >= 0.3:
        return "weakly tracks"
    return "does NOT track"


def print_mechanism_report(res: dict, scheme_name: str) -> None:
    """Print the three rankings, both rank correlations, and the
    qualitative verdict. Thresholds (0.7/0.3) are labeled as
    qualitative reads, not decision rules — the numbers are the
    result."""
    fp32 = res["fp32_ppl"]
    print(f"\n=== sensitivity-mechanism report ({scheme_name}) ===")
    print(f"fp32 baseline ppl: {fp32:.2f}")
    probed = sorted(r["block"] for r in res["leave_one_out"])
    if probed != sorted(res["trace_fp32"]):
        print(f"(leave-one-out restricted to blocks {probed})")
    print("\nfp32-activation Fisher trace ranking (block: trace):")
    for i in res["order_fp32"]:
        print(f"  block {i:2d}: {res['trace_fp32'][i]:.6g}")
    print("\nleave-one-out damage ranking (block: ppl, delta vs fp32):")
    by_block = {r["block"]: r for r in res["leave_one_out"]}
    for i in res["order_damage"]:
        ppl = by_block[i]["ppl"]
        print(f"  block {i:2d}: ppl {ppl:8.2f}  "
              f"delta {ppl - fp32:+8.2f}")
    print("\nquantized-activation Fisher trace ranking (block: trace):")
    for i in res["order_drift"]:
        print(f"  block {i:2d}: {res['trace_quantized'][i]:.6g}")
    rd, rr = res["rho_damage"], res["rho_drift"]
    print(f"\nrho(fp32-trace, leave-one-out damage) = {rd:+.3f} -> "
          f"fp32 ranking {_qualify_rho(rd)} real block damage")
    print(f"rho(fp32-trace, quantized-activation trace) = {rr:+.3f} -> "
          f"quantization {'shifts' if rr < 0.7 else 'does not shift'} "
          "the sensitivity ranking")
    print("caveats: collapse territory (ppl deltas are directional); "
          "calibrated on the eval text (no held-out corpus); GPT-2 "
          "124M scale")


def main() -> None:
    args = parse_args()
    if args.one_layer and args.only_names:
        raise SystemExit("--one-layer and --only-names are mutually "
                         "exclusive")
    if args.only_names and (args.obq or args.obq_all):
        raise SystemExit("--only-names is not supported with "
                         "--obq/--obq-all (OBQ has its own targeting)")
    tensors = read_safetensors(args.safetensors)
    mats = linear_weight_tensors(tensors)
    print(f"{len(mats)} linear matrices quantized, group size "
          f"{args.group_size}")
    if args.quantize_embeddings:
        print("quantize_embeddings ON: embedding tables (wte, wpe) also "
              "quantized with the same scheme; biases/LN stay fp32")
    texts = []
    for p in eval_text_paths(args):
        with open(p) as f:
            text = f.read()
        ids = GPT2Tokenizer(args.tokenizer_dir).encode(text)
        label = os.path.basename(p)
        texts.append((label, ids))
        print(f"eval text {label}: {len(ids)} tokens")
    check_obq_args(args)
    check_mixed_args(args)
    check_mechanism_args(args)
    if len(texts) > 1 and (args.obq_all or args.obq):
        print("note: OBQ probes run on the first eval text only "
              f"({texts[0][0]})")
    ids = texts[0][1]
    if args.sensitivity_mechanism is not None:
        if len(texts) > 1:
            print("note: --sensitivity-mechanism runs on the first eval "
                  f"text only ({texts[0][0]})")
        res = run_sensitivity_mechanism(tensors, ids,
                                        args.sensitivity_mechanism,
                                        args.group_size)
        print_mechanism_report(res, args.sensitivity_mechanism)
        return
    mixed_plan = None
    if args.per_layer_schemes or args.sensitive_layers is not None:
        mixed_plan = resolve_layer_schemes(args, tensors, ids, texts[0][0])
    if args.obq_all:
        if args.obq_ckpt_dir:
            main_obq_ckpt(args, tensors, ids, texts[0][0])
            return
        t0 = time.time()
        print(f"OBQ-quantizing all linear layers "
              f"(damp {args.obq_damp})...", flush=True)
        qw, bpw = quantize_model_obq(tensors, ids,
                                     args.group_size, args.obq_damp)
        ppl = perplexity_of(qw, ids)
        dt = time.time() - t0
        print(f"{'int2_kmeans_q8+obq-all':<22} ppl {ppl:8.2f}  "
              f"bpw {bpw:5.3f}  ({dt:5.1f} s)")
        return
    if args.obq:
        t0 = time.time()
        print(f"OBQ-quantizing {args.one_layer} "
              f"(damp {args.obq_damp})...", flush=True)
        qw, bpw = quantize_one_layer_obq(tensors, ids, args.one_layer,
                                        args.group_size, args.obq_damp)
        ppl = perplexity_of(qw, ids)
        dt = time.time() - t0
        print(f"{'int2_kmeans_q8+obq':<22} ppl {ppl:8.2f}  "
              f"bpw {bpw:5.3f}  ({dt:5.1f} s)")
        return
    scheme_names = [s for s in args.schemes.split(",") if s]
    if args.one_layer:
        only = {args.one_layer}
        print(f"one-layer mode: only {args.one_layer} quantized, "
              "rest fp32")
    elif args.only_names:
        only = {n for n in args.only_names.split(",") if n}
        print(f"only-names mode: only {sorted(only)} quantized, "
              "rest fp32")
    else:
        only = None
    if len(texts) == 1:
        # Original single-text protocol: output format unchanged.
        sample_weights = None
        if args.fisher:
            from .fisher import per_weight_importance
            print("collecting diagonal Fisher weights (one fp32 "
                  "forward)...", flush=True)
            sample_weights = per_weight_importance(tensors, ids)
            print("fisher weighting on; applies to schemes accepting "
                  "sample_weight (int2_kmeans_q8)", flush=True)
        results = run_sweep(tensors, ids, scheme_names, args.group_size,
                            sample_weights=sample_weights,
                            quantize_embeddings=args.quantize_embeddings,
                            only_names=only)
        if mixed_plan is not None:
            assignment, mlabel = mixed_plan
            mlabel, entry = run_mixed(tensors, texts, assignment, mlabel,
                                      args.group_size)
            results.append({"scheme": mlabel,
                            "ppl": entry["ppls"][texts[0][0]],
                            "bpw": entry["bpw"], "secs": entry["secs"]})
            results.sort(key=lambda r: r["ppl"])
        print_report(results)
        return
    results = run_multitext_sweep(tensors, texts, scheme_names,
                                  args.group_size,
                                  use_fisher=args.fisher,
                                  quantize_embeddings=args.quantize_embeddings,
                                  only_names=only)
    if mixed_plan is not None:
        assignment, mlabel = mixed_plan
        mlabel, entry = run_mixed(tensors, texts, assignment, mlabel,
                                  args.group_size)
        results[mlabel] = entry
    rows = aggregate_multitext(results)
    print_multitext_report(rows, [label for label, _ in texts])


if __name__ == "__main__":
    main()
