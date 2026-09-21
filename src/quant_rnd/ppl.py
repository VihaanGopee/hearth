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

Usage: python3 -m src.quant_rnd.ppl <model.safetensors> [--schemes ...]
"""
import argparse
import inspect
import time

import numpy as np

from .gpt2_forward import GPT2
from .gpt2_tokenizer import GPT2Tokenizer
from .realweights import read_safetensors, linear_weight_tensors
from .schemes import GROUP_SIZE, SCHEMES

# Mirrors the exclusion list in realweights.linear_weight_tensors: tensors
# whose name matches one of these markers are the embedding tables that
# the weight-only protocol normally leaves in fp32.
EMBEDDING_MARKERS = ("embed", "wte", "wpe")


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
    """
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


def parse_args(argv: list | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="quantized perplexity probe")
    ap.add_argument("safetensors", help="path to model.safetensors")
    ap.add_argument("--eval-text", default="research/data/eval_text.txt")
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
    return ap.parse_args(argv)


def check_obq_args(args: argparse.Namespace) -> None:
    """Validate the --obq/--obq-all/--one-layer combination; SystemExit
    on misuse. Factored for testability."""
    if args.obq and not args.one_layer:
        raise SystemExit("--obq requires --one-layer NAME")
    if args.obq_all and (args.obq or args.one_layer):
        raise SystemExit("--obq-all is mutually exclusive with "
                         "--one-layer/--obq")


def main() -> None:
    args = parse_args()
    tensors = read_safetensors(args.safetensors)
    mats = linear_weight_tensors(tensors)
    print(f"{len(mats)} linear matrices quantized, group size "
          f"{args.group_size}")
    if args.quantize_embeddings:
        print("quantize_embeddings ON: embedding tables (wte, wpe) also "
              "quantized with the same scheme; biases/LN stay fp32")
    with open(args.eval_text) as f:
        text = f.read()
    ids = GPT2Tokenizer(args.tokenizer_dir).encode(text)
    print(f"eval text: {len(ids)} tokens")
    check_obq_args(args)
    if args.obq_all:
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
    only = {args.one_layer} if args.one_layer else None
    if only:
        print(f"one-layer mode: only {args.one_layer} quantized, "
              "rest fp32")
    sample_weights = None
    if args.fisher:
        from .fisher import per_weight_importance
        print("collecting diagonal Fisher weights (one fp32 forward)...",
              flush=True)
        sample_weights = per_weight_importance(tensors, ids)
        print("fisher weighting on; applies to schemes accepting "
              "sample_weight (int2_kmeans_q8)", flush=True)
    results = run_sweep(tensors, ids, scheme_names, args.group_size,
                        sample_weights=sample_weights,
                        quantize_embeddings=args.quantize_embeddings,
                        only_names=only)
    print_report(results)


if __name__ == "__main__":
    main()
