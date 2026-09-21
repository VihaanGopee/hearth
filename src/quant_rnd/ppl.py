"""Per-scheme quantized perplexity on GPT-2 124M (NumPy only).

Step 2b of the roadmap's "validate candidates on a real tiny model" item:
take the fp32 GPT-2 checkpoint, quantize every linear weight matrix
group-wise with each scheme, reconstruct in fp32, and measure perplexity
on research/data/eval_text.txt against the 53.50 fp32 reference
(see research/log/2026-09-21.md for the 2a session).

Scope notes (honest):
- Embeddings (wte, which is also the tied lm_head), position embeddings,
  biases, and LayerNorm weights/biases stay fp32. This is the standard
  weight-only quantization protocol; embedding-table quantization is a
  separate backlog item.
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
import time

import numpy as np

from .gpt2_forward import GPT2
from .gpt2_tokenizer import GPT2Tokenizer
from .realweights import read_safetensors, linear_weight_tensors
from .schemes import GROUP_SIZE, SCHEMES

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
                   group_size: int = GROUP_SIZE) -> tuple:
    """Quantize + reconstruct every linear weight with `scheme_name`.

    Returns (weights, bpw): a new {name: float32 ndarray} dict with the
    same keys and shapes as the input; linear matrices (the same set
    linear_weight_tensors selects) are quantized group-wise over the flat
    array and reconstructed, everything else passes through as a float32
    copy. The input dict is not modified.
    """
    if scheme_name not in SCHEMES:
        raise KeyError(f"unknown scheme {scheme_name!r}; "
                       f"have {sorted(SCHEMES)}")
    fn = SCHEMES[scheme_name]
    linear = set(linear_weight_tensors(tensors))
    out = {}
    bpw = None
    for name, t in tensors.items():
        t32 = np.ascontiguousarray(t, dtype=np.float32)
        if name in linear:
            q = fn(t32.ravel(), group_size=group_size)
            out[name] = q.reconstruct().reshape(t.shape)
            bpw = q.bpw
        else:
            out[name] = t32
    return out, bpw


def perplexity_of(tensors: dict, token_ids) -> float:
    """Perplexity of a float32 weight dict on a token id sequence."""
    return GPT2(tensors).perplexity(token_ids)


def run_sweep(tensors: dict, token_ids, scheme_names: list,
              group_size: int = GROUP_SIZE) -> list:
    """Quantize + forward per scheme; results sorted by perplexity asc."""
    results = []
    for name in scheme_names:
        t0 = time.time()
        qw, bpw = quantize_model(tensors, name, group_size)
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
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    tensors = read_safetensors(args.safetensors)
    mats = linear_weight_tensors(tensors)
    print(f"{len(mats)} linear matrices quantized, group size "
          f"{args.group_size}")
    with open(args.eval_text) as f:
        text = f.read()
    ids = GPT2Tokenizer(args.tokenizer_dir).encode(text)
    print(f"eval text: {len(ids)} tokens")
    scheme_names = [s for s in args.schemes.split(",") if s]
    results = run_sweep(tensors, ids, scheme_names, args.group_size)
    print_report(results)


if __name__ == "__main__":
    main()
