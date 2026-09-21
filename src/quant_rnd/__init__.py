"""Quantization R&D prototypes for the 70B-quest (avenue H).

Small-scale, CPU-only experiments: novel sub-2-bit / ternary schemes,
benchmarked by signal-to-quantization-noise ratio (SQNR) against honest
baselines on synthetic weight distributions that mimic real LLM stats
(Gaussian body + sparse large outliers).

Honest limits, stated up front:
- This is synthetic-tensor R&D, not real-model perplexity. SQNR on
  synthetic weights is a cheap proxy that lets us iterate fast; any
  scheme that "wins" here still needs validation on a real model (see
  roadmap backlog item "wire bench to real tiny model").
- bpw figures account for payload + scale/index overhead per 128-weight
  group, so cross-scheme comparisons are at (roughly) matched bitrates.
- realweights.py runs the same SQNR ranking on real model weights
  (dependency-free .safetensors parser) to check whether the synthetic
  ranking reproduces on real distributions. Still a proxy for perplexity,
  but it removes the "synthetic distribution" caveat.
- gpt2_tokenizer.py + gpt2_forward.py close the perplexity gap: a
  dependency-free GPT-2 byte-level BPE tokenizer and NumPy forward pass
  over research/data/gpt2.safetensors, giving a real fp32 perplexity
  reference on a few hundred tokens of text. ppl.py adds per-scheme
  quantized perplexity (step 2b); see research/log for the numbers.
"""
from .schemes import (
    QuantResult,
    quantize_ternary_uniform,
    quantize_ternary_lloyd,
    quantize_ternary_lloyd_ds,
    quantize_ternary_1step,
    quantize_ternary_1step_ds,
    quantize_int2_symmetric,
    quantize_int2_kmeans,
    quantize_int2_kmeans_q8,
    quantize_int2_outlier_retain,
    quantize_dual_scale_ternary,
    quantize_ternary_outlier,
    SCHEMES,
)


def __getattr__(name):
    # Lazy so `python3 -m src.quant_rnd.bench` doesn't double-import bench.
    if name in ("run_bench", "sqnr_db", "synthetic_weights"):
        from . import bench as _bench

        return getattr(_bench, name)
    if name in ("M1_PRO_MEM_BW_GBS", "N_PARAMS_70B", "decode_roofline_tps",
                "kv_cache_bytes", "weight_bytes", "ternary_opcount",
                "codebook_opcount", "equiv_adds", "measured_sparsity",
                "scheme_report", "print_report", "is_bandwidth_bound"):
        from . import opcount as _opcount

        return getattr(_opcount, name)
    if name in ("read_safetensors", "linear_weight_tensors", "sample_groups",
                "rank_on_real_weights"):
        from . import realweights as _rw

        return getattr(_rw, name)
    if name in ("GPT2", "load_gpt2", "gelu", "layer_norm", "attention"):
        from . import gpt2_forward as _gf

        return getattr(_gf, name)
    if name == "GPT2Tokenizer":
        from . import gpt2_tokenizer as _gt

        return getattr(_gt, name)
    if name in ("inverse_hessian", "quantize_layer_obq",
                "quantize_layers_obq", "hessian_weighted_sse"):
        from . import obq as _obq

        return getattr(_obq, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "QuantResult",
    "quantize_ternary_uniform",
    "quantize_ternary_lloyd",
    "quantize_ternary_lloyd_ds",
    "quantize_ternary_1step",
    "quantize_ternary_1step_ds",
    "quantize_int2_symmetric",
    "quantize_int2_kmeans",
    "quantize_int2_kmeans_q8",
    "quantize_int2_outlier_retain",
    "quantize_dual_scale_ternary",
    "quantize_ternary_outlier",
    "SCHEMES",
    "run_bench",
    "sqnr_db",
    "synthetic_weights",
    "M1_PRO_MEM_BW_GBS",
    "N_PARAMS_70B",
    "decode_roofline_tps",
    "kv_cache_bytes",
    "weight_bytes",
    "ternary_opcount",
    "codebook_opcount",
    "equiv_adds",
    "measured_sparsity",
    "scheme_report",
    "print_report",
    "is_bandwidth_bound",
    "read_safetensors",
    "linear_weight_tensors",
    "sample_groups",
    "rank_on_real_weights",
    "GPT2",
    "load_gpt2",
    "gelu",
    "layer_norm",
    "attention",
    "GPT2Tokenizer",
    "inverse_hessian",
    "quantize_layer_obq",
    "quantize_layers_obq",
    "hessian_weighted_sse",
]
