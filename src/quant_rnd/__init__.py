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
"""
from .schemes import (
    QuantResult,
    quantize_ternary_uniform,
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "QuantResult",
    "quantize_ternary_uniform",
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
]
