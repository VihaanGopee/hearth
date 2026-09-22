"""estimate_fit tool: can this model fit in this machine's RAM?

Used by the agent (and by Justin) to answer the 70B-quest question with
math instead of vibes: weights + KV cache + runtime vs the usable budget.
All figures are estimates.
"""
from __future__ import annotations
from .registry import register_tool

# Approximate bits per weight for common GGUF quantization types.
QUANT_BPW = {
    "f16": 16.0, "f32": 32.0,
    "q8_0": 8.5,
    "q6_k": 6.6, "q5_k_m": 5.7, "q5_0": 5.5,
    "q4_k_m": 4.9, "q4_k_s": 4.4, "q4_0": 4.5,
    "q3_k_m": 4.0, "q3_k_s": 3.4,
    "q2_k": 3.35,
    "iq4_xs": 4.4, "iq3_m": 3.7, "iq3_s": 3.66, "iq3_xxs": 3.1,
    "iq2_m": 2.7, "iq2_s": 2.56, "iq2_xs": 2.3, "iq2_xxs": 2.06,
    "iq1_m": 1.75, "iq1_s": 1.56,
    "tq2_0": 2.06, "tq1_0": 1.69,
}

# KV-cache bytes per element for llama.cpp cache types (approx).
CACHE_BYTES = {"f16": 2.0, "q8_0": 1.15, "q4_0": 0.65, "q4_1": 0.65}

# (n_layer, n_kv_heads, head_dim) for well-known architectures. Approximate.
# Optional 4th element: n_kv_layers — how many layers actually carry a KV
# cache. Defaults to n_layer; hybrid architectures (full attention mixed
# with linear-attention/SSM layers) only pay KV cache on the former.
ARCHES = {
    "llama-70b": (80, 8, 128),
    "llama-8b": (32, 8, 128),
    "llama-3b": (28, 8, 128),
    "qwen3-32b": (64, 8, 128),
    "qwen3-14b": (40, 8, 128),
    "qwen3-8b": (36, 8, 128),
    "qwen3-4b": (36, 8, 128),
    "qwen3-30b-a3b": (48, 8, 128),   # MoE; weights counted on total params
    # Qwen3.5-35B-A3B: 40 layers, GQA-2 KV heads, head_dim 256 (from the
    # official config.json, fetched via the HF API 2026-09-21). Hybrid
    # architecture: only the 10 full-attention layers carry a KV cache;
    # the 30 Gated DeltaNet layers carry small recurrent state instead
    # (unmodeled here). MoE; weights counted on total params (34.65B),
    # decode traffic on active (~3.3B).
    "qwen3.5-35b-a3b": (40, 2, 256, 10),
    # Qwen3.6-35B-A3B: same hybrid MoE skeleton (40 layers, GQA-2 KV heads,
    # head_dim 256, full_attention_interval=4 -> 10 KV layers), per the
    # Jundot/Qwen3.6-35B-A3B-oQ2 config.json text_config (HF API raw fetch
    # 2026-09-21). Used for the rung-4 oQ2/Smelt recipe. The oQ2 weights
    # include a 333-tensor vision tower (unused for text eval).
    "qwen3.6-35b-a3b": (40, 2, 256, 10),
}


def estimate(params_b: float, quant: str, n_ctx: int,
             arch: str | None = None, n_layer: int | None = None,
             n_kv_heads: int | None = None, head_dim: int = 128,
             cache_type: str = "q8_0", budget_gb: float = 11.0) -> dict:
    q = quant.lower()
    if q not in QUANT_BPW:
        return {"ok": False,
                "error": f"Unknown quant '{quant}'. Known: {sorted(QUANT_BPW)}"}
    bpw = QUANT_BPW[q]
    weights_gb = params_b * 1e9 * bpw / 8 / 1e9

    n_kv_layers = n_layer  # layers carrying KV cache; hybrid archs override
    if arch:
        if arch not in ARCHES:
            return {"ok": False,
                    "error": f"Unknown arch '{arch}'. Known: {sorted(ARCHES)} "
                             f"or pass n_layer/n_kv_heads directly."}
        entry = ARCHES[arch]
        n_layer, n_kv_heads, head_dim = entry[0], entry[1], entry[2]
        n_kv_layers = entry[3] if len(entry) > 3 else n_layer
    if n_layer is None or n_kv_heads is None:
        return {"ok": False,
                "error": "Need arch= or n_layer= and n_kv_heads= for KV cache math."}
    cb = CACHE_BYTES.get(cache_type.lower(), 2.0)
    # 2x for K and V; only the layers that carry KV cache count (hybrid
    # architectures mix full-attention layers with SSM/linear layers).
    kv_bytes = 2 * n_kv_layers * n_kv_heads * head_dim * n_ctx * cb
    kv_gb = kv_bytes / 1e9
    runtime_gb = 0.5  # compute buffers, tokenizer, overhead
    total_gb = weights_gb + kv_gb + runtime_gb
    fits = total_gb <= budget_gb
    verdict = ("FITS" if fits else "DOES NOT FIT") + \
        f" the ~{budget_gb:.0f} GB usable budget " \
        f"({total_gb:.1f} GB needed: {weights_gb:.1f} weights + " \
        f"{kv_gb:.1f} KV cache @ {n_ctx} ctx + {runtime_gb:.1f} runtime)."
    return {
        "ok": True,
        "params_b": params_b, "quant": q, "bits_per_weight": bpw,
        "weights_gb": round(weights_gb, 2),
        "kv_cache_gb": round(kv_gb, 2),
        "runtime_gb": runtime_gb,
        "total_gb": round(total_gb, 2),
        "budget_gb": budget_gb,
        "fits": fits,
        "verdict": verdict,
    }


def moe_expert_gb(n_layer: int, n_experts: int, hidden_size: int,
                  moe_intermediate_size: int, expert_bits: float,
                  n_proj: int = 3) -> float:
    """GB of routed-expert weights for an MoE layer stack.

    Each routed expert is n_proj projections of (hidden_size x
    moe_intermediate_size) at expert_bits per weight. For
    Qwen3.5/3.6-35B-A3B: 40 layers x 256 experts x 3 x 2048 x 512 at
    2-bit = ~8.05 GB. (The oQ2 config fuses routed experts as
    switch_mlp.{gate,up,down}_proj; the parameter count is the same.)
    """
    params = n_layer * n_experts * n_proj * hidden_size * moe_intermediate_size
    return params * expert_bits / 8 / 1e9


def smelt_resident(total_gb: float, expert_gb: float,
                   smelt_frac: float) -> dict:
    """Resident-RAM estimate for vmlx Smelt (partial expert loading).

    Smelt keeps the backbone (attention, shared experts, embeddings,
    lm_head, norms, vision tower) fully resident and pages smelt_frac
    of the routed-expert weights from SSD; routing is biased toward the
    resident experts. The backbone is the residual total - experts, so
    this needs the MEASURED total file size, not a nominal bpw figure.

    Returns backbone_gb, resident_experts_gb, resident_gb (all weights;
    add KV cache + runtime separately).
    """
    if not 0.0 < smelt_frac <= 1.0:
        return {"ok": False,
                "error": f"smelt_frac must be in (0, 1], got {smelt_frac}"}
    if expert_gb > total_gb:
        return {"ok": False,
                "error": f"expert_gb ({expert_gb}) exceeds total_gb ({total_gb})"}
    backbone_gb = total_gb - expert_gb
    resident_experts_gb = expert_gb * smelt_frac
    resident_gb = backbone_gb + resident_experts_gb
    return {
        "ok": True,
        "backbone_gb": round(backbone_gb, 2),
        "resident_experts_gb": round(resident_experts_gb, 2),
        "resident_gb": round(resident_gb, 2),
        "smelt_frac": smelt_frac,
    }


def register(ctx: dict) -> None:
    budget = float(ctx.get("memory_budget_gb", 11.0))

    def estimate_fit(params_b: float, quant: str, n_ctx: int = 8192,
                     arch: str = "", n_layer: int = 0, n_kv_heads: int = 0,
                     cache_type: str = "q8_0"):
        return estimate(
            params_b, quant, n_ctx,
            arch=arch or None,
            n_layer=n_layer or None, n_kv_heads=n_kv_heads or None,
            cache_type=cache_type, budget_gb=budget)

    register_tool(
        "estimate_fit",
        "Estimate whether a model fits in this machine's RAM. Give params_b "
        "(billions), quant (e.g. q4_k_m, iq1_m, tq1_0), context length, and "
        "either arch (llama-70b, qwen3-8b, …) or n_layer/n_kv_heads. Returns "
        "weights + KV-cache math and a FITS / DOES NOT FIT verdict.",
        {"properties": {
            "params_b": {"type": "number"},
            "quant": {"type": "string"},
            "n_ctx": {"type": "integer"},
            "arch": {"type": "string"},
            "n_layer": {"type": "integer"},
            "n_kv_heads": {"type": "integer"},
            "cache_type": {"type": "string"},
        }, "required": ["params_b", "quant"]},
        estimate_fit)
