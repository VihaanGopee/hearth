# Rung 4 validation recipe: Qwen3.6-35B-A3B-oQ2 under vmlx Smelt (M1 Pro)

**Goal:** rung 4 of the Hearth capability ladder — the best-benchmarked
~2-bit MoE quality point (oQ2, 64% MMLU on the 3.5 variant per the oMLX
docs) running on Justin's M1 Pro at **usable speed**, with its quality
measured against rung 3's IQ2_XXS. Rung 3 proved a 35B MoE fits resident;
rung 4 asks whether *better-quantized* 35B weights beat it on
intelligence-per-GB — the intelligence-first question, not a speed race.

**Pass criteria:** mean decode tok/s **≥ 10** **and** all 3 quality sanity
checks pass, **and** the model loads under `--smelt 50` without swap
pressure. The intelligence verdict is then the side-by-side battery score
vs rung 3 (`qwen35-35b-a3b`), not the pass/fail line.

## What to download

`Jundot/Qwen3.6-35B-A3B-oQ2` — **measured 13.10 GB** via the HF tree API
2026-09-21 (3 shards: 5.03 + 5.03 + 3.02 GB; oMLX v0.3.7, 2-bit affine,
group 64, MLX safetensors). On the Mac (do **not** download this on the
dev VM):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Jundot/Qwen3.6-35B-A3B-oQ2 \
  --local-dir ~/models/Qwen3.6-35B-A3B-oQ2
```

## Measured architecture (config.json text_config, HF raw fetch 2026-09-21)

Same hybrid MoE skeleton as the 3.5: **40 layers, hidden 2048, 16 attn
heads, GQA-2 KV heads, head_dim 256, `full_attention_interval=4`** (10
full-attention layers carry KV cache; 30 Gated DeltaNet layers carry
recurrent state), **256 experts, 8 routed + 1 shared per token**,
moe_intermediate 512, vocab 248320, max ctx 262144. Routed experts are
fused per layer as `switch_mlp.{gate,up,down}_proj` in the weights.

Quant layout (`quantization_config`, measured — this is the oQ
sensitivity-driven allocation, and the reason oQ2 holds quality at 2-bit):
routed experts **2-bit g64**; linear-attn projections 4–6 bit; self-attn
q/k/v/o **8-bit**; shared experts + gate **8-bit**; embeddings **8-bit**;
lm_head **8-bit**.

Three facts that matter for validation:

1. **Vision tower present** (333 tensors in the weights). vmlx ≥ 1.3.33
   forcibly disables VLM mode under `--smelt` (the partial-expert loader
   does not wire the vision tower, so image input produces garbage) —
   **text-only evaluation**, which is what the battery measures anyway.
2. **No MTP tensors in the weights** despite `mtp_num_hidden_layers=1`
   in the config — there is no MTP speculative path in vmlx for this
   quant. Do not expect MTP acceleration.
3. The oQ2 **64% MMLU figure is from the 3.5 variant** (oMLX docs). The
   3.6 variant is unmeasured — the Mac-side battery score IS the
   measurement.

## Expected RAM (fitcheck arch `qwen3.6-35b-a3b`, measured file size)

Routed experts: 40 × 256 × 3 × 2048 × 512 params @ 2-bit =
**8.05 GB** (`moe_expert_gb` in fitcheck). Backbone (attention @ 4–8 bit,
shared experts, embeddings, lm_head, vision tower, scales) is the
residual: 13.10 − 8.05 = **5.05 GB**.

Smelt keeps the backbone fully resident and pages a fraction of the
routed experts from SSD (`smelt_resident` in fitcheck):

| `--smelt-experts` | Resident experts | Resident weights | + KV @ 2048 f16 (0.04) + runtime ~1.0 | **Total** |
|---|---|---|---|---|
| 50 | 4.03 GB | 9.07 GB | 1.04 GB | **~10.1 GB** ✓ fits ~11 GB |
| 25 | 2.01 GB | 7.06 GB | 1.04 GB | **~8.1 GB** ✓ comfortable |

Reference benchmarks (vmlx README, jjang-ai — **M3 Ultra / 128 GB** on
`Nemotron-Cascade-2-30B-A3B-JANG_4M`, not the M1 Pro, not this model):
baseline 17,408 MB @ 89.9 tok/s; smelt-50 → 9,529 MB (−45%) @
**66.5 tok/s**; smelt-25 → 5,590 MB (−68%). Throughput scales inversely
with the loaded expert fraction (expert swaps hit SSD on the hot path);
the M1 Pro number is what the measurement below produces.

## Install + serve

```bash
pip install "vmlx[jang]"
vmlx serve ~/models/Qwen3.6-35B-A3B-oQ2 --smelt --smelt-experts 50
```

(If `vmlx serve` accepts an HF repo id directly, the download step can be
skipped — verify and note which form was used. Default when `--smelt` is
passed alone is 50.)

## Measure speed (OpenAI-compatible endpoint)

vmlx serves an OpenAI-compatible API, so `tools/measure_rung.py`
(Ollama-shaped) does not apply. Time streamed tokens instead:

```python
import json, time, urllib.request
def tps(prompt, n=256):
    body = {"model": "Qwen3.6-35B-A3B-oQ2", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": n, "stream": True}
    req = urllib.request.Request("http://localhost:8080/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0, ntok = None, 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                d = json.loads(line[6:])
                if d["choices"][0]["delta"].get("content"):
                    t0 = t0 or time.time(); ntok += 1
    return ntok / (time.time() - t0)
```

(Adjust the port/model name to what `vmlx serve` prints.) Run the 3 rung
prompts — short factual answer, code generation, ~150-word passage +
summary — 256 tokens each, temp 0; report per-prompt and mean tok/s.
Record resident RAM from Activity Monitor during the run: green memory
pressure with no swap is part of the pass.

## Measure quality

1. **Sanity (same 3 as rungs 1–3):** `17*23` → `391`; capital of France;
   repeat "quasar" exactly 3×. All at temp 0.
2. **Intelligence verdict:** the side-by-side battery score vs rung 3.
   `tools/eval_battery.py` is Ollama-shaped — until an OpenAI-compatible
   transport exists, run its 19 prompts manually against the vmlx endpoint
   and score with the same checks. A win on math/logic/honesty categories
   over `qwen35-35b-a3b` is what promotes this from "fits" to "rung 4".
3. **Caveat to measure, not assume:** Smelt biases routing toward
   resident experts. If quality under smelt-50 disappoints vs the 64%
   MMLU baseline expectation, re-run at `--smelt-experts 75` before
   concluding anything about the quant — the routing bias and the quant
   are two different variables.

## If below 10 tok/s or OOM — tune in this order

1. **`--smelt-experts 25`** (resident ~8.1 GB — the RAM headroom lever;
   expect a tok/s hit from more SSD swaps; that tradeoff is the
   measurement).
2. **Context 2048** (default serve ctx may be larger; KV is tiny here but
   the serve default matters for prefill).
3. **`--smelt-experts 75`** for speed if RAM allows (~11.1 GB resident —
   tight; only from a clean boot).
4. **Record, don't force:** if smelt-25 is still swapping or < 10 tok/s,
   log resident RAM, tok/s, and which smelt fraction — that closes the
   rung-4 path honestly and the ladder waits on the next release.

## Hearth wiring (not in this recipe)

Hearth has no transport for this yet, deliberately:

- `backend: mlx` (mlx-lm) **cannot** load oQ2's affine 2-bit MLX format —
  do not try; the load will fail on the quant layout.
- vmlx serves OpenAI-compatible HTTP, and Hearth has no
  OpenAI-compatible backend. The wiring work is a new backlog item
  (it also unblocks the `mlx_lm.server` transport for the MLX
  tool-calling follow-up): when it lands, the cascade big-model becomes
  `backend: openai-compat` → vmlx-served oQ2-smelt, with qwen3:8b
  resident as the small model.

## What a validated rung 4 unlocks

Rung 4 passing proves the quality-per-GB climb: the same 35B skeleton at
~2-bit *adaptive* quantization measurably out-reasons the GGUF IQ2_XXS
path, and Smelt paging makes a 13 GB model a 10 GB resident. Past rung 4,
the ladder is release-driven: oQ3-class quants (85% MMLU ≈ FP16 at
~15 GB — needs better paging or smaller skeletons), native ternary 70B,
or the next MoE generation.
