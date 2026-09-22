# Rung 4 validation recipe: Qwen3.5-35B-A3B-JANG_2S under vmlx Smelt (M1 Pro)

**Goal:** rung 4 of the Hearth capability ladder — the best-measured ~2-bit
MoE quality point that can actually page into 16 GB, running on Justin's
M1 Pro at **usable speed**, with its quality measured against rung 3's
IQ2_XXS GGUF. Rung 3 proved a 35B MoE fits resident; rung 4 asks whether
*better-quantized* 35B weights beat it on intelligence-per-GB — the
intelligence-first question, not a speed race. Bonus: JANG_2S is the 3.5
variant, the same base model as rung 3, so the comparison is
quant-vs-quant on identical weights.

**Pass criteria:** mean decode tok/s **≥ 10** **and** all 3 quality sanity
checks pass, **and** the model loads under `--smelt 50` without swap
pressure. The intelligence verdict is then the side-by-side battery score
vs rung 3 (`qwen35-35b-a3b`), not the pass/fail line.

## Why JANG_2S, not the oQ2 (supersedes the 2026-09-21 oQ2 plan)

The 2026-09-21 plan (`vmlx serve Jundot/Qwen3.6-35B-A3B-oQ2 --smelt 50`)
is **invalid — code-verified 2026-09-22** against jjang-ai/vmlx main:

- Smelt requires a JANG-format model. vmlx README: "Smelt requires an MoE
  model in JANG format. Not compatible with dense models (no experts to
  partial-load) or with non-JANG formats." The oQ2 weights are oQ (oMLX
  affine 2-bit), not JANG — `--smelt` on them is unsupported, and vmlx
  ships no oQ loader (loaders are jang / jangtq / laguna / mistral3 /
  zaya / qwen4_exp / dsv4).
- vmlx **does** serve JANGTQ/mxtq
  (`vmlx_engine/loaders/load_jangtq.py` re-exports
  `jang_tools.load_jangtq.load_jangtq_model`, bundled in vMLX's runtime —
  correcting the 2026-09-22 README-only read that said otherwise). But
  `--smelt` on JANGTQ raises an explicit error (vmlx#81 guard in
  `vmlx_engine/utils/smelt_loader.py`): JANGTQ's custom TurboQuantLinear
  modules (tq_packed + tq_norms + codebook + signs) can't be
  subset by the Smelt patches. JANGTQ is fully-resident-only: 10.74 GB
  text-only + runtime ≈ 12 GB — over budget. Dead on 16 GB.
- `JANGQ-AI/Qwen3.5-35B-A3B-JANG_2S` **is** a JANG profile (present in
  vmlx's `HYBRID_JANG_PROFILES`, `panel/src/renderer/src/lib/jangCompat.ts`)
  → Smelt-eligible, and at 2.17 bpw actual it is the best-measured
  ~2-bit 35B-A3B upload with a complete budget picture.

## What to download

`JANGQ-AI/Qwen3.5-35B-A3B-JANG_2S` — **11.65 GB measured** via
safetensors-header probe 2026-09-22 (HTTP Range only, zero weight bytes:
**10.75 GB text** (1611 tensors) + **0.89 GB vision** (333 fp16 tensors);
9.64 GB packed U32 + 2.01 GB F16 scales/metadata; mixed 2/4/6-bit,
**2.17 bpw actual** per jang_config.json).

Card quality claim (author-reported, 200-question MMLU subset): **65.5%**
vs ~20% for naive MLX 2-bit at 10 GB. Our battery is the arbiter, not the
card.

No manual download needed: `vmlx serve` accepts an HF repo id directly
(vmlx README: "Point it at a HuggingFace repo or local path and go";
doc-checked 2026-09-22). If you want a pinned local snapshot instead:
`huggingface-cli download JANGQ-AI/Qwen3.5-35B-A3B-JANG_2S --local-dir
~/models/Qwen3.5-35B-A3B-JANG_2S`, then point `vmlx serve` at the dir.

## Expected RAM (fitcheck arch `qwen3.5-35b-a3b`, measured file size)

Smelt disables VLM mode (the partial-expert loader doesn't wire the
vision tower — image input would produce garbage), so the served model is
**text-only: 10.75 GB**. Routed experts: 40 × 256 × 3 × 2048 × 512 @
2-bit = **8.05 GB** (`moe_expert_gb`); backbone (attention @ 4–6 bit,
shared experts, embeddings, lm_head, norms, scales) is the residual:
10.75 − 8.05 = **2.70 GB**.

Smelt keeps the backbone fully resident and pages a fraction of the
routed experts from SSD (`smelt_resident` in fitcheck):

| `--smelt-experts` | Resident experts | Resident weights | + KV @ 2048 f16 (0.04) + runtime ~1.0 | **Total** |
|---|---|---|---|---|
| 50 | 4.03 GB | 6.72 GB | 1.04 GB | **~7.8 GB** ✓ comfortable |
| 25 | 2.01 GB | 4.71 GB | 1.04 GB | **~5.8 GB** ✓ lots of headroom |

Reference benchmarks (vmlx README — **M3 Ultra / 128 GB** on
`Nemotron-Cascade-2-30B-A3B-JANG_4M`, not the M1 Pro, not this model):
baseline 17,408 MB @ 89.9 tok/s; smelt-50 → 9,529 MB (−45%) @
**66.5 tok/s**; smelt-25 → 5,590 MB (−68%). Throughput scales inversely
with the loaded expert fraction (expert swaps hit SSD on the hot path);
the M1 Pro number is what the measurement below produces.

## Install + serve

```bash
pip install "vmlx[jang]"
vmlx serve JANGQ-AI/Qwen3.5-35B-A3B-JANG_2S --smelt --smelt-experts 50
```

(HF repo id goes straight into `vmlx serve` — doc-verified 2026-09-22.
Default when `--smelt` is passed alone is 50. If the `[jang]` extra
doesn't pull the loader, `pip install jang-tools` — the vmlx JANGTQ
loader re-exports `jang_tools.load_jangtq`, same package family.)

GUI alternative: **MLX Studio** (jjang-ai/mlxstudio) is an Electron
wrapper around the vmlx server (its build clones jjang-ai/vmlx — same
format support, same Smelt). Signed + notarized DMG, macOS 14+ (M1 OK),
no terminal needed; its Server mode exposes the same OpenAI-compatible
API on localhost. Use it instead of the CLI if you prefer — the
measurements below are identical.

## Measure speed (OpenAI-compatible endpoint)

vmlx serves an OpenAI-compatible API, so `tools/measure_rung.py`
(Ollama-shaped) does not apply. Use `tools/measure_openai.py` — the same
rung protocol (3 speed prompts + 3 quality sanity checks + PASS/FAIL
report), but with streamed-token timing against `/v1/chat/completions`:

```bash
python3 tools/measure_openai.py --model JANGQ-AI/Qwen3.5-35B-A3B-JANG_2S --target 10 \
  --base-url http://localhost:8000
# --base-url accepts root, /v1, or the full chat/completions path
# (vmlx's CLI server default is port 8000, not the tool's 8080 default);
# --api-key or OPENAI_API_KEY if needed
```

Token counts come from the stream's `usage.completion_tokens` when the
server provides it, else from chunk counting (the report labels which).
`<think>` blocks are stripped from the quality-check text (Qwen thinking
tokens still count toward decode tok/s — they cost real decode time).
(Adjust the port/model name to what `vmlx serve` prints.) Run the 3 rung
prompts — short factual answer, code generation, ~150-word passage +
summary — 256 tokens each, temp 0; report per-prompt and mean tok/s.
Record resident RAM from Activity Monitor during the run: green memory
pressure with no swap is part of the pass.

## Measure quality

1. **Sanity (same 3 as rungs 1–3):** `17*23` → `391`; capital of France;
   repeat "quasar" exactly 3×. All at temp 0.
2. **Intelligence verdict:** the side-by-side battery score vs rung 3.
   `tools/eval_battery_openai.py` (2026-09-21) runs the same 19 prompts +
   checks as `tools/eval_battery.py` over the vmlx endpoint:
   `python3 tools/eval_battery_openai.py --models JANGQ-AI/Qwen3.5-35B-A3B-JANG_2S`
   then compare the scorecard against the recorded rung-3
   (`qwen35-35b-a3b`) numbers — 15/19 nominal (~17/19 effective). A win on
   math/logic/honesty categories over rung 3 is what promotes this from
   "fits" to "rung 4". Same-base-model comparison: any delta is the
   quant + Smelt routing bias, not the weights.
3. **Caveat to measure, not assume:** Smelt biases routing toward
   resident experts. If quality under smelt-50 disappoints vs the 65.5%
   MMLU-claim expectation, re-run at `--smelt-experts 75` before
   concluding anything about the quant — the routing bias and the quant
   are two different variables.
4. **Known Smelt failure mode — don't misread it as a bad quant:** if
   short deterministic prompts under `--smelt` return repeated junk
   tokens (or stop-by-length garbage) while the server started and loaded
   cleanly, that is the exact failure signature of vmlx#222 (closed; fix
   landed 2026-06-30): the Smelt path once skipped the internal JANG
   norm-shift correction for Qwen/Ornith artifacts. It is fixed in
   current vmlx, but treat it as a known regression surface: before
   concluding anything about the model, run the plain non-Smelt serve as
   a control. If plain serve is coherent and Smelt serve emits junk, it
   is a loader bug, not the quant — pin the vmlx version and report it.

## If below 10 tok/s or OOM — tune in this order

1. **`--smelt-experts 25`** (resident ~5.8 GB — the RAM headroom lever;
   expect a tok/s hit from more SSD swaps; that tradeoff is the
   measurement).
2. **Context 2048** (default serve ctx may be larger; KV is tiny here but
   the serve default matters for prefill).
3. **`--smelt-experts 75`** for quality if RAM allows (~8.7 GB resident —
   still comfortable; separates routing-bias effects from quant effects).
4. **Record, don't force:** if smelt-25 is still swapping or < 10 tok/s,
   log resident RAM, tok/s, and which smelt fraction — that closes the
   rung-4 path honestly and the ladder waits on the next release.

## Hearth wiring (after rung-4 validates)

When vmlx (or MLX Studio's Server mode) is serving, Hearth speaks to it
via `backend: openai` (`src/openai_backend.py`, landed 2026-09-21 —
Mac-side transport still untested). Drop this into `config.yaml`
(also mirrored as a pinned, commented example there) and set
`backend: cascade`:

```yaml
backend: cascade
cascade:
  # small omitted: defaults to the [ollama] block (qwen3:8b, resident)
  router: heuristic        # or "verify"; see src/cascade.py
  big:
    backend: openai
    openai:
      base_url: "http://localhost:8000"   # vmlx serve default port
      model: "JANGQ-AI/Qwen3.5-35B-A3B-JANG_2S"
      temperature: 0.6
```

The big client builds lazily on first escalation, so the JANG_2S Smelt
model costs no resident RAM until a hard query actually fires. Only turn
this on after the "Measure speed" / "Measure quality" items above pass —
the cascade inherits whatever quality and tok/s the served model has.
This wiring also unblocks the `mlx_lm.server` transport for the MLX
tool-calling follow-up.

## What a validated rung 4 unlocks

Rung 4 passing proves the quality-per-GB climb: the same 35B skeleton at
~2-bit *adaptive* quantization under Smelt paging measurably out-reasons
the GGUF IQ2_XXS path at lower resident RAM (7.8 vs 10.66 GB). Past
rung 4, the ladder is release-driven: a JANG_2L/JANG_2M upload of the 3.6
variant, native ternary 70B, or the next MoE generation. The oQ2 weights
stay on the watch list — if vmlx ever ships an oQ loader, or JANGQ-AI
ships a JANG-format 3.6 at ~2-bit, the recipe gets a second candidate.
