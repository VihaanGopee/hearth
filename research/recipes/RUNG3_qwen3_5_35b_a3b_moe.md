# Rung 3 validation recipe: Qwen3.5-35B-A3B (MoE) at usable speed (M1 Pro)

**Goal:** rung 3 of the Hearth capability ladder — a 35B-class MoE
sustaining **usable decode speed** on Justin's M1 Pro, **measured** (not
estimated), plus a quality sanity check. A rung counts only when both pass.

**Pass criteria:** mean decode tok/s **≥ 10** **and** all 3 quality sanity
checks pass. The script exits 0 and prints `RESULT: RUNG PASS`. Paste the
full output back as the measurement record. ≥ 15 is comfortable, ≥ 20 is
gravy.

Unlike rungs 1–2, the bar here is "usable", not 20 — per the
intelligence-first direction, speed is secondary and ~10+ tok/s is enough
for interactive use. The decode ceiling for a MoE is kernel-dependent (see
the roofline section), so 10 is the honest bar and anything above it is
gravy.

## What to download

The file is `Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf` from
`unsloth/Qwen3.5-35B-A3B-GGUF` — **measured 10.66 GB** (HF tree API,
2026-09-21).

> **Correction to the 2026-09-21 morning survey:** the survey quoted
> 10.6 GB for the **IQ2_M** file. Re-measured the same afternoon via the HF
> tree API: **IQ2_M = 11.39 GB, IQ2_XXS = 10.66 GB.** At 11.39 GB the IQ2_M
> file needs ~12.0 GB resident (weights + KV + runtime) and does not fit
> the ~11 GB usable budget — so the recipe's primary file is the IQ2_XXS
> variant (~2.46 effective bpw). IQ2_M is demoted to fallback.

On the Mac (do **not** download this on the dev VM):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
  Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf --local-dir ~/models
```

Load it into Ollama with a Modelfile so `tools/measure_rung.py` works
unchanged:

```
# ~/models/Modelfile.qwen35  (FROM is relative to the Modelfile)
FROM ./Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf
PARAMETER num_ctx 2048
```

```bash
cd ~/models && ollama create qwen35-35b-a3b -f Modelfile.qwen35
```

## Expected RAM (fitcheck arch `qwen3.5-35b-a3b`, measured file size)

Qwen3.5-35B-A3B: 34.65B total params, 40 layers, GQA-2 KV heads, head_dim
256 (from the official `Qwen/Qwen3.5-35B-A3B` config.json via the HF API,
2026-09-21). The 10.66 GB file is ~2.46 effective bpw (unsloth's dynamic
mix keeps some tensors above the nominal IQ2_XXS 2.06 bpw). Weights figures
below use the **measured file size**, not fitcheck's nominal bpw.

| Component | Size |
|---|---|
| Weights (UD-IQ2_XXS, measured file) | 10.66 GB |
| KV cache @ 2048 ctx, f16 | 0.04 GB |
| KV cache @ 2048 ctx, q8_0 | 0.02 GB |
| KV cache @ 2048 ctx, q4_0 | 0.01 GB |
| KV cache @ 4096 ctx, q8_0 | 0.05 GB |
| Runtime overhead | ~0.5 GB |
| **Total @ 2048, q8_0** | **~11.2 GB** |
| **Total @ 2048, f16** | **~11.2 GB** |
| **Total @ 4096, q8_0** | **~11.2 GB** |

> **Correction (2026-09-21):** an earlier version of this table applied the
> KV cache to all 40 layers (0.17 GB @ 2048 f16). Qwen3.5-35B-A3B is a
> hybrid architecture — only the 10 full-attention layers carry a KV
> cache; the 30 Gated DeltaNet layers carry small recurrent state instead
> (unmodeled here). Corrected figures above; `fitcheck` arch
> `qwen3.5-35b-a3b` now models this.

**This is the tightest rung on RAM.** The file alone is ~97% of the ~11 GB
usable budget; the whole thing only fits if macOS + everything else stays
under ~4.8 GB. Preconditions: clean boot (or at least close other apps),
then verify the full model is resident with `ollama ps` / Activity
Monitor. Any swap activity during the run fails the *file*, not the
measurement — record it and move to the fallbacks.

## Measure

With `ollama serve` running (start it with `OLLAMA_KV_CACHE_TYPE=q8_0` in
the environment — see rung 2's recipe for the `launchctl setenv` form):

```bash
cd ~/workspace/local-agent
python3 tools/measure_rung.py --model qwen35-35b-a3b --target 10 --num-ctx 2048
```

Same harness as rungs 1–2: POSTs to Ollama's `/api/generate`
(`stream=false`, temp 0), reads `eval_count`/`eval_duration` straight from
the server. 3 fixed speed prompts (short answer, code gen, ~150-word
passage + summary), 256 generated tokens each; per-prompt and mean decode
tok/s. Quality sanity: arithmetic (`17*23` → `391`), factual (capital of
France), instruction-following (repeat "quasar" exactly 3×), all at temp 0.

## Roofline sanity check (two bounds, not one number)

MoE decode streams **active** params per token, not the full file:
~3.3B active (8 routed + 1 shared expert per layer) at 2.46 bpw ≈ 1.02 GB
per token → a perfect-kernel ceiling of ≈ 200 / 1.02 ≈ **197 tok/s**.
That number is fiction — expert-gather overhead, 256-way routing, and
small per-expert GEMMs dominate real kernels, and no honest model lands
there.

The conservative floor: if the kernel streamed the whole 10.66 GB per
token → 200 / 10.66 ≈ **18.8 tok/s** — *below* the old 20 bar. The truth
is between the two. The one community datapoint (unverified, not M1 Pro):
**30 tok/s on a Mac mini M4** (120 GB/s bandwidth; the M1 Pro has 200).

That uncertainty is exactly why rung 3's bar is "usable speed" (10) rather
than 20: the ceiling is kernel-dependent and only the measurement counts.
If the M1 Pro lands near the M4 datapoint scaled by bandwidth, 20+ is in
reach — record whatever the harness prints, honestly.

## If below 10 tok/s or OOM — tune in this order

1. **RAM first — this rung is RAM-bound, not compute-bound.** Close
   everything; re-run from a clean boot if needed. `ollama ps` must show
   the full model resident and Activity Monitor's memory pressure must
   stay green. Swapping masquerades as slowness.
2. **Quantized KV cache** (recipe default: `OLLAMA_KV_CACHE_TYPE=q8_0`).
   `q4_0` is the next lever (~0.05 GB more at 2048 ctx) — small, but this
   rung has no slack.
3. **Context 2048** (already the recipe default). Do not cut to 1024 and
   call it a win — the rung's context must stay useful.
4. **Thread count.** `OLLAMA_NUM_THREAD` default (all cores) is usually
   right on the M1 Pro; only touch if something looks wrong.
5. **Fallback files** (each needs its own full harness run + quality
   sanity — a rung measured on a fallback still counts if the file and the
   sanity results are in the record):
   - `Qwen3.5-35B-A3B-UD-IQ2_M.gguf` (**11.39 GB**, ~11.9 GB resident @
     2048/q8_0): higher effective bpw (~2.63) if the XXS quality
     disappoints — **only if the Mac has the headroom**, it does not fit
     the standard ~11 GB budget.
   - `mtrpires/Qwen3.5-35B-A3B-IQK-RPi5-16GB` (11.38 GB disk): principled
     mixed-IQK recipe (attention + shared expert protected); same RAM
     class as IQ2_M, quality unverified.
   - `Qwen3.6-35B-A3B-MTP` UD-IQ2_XXS (**11.819 GB** per the 2026-09-21
     survey — verify the exact filename on HF before downloading): ships
     a native MTP speculative head; pair with Hearth's llamacpp backend
     `speculative: prompt_lookup` (config below).
   - oQ2 MLX (~12.6 GB, the benchmarked-quality ~2-bit option at 64%
     MMLU) via `vmlx --smelt 50` (see the backlog's Smelt item): the
     flash-paged path if nothing fits fully resident.
6. **If nothing is both resident and ≥ 10 tok/s:** that is a result, not a
   failure. Log the best measured number, which files were tried, and
   where RAM vs speed bit. Rung 3 then waits on the oQ2/Smelt path or a
   smaller MoE — do not force it.

Report which step (if any) was needed, and re-run the full harness after
each change — one variable at a time.

## Hearth config snippets (opt-in; default `backend: ollama` + `qwen3:8b` untouched)

llamacpp backend for the MTP variant with speculative prompt-lookup
decoding (the speculative support landed 2026-09-21; `prompt_lookup` needs
no draft model):

```yaml
backend: llamacpp
llamacpp:
  model_path: ~/models/Qwen3.6-35B-A3B-MTP-UD-IQ2_XXS.gguf
  num_ctx: 2048
  n_gpu_layers: -1
  cache_type_k: q8_0
  cache_type_v: q8_0
  speculative: prompt_lookup
  draft_n_tokens: 10
```

Or as the big model in a cascade (qwen3:8b stays resident; the 35B loads
lazily on first escalation):

```yaml
backend: cascade
cascade:
  router: heuristic
  big:
    backend: llamacpp
    llamacpp:
      model_path: ~/models/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf
      num_ctx: 2048
      n_gpu_layers: -1
      cache_type_k: q8_0
      cache_type_v: q8_0
```

## What a validated rung 3 unlocks

Rung 3 passing proves the MoE path on the M1 Pro: 35B-class weights
resident in 16 GB at interactive speed, with decode cost set by active
params. The next measurement is then **quality-per-GB** — IQ2_XXS vs dense
qwen3:8b on a real task battery (MMLU subset), because speed without
quality doesn't climb the ladder. Past rung 3, the climb goes through
bigger MoE / better quants (oQ2-class MLX, Smelt-paged 12 GB models), not
through denser models.
