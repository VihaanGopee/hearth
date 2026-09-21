# Rung 2 validation recipe: qwen3:14b at 20 tok/s (M1 Pro)

**Goal:** rung 2 of the Hearth capability ladder — a 14B-class model
sustaining **≥ 20 tokens/sec decode** on Justin's M1 Pro, **measured** (not
estimated), plus a quality sanity check. A rung counts only when both pass.

This recipe follows the rung-1 template
(`research/recipes/RUNG1_qwen3_8b_20tps.md`); the difference is that rung 2
is **tight against the bar** — the roofline ceiling sits at ~22 tok/s, so
expect to walk the tuning ladder before the harness prints PASS.

## What to download

```bash
ollama pull qwen3:14b        # ~9 GB, Q4_K_M default tag
```

## Expected RAM (from `src/tools/fitcheck.py`, arch `qwen3-14b`)

qwen3:14b is 14.7B params at 4.9 bpw (Q4_K_M) → **9.0 GB weights**.
Arch (40 layers, 8 KV heads, head_dim 128):

| Component | Size |
|---|---|
| Weights (Q4_K_M, 4.9 bpw) | ~9.0 GB |
| KV cache @ 2048 ctx, f16 | 0.34 GB |
| KV cache @ 4096 ctx, f16 | 0.67 GB |
| KV cache @ 2048 ctx, q8_0 | 0.19 GB |
| KV cache @ 4096 ctx, q8_0 | 0.39 GB |
| Runtime overhead | ~0.5 GB |
| **Total @ 2048, f16** | **~9.8 GB** |
| **Total @ 4096, f16** | **~10.2 GB** |
| **Total @ 8192, f16** | **~10.9 GB** |

Fits inside the ~10–11 GB usable budget at 2048/4096 ctx, but there is no
slack left — keep the context at 2048 for the first measurement and treat
4096 as the ceiling unless 2048 passes with room to spare. Verify resident
size on the Mac with `ollama ps`.

## Measure

With `ollama serve` running:

```bash
cd ~/workspace/local-agent
python3 tools/measure_rung.py --model qwen3:14b --target 20 --num-ctx 2048
```

Same harness as rung 1: POSTs to Ollama's `/api/generate`
(`stream=false`, temp 0), reads `eval_count`/`eval_duration` straight from
the server. 3 fixed speed prompts (short answer, code gen, ~150-word
passage + summary), 256 generated tokens each; per-prompt and mean decode
tok/s. Quality sanity: arithmetic (`17*23` → `391`), factual (capital of
France), instruction-following (repeat "quasar" exactly 3×), all at temp 0.

**Pass criteria:** mean decode tok/s ≥ 20 **and** all 3 quality checks pass.
The script exits 0 and prints `RESULT: RUNG PASS`. Paste the full output
back as the measurement record.

## Roofline sanity check (why this rung is a real test)

M1 Pro unified-memory bandwidth ≈ 200 GB/s. Decode is bandwidth-bound: each
token streams the weights once, so the ceiling is ≈ bandwidth / model size
≈ 200 / 9.0 ≈ **22 tok/s**. The 20 tok/s bar is ~90% of the ceiling —
there is almost no headroom for kernel overhead. Unlike rung 1 (bar at ~50%
of ceiling), this rung measures how efficient Ollama's Metal decode path
actually is at 14B scale. If it misses, the tuning ladder below is the
experiment, not an afterthought — record which step was needed.

## If below 20 tok/s — tune in this order

1. **Verify full Metal offload.** `ollama ps` should show the model at ~100%
   GPU. With 9.8 GB resident against an ~11 GB usable budget, a partial
   offload is the most likely way the model stays loaded while running
   slow — check this first.
2. **Reduce context to 2048** (already the recipe default above). Newer
   Ollama defaults to 4096; the recipe starts at 2048 to cap worst-case KV
   and attention cost. If 2048 fails, this lever is spent — move on, don't
   cut to 1024 and call it a win (the rung's context must stay useful).
3. **Quantized KV cache.** Stop Ollama, restart with
   `OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve` (or `launchctl setenv`
   `OLLAMA_KV_CACHE_TYPE q8_0` for the launchd service), re-run. Saves
   ~0.15 GB at 2048 ctx; small speed gain, mostly safety margin under
   memory pressure.
4. **Thread count.** `OLLAMA_NUM_THREAD` — the default (all efficiency +
   performance cores) is usually right on the M1 Pro; only touch if
   something looks wrong.
5. **Lighter quant tag** (last resort — changes quality, so re-run the full
   quality sanity section): a `Q4_K_S` variant (~8.0 GB weights, ceiling
   ≈ 25 tok/s) or `Q4_0` (~8.3 GB, ceiling ≈ 24 tok/s) if published for
   qwen3:14b. Record the quality-sanity outcome per tag — a rung measured
   on a lighter quant still counts, as long as the tag and the sanity
   results are in the record.
6. **If nothing reaches 20:** that is a result, not a failure. Log the
   best measured number and which steps were tried. The rung-2 fallback
   candidates (conditional, to be added to the backlog): qwen3:14b at
   IQ3_M (~6.8 GB, ceiling ≈ 29 tok/s — quality unverified at 3.7 bpw) or
   Qwen3-8B-class at 20 tok/s with 14B kept as a cascade big model.

Report which step (if any) was needed, and re-run the full harness after
each change — one variable at a time.

## What a validated rung 2 unlocks

Rung 2 passing proves Hearth's Mac path can serve a 14B-class model at
interactive speed — the ceiling is near the bar, so the measurement also
calibrates how much kernel headroom Ollama really has on this machine.
Rung 3 (35B-class MoE at usable speed) then moves off Ollama onto the
llamacpp/MLX path, where active-params-per-token (not total params) sets
decode speed — a different regime from the two rungs below.
