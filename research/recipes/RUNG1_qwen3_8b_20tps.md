# Rung 1 validation recipe: qwen3:8b at 20 tok/s (M1 Pro)

**Goal:** rung 1 of the Hearth capability ladder — a 7B/8B-class model
sustaining **≥ 20 tokens/sec decode** on Justin's M1 Pro, **measured** (not
estimated), plus a quality sanity check. A rung counts only when both pass.

## What to download

```bash
ollama pull qwen3:8b        # ~5 GB, Q4_K_M default tag
```

This is Hearth's current default model, so nothing else needs installing.

## Expected RAM (from `src/tools/fitcheck.py`, arch `qwen3-8b`)

| Component | Size |
|---|---|
| Weights (Q4_K_M, 4.9 bpw) | ~4.9–5.2 GB |
| KV cache @ 2048 ctx, f16 | 0.30 GB |
| KV cache @ 4096 ctx, f16 | 0.60 GB |
| KV cache @ 2048 ctx, q8_0 | 0.17 GB |
| Runtime overhead | ~0.5 GB |
| **Total** | **~5.6–6.0 GB** |

Comfortably inside the ~10–11 GB usable budget (macOS takes the rest of the
16 GB). No memory tuning is needed for this rung; KV settings below are
speed levers only. Verify resident size on the Mac with `ollama ps`.

## Measure

With `ollama serve` running:

```bash
cd ~/workspace/local-agent
python3 tools/measure_rung.py --model qwen3:8b --target 20
```

The harness POSTs to Ollama's `/api/generate` (`stream=false`, temp 0),
reads `eval_count`/`eval_duration` straight from the server, and runs:

- **Speed:** 3 fixed prompts (short answer, code gen, ~150-word
  passage + summary), 256 generated tokens each. Reports per-prompt and mean
  decode tok/s.
- **Quality sanity:** arithmetic (`17*23` → `391`), factual (capital of
  France), instruction-following (repeat "quasar" exactly 3×), all at temp 0.

**Pass criteria:** mean decode tok/s ≥ 20 **and** all 3 quality checks pass.
The script exits 0 and prints `RESULT: RUNG PASS`. Paste the full output
back as the measurement record.

## Roofline sanity check (why 20 tok/s is expected, not hoped)

M1 Pro unified-memory bandwidth ≈ 200 GB/s. Decode is bandwidth-bound: each
token streams the weights once, so the ceiling is ≈ bandwidth / model size
≈ 200 / 5 ≈ **40 tok/s**. The 20 tok/s bar is ~half the ceiling — there is
headroom for kernel overhead, and a healthy run should clear it. If it
doesn't, something is off (usually GPU offload), not the model.

## If below 20 tok/s — tune in this order

1. **Verify full Metal offload.** `ollama ps` should show the model at ~100%
   GPU. If layers are on CPU, that's the whole story — fix the offload
   before touching anything else.
2. **Reduce context.** Re-run with `--num-ctx 2048` (newer Ollama defaults to
   4096; a smaller context shrinks per-token KV traffic):
   `python3 tools/measure_rung.py --model qwen3:8b --num-ctx 2048 --target 20`
3. **Quantized KV cache.** Stop Ollama, restart with
   `OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve` (or `launchctl setenv`
   `OLLAMA_KV_CACHE_TYPE q8_0` for the launchd service), re-run. Halves KV
   traffic; small speed gain, mostly memory headroom.
4. **Thread count.** `OLLAMA_NUM_THREAD` — the default (all efficiency +
   performance cores) is usually right on the M1 Pro; only touch if
   something looks wrong.
5. **Lighter quant tag** (last resort — changes quality, so re-check the
   sanity section): try a `q4_0` variant of the 8B model if published.

Report which step (if any) was needed, and re-run the full harness after
each change — one variable at a time.

## What a validated rung 1 unlocks

Rung 1 passing fixes the ladder's anchor: Hearth's default model holds
20 tok/s on the target machine. Rung 2 (14B-class at 20 tok/s) then repeats
this recipe with the 14B candidate; rung 3 (35B-class MoE) uses the
llamacpp/MLX path instead of Ollama.
