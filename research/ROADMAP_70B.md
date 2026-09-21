# Project 70B: fit a 70B-class model in 16 GB

**Goal:** make Hearth able to run a 70B-parameter-class model on a 16 GB
M-series Mac (M1 Pro target) with usable quality and speed.

**Honest budget:** macOS itself needs ~4–6 GB, so the model weights + KV
cache + runtime realistically get **~10–11 GB**. Everything below is
measured against that number, not the full 16 GB.

## RAM math (estimates, 70B dense)

| Quantization | bits/param | ~Size | Fits 10–11 GB budget? |
|---|---|---|---|
| FP16 | 16.0 | 140 GB | No |
| Q8_0 | 8.5 | ~74 GB | No |
| Q4_K_M | ~4.9 | ~43 GB | No |
| Q2_K | ~3.4 | ~29 GB | No |
| IQ2_XXS | ~2.1 | ~18 GB | No |
| IQ1_M | ~1.75 | ~15.3 GB | Borderline — likely OOM once macOS + KV cache are counted |
| TQ1_0 (ternary) | ~1.69 | ~14.9 GB | Borderline — same caveat |
| Native ternary 1.58-bit (BitNet-style, trained from scratch) | 1.58 | ~13.8 GB | Closest; needs a well-trained 70B ternary model, which does not publicly exist yet |

**Conclusion:** dense 70B needs ≤ ~1.3 bits/param to fit comfortably, which
no GGUF quant reaches today. The realistic paths are: (1) a native
ternary 70B release, (2) MoE + mixed-precision 2-bit quants
("70B-class" quality at 8–10 GB), (3) pushing GGUF sub-2-bit quants and
measuring where quality actually breaks.

## Ranked avenues

### A. Native ternary 70B (BitNet b1.58 style) — WATCH
- A 70B model trained from scratch in ternary {-1,0,1} is ~13.8 GB and
  bitnet.cpp runs it with CPU-friendly add/sub ops (no multiplies).
- Status 2026-09-20: public ternary models top out around 8–10B, and the
  Llama3-8B-1.58 (100B-token) conversion is reported as poor quality. No
  well-trained 70B ternary model is announced.
- Trigger: if Microsoft/community releases a 70B ternary model trained on
  1T+ tokens, evaluate immediately — this is the single development that
  would most directly achieve the goal.

### B. MoE + mixed-precision 2-bit quants (MLX) — ACTIVE / PRACTICAL PATH
- MoE models activate few params per token, so on bandwidth-bound Apple
  Silicon, *active* params set decode speed. Mixed-precision quants keep
  sensitive tensors (attention, routers) at 6–8 bit and compress experts
  to 2–4 bit.
- Data point: oQ (oMLX) 2-bit quant of Qwen3.5-35B-A3B scores 64% MMLU vs
  14% for naive mlx-lm 2-bit; JANG does similar adaptive allocation.
- A 30–35B MoE at effective ~2 bit ≈ 8–9 GB — fits the budget with room
  for context, and quality is "70B-class" on many tasks.
- Work item: add an **MLX backend** to Hearth (Apple Silicon native,
  unified memory). This is likely the eventual engine for the Mac.

### C. GGUF sub-2-bit 70B (IQ1_M / TQ1_0) — EXPERIMENT
- llama.cpp now ships TQ1_0/TQ2_0 ternary quants (1.7–2.1 bpw) and
  IQ1_S/IQ1_M (1.5–1.75 bpw). A 70B IQ1_M is ~15.3 GB on disk.
- Open questions: does Ollama's bundled llama.cpp support TQ quants yet?
  Do 70B IQ1_M/TQ1_0 GGUFs exist on HuggingFace, and is their quality
  (esp. tool calling, reasoning) usable?
- Work item: when files exist, pull one on the Mac, measure quality vs a
  32B Q4 baseline, record results here.

### D. KV-cache compression — SUPPORTING
- KV cache is small for 70B (~1.3 GB at 4k ctx fp16) but every GB counts.
- Ollama: `OLLAMA_KV_CACHE_TYPE=q8_0` (or q4_0). MLX: VeloxQuant /
  TurboQuant claim 5–8x KV cache reduction at near-fp16 quality.
- Work item: set `OLLAMA_KV_CACHE_TYPE=q8_0` in Hearth's Ollama launch env;
  document measured savings.

### E. Speculative decoding — SPEED, once a 70B fits
- Small draft model + 70B target. Doesn't reduce memory; makes a fitting
  70B actually pleasant to use. Revisit after A/B/C lands.

### F. Layer-wise / SSD offloading (AirLLM-style) — LAST RESORT
- Fits anything by paging layers, but tok/s is usually too slow for an
  interactive agent. Only if all else fails.

### H. Original quantization R&D — us, not waiting (NEW)
- Instead of only tracking other people's releases, we run our own
  small-scale quantization research: prototype novel sub-2-bit / ternary
  schemes in Python on small models we can actually run here, measure
  perplexity properly against IQ1_M/TQ1_0, and publish the results.
- If a prototype wins: implement ggml CPU kernels, validate, write it up,
  and contribute it upstream (llama.cpp / bitnet.cpp).
- Hard limit, stated honestly: this VM cannot train or convert an actual
  70B model (no GPU, can't hold 140GB weights, can't validate at scale).
  We do real R&D at small scale and contribute upward — we don't pretend
  to ship a 70B artifact from here.

## Experiment backlog

- [x] Direct llama.cpp backend (`src/llamacpp_backend.py`, `backend: llamacpp`) — unlocks IQ1_M/TQ1_0, KV-cache types, Metal layers (2026-09-20)
- [x] `estimate_fit` tool (`src/tools/fitcheck.py`) — weights + KV cache math vs usable budget (2026-09-20)
- [ ] Quant R&D: prototype novel ternary/sub-2-bit schemes on small models, perplexity vs IQ1_M/TQ1_0
- [ ] Quant R&D: if a prototype wins, ggml CPU kernels + upstream write-up/PR
- [ ] Check whether current Ollama release supports TQ1_0/TQ2_0 GGUFs
- [ ] Survey HuggingFace for 70B IQ1_M / TQ1_0 GGUFs; record sizes + quality reports
- [ ] Set `OLLAMA_KV_CACHE_TYPE=q8_0` in run.sh and document
- [ ] Speculative decoding support in the llamacpp backend (draft + target model)
- [ ] Model cascade: small fast model by default, escalate hard queries to the big model
- [ ] Prototype MLX backend (mac-only; can't be tested on Linux — needs Justin's Mac)
- [ ] Benchmark harness: quality-vs-quant curves on small models to validate the pipeline
- [ ] Track BitNet.cpp releases + any 70B ternary model announcement
- [ ] Track oQ/JANG releases and 2-bit MoE quality reports

## Ground rules for this research track

- Every change is a git commit in `~/workspace/local-agent`. Revert with
  `git log --oneline` + `git revert <hash>`.
- Never break the working default (`qwen3:8b` via Ollama). Experimental
  profiles stay opt-in.
- Don't download multi-GB models on the dev VM; use file-size metadata.
- Each research sweep appends a dated entry under `research/log/`.
