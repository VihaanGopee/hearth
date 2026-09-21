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
- [x] Quant R&D: harness scaffolded — `src/quant_rnd/` (5 schemes, synthetic
      SQNR bench, 12 tests). Baselines: ternary_uniform (T1_0-like),
      int2_symmetric (naive), int2_outlier_retain (mixed-precision).
      Candidates: dual_scale_ternary (asymmetric ternary), ternary_outlier
      (T1 + 2 exact fp16 outliers/group). Synthetic result 2026-09-20:
      ternary_outlier 6.73 dB @ 2.06 bpw > int2_symmetric 2.51 dB @ 2.13 bpw;
      dual_scale_ternary beats symmetric ternary on skewed tensors.
      PROXY ONLY — not real-model perplexity.
- [x] Quant R&D: K-means (Lloyd) 2-bit baseline landed 2026-09-20
      (`int2_kmeans`, 4 fitted centroids/group, 2.5 bpw, 32 tests green).
      Synthetic result: 9.68 dB — best SQNR, but at the highest bitrate.
      Honest read: the classical baseline beats our candidates on synthetic
      SQNR; they are cheaper in bpw. Follow-up below.
- [x] Quant R&D: matched-bitrate k-means variant — landed 2026-09-20 as
      `int2_kmeans_q8` (Lloyd fit, 4 centroids in 8-bit + one fp16
      codebook scale -> 2.375 bpw; 39 tests green). Result: 8-bit codebook
      rounding is nearly free (9.0486 vs 9.0489 dB on the test tensor), and
      **ternary_outlier still loses to the classical baseline** (6.60 dB @
      2.06 bpw vs 9.05 dB @ 2.375 bpw). Negative result, recorded honestly:
      on synthetic SQNR, Lloyd k-means owns the ~2.4 bpw Pareto point.
- [ ] Quant R&D: pivot the candidate story — since Lloyd beats our
      candidates on SQNR, the case for ternary schemes must rest on
      inference cost (add/sub vs multiplies), not fidelity. Part (a) is
      now ANSWERED NO (2026-09-20 Pareto sweep, `src/quant_rnd/sweep.py`):
      Lloyd owns every bitrate ≥2.19 bpw — `int2_kmeans_q8` at group 256
      scores **9.12 dB @ 2.188 bpw** vs ternary_outlier's 6.73 @ 2.060 —
      on both clean and skewed tensors, and Lloyd is skew-invariant. Don't
      chase SQNR parity with Lloyd. Part (b) is now the main thread:
      quantify the ternary compute advantage and aim the ggml kernel
      work there. Part (b) QUANTIFIED 2026-09-20 (`src/quant_rnd/opcount.py`):
      ternary_uniform 1.36x higher decode ceiling + 3.0x lower
      energy-proxy op cost vs int2_kmeans_q8 at 70B scale (see checked
      op-count item below); next: Mac-side validation, prefill roofline.
- [ ] Quant R&D: when the real-tiny-model perplexity run happens, check
      whether the synthetic ranking (k-means > ternary_outlier >
      dual_scale_ternary) reproduces on real weights — the ranking, not
      the absolute dB, is what transfers.
- [ ] Quant R&D: validate candidates on a real tiny model (60–130M params,
      CPU-friendly) — real perplexity vs the synthetic SQNR ranking
- [x] Quant R&D: sweep outlier_frac / n_outliers for the Pareto frontier
      (SQNR vs bpw) — landed 2026-09-20 as `src/quant_rnd/sweep.py`
      (seed 7). Frontier: ternary family below ~2.06 bpw (ternary_uniform
      5.56 dB @ 1.710 → DST 5.59 @ 1.835 → T1+out n=1 6.41 @ 1.885 →
      n=2 6.73 @ 2.060), then classical Lloyd takes over (q8 g=256:
      9.12 @ 2.188; q8 g=128: 9.68 @ 2.375; kmeans g=64: 10.09 @ 3.000).
      Skewed run (mean shift 0.5): Lloyd unchanged (skew-invariant), DST
      keeps a small edge over symmetric ternary (5.06 vs 4.76 dB).
      Verdict: no ternary scheme beats Lloyd below ~2.2 bpw on synthetic
      SQNR — the fidelity path is exhausted; pivot to compute (see above).
      Side fix the sweep exposed: `QuantResult.reconstruct()` hard-coded
      GROUP_SIZE=128 and crashed on other group sizes; group_size is now
      stored on the result (46 tests green).
- [ ] Quant R&D: candidate C — "Lloyd-fit ternary": 1-D Lloyd with the
      codebook constrained to ternary {-s,0,+s} (or dual-scale
      {-s_neg,0,+s_pos}). The sweep suggests k-means *fitting* is doing
      the heavy lifting, not the codebook size; this tests whether Lloyd
      fitting rescues ternary at 1.71–1.84 bpw. If it beats DST/T1+out
      there, it becomes the new ternary reference.
- [ ] Quant R&D: adopt `int2_kmeans_q8` at group 256 (2.188 bpw) as the
      reference baseline-to-beat for future candidate schemes — it
      undercuts the old 2.375 bpw reference and matches our candidates'
      bitrate band.
- [x] Quant R&D: op-count model of ternary matmul (add/sub per MAC) vs
      int2 codebook-lookup + fp dequant-multiply, for the bandwidth-bound
      Apple Silicon decode regime — landed 2026-09-20 as
      `src/quant_rnd/opcount.py` (11 new tests, 57/57 green). The
      quantitative case the pivot now rests on, at 70B scale / 200 GB/s
      / 1.34 GB KV (4k ctx fp16):
      ternary_uniform (1.71 bpw, measured sparsity 31%): 14.96 GB,
      roofline 12.3 t/s, 0.71 equiv-adds/weight;
      int2_kmeans_q8 (2.375 bpw, histogram dequant): 20.78 GB, roofline
      9.0 t/s, 2.17 equiv-adds/weight.
      => **1.36x higher decode ceiling, 3.0x lower energy-proxy op cost**
      for ternary over the Lloyd reference. All schemes are
      bandwidth-bound (2.8-7.7 FLOP/byte vs ~10 machine balance), so the
      bpw-driven byte reduction is the first-order effect; the add/sub
      MAC advantage is second-order on decode, first-order in
      compute-bound regimes (prefill). MODEL ONLY — ceilings, not
      measured tok/s; Mac validation is a follow-up below. Caveat:
      14.96 GB still exceeds the ~10-11 GB usable budget, so this
      quantifies the *speed* case contingent on a fitting path.
- [ ] Quant R&D: Mac-side validation of the roofline ratios — measure real
      decode tok/s on Justin's Mac for a ternary/TQ1_0 model vs a 2-bit
      codebook model at matched size; compare against the 1.3-1.4x
      ceiling ratio predicted by opcount.py (needs Justin's Mac)
- [ ] Quant R&D: extend opcount.py with a prefill (compute-bound) roofline
      — the regime where ternary's 3x equiv-adds advantage becomes the
      first-order effect
- [ ] Quant R&D: if a candidate holds up on real perplexity, design the
      ggml CPU kernel (ternary add/sub path) + upstream write-up/PR
- [x] Quant R&D: add a K-means (Lloyd) 2-bit baseline — the fair classical
      comparison the current naive int2 baseline lacks (landed 2026-09-20,
      `int2_kmeans`, 2.5 bpw; see result note above)
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
