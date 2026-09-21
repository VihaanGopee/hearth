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
| IQ1_M | ~1.75 | **16.75 GB measured** (bartowski 70B GGUFs, HF API 2026-09-21) | No — over budget even before macOS + KV cache |
| TQ1_0 (ternary) | ~1.69 | ~14.9 GB (est.; no 70B file exists on HF as of 2026-09-21) | No — est. still over the 10–11 GB budget |
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

### C. GGUF sub-2-bit 70B (IQ1_M / TQ1_0) — SURVEYED 2026-09-21, effectively closed
- llama.cpp ships TQ1_0/TQ2_0 ternary quants (1.7–2.1 bpw) and
  IQ1_S/IQ1_M (1.5–1.75 bpw).
- **Ollama TQ support: YES.** Ollama main and release v0.34.3-rc1 both pin
  llama.cpp b10969; at that revision upstream `ggml/include/ggml.h`
  defines `GGML_TYPE_TQ1_0` and `GGML_TYPE_TQ2_0` (verified 2026-09-21
  via raw file fetch), so Ollama loads TQ1_0/TQ2_0 GGUFs. Caveat: type
  support only — the Metal kernel/perf path for ternary quants is
  unverified (see backlog).
- **HF survey 2026-09-21: no 70B-scale TQ1_0/TQ2_0 GGUF exists.**
  HF API search for TQ1_0/TQ2_0 returns only unrelated small repos and one
  397B oddity; no 70B ternary file to evaluate.
- **70B IQ1_M exists but fails the budget.** bartowski's Llama-3.1-70B,
  Llama-3.3-70B, r1-1776-distill-llama-70b IQ1_M files all measure
  **16.75 GB** (HF API tree sizes) vs the ~15.3 GB estimate — over the
  10–11 GB usable budget even before macOS + KV cache. It cannot fit a
  16 GB Mac, full stop.
- **Quality would not save it anyway.** bartowski labels 70B IQ1_M
  "Extremely low quality, *not* recommended."; mradermacher's i1-IQ1_M
  (16.0 GB) is "mostly desperate"; an independent 20-model Llama-3-70B
  comparison found IQ1_M/IQ1_S at 12–15/18 vs reference with degraded
  instruction-following and coherence ("1-bit quantization doesn't seem
  viable yet" for agentic use).
- Net: even the best sub-2-bit dense-70B option (a) doesn't exist as a
  file (TQ), (b) doesn't fit when it does exist (IQ1_M), and (c) loses
  too much quality for an agent. Avenue C is now WATCH-only for future
  ≤1.3-bpw GGUF quants or TQ files on smaller dense models (e.g. a ~50B
  TQ1_0 ≈ 10.6 GB would land inside the budget).

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
- [x] Quant R&D: does the synthetic SQNR ranking reproduce on real weights
      — ANSWERED YES 2026-09-21 (`src/quant_rnd/realweights.py`,
      dependency-free .safetensors parser, 97 tests green). GPT-2 124M,
      48 linear weight matrices, 64 groups/matrix, seeds 7+8: ranking
      matches synthetic almost exactly — Lloyd k-means 9.4-9.5 dB >>
      ternary_lloyd_ds 7.0-7.1 > ternary_lloyd 6.8-6.9 >
      ternary_1step_ds ~= ternary_outlier 6.5-6.6 (tied within noise) >
      ternary_1step 6.4-6.5 > int2_outlier_retain 5.8-5.9 >
      dual_scale_ternary 5.3-5.4 > ternary_uniform 5.25-5.33 >
      int2_symmetric 2.8-3.1. Absolute SQNR within ~0.3 dB of synthetic
      for all schemes except int2_outlier_retain (-0.6 dB, real outliers
      are harder to isolate than planted ones) and int2_symmetric
      (+0.3-0.6 dB). Weights live at research/data/gpt2.safetensors
      (gitignored, reusable). Caveat: SQNR, not perplexity — the
      distribution caveat is closed, the PPL caveat is not.
- [x] Quant R&D: validate candidates on a real tiny model (60–130M params,
      CPU-friendly) — real perplexity vs the synthetic SQNR ranking.
      Step 1 landed (real-weight SQNR probe above); step 2a LANDED
      2026-09-21 as `src/quant_rnd/gpt2_forward.py` + `gpt2_tokenizer.py`
      (commit 48dd13e, 116 tests green): dependency-free NumPy GPT-2 124M
      forward pass and byte-level BPE tokenizer over
      research/data/gpt2.safetensors. fp32 reference: **perplexity 53.50**
      on 406 tokens of hand-composed ASCII prose
      (research/data/eval_text.txt) — sane band for GPT-2 124M, so the
      forward pass is verified end to end. The parser, tensor selection,
      and sampled-block harness all exist already. Step 2b LANDED
      2026-09-21 (`python3 -m src.quant_rnd.ppl`, 7-scheme default sweep,
      g128, ~7.7 min): **everything collapses vs fp32** — best
      ternary_lloyd 2764.68 @ 1.710 bpw vs 53.50 reference (52x worse).
      Top tier transfers directionally (fitted ternary family dominates;
      lloyd beats naive uniform 186x at matched bitrate), but below the
      fitted tier SQNR ordering scrambles (dual_scale SQNR-beats uniform
      yet ppl-loses 4.7x; diagnostics rule out bad/dead groups — it is
      cliff noise). Surprise revising a prior conclusion: full Lloyd
      beats 1-step 25x in ppl (2764 vs 70997) at only 0.2 dB SQNR
      difference — "1-step captures 85%" was SQNR-true but
      perplexity-incomplete; in collapse territory small fidelity deltas
      amplify wildly through 12 layers. Verdict: simple group-wise
      quantizers at <=2.2 bpw do not yield a usable GPT-2 124M (honest
      negative); the fidelity path needs second-order correction (see
      follow-ups below). The opcount compute-advantage story is
      untouched — it never depended on pure group-wise usability.
- [x] Quant R&D: int2_kmeans_q8 perplexity run — ANSWERED 2026-09-21
      (`python3 -m src.quant_rnd.ppl ... --schemes int2_kmeans_q8`, g128):
      **795.96 @ 2.375 bpw** vs fp32 53.50 (14.9x — still above the 10x
      discrimination threshold, so not out of collapse territory, but the
      harness measures a REAL fidelity gradient: 9.68 dB SQNR -> 795 ppl
      is 3.5x better than the previous best (ternary_lloyd 2764). More
      fidelity is still the right direction; OBQ-style correction next.
- [x] Quant R&D: full ternary_lloyd_ds perplexity — ANSWERED 2026-09-21
      (n_iter=20, g128): **2719.46 @ 1.835 bpw** — essentially TIED with
      symmetric ternary_lloyd (2764 @ 1.710). Dual fitting buys ~nothing
      extra in perplexity at this scale despite +0.07 dB SQNR and higher
      bitrate; the symmetric full fit stays the ternary reference.
      Full-vs-1step dual gap is 4.9x (13225 -> 2719), smaller than the
      symmetric 25x gap — dual is closer to converged at n_iter=1.
- [x] Quant R&D: int2_kmeans_q8 perplexity at g256 (2.188 bpw) — the
      opcount reference config; SQNR dips 9.68 -> 9.09 g128 -> g256, so
      this checks whether the ppl gradient prefers g128 or g256 and
      pins the fidelity anchor the OBQ work must beat. ANSWERED
      2026-09-21: **g256 LOSES — 1085.61 @ 2.188 bpw** (vs 795.96 @
      2.375 bpw at g128), despite only a -0.59 dB SQNR dip. The
      perplexity gradient prefers the smaller group size: more
      per-group codebook fidelity beats the 0.19 bpw rate saving. The
      OBQ anchor is therefore **795.96 @ g128 (2.375 bpw)**.
- [x] Quant R&D: is the 795 ppl at 2.375 bpw limited by embeddings/LN?
      ANSWERED NO 2026-09-21 — `ppl.py --quantize-embeddings` landed
      (quantizes wte+wpe with the same scheme; biases/LN stay fp32; 145
      tests green). int2_kmeans_q8 g128 on text1 with embeddings
      quantized: **ppl 1.3e14 @ 2.375 bpw** vs 795.96 with fp32
      embeddings. Total collapse (wte is the tied lm_head) — the fp32
      passthrough is load-bearing, not a bottleneck. The fidelity problem
      is in the LINEAR weights; error-compensation (OBQ-style) stays the
      right next thread, applied to linears only.
      Follow-up: wte-vs-wpe split not yet run (one ~11 min run each) —
      folded into the embedding-table probe item below.
- [ ] Quant R&D: OBQ/GPTQ-style second-order correction prototype — the
      honest next fidelity step now that pure group-wise is exhausted on
      both SQNR and perplexity. Start with per-group Hessian-diagonal
      (empirical Fisher from a few forward passes) reweighting of the
      Lloyd/ternary fit; measure whether ppl exits collapse territory.
- [ ] Quant R&D: methodology caveat for future sweeps — perplexity only
      discriminates below ~10x the fp32 reference; in collapse territory
      report ordering as directional and do not quote ratios as quality
      figures for the 70B case.
- [x] Quant R&D: ternary_lloyd perplexity at g64 (SQNR-best group size,
      ~5 min quantize) — ANSWERED 2026-09-21 (see robustness item):
      2729.23 @ 1.835 vs 2764.68 @ 1.710 g128 — the SQNR edge buys ~1%
      ppl at +0.125 bpw. g128 stays the reference config.
- [ ] Quant R&D: this VM has no optimized BLAS, so one 406-token forward
      pass costs ~40 s (the (T,768)@(768,50257) logits matmul dominates).
      The step-2b per-scheme sweep is ~10 forwards; if that gets slow,
      chunk the logits matmul or shorten the eval text. The fp32 53.50
      reference is logged and reproducible either way.
- [x] Quant R&D: group-size sensitivity of the real-weight ranking —
      ANSWERED 2026-09-21: re-ran realweights.py at g64 and g256
      (CLI gained a `--group-size` flag, `parse_args` factored for
      testability; 99 tests green). The ranking is essentially
      group-size invariant (seed 7, 48 matrices): Lloyd k-means first,
      then ternary_lloyd_ds, then ternary_lloyd at ALL three sizes.
      g64: 9.95 / 7.34 / 7.07 dB; g128: ~9.4 / 7.0 / 6.8;
      g256: 9.09 / 6.83 / 6.72. All schemes lose SQNR as groups grow
      (Lloyd 9.95→9.09 g64→g256; naive int2_symmetric collapses at g256:
      1.21 dB). One outlier-rate-driven shuffle: ternary_outlier passes
      ternary_lloyd at g64 (7.21 vs 7.07 @ 2.535 bpw) and falls below
      ternary_1step at g256 (6.21 vs 6.30 @ 1.823 bpw) — expected from
      its n_outliers/group bitrate, not a story change. Feeds the
      opcount-table item below: the ranking does not prefer a group
      size; the perplexity run still needs to pick 128 vs 256.
- [ ] Quant R&D: add embedding-table (wte) quantization to the real-weight
      probe — currently excluded by name; embedding rows have a very
      different distribution and are the largest single tensor in small
      models. NOTE 2026-09-21: the ppl ablation above (wte+wpe at 2-bit
      group-wise → 1.3e14 collapse) says naive low-bit embedding
      quantization is catastrophic — the interesting probe is now
      higher-precision (q8/q4) or per-row-scaled embedding schemes, plus
      the wte-vs-wpe split the ablation skipped.
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
- [x] Quant R&D: candidate C — "Lloyd-fit ternary" — LANDED 2026-09-20 as
      `ternary_lloyd` / `ternary_lloyd_ds` (65 tests green). Verdict: YES,
      fitting is the ingredient. At EXACTLY matched bitrate, Lloyd fitting
      beats the absmean heuristic by **+1.37 dB symmetric** (6.93 vs 5.56 @
      1.710 bpw) and **+1.41 dB dual-scale** (7.00 vs 5.59 @ 1.835 bpw).
      Both are new Pareto-frontier points; on skewed tensors
      ternary_lloyd_ds (6.55 @ 1.835) > ternary_lloyd (6.11 @ 1.710).
      ternary_lloyd is now the ternary reference (beats T1+out n=2 at a
      LOWER bitrate). The ~2.19 bpw Lloyd crossover still holds: below it,
      fit the ternary codebook; at/above it, use 4-centroid Lloyd.
- [x] Quant R&D: adopt `ternary_lloyd` (1.710 bpw) as the ternary reference
      baseline-to-beat for future candidate schemes, alongside
      `int2_kmeans_q8` at group 256 (2.188 bpw) as the classical reference.
      SUPERSEDED 2026-09-21: the fitted-ternary reference was adopted as
      `ternary_1step` (the practical 1-iteration encoder) instead — see the
      checked item above; and perplexity selects g128, not g256, for the
      k-means reference (anchor 795.96 @ 2.375 bpw).
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
- [x] Quant R&D: extend opcount.py with a prefill (compute-bound) roofline
      — landed 2026-09-21 as `prefill_roofline_tps` (91 tests green).
      Verdict: the regime flip is real and the prediction holds. At L=1
      (decode-like) both references are bandwidth-bound (AI 2.8 vs machine
      balance 26 FLOP/byte); at L=512+ both are compute-bound (AI 100x+
      balance) and the ceiling ratio is EXACTLY the FLOP-per-weight ratio:
      **1.795x ternary_1step over int2_kmeans_q8** (peak/efficiency/length
      all cancel). Conservative by construction: adds count as 1 FLOP
      against an FMA-counted peak, so a real add-dominated ternary kernel
      has up to ~2x headroom above this ceiling on FMA hardware — stated,
      not folded in. Peak is an explicit argument (Apple-published 5.2
      TFLOPS FP32 M1 Pro GPU used in tests); no baked-in constants.
      Attention O(L^2) ignored (<5% at 4k) — a known under-count at long
      context, see follow-up below.
- [ ] Quant R&D: Mac-side validation of the prefill ceiling ratio —
      measure prompt-processing tok/s on Justin's Mac for a ternary/TQ1_0
      model vs a 2-bit codebook model at matched size; compare against the
      1.79x ceiling ratio predicted by prefill_roofline_tps (needs
      Justin's Mac)
- [ ] Quant R&D: add the attention O(L^2) term to prefill_roofline_tps for
      long context (32k+) — currently a documented under-count; at 128k
      attention is no longer negligible vs the 70B matmuls
- [ ] Quant R&D: if a candidate holds up on real perplexity, design the
      ggml CPU kernel (ternary add/sub path) + upstream write-up/PR
- [x] Quant R&D: add a K-means (Lloyd) 2-bit baseline — the fair classical
      comparison the current naive int2 baseline lacks (landed 2026-09-20,
      `int2_kmeans`, 2.5 bpw; see result note above)
- [x] Check whether current Ollama release supports TQ1_0/TQ2_0 GGUFs —
      ANSWERED YES 2026-09-21: Ollama main and v0.34.3-rc1 both pin
      llama.cpp b10969, whose ggml.h defines GGML_TYPE_TQ1_0/TQ2_0
      (verified via raw fetch). Type-level support confirmed; Metal
      kernel/perf path for ternary quants unverified (see new item).
- [x] Survey HuggingFace for 70B IQ1_M / TQ1_0 GGUFs; record sizes +
      quality reports — SURVEYED 2026-09-21: no 70B-scale TQ1_0/TQ2_0
      file exists on HF; 70B IQ1_M files exist (bartowski 3.1/3.3/r1-1776)
      but measure 16.75 GB — over the 10–11 GB usable budget, cannot fit
      a 16 GB Mac. Quality: bartowski "extremely low quality, not
      recommended"; mradermacher i1-IQ1_M "mostly desperate";
      independent 20-model test found IQ1_M at 12–15/18 with degraded
      instruction-following. Avenue C reclassified WATCH-only.
- [ ] Re-survey HF for 70B-class TQ1_0/TQ2_0 GGUFs when the ternary-quant
      ecosystem matures (none exist as of 2026-09-21). If a file appears,
      check its size against the 10–11 GB budget FIRST (a 70B TQ1_0 est.
      ~14.9 GB would still not fit; a ~50B TQ1_0 ≈ 10.6 GB would) before
      any quality evaluation.
- [ ] Verify Ollama's Metal kernel path for TQ1_0/TQ2_0 quants on the Mac
      (type support confirmed at llama.cpp b10969; whether ternary matmuls
      take an optimized path or a slow fallback on Apple Silicon is
      unknown) — needs Justin's Mac, low priority until a TQ file exists.
- [ ] Set `OLLAMA_KV_CACHE_TYPE=q8_0` — NOTE 2026-09-21: Hearth talks to
      an already-running `ollama serve` over HTTP (see src/llm.py,
      run.sh); there is no launcher code to wire the env var into, it
      belongs in the *Mac's* `ollama serve` environment (e.g.
      `launchctl setenv` / the plist that starts Ollama). When Justin
      validates on the Mac: export it before starting Ollama and document
      measured KV savings; nothing to commit here beyond this note.
- [ ] Speculative decoding support in the llamacpp backend (draft + target model)
- [ ] Model cascade: small fast model by default, escalate hard queries to the big model
- [ ] Prototype MLX backend (mac-only; can't be tested on Linux — needs Justin's Mac)
- [ ] Benchmark harness: quality-vs-quant curves on small models to validate the pipeline
- [ ] Track BitNet.cpp releases + any 70B ternary model announcement
- [ ] Track oQ/JANG releases and 2-bit MoE quality reports
- [x] Quant R&D: re-run the op-count energy/speed comparison with the
      fitted ternary reference — landed 2026-09-21 as
      `test_fitted_ternary_opcount_reference` (81 tests green). Verdict:
      the figures barely move. Decode ceiling ratio UNCHANGED at 1.357x
      (roofline is byte-driven, sparsity-independent); energy-proxy ratio
      RISES from 3.04x to 3.52x because the fitted encoder's measured
      zero-rate is 0.41 vs ternary_uniform's 0.31 (fewer adds/weight).
      The ternary compute case is, if anything, slightly stronger than
      first reported.
- [x] Quant R&D: diagnostic — decompose the +1.4 dB Lloyd win into
      threshold adaptation vs scale refit (fix one, vary the other) to
      find the cheaper approximation of the fitted optimum — LANDED
      2026-09-20 as `src/quant_rnd/diagnose.py` (74 tests green). Verdict:
      the win is ~85% scale refit (+1.16 dB at step 1) and ~15% threshold
      adaptation (+0.20 dB to convergence) on clean tensors; skewed
      tensors split +1.13/+0.22 dB. A threshold-grid ablation with the
      heuristic scale fixed finds alpha=0.5 optimal — with the heuristic
      scale the heuristic threshold was already optimal; the problem was
      the scale. => the follow-up item below.
- [x] Quant R&D: add a "1-step Lloyd" ternary scheme — heuristic
      thresholds (+/-absmean/2) + a single L2-optimal scale refit — and
      check whether it captures ~85% of ternary_lloyd's SQNR at O(1)
      extra cost (no iteration) — LANDED 2026-09-21 as `ternary_1step`
      (`_ternary_lloyd_fit` with n_iter=1; 81 tests green). HOLDS: capture
      0.84–0.87 across seeds 7–9 and skew 0.0/0.5 (seed 7 clean: 6.72 dB
      vs uniform 5.56 / full Lloyd 6.93). One iteration IS the practical
      encoder for fitted ternary. Note: the strict "refit scale, keep
      heuristic codes" variant was tried and scores worse (6.45 dB) —
      decode must reassign at the refit thresholds. Measured zero-rate
      0.41 vs uniform's 0.31 (refit widens thresholds).
- [x] Quant R&D: adopt a fitted-ternary reference — landed as
      `ternary_1step` (the practical encoder) as the ternary reference
      baseline-to-beat, alongside `int2_kmeans_q8` at group 256
      (2.188 bpw) as the classical reference (see opcount re-run below).
- [x] Quant R&D: "1-step Lloyd" dual-scale variant — heuristic dual
      thresholds -> one per-side refit -> reassign; check the capture on
      skewed tensors against ternary_lloyd_ds (currently 0.84–0.87 for
      the symmetric twin) — LANDED 2026-09-21 as `ternary_1step_ds`
      (`_ternary_lloyd_fit` with dual=True, n_iter=1; 88 tests green).
      ANSWER: PARTIAL, not the symmetric twin's 0.84–0.87. One iteration
      captures 0.55–0.85 of the dual-Lloyd win across seeds 7–9 (clean +
      skew 0.5); 2 iterations 0.63–0.97; 3 iterations 0.65–0.99. The
      dual case converges slower — the practical dual encoder is 2–3
      fixed iterations (still O(1)), not 1. (Side note the experiment
      caught: an unregistered dual-scheme name silently mis-decodes in
      reconstruct() via the single-scale fallback — the
      _DUAL_SCALE_SCHEMES registration is load-bearing for correctness.)
- [x] Quant R&D: robustness of the g128-vs-g256 ppl gap — PARTIALLY
      ANSWERED 2026-09-21: second eval text landed
      (`research/data/eval_text2.txt`, hand-composed non-fiction prose,
      393 tokens, fp32 ppl 63.18 — sane). int2_kmeans_q8 g128 on text2:
      **1047.35 @ 2.375 bpw** vs 795.96 on text1. Relative to fp32:
      16.6x vs 14.9x. The anchor is text-sensitive (~+30% absolute,
      ~+11% relative): 795.96 is NOT gospel — quote the OBQ anchor as a
      text-conditioned range and re-measure OBQ candidates on the SAME
      text(s). Directional verdict holds (still far best, same band).
      Still open: the "shuffled block sample" half of this item; g256 on
      text2 (does the g128-beats-g256 verdict transfer? — see new item).
- [x] Quant R&D: ternary_lloyd perplexity at g64 — ANSWERED 2026-09-21:
      **2729.23 @ 1.835 bpw** vs 2764.68 @ 1.710 at g128. NO — the g64
      SQNR edge (7.07 vs ~6.8 dB) buys nothing meaningful in ppl (1.3%
      better at +0.125 bpw). g128 stays the group size for the ternary
      reference; the fidelity anchor stays g128 across the board.
- [x] Quant R&D: does the g128-beats-g256 ppl verdict transfer to
      eval_text2 — ANSWERED 2026-09-21: YES, it transfers and the gap
      widens. int2_kmeans_q8 g256 on text2: **1971.77 @ 2.188 bpw** vs
      g128 1047.35 @ 2.375 (text1: 1085.61 vs 795.96 — 37% gap). g128
      wins on both texts, so the 795.96 anchor decision was not
      text-luck; the group-size verdict is text-robust.
- [ ] Quant R&D: shuffled-block eval sample — PARTIALLY LANDED
      2026-09-21: `ppl.py --eval-texts A.txt,B.txt` now evaluates every
      scheme on every text and reports per-text ppl + mean/std + x_fp32
      (184 tests green; `fp32` accepted as a pseudo-scheme for a
      same-table reference). Still open: the shuffled-block half —
      sample N disjoint blocks from a longer corpus (needs a corpus
      first; currently only two ~400-token hand-composed texts exist)
      and report mean/std per scheme across blocks.
- [x] Quant R&D: re-run the default (non-Lloyd) ternary sweep on
      eval_text2 via `--eval-texts` — PARTIALLY ANSWERED 2026-09-21
      (3-scheme slice: the 1.710-bpw fitted-vs-naive tiering question).
      lloyd >> 1step >> uniform on BOTH texts at matched bitrate, so the
      step-2b tiering is text-robust, not text-luck; the fitted-vs-1step
      gap widens on text2 (~40x vs ~25x). Remaining schemes
      (1step_ds, outlier, dual_scale) on text2: still open, low value
      (all in collapse territory, directional-only per the methodology
      item).
- [x] Quant R&D: OBQ first slice — ANSWERED 2026-09-21, delta ~0.
      Landed as `src/quant_rnd/fisher.py` (activation capture in
      gpt2_forward.py, diag-Fisher d_j = mean_t(x_{t,j}^2) per input
      channel), weighted `_lloyd_1d`, `sample_weight` on
      int2_kmeans_q8, and `ppl.py --fisher` (141 tests green, incl. a
      bit-identical unweighted-path pin). int2_kmeans_q8 g128 with
      Fisher reweighting: **795.93 @ 2.375 bpw** vs 795.96 unweighted —
      the reweighting provably changes the codebook (synthetic test)
      but buys nothing in perplexity. Per the item's decision rule:
      skip to full per-weight error-compensation (new item below); do
      NOT bother with a Fisher-weighted ternary_lloyd variant.
      Caveats: diagonal Hessian only; calibrated on the eval text
      itself (no held-out corpus on this VM yet).
- [x] Quant R&D: full per-weight error-compensation prototype
      (GPTQ/OBQ-style) — ONE-LAYER SLICE LANDED 2026-09-21 as
      `src/quant_rnd/obq.py` + `ppl.py --one-layer/--obq` (167 tests
      green). Damped inverse Hessian (Cholesky; 1% GPTQ damping,
      load-bearing: T=406 < d_in=768 so H is rank-deficient), GPTQ block
      update over input-channel rows, per-column 4-centroid Lloyd +
      8-bit codebook = int2_kmeans_q8 math at matched 2.375 bpw.
      Unit-verified: single-block == naive bit-identical; compensation
      strictly reduces the Hessian-weighted objective; block update
      pinned against a manual formula recomputation. Measured on
      h.0.attn.c_attn (768x2304), eval_text1: naive one-layer 53.34 vs
      OBQ one-layer **52.65** vs fp32 53.50 — directionally right at
      matched bitrate, machinery verified end to end. BUT: one layer is
      ~1.4% of params, so the probe is near the noise floor — the real
      verdict needs the full-model slice below.
- [ ] Quant R&D: OBQ full-model slice — apply quantize_layer_obq to all
      48 linear layers (one capture forward already gets every layer's
      X; per-layer Hessian inverse + block update, ~10-15 min on this
      VM) and measure ppl vs the 795.96 naive anchor at g128. Wire as
      `ppl.py --obq-all` reusing quantize_model's target loop. Decision
      rule: if full-model OBQ exits collapse territory materially
      (< ~400 ppl), the fidelity path is alive; if delta ~0, the
      Hessian-second-order story is exhausted on GPT-2 124M at this
      scale and the honest conclusion is logged.
- [ ] Quant R&D: unblock the full-model OBQ measurement — the 48-layer
      `ppl.py --obq-all` run cannot finish inside a ~25-min session
      (two runs killed by the execution timeout on 2026-09-21; machinery
      is committed at 7cb610e, 179 tests green — only the measurement
      is missing). Candidate unblockers, none attempted yet: (a)
      checkpoint per-block quantized weights to research/data/ so the
      run resumes across sessions; (b) detached nohup runner writing to
      durable storage, polled by later sessions (beware /tmp wipes on
      service restarts); (c) fewer eval tokens for the measurement run
      only (pre-registered methodology deviation). Build ONE of these
      (small, tested, committed) before re-attempting the measurement.
- [ ] Quant R&D: 1-step Lloyd re-fit of centroids is already the cheap
      fitted-ternary encoder; check whether its measured 0.41 zero-rate
      (vs 0.31 uniform) can be raised toward 0.5 (more sparsity -> more
      add-skip) without ppl collapse, e.g. threshold widening with the
      n_iter=20 scale.
- [x] Quant R&D: re-run the opcount energy/speed comparison with the
      dual fitted reference (`ternary_1step_ds` at n_iter=2, measured
      zero-rate per side) — LANDED 2026-09-21 as
      `test_dual_fitted_ternary_opcount_reference` (+ `side_fractions()`
      in opcount.py; 90 tests green). Verdict: figures barely move, as
      expected. Decode ceiling ratio 1.272x vs symmetric 1-step's 1.357x
      (lower because the dual reference carries 1.835 vs 1.710 bpw —
      byte-driven); energy-proxy ratio 3.53x clean / 3.40x skewed vs
      symmetric's 3.52x. Dual zero-rates: 0.441 clean, 0.418 skew 0.5;
      per-side split on skewed tensor pos=0.415 / neg=0.168 (the op-profile
      shift the re-run was meant to capture). n_scales=2 costs only
      ~0.016 muls/weight — genuinely "slight".
- [x] Quant R&D: when the tiny-model perplexity run selects a real group
      size (128 vs 256), re-run the opcount table at that group size —
      RESOLVED 2026-09-21 without re-running: perplexity selects g128
      (795.96 vs 1085.61 at g256), and the current decode ratios (1.36x
      sym / 1.27x dual) are already computed against int2_kmeans_q8 at
      g128 — the reference config stands.
- [ ] Quant R&D: vectorize the ternary Lloyd-fit encoder — currently a
      pure-Python per-group loop at ~2 Mparams/s (full-model
      ternary_1step ≈ 45 s, ternary_lloyd n_iter=20 ≈ 2 min). The
      2026-09-21 reconstruct fix (O(n^2) -> O(n), 1000x) unblocked the
      step-2b sweep; the encoder is now the dominant cost per scheme.
      Careful: keep bit-identical numerics vs the per-group float64 path
      (pin with a test), or document any intentional deviation.

## Ground rules for this research track

- Every change is a git commit in `~/workspace/local-agent`. Revert with
  `git log --oneline` + `git revert <hash>`.
- Never break the working default (`qwen3:8b` via Ollama). Experimental
  profiles stay opt-in.
- Don't download multi-GB models on the dev VM; use file-size metadata.
- Each research sweep appends a dated entry under `research/log/`.
