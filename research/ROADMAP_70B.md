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
- [x] Quant R&D: OBQ/GPTQ-style second-order correction prototype — the
      honest next fidelity step now that pure group-wise is exhausted on
      both SQNR and perplexity. Start with per-group Hessian-diagonal
      (empirical Fisher from a few forward passes) reweighting of the
      Lloyd/ternary fit; measure whether ppl exits collapse territory.
      CLOSED 2026-09-21 — superseded by the three landed slices below:
      Fisher reweighting (795.93, delta ~0), one-layer OBQ probe (52.65,
      directionally right), and the full-model OBQ verdict (475.48 @
      2.375 bpw vs 795.96 naive, missed the <400 bar — the group-wise
      fidelity ladder is exhausted; the climb now goes through bigger
      models + speed optimization, per standing direction).
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
- [x] Quant R&D: add embedding-table (wte) quantization to the real-weight
      probe — ANSWERED 2026-09-21 (commit 564b53a): the collapse is 100%
      the wte table. `--only-names` + `int8_uniform` (q8 reference, 8.125
      bpw @ g128) landed in ppl.py. Measured on eval_text1, g128, fp32
      ref 53.50: wte-only @ int2_kmeans_q8 -> ppl inf (complete collapse;
      wte is the tied lm_head, so 2-bit destroys the output projection);
      wpe-only @ 2-bit -> 53.81 (unharmed); wte-only @ q8 -> 54.76
      (+2.4%, survives); wpe-only @ q8 -> 53.48 (unharmed). Mac-relevant
      rule, now measured: recipes must keep embedding/lm_head tables at
      >= 8-bit. Supersedes the old note about naive low-bit embedding
      probes; per-row-scaled is a refinement, not a question (see new
      follow-up below).
- [ ] Quant R&D: q4 / per-row-scaled embedding probe — refinement of the
      answered wte item: is there a bitrate between 2-bit (collapse) and
      q8 (54.76) where the tied head survives? Only worth a session if a
      Mac recipe needs sub-8-bit embeddings (low priority).
- [ ] Quant R&D: layer-wise mixed-precision harness — the `--only-names`
      flag (landed 2026-09-21) gives per-tensor targeting; the remaining
      piece for the fidelity-retrospective's sensitivity-adaptive idea is
      a per-layer scheme selector (early/late layers at higher precision,
      matched average bpw). One matched-bitrate experiment max before
      re-evaluation, per the retrospective's rule. (NEW 2026-09-21)
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
- [x] Quant R&D: add the attention O(L^2) term to prefill_roofline_tps for
      long context — LANDED 2026-09-21 (commit cd60e81, mirrored, 203
      tests green). `include_attention=True` default (n_q_heads=64):
      4*n_layers*n_q_heads*L^2*head_dim FLOPs (MAC=2) + KV read = KV
      write; `include_attention=False` recovers the matmul-only model.
      Honest consequence at 32k+: attention DOMINATES the matmuls, so
      the quant-scheme prefill ceiling ratio compresses toward 1
      (scheme-independent term swamps the per-weight FLOP difference) —
      the prefill case for ternary over k-means largely evaporates at
      very long context, while decode (bandwidth-bound, attention
      O(L^2) absent from decode per-token) keeps the 1.36x ratio. The
      "under-count at 128k" caveat is now closed.
- [x] Quant R&D: crossover context length — PINNED 2026-09-21 via
      `prefill_crossover_L` in opcount.py (205 tests green). For the
      reference pair (ternary_1step 1.710 bpw vs int2_kmeans_q8 2.375 bpw,
      70B scale / 200 GB/s / 5.2 TFLOPS peak): prefill ceiling ratios
      4k=1.63x, 8k=1.52x, 16k=1.39x, 32k=1.26x, 64k=1.15x, 128k=1.09x.
      The ratio drops below 1.2x between 32k and 64k (crossover_L=65536),
      below 1.5x at 16384. Practical read: the ternary prefill compute
      case survives at ordinary prompt lengths (<=32k) but compresses to
      noise at 64k+; decode (bandwidth-bound, no attention term) keeps
      the 1.36x ratio at ALL context lengths. Crossover is a model
      output (70B-shaped Llama geometry, GQA-8, head_dim 128) — re-pin
      if the target architecture changes.
- [ ] Quant R&D: Mac-side validation of the crossover — on Justin's Mac,
      measure prefill tok/s for matched-size ternary/TQ1_0 vs 2-bit
      codebook models at prompt lengths spanning 4k-64k and check
      whether the measured ratio curve tracks the model's 1.63 -> 1.15x
      compression (needs Justin's Mac; extends both prefill-validation
      items below)
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
- [x] Speculative decoding support in the llamacpp backend (draft + target model)
      — LANDED 2026-09-21 (13:15 session): LlamaCppClient gains
      speculative="off"|"prompt_lookup"|"draft_model", draft_model_path,
      draft_n_tokens (upstream default 10; llama.cpp docs note 2 is better
      on CPU-only). prompt_lookup uses the documented
      LlamaPromptLookupDecoding(num_pred_tokens=...) API wired through
      Llama(draft_model=...); draft_model mode loads a small draft GGUF
      via LlamaDraftModel — constructor signature not verifiable on this
      VM (package absent), so it is marked untested-on-Mac with a graceful
      TypeError->LLMError translation. Validates mode/paths up front.
      Default "off": existing behavior unchanged (suite pins it). Config
      keys llamacpp.speculative / draft_model_path / draft_n_tokens wired
      in agent.py as keyword args. 14 new tests green (fake llama_cpp
      module injection), full suite green.
- [ ] Speculative decoding Mac-side measurement — on Justin's Mac, measure
      decode tok/s with speculative="prompt_lookup" (num_pred_tokens=10
      vs 2) and with a small draft GGUF (draft_model mode) against the
      non-speculative baseline, at matched quant; quantify the realized
      decode speedup vs the opcount ceiling ratios (decode is
      bandwidth-bound, so acceptance rate decides).
- [x] Model cascade: small fast model by default, escalate hard queries to
      the big model — LANDED 2026-09-21 as `src/cascade.py` + opt-in
      `backend: cascade` config (commit 6eb10d8, 236 tests green). Two
      routers: "heuristic" (one pass; escalate if last user message >2000
      chars or hits >=2 complexity keywords — rule is documented and
      tunable) and "verify" (always try small first; escalate on refusal
      phrasing or empty-with-no-tool-calls; a bare empty reply WITH tool
      calls is a normal tool-use turn and does not escalate). Big client
      built lazily via factory on first escalation, so the small model is
      the only resident RAM cost until the cascade fires; no unload API
      in v1 (restart drops the big model). Per-backend dicts take the same
      keys as the top-level config blocks. Default `backend: ollama` +
      `qwen3:8b` untouched; clear LLMError if no `big` spec.
- [ ] Mac-side: cascade validation — qwen3:8b resident + 35B-A3B IQ2_M
      (10.6 GB) via llamacpp as the big model; measure escalation rate on
      real usage, resident-RAM before/after first escalation, and whether
      the heuristic thresholds (≈2000 chars / 2 keyword hits) over- or
      under-escalate (needs Justin's Mac)
- [ ] Cascade router calibration — log (heuristic score, verify decision,
      final route) per turn and review after a week of real use to tune
      the keyword set and thresholds; consider a learned router only if
      the rule-based one mis-routes measurably
- [x] Prototype MLX backend (mac-only; can't be tested on Linux — needs Justin's Mac)
      — LANDED 2026-09-21 as `src/mlx_backend.py` + opt-in `backend: mlx`
      wiring (commit 02e4d77, 254 tests green). `MlxClient` wraps mlx-lm's
      long-stable API (`load` / `generate` / `apply_chat_template`,
      verified against upstream docs 2026-09-21) with graceful LLMError
      degradation when mlx-lm is absent. Config: model (HF repo id or local
      dir), temperature, top_p, max_tokens (1024 default — the agent
      loops), repetition_penalty, seed, adapter_path. Same
      chat(messages, tools) interface; tool calls use the standard
      text-based fallback (tool schemas -> system instruction ->
      {"tool_calls": [...]} JSON parse, unknown names/malformed JSON fall
      back to plain text) — marked EXPERIMENTAL, untested-on-Mac. Tool
      results are folded into user text since chat templates have no tool
      role. Default `backend: ollama` + `qwen3:8b` untouched. Fake-module
      tests (19 new) verify prompt construction, sampler mapping, tool
      parse/fallback, and error paths; real behavior needs the Mac.
- [ ] Mac-side: MLX backend smoke test — `pip install mlx-lm`,
      `backend: mlx` with a small mlx-community 4-bit model; confirm
      chat works, then measure decode tok/s vs the ollama/llamacpp
      backends at matched quant (needs Justin's Mac)
- [ ] Mac-side: MLX backend tool-calling reliability — measure how often
      the text-based {"tool_calls": [...]} parse succeeds on real agent
      turns vs Ollama's server-side tool calling; if unreliable, consider
      wiring `mlx_lm.server` (OpenAI-compatible, localhost:8080) as the
      transport instead of the Python API (needs Justin's Mac)
- [x] Hearth: documented MLX recipe for the 35B-A3B — LANDED 2026-09-21
      as `research/recipes/RUNG4_oq2_35b_smelt.md` (rung-4 ladder recipe:
      Jundot/Qwen3.6-35B-A3B-oQ2 13.10 GB measured via HF tree API;
      arch from config.json text_config — 40 layers, GQA-2 KV heads,
      head_dim 256, full_attention_interval=4 (10 KV layers), 256 experts
      8+shared; quant layout measured from quantization_config: routed
      experts 2-bit g64, linear_attn 4-6 bit, self_attn/shared/embeds/
      lm_head 8-bit; 333-tensor vision tower present, text-only under
      smelt; NO mtp tensors in the weights despite mtp_num_hidden_layers=1
      — no MTP speculative path). fitcheck gained arch `qwen3.6-35b-a3b`
      + `moe_expert_gb()` (8.05 GB routed experts) + `smelt_resident()`
      (smelt-50 -> 9.07 GB weights, ~10.1 GB total, fits ~11 GB; smelt-25
      -> 7.06 GB, ~8.1 GB total); 8 new recipe-consistency tests, 325
      green. Honest correction folded in: the oQ2 profile's old "~7 GB
      resident est." is now the computed ~10.1 GB. The recipe's vmlx path
      is NOT the `mlx` backend (mlx-lm cannot load oQ2's affine format)
      and NOT a cascade big model yet — both need the OpenAI-compatible
      transport item below; the recipe documents that explicitly.
- [x] Hearth: OpenAI-compatible HTTP backend (vmlx / mlx_lm.server
      transport) — LANDED 2026-09-21 as `src/openai_backend.py` +
      `backend: openai` wiring (353 tests green, incl. 19 new against a
      stub HTTP server). `OpenAICompatClient`: provider-neutral
      /v1/chat/completions, OpenAI-native tool_calls passthrough, optional
      api_key (or OPENAI_API_KEY env), base_url normalization (root / /v1
      / full path). Wired into Agent + cascade big-spec + config.yaml
      commented example; RUNG4 recipe's wiring note updated to point at
      it. Unblocks the rung-4 recipe's Hearth wiring (cascade big =
      vmlx-served oQ2-smelt) AND the MLX tool-calling follow-up
      (mlx_lm.server transport). Still Mac-side: the transport itself is
      untested against a real server (see new item below).
- [ ] Mac-side: OpenAI-backend transport validation — against `vmlx serve`
      (with the Jundot oQ2 model) and `mlx_lm.server`: confirm chat
      works, measure whether the server's tool_calls actually fire on
      real agent turns (mlx_lm.server tool support is the specific
      unknown), then wire the cascade big = vmlx-smelt config for the
      rung-4 validation item (NEW 2026-09-21; transport is untested-on-Mac).
- [x] tools/measure_openai.py — LANDED 2026-09-21: generalizes the rung
      harness to OpenAI-compatible endpoints (vmlx, mlx_lm.server).
      Reuses measure_rung's pure core (SPEED_PROMPTS, QUALITY_CHECKS,
      run_rung, format_report) via import; adds a streamed
      /v1/chat/completions generate_fn returning an Ollama-shaped payload
      (eval_count from usage.completion_tokens when the server provides it,
      else chunk-counted and labeled; wall-clock eval_duration). Same
      PASS/FAIL report + extra STREAM META lines (ttft, token source).
      `<think>` blocks stripped from quality-check text (thinking tokens
      still count toward decode tok/s — honest). Base-URL accepts root,
      /v1, or the full path; api key via --api-key/OPENAI_API_KEY. Default
      --target 10.0 (the rung-4 pass bar). 11 new tests (fake SSE server),
      suite green. RUNG4 recipe's manual snippet replaced by the tool.
- [ ] Track: oQ2 64% MMLU figure is measured on the 3.5 variant (oMLX
      docs); the Jundot 3.6 variant is unmeasured — the Mac-side battery
      score vs rung 3 is the measurement, not this figure (NEW 2026-09-21).
- [ ] Mac-side: verify whether `vmlx serve` accepts an HF repo id directly
      or needs a local snapshot dir (the RUNG4 recipe downloads first —
      note which form was used) (NEW 2026-09-21).
- [x] tools/eval_battery.py — OpenAI-compatible transport for the 19-prompt
      battery, so rung 4+ (vmlx / mlx_lm.server) can be scored with the same
      checks as rungs 1-3. LANDED 2026-09-21 as
      `tools/eval_battery_openai.py` (15 new tests, suite green). Same trick
      as measure_openai.py: reuses the battery's check/prompt/scorecard
      definitions via import (identity-pinned in tests), swaps the generate
      transport for non-streamed /v1/chat/completions (temp=0, max_tokens
      512, stream_options unnecessary — timing isn't measured). Reuses
      measure_openai's normalize_base_url + strip_think; OpenAI error
      shapes (dict/string) surface loudly via BatteryError. One-or-two
      model CLI, completion-token tally from usage per run, exit 0/1/2
      semantics (1 = any check or request failed). RUNG4 recipe's manual
      prompt note replaced with the tool invocation — the rung-4
      intelligence verdict gap is closed.
- [ ] Cross-transport battery parity — run tools/eval_battery.py (Ollama
      /api/generate, temp=0, num_predict 512, think=false) and
      tools/eval_battery_openai.py (vmlx/mlx_lm.server, temp=0,
      max_tokens 512, <think>-stripped) against the SAME model and
      confirm identical scorecards, so the rung-3-vs-rung-4 comparison
      isn't confounded by transport differences (needs Justin's Mac;
      only meaningful once a model is served both ways). (NEW 2026-09-21)
- [ ] Benchmark harness: quality-vs-quant curves on small models to validate the pipeline
- [ ] Track BitNet.cpp releases + any 70B ternary model announcement
- [ ] Track oQ/JANG releases and 2-bit MoE quality reports
- [x] Quant R&D: low-active-parameter MoE survey — SURVEYED 2026-09-21
      (research/moe_survey_2026-09-21.md). Central candidate confirmed:
      Qwen3.5-35B-A3B (35B total / 3.3B active, 256 experts 8+1 shared,
      40 layers). In-budget in-RAM: unsloth IQ2_M GGUF (10.6 GB, ~2.45
      bpw) — quality at that bitrate UNVERIFIED, the week's key unknown.
      Borderline: mtrpires mixed-IQK (11.38 GB disk / ~12.5 GB RAM,
      principled attention/shared-expert protection) and
      unsloth Qwen3.6-35B-A3B-MTP IQ2_XXS (11.819 GB, native MTP head —
      composes with the speculative backend). Best benchmarked ~2-bit:
      oQ2 MLX (MMLU 64.0% vs naive 2-bit 14.0%, HumanEval 78.0%,
      MBPP 63.3%, TruthfulQA 80.0%) but ~12.6 GB — over in-RAM, needs
      flash-paging. Jundot publishes prebuilt oQ models on HF. Verdict:
      no option is 70B-dense quality; "70B-class" must be measured vs
      dense models that fit 16 GB, not assumed. Recommended Mac
      validation order in the survey note.
- [ ] Mac-side: Qwen3.5-35B-A3B UD-IQ2_XXS (10.66 GB measured) — measure
      tok/s + quality (MMLU subset) on the M1 Pro per the rung-3 recipe
      (research/recipes/RUNG3_qwen3_5_35b_a3b_moe.md); compare vs the oQ2
      64% MMLU datapoint (needs Justin's Mac). NOTE 2026-09-21 pm: the
      survey's "IQ2_M 10.6 GB" figure was wrong — HF tree API re-measure:
      IQ2_M = 11.39 GB (does not fit the ~11 GB budget), IQ2_XXS =
      10.66 GB. The rung-3 file is now IQ2_XXS, not IQ2_M.
- [ ] Rung-3 Mac-side validation — run the rung-3 recipe end to end on
      Justin's Mac (download, Ollama Modelfile, measure_rung.py --target
      15); record measured tok/s, which tuning steps were needed, and the
      quality-sanity outcome. The IQ2_XXS-vs-dense-8B quality-per-GB
      comparison (MMLU subset or fixed task battery) is the follow-up
      measurement that decides whether the rung means anything.
- [ ] Mac-side: mtrpires mixed-IQK (11.38 GB) — does it hold under memory
      pressure with q4_0 KV cache at 8–12k ctx? (needs Justin's Mac)
- [x] Hearth: documented rung-3 recipe/profile for the 35B-A3B — LANDED
      2026-09-21 as `research/recipes/RUNG3_qwen3_5_35b_a3b_moe.md`
      (primary file UD-IQ2_XXS 10.66 GB measured, not the survey's stale
      IQ2_M figure; fitcheck gained arch `qwen3.5-35b-a3b` = (40, 2, 256)
      from the official config.json; pass bar >= 15 tok/s usable, stretch
      20+; two-bound roofline story 18.8 tok/s floor vs ~197 active-traffic
      fiction; llamacpp + cascade opt-in snippets incl. prompt_lookup for
      the MTP variant). 8 new tests, recipe-consistency tests pin the
      RAM/roofline figures. The Mac-side validation item above is the
      remaining half.
- [x] Re-check HF for a prebuilt oQ2/oQ2.5 35B-A3B MLX upload (Jundot org) —
      ANSWERED 2026-09-21: `Jundot/Qwen3.6-35B-A3B-oQ2` EXISTS (uploaded
      2026-04-23, oMLX v0.3.7, 2-bit group 64, MLX safetensors) and
      measures **13.10 GB** via the HF tree API (3 shards: 5.03 + 5.03 +
      3.02 GB). Slightly above the ~12.6 GB survey estimate (est. was for
      3.5; this is the 3.6 variant, ~3.03 eff. bpw). Quality-per-GB
      re-evaluation: at 64% MMLU it is the best-benchmarked ~2-bit option,
      but 13.1 GB cannot fit fully resident — the fit story is Smelt
      paging, not residency.
- [x] Data hygiene: `research/model_profiles.yaml` oQ2 entry corrected
      2026-09-21 — the stale `qwen3.5-35b-a3b-oQ-2bit @ est_gb 8.8`
      ("Fits comfortably") is now `qwen3.6-35b-a3b-oQ2 @ est_gb 13.1`
      measured, with the real repo id and the Smelt fit story in notes.
      A test (`TestModelProfiles.test_oq2_profile_size_measured`) pins the
      measured figure so the profile can't silently drift again.
- [ ] Mac-side: Jundot oQ2 (13.1 GB) under vmlx --smelt 50 — measure
      resident RAM, decode tok/s, and quality (MMLU subset) on Justin's
      M1 Pro; check whether quality tracks the oQ2 64% MMLU baseline or
      degrades from the resident-expert routing bias. If IQ2_XXS quality
      disappoints on rung 3, this is the first fallback path.
- [ ] Quant R&D: vmlx "Smelt" mode (partial expert loading) — NEW 2026-09-21
      from the JANG release watch: vmlx README documents `--smelt` /
      `--smelt-experts N` for MoE models that don't fit in RAM - keeps the
      backbone resident, loads a subset of experts/layer from SSD, biases
      routing toward resident experts. Benchmark (Nemotron-Cascade-2-30B-A3B
      -JANG_4M, M3 Ultra 128 GB): 50% experts -> 9.5 GB RAM (-45%), 66.5
      tok/s (vs 17.4 GB / 89.9 baseline); 25% -> 5.6 GB (-68%).
      Implication for the 35B-A3B path: the oQ2 35B-A3B (13.1 GB measured
      2026-09-21) is over-budget fully resident but could fit 16 GB under
      Smelt-50 at ~2/3 speed - changes the "borderline" classification. Also of note
      from the same watch: oQ+ adds GPTQ weight optimization before
      quantization (sensitivity-driven bit allocation, batched over all
      routed experts), and JANG profiles are now explicit (JANG_2M/2L/3M/
      4M/6M, attn 8-bit / MLP 2-6-bit). Needs Justin's Mac: install vmlx,
      serve the oQ2 35B-A3B with --smelt 50, measure quality + tok/s.
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
- [x] Quant R&D: OBQ full-model slice — ANSWERED 2026-09-21
      (checkpointed protocol completed across 3 runs: 12 + 24 + 12
      layers; manifest at research/data/obq_ckpt/, per-column 4-centroid
      Lloyd + 8-bit codebook at matched 2.375 bpw g128, damped inverse
      Hessian, calibrated on eval_text.txt). **Full-model OBQ ppl:
      475.48 @ 2.375 bpw** vs naive anchor 795.96 (eval_text1, 406
      tokens). Verdict: 1.67x better than naive, but 8.9x above fp32
      (53.50) — still in collapse territory per the harness's own
      decision rule. The <~400 bar for "fidelity path alive" was NOT
      met. Honest conclusion per the item's protocol: second-order
      error compensation at ~2.4 bpw cannot reach usable quality on
      this scale; the pure-group-wise fidelity ladder is exhausted
      (naive -> Lloyd -> Fisher-reweight -> full OBQ), and no further
      small-scale fidelity iteration at this bitrate is expected to
      change that. The quant-R&D story now rests on the compute side
      (ternary add/sub kernels, opcount 1.36x decode / 1.79x prefill
      ceilings) and on fundamentally different foundations
      (native-ternary models, MoE, Mac-side validation). The
      checkpointing protocol worked as designed: 12 + 24 + 12 layers
      across three sessions, bit-identical assembly, survived a
      background restart.
- [ ] Quant R&D: fidelity-ladder retrospective — naive -> Lloyd k-means
      -> Fisher reweighting -> full OBQ all collapsed at ~2.4 bpw
      (475 best), so the next fidelity ideas must be STRUCTURALLY
      different to be worth a session: e.g. sensitivity-adaptive bit
      allocation across layers (early/late layers at higher precision)
      at matched average bpw, or per-layer mixed schemes with the bpw
      budget reallocated by Fisher-trace sensitivity. Any such idea
      gets one matched-bitrate experiment max before re-evaluation.
- [x] Quant R&D: unblock the full-model OBQ measurement — PARTIALLY
      LANDED 2026-09-21 as `src/quant_rnd/obq_ckpt.py` + `ppl.py
      --obq-all --obq-ckpt-dir DIR [--obq-max-layers N]` (commit
      6fa4635, 202 tests green, mirrored). Per-layer quantized weights
      checkpoint to research/data/obq_ckpt/ (gitignored) as each layer
      finishes; runs skip done layers and resume from the manifest,
      which refuses to mix incompatible params (group_size / damp /
      n_iter / calibration-text hash). Resumed assembly is
      bit-identical to an in-memory --obq-all (test-pinned). First
      chunk (12 of 48 layers) launched this session; when the last
      layer lands the run assembles the model and prints the verdict
      vs the 795.96 anchor with the <~400 decision rule. Remaining
      work is just running the chunks — the full-model slice item
      below is now unblocked and carries the protocol note.
- [x] Quant R&D: 1-step Lloyd re-fit of centroids is already the cheap
      fitted-ternary encoder; check whether its measured 0.41 zero-rate
      (vs 0.31 uniform) can be raised toward 0.5 (more sparsity -> more
      add-skip) without ppl collapse, e.g. threshold widening with the
      n_iter=20 scale. ANSWERED NO 2026-09-21 (sharp honest negative):
      `_ternary_lloyd_fit` gained a `thresh_factor` kwarg (threshold-biased
      Lloyd, default 1.0 bit-identical, 5 new tests) and
      `ternary_1step_sp` (k=1.2) is registered in SCHEMES. Probe on all 48
      GPT-2 linear matrices, g128: k=1.2 -> zero-rate 0.419 -> 0.514 at
      only -0.05 dB SQNR (6.52 -> 6.47). Perplexity (eval_text1, 406
      tokens): **45242 @ 1.710 bpw vs 2764 for ternary_1step** - a 16x
      blowup from a -0.05 dB fidelity delta. The collapse-territory
      cliff is razor-sharp (same phenomenon as the 25x full-vs-1step
      Lloyd gap at 0.2 dB): even the ~0.4 zero-rate of ternary_1step is
      load-bearing, and SQNR cannot see the damage. Zero-rate 0.5 is not
      a free add-skip; sparsity must come from the fit itself, not the
      threshold. Energy-proxy side (unrealizable at this fidelity):
      3.51x -> 4.22x over int2_kmeans_q8 at zero-rate 0.514.
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
- [ ] Rung-1 Mac-side validation — RECIPE READY 2026-09-21:
      `research/recipes/RUNG1_qwen3_8b_20tps.md` + `tools/measure_rung.py`
      (282 tests green incl. 23 new). On Justin's Mac: `ollama pull
      qwen3:8b`, then `python3 tools/measure_rung.py --model qwen3:8b
      --target 20`. Expected RAM ~5.6-6.0 GB (fitcheck: 4.9 GB weights +
      0.3 GB KV @ 2048 ctx); roofline ~40 tok/s so 20 is expected but only
      the measurement counts. Pass = mean decode >= 20 tok/s + 3 quality
      sanity checks. Tuning ladder in the recipe (Metal offload check,
      --num-ctx 2048, OLLAMA_KV_CACHE_TYPE=q8_0). Paste the script output
      back as the measurement record.
- [x] Rung-2 recipe: 14B-class at 20 tok/s — LANDED 2026-09-21 as
      `research/recipes/RUNG2_qwen3_14b_20tps.md` (roofline 200/9.0 ≈
      22 tok/s — TIGHT, expect tuning), `tests/test_rung_recipes.py` pins
      the recipe's RAM/roofline figures against fitcheck + checks every
      recipe referenced in model_profiles.yaml exists (291 tests green).
      qwen3:14b profile est_gb corrected 8.6 -> 9.0 (14.7B params x
      4.9 bpw). measure_rung.py needed no new knobs (--num-ctx/--target
      already exist).
- [ ] Rung-2 Mac-side validation — run the rung-2 recipe on Justin's Mac;
      record measured tok/s + which tuning steps were needed.
- [ ] Rung-2 fallback candidates (CONDITIONAL — only if rung-2 misses 20
      tok/s at Q4_K_M after the tuning ladder): qwen3:14b at Q4_K_S
      (~8.0 GB, ceiling ≈ 25 tok/s) / Q4_0 (~8.3 GB, ≈ 24 tok/s), or
      IQ3_M (~6.8 GB, ceiling ≈ 29 tok/s — quality at 3.7 bpw
      unverified); each needs the quality sanity re-run + a fitcheck
      entry before it counts as a rung.
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
