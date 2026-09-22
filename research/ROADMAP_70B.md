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
- [x] Quant R&D: methodology caveat for future sweeps — WORKED EXAMPLE
      LANDED 2026-09-22: perplexity only discriminates below ~10x the fp32
      reference. The shuffled-block run (below) measures x_fp32 = 15.67,
      so in collapse territory the report is a directional distribution,
      not a quality figure for the 70B case. The naive anchor is now
      quoted as a text-conditioned range (676 +/- 288 over 8 blocks),
      not the 795.96 point.
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
- [x] Quant R&D: layer-wise mixed-precision harness — LANDED 2026-09-21
      as `ppl.py --per-layer-schemes SPEC` (explicit per-block assignment,
      e.g. "0-5:int8_uniform,6-11:ternary_1step") and
      `--sensitive-layers K --sensitive-scheme S --base-scheme B` (top-K
      Fisher-trace blocks at S, rest at B), `fisher.layer_fisher_trace`
      (per-block diag-Fisher trace on the eval text — the same
      calibration-on-eval caveat as --fisher), `quantize_model_per_layer`
      with PARAMETER-WEIGHTED average bpw (the matched-bitrate
      denominator), strict coverage rules (every block assigned exactly
      once; KeyError on gaps) so no block is silently fp32. 16 new tests
      (incl. a gated trace test on the real checkpoint), 404 green.
      Per the fidelity-retrospective's one-experiment rule the matched
      experiment is DONE and it is a NEGATIVE (see that item).
- [ ] Quant R&D: per-tensor (not per-block) sensitivity granularity —
      the block-level harness can't hit arbitrary target bitrates exactly
      (12 coarse blocks: K=1 int8 + rest ternary_1step_ds lands 2.359 vs
      the 2.375 anchor). A per-tensor top-K selector would allow exact
      bitrate matching and test whether the negative is granularity, not
      principle. Low priority given the verdict below. (NEW 2026-09-21)
- [x] Quant R&D: measure activation drift under quantization — the
      sensitivity experiment's leading explanation is that the fp32
      Fisher ranking is invalidated once other blocks are quantized.
      ANSWERED 2026-09-21 (harness landed commit c7aa8b6:
      `ppl.py --sensitivity-mechanism SCHEME` — leave_one_out_ppl +
      trace_on_quantized + spearman_rho, 415 tests green; full 12-block
      run completed this session with ternary_1step_ds, g128,
      eval_text1). The verdict is STRONGER than the drift hypothesis:
      **rho(fp32-trace, leave-one-out damage) = -0.035** — the fp32
      ranking never predicted block-wise damage AT ALL (it isn't a valid
      ranking that quantization then invalidates). Block 0 is the
      cleanest proof: LEAST sensitive by fp32 trace (180633) yet MOST
      damaging when quantized alone (ppl 1061.85, +1008 delta) — the
      first block's errors cascade through all 11 downstream blocks,
      which a local trace measure fundamentally cannot see. Block 11 is
      the one case the ranking gets right-ish (most sensitive 767860,
      second-most damaging 143.65). rho(fp32-trace,
      quantized-activation trace) = +0.217 — quantization also shifts
      the trace ranking (weakly tracks). Honest read: fp32-trace-guided
      bit allocation was unprincipled from the start, not just drifted;
      the sensitivity-adaptive negative needed no drift to lose. Thread
      CLOSED. (Determinism note: blocks 0-5 of a previous-session run
      killed at block 5 reproduced bit-identically in the re-run.)
      Caveats: collapse territory (directional), calibration on the eval
      text, GPT-2 124M scale.
- [x] Quant R&D: activation-drift probe, scheme-specificity check —
      DECIDED AGAINST 2026-09-21: the ternary_1step_ds run answered the
      mechanism question decisively (rho_damage = -0.035 — the fp32
      ranking never predicted damage, so there is no scheme-dependent
      drift pattern left to test). A second scheme's damage ranking
      would test whether DAMAGE is scheme-dependent, not whether the
      fp32 ranking is valid — and that doesn't change the verdict that
      fp32-trace-guided allocation was unprincipled. Fidelity thread
      fully closed; no more session time on it.
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
      any quality evaluation. RE-SURVEYED 2026-09-21 (pm session): still
      nothing — TQ1_0 hits are the same 397B oddity
      (Anjielon/ODINO-397B-v34a-TQ1_0) plus unrelated small repos; TQ2_0
      hits are all non-GGUF noise. RE-SURVEYED 2026-09-22 (am): still nothing
      >=70B — TQ1_0 hits unchanged; the TQ2_0 ecosystem is visibly maturing
      at small scale (TriLM, Ternary-Bonsai, ERNIE-4.5-VL-28B-A3B-TQ1_0/TQ2_0,
      Gemma3-27B-TQ2_0, bitnet-family TQ2_0 conversions) — real tooling
      love for the format, but no 70B file. Item stays WATCH.
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
- [ ] Mac-side: vmlx speculative decoding — the vmlx README (doc-checked
      2026-09-22) exposes server-side `--speculative-model <model>`
      (draft model, 20-90% speedup claimed) and `--enable-pld` (prompt
      lookup decoding, no draft model — "best for structured or
      repetitive output: code, JSON, schemas"). Once the rung-4 oQ2
      Smelt path is measured, quantify decode tok/s with PLD on/off and
      with a small draft model at matched quant; this is the rung-4
      speed-optimization layer via the vmlx transport, complementing the
      llamacpp-backend speculative item above. (NEW 2026-09-22)
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
- [x] Mac-side: verify whether `vmlx serve` accepts an HF repo id directly
      or needs a local snapshot dir — ANSWERED YES 2026-09-22 (doc check,
      no Mac needed): vmlx README says "Point it at a HuggingFace repo or
      local path and go" (quickstart: `vmlx serve
      mlx-community/Qwen3-8B-4bit`; distributed example: `vmlx serve
      JANGQ-AI/... --distributed`). RUNG4 recipe updated: serve
      `Jundot/Qwen3.6-35B-A3B-oQ2` straight from the repo id, download
      step dropped (snapshot fallback documented). Related correction
      folded into the recipe: vmlx's CLI server defaults to port 8000,
      not the measure_openai.py 8080 default. (NEW 2026-09-21)
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
- [x] Benchmark harness: quality-vs-quant curves on small models to
      validate the pipeline — CLOSED 2026-09-22: landed in substance, no
      separate harness needed. `python3 -m src.quant_rnd.ppl` runs
      per-scheme perplexity sweeps with measured bpw on GPT-2 124M
      (7-scheme default; fp32 reference 53.50, sane), and
      `src/quant_rnd/sweep.py` draws the SQNR-vs-bpw Pareto frontier
      (matched-bitrate sweeps, seed 7, skew variants). The pipeline is
      validated end to end by the honest fidelity ladder it produced
      (naive -> Lloyd -> Fisher-reweight -> full-OBQ verdicts, all
      measured against the 795.96 anchor).
- [ ] Track BitNet.cpp releases + any 70B ternary model announcement
      (SWEEP 2026-09-21 pm: still nothing — avenue-A trigger not hit.
      microsoft/BitNet's largest public model remains BitNet-b1.58-2B-4T
      (2.4B, 4T tokens); an independent 2026 well-trained-models survey
      (updated ~2026-09-16) finds no announced or in-progress 7B+ 1.58-bit
      model with 1T+ tokens expected in 2026, and microsoft/BitNet's
      "model-release" issues are empty. SWEEP 2026-09-22 (am): still nothing —
      microsoft/BitNet's largest public model remains BitNet-b1.58-2B-4T
      (2.4B, 4T tokens); the supported-model ceiling is still ~2-8B ternary.
      Avenue-A trigger not hit. Re-check periodically.)
- [ ] Track oQ/JANG releases and 2-bit MoE quality reports (SWEEP
      2026-09-21 pm: vmlx README (updated ~2026-09-17) adds an explicit
      JANG profile table, Smelt benchmarks unchanged, and a new
      `--calibration-method activations` flag ("better at 2-3 bit") —
      use it when converting locally. oMLX 0.6.4 allows TurboQuant KV
      cache + Lightning MTP together — relevant to the MTP speculative
      path. MiniMax-M2.5 JANG_2L datapoint: 74% MMLU at 82.5 GB vs 26.5%
      for standard MLX 4-bit at 119.8 GB (author-reported). SWEEP
      2026-09-22 (am): vmlx has moved orgs (vink-ai -> jjang-ai/vmlx);
      Smelt benchmarks unchanged. JANGQ-AI is very active — new
      Qwen3.8-Flash-Next JANG 4S/4M/6S uploads (1.3-1.7k downloads) and
      Qwen3.6-35B-A3B JANGTQ4 (19.71 GB) / JANG_4K (19.67 GB) — 4-bit-ish
      profiles, too big to matter. Jundot: Qwen3.6-35B-A3B oQ3e-mtp
      (17.23 GB, 2026-07-02) and oQ4e-mtp (21.64 GB) — both larger than
      the rung-4 pick (oQ2, 13.10 GB), so they don't improve the fit
      story; the -mtp variants carry the MTP head the oQ2 weights lack.
      NEW ecosystem players this sweep (items below): ddalcu/mlx-serve,
      novamlx (MoE-aware SSD streaming), mlxl3 (EXL3 on MLX). Re-check
      periodically.)
- [ ] Track: novamlx (cnshsliu/novamlx) — NEW 2026-09-22: pure-Swift
      Mac-native LLM server with NovaMLX-TIE, a 3-tier (wired / LRU /
      SSD-mmap) inference engine with MoE-aware router-driven expert
      prefetch and per-expert LRU — architecturally a smarter Smelt for
      the 13.1 GB oQ2 on 16 GB (dynamic prefetch vs vmlx Smelt's fixed
      expert subset + routing bias). Compare resident RAM, decode tok/s,
      and quality head-to-head with vmlx --smelt 50 when the rung-4
      validation runs. (SWEEP 2026-09-22)
      DOC CHECK 2026-09-22 (am): features.md pins model formats to
      SafeTensors 4-bit / 8-bit / FP16 / NVFP4 pre-quantized — NO 2-bit,
      no oQ/JANG custom layouts, so it CANNOT serve the Jundot oQ2
      2-bit weights; the "smarter Smelt" comparison is moot for rung 4's
      pick. TIE conversion is a per-model script
      (expert_shard_layout.py). API is SSE OpenAI/Anthropic/Responses on
      localhost:8080 (Hearth's openai backend could drive it IF the
      format loaded). Requires macOS 15+ (Sequoia) — check Justin's OS
      version before any install attempt. Verdict: not a rung-4
      transport; keep as WATCH for (a) a future 4-bit 35B-A3B path where
      TIE-vs-Smelt paging could actually be compared, (b) 2-bit format
      support landing. (NEW 2026-09-22)
- [ ] Track: ddalcu/mlx-serve (MLX Core.app) — NEW 2026-09-22: Mac-native
      MLX server speaking the Ollama API (/api/chat, /api/generate,
      /api/tags) alongside OpenAI/Anthropic — Hearth's existing ollama
      backend could drive it unchanged, and it needs no custom oQ2/JANG
      format (runs standard MLX quants). MLX 0.32.2, Qwen 3.8 Flash
      Next support. Evaluate as the rung-4 transport if vmlx Smelt
      quality disappoints, and as the mlx_lm.server alternative for the
      MLX tool-calling follow-up. (SWEEP 2026-09-22)
      DOC CHECK 2026-09-22 (am): docs/models.md lists Qwen 3/3.5/3.6/3.8
      incl. qwen3_5_moe (Qwen3.6-35B-A3B named) and claims a faster
      Qwen-MoE decode path (+26% raw on 35B-A3B vs LM Studio); GGUF runs
      via embedded llama.cpp — so the rung-3 IQ2_XXS GGUF serves through
      it TODAY, driven by Hearth's existing backend unchanged (concrete
      Mac-side test: tok/s vs Ollama on the same file). Two blockers for
      oQ2: (1) whether it parses oQ2's affine-2bit custom MLX layout is
      UNVERIFIED (docs say "MLX models" broadly); (2) no documented
      Smelt-equivalent expert paging, and 13.1 GB fully resident exceeds
      the ~11 GB budget regardless — so oQ2 can't fit under mlx-serve
      even if the format loads. Realistic role: faster Ollama-API
      transport for the rung-3 GGUF + the mlx_lm.server alternative for
      MLX tool-calling, NOT an oQ2 Smelt replacement. Also watch ddalcu's
      own iQ-MLX family (imatrix-calibrated mixed-width MLX, e.g.
      Qwen3.8-27B 3.8bpw 13.0 GB text-only) — a Qwen3.6-35B-A3B-iQ-MLX
      upload would be a native mlx-serve rung-4 candidate. (NEW 2026-09-22)
- [ ] Track: mlxl3 (0xZKnw/mlxl3) — NEW 2026-09-22: EXL3
      inference/conversion engine on MLX with JIT Metal kernels and
      CPU-vs-Metal conformance tests at every bit width 1-8. EXL3 is a
      new format family on the Mac side; watch for a sub-2-bit EXL3
      35B-A3B / 70B-class upload that fits the budget. (SWEEP 2026-09-22)
      DOC CHECK 2026-09-22 (am): it is a standalone Desktop app
      (v1.0.2) + engine, NOT an HTTP server — no documented
      Ollama/OpenAI API surface, so Hearth has no transport to it today
      (verify if a 35B-A3B EXL3 upload appears). Requires macOS 26.2+
      and is ad-hoc signed, NOT notarized (Gatekeeper "Open Anyway"
      friction on install). MoE support is in progress (mapped
      two-launch SwitchGLU path for selected experts). EXL3 quants
      already exist on HF at 1.40-2.00 bpw — but for Qwen3.8-27B (dense
      hybrid), not the 35B-A3B MoE; at 2.00 bpw a 27B is ~6.75 GB, which
      fits — a 35B-A3B EXL3 ~2bpw upload would be ~9 GB and inside the
      budget. Watch item stands, transport question added. (NEW 2026-09-22)
- [x] Quant R&D: JANGQ-AI/Qwen3.5-35B-A3B-JANG_2S candidate — NEW
      2026-09-21 (pm sweep): prebuilt MLX JANG 2-bit for OUR rung-3
      model. Measured **11.67 GB** via HF tree API — but the model card
      claims 65.5% MMLU at **9.0 GB** (200-question subset, author-reported;
      MLX 2-bit ~20% at 10 GB on the same table). The 9.0-vs-11.67 gap is
      unresolved (likely text-only vs vision-tower-included accounting —
      the 3.6 JANGTQ card ships 333 fp16 vision tensors). Borderline over
      the ~11 GB budget fully resident → Smelt or text-only strip.
      PROBE 2026-09-22 (am, safetensors-header probe via HTTP Range only,
      no weight download): the gap is now MEASURED, not hypothesized.
      Full file = **11.65 GB** (10.75 GB text + 0.89 GB vision, 333
      tensors) — vision stripping explains only 0.89 of the 2.65 GB gap;
      the rest is real: **9.64 GB packed U32 weights + 2.01 GB F16
      scales/metadata** (1611 tensors; JANG_2S is mixed 2/4/6-bit at
      actual 2.17 bpw per jang_config.json). The card's 9.0 GB is the
      quantizer's own `total_weight_gb: 8.98` runtime accounting in
      jang_config.json — a theoretical figure that matches neither the
      on-disk 11.65 GB nor even the 9.64 GB packed payload alone. The
      card's size column is stale/optimistic across the board (its
      JANG_4K 16.4 GB row vs 19.67 GB measured on the 3.6 upload).
      Budget verdict TIGHTENED: 10.75 GB text-only + ~1 GB runtime + KV
      > ~11 GB usable → fully resident is a NO even stripped; needs
      Smelt (if the server parses JANG_2S) or the jang_tools path with
      the vision tower dropped. Mac-side resident-RAM measurement is the
      arbiter — do not trust the card's 9.0. Loader:
      `jang_tools.loader.load_jang_model` (pip install "jang[mlx]");
      card states LM Studio / Ollama / oMLX / Inferencer do NOT support
      JANG yet — vMLX / MLX Studio are the serving paths.
      Mac-side: run our battery (not their MMLU claim), compare vs
      IQ2_XXS GGUF rung-3 and oQ2 rung-4. This is the leading
      rung-3 fallback if IQ2_XXS quality disappoints.
      PROMOTED 2026-09-22: this is now the RUNG-4 PICK (not a fallback).
      The oQ2+Smelt plan was invalidated by code check (Smelt is
      JANG-only; no oQ loader in vmlx), and JANGTQ+Smelt raises an
      explicit vmlx#81 error — JANG_2S is the only ~2-bit 35B-A3B upload
      that is both Smelt-eligible (JANG profile, confirmed in vmlx
      source) and inside the budget under Smelt (~7.8 GB @ smelt-50).
      Recipe: research/recipes/RUNG4_jang2s_35b_smelt.md; the Mac-side
      validation item above is its execution half.
- [ ] Quant R&D: JANGQ-AI/Qwen3.6-35B-A3B-JANGTQ candidate — NEW
      2026-09-21 (pm sweep): "TurboQuant" codebook 2-bit routed experts
      (Lloyd-Max + Hadamard rotation, no dequant at inference) with
      attention/embed/shared-expert/lm_head at 8-bit affine, router fp16.
      Measured **11.63 GB** (12 shards). Requires the custom
      `jang-tools` loader (stock mlx_lm can't parse `.tq_packed`);
      card's usage block references `OsaurusAI/Qwen3.6-35B-A3B-JANGTQ2`
      while the repo is `JANGQ-AI/Qwen3.6-35B-A3B-JANGTQ` — naming
      inconsistency, pin the exact id on the Mac. Also over budget fully
      resident (11.63 GB incl. fp16 vision tower dead weight for
      text-only) → Smelt. Mac-side: quality battery vs oQ2 (rung-4's
      current pick); compare the TurboQuant codebook claim ("better
      quality AND faster decode than affine 2-bit at the same budget")
      PROBE 2026-09-22 (am, safetensors-header probe via HTTP Range only,
      zero weight bytes downloaded — same method as the JANG_2S probe):
      on-disk **11.63 GB** = 10.74 GB text (1597 tensors) + 0.89 GB
      vision (333 tensors, fp16 ViT) — matches the card's stated 11.63 GB
      exactly (this card is honest about size, unlike the JANG_2S 9.0 GB
      claim). Composition: **10.48 GB U32 packed** (`.tq_packed` expert
      indices + MLX 8-bit-affine packed weights) + **1.15 GB F16**
      (`.tq_norms`, router/norms/vision passthrough) — TurboQuant's
      codebook+norm metadata is much leaner than JANG_2S's 2.01 GB F16
      scale volume. Naming RESOLVED: `OsaurusAI/Qwen3.6-35B-A3B-JANGTQ2`
      (the id in the card's usage snippet) is not publicly reachable
      (HTTP 401); the pinned public id is
      `JANGQ-AI/Qwen3.6-35B-A3B-JANGTQ`. Loader corrected from the card:
      `jang_tools.load_jangtq.load_jangtq_model` (pip install from the
      jang-tools dir of github.com/jjang-ai/jangq) — differs from the
      JANG_2S card's `jang_tools.loader.load_jang_model`. Budget verdict
      TIGHTENED by the card itself: "expect ~12–14 GB resident after
      load, plus KV cache", and its hardware table lists 24/32 GB rows
      only — NO 16 GB row. 10.74 GB text-only + ~1 GB runtime + KV >
      ~11 GB usable → fully resident is a NO even vision-stripped;
      needs Smelt (whether vmlx parses the mxtq/TurboQuant layout is
      UNVERIFIED — vmlx documents JANG profiles, not mxtq) or the
      jang_tools loader with the vision tower dropped. Quality gap: the
      card carries NO measured quant quality — its benchmark table is
      base-model fp16 references only (MMLU-Pro 85.2 etc.), "independent
      JANGTQ-quant evaluation ... will land in future README revisions";
      the "better quality AND faster decode than affine 2-bit" claim is
      author-stated, unmeasured. Our battery (vs oQ2 / IQ2_XXS) is the
      arbiter. The rung-3-fallback budget picture is now complete to the
      same granularity as JANG_2S.
- [x] Mac-side: verify whether vmlx parses the JANGTQ mxtq/TurboQuant
      layout — ANSWERED 2026-09-22 by CODE CHECK (not just the README):
      **vmlx DOES serve JANGTQ.** `vmlx_engine/loaders/load_jangtq.py`
      re-exports `jang_tools.load_jangtq.load_jangtq_model`, bundled in
      vMLX's Python runtime; the panel UI parses `weight_format: mxtq`
      (`jangQuantization.ts`). This CORRECTS the 2026-09-22 README-only
      read that said "vmlx does NOT parse the mxtq layout" — the README's
      JANG Profiles section is stale relative to the code. BUT: `--smelt`
      on JANGTQ raises an explicit error (vmlx#81 guard in
      `vmlx_engine/utils/smelt_loader.py`): JANGTQ's custom
      TurboQuantLinear modules (tq_packed + tq_norms + codebook + signs)
      cannot be subset by the Smelt patches. JANGTQ is
      **fully-resident-only**: 10.74 GB text-only + ~1 GB runtime + KV ≈
      12 GB > ~11 GB budget. The JANGTQ rung-3 fallback is therefore
      DOWNGRADED to effectively dead on 16 GB (a Mac-side resident-RAM
      measurement could still surprise, but the budget math says no).
      Also resolved from the same code read: `JANG_2S` IS a JANG profile
      (`HYBRID_JANG_PROFILES` in
      `panel/src/renderer/src/lib/jangCompat.ts`) — the README profile
      list (2M/2L/3M/4M/6M) was incomplete; JANG_2S is Smelt-eligible,
      which is what makes it the rung-4 pick below.
      against our own opcount/SQNR numbers. NOTE 2026-09-22: the vmlx
      org also ships jjang-ai/mlxstudio, a Mac app wrapping the vmlx
      server (its build clones github.com/jjang-ai/vmlx — SAME format
      support and Smelt behavior, not a separate engine). Signed +
      notarized DMG, macOS 14+ (M1 OK), no-terminal install; its Server
      mode exposes the same OpenAI-compatible API, so Hearth's `openai`
      backend can drive it. It is the GUI path for the JANG_2S rung-4
      run, but NOT a JANGTQ-Smelt path (same vmlx#81 error applies).
- [ ] Watch: Nemotron-3-Nano-Omni-30B-A3B-JANGTQ2 (JANGQ-AI, 0 downloads
      2026-09-21) — a NEW 30B-A3B MoE family with the JANGTQ2 format;
      if quality reports appear, evaluate as a 35B-A3B alternative at
      smaller size. (NEW 2026-09-21 pm sweep)
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
- [x] Mac-side: Qwen3.5-35B-A3B UD-IQ2_XXS (10.66 GB measured) — VALIDATED
      2026-09-21 ~17:42 PDT (Justin): `measure_rung.py --model qwen35-35b-a3b
      --target 10 --num-ctx 2048` (Ollama 0.34.2) → 34.46 / 34.83 / 32.15
      tok/s, mean decode **33.81 tok/s**, quality sanity PASS (arithmetic,
      factual, repeat-3x). MoE sparsity delivers: ~3x the speed of dense
      14B at ~2.6x the weight budget. NOTE: the 19-prompt battery was run
      on the canonical -full entry after the IQ2_XXS entry wedged (see
      rung-3 item); canonical rung-3 model = qwen35-35b-a3b-full. NOTE
      2026-09-21 pm: the survey's "IQ2_M 10.6 GB" figure was wrong — HF
      tree API re-measure: IQ2_M = 11.39 GB (does not fit the ~11 GB
      budget), IQ2_XXS = 10.66 GB. The rung-3 file was IQ2_XXS, not IQ2_M.
- [x] Rung-3 Mac-side validation — VALIDATED 2026-09-21 ~18:40 PDT
      (Justin): qwen35-35b-a3b-full (canonical entry after the IQ2_XXS
      entry wedged) → measure_rung 22.54 / 27.83 / 33.64, mean decode
      **28.00 tok/s**; quality sanity PASS; 19-prompt battery **15/19
      nominal (~17/19 effective** — honesty 0/2 is checker pedantry: the
      accord was correctly identified as nonexistent, the Nobel correctly
      noted as not yet awarded). Intelligence verdict vs rung 2 (14B): at
      least as smart, genuine wins (bat-and-ball correct, honesty
      calibration), but not yet decisively smarter across the board — the
      two multi-step math failures are identical on both models. RUNG 3
      STATUS: ACHIEVED (reliable, 28 tok/s, quality pass).
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
- [x] Mac-side: Jundot oQ2 (13.1 GB) under vmlx --smelt 50 — SUPERSEDED
      2026-09-22: the plan was INVALID. Code check of jjang-ai/vmlx main
      shows Smelt requires a JANG-format model (README: "Smelt requires
      an MoE model in JANG format. Not compatible with ... non-JANG
      formats") and vmlx ships NO oQ loader (loaders: jang / jangtq /
      laguna / mistral3 / zaya / qwen4_exp / dsv4). The oQ2 weights are oQ
      (oMLX affine 2-bit), not JANG — `--smelt` on them is unsupported
      and serving them under vmlx is unverified. The oQ2 weights stay on
      the watch list (if vmlx ever ships an oQ loader, or JANGQ-AI ships
      a JANG-format 3.6 at ~2-bit, the recipe gets a second candidate),
      but the rung-4 pick is now JANG_2S (below).
- [ ] Mac-side: JANGQ-AI/Qwen3.5-35B-A3B-JANG_2S under vmlx --smelt 50 —
      the rung-4 validation (recipe:
      research/recipes/RUNG4_jang2s_35b_smelt.md). Serve
      `JANGQ-AI/Qwen3.5-35B-A3B-JANG_2S` straight from the HF repo id
      (`pip install "vmlx[jang]"`; GUI alternative: MLX Studio Server
      mode). JANG_2S is a confirmed JANG profile in the vmlx source
      (HYBRID_JANG_PROFILES) so Smelt applies; Smelt disables VLM mode,
      served text-only (10.75 GB). Expected resident: smelt-50 → 6.72 GB
      weights, ~7.8 GB total (fitcheck); smelt-25 → 4.71 GB, ~5.8 GB.
      Measure: resident RAM (Activity Monitor, green pressure no swap),
      decode tok/s via tools/measure_openai.py (--base-url
      http://localhost:8000, target 10), quality sanity, then the
      19-prompt battery (tools/eval_battery_openai.py) vs rung 3's
      recorded scorecard (15/19 nominal, ~17/19 effective). Same-base-model
      comparison (both 3.5): any delta is the quant + Smelt routing bias,
      not the weights. If smelt-50 quality disappoints, re-run smelt-75
      before concluding anything about the quant (routing bias vs quant
      are two variables). (NEW 2026-09-22 — replaces the oQ2 item)
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
- [x] Quant R&D: shuffled-block eval sample — LANDED 2026-09-22 as
      `ppl.py --eval-corpus FILE --eval-blocks N [--eval-block-len L]
      [--eval-block-seed S]` (commit fecd32d; deterministic disjoint
      block sampling, labels carry the corpus offset, blocks reuse the
      multitext report machinery; 10 new tests, 429 green). Corpus:
      research/data/corpus_pg11_alice.txt (Alice in Wonderland, Project
      Gutenberg, header/footer stripped, 44,525 tokens). First
      measurement (fp32 + int2_kmeans_q8 g128 @ 2.375 bpw, 8 x 256-token
      blocks): fp32 mean 43.18 +/- 14.30 (range 21.77-60.76);
      int2_kmeans_q8 mean 676.49 +/- 288.18 (range 234.88-1210.21),
      x_fp32 = 15.67. Verdict: the 795.96 anchor is text-conditioned and
      is now quoted as 676 +/- 288; both hand-composed texts fall within
      one std. Much of the spread is inherited text difficulty (fp32
      spans 2.8x), with scheme x text interaction on top. Follow-up
      CLOSED 2026-09-22 (am session): g256-on-blocks slice ran
      (int2_kmeans_q8 @ 2.188 bpw, same 8 blocks, seed 7): mean
      778.90 +/- 290.38 (range 225.83-1308.73) vs g128's 676.49 +/-
      288.18 — g128 still wins on the block distribution (x_fp32 15.67
      vs 18.04). Honest caveat: the 102.4 mean gap sits inside the
      standard error of the difference (~145), so blocks alone are not
      significant — but this is now the THIRD independent datum with
      the same sign (text1: 795.96 vs 1085.61; text2: 1047.35 vs
      1971.77), all in collapse territory, all directional. The
      g128-beats-g256 verdict is as tightened as it will get; no further
      slices planned.
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
- [x] Quant R&D: fidelity-ladder retrospective — CLOSED NEGATIVE
      2026-09-21. The structurally-different idea (sensitivity-adaptive
      bit allocation across layers at matched average bpw) got its one
      experiment via the landed per-layer harness, and it LOSES.
      Matched-bitrate result (eval_text1, g128): the single most
      Fisher-sensitive block (block 11, trace 767860 — late blocks
      dominate, blocks 0/1 least sensitive) at int8_uniform, the other
      11 at ternary_1step_ds -> **2008.36 @ 2.359 bpw vs the naive
      int2_kmeans_q8 anchor 795.96 @ 2.375** — 2.5x worse at matched
      bitrate (0.7% fewer bits, so the loss is conservative). Second
      datapoint: top-5 sensitive blocks at int8 + rest ternary_1step ->
      4070.67 @ 4.383 bpw, WORSE than uniform ternary_1step (2764 @
      1.710) at 2.56x the bits. Honest mechanism: the Fisher trace is
      measured on fp32 activations, but once the unprotected blocks go
      ternary the activation distribution shifts and the fp32 ranking
      no longer describes where the error hurts — uniform Lloyd at
      2.375 bpw is simply a better use of bits than ternary-anywhere +
      int8 islands. Caveats: calibrated on the eval text (no held-out
      corpus); GPT-2 124M scale; collapse territory so ordering is
      directional per the methodology item — but both gaps are 2.5x+,
      far outside noise. No further fidelity iteration at this bitrate
      is planned; the follow-ups (per-tensor granularity, activation
      drift measurement) are logged above as low-priority mechanism
      work. The quant-R&D story rests on the compute side and on
      bigger models / Mac-side validation, as before.
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
- [x] Rung-1 Mac-side validation — VALIDATED 2026-09-21 (Justin, his M1
      Pro): qwen3:8b at 20 tok/s. Rung 1 DONE per standing direction. Recipe
      at `research/recipes/RUNG1_qwen3_8b_20tps.md`, harness
      `tools/measure_rung.py` (282 tests green incl. 23 new).
- [x] Rung-2 recipe: 14B-class at 20 tok/s — LANDED 2026-09-21 as
      `research/recipes/RUNG2_qwen3_14b_20tps.md` (roofline 200/9.0 ≈
      22 tok/s — TIGHT, expect tuning), `tests/test_rung_recipes.py` pins
      the recipe's RAM/roofline figures against fitcheck + checks every
      recipe referenced in model_profiles.yaml exists (291 tests green).
      qwen3:14b profile est_gb corrected 8.6 -> 9.0 (14.7B params x
      4.9 bpw). measure_rung.py needed no new knobs (--num-ctx/--target
      already exist).
- [x] Rung-2 Mac-side validation — ACHIEVED 2026-09-21 (Justin):
      qwen3:14b Q4_K_M measured 13.0 tok/s on his Mac, 100% GPU, quality
      sanity all pass — usable speed, speed tuning deferred per the
      standing direction. Rung 2 DONE.
- [x] Rung-2 fallback candidates — SUPERSEDED 2026-09-21: rung 2 is
      DONE at usable speed (13.0 tok/s) per the standing direction, which
      replaced the assistant's 20 tok/s construct with "~10+ tok/s usable"
      — the conditional (miss after tuning ladder) never fired and the
      fallbacks are moot. If rung-2 quality (not speed) ever becomes the
      question, the ladder moves up to rung 3, not sideways.
- [x] Quant R&D: vectorize the ternary Lloyd-fit encoder — LANDED
      2026-09-22 as `_ternary_lloyd_fit_batch` in schemes.py (whole-array
      numpy ops over all groups, per-group early exit preserved). Measured
      on 2M params: ternary_1step 81 ms (**24.7 vs 2.8 Mparams/s**, 8.8x),
      ternary_lloyd n_iter=20 600 ms (3.3 vs 0.8 Mparams/s, 4.4x),
      1step_ds 15.9 Mparams/s (5.4x), lloyd_ds 3.5 Mparams/s (3.1x).
      Bit-identical numerics vs the old per-group float64 path, pinned by
      test against a FROZEN copy of the old loop (dual T/F, n_iter 1/2/20,
      thresh_factor 1.0/1.2, group sizes 64/128/256, incl. all-zero,
      one-sided, and ragged/padded edge tensors) — every published anchor
      number (SQNR tables, ppl 2764/795.96, opcount zero-rates) reproduces
      exactly; diagnose.py's +1.47 dB decomposition re-verified. The old
      `_ternary_lloyd_fit` is now a thin 1-group wrapper (same signature,
      same history semantics for diagnose.py). 419 tests green.

- [ ] Quant R&D: vectorize `_lloyd_1d` (the int2_kmeans/int2_kmeans_q8
      per-group loop) the same way the ternary encoder was vectorized
      2026-09-22 — it is now the remaining pure-Python per-group encoder
      bottleneck. Harder to keep bit-identical (argmin of weighted
      distances per group); the frozen-oracle pin pattern from the
      ternary work applies. Only worth it if a future experiment needs
      many k-means re-fits (the fidelity thread that needed them is
      closed, so this is opportunistic). (NEW 2026-09-22)
- [ ] Mac-side experiment: local JANG_2L conversion — vmlx documents
      JANG conversion with `--calibration-method activations` ("better at
      2-3 bit", from the 2026-09-21 pm sweep) and MLX Studio ships
      conversion tooling. Converting the bf16 35B-A3B source (~70 GB
      download on the Mac) to JANG_2L locally would produce a native
      2-bit JANG with full Smelt support for EITHER the 3.5 or the 3.6
      variant — a second rung-4 candidate independent of JANGQ-AI's
      upload cadence, and the way to get a 3.6 JANG_2L (the oQ2
      equivalent that Smelt can actually page). Quality unknown until the
      battery runs; compare vs JANG_2S smelt-50. (NEW 2026-09-22)
- [ ] Track: JANGQ-AI JANG_2L/JANG_2M upload for Qwen3.6-35B-A3B — a
      prebuilt native-JANG ~2-bit 3.6 (the oQ2 weights in a
      Smelt-compatible format) would slot straight into the rung-4
      recipe as a second candidate. Their 3.6 uploads so far are JANGTQ
      (mxtq, no Smelt) and JANG_4K/JANGTQ4 (4-bit-ish, too big).
      (NEW 2026-09-22)
- [ ] Mac-side: confirm the JANG_2S serve path end to end — the JANG_2S
      profile is confirmed in the vmlx source (HYBRID_JANG_PROFILES) and
      Smelt's ExpertIndex scans the Qwen 3.5 `switch_mlp` naming, but no
      one has actually run `vmlx serve JANGQ-AI/Qwen3.5-35B-A3B-JANG_2S
      --smelt` yet. Fold into the rung-4 validation item: if serve
      fails, the fallback is the plain (non-Smelt) vmlx serve at 10.75
      GB text-only — borderline over budget, measure resident RAM.
      (NEW 2026-09-22)

## Ground rules for this research track

- Every change is a git commit in `~/workspace/local-agent`. Revert with
  `git log --oneline` + `git revert <hash>`.
- Never break the working default (`qwen3:8b` via Ollama). Experimental
  profiles stay opt-in.
- Don't download multi-GB models on the dev VM; use file-size metadata.
- Each research sweep appends a dated entry under `research/log/`.
