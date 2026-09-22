# Ternary add/sub kernel: design study for a fitted-ternary ggml type

**Status:** design study, 2026-09-22. Not a PR, not a kernel — the
quantitative and layout groundwork for one, if the fidelity evidence
ever justifies it.

## Why this document exists

The quant-R&D fidelity ladder closed negative on 2026-09-21: pure
group-wise quantization at ~2.2 bpw cannot reach usable perplexity at
small scale (naive -> Lloyd -> Fisher-reweight -> full-model OBQ, best
475.48 vs 53.50 fp32 — still collapse territory). The remaining
ternary case is **inference cost**, quantified in
`src/quant_rnd/opcount.py`:

- Fitted ternary (`ternary_1step`, 1-step Lloyd, symmetric, g128):
  measured zero-rate **0.41** on real GPT-2 124M weights (vs 0.31 for
  absmean-uniform ternary), **0.617 equiv-adds/weight**, **3.52x**
  energy-proxy advantage over the Lloyd codebook reference
  (`int2_kmeans_q8` @ 2.375 bpw, 2.17 equiv-adds/weight).
- Prefill (compute-bound): ceiling ratio **1.795x** ternary over
  codebook; compresses below 1.2x past 64k context (attention O(L^2)
  swamps the matmul term; `prefill_crossover_L` pins 65536).

This document turns those numbers into a concrete kernel design: the
exact data layout, the decode loop, the SIMD sketch, and what an
upstream contribution would actually contain. It also records one
honest correction to the opcount model found while writing it (below).

## Reference scheme: `ternary_1step`

- Symmetric ternary codes in {-1, 0, 1}, group size 128.
- Encoder: heuristic absmean assignment -> one L2-optimal scale refit
  -> reassignment at the refit thresholds (captures ~85% of the full
  Lloyd-ternary SQNR win: +1.37 dB over absmean-uniform at exactly
  matched bitrate, on synthetic AND real GPT-2 weights).
- One fp16 scale per group. Dual-scale twin (`ternary_1step_ds`,
  1.835 bpw) exists; the symmetric variant is the kernel reference.

## Layout spec (per group of 128 weights)

A real kernel cannot store 1.585 bits/weight — that figure is the
*entropy* of a ternary alphabet, achievable only with arithmetic
coding, which no matmul kernel uses. Packable storage is 2 bits/weight:

| Field | Bytes/group | Notes |
|---|---|---|
| codes | 32 | 2 bits/weight, 4 weights/byte; code mapping 0=+1, 1=-1, 2/3=zero |
| scale | 2 | fp16, one per group |

**34 bytes / 128 weights = 2.125 storage bpw** (not 1.710).

### The packing correction (new, honest)

`src/quant_rnd/opcount.py`'s decode roofline used the entropy bpw
(1.710) for ternary schemes but the true storage bpw (2.375) for the
k-means reference — an apples-to-oranges byte count. Corrected at 70B
scale, 200 GB/s, 1.34 GB KV:

| Scheme | Storage GB | Decode ceiling | vs k-means |
|---|---|---|---|
| int2_kmeans_q8 (2.375 bpw) | 20.78 | 9.0 t/s | — |
| ternary_1step, entropy (1.710) | 14.96 | 12.3 t/s | 1.36x |
| ternary_1step, **packed (2.125)** | **18.59** | **10.0 t/s** | **1.11x** |

The decode advantage shrinks from 1.36x to **~1.11x** once bytes are
counted honestly. The energy-proxy (3.52x) and prefill compute-bound
(1.795x) ratios are op-driven and survive — they never depended on the
byte count. Budget consequence: 18.59 GB is even further over the
10–11 GB usable budget than the entropy figure suggested. Dense-70B
ternary stays closed (avenue C); this strengthens, not weakens, that
verdict. Follow-up filed: add a `storage_bpw` notion to `opcount.py`'s
`scheme_report` and re-pin the decode figures (backlog).

## Kernel loop design (decode: y += Wx)

Per group, with fp32 accumulator (one scale multiply per group):

```
for each group g of 128 weights:
    acc = 0.0f
    for i in 0..127:
        c = (codes[i >> 2] >> (2 * (i & 3))) & 3
        if (c > 1) continue          // zero code: skip the add
        acc += (c == 0) ? x[i] : -x[i]
    y[row] += scale[g] * acc          // single fma per group
```

Op profile per weight (matches `ternary_opcount`): (1 − 0.41) = 0.59
signed adds, 1/128 multiplies, 0 lookups. The 25% unused code space
(2 of 4 states both decode as zero) is the price of packable ternary —
exactly the entropy gap.

Dual-scale variant: two accumulators; `y += s_pos*acc_pos −
s_neg*acc_neg`, matching the prototype's decode (code +1 -> +s_pos,
code −1 -> −s_neg).

SIMD sketch (not implemented): unpack 8 bytes of codes (16 weights)
to int8 lanes per 128-bit register; the signed-add becomes a blend of
`+x`/`−x`/`0` selected by lane value — the same shape as bitnet.cpp's
TL1/i2_s kernels and llama.cpp's TQ kernels. ARM NEON (the M1 target)
does this with `vtbl`-style shuffles; AVX2 with `vpshufb`. Zero-skip
at SIMD width needs a popcount/mask test per 16-weight chunk; whether
branchy skip beats branchless masked-add at 41% sparsity is a
measurement, not a design decision — both shapes are one `#ifdef`
apart. Prefill (compute-bound) wants the branchless form; decode
(bandwidth-bound) is indifferent to ALU shape, only to bytes moved.

## Relationship to existing art

- **llama.cpp TQ1_0/TQ2_0** (in `ggml.h` at b10969, verified
  2026-09-21): naive absmean ternary. Our fitted variant is an
  *encoder* improvement on the same idea (+1.37 dB measured), not a new
  arithmetic. An upstream contribution would most plausibly be a
  fitted encoder for the TQ1_0 packing — reusing their layout and
  kernel — rather than a new `GGML_TYPE_*`.
- **bitnet.cpp**: ships trained-ternary kernels (i2_s AVX2 path seen in
  their issue tracker); different problem (trained ternary, not
  post-training quant of fp weights) but the add/sub MAC shape is the
  reference implementation to read before writing anything.
- **Metal path**: ggml's Metal backend would need the kernel ported;
  whether ternary matmuls take an optimized path on Apple Silicon is
  unverified (existing backlog item) — CPU-first is the honest order.

## What a PR would contain, and the blockers

1. Encoder: 1-step Lloyd fit emitting the TQ1_0 packing (drop-in
   quality uplift for an existing type).
2. Kernel: add/sub MAC with zero-skip, x86 AVX2 + ARM NEON, behind the
   existing ternary type.
3. Write-up: the SQNR evidence (synthetic + real-weight, both done)
   and the op-profile numbers (done).

Blockers, stated plainly:
- **Fidelity evidence at scale.** Our perplexity runs are
  collapse-territory at GPT-2 124M (2764 vs 53.50 fp32). A PR needs at
  least 7B-scale perplexity showing the +1.37 dB SQNR win transfers to
  real quality. Small-scale evidence does not qualify.
- **Mac-side validation of the opcount ratios** (roofline 1.11x
  decode / 1.795x prefill ceilings need measured tok/s; backlog).
- Upstream discussion first: TQ1_0 already covers naive ternary, so
  the pitch is encoder-only — a smaller, more mergeable change than a
  new type, but it needs a maintainer's read on whether the fitted
  encoder is worth the complexity.

## Bottom line

The design is small and fully specified above; the arithmetic is
settled (add/sub + one fma per group, 2.125 storage bpw, ~1.11x honest
decode ceiling over the codebook reference). What is *not* settled is
whether fitted ternary deserves to exist at all — that is a fidelity
question at 7B+ scale, and this VM cannot answer it. The document
stands as the compute thread's deliverable: if a future 70B-scale
measurement revives the ternary case, the kernel work starts here,
not from scratch.
