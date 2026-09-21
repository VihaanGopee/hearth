# Low-active-parameter MoE survey — 2026-09-21

**Question:** what is the best-evidenced "70B-class quality on a 16 GB Mac"
option available TODAY, for the end-of-week usable-solution push?
**Budget:** ~10–11 GB usable for weights (+ KV cache), macOS takes the rest.
Qwen3.5-35B-A3B is 34.65 B params (FP16 = 69.3 GB), so GB ≈ 34.65 × bpw / 8.

Architecture note (Qwen3.5-35B-A3B): 35 B total / **3.3 B active per token**
(256 routed experts, 8 active + 1 shared expert), 40 layers, 262 k context.
On bandwidth-bound Apple Silicon, *active* params set decode speed, so a
35 B MoE decodes roughly like a 3 B dense model — this is why the MoE path
is the only one with both fit AND speed evidence. FP16 reference quality:
MMMLU 85.2, MMLU-ProX 81.0.

## Candidates that fit (or nearly fit) the budget

| Option | Size | Est. bpw | Fit verdict | Quality evidence |
|---|---|---|---|---|
| `unsloth/Qwen3.5-35B-A3B-GGUF` / `Qwen3.5-35B-A3B-UD-IQ2_M.gguf` | **10.6 GB** | ~2.45 | ✅ fits in RAM | quality at IQ2_M **not benchmarked** (gap — see follow-ups). Community claim: 30 tok/s on Mac mini M4 (unverified, not M1 Pro) |
| `mtrpires/Qwen3.5-35B-A3B-IQK-RPi5-16GB` (mixed IQK recipe) | 11.38 GB disk / ~12.5 GB RAM | ~2.63 | ⚠️ borderline — over the 10–11 GB usable budget; needs q4_0 KV + lean ctx | principled recipe: attention + shared expert at high precision, routed experts aggressive. No independent bench numbers found |
| `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` / `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` | 11.819 GB | ~2.73 | ⚠️ borderline | ships a native NextN/MTP speculative head — pairs with the speculative-decoding backend landed 2026-09-21 |
| oQ2 MLX (`oMLX`, self-quantize on the Mac) | ~12.6 GB | ~2.9 | ❌ over in-RAM; flash-paging only | **best quality data at ~2-bit**: MMLU 64.0% vs naive mlx-lm 2-bit 14.0%; TruthfulQA 80.0%; HumanEval 78.0%; MBPP 63.3% (oMLX docs, 300 samples) |
| oQ3 MLX | ~15.2 GB | ~3.5 | ❌ over | MMLU 85.0% (≈ FP16 85.2 MMMLU), HumanEval 86.6% — near-lossless, but needs 48 GB-class machine in RAM |
| `Jundot/Qwen3.6-35B-A3B-oQ4` (prebuilt oQ download exists) | ~20 GB | ~4.6 | ❌ over | oQ4 is the "recommended" level; proves Jundot publishes prebuilt oQ models — an oQ2 35B-A3B upload would land ~12.6 GB (still slightly over) |

Also checked and excluded: `Qwen3-30B-A3B` 4-bit MLX (~17 GB — over, "Expert
Sniper" paging only 4.3 tok/s); EOQ Q5 (35.2 GB, PPL 5.39 vs 5.19 — too big);
MINT mixed-precision (37 GB min — too big); `Qwen3.5-35B-A3B-Q8_0.gguf`
(36.9 GB — too big); Q4_K_M 20–22 GB variants (flash-streaming only, ~1.5
tok/s — not "usable").

## Honest verdict

1. **The only in-budget, in-RAM option with a file you can download today is
   the unsloth IQ2_M (10.6 GB).** Its quality at that bitrate is unverified —
   that is now the single most important unknown for the week.
2. **The only ~2-bit option with real benchmarks is oQ2 (MMLU 64%)**, and it
   is ~1.5–2 GB over the usable budget — close enough that oMLX's SSD KV
   cache / flash-paging might make it workable, but it is not the clean
   in-RAM story. oQ's sensitivity-driven mixed-precision (lm_head + router +
   shared-expert-gate protected at 8-bit, routed experts at base bits) is the
   MLX-side technique to watch; Jundot already publishes prebuilt oQ models.
3. **Qwen3.6-35B-A3B-MTP is the speculative-decoding-native pick** — the MTP
   head composes with the prompt_lookup/draft_model support landed in
   Hearth's llamacpp backend this morning.
4. No option delivers 70B-dense quality: oQ2 at 64% MMLU vs FP16 ~85% is a
   ~20-point gap. "70B-class" for the agent use case means "better than any
   dense model that fits 16 GB" — plausibly true for IQ2_M/oQ2 vs an 8 B
   dense at Q4, but unmeasured. The week's Mac validation must measure this,
   not assume it.

## Recommended Mac validation order (needs Justin's Mac)

1. Download `unsloth/Qwen3.5-35B-A3B-GGUF` IQ2_M (10.6 GB); run via Hearth's
   `llamacpp` backend or Ollama Modelfile; measure decode tok/s on M1 Pro
   and spot-check quality (MMLU subset or a fixed agentic task battery).
2. If IQ2_M quality disappoints: try the mtrpires mixed-IQK file (11.38 GB)
   with `--cache-type-k q4_0 --cache-type-v q4_0`, short context.
3. In parallel: install oMLX on the Mac and self-quantize oQ2 (streaming
   path is designed for 16 GB Macs) — compares benchmarked-quality ~2-bit
   against the in-RAM IQ2_M.

## Follow-ups (added to roadmap backlog)

- [ ] Mac-side: IQ2_M 10.6 GB — measure tok/s + quality (MMLU subset) on the
      M1 Pro; compare vs the oQ2 64% MMLU datapoint
- [ ] Mac-side: mtrpires mixed-IQK (11.38 GB) — does it actually stay under
      memory pressure with q4_0 KV cache at 8–12 k ctx?
- [ ] Hearth: documented `llamacpp` recipe/profile for the 35B-A3B IQ2_M
      (opt-in config snippet), wiring the speculative prompt_lookup mode
      with the Qwen3.6 MTP variant
- [ ] Re-check HF for a prebuilt oQ2/oQ2.5 35B-A3B MLX upload (Jundot org);
      if one appears at ~12.6 GB, re-evaluate vs IQ2_M on quality-per-GB
- [ ] JANG release tracking stays open (no new 2-bit MoE quality reports
      found 2026-09-21)
