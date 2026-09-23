# Rung-4 candidate recipe: Qwen3.6-35B-A3B EXL3 2.08bpw under mlxl3 (M1 Pro)

**Goal:** the second rung-4 candidate (the intelligence-first rung): the
best ~2-bpw 35B-A3B fidelity datapoint on record — ppl +7.6% vs bf16 —
fully resident on a 16 GB Mac via mlxl3's EXL3 engine. Same pass bar as
rung 4: **usable speed** (≥ 10 tok/s) **and** all quality sanity checks,
with the intelligence verdict being the side-by-side battery scorecard
vs rung 3's IQ2_XXS GGUF.

**Status:** the weights exist; the engine exists; the *transport* is the
gap. mlxl3 ships no HTTP/OpenAI server surface, so Hearth drives it
over the one-shot CLI bridge (`src/mlxl3_cli.py`, UNTESTED-ON-MAC), and
the 19-prompt battery has a `--transport mlxl3` mode ready for the
Mac-side run. This recipe is the Mac-side validation procedure for that
path. The primary rung-4 candidate remains JANG_2S under vmlx Smelt
(`RUNG4_jang2s_35b_smelt.md`) — same-base 3.5, real HTTP transport,
measurable today. If both candidates measure, compare them.

## The candidate

`yeasah/Qwen3.6-35B-A3B-exl3` — the **2.08bpw revision** (published
2026-09-21; per-bitrate weights live as git revisions, invisible on
main).

- qbench (openwebtext10k, 10 × 2048-token rows, author-reported):
  **2.00bpw-H5 ppl 10.865 / KLD 0.1136 vs bf16 10.097 (+7.6%)** at 2.13
  bpw_layer; 3.00bpw-H5 ppl 10.205 (+1.1%) / KLD 0.0302; 6.00bpw is
  ppl-identical to bf16. This is the best ~2-bpw 35B-A3B fidelity
  datapoint on record (vs JANG_2S's author-claimed 65.5% MMLU and oQ2's
  measured 64% MMLU on the 3.5 variant — different metrics, so our
  battery is the arbiter, not the card).
- Sibling `yeasah/Ornith-1.5-35B-A3B-exl3` confirms the curve on a second
  35B-A3B MoE: 2.00bpw-H5 ppl 12.336 vs bf16 11.431 (+7.9%), KLD 0.1775,
  9.6 GB VRAM.
- Size: README claims 10.83 GiB disk / **9.88 GiB VRAM** (exllamav3, with
  embeddings CPU-offloaded); HF tree API measures **11.72 GB across 2
  shards** on disk.

**Caveats (read before running):**

- The figures are **author-reported, not independently measured**.
- **Not same-base:** this is the 3.6 variant, rung 3 is 3.5. A battery
  delta confounds the base-model change with the quant. A Qwen3.5-35B-A3B
  2bpw EXL3 upload would enable the apples-to-apples comparison vs
  IQ2_XXS — watch for it.
- EXL3 does not quantize embeddings (CPU-offload on exllamav3 — perf
  cost) and the vision encoder is unquantized; serve text-only.

## Expected RAM (fitcheck arch `qwen3.6-35b-a3b`)

- Resident weights: **9.88 GB** (author-reported, exllamav3 accounting
  with embeddings CPU-offloaded).
- KV cache f16 @ 2048 ctx: **0.04 GB** (fitcheck arch
  `qwen3.6-35b-a3b`: 40 layers, GQA-2 KV heads, head_dim 256 — only the
  10 full-attention layers carry KV, same geometry as the JANG_2S
  recipe's 3.5 arch).
- Total: 9.88 + 0.04 + ~1.0 runtime ≈ **10.9 GB** — inside the ~11 GB
  usable budget, but **tight**.
- **Honest caveat:** the 9.88 figure is a CUDA engine's accounting.
  mlxl3's resident number is UNMEASURED — step 6 of the Mac-side
  procedure is the arbiter. If mlxl3's resident lands over budget, this
  candidate drops back to watch.
- Roofline floor: 200 GB/s ÷ 9.88 GB ≈ **20.2 tok/s** (bandwidth floor,
  not a prediction).

## The engine: mlxl3 v1.1.1

mlxl3 (0xZKnw/mlxl3) is the only Apple-Silicon EXL3 engine:

- **Qwen3.6-35B-A3B EXL3 2.49bpw is TESTED on Apple Silicon**
  (docs/v1-validation.md: text generation + multi-turn + Qwen
  structured-parser fixtures). README headline: **47.545 tok/s median
  decode** (greedy, 48 tokens, M5), 0.153 s prefill; experimental
  DFlash2 speculative path (draft files from
  incoai/Qwen3.6-35B-A3B-Splash, ~457 MiB) hits 74.645 tok/s — opt-in,
  greedy-only.
- **M1 Pro perf is UNMEASURED.** mlxl3's dev + perf campaign is M5-only;
  M1–M4 "use compatible Metal paths where M5 TensorOps are unavailable".
  Do not assume 47.5 on the M1 Pro.
- MoE kernels are a "working correctness-first path" (kernel-port.md) —
  quality before speed, which fits the intelligence-first rung.
- Release build requires **macOS 26.2+**; ad-hoc signed, **NOT notarized**
  (Gatekeeper "Open Anyway" friction on install).
- HTTP/OpenAI server surface: **none** (README re-checked 2026-09-22
  v1.1.1). Desktop chat app + streaming `mlxl3` CLI only: `mlxl3 run`
  interactive chat, one-shot `--prompt ... --max-tokens N`, `mlxl3 hub
  download` pulls EXL3 repos.

## Transport: the CLI bridge (the gap)

Hearth's path to mlxl3 is `src/mlxl3_cli.py` (`Mlxl3CliClient`:
chat(messages, tools) → one-shot `mlxl3 run <model> --prompt ...
--max-tokens N` via subprocess; greedy-only; chat() with non-empty
tools raises LLMError rather than silently dropping schemas).

- **UNTESTED-ON-MAC.** The real one-shot stdout shape is the open
  question — see Mac-side step 3. The bridge is deliberately **not**
  wired into agent.py yet: wiring waits on the Mac-side stdout-parse
  validation (don't wire a bridge whose real stdout shape is still a
  guess).
- **Speed is not measurable on this transport:** `tools/measure_openai.py`
  deliberately REFUSES `--transport mlxl3` (the one-shot CLI exposes no
  token counts and the per-invocation model-load cost is unmeasured —
  any tok/s figure would be meaningless). Speed stays mlxl3's published
  numbers plus whatever the Mac measures until a server transport
  exists.

## Mac-side validation steps

1. Confirm **macOS 26.2+**. Install mlxl3 (accept the Gatekeeper "Open
   Anyway" step — ad-hoc signed, not notarized).
2. Download the 2.08bpw revision:
   `mlxl3 hub download yeasah/Qwen3.6-35B-A3B-exl3` (select the 2.08bpw
   revision; ~11.72 GB disk).
3. **Validate the one-shot stdout shape** per the checklist in
   `src/mlxl3_cli.py`'s module docstring: real stdout (echo/banners),
   per-invocation model-load cost (batch tool vs interactive backend),
   whether `--prompt` gets the chat template applied (transcript vs raw
   last user message), DFlash2 CLI exposure. If the parse is wrong, this
   candidate stays blocked — report the real stdout shape back.
4. Run the intelligence battery (the rung verdict that matters per the
   intelligence-first direction):
   `python3 tools/eval_battery_openai.py --transport mlxl3
   --mlxl3-model <name-as-mlxl3-expects> --models qwen36-35b-a3b-exl3`
   (single-model run; no server needed; --max-tokens 512 per item;
   `<think>` stripped as usual).
5. Compare against rung-3's recorded scorecard (**15/19 nominal, ~17/19
   effective**) — with the 3.6-vs-3.5 caveat above. The verdict is the
   side-by-side score, not a pass/fail line.
6. Measure **resident RAM** (Activity Monitor: green pressure, no swap)
   at idle and during generation, and decode tok/s as observed. Note
   whether the embeddings CPU-offload behaves like exllamav3's 9.88 GiB
   accounting suggests.

**Pass criteria:** the battery completes; resident < ~11 GB with no swap
pressure; decode ≥ 10 tok/s on the M1 Pro (mlxl3's 47.545 is M5-only).
The intelligence verdict is then the side-by-side battery score vs rung
3 — the question is whether 2.08bpw EXL3's +7.6% ppl is real on our 19
prompts and whether it beats IQ2_XXS enough to displace the JANG_2S
primary.
