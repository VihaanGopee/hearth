"""Pins the RAM/roofline figures quoted in the rung validation recipes against
fitcheck, so the docs can't silently drift from the math.

If a recipe's numbers change (different params_b, quant, arch), update BOTH
the recipe markdown AND this test — the test fails until they agree.
"""
import math
import os
import unittest

import yaml

from src.tools.fitcheck import estimate, moe_expert_gb, smelt_resident

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECIPES = os.path.join(REPO, "research", "recipes")


class TestRung1RecipeNumbers(unittest.TestCase):
    """research/recipes/RUNG1_qwen3_8b_20tps.md."""

    def test_weights_kv_totals(self):
        r2048 = estimate(8.03, "Q4_K_M", 2048, arch="qwen3-8b", cache_type="f16")
        self.assertTrue(r2048["ok"], r2048.get("error"))
        # Recipe: ~4.9-5.2 GB weights, 0.30 GB KV @ 2048, ~5.6-6.0 GB total.
        self.assertAlmostEqual(r2048["weights_gb"], 4.9, delta=0.3)
        self.assertAlmostEqual(r2048["kv_cache_gb"], 0.30, delta=0.05)
        self.assertAlmostEqual(r2048["total_gb"], 5.8, delta=0.4)
        self.assertTrue(r2048["fits"])

    def test_roofline(self):
        # Recipe: ~40 tok/s ceiling at 200 GB/s on ~5 GB weights.
        self.assertAlmostEqual(200.0 / 4.9, 40.8, delta=2.0)

    def test_recipe_file_exists(self):
        self.assertTrue(
            os.path.isfile(os.path.join(RECIPES, "RUNG1_qwen3_8b_20tps.md")))


class TestRung2RecipeNumbers(unittest.TestCase):
    """research/recipes/RUNG2_qwen3_14b_20tps.md."""

    def test_weights_kv_totals_2048(self):
        r = estimate(14.7, "Q4_K_M", 2048, arch="qwen3-14b", cache_type="f16")
        self.assertTrue(r["ok"], r.get("error"))
        # Recipe: 9.0 GB weights, 0.34 GB KV @ 2048, ~9.8 GB total.
        self.assertAlmostEqual(r["weights_gb"], 9.0, delta=0.15)
        self.assertAlmostEqual(r["kv_cache_gb"], 0.34, delta=0.05)
        self.assertAlmostEqual(r["total_gb"], 9.8, delta=0.25)
        self.assertTrue(r["fits"])

    def test_weights_kv_totals_4096(self):
        r = estimate(14.7, "Q4_K_M", 4096, arch="qwen3-14b", cache_type="f16")
        self.assertTrue(r["ok"], r.get("error"))
        # Recipe: 0.67 GB KV @ 4096, ~10.2 GB total.
        self.assertAlmostEqual(r["kv_cache_gb"], 0.67, delta=0.05)
        self.assertAlmostEqual(r["total_gb"], 10.2, delta=0.25)
        self.assertTrue(r["fits"])

    def test_kv_q8_0_savings(self):
        # Recipe claims q8_0 halves KV traffic roughly.
        f16 = estimate(14.7, "Q4_K_M", 2048, arch="qwen3-14b", cache_type="f16")
        q8 = estimate(14.7, "Q4_K_M", 2048, arch="qwen3-14b", cache_type="q8_0")
        self.assertTrue(q8["ok"] and f16["ok"])
        self.assertAlmostEqual(q8["kv_cache_gb"], 0.19, delta=0.05)
        self.assertLess(q8["kv_cache_gb"], 0.6 * f16["kv_cache_gb"])

    def test_roofline_is_tight(self):
        # Recipe: ceiling ~22 tok/s at 200 GB/s -- tight vs the 20 bar.
        ceiling = 200.0 / 9.0
        self.assertAlmostEqual(ceiling, 22.2, delta=0.6)
        self.assertLess(ceiling, 25.0)   # tight: no headroom claim
        self.assertGreater(ceiling, 20.0)

    def test_recipe_file_exists(self):
        self.assertTrue(
            os.path.isfile(os.path.join(RECIPES, "RUNG2_qwen3_14b_20tps.md")))


class TestRung3RecipeNumbers(unittest.TestCase):
    """research/recipes/RUNG3_qwen3_5_35b_a3b_moe.md."""

    def _estimate(self, n_ctx, cache_type):
        r = estimate(34.65, "iq2_m", n_ctx, arch="qwen3.5-35b-a3b",
                     cache_type=cache_type)
        self.assertTrue(r["ok"], r.get("error"))
        return r

    def test_arch_entry_kv_2048_f16(self):
        # GQA-2 KV heads, head_dim 256, but only the 10 full-attention
        # layers carry KV cache (the 30 Gated DeltaNet layers carry small
        # recurrent state instead): KV = 2*10*2*256*2048*2 B = 0.042 GB.
        r = self._estimate(2048, "f16")
        self.assertAlmostEqual(r["kv_cache_gb"], 0.04, delta=0.03)

    def test_arch_entry_kv_2048_q8_0(self):
        r = self._estimate(2048, "q8_0")
        self.assertAlmostEqual(r["kv_cache_gb"], 0.02, delta=0.03)

    def test_total_with_measured_file_size(self):
        # Recipe uses the MEASURED file (10.66 GB, HF tree API 2026-09-21),
        # not fitcheck's nominal iq2_m bpw (2.7 -> 11.69 GB). Total @ 2048
        # ctx + q8_0 KV: 10.66 + 0.02 + 0.5 runtime = ~11.2 GB.
        r = self._estimate(2048, "q8_0")
        total = 10.66 + r["kv_cache_gb"] + r["runtime_gb"]
        self.assertAlmostEqual(total, 11.18, delta=0.15)

    def test_total_2048_f16(self):
        r = self._estimate(2048, "f16")
        total = 10.66 + r["kv_cache_gb"] + r["runtime_gb"]
        self.assertAlmostEqual(total, 11.20, delta=0.15)

    def test_roofline_floor_below_20(self):
        # Honest bound: even streaming the whole 10.66 GB file per token,
        # the ceiling (18.8 tok/s) sits BELOW the old 20 bar -- the rung is
        # decided by measurement, and the pass bar is 10 (usable, per the
        # intelligence-first direction: speed is secondary).
        floor = 200.0 / 10.66
        self.assertAlmostEqual(floor, 18.8, delta=0.3)
        self.assertLess(floor, 20.0)

    def test_recipe_file_exists(self):
        self.assertTrue(
            os.path.isfile(os.path.join(RECIPES, "RUNG3_qwen3_5_35b_a3b_moe.md")))


class TestRung4RecipeNumbers(unittest.TestCase):
    """research/recipes/RUNG4_jang2s_35b_smelt.md."""

    def _estimate(self, n_ctx, cache_type):
        r = estimate(35.0, "q4_k_m", n_ctx, arch="qwen3.5-35b-a3b",
                     cache_type=cache_type)
        self.assertTrue(r["ok"], r.get("error"))
        return r

    def test_arch_entry_kv_2048_f16(self):
        # 3.5 skeleton: GQA-2 KV heads, head_dim 256, only the 10
        # full-attention layers carry KV (full_attention_interval=4;
        # same geometry as the 3.6 measured 2026-09-21).
        r = self._estimate(2048, "f16")
        self.assertAlmostEqual(r["kv_cache_gb"], 0.04, delta=0.03)

    def test_routed_expert_bytes(self):
        # 40 layers x 256 experts x 3 projs x 2048 x 512 @ 2-bit.
        self.assertAlmostEqual(
            moe_expert_gb(40, 256, 2048, 512, 2.0), 8.05, delta=0.05)

    def test_smelt_resident_50(self):
        # Measured text-only total 10.75 GB (Smelt disables VLM mode, the
        # 0.89 GB vision tower is not loaded); backbone = 10.75 - 8.05 =
        # 2.70 GB; smelt-50 pages half the routed experts.
        s = smelt_resident(10.75, 8.05, 0.5)
        self.assertTrue(s["ok"], s.get("error"))
        self.assertAlmostEqual(s["backbone_gb"], 2.70, delta=0.05)
        self.assertAlmostEqual(s["resident_experts_gb"], 4.03, delta=0.05)
        self.assertAlmostEqual(s["resident_gb"], 6.72, delta=0.10)

    def test_smelt_resident_25(self):
        s = smelt_resident(10.75, 8.05, 0.25)
        self.assertTrue(s["ok"], s.get("error"))
        self.assertAlmostEqual(s["resident_gb"], 4.71, delta=0.10)

    def test_smelt_rejects_bad_frac(self):
        self.assertFalse(smelt_resident(10.75, 8.05, 0.0)["ok"])
        self.assertFalse(smelt_resident(10.75, 8.05, 1.5)["ok"])
        self.assertFalse(smelt_resident(8.05, 10.75, 0.5)["ok"])

    def test_total_smelt50_fits_budget(self):
        # Recipe: ~7.8 GB total @ smelt-50 (6.72 resident weights + 0.04
        # KV @ 2048 f16 + ~1.0 runtime) — comfortably inside ~11 GB.
        kv = self._estimate(2048, "f16")["kv_cache_gb"]
        s = smelt_resident(10.75, 8.05, 0.5)
        total = s["resident_gb"] + kv + 1.0
        self.assertLess(total, 11.0)
        self.assertAlmostEqual(total, 7.8, delta=0.3)

    def test_recipe_file_exists(self):
        self.assertTrue(
            os.path.isfile(os.path.join(RECIPES, "RUNG4_jang2s_35b_smelt.md")))


class TestRung4RecipeSmeltFailureNote(unittest.TestCase):
    """The RUNG4 recipe must keep the vmlx#222 Smelt failure-mode note.

    This pins a Mac-side troubleshooting fact verified from the vmlx issue
    tracker (2026-09-22): short deterministic prompts returning repeated
    junk tokens under --smelt is the loader-norm-shift signature, not a
    bad quant — the non-Smelt serve is the control. Without the pin, a
    future recipe edit could silently drop it.
    """

    def test_failure_mode_note_present(self):
        with open(os.path.join(RECIPES, "RUNG4_jang2s_35b_smelt.md")) as f:
            text = f.read()
        self.assertIn("vmlx#222", text)
        self.assertIn("norm-shift", text)
        self.assertIn("junk", text)
        self.assertIn("non-Smelt serve", text)


class TestRecipeConsistency(unittest.TestCase):
    """Every recipe named in model_profiles.yaml must exist on disk."""

    def test_profiled_recipes_exist(self):
        import re
        profiles = os.path.join(REPO, "research", "model_profiles.yaml")
        with open(profiles) as f:
            text = f.read()
        paths = re.findall(r"research/recipes/[\w.-]+\.md", text)
        self.assertTrue(paths, "no recipe references found in profiles")
        for p in paths:
            self.assertTrue(os.path.isfile(os.path.join(REPO, p)),
                            f"missing recipe file: {p}")
        self.assertTrue(os.path.isfile(os.path.join(RECIPES, "RUNG1_qwen3_8b_20tps.md")))
        self.assertTrue(os.path.isfile(os.path.join(RECIPES, "RUNG2_qwen3_14b_20tps.md")))
        self.assertTrue(os.path.isfile(os.path.join(RECIPES, "RUNG3_qwen3_5_35b_a3b_moe.md")))
        self.assertTrue(os.path.isfile(os.path.join(RECIPES, "RUNG4_jang2s_35b_smelt.md")))


class TestModelProfiles(unittest.TestCase):
    """model_profiles.yaml must not drift from measured file sizes
    (the IQ2_M and oQ2 misattributions both came from stale estimates)."""

    def _profiles(self):
        with open(os.path.join(REPO, "research", "model_profiles.yaml")) as f:
            return yaml.safe_load(f)["profiles"]

    def _by_name(self, name):
        for p in self._profiles():
            if p["name"] == name:
                return p
        self.fail(f"profile {name!r} missing from model_profiles.yaml")

    def test_oq2_profile_size_measured(self):
        # Jundot/Qwen3.6-35B-A3B-oQ2 measured 13.10 GB via the HF tree API
        # 2026-09-21; the old 8.8 GB "fits comfortably" estimate was wrong.
        p = self._by_name("qwen3.6-35b-a3b-oQ2")
        self.assertAlmostEqual(p["est_gb"], 13.1, delta=0.5)
        self.assertGreater(p["est_gb"], 11.0, "oQ2 does not fit the usable budget")
        self.assertIn("Jundot/Qwen3.6-35B-A3B-oQ2", p["notes"])
        self.assertNotIn("Fits comfortably", p["notes"])

    def test_exl3_profile_figures_pinned(self):
        # yeasah/Qwen3.6-35B-A3B-exl3, published 2026-09-21. Figures are
        # author-reported in the repo README (not independently measured):
        # 2.08 bpw = 10.83 GiB disk, 9.88 GiB VRAM under exllamav3 with
        # embeddings CPU-offloaded; qbench 2.00bpw-H5 ppl 10.865 vs bf16
        # 10.097 (+7.6%). est_gb carries the on-disk size (10.83 GiB).
        p = self._by_name("qwen3.6-35b-a3b-exl3-2bpw")
        self.assertAlmostEqual(p["est_gb"], 11.6, delta=0.5)
        self.assertEqual(p["status"], "watch", "EXL3 is transport-blocked")
        self.assertEqual(p["backend"], "exl3")
        self.assertIn("yeasah/Qwen3.6-35B-A3B-exl3", p["notes"])
        self.assertIn("9.88", p["notes"])
        self.assertIn("Author-reported", p["notes"])
        self.assertIn("NO HTTP/OpenAI server surface", p["notes"])

    def test_profiles_have_known_backends(self):
        for p in self._profiles():
            self.assertIn(p["backend"], ("ollama", "llamacpp", "mlx", "bitnet.cpp", "openai", "exl3"),
                          p["name"])


class TestRung4CascadeExample(unittest.TestCase):
    """The commented rung-4 cascade drop-in in config.yaml must stay a valid,
    buildable cascade spec pointing at the current rung-4 pick (JANG_2S).

    This pins the exact drift that just got fixed: the example used to point
    at the superseded oQ2 plan after the rung-4 pick changed to JANG_2S.
    """

    BEGIN = "# RUNG4_CASCADE_EXAMPLE_BEGIN"
    END = "# RUNG4_CASCADE_EXAMPLE_END"
    MODEL = "JANGQ-AI/Qwen3.5-35B-A3B-JANG_2S"

    def _example(self):
        with open(os.path.join(REPO, "config.yaml"), encoding="utf-8") as f:
            lines = f.read().splitlines()
        start = next(i for i, l in enumerate(lines) if l.strip() == self.BEGIN)
        end = next(i for i, l in enumerate(lines) if l.strip() == self.END)
        self.assertGreater(end, start, "cascade example markers out of order")
        body = []
        for l in lines[start + 1:end]:
            s = l.lstrip()
            self.assertTrue(s.startswith("#"), f"example line not commented: {l!r}")
            rest = s[1:]
            # Strip exactly one space: the original YAML indentation must
            # survive for safe_load to see the cascade/big nesting.
            if rest.startswith(" "):
                rest = rest[1:]
            body.append(rest)
        return yaml.safe_load("\n".join(body))

    def test_example_parses_and_targets_jang2s(self):
        ex = self._example()
        cascade = ex["cascade"]
        self.assertIn(cascade.get("router"), ("heuristic", "verify"))
        # `small` omitted -> defaults to the [ollama] block (qwen3:8b).
        self.assertNotIn("small", cascade)
        big = cascade["big"]
        self.assertEqual(big["backend"], "openai")
        self.assertEqual(big["openai"]["base_url"], "http://localhost:8000")
        self.assertEqual(big["openai"]["model"], self.MODEL)

    def test_big_spec_builds_openai_client(self):
        from src.agent import _build_llm_client
        big = self._example()["cascade"]["big"]
        client = _build_llm_client(big)  # no network: constructor is pure
        self.assertEqual(client.model, self.MODEL)
        self.assertTrue(
            client.url.startswith("http://localhost:8000/v1"),
            client.url)

    def test_no_stale_oq2_references_in_config(self):
        with open(os.path.join(REPO, "config.yaml"), encoding="utf-8") as f:
            text = f.read()
        self.assertNotIn(
            "Jundot/Qwen3.6-35B-A3B-oQ2", text,
            "superseded rung-4 pick still referenced in config.yaml")


if __name__ == "__main__":
    unittest.main()
