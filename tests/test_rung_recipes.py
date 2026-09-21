"""Pins the RAM/roofline figures quoted in the rung validation recipes against
fitcheck, so the docs can't silently drift from the math.

If a recipe's numbers change (different params_b, quant, arch), update BOTH
the recipe markdown AND this test — the test fails until they agree.
"""
import math
import os
import unittest

from src.tools.fitcheck import estimate

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


if __name__ == "__main__":
    unittest.main()
