"""Tests for the estimate_fit RAM-budget math (70B-quest core)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.fitcheck import estimate, QUANT_BPW


class TestEstimateFit(unittest.TestCase):
    def test_70b_iq1m_weights(self):
        r = estimate(70, "iq1_m", 8192, arch="llama-70b", budget_gb=11.0)
        self.assertTrue(r["ok"])
        # 70e9 * 1.75 / 8 / 1e9 = 15.3125
        self.assertAlmostEqual(r["weights_gb"], 15.31, places=2)
        self.assertFalse(r["fits"])  # honest: doesn't fit 11 GB budget

    def test_70b_tq1_0(self):
        r = estimate(70, "tq1_0", 4096, arch="llama-70b", budget_gb=11.0)
        self.assertTrue(r["ok"])
        self.assertAlmostEqual(r["weights_gb"], 70 * 1.69 / 8, places=2)
        self.assertFalse(r["fits"])

    def test_kv_cache_formula(self):
        # 2 * 80 layers * 8 kv_heads * 128 head_dim * 8192 ctx * 2 bytes
        r = estimate(70, "q4_k_m", 8192, arch="llama-70b",
                     cache_type="f16", budget_gb=100)
        self.assertTrue(r["ok"])
        expected = 2 * 80 * 8 * 128 * 8192 * 2 / 1e9
        self.assertAlmostEqual(r["kv_cache_gb"], expected, places=2)

    def test_8b_q4_fits(self):
        r = estimate(8, "q4_k_m", 16384, arch="qwen3-8b", budget_gb=11.0)
        self.assertTrue(r["ok"])
        self.assertTrue(r["fits"])

    def test_quant_table_sane(self):
        # every entry must be a plausible bits-per-weight value
        for q, bpw in QUANT_BPW.items():
            self.assertGreater(bpw, 0, q)
            self.assertLessEqual(bpw, 32, q)

    def test_unknown_quant_errors(self):
        r = estimate(70, "q99_zzz", 8192, arch="llama-70b")
        self.assertFalse(r["ok"])

    def test_unknown_arch_errors(self):
        r = estimate(70, "q4_k_m", 8192, arch="nope-999b")
        self.assertFalse(r["ok"])

    def test_missing_layer_info_errors(self):
        r = estimate(70, "q4_k_m", 8192)
        self.assertFalse(r["ok"])

    def test_verdict_mentions_numbers(self):
        r = estimate(8, "q4_k_m", 8192, arch="qwen3-8b", budget_gb=11.0)
        self.assertIn("GB", r["verdict"])
        self.assertIn("FITS", r["verdict"])


if __name__ == "__main__":
    unittest.main()
