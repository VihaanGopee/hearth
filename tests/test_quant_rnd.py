"""Tests for the quant R&D prototypes (avenue H): baselines + candidates."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quant_rnd import (
    SCHEMES,
    quantize_dual_scale_ternary,
    quantize_int2_kmeans,
    quantize_int2_kmeans_q8,
    quantize_int2_symmetric,
    quantize_ternary_outlier,
    quantize_ternary_uniform,
)
from src.quant_rnd.bench import run_bench, sqnr_db, synthetic_weights


def _mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


class TestSchemes(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.w = synthetic_weights(rng, n_groups=8)

    def test_registry_has_baselines_and_candidates(self):
        self.assertEqual(set(SCHEMES),
                         {"ternary_uniform", "int2_symmetric", "int2_kmeans",
                          "int2_kmeans_q8",
                          "int2_outlier_retain", "dual_scale_ternary",
                          "ternary_outlier"})

    def test_ternary_codes_valid(self):
        q = quantize_ternary_uniform(self.w)
        self.assertTrue(set(np.unique(q.codes)) <= {-1, 0, 1})

    def test_int2_codes_valid(self):
        q = quantize_int2_symmetric(self.w)
        self.assertTrue(set(np.unique(q.codes)) <= {-3, -1, 1, 3})

    def test_roundtrip_shape_finite(self):
        for name, fn in SCHEMES.items():
            q = fn(self.w)
            r = q.reconstruct()
            self.assertEqual(r.shape, self.w.shape, name)
            self.assertTrue(np.all(np.isfinite(r)), name)

    def test_all_zero_input_no_nan(self):
        for name, fn in SCHEMES.items():
            q = fn(np.zeros(256, dtype=np.float32))
            self.assertTrue(np.all(np.isfinite(q.reconstruct())), name)

    def test_bpw_accounting(self):
        # ternary payload log2(3) + one fp16 scale per 128-group
        q = quantize_ternary_uniform(self.w)
        self.assertAlmostEqual(q.bpw, math.log2(3) + 16 / 128, places=6)
        # dual-scale pays for a second fp16 scale per group
        d = quantize_dual_scale_ternary(self.w)
        self.assertAlmostEqual(d.bpw, math.log2(3) + 32 / 128, places=6)
        self.assertGreater(d.bpw, q.bpw)
        # int2: 2-bit payload + one scale
        i = quantize_int2_symmetric(self.w)
        self.assertAlmostEqual(i.bpw, 2.0 + 16 / 128, places=6)

    def test_dual_scale_wins_on_skewed_data(self):
        # Skewed but two-sided: large positives, small negatives.
        # A symmetric scale compromises both sides; DST fits each side.
        rng = np.random.default_rng(1234)
        n = 8 * 128
        pos = np.abs(rng.standard_normal(n // 2)) * 2.0 + 0.5
        neg = -np.abs(rng.standard_normal(n - n // 2)) * 0.1
        w = rng.permutation(np.concatenate([pos, neg])).astype(np.float32)
        uni = quantize_ternary_uniform(w)
        dst = quantize_dual_scale_ternary(w)
        self.assertGreater(sqnr_db(w, dst.reconstruct()),
                           sqnr_db(w, uni.reconstruct()) + 1.0)

    def test_int2_kmeans_codes_valid(self):
        q = quantize_int2_kmeans(self.w)
        self.assertTrue(set(np.unique(q.codes)) <= {0, 1, 2, 3})

    def test_int2_kmeans_bpw(self):
        # 2-bit payload + 4 fp16 centroids per 128-group = 2.5 bpw.
        # Heavier than the ~2.06 bpw candidates; honest, and stated in docs.
        q = quantize_int2_kmeans(self.w)
        self.assertAlmostEqual(q.bpw, 2.0 + 4 * 16 / 128, places=6)
        self.assertEqual(q.scales.shape, (8, 4))

    def test_int2_kmeans_deterministic(self):
        a = quantize_int2_kmeans(self.w)
        b = quantize_int2_kmeans(self.w)
        self.assertTrue(np.array_equal(a.codes, b.codes))
        self.assertTrue(np.array_equal(a.scales, b.scales))

    def test_int2_kmeans_reconstruct_is_centroid_lookup(self):
        q = quantize_int2_kmeans(self.w)
        r = q.reconstruct()
        n = self.w.shape[0]
        group_id = np.arange(n) // 128
        expected = q.scales[group_id, q.codes.astype(int)]
        self.assertTrue(np.allclose(r, expected))

    def test_int2_kmeans_beats_naive_int2_on_outliers(self):
        # The whole point of the baseline: fitting centroids to the group
        # distribution (instead of stretching a fixed codebook by amax)
        # recovers the body that outliers destroy.
        k = quantize_int2_kmeans(self.w)
        i = quantize_int2_symmetric(self.w)
        self.assertGreater(sqnr_db(self.w, k.reconstruct()),
                           sqnr_db(self.w, i.reconstruct()) + 3.0)

    def test_int2_kmeans_constant_input_exact(self):
        q = quantize_int2_kmeans(np.full(256, 0.7, dtype=np.float32))
        self.assertTrue(np.allclose(q.reconstruct(), 0.7))

    def test_int2_kmeans_q8_codes_valid(self):
        q = quantize_int2_kmeans_q8(self.w)
        self.assertTrue(set(np.unique(q.codes)) <= {0, 1, 2, 3})

    def test_int2_kmeans_q8_bpw(self):
        # 2-bit payload + 4 int8 centroids + one fp16 codebook scale
        # per 128-group = 2.375 bpw (vs 2.5 for the fp16-codebook variant).
        q = quantize_int2_kmeans_q8(self.w)
        self.assertAlmostEqual(q.bpw, 2.0 + 4 * 8 / 128 + 16 / 128, places=6)
        self.assertEqual(q.scales.shape, (8, 4))

    def test_int2_kmeans_q8_deterministic(self):
        a = quantize_int2_kmeans_q8(self.w)
        b = quantize_int2_kmeans_q8(self.w)
        self.assertTrue(np.array_equal(a.codes, b.codes))
        self.assertTrue(np.array_equal(a.scales, b.scales))

    def test_int2_kmeans_q8_reconstruct_is_codebook_lookup(self):
        # scales hold the DEQUANTIZED (8-bit-rounded) centroids.
        q = quantize_int2_kmeans_q8(self.w)
        r = q.reconstruct()
        n = self.w.shape[0]
        group_id = np.arange(n) // 128
        expected = q.scales[group_id, q.codes.astype(int)]
        self.assertTrue(np.allclose(r, expected))

    def test_int2_kmeans_q8_codebook_rounding_is_nearly_free(self):
        # 8-bit codebook storage must not beat the fp16 fit, and on
        # Gaussian-ish data the rounding cost is tiny (measured 0.0003 dB).
        k = quantize_int2_kmeans(self.w)
        q8 = quantize_int2_kmeans_q8(self.w)
        s_fp16 = sqnr_db(self.w, k.reconstruct())
        s_q8 = sqnr_db(self.w, q8.reconstruct())
        self.assertLessEqual(s_q8, s_fp16)
        self.assertLess(s_fp16 - s_q8, 0.1)

    def test_int2_kmeans_q8_still_beats_ternary_outlier(self):
        # The session's key question: at (roughly) matched bitrate, does
        # our best candidate beat the classical baseline? Answer: no -
        # Lloyd k-means wins by >2 dB even with an 8-bit codebook.
        # Recorded as a negative result; the candidates' case now rests on
        # ternary compute (add/sub) rather than SQNR.
        q8 = quantize_int2_kmeans_q8(self.w)
        t = quantize_ternary_outlier(self.w)
        self.assertLess(q8.bpw - t.bpw, 0.35)  # near-matched bitrate
        self.assertGreater(sqnr_db(self.w, q8.reconstruct()),
                           sqnr_db(self.w, t.reconstruct()) + 2.0)

    def test_int2_kmeans_q8_constant_input_exact(self):
        q = quantize_int2_kmeans_q8(np.full(256, 0.7, dtype=np.float32))
        self.assertTrue(np.allclose(q.reconstruct(), 0.7))

    def test_ternary_outlier_stores_outliers_exactly(self):
        q = quantize_ternary_outlier(self.w)
        r = q.reconstruct()
        # every stored outlier is kept in fp16 -> exact reconstruction
        self.assertTrue(np.array_equal(r[q._outlier_idx], q._outlier_vals))

    def test_ternary_outlier_matched_bitrate_vs_int2(self):
        # Same synthetic tensor: ternary+outliers should not lose badly to
        # plain int2 despite a slightly LOWER bitrate (honest comparison).
        q = quantize_ternary_outlier(self.w)
        i = quantize_int2_symmetric(self.w)
        self.assertLessEqual(q.bpw, i.bpw)
        self.assertGreaterEqual(sqnr_db(self.w, q.reconstruct()),
                                sqnr_db(self.w, i.reconstruct()) - 1.0)


class TestBench(unittest.TestCase):
    def test_deterministic(self):
        a = run_bench(seed=7, n_groups=16)
        b = run_bench(seed=7, n_groups=16)
        self.assertEqual(a, b)

    def test_results_sorted_and_sane(self):
        res = run_bench(seed=7, n_groups=16)
        self.assertEqual(len(res), len(SCHEMES))
        sqnrs = [r["sqnr_db"] for r in res]
        self.assertEqual(sqnrs, sorted(sqnrs, reverse=True))
        for r in res:
            self.assertTrue(math.isfinite(r["sqnr_db"]))
            self.assertGreater(r["bpw"], 1.0)
            self.assertLess(r["bpw"], 3.0)

    def test_sqnr_perfect_reconstruction(self):
        w = np.array([1.0, -2.0, 0.5], dtype=np.float32)
        self.assertEqual(sqnr_db(w, w), float("inf"))


if __name__ == "__main__":
    unittest.main()
