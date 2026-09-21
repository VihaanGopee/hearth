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
from src.quant_rnd.sweep import CONFIGS, pareto_frontier, run_sweep
from src.quant_rnd.opcount import (
    M1_PRO_MEM_BW_GBS,
    N_PARAMS_70B,
    codebook_opcount,
    decode_roofline_tps,
    equiv_adds,
    is_bandwidth_bound,
    kv_cache_bytes,
    measured_sparsity,
    scheme_report,
    ternary_opcount,
    weight_bytes,
)


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


class TestSweep(unittest.TestCase):
    def test_reconstruct_respects_nondefault_group_size(self):
        # Regression: reconstruct() used to hard-code GROUP_SIZE=128 and
        # crashed (IndexError) for any other group size.
        rng = np.random.default_rng(42)
        w = synthetic_weights(rng, n_groups=8)
        for g in (64, 128, 256):
            q = quantize_int2_kmeans_q8(w, group_size=g)
            rec = q.reconstruct()
            self.assertEqual(rec.shape, w.shape)
            self.assertTrue(math.isfinite(sqnr_db(w, rec)))
        # and the bpw accounting must match the group size
        self.assertAlmostEqual(
            quantize_int2_kmeans_q8(w, group_size=256).bpw,
            2.0 + 32 / 256 + 16 / 256, places=9)

    def test_pareto_frontier_picks_monotone_envelope(self):
        pts = [
            {"bpw": 1.7, "sqnr_db": 5.0},
            {"bpw": 2.0, "sqnr_db": 4.0},   # dominated by the 1.7 point
            {"bpw": 2.2, "sqnr_db": 6.0},   # frontier: new best
            {"bpw": 2.4, "sqnr_db": 6.0},   # tie at higher bpw: not a new best
            {"bpw": 2.6, "sqnr_db": 9.0},   # frontier
        ]
        f = pareto_frontier(pts)
        self.assertEqual([(p["bpw"], p["sqnr_db"]) for p in f],
                         [(1.7, 5.0), (2.2, 6.0), (2.6, 9.0)])

    def test_sweep_is_deterministic(self):
        a = run_sweep(seed=7, n_groups=16)
        b = run_sweep(seed=7, n_groups=16)
        self.assertEqual(a, b)

    def test_sweep_covers_all_configs_and_marks_frontier(self):
        res = run_sweep(seed=7, n_groups=16)
        self.assertEqual(len(res), len(CONFIGS))
        labels = [r["label"] for r in res]
        self.assertIn("ternary_outlier n=2", labels)
        self.assertIn("int2_kmeans_q8 g=128", labels)
        self.assertIn("dual_scale_ternary", labels)
        # sorted by ascending bpw
        bpws = [r["bpw"] for r in res]
        self.assertEqual(bpws, sorted(bpws))
        # the global best-SQNR point is always on the frontier
        best = max(res, key=lambda r: r["sqnr_db"])
        self.assertTrue(best["on_frontier"])

    def test_sweep_bpw_monotonic_in_outliers(self):
        res = run_sweep(seed=7, n_groups=16)
        by_label = {r["label"]: r for r in res}
        for frac_lo, frac_hi in [("0.001", "0.005"), ("0.005", "0.01"),
                                 ("0.01", "0.02")]:
            self.assertLess(by_label[f"int2_outlier_retain f={frac_lo}"]["bpw"],
                            by_label[f"int2_outlier_retain f={frac_hi}"]["bpw"])
        for n_lo, n_hi in [(1, 2), (2, 4), (4, 8)]:
            self.assertLess(by_label[f"ternary_outlier n={n_lo}"]["bpw"],
                            by_label[f"ternary_outlier n={n_hi}"]["bpw"])
        # smaller k-means groups cost more codebook overhead
        self.assertLess(by_label["int2_kmeans_q8 g=256"]["bpw"],
                        by_label["int2_kmeans_q8 g=128"]["bpw"])
        self.assertLess(by_label["int2_kmeans_q8 g=128"]["bpw"],
                        by_label["int2_kmeans_q8 g=64"]["bpw"])

    def test_sweep_kmeans_reference_honest_bpw(self):
        # The reference point the candidates must beat: q8 at g=128 is the
        # advertised 2.375 bpw matched-bitrate baseline.
        res = run_sweep(seed=7, n_groups=16)
        by_label = {r["label"]: r for r in res}
        ref = by_label["int2_kmeans_q8 g=128"]
        self.assertAlmostEqual(ref["bpw"], 2.375, places=6)
        self.assertTrue(math.isfinite(ref["sqnr_db"]))

    def test_sweep_ternary_outlier_n2_matches_bench_bpw(self):
        # The default ternary_outlier (n=2) swept point must agree with the
        # unswept bench bpw: same tensor, same quantizer, same number.
        bench = {r["scheme"]: r for r in run_bench(seed=7, n_groups=16)}
        sweep = {r["label"]: r for r in run_sweep(seed=7, n_groups=16)}
        self.assertAlmostEqual(sweep["ternary_outlier n=2"]["bpw"],
                               bench["ternary_outlier"]["bpw"], places=9)
        self.assertAlmostEqual(sweep["ternary_outlier n=2"]["sqnr_db"],
                               bench["ternary_outlier"]["sqnr_db"], places=6)


class TestOpCount(unittest.TestCase):
    """Tests for the op-count + roofline model (pivot part b)."""

    def test_kv_cache_matches_roadmap_ram_math(self):
        # Roadmap says 70B KV cache is ~1.3 GB at 4k ctx fp16.
        kv = kv_cache_bytes()  # Llama-70B-shaped defaults
        self.assertAlmostEqual(kv, 1.34217728e9, delta=1e6)

    def test_weight_bytes(self):
        # 70B at 2 bpw = 17.5 GB.
        self.assertAlmostEqual(weight_bytes(N_PARAMS_70B, 2.0), 17.5e9)

    def test_roofline_halving_bpw_doubles_tps(self):
        kv = kv_cache_bytes()
        fast = decode_roofline_tps(M1_PRO_MEM_BW_GBS,
                                   weight_bytes(N_PARAMS_70B, 1.0), kv)
        slow = decode_roofline_tps(M1_PRO_MEM_BW_GBS,
                                   weight_bytes(N_PARAMS_70B, 2.0), kv)
        # With KV held fixed the ratio is diluted below 2.0 but still > 1.
        self.assertGreater(fast / slow, 1.0)
        self.assertLessEqual(fast / slow, 2.0)
        # Exact: tok/s = BW / bytes_per_token.
        self.assertAlmostEqual(
            fast, M1_PRO_MEM_BW_GBS * 1e9 / (weight_bytes(N_PARAMS_70B, 1.0) + kv))

    def test_roofline_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            decode_roofline_tps(M1_PRO_MEM_BW_GBS, 0.0)
        with self.assertRaises(ValueError):
            decode_roofline_tps(M1_PRO_MEM_BW_GBS, 1e9, efficiency=1.5)

    def test_ternary_opcount_zeros_are_skipped(self):
        oc = ternary_opcount(sparsity=0.5, n_scales=1, group_size=128)
        self.assertAlmostEqual(oc["adds"], 0.5)
        self.assertAlmostEqual(oc["muls"], 1.0 / 128)
        self.assertEqual(oc["lookups"], 0.0)
        # Dual-scale costs two scale multiplies per group; outliers add MACs.
        oc2 = ternary_opcount(sparsity=0.5, n_scales=2, group_size=128,
                              n_outliers=2)
        self.assertAlmostEqual(oc2["muls"], 4.0 / 128)
        self.assertAlmostEqual(oc2["adds"], 0.5 + 2.0 / 128)

    def test_ternary_opcount_rejects_bad_sparsity(self):
        with self.assertRaises(ValueError):
            ternary_opcount(sparsity=1.5)

    def test_codebook_histogram_beats_naive(self):
        naive = codebook_opcount(method="naive")
        hist = codebook_opcount(method="histogram")
        # The histogram trick must cut multiplies far below one per weight.
        self.assertLess(hist["muls"], 0.1)
        self.assertAlmostEqual(naive["muls"], 1.0 + 2.0 / 128)
        self.assertLess(equiv_adds(hist), equiv_adds(naive))

    def test_codebook_rejects_unknown_method(self):
        with self.assertRaises(ValueError):
            codebook_opcount(method="magic")

    def test_measured_sparsity_on_real_quantizer(self):
        rng = np.random.default_rng(42)
        w = synthetic_weights(rng, n_groups=8)
        q = quantize_ternary_uniform(w)
        s = measured_sparsity(q)
        self.assertGreater(s, 0.3)  # symmetric ternary zeroes the body
        self.assertLess(s, 0.8)

    def test_end_to_end_ternary_beats_kmeans_on_roofline_and_ops(self):
        # The headline of pivot part (b): with measured sparsity from the
        # real quantizers, ternary_uniform must show BOTH a higher decode
        # ceiling (fewer bytes) AND lower energy-proxy op cost than the
        # k-means-q8 reference at group 128.
        rng = np.random.default_rng(7)
        w = synthetic_weights(rng, n_groups=64)
        qt = quantize_ternary_uniform(w, group_size=128)
        qk = quantize_int2_kmeans_q8(w, group_size=128)
        rt = scheme_report("ternary_uniform", bpw=qt.bpw, kind="ternary",
                           sparsity=measured_sparsity(qt))
        rk = scheme_report("int2_kmeans_q8", bpw=qk.bpw, kind="codebook",
                           method="histogram")
        self.assertLess(rt["bpw"], rk["bpw"])
        self.assertGreater(rt["roofline_tps"], rk["roofline_tps"])
        # Speedup ratio is bounded above by the bpw ratio (KV dilutes it).
        self.assertLessEqual(rt["roofline_tps"] / rk["roofline_tps"],
                             rk["bpw"] / rt["bpw"] + 1e-9)
        self.assertLess(rt["equiv_adds_per_w"], rk["equiv_adds_per_w"])
        # Both are bandwidth-bound on M1-Pro-class hardware: arithmetic
        # intensity well under a 10 FLOP/byte machine balance.
        for r in (rt, rk):
            self.assertTrue(is_bandwidth_bound(r["flops_per_byte"],
                                               peak_flops=2e12,
                                               bandwidth_gbs=200.0))
            self.assertLess(r["flops_per_byte"], 10.0)

    def test_scheme_report_rejects_unknown_kind(self):
        with self.assertRaises(ValueError):
            scheme_report("x", bpw=2.0, kind="magic")


if __name__ == "__main__":
    unittest.main()
