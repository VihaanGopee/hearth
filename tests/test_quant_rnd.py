"""Tests for the quant R&D prototypes (avenue H): baselines + candidates."""
import json
import math
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quant_rnd import (
    SCHEMES,
    quantize_dual_scale_ternary,
    quantize_int2_kmeans,
    quantize_int2_kmeans_q8,
    quantize_int2_symmetric,
    quantize_ternary_lloyd,
    quantize_ternary_lloyd_ds,
    quantize_ternary_1step,
    quantize_ternary_1step_ds,
    quantize_ternary_outlier,
    quantize_ternary_uniform,
)
from src.quant_rnd.bench import run_bench, sqnr_db, synthetic_weights
from src.quant_rnd.realweights import (
    linear_weight_tensors,
    parse_args,
    rank_on_real_weights,
    read_safetensors,
    sample_groups,
)
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
    prefill_roofline_tps,
    scheme_report,
    side_fractions,
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
                         {"ternary_uniform", "ternary_lloyd", "ternary_lloyd_ds",
                          "ternary_1step", "ternary_1step_ds",
                          "int2_symmetric", "int2_kmeans",
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

    def test_fitted_ternary_opcount_reference(self):
        # Opcount re-run with the FITTED ternary reference (ternary_1step,
        # the practical fitted encoder) instead of ternary_uniform. The
        # refit widens the thresholds -> higher zero-rate (~0.41 vs ~0.31)
        # -> fewer adds/weight. The decode ceiling is byte-driven and
        # must not move (1.36x), while the energy-proxy ratio must be at
        # least as large as uniform's 3.04x. Measured: 1.357x / 3.52x.
        rng = np.random.default_rng(7)
        w = synthetic_weights(rng, n_groups=64)
        q1 = quantize_ternary_1step(w, group_size=128)
        qk = quantize_int2_kmeans_q8(w, group_size=128)
        r1 = scheme_report("ternary_1step", bpw=q1.bpw, kind="ternary",
                           sparsity=measured_sparsity(q1))
        rk = scheme_report("int2_kmeans_q8", bpw=qk.bpw, kind="codebook",
                           method="histogram")
        self.assertAlmostEqual(r1["roofline_tps"] / rk["roofline_tps"],
                               1.357, delta=0.01)
        self.assertGreaterEqual(rk["equiv_adds_per_w"]
                                / r1["equiv_adds_per_w"], 3.04)
        self.assertLess(r1["equiv_adds_per_w"], rk["equiv_adds_per_w"])
        # The fitted reference is sparser than uniform - the numbers are
        # driven by a real, measured quantity, not an assumption.
        zu = float(np.mean(quantize_ternary_uniform(w).codes == 0))
        self.assertGreater(measured_sparsity(q1), zu)

    def test_side_fractions(self):
        # Unit coverage for the dual op-count input: fractions sum to 1,
        # zero matches measured_sparsity, and non-ternary codes are
        # rejected instead of silently mis-measured.
        rng = np.random.default_rng(7)
        w = synthetic_weights(rng, n_groups=64)
        qd = quantize_ternary_1step_ds(w, group_size=128, n_iter=2)
        sf = side_fractions(qd)
        self.assertAlmostEqual(sf["pos"] + sf["neg"] + sf["zero"], 1.0)
        self.assertAlmostEqual(sf["zero"], measured_sparsity(qd))
        self.assertGreater(sf["pos"], 0.0)
        self.assertGreater(sf["neg"], 0.0)
        with self.assertRaises(ValueError):
            side_fractions(quantize_int2_kmeans(w, group_size=128))

    def test_dual_fitted_ternary_opcount_reference(self):
        # Opcount re-run with the DUAL fitted reference (ternary_1step_ds,
        # n_iter=2 - the practical dual encoder from the 1-step_ds session)
        # instead of the symmetric 1-step. The dual refit stores two fp16
        # scales per group (1.835 bpw at g128) and adapts each side's
        # threshold independently, so both the zero-rate and the
        # pos/neg split are measured per scheme, not borrowed from the
        # symmetric run. Measured seed 7, 64 groups:
        #   clean:  zero=0.441 (vs symmetric 1-step's 0.41)
        #   skew 0.5: pos=0.415 / neg=0.168 - visibly asymmetric,
        #           which is exactly the per-side op profile shift the
        #           dual re-run was meant to capture.
        # n_scales=2 (not 1) feeds the extra scale multiply into the
        # energy proxy; it is ~0.016 muls/weight - genuinely "slight".
        rng = np.random.default_rng(7)
        w = synthetic_weights(rng, n_groups=64)
        w_skew = w + 0.5
        qd = quantize_ternary_1step_ds(w, group_size=128, n_iter=2)
        qd_s = quantize_ternary_1step_ds(w_skew, group_size=128, n_iter=2)
        qk = quantize_int2_kmeans_q8(w, group_size=128)
        self.assertAlmostEqual(qd.bpw, math.log2(3) + 32 / 128, places=6)
        for q in (qd, qd_s):
            sf = side_fractions(q)
            rd = scheme_report("ternary_1step_ds", bpw=q.bpw, kind="ternary",
                               sparsity=sf["zero"], n_scales=2,
                               group_size=128)
            rk = scheme_report("int2_kmeans_q8", bpw=qk.bpw, kind="codebook",
                               method="histogram")
            # Decode ceiling is byte-driven: lower than symmetric 1-step's
            # 1.357x because the dual reference carries 1.835 vs 1.710 bpw.
            self.assertAlmostEqual(rd["roofline_tps"] / rk["roofline_tps"],
                                   1.272, delta=0.01)
            # Energy-proxy advantage holds at >= 3.3x on both clean and
            # skewed tensors (measured 3.53 clean / 3.40 skewed), and the
            # dual's extra scale multiply is real but negligible.
            self.assertGreaterEqual(rk["equiv_adds_per_w"]
                                    / rd["equiv_adds_per_w"], 3.3)
            self.assertAlmostEqual(rd["muls_per_w"], 2 / 128)
        # The per-side split the single-scale model cannot express: on a
        # skewed tensor the dual refit keeps ~2.5x more weights positive
        # than negative.
        sf_s = side_fractions(qd_s)
        self.assertGreater(sf_s["pos"], 2.0 * sf_s["neg"])

    def test_prefill_roofline_regime_flip_and_ratio(self):
        # Prefill roofline with the two adopted references: ternary_1step
        # (fitted ternary, the practical encoder) vs int2_kmeans_q8 (the
        # classical Lloyd reference), both at g128, seed 7. The roadmap
        # predicts the ternary op advantage becomes first-order in the
        # compute-bound prefill regime; this pins the regime flip and the
        # ceiling ratio the model actually produces.
        # peak_flops is explicit: Apple-published 5.2 TFLOPS FP32 for the
        # M1 Pro GPU. The ratio assertions do not depend on its value
        # (it cancels); the regime assertions hold for any plausible peak.
        peak = 5.2e12
        rng = np.random.default_rng(7)
        w = synthetic_weights(rng, n_groups=64)
        q1 = quantize_ternary_1step(w, group_size=128)
        qk = quantize_int2_kmeans_q8(w, group_size=128)
        oc_t = ternary_opcount(measured_sparsity(q1))
        oc_k = codebook_opcount(group_size=128, method="histogram")
        kw = dict(n_params=N_PARAMS_70B, bandwidth_gbs=M1_PRO_MEM_BW_GBS,
                  peak_flops=peak)
        # L=1 is decode-like: arithmetic intensity ~3 FLOP/byte, far below
        # the machine balance (~26) -> bandwidth-bound for both.
        p1_t = prefill_roofline_tps(bpw=q1.bpw, opcount=oc_t, prompt_len=1,
                                    **kw)
        p1_k = prefill_roofline_tps(bpw=qk.bpw, opcount=oc_k, prompt_len=1,
                                    **kw)
        self.assertTrue(p1_t["bandwidth_bound"])
        self.assertTrue(p1_k["bandwidth_bound"])
        # L=4096 flips the regime: every weight is reused 4096x, so AI is
        # ~100x the machine balance -> compute-bound for both.
        p4_t = prefill_roofline_tps(bpw=q1.bpw, opcount=oc_t,
                                    prompt_len=4096, **kw)
        p4_k = prefill_roofline_tps(bpw=qk.bpw, opcount=oc_k,
                                    prompt_len=4096, **kw)
        self.assertFalse(p4_t["bandwidth_bound"])
        self.assertFalse(p4_k["bandwidth_bound"])
        self.assertGreater(
            p4_t["arith_intensity_flops_per_byte"],
            100 * peak / (M1_PRO_MEM_BW_GBS * 1e9))
        # Compute-bound -> the ceiling ratio is exactly the FLOP-per-weight
        # ratio (peak, efficiency and prompt_len all cancel). Measured on
        # seed 7 this is ~1.79x: ternary's add/skip MAC vs the codebook
        # histogram MAC. Conservative: adds count as 1 FLOP against an
        # FMA-counted peak, so a real add-dominated ternary kernel has up
        # to ~2x headroom above this ceiling on FMA hardware.
        f_t = oc_t["adds"] + oc_t["muls"]
        f_k = oc_k["adds"] + oc_k["muls"]
        ratio = p4_t["ceil_tps"] / p4_k["ceil_tps"]
        self.assertAlmostEqual(ratio, f_k / f_t, delta=1e-6)
        self.assertGreater(ratio, 1.5)
        # Sanity: the compute-bound ceiling reduces to peak/flops_per_token.
        self.assertAlmostEqual(p4_t["ceil_tps"], peak / (N_PARAMS_70B * f_t),
                               delta=1.0)
        # Input validation.
        with self.assertRaises(ValueError):
            prefill_roofline_tps(n_params=N_PARAMS_70B, bpw=q1.bpw,
                                 opcount=oc_t, prompt_len=0,
                                 bandwidth_gbs=M1_PRO_MEM_BW_GBS,
                                 peak_flops=peak)
        with self.assertRaises(ValueError):
            prefill_roofline_tps(n_params=N_PARAMS_70B, bpw=q1.bpw,
                                 opcount=oc_t, prompt_len=8,
                                 bandwidth_gbs=M1_PRO_MEM_BW_GBS,
                                 peak_flops=0.0)


class TestTernaryLloyd(unittest.TestCase):
    """Candidate C: Lloyd with the codebook constrained to ternary."""

    def setUp(self):
        rng = np.random.default_rng(7)
        self.w = synthetic_weights(rng, n_groups=64)

    def test_codes_valid(self):
        for fn in (quantize_ternary_lloyd, quantize_ternary_lloyd_ds):
            q = fn(self.w)
            self.assertTrue(set(np.unique(q.codes)) <= {-1, 0, 1},
                            fn.__name__)

    def test_bpw_matched_to_heuristic_twins(self):
        # Same storage as ternary_uniform (1 fp16 scale) and
        # dual_scale_ternary (2 fp16 scales): the fidelity comparison is at
        # exactly matched bitrate.
        ql = quantize_ternary_lloyd(self.w)
        self.assertAlmostEqual(ql.bpw, math.log2(3) + 16 / 128, places=6)
        self.assertAlmostEqual(
            ql.bpw, quantize_ternary_uniform(self.w).bpw, places=9)
        qd = quantize_ternary_lloyd_ds(self.w)
        self.assertAlmostEqual(qd.bpw, math.log2(3) + 32 / 128, places=6)
        self.assertAlmostEqual(
            qd.bpw, quantize_dual_scale_ternary(self.w).bpw, places=9)

    def test_deterministic(self):
        for fn in (quantize_ternary_lloyd, quantize_ternary_lloyd_ds):
            a, b = fn(self.w), fn(self.w)
            self.assertTrue(np.array_equal(a.codes, b.codes), fn.__name__)
            self.assertTrue(np.array_equal(a.scales, b.scales), fn.__name__)

    def test_dual_scale_reconstruct_matches_scales(self):
        q = quantize_ternary_lloyd_ds(self.w)
        r = q.reconstruct()
        g0 = q.scales[0]
        c0 = q.codes[:128].astype(np.float32)
        expected = np.where(c0 > 0, g0[0], np.where(c0 < 0, -g0[1], 0.0))
        np.testing.assert_allclose(r[:128], expected, rtol=1e-6)

    def test_lloyd_beats_heuristic_at_same_bitrate(self):
        # The experiment this candidate was built to answer: does Lloyd
        # *fitting* rescue ternary at matched bitrate? Seed 7 reproduces
        # the bench numbers: +1.37 dB symmetric, +1.41 dB dual-scale.
        sqnr_l = sqnr_db(self.w, quantize_ternary_lloyd(self.w).reconstruct())
        sqnr_u = sqnr_db(self.w,
                         quantize_ternary_uniform(self.w).reconstruct())
        self.assertGreater(sqnr_l, sqnr_u + 1.0)
        sqnr_d = sqnr_db(self.w,
                         quantize_ternary_lloyd_ds(self.w).reconstruct())
        sqnr_dst = sqnr_db(self.w,
                           quantize_dual_scale_ternary(self.w).reconstruct())
        self.assertGreater(sqnr_d, sqnr_dst + 1.0)

    def test_dual_scale_lloyd_captures_skew(self):
        # Asymmetric scales fit each side: on a skewed tensor the
        # dual-scale Lloyd variant should beat the symmetric one.
        rng = np.random.default_rng(1234)
        n = 16 * 128
        pos = np.abs(rng.standard_normal(n // 2)) * 2.0 + 0.5
        neg = -np.abs(rng.standard_normal(n - n // 2)) * 0.1
        w = rng.permutation(np.concatenate([pos, neg])).astype(np.float32)
        sqnr_d = sqnr_db(w, quantize_ternary_lloyd_ds(w).reconstruct())
        sqnr_l = sqnr_db(w, quantize_ternary_lloyd(w).reconstruct())
        self.assertGreater(sqnr_d, sqnr_l)

    def test_ternary_lloyd_beats_outlier_schemes_below_2bpw(self):
        # Sweep result pinned: Lloyd-fit ternary owns the sub-2.06 bpw
        # region, beating ternary_outlier n=2 at a lower bitrate.
        ql = quantize_ternary_lloyd(self.w)
        qo = quantize_ternary_outlier(self.w, n_outliers=2)
        self.assertLess(ql.bpw, qo.bpw)
        self.assertGreater(sqnr_db(self.w, ql.reconstruct()),
                           sqnr_db(self.w, qo.reconstruct()))

    def test_one_sided_group_no_nan(self):
        w = np.abs(self.w)  # no negative weights anywhere
        for fn in (quantize_ternary_lloyd, quantize_ternary_lloyd_ds):
            q = fn(w)
            r = q.reconstruct()
            self.assertTrue(np.all(np.isfinite(r)), fn.__name__)


from src.quant_rnd.diagnose import (
    diagnostic_report,
    lloyd_step_sqnr,
    threshold_grid_ablation,
)
from src.quant_rnd.schemes import _groups, _ternary_lloyd_fit


class TestLloydGainDecomposition(unittest.TestCase):
    """Decompose the +1.4 dB Lloyd-fit ternary win (diagnose.py)."""

    def setUp(self):
        rng = np.random.default_rng(7)
        self.w = synthetic_weights(rng, n_groups=64)

    def test_lloyd_history_records_heuristic_init(self):
        wp, n_groups, _n = _groups(self.w)
        h = []
        _ternary_lloyd_fit(wp[0], dual=False, history=h)
        self.assertGreaterEqual(len(h), 1)
        # First recorded state is the absmean heuristic init (the fitter
        # works in float64, so compare against the float64 absmean).
        g64 = wp[0].astype(np.float64)
        self.assertAlmostEqual(h[0][0], float(np.mean(np.abs(g64))),
                               places=9)
        self.assertAlmostEqual(h[0][1], h[0][0], places=9)  # symmetric

    def test_lloyd_history_does_not_change_scheme_output(self):
        # The history hook is diagnostics-only: same output with/without.
        wp, _ng, _n = _groups(self.w)
        h = []
        a = _ternary_lloyd_fit(wp[0], dual=False, history=h)
        b = _ternary_lloyd_fit(wp[0], dual=False)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])
        np.testing.assert_array_equal(a[2], b[2])

    def test_lloyd_step0_matches_uniform(self):
        steps = lloyd_step_sqnr(self.w)
        uniform = sqnr_db(self.w, quantize_ternary_uniform(self.w).reconstruct())
        self.assertAlmostEqual(steps[0], uniform, places=6)

    def test_lloyd_step_sqnr_monotone(self):
        # Alternating minimization: MSE cannot increase step to step.
        steps = lloyd_step_sqnr(self.w)
        for prev, cur in zip(steps, steps[1:]):
            self.assertGreaterEqual(cur, prev - 1e-9)

    def test_lloyd_converged_matches_scheme(self):
        steps = lloyd_step_sqnr(self.w)
        lloyd = sqnr_db(self.w, quantize_ternary_lloyd(self.w).reconstruct())
        self.assertAlmostEqual(steps[-1], lloyd, places=3)

    def test_scale_refit_is_dominant_gain_term(self):
        # Verified decomposition: the one-step L2 scale refit captures
        # most of the Lloyd gain; threshold adaptation is the remainder.
        steps = lloyd_step_sqnr(self.w)
        self.assertGreater(len(steps), 1)
        scale_refit = steps[1] - steps[0]
        threshold_adapt = steps[-1] - steps[1]
        self.assertGreater(scale_refit, 0.0)
        self.assertGreater(scale_refit, threshold_adapt)

    def test_decomposition_sums_to_total_gain(self):
        r = diagnostic_report(seed=7)
        self.assertAlmostEqual(
            r["scale_refit_gain_db"] + r["threshold_adapt_gain_db"],
            r["total_gain_db"], places=9)

    def test_threshold_grid_at_least_uniform(self):
        # The grid includes alpha=0.5 (uniform's threshold), so the best
        # grid point is provably no worse than the heuristic.
        uniform = sqnr_db(self.w, quantize_ternary_uniform(self.w).reconstruct())
        grid_db, _alpha = threshold_grid_ablation(self.w)
        self.assertGreaterEqual(grid_db, uniform - 1e-9)

    def test_threshold_grid_confirms_threshold_was_not_bottleneck(self):
        # With the heuristic scale held fixed, the grid's best threshold
        # is the heuristic one: the problem was the scale, not the
        # threshold placement.
        _grid_db, alpha = threshold_grid_ablation(self.w)
        self.assertAlmostEqual(alpha, 0.5, places=6)


class TestTernary1Step(unittest.TestCase):
    """Candidate D: "1-step Lloyd" ternary - one fit iteration, O(1) cost."""

    def setUp(self):
        rng = np.random.default_rng(7)
        self.w = synthetic_weights(rng, n_groups=64)

    def test_codes_valid_and_bpw_matched(self):
        q = quantize_ternary_1step(self.w)
        self.assertTrue(set(np.unique(q.codes)) <= {-1, 0, 1})
        self.assertEqual(q.name, "ternary_1step")
        # Identical storage to ternary_uniform: 1.585-bit payload + 1 fp16
        # scale per group - the fidelity comparison is exactly matched.
        self.assertAlmostEqual(q.bpw, math.log2(3) + 16 / 128, places=6)
        self.assertAlmostEqual(q.bpw,
                               quantize_ternary_uniform(self.w).bpw,
                               places=9)

    def test_deterministic(self):
        a, b = quantize_ternary_1step(self.w), quantize_ternary_1step(self.w)
        self.assertTrue(np.array_equal(a.codes, b.codes))
        self.assertTrue(np.array_equal(a.scales, b.scales))

    def test_all_zero_input_no_nan(self):
        q = quantize_ternary_1step(np.zeros(512, dtype=np.float32))
        r = q.reconstruct()
        self.assertTrue(np.all(np.isfinite(r)))
        self.assertTrue(np.all(r == 0.0))

    def test_captures_most_of_lloyd_gain(self):
        # The backlog question: does ONE Lloyd iteration capture ~85% of
        # the full ternary_lloyd win over ternary_uniform? Seed 7 clean
        # tensor measures 6.72 vs 5.56 vs 6.93 dB -> capture 0.85. Assert
        # >= 0.80 so the test has margin against float noise, and assert
        # the same on a skewed tensor (capture 0.84 measured there).
        sqnr_u = sqnr_db(self.w, quantize_ternary_uniform(self.w).reconstruct())
        sqnr_l = sqnr_db(self.w, quantize_ternary_lloyd(self.w).reconstruct())
        sqnr_1 = sqnr_db(self.w, quantize_ternary_1step(self.w).reconstruct())
        self.assertGreater(sqnr_l, sqnr_u + 1.0)  # sanity: full Lloyd wins
        capture = (sqnr_1 - sqnr_u) / (sqnr_l - sqnr_u)
        self.assertGreaterEqual(capture, 0.80)

        rng = np.random.default_rng(7)
        ws = synthetic_weights(rng, n_groups=64, skew=0.5)
        u = sqnr_db(ws, quantize_ternary_uniform(ws).reconstruct())
        l = sqnr_db(ws, quantize_ternary_lloyd(ws).reconstruct())
        o = sqnr_db(ws, quantize_ternary_1step(ws).reconstruct())
        self.assertGreaterEqual((o - u) / (l - u), 0.80)

    def test_scale_differs_from_absmean_heuristic(self):
        # The refit must actually move the scale: refit s > absmean s0
        # on these tensors (absmean underestimates the L2-optimal scale).
        q1 = quantize_ternary_1step(self.w)
        qu = quantize_ternary_uniform(self.w)
        s1 = q1.scales.ravel().astype(np.float64)
        s0 = qu.scales.ravel().astype(np.float64)
        self.assertGreater(np.mean(s1 / s0), 1.0)

    def test_zero_rate_measured_and_higher_than_uniform(self):
        # For the opcount re-run: fitted ternary's sparsity differs from
        # ternary_uniform's (the refit widens the thresholds). Seed 7
        # measures ~0.41 for 1-step vs ~0.31 for uniform; assert 1-step's
        # zero rate is within [0.35, 0.50] and exceeds uniform's.
        zr_1 = float(np.mean(quantize_ternary_1step(self.w).codes == 0))
        zr_u = float(np.mean(quantize_ternary_uniform(self.w).codes == 0))
        self.assertGreater(zr_1, zr_u)
        self.assertGreaterEqual(zr_1, 0.35)
        self.assertLessEqual(zr_1, 0.50)


class TestTernary1StepDs(unittest.TestCase):
    """Candidate D, dual-scale twin: "1-step Lloyd" with {-s_neg,0,+s_pos}.

    The backlog question: does one fit iteration also capture the
    dual-scale Lloyd win (ternary_lloyd_ds over dual_scale_ternary) the
    way the symmetric 1-step captures 0.84-0.87 of the symmetric win?
    Answer (seeds 7-9, clean + skew 0.5): only partially - one iteration
    captures 0.55-0.85, two iterations 0.63-0.97, three 0.65-0.99. The
    dual case converges slower, so the practical dual encoder is 2-3
    fixed iterations (still O(1)), not 1.
    """

    def setUp(self):
        rng = np.random.default_rng(7)
        self.w = synthetic_weights(rng, n_groups=64)
        rng2 = np.random.default_rng(7)
        self.ws = synthetic_weights(rng2, n_groups=64, skew=0.5)

    def _capture(self, w, n_iter):
        h = sqnr_db(w, quantize_dual_scale_ternary(w).reconstruct())
        f = sqnr_db(w, quantize_ternary_lloyd_ds(w).reconstruct())
        o = sqnr_db(w, quantize_ternary_1step_ds(w,
                                                n_iter=n_iter).reconstruct())
        self.assertGreater(f, h + 1.0)  # sanity: full dual Lloyd wins
        return (o - h) / (f - h)

    def test_codes_valid_and_bpw_matched(self):
        q = quantize_ternary_1step_ds(self.w)
        self.assertTrue(set(np.unique(q.codes)) <= {-1, 0, 1})
        self.assertEqual(q.name, "ternary_1step_ds")
        self.assertEqual(q.scales.shape[1], 2)  # (s_pos, s_neg) per group
        # Identical storage to ternary_lloyd_ds: 1.585-bit payload + 2
        # fp16 scales per group - the comparison is exactly matched.
        self.assertAlmostEqual(q.bpw, math.log2(3) + 32 / 128, places=6)
        self.assertAlmostEqual(q.bpw,
                               quantize_ternary_lloyd_ds(self.w).bpw,
                               places=9)

    def test_deterministic(self):
        a = quantize_ternary_1step_ds(self.w)
        b = quantize_ternary_1step_ds(self.w)
        self.assertTrue(np.array_equal(a.codes, b.codes))
        self.assertTrue(np.array_equal(a.scales, b.scales))

    def test_all_zero_input_no_nan(self):
        q = quantize_ternary_1step_ds(np.zeros(512, dtype=np.float32))
        r = q.reconstruct()
        self.assertTrue(np.all(np.isfinite(r)))
        self.assertTrue(np.all(r == 0.0))

    def test_dual_reconstruction_uses_both_scales(self):
        # On skewed weights the fitted scales are asymmetric; the +1
        # codes must decode to s_pos and -1 codes to -s_neg, not to a
        # single shared scale.
        q = quantize_ternary_1step_ds(self.ws)
        r = q.reconstruct()
        self.assertFalse(np.allclose(q.scales[:, 0], q.scales[:, 1]))
        gid = np.arange(q.codes.shape[0]) // 128
        pos = q.codes == 1
        neg = q.codes == -1
        self.assertTrue(pos.any() and neg.any())
        self.assertTrue(np.allclose(r[pos],
                                   q.scales[gid[pos], 0].astype(np.float32)))
        self.assertTrue(np.allclose(r[neg],
                                   -q.scales[gid[neg], 1].astype(np.float32)))
        self.assertTrue(np.all(r[q.codes == 0] == 0.0))

    def test_one_step_captures_partially(self):
        # Seed 7 measures 0.83 clean / 0.73 skewed; assert >= 0.50 for
        # margin against float noise and seed variance (seed 9 skew 0.5
        # is the hardest measured at 0.57).
        self.assertGreaterEqual(self._capture(self.w, 1), 0.50)
        self.assertGreaterEqual(self._capture(self.ws, 1), 0.50)

    def test_two_steps_capture_most(self):
        # Two fixed iterations get within striking distance of full
        # convergence on seed 7: 0.96 clean / 0.84 skewed measured.
        # Assert >= 0.80 so the test has margin; the dual practical
        # encoder is 2-3 iterations, still O(1).
        self.assertGreaterEqual(self._capture(self.w, 2), 0.80)
        self.assertGreaterEqual(self._capture(self.ws, 2), 0.80)

    def test_iteration_count_increases_capture_on_hard_seed(self):
        # Seed 9 skew 0.5 is the slowest-converging case measured
        # (n1=0.57, n3=0.76, n5=1.00): more iterations must not hurt and
        # must eventually reach the full win.
        rng = np.random.default_rng(9)
        w9 = synthetic_weights(rng, n_groups=64, skew=0.5)
        c1 = self._capture(w9, 1)
        c3 = self._capture(w9, 3)
        self.assertGreaterEqual(c3, c1)
        self.assertGreaterEqual(self._capture(w9, 5), 0.95)


def _write_safetensors(path, tensors):
    """Write a minimal .safetensors file (test helper, NumPy only)."""
    header, blobs, offset = {}, [], 0
    dtype_names = {np.dtype("float32"): "F32", np.dtype("float16"): "F16"}
    for name, arr in tensors.items():
        blob = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dtype_names[arr.dtype],
                        "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
        blobs.append(blob)
    hb = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


class TestRealWeights(unittest.TestCase):
    def test_safetensors_roundtrip(self):
        rng = np.random.default_rng(3)
        want = {
            "h.0.attn.c_attn.weight": rng.standard_normal((8, 16)).astype(np.float32),
            "h.0.ln_1.weight": rng.standard_normal(8).astype(np.float16),
        }
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.safetensors")
            _write_safetensors(p, want)
            got = read_safetensors(p)
        self.assertEqual(set(got), set(want))
        for k in want:
            self.assertEqual(got[k].shape, want[k].shape)
            self.assertEqual(got[k].dtype, want[k].dtype)
            np.testing.assert_array_equal(got[k], want[k])

    def test_safetensors_rejects_unsupported_dtype(self):
        header = {"w": {"dtype": "BF16", "shape": [4],
                        "data_offsets": [0, 8]}}
        hb = json.dumps(header).encode("utf-8")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.safetensors")
            with open(p, "wb") as f:
                f.write(struct.pack("<Q", len(hb)))
                f.write(hb)
                f.write(b"\x00" * 8)
            with self.assertRaises(ValueError):
                read_safetensors(p)

    def test_linear_weight_tensors_filters(self):
        fake = {
            "h.0.attn.c_attn.weight": np.zeros((8, 8), np.float32),
            "h.0.attn.c_attn.bias": np.zeros(8, np.float32),
            "h.0.ln_1.weight": np.zeros(8, np.float32),
            "transformer.wte.weight": np.zeros((16, 8), np.float32),
            "lm_head.weight": np.zeros((16, 8), np.float32),
        }
        got = linear_weight_tensors(fake)
        self.assertEqual(list(got), ["h.0.attn.c_attn.weight",
                                     "lm_head.weight"])

    def test_sample_groups_deterministic_and_aligned(self):
        rng = np.random.default_rng(11)
        t = rng.standard_normal((64, 128)).astype(np.float32)
        a = sample_groups(t, 5, 128, np.random.default_rng(5))
        b = sample_groups(t, 5, 128, np.random.default_rng(5))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (5 * 128,))
        # every sampled block is a contiguous slice of the flattened tensor
        flat = t.ravel()
        for i in range(5):
            block = a[i * 128:(i + 1) * 128]
            starts = np.flatnonzero((flat[:len(flat) - 127] == block[0]))
            self.assertTrue(
                any(np.array_equal(flat[s:s + 128], block) for s in starts),
                "sampled block not found as a contiguous slice")

    def test_sample_groups_clamps_to_available_blocks(self):
        t = np.zeros((2, 128), np.float32)
        got = sample_groups(t, 10, 128, np.random.default_rng(1))
        self.assertEqual(got.shape, (2 * 128,))

    def test_rank_on_real_weights_smoke(self):
        rng = np.random.default_rng(13)
        mats = {
            "a.weight": synthetic_weights(rng, n_groups=4),
            "b.weight": synthetic_weights(rng, n_groups=4),
        }
        # reshape to 2-D so they look like real matrices
        mats = {k: v.reshape(4, 128) for k, v in mats.items()}
        schemes = {"ternary_uniform": quantize_ternary_uniform,
                   "int2_kmeans_q8": quantize_int2_kmeans_q8}
        res = rank_on_real_weights(mats, n_groups_per_matrix=4, seed=7,
                                   schemes=schemes)
        self.assertEqual([r["scheme"] for r in res],
                         sorted([r["scheme"] for r in res],
                                key=lambda n: next(
                                    x["sqnr_db"] for x in res
                                    if x["scheme"] == n),
                                reverse=True))
        self.assertTrue(all(r["n_matrices"] == 2 for r in res))
        # aggregate equals the mean of direct per-matrix SQNR
        for r in res:
            direct = []
            rng2 = np.random.default_rng(7)
            for name in ("a.weight", "b.weight"):
                w = sample_groups(mats[name], 4, 128, rng2)
                q = schemes[r["scheme"]](w)
                direct.append(sqnr_db(w, q.reconstruct()))
            self.assertAlmostEqual(r["sqnr_db"], sum(direct) / 2, places=9)

    def test_parse_args_group_size(self):
        args = parse_args(["model.safetensors"])
        self.assertEqual(args.group_size, 128)
        args = parse_args(["model.safetensors", "--group-size", "64"])
        self.assertEqual(args.group_size, 64)
        args = parse_args(["model.safetensors", "--group-size", "256",
                           "--seed", "9"])
        self.assertEqual((args.group_size, args.seed), (256, 9))

    def test_rank_group_size_sensitivity_smoke(self):
        # The ranking must stay Lloyd-first across group sizes (the
        # g64/g256 real-weight runs answer whether it holds exactly;
        # here we pin the harness behavior on synthetic data).
        rng = np.random.default_rng(13)
        mats = {f"{k}.weight": synthetic_weights(rng, n_groups=8).reshape(4, 256)
                for k in ("a", "b")}
        schemes = {"ternary_uniform": quantize_ternary_uniform,
                   "ternary_1step": quantize_ternary_1step,
                   "int2_kmeans_q8": quantize_int2_kmeans_q8}
        orders = {}
        for gs in (64, 128, 256):
            res = rank_on_real_weights(mats, n_groups_per_matrix=4, seed=7,
                                       group_size=gs, schemes=schemes)
            orders[gs] = [r["scheme"] for r in res]
            # stored bpw must track group size: 2.0 + 48/gs for kmeans_q8
            got = {r["scheme"]: r["bpw"] for r in res}
            self.assertAlmostEqual(got["int2_kmeans_q8"], 2.0 + 48 / gs,
                                   places=9)
        self.assertEqual(orders[64], orders[128])
        self.assertEqual(orders[128], orders[256])
        # classical Lloyd reference on top at every group size
        for order in orders.values():
            self.assertEqual(order[0], "int2_kmeans_q8")


if __name__ == "__main__":
    unittest.main()