"""Tests for the quant R&D prototypes (avenue H): baselines + candidates."""
import json
import math
import os
import struct
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quant_rnd import (
    SCHEMES,
    quantize_dual_scale_ternary,
    quantize_int2_kmeans,
    quantize_int2_kmeans_q8,
    quantize_int2_symmetric,
    quantize_int4_uniform,
    quantize_int8_uniform,
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
from src.quant_rnd.gpt2_tokenizer import GPT2Tokenizer
from src.quant_rnd.gpt2_forward import (
    GPT2,
    attention,
    gelu,
    layer_norm,
    load_gpt2,
)
from src.quant_rnd.opcount import (
    M1_PRO_MEM_BW_GBS,
    N_PARAMS_70B,
    codebook_opcount,
    decode_roofline_tps,
    equiv_adds,
    is_bandwidth_bound,
    kv_cache_bytes,
    measured_sparsity,
    prefill_crossover_L,
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
                          "ternary_1step", "ternary_1step_sp",
                          "ternary_1step_ds",
                          "int2_symmetric", "int8_uniform", "int4_uniform",
                          "int2_kmeans",
                          "int2_kmeans_q8",
                          "int2_outlier_retain", "dual_scale_ternary",
                          "ternary_outlier"})

    def test_ternary_codes_valid(self):
        q = quantize_ternary_uniform(self.w)
        self.assertTrue(set(np.unique(q.codes)) <= {-1, 0, 1})

    def test_int2_codes_valid(self):
        q = quantize_int2_symmetric(self.w)
        self.assertTrue(set(np.unique(q.codes)) <= {-3, -1, 1, 3})

    def test_int8_codes_valid(self):
        q = quantize_int8_uniform(self.w)
        self.assertEqual(q.codes.dtype, np.int8)
        self.assertTrue(np.all(q.codes >= -127))
        self.assertTrue(np.all(q.codes <= 127))

    def test_int8_bpw_accounting(self):
        # 8-bit payload + one fp16 scale per group; overhead halves
        # when the group size doubles.
        q = quantize_int8_uniform(self.w)
        self.assertAlmostEqual(q.bpw, 8.0 + 16 / 128, places=6)
        q64 = quantize_int8_uniform(self.w, group_size=64)
        self.assertAlmostEqual(q64.bpw, 8.0 + 16 / 64, places=6)

    def test_int8_multiplicative_decode(self):
        # int8_uniform is neither a codebook nor a dual-scale scheme:
        # decode is the plain codes * scale path, pinned against a
        # manual recomputation so a mis-registration can't silently
        # re-decode through the wrong branch.
        q = quantize_int8_uniform(self.w)
        n = self.w.shape[0]
        group_id = np.arange(n) // 128
        manual = q.codes.astype(np.float32) * q.scales[group_id, 0]
        np.testing.assert_array_equal(q.reconstruct(), manual)

    def test_int4_codes_valid(self):
        q = quantize_int4_uniform(self.w)
        self.assertEqual(q.codes.dtype, np.int8)
        self.assertTrue(np.all(q.codes >= -7))
        self.assertTrue(np.all(q.codes <= 7))

    def test_int4_bpw_accounting(self):
        # 4-bit payload + one fp16 scale per group; overhead halves
        # when the group size doubles.
        q = quantize_int4_uniform(self.w)
        self.assertAlmostEqual(q.bpw, 4.0 + 16 / 128, places=6)
        q64 = quantize_int4_uniform(self.w, group_size=64)
        self.assertAlmostEqual(q64.bpw, 4.0 + 16 / 64, places=6)

    def test_int4_multiplicative_decode(self):
        # int4_uniform is neither a codebook nor a dual-scale scheme:
        # decode is the plain codes * scale path, pinned against a
        # manual recomputation so a mis-registration can't silently
        # re-decode through the wrong branch.
        q = quantize_int4_uniform(self.w)
        n = self.w.shape[0]
        group_id = np.arange(n) // 128
        manual = q.codes.astype(np.float32) * q.scales[group_id, 0]
        np.testing.assert_array_equal(q.reconstruct(), manual)

    def test_int4_deterministic_and_near_lossless(self):
        q1 = quantize_int4_uniform(self.w)
        q2 = quantize_int4_uniform(self.w)
        np.testing.assert_array_equal(q1.codes, q2.codes)
        rel = np.linalg.norm(q1.reconstruct() - self.w) / np.linalg.norm(self.w)
        # q4 on synthetic weights: error is ~7.7x the q8 reference
        # (0.154 vs 0.02), as expected for 7 vs 127 levels. The bound
        # pins the encoder's behavior on this distribution; if it ever
        # fails the encoder changed, not the tolerance.
        self.assertLess(rel, 0.20)

    def test_int8_deterministic_and_near_lossless(self):
        q1 = quantize_int8_uniform(self.w)
        q2 = quantize_int8_uniform(self.w)
        np.testing.assert_array_equal(q1.codes, q2.codes)
        rel = np.linalg.norm(q1.reconstruct() - self.w) / np.linalg.norm(self.w)
        # q8 on synthetic weights is near-lossless; if this ever fails
        # the encoder changed, not the tolerance.
        self.assertLess(rel, 0.02)

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


class TestReconstructVectorized(unittest.TestCase):
    """Regression: reconstruct() was a per-group boolean-mask loop --
    O(n^2/group_size), ~25 s per 2M params, which made full-model
    dequantization take ~18 min per scheme and killed the step-2b sweep.
    It is now vectorized O(n) (~25 ms per 2M). These tests pin the new
    code to the old per-group semantics and guard the model-scale runtime.
    """

    @staticmethod
    def _naive_reconstruct(q):
        # The pre-fix algorithm, written out as an independent reference.
        n = q.codes.shape[0]
        g = q.group_size
        out = np.zeros(n, dtype=np.float32)
        group_id = np.arange(n) // g
        for gi in range((n + g - 1) // g):
            mask = group_id == gi
            c = q.codes[mask].astype(np.float32)
            s = q.scales[gi]
            if q.name in ("dual_scale_ternary", "ternary_lloyd_ds",
                          "ternary_1step_ds"):
                rec = np.where(c > 0, s[0], np.where(c < 0, -s[1], 0.0))
            elif q.name in ("int2_kmeans", "int2_kmeans_q8"):
                rec = s[c.astype(np.int64)]
            else:
                rec = c * s[0]
            out[mask] = rec
        if q._outlier_vals is not None:
            out[q._outlier_idx] = q._outlier_vals
        return out

    def test_matches_naive_loop_all_decode_branches(self):
        # Non-multiple-of-group-size length exercises the trailing
        # partial group, the edge case the vectorized indexing must get
        # right.
        rng = np.random.default_rng(1234)
        w = (rng.standard_normal(3000) * 0.02).astype(np.float32)
        cases = [
            ("ternary_1step", {}),        # single-scale branch
            ("ternary_1step_ds", {}),     # dual-scale branch
            ("int2_kmeans_q8", {}),       # codebook branch
            ("ternary_outlier", {"n_outliers": 2}),  # + exact outliers
        ]
        for name, kw in cases:
            q = SCHEMES[name](w, group_size=128, **kw)
            np.testing.assert_array_equal(q.reconstruct(),
                                          self._naive_reconstruct(q),
                                          err_msg=name)

    def test_reconstruct_model_scale_is_fast(self):
        # Generous bound: pre-fix took ~25 s here; fixed takes ~25 ms.
        rng = np.random.default_rng(7)
        w = (rng.standard_normal(2_000_000) * 0.02).astype(np.float32)
        q = SCHEMES["ternary_1step"](w, group_size=128)
        t0 = time.time()
        r = q.reconstruct()
        self.assertLess(time.time() - t0, 10.0)
        self.assertEqual(r.shape, w.shape)
        self.assertTrue(np.all(np.isfinite(r)))


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
            # Family spans the sub-2-bit candidates up to the int8_uniform
            # q8 ceiling reference (8.125 bpw @ g128).
            self.assertLess(r["bpw"], 9.0)

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
        # Compute-bound -> the ceiling ratio is the ratio of TOTAL
        # flops/token, i.e. (matmul flops + attention flops)/token.
        # Attention is scheme-independent, so the quant ratio is
        # compressed toward 1 vs the matmul-only figure (exactly the
        # effect this term was added to capture); include_attention=False
        # recovers the exact per-weight ratio pinned below.
        f_t = oc_t["adds"] + oc_t["muls"]
        f_k = oc_k["adds"] + oc_k["muls"]
        attn_per_token = p4_t["attention_flops"] / 4096
        self.assertAlmostEqual(
            p4_t["ceil_tps"] / p4_k["ceil_tps"],
            (N_PARAMS_70B * f_k + attn_per_token)
            / (N_PARAMS_70B * f_t + attn_per_token),
            delta=1e-6)
        # Sanity: the compute-bound ceiling reduces to peak/flops_per_token
        # including the attention term.
        self.assertAlmostEqual(p4_t["ceil_tps"],
                               peak / (N_PARAMS_70B * f_t + attn_per_token),
                               delta=1.0)
        # include_attention=False recovers the exact matmul-only ratio.
        q4_t = prefill_roofline_tps(bpw=q1.bpw, opcount=oc_t,
                                    prompt_len=4096, include_attention=False,
                                    **kw)
        q4_k = prefill_roofline_tps(bpw=qk.bpw, opcount=oc_k,
                                    prompt_len=4096, include_attention=False,
                                    **kw)
        ratio = q4_t["ceil_tps"] / q4_k["ceil_tps"]
        self.assertAlmostEqual(ratio, f_k / f_t, delta=1e-6)
        self.assertGreater(ratio, 1.5)
        self.assertAlmostEqual(q4_t["ceil_tps"],
                               peak / (N_PARAMS_70B * f_t), delta=1.0)
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

    def test_prefill_attention_term_math(self):
        """The O(L^2) attention term: pinned FLOPs formula, KV-read byte
        accounting, prompt-length scaling, and the long-context regime
        verdict the term was added to model."""
        kw = dict(n_params=N_PARAMS_70B, bpw=2.0,
                  opcount={"adds": 0.5, "muls": 0.01},
                  bandwidth_gbs=M1_PRO_MEM_BW_GBS, peak_flops=5.2e12,
                  n_layers=80, n_kv_heads=8, head_dim=128)
        L = 32768
        p = prefill_roofline_tps(prompt_len=L, **kw)
        # Q@K^T + attn@V: 2 matmuls x (2 FLOPs/MAC) x heads x L^2 x d_head.
        expect_attn = 4.0 * 80 * 64 * L * L * 128
        self.assertAlmostEqual(p["attention_flops"], expect_attn, delta=1.0)
        self.assertAlmostEqual(p["total_flops"],
                               N_PARAMS_70B * 0.51 * L + expect_attn,
                               delta=1.0)
        # KV is read once during attention on top of the prompt KV write:
        # KV traffic doubles relative to the matmul-only model.
        no_attn = prefill_roofline_tps(prompt_len=L, include_attention=False,
                                       **kw)
        kv_write = 2.0 * 80 * 8 * 128 * L * 2
        self.assertAlmostEqual(p["bytes_moved"] - no_attn["bytes_moved"],
                               kv_write, delta=1.0)
        # Term disabled -> zero attention FLOPs and exact matmul-only
        # numerics (byte traffic = weight + act + KV write only).
        self.assertEqual(no_attn["attention_flops"], 0.0)
        # At 32k the attention term dominates the matmuls for a 70B-shaped
        # model -- the documented reason the term matters at long context.
        self.assertGreater(p["attention_flops"],
                           N_PARAMS_70B * 0.51 * L)
        # Regime stays compute-bound at 32k; the ceiling is honest about
        # attention dominating the quant-scheme difference.
        self.assertFalse(p["bandwidth_bound"])
        # GQA scaling: doubling query heads doubles the attention FLOPs;
        # the L^2 scaling is exact across prompt lengths.
        p2 = prefill_roofline_tps(prompt_len=L, n_q_heads=128, **kw)
        self.assertAlmostEqual(p2["attention_flops"],
                               2.0 * p["attention_flops"], delta=1.0)
        p4 = prefill_roofline_tps(prompt_len=16384, **kw)
        self.assertAlmostEqual(p["attention_flops"],
                               4.0 * p4["attention_flops"], delta=1.0)
        with self.assertRaises(ValueError):
            prefill_roofline_tps(prompt_len=L, n_q_heads=0, **kw)

    def test_prefill_crossover_L_reference_pair(self):
        # The pinned crossover for the reference pair at 70B scale /
        # 200 GB/s / 5.2 TFLOPS peak (M1 Pro GPU published figure):
        # ternary_1step (1.710 bpw, measured sparsity 0.41) vs
        # int2_kmeans_q8 (2.375 bpw, histogram dequant). The prefill
        # ceiling ratio compresses from 1.63x at 4k toward 1 as the
        # scheme-independent attention O(L^2) term swamps the matmul
        # difference, dropping below 1.2x between 32k and 64k.
        op_t = ternary_opcount(0.41, n_scales=1, group_size=128)
        op_k = codebook_opcount(128, n_centroids=4, n_scales=1,
                               method="histogram")
        kw = dict(n_params=N_PARAMS_70B, bandwidth_gbs=M1_PRO_MEM_BW_GBS,
                  peak_flops=5.2e12, n_layers=80, n_q_heads=64,
                  n_kv_heads=8, head_dim=128)
        r = prefill_crossover_L(op_t, 1.710, op_k, 2.375, **kw)
        self.assertEqual(r["crossover_L"], 65536)
        self.assertFalse(r["below_at_start"])
        self.assertEqual(r["prompt_lens"],
                         [4096, 8192, 16384, 32768, 65536, 131072])
        ratios = r["ratios"]
        self.assertEqual(len(ratios), 6)
        # The ratio compresses monotonically toward 1 with context.
        for a, b in zip(ratios, ratios[1:]):
            self.assertLess(b, a)
        self.assertAlmostEqual(ratios[0], 1.629, delta=0.01)
        self.assertAlmostEqual(ratios[3], 1.259, delta=0.01)
        self.assertAlmostEqual(ratios[5], 1.086, delta=0.01)
        self.assertGreater(ratios[-1], 1.0)
        # Per-L ceilings are sane (monotone, same length as lens).
        self.assertEqual(len(r["ceil_a_tps"]), 6)
        self.assertEqual(len(r["ceil_b_tps"]), 6)
        # Prefill tok/s falls with L: attention O(L^2) FLOPs and KV/act
        # traffic grow per token while weight bytes are amortized once.
        self.assertGreater(r["ceil_a_tps"][0], r["ceil_a_tps"][1])

    def test_prefill_crossover_L_edges(self):
        op_t = ternary_opcount(0.41, n_scales=1, group_size=128)
        op_k = codebook_opcount(128, n_centroids=4, n_scales=1,
                               method="histogram")
        kw = dict(n_params=N_PARAMS_70B, bandwidth_gbs=M1_PRO_MEM_BW_GBS,
                  peak_flops=5.2e12)
        # threshold 1.0: A is strictly cheaper in FLOPs and bytes, so
        # the ratio can never dip below 1 -> no crossover.
        r = prefill_crossover_L(op_t, 1.710, op_k, 2.375, threshold=1.0,
                               **kw)
        self.assertIsNone(r["crossover_L"])
        self.assertFalse(r["below_at_start"])
        # Already below at the first swept length: crossover reports
        # that length and flags below_at_start.
        r = prefill_crossover_L(op_t, 1.710, op_k, 2.375,
                               prompt_lens=(131072,), threshold=1.2, **kw)
        self.assertTrue(r["below_at_start"])
        self.assertEqual(r["crossover_L"], 131072)
        # Custom thresholds move the crossover: 1.5x is crossed earlier.
        r = prefill_crossover_L(op_t, 1.710, op_k, 2.375, threshold=1.5,
                               **kw)
        self.assertEqual(r["crossover_L"], 16384)
        # Validation.
        with self.assertRaises(ValueError):
            prefill_crossover_L(op_t, 1.710, op_k, 2.375, threshold=0.0,
                               **kw)
        with self.assertRaises(ValueError):
            prefill_crossover_L(op_t, 1.710, op_k, 2.375, prompt_lens=(),
                               **kw)
        with self.assertRaises(ValueError):
            prefill_crossover_L(op_t, 1.710, op_k, 2.375,
                               prompt_lens=(8192, 4096), **kw)
        with self.assertRaises(ValueError):
            prefill_crossover_L(op_t, 1.710, op_k, 2.375,
                               prompt_lens=(4096, 4096), **kw)


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


class TestThresholdBiasedLloyd(unittest.TestCase):
    """Threshold-widening (thresh_factor) for the zero-rate sparsity probe.

    Raises ternary_1step's zero bin (thresholds at +/-k*s/2) with the
    fit staying L2-optimal under the widened thresholds - the honest
    sparse encoder for the compute-side (add-skip) story.
    """

    def setUp(self):
        rng = np.random.default_rng(7)
        self.w = synthetic_weights(rng, n_groups=64)
        wp, _ng, _n = _groups(self.w)
        self.g = wp[0]

    def zero_rate(self, w, k):
        return float(np.mean(quantize_ternary_1step(w, thresh_factor=k).codes == 0))

    def test_default_factor_is_bit_identical(self):
        a = _ternary_lloyd_fit(self.g, dual=False)
        b = _ternary_lloyd_fit(self.g, dual=False, thresh_factor=1.0)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])
        np.testing.assert_array_equal(a[2], b[2])
        qa = quantize_ternary_1step(self.w)
        qb = quantize_ternary_1step(self.w, thresh_factor=1.0)
        np.testing.assert_array_equal(qa.codes, qb.codes)
        np.testing.assert_array_equal(qa.scales, qb.scales)

    def test_widening_raises_zero_rate_monotonically(self):
        zr_10 = self.zero_rate(self.w, 1.0)
        zr_12 = self.zero_rate(self.w, 1.2)
        zr_20 = self.zero_rate(self.w, 2.0)
        self.assertLessEqual(zr_10, zr_12)
        self.assertLessEqual(zr_12, zr_20)
        self.assertGreater(zr_20, zr_10)  # strictly more sparse somewhere

    def test_widened_decode_stays_symmetric_ternary(self):
        # The sparse variant is still {-s, 0, +s} per group: one scale,
        # codes in {-1, 0, 1}, reconstruct decodes the symmetric branch.
        q = quantize_ternary_1step(self.w, thresh_factor=1.2)
        self.assertEqual(q.name, "ternary_1step")
        self.assertTrue(set(np.unique(q.codes)) <= {-1, 0, 1})
        self.assertEqual(q.scales.shape[1], 1)
        r = q.reconstruct()
        n, g = q.codes.shape[0], q.group_size
        group_id = np.arange(n) // g
        np.testing.assert_allclose(
            r, q.codes.astype(np.float32) * q.scales[group_id, 0],
            rtol=1e-6)

    def test_sparse_scheme_registered_and_matched_bitrate(self):
        fn = SCHEMES["ternary_1step_sp"]
        q = fn(self.w, 128)
        # Matched 1.710 bpw: widening moves zero-rate, not bitrate.
        self.assertAlmostEqual(q.bpw, math.log2(3) + 16 / 128, places=6)
        self.assertAlmostEqual(q.bpw,
                               quantize_ternary_1step(self.w).bpw,
                               places=9)
        zr_sp = float(np.mean(q.codes == 0))
        zr_1 = float(np.mean(quantize_ternary_1step(self.w).codes == 0))
        self.assertGreater(zr_sp, zr_1)
        # Seed 7 synthetic: factor 1.2 lands in the 0.45-0.60 band
        # (0.514 on real GPT-2 weights); assert it is genuinely sparser
        # without claiming the exact real-weight number here.
        self.assertGreater(zr_sp, 0.44)

    def test_all_zero_input_still_safe(self):
        q = quantize_ternary_1step(np.zeros(512, dtype=np.float32),
                                   thresh_factor=2.0)
        r = q.reconstruct()
        self.assertTrue(np.all(np.isfinite(r)))
        self.assertTrue(np.all(r == 0.0))


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


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_DIR = os.path.join(REPO_ROOT, "research", "data", "tokenizer")
GPT2_WEIGHTS = os.path.join(REPO_ROOT, "research", "data", "gpt2.safetensors")
EVAL_TEXT = os.path.join(REPO_ROOT, "research", "data", "eval_text.txt")


def _read_eval_text() -> str:
    with open(EVAL_TEXT, encoding="utf-8") as f:
        return f.read()


class TestGPT2Tokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(TOKENIZER_DIR):
            raise unittest.SkipTest("tokenizer data not present")

    def _tok(self):
        return GPT2Tokenizer(TOKENIZER_DIR)

    def test_roundtrip_ascii_prose(self):
        tok = self._tok()
        text = _read_eval_text()
        self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_roundtrip_tricky_ascii(self):
        tok = self._tok()
        text = "Don't stop: 3.14 is pi-ish (well, roughly).\nNew\tlines\r\n too!"
        self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_roundtrip_non_ascii_falls_back_to_bytes(self):
        # The pre-tokenizer pattern is ASCII-safe; non-ASCII chars must
        # still round-trip via the byte-level fallback, never crash.
        tok = self._tok()
        text = "caf\u00e9 na\u00efve \u4e2d\u6587"
        self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_deterministic(self):
        tok = self._tok()
        text = "The quick brown fox jumps over 13 lazy dogs."
        self.assertEqual(tok.encode(text), tok.encode(text))

    def test_ids_in_vocab_range(self):
        tok = self._tok()
        ids = tok.encode(_read_eval_text())
        self.assertTrue(len(ids) > 300)  # a few hundred tokens of text
        self.assertTrue(all(0 <= i < tok.vocab_size for i in ids))

    def test_bpe_goldens_from_merge_table(self):
        # ' t' and 'he' are merge products early in merges.txt
        # ("\u0120 t" is rank 0, "h e" is rank 2), so each must encode
        # to a single token id. Derives goldens from the table itself.
        tok = self._tok()
        self.assertEqual(len(tok.encode(" t")), 1)
        self.assertEqual(len(tok.encode("he")), 1)
        self.assertEqual(tok.encode(" t")[0], tok.encoder["\u0120t"])
        self.assertEqual(tok.encode("he")[0], tok.encoder["he"])

    def test_contraction_splits_like_gpt2(self):
        # GPT-2's pre-tokenizer splits "don't" into "don" + "'t".
        tok = self._tok()
        ids = tok.encode("don't")
        self.assertEqual(len(ids), 2)
        self.assertEqual(tok.decode(ids), "don't")


class TestGPT2ForwardMath(unittest.TestCase):
    def test_gelu_matches_erf_definition(self):
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(2000) * 3).astype(np.float32)
        ref = 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
        np.testing.assert_allclose(gelu(x), ref, rtol=1e-5, atol=1e-6)

    def test_layer_norm_matches_definition(self):
        rng = np.random.default_rng(1)
        x = (rng.standard_normal((5, 32)) * 2 + 1).astype(np.float32)
        w = rng.standard_normal(32).astype(np.float32)
        b = rng.standard_normal(32).astype(np.float32)
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        ref = (x - mu) / np.sqrt(var + 1e-5) * w + b
        np.testing.assert_allclose(layer_norm(x, w, b), ref,
                                   rtol=1e-6, atol=1e-6)

    def test_layer_norm_unit_output_stats(self):
        rng = np.random.default_rng(2)
        x = (rng.standard_normal((7, 64)) * 5 - 3).astype(np.float32)
        w = np.ones(64, dtype=np.float32)
        b = np.zeros(64, dtype=np.float32)
        y = layer_norm(x, w, b)
        np.testing.assert_allclose(y.mean(-1), np.zeros(7), atol=1e-5)
        np.testing.assert_allclose(y.var(-1), np.ones(7), atol=1e-5)

    def test_attention_matches_naive_reference(self):
        # Independent triple-loop causal attention vs the vectorized path.
        rng = np.random.default_rng(3)
        t, c, n_head = 6, 8, 2
        hd = c // n_head
        x = rng.standard_normal((t, c)).astype(np.float32)
        w_qkv = rng.standard_normal((c, 3 * c)).astype(np.float32) * 0.3
        b_qkv = rng.standard_normal(3 * c).astype(np.float32) * 0.1
        w_proj = rng.standard_normal((c, c)).astype(np.float32) * 0.3
        b_proj = rng.standard_normal(c).astype(np.float32) * 0.1
        got = attention(x, w_qkv, b_qkv, w_proj, b_proj, n_head)

        qkv = x @ w_qkv + b_qkv
        q = qkv[:, :c].reshape(t, n_head, hd)
        k = qkv[:, c:2 * c].reshape(t, n_head, hd)
        v = qkv[:, 2 * c:].reshape(t, n_head, hd)
        out = np.zeros((t, n_head, hd), dtype=np.float32)
        for h in range(n_head):
            for i in range(t):
                s = np.array([q[i, h] @ k[j, h] / math.sqrt(hd)
                              for j in range(i + 1)])
                e = np.exp(s - s.max())
                a = e / e.sum()
                out[i, h] = sum(a[j] * v[j, h] for j in range(i + 1))
        ref = out.reshape(t, c) @ w_proj + b_proj
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


class TestQuantizeLayersObq(unittest.TestCase):
    """Multi-layer OBQ helper (obq.py) on synthetic activations.

    No model forward needed: activations are synthetic (T, d_in) arrays,
    so the whole loop is fast and unit-testable.
    """

    @staticmethod
    def _linear():
        rng = np.random.default_rng(7)
        return {
            "h.0.mlp.c_fc.weight":
                rng.standard_normal((64, 32)).astype(np.float32),
            "h.0.attn.c_proj.weight":
                (0.1 * rng.standard_normal((32, 64))).astype(np.float32),
        }

    @staticmethod
    def _inputs(linear):
        rng = np.random.default_rng(11)
        return {n: rng.standard_normal((48, w.shape[0])).astype(np.float32)
                for n, w in linear.items()}

    def test_all_layers_shapes_and_input_untouched(self):
        lin = self._linear()
        before = {k: v.copy() for k, v in lin.items()}
        out, bpw = quantize_layers_obq(lin, self._inputs(lin),
                                       group_size=16)
        self.assertEqual(set(out), set(lin))
        for k in lin:
            self.assertEqual(out[k].shape, lin[k].shape)
            np.testing.assert_array_equal(lin[k], before[k])
            self.assertTrue(np.all(np.isfinite(out[k])), k)
            self.assertFalse(np.allclose(out[k], lin[k]), k)
        self.assertAlmostEqual(bpw, 2.0 + 4 * 8 / 16 + 16 / 16, places=9)

    def test_single_target_equals_direct_call(self):
        # Plumbing pin: the loop must not change the per-layer math vs
        # quantize_layer_obq.
        lin = self._linear()
        inp = self._inputs(lin)
        out, _ = quantize_layers_obq(lin, inp, group_size=16,
                                     only_names={"h.0.mlp.c_fc.weight"})
        self.assertEqual(set(out), {"h.0.mlp.c_fc.weight"})
        direct, _ = quantize_layer_obq(lin["h.0.mlp.c_fc.weight"],
                                       inp["h.0.mlp.c_fc.weight"],
                                       group_size=16)
        np.testing.assert_array_equal(out["h.0.mlp.c_fc.weight"], direct)

    def test_unknown_only_names_raises(self):
        lin = self._linear()
        with self.assertRaises(KeyError):
            quantize_layers_obq(lin, self._inputs(lin), group_size=16,
                                only_names={"nope.weight"})

    def test_missing_inputs_raises(self):
        lin = self._linear()
        inp = {n: a for n, a in self._inputs(lin).items()
               if n != "h.0.attn.c_proj.weight"}
        with self.assertRaises(KeyError):
            quantize_layers_obq(lin, inp, group_size=16)

    def test_empty_only_names_raises(self):
        lin = self._linear()
        with self.assertRaises(ValueError):
            quantize_layers_obq(lin, self._inputs(lin), group_size=16,
                                only_names=set())


@unittest.skipUnless(os.path.exists(GPT2_WEIGHTS), "gpt2 weights not present")
class TestGPT2RealWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_gpt2(GPT2_WEIGHTS)
        cls.tok = GPT2Tokenizer(TOKENIZER_DIR)
        cls.ids = cls.tok.encode(_read_eval_text())

    def test_config_matches_gpt2_124m(self):
        m = self.model
        self.assertEqual((m.n_layer, m.n_embd, m.n_head), (12, 768, 12))
        self.assertEqual((m.vocab_size, m.n_ctx), (50257, 1024))

    def test_forward_shape_and_finite(self):
        logits = self.model.forward(self.ids[:16])
        self.assertEqual(logits.shape, (16, 50257))
        self.assertTrue(np.all(np.isfinite(logits)))

    def test_forward_deterministic(self):
        a = self.model.forward(self.ids[:16])
        b = self.model.forward(self.ids[:16])
        np.testing.assert_array_equal(a, b)

    def test_causality(self):
        # Changing only the last token must not move earlier logits:
        # logits[i] may only depend on tokens <= i.
        ids_a = self.ids[:16]
        ids_b = list(ids_a)
        ids_b[-1] = (ids_b[-1] + 1) % self.model.vocab_size
        la = self.model.forward(ids_a)
        lb = self.model.forward(ids_b)
        np.testing.assert_allclose(la[:15], lb[:15], rtol=1e-6, atol=1e-6)
        self.assertFalse(np.allclose(la[15], lb[15]))

    def test_forward_rejects_overlong_sequence(self):
        with self.assertRaises(ValueError):
            self.model.forward([0] * (self.model.n_ctx + 1))

    def test_fp32_perplexity_sane(self):
        # The end-to-end architecture check: a broken forward pass
        # (wrong mask, wrong LN, wrong gelu, untied head) gives ppl in
        # the thousands; random logits give ~50257. fp32 GPT-2 124M on
        # plain English prose must land well under 500.
        ppl = self.model.perplexity(self.ids[:64])
        self.assertTrue(math.isfinite(ppl))
        self.assertGreater(ppl, 1.0)
        self.assertLess(ppl, 500.0)


from src.quant_rnd.ppl import (
    DEFAULT_SWEEP,
    FP32_SCHEME,
    aggregate_multitext,
    check_corpus_args,
    check_obq_args,
    eval_text_paths,
    is_embedding,
    parse_args as ppl_parse_args,
    main as ppl_main,
    perplexity_of,
    quantize_model,
    quantize_model_obq,
    quantize_one_layer_obq,
    sample_eval_blocks,
)
from src.quant_rnd.obq import quantize_layer_obq, quantize_layers_obq


class TestQuantizeModel(unittest.TestCase):
    """Whole-model quantization plumbing (ppl.py), on a fake weight dict.

    Real-model perplexity numbers live in the research log, not in the
    suite: a full forward costs ~40 s on this VM. These tests cover the
    mechanics; one gated end-to-end test below runs the real model on
    16 tokens.
    """

    @staticmethod
    def _fake_dict():
        rng = np.random.default_rng(7)
        return {
            # linear weights: quantized
            "h.0.mlp.c_fc.weight": rng.standard_normal((64, 32)).astype(np.float32),
            "h.0.attn.c_proj.weight": (0.1 * rng.standard_normal((32, 64))).astype(np.float32),
            # pass-through: embeddings, biases, layernorm
            "wte.weight": rng.standard_normal((50, 64)).astype(np.float32),
            "h.0.mlp.c_fc.bias": rng.standard_normal((32,)).astype(np.float32),
            "h.0.ln_1.weight": np.ones(64, dtype=np.float32),
        }

    def test_keys_shapes_and_input_untouched(self):
        for name in ("ternary_uniform", "ternary_1step", "ternary_1step_ds",
                     "int2_symmetric", "int2_kmeans_q8"):
            fake = self._fake_dict()
            before = {k: v.copy() for k, v in fake.items()}
            out, bpw = quantize_model(fake, name, group_size=32)
            self.assertEqual(set(out), set(fake))
            for k in fake:
                self.assertEqual(out[k].shape, fake[k].shape)
                np.testing.assert_array_equal(fake[k], before[k])
            self.assertTrue(math.isfinite(bpw) and bpw > 0)
            for k, v in out.items():
                self.assertTrue(np.all(np.isfinite(v)), k)

    def test_passthrough_tensors_unchanged(self):
        fake = self._fake_dict()
        out, _ = quantize_model(fake, "ternary_uniform", group_size=32)
        for k in ("wte.weight", "h.0.mlp.c_fc.bias", "h.0.ln_1.weight"):
            np.testing.assert_array_equal(out[k], fake[k])
        # but a linear matrix must actually change under quantization
        self.assertFalse(
            np.allclose(out["h.0.mlp.c_fc.weight"],
                        fake["h.0.mlp.c_fc.weight"]))

    def test_unknown_scheme_raises(self):
        with self.assertRaises(KeyError):
            quantize_model(self._fake_dict(), "nope", group_size=32)

    def test_deterministic(self):
        a, b1 = quantize_model(self._fake_dict(), "ternary_1step", group_size=32)
        b, b2 = quantize_model(self._fake_dict(), "ternary_1step", group_size=32)
        self.assertEqual(b1, b2)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])

    def test_bpw_matches_scheme_bookkeeping(self):
        # group 32 keeps the kmeans_q8 test fast and still exercises the
        # codebook path end to end.
        _, bpw = quantize_model(self._fake_dict(), "int2_kmeans_q8",
                                group_size=32)
        self.assertAlmostEqual(bpw, 2.0 + 4 * 8 / 32 + 16 / 32, places=9)

    def test_only_names_restricts_targets(self):
        fake = self._fake_dict()
        out, _ = quantize_model(fake, "int2_kmeans_q8", group_size=32,
                                only_names={"h.0.mlp.c_fc.weight"})
        # the named layer is quantized...
        self.assertFalse(np.allclose(out["h.0.mlp.c_fc.weight"],
                                     fake["h.0.mlp.c_fc.weight"]))
        # ...the other linear passes through fp32 unchanged...
        np.testing.assert_array_equal(out["h.0.attn.c_proj.weight"],
                                      fake["h.0.attn.c_proj.weight"])
        # ...and so do the non-linear tensors.
        np.testing.assert_array_equal(out["wte.weight"], fake["wte.weight"])

    def test_only_names_unknown_raises(self):
        with self.assertRaises(KeyError):
            quantize_model(self._fake_dict(), "int2_kmeans_q8",
                           group_size=32, only_names={"nope.weight"})

    def test_only_names_embedding_without_embeddings_flag_raises(self):
        # The silent-no-op trap: wte is not a linear target, so
        # only_names={"wte.weight"} alone selects nothing. Must fail
        # fast with a hint, not burn a quantize+forward and crash the
        # printer on bpw=None.
        with self.assertRaises(ValueError):
            quantize_model(self._fake_dict(), "int4_uniform",
                           group_size=32, only_names={"wte.weight"})

    def test_only_names_embedding_with_embeddings_flag_quantizes(self):
        fake = self._fake_dict()
        out, bpw = quantize_model(fake, "int4_uniform", group_size=32,
                                  quantize_embeddings=True,
                                  only_names={"wte.weight"})
        # the named table is quantized...
        self.assertFalse(np.allclose(out["wte.weight"],
                                     fake["wte.weight"]))
        # ...everything else passes through fp32 unchanged...
        np.testing.assert_array_equal(out["h.0.mlp.c_fc.weight"],
                                      fake["h.0.mlp.c_fc.weight"])
        # ...and the reported bpw is the scheme's, not None.
        self.assertAlmostEqual(bpw, 4.0 + 16 / 32, places=6)

    def test_only_names_none_is_default_all_linear(self):
        a, _ = quantize_model(self._fake_dict(), "int2_symmetric",
                              group_size=32, only_names=None)
        b, _ = quantize_model(self._fake_dict(), "int2_symmetric",
                              group_size=32)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])

    def test_one_layer_obq_unknown_layer_raises(self):
        with self.assertRaises(KeyError):
            quantize_one_layer_obq(self._fake_dict(), [1, 2, 3],
                                   "nope.weight")

    def test_cli_one_layer_obq_flags(self):
        args = ppl_parse_args(["m.safetensors", "--one-layer",
                               "h.0.attn.c_attn.weight", "--obq"])
        self.assertEqual(args.one_layer, "h.0.attn.c_attn.weight")
        self.assertTrue(args.obq)
        self.assertAlmostEqual(args.obq_damp, 0.01)
        args = ppl_parse_args(["m.safetensors", "--one-layer",
                               "h.0.attn.c_attn.weight", "--obq",
                               "--obq-damp", "0.05"])
        self.assertAlmostEqual(args.obq_damp, 0.05)
        # defaults: off
        args = ppl_parse_args(["m.safetensors"])
        self.assertIsNone(args.one_layer)
        self.assertFalse(args.obq)

    def test_cli_scheme_parsing(self):
        # parse_args is factored for testability (same pattern as
        # realweights.parse_args); run_sweep itself needs a real model.
        args = ppl_parse_args(["m.safetensors", "--schemes",
                               "ternary_uniform,int2_symmetric"])
        self.assertEqual(args.schemes, "ternary_uniform,int2_symmetric")
        self.assertEqual(args.group_size, 128)

    def test_default_sweep_covers_both_references(self):
        # The practical ternary reference and both naive floors must be in
        # the default sweep; the k-means family is deliberately excluded
        # (see ppl.py module docstring) and runs as a follow-up slice.
        for s in ("ternary_1step", "ternary_uniform", "int2_symmetric"):
            self.assertIn(s, DEFAULT_SWEEP)
        for s in ("int2_kmeans", "int2_kmeans_q8"):
            self.assertNotIn(s, DEFAULT_SWEEP)

    def test_is_embedding_markers(self):
        self.assertTrue(is_embedding("transformer.wte.weight"))
        self.assertTrue(is_embedding("transformer.wpe.weight"))
        self.assertTrue(is_embedding("wte.weight"))
        self.assertFalse(is_embedding("h.0.attn.c_attn.weight"))
        self.assertFalse(is_embedding("h.0.mlp.c_fc.bias"))
        self.assertFalse(is_embedding("h.0.ln_1.weight"))

    def test_quantize_embeddings_flag(self):
        # Ablation protocol: flag ON quantizes wte with the same scheme,
        # biases and LayerNorm still pass through fp32.
        fake = self._fake_dict()
        out, bpw = quantize_model(fake, "int2_symmetric", group_size=32,
                                  quantize_embeddings=True)
        self.assertFalse(
            np.allclose(out["wte.weight"], fake["wte.weight"]),
            "wte must be quantized when the flag is on")
        self.assertEqual(out["wte.weight"].shape, fake["wte.weight"].shape)
        for k in ("h.0.mlp.c_fc.bias", "h.0.ln_1.weight"):
            np.testing.assert_array_equal(out[k], fake[k])
        # bpw bookkeeping is unaffected: same scheme, same group size
        _, bpw_off = quantize_model(fake, "int2_symmetric", group_size=32)
        self.assertEqual(bpw, bpw_off)

    def test_quantize_embeddings_flag_default_off(self):
        # Default behavior is unchanged: wte stays fp32 (the weight-only
        # protocol the published anchors use).
        fake = self._fake_dict()
        out, _ = quantize_model(fake, "int2_symmetric", group_size=32)
        np.testing.assert_array_equal(out["wte.weight"], fake["wte.weight"])

    def test_cli_quantize_embeddings_flag(self):
        args = ppl_parse_args(["m.safetensors"])
        self.assertFalse(args.quantize_embeddings)
        args = ppl_parse_args(["m.safetensors", "--quantize-embeddings"])
        self.assertTrue(args.quantize_embeddings)

    def test_cli_only_names_flag(self):
        args = ppl_parse_args(["m.safetensors"])
        self.assertIsNone(args.only_names)
        args = ppl_parse_args(["m.safetensors", "--only-names",
                               "wte.weight,wpe.weight"])
        self.assertEqual(args.only_names, "wte.weight,wpe.weight")

    def test_one_layer_and_only_names_mutually_exclusive(self):
        # The mutual-exclusion check runs before any model loading, so
        # a bogus model path never gets touched.
        import sys
        old = sys.argv
        sys.argv = ["ppl", "nope.safetensors", "--one-layer", "a.weight",
                    "--only-names", "b.weight"]
        try:
            with self.assertRaises(SystemExit):
                ppl_main()
        finally:
            sys.argv = old

    def test_only_names_rejected_with_obq(self):
        import sys
        old = sys.argv
        sys.argv = ["ppl", "nope.safetensors", "--obq-all",
                    "--only-names", "wte.weight"]
        try:
            with self.assertRaises(SystemExit):
                ppl_main()
        finally:
            sys.argv = old

    def test_only_names_splits_embedding_ablation(self):
        # --only-names + --quantize-embeddings isolates a single table:
        # the wte-vs-wpe split for the embedding-collapse ablation.
        rng = np.random.default_rng(7)
        fake = self._fake_dict()
        fake["wpe.weight"] = rng.standard_normal((8, 64)).astype(np.float32)
        out, _ = quantize_model(fake, "int2_symmetric", group_size=32,
                                quantize_embeddings=True,
                                only_names={"wte.weight"})
        self.assertFalse(np.allclose(out["wte.weight"], fake["wte.weight"]),
                         "wte must be quantized when targeted")
        np.testing.assert_array_equal(out["wpe.weight"], fake["wpe.weight"])
        np.testing.assert_array_equal(out["h.0.mlp.c_fc.weight"],
                                      fake["h.0.mlp.c_fc.weight"])
        out2, _ = quantize_model(fake, "int2_symmetric", group_size=32,
                                 quantize_embeddings=True,
                                 only_names={"wpe.weight"})
        self.assertFalse(np.allclose(out2["wpe.weight"], fake["wpe.weight"]),
                         "wpe must be quantized when targeted")
        np.testing.assert_array_equal(out2["wte.weight"], fake["wte.weight"])

    def test_only_names_unknown_tensor_raises(self):
        with self.assertRaises(KeyError):
            quantize_model(self._fake_dict(), "int2_symmetric",
                           only_names={"nope.weight"})

    @staticmethod
    def _fake_capture():
        # Deterministic synthetic activations matching the fake dict's
        # linears; used to patch fisher.capture_linear_inputs so the
        # full-model OBQ plumbing is testable without a real model.
        def fake_cap(tensors, ids):
            rng = np.random.default_rng(11)
            lin = linear_weight_tensors(tensors)
            return {n: rng.standard_normal((48, w.shape[0])).astype(np.float32)
                    for n, w in lin.items()}
        return fake_cap

    def _patched_obq(self, fake, **kw):
        import src.quant_rnd.fisher as fisher_mod
        orig = fisher_mod.capture_linear_inputs
        fisher_mod.capture_linear_inputs = self._fake_capture()
        try:
            return quantize_model_obq(fake, [1, 2, 3], group_size=16, **kw)
        finally:
            fisher_mod.capture_linear_inputs = orig

    def test_quantize_model_obq_all_layers(self):
        # Every linear is OBQ-quantized off ONE capture forward; the rest
        # passes through fp32. bpw matches the int2_kmeans_q8 bookkeeping.
        fake = self._fake_dict()
        lin = linear_weight_tensors(fake)
        out, bpw = self._patched_obq(fake)
        self.assertEqual(set(out), set(fake))
        for k in fake:
            self.assertEqual(out[k].shape, fake[k].shape)
        for k in lin:
            self.assertFalse(np.allclose(out[k], fake[k]), k)
            self.assertTrue(np.all(np.isfinite(out[k])), k)
        for k in ("wte.weight", "h.0.mlp.c_fc.bias", "h.0.ln_1.weight"):
            np.testing.assert_array_equal(out[k], fake[k])
        self.assertAlmostEqual(bpw, 2.0 + 4 * 8 / 16 + 16 / 16, places=9)

    def test_quantize_model_obq_only_names_restricts(self):
        fake = self._fake_dict()
        out, _ = self._patched_obq(
            fake, only_names={"h.0.mlp.c_fc.weight"})
        self.assertFalse(np.allclose(out["h.0.mlp.c_fc.weight"],
                                     fake["h.0.mlp.c_fc.weight"]))
        np.testing.assert_array_equal(out["h.0.attn.c_proj.weight"],
                                      fake["h.0.attn.c_proj.weight"])

    def test_quantize_model_obq_unknown_only_names_raises(self):
        with self.assertRaises(KeyError):
            self._patched_obq(self._fake_dict(),
                              only_names={"nope.weight"})

    def test_quantize_model_obq_non_linear_only_names_raises(self):
        with self.assertRaises(KeyError):
            self._patched_obq(self._fake_dict(),
                              only_names={"wte.weight"})

    def test_one_layer_obq_delegates_to_model_obq(self):
        # The old entry point must give the same one-layer result as the
        # new shared-capture path, given identical activations.
        fake = self._fake_dict()
        import src.quant_rnd.fisher as fisher_mod
        orig = fisher_mod.capture_linear_inputs
        fisher_mod.capture_linear_inputs = self._fake_capture()
        try:
            a, _ = quantize_one_layer_obq(
                fake, [1, 2, 3], "h.0.mlp.c_fc.weight", group_size=16)
            b, _ = quantize_model_obq(
                fake, [1, 2, 3], group_size=16,
                only_names={"h.0.mlp.c_fc.weight"})
        finally:
            fisher_mod.capture_linear_inputs = orig
        np.testing.assert_array_equal(a["h.0.mlp.c_fc.weight"],
                                      b["h.0.mlp.c_fc.weight"])
        for k in fake:
            if k != "h.0.mlp.c_fc.weight":
                np.testing.assert_array_equal(a[k], b[k])

    def test_cli_obq_all_flag(self):
        args = ppl_parse_args(["m.safetensors"])
        self.assertFalse(args.obq_all)
        args = ppl_parse_args(["m.safetensors", "--obq-all"])
        self.assertTrue(args.obq_all)

    def test_check_obq_args(self):
        check_obq_args(ppl_parse_args(["m.safetensors", "--obq-all"]))
        check_obq_args(ppl_parse_args(["m.safetensors", "--obq",
                                       "--one-layer", "h.0.x"]))
        with self.assertRaises(SystemExit):
            check_obq_args(ppl_parse_args(["m.safetensors", "--obq"]))
        with self.assertRaises(SystemExit):
            check_obq_args(ppl_parse_args(["m.safetensors", "--obq-all",
                                           "--one-layer", "h.0.x"]))
        with self.assertRaises(SystemExit):
            check_obq_args(ppl_parse_args(["m.safetensors", "--obq-all",
                                           "--obq", "--one-layer",
                                           "h.0.x"]))


@unittest.skipUnless(os.path.exists(GPT2_WEIGHTS), "gpt2 weights not present")
class TestMultiTextEval(unittest.TestCase):
    """--eval-texts plumbing, the fp32 pseudo-scheme, and the
    mean/std aggregation (the text-robustness protocol)."""

    def test_cli_eval_texts_parsing(self):
        args = ppl_parse_args(["m.safetensors"])
        self.assertIsNone(args.eval_texts)
        self.assertEqual(args.eval_text, "research/data/eval_text.txt")
        args = ppl_parse_args(["m.safetensors", "--eval-texts",
                               "a.txt,b.txt"])
        self.assertEqual(args.eval_texts, "a.txt,b.txt")

    def test_eval_text_paths_prefers_plural(self):
        args = ppl_parse_args(["m.safetensors", "--eval-texts",
                               "a.txt,b.txt", "--eval-text", "c.txt"])
        self.assertEqual(eval_text_paths(args), ["a.txt", "b.txt"])
        args = ppl_parse_args(["m.safetensors", "--eval-text", "c.txt"])
        self.assertEqual(eval_text_paths(args), ["c.txt"])
        # empty --eval-texts falls back to the singular default
        args = ppl_parse_args(["m.safetensors", "--eval-texts", ""])
        self.assertEqual(eval_text_paths(args),
                         ["research/data/eval_text.txt"])

    def test_fp32_scheme_passthrough(self):
        fake = TestQuantizeModel._fake_dict()
        before = {k: v.copy() for k, v in fake.items()}
        out, bpw = quantize_model(fake, FP32_SCHEME, group_size=32)
        self.assertEqual(bpw, 32.0)
        self.assertEqual(set(out), set(fake))
        for k in fake:
            # bit-identical passthrough, input dict untouched
            np.testing.assert_array_equal(out[k], fake[k])
            np.testing.assert_array_equal(fake[k], before[k])
            self.assertEqual(out[k].dtype, np.float32)

    def test_aggregate_multitext_mean_std_sort(self):
        results = {
            "ternary_uniform": {"bpw": 1.710,
                                "ppls": {"t1": 100.0, "t2": 140.0},
                                "secs": 1.0},
            FP32_SCHEME: {"bpw": 32.0,
                          "ppls": {"t1": 50.0, "t2": 60.0},
                          "secs": 1.0},
        }
        rows = aggregate_multitext(results)
        # sorted by mean ascending: fp32 first
        self.assertEqual([r["scheme"] for r in rows],
                         [FP32_SCHEME, "ternary_uniform"])
        tu = rows[1]
        self.assertAlmostEqual(tu["mean"], 120.0)
        self.assertAlmostEqual(tu["std"], 20.0)  # population std
        self.assertAlmostEqual(tu["x_fp32"], 120.0 / 55.0)
        self.assertEqual(tu["ppls"], {"t1": 100.0, "t2": 140.0})

    def test_aggregate_multitext_no_fp32(self):
        results = {"ternary_uniform": {"bpw": 1.710,
                                       "ppls": {"t1": 100.0},
                                       "secs": 1.0}}
        rows = aggregate_multitext(results)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["mean"], 100.0)
        self.assertAlmostEqual(rows[0]["std"], 0.0)
        self.assertIsNone(rows[0]["x_fp32"])


class TestEvalBlocks(unittest.TestCase):
    """--eval-corpus shuffled-block sampling (sample_eval_blocks) and its
    CLI validation (check_corpus_args): the missing half of the
    text-robustness protocol, closed 2026-09-22."""

    def test_sample_blocks_deterministic(self):
        corpus = list(range(10000))
        a = sample_eval_blocks(corpus, 4, 256, seed=7)
        b = sample_eval_blocks(corpus, 4, 256, seed=7)
        self.assertEqual([l for l, _ in a], [l for l, _ in b])
        for (_, ia), (_, ib) in zip(a, b):
            self.assertEqual(ia, ib)

    def test_sample_blocks_disjoint_and_content_matches_offset(self):
        corpus = list(range(10000))
        blocks = sample_eval_blocks(corpus, 8, 256, seed=7)
        self.assertEqual(len(blocks), 8)
        ranges = []
        for label, ids in blocks:
            self.assertEqual(len(ids), 256)
            # label records the block index and the corpus offset
            idx, at = label.split("@")
            self.assertTrue(idx.startswith("b"))
            start = int(at)
            self.assertEqual(ids, corpus[start:start + 256])
            ranges.append((start, start + 256))
        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                a0, a1 = ranges[i]
                b0, b1 = ranges[j]
                self.assertTrue(a1 <= b0 or b1 <= a0,
                                f"blocks overlap: {ranges[i]} {ranges[j]}")

    def test_sample_blocks_different_seeds_differ(self):
        corpus = list(range(10000))
        a = [l for l, _ in sample_eval_blocks(corpus, 8, 256, seed=7)]
        b = [l for l, _ in sample_eval_blocks(corpus, 8, 256, seed=8)]
        self.assertNotEqual(a, b)

    def test_sample_blocks_too_short_raises(self):
        # 500 tokens -> only 1 non-overlapping 256-token slot
        with self.assertRaises(ValueError):
            sample_eval_blocks(list(range(500)), 2, 256, seed=7)
        # exactly enough slots works
        blocks = sample_eval_blocks(list(range(512)), 2, 256, seed=7)
        self.assertEqual(len(blocks), 2)

    def test_sample_blocks_n_blocks_zero_raises(self):
        with self.assertRaises(ValueError):
            sample_eval_blocks(list(range(10000)), 0, 256, seed=7)

    def test_check_corpus_args_valid_combo(self):
        args = ppl_parse_args(["m.safetensors", "--eval-corpus", "c.txt",
                               "--eval-blocks", "8"])
        check_corpus_args(args)  # must not raise
        self.assertEqual(args.eval_block_len, 256)
        self.assertEqual(args.eval_block_seed, 7)

    def test_check_corpus_args_blocks_without_corpus(self):
        args = ppl_parse_args(["m.safetensors", "--eval-blocks", "8"])
        with self.assertRaises(SystemExit):
            check_corpus_args(args)

    def test_check_corpus_args_corpus_without_blocks(self):
        args = ppl_parse_args(["m.safetensors", "--eval-corpus", "c.txt"])
        with self.assertRaises(SystemExit):
            check_corpus_args(args)

    def test_check_corpus_args_corpus_conflicts_with_eval_texts(self):
        args = ppl_parse_args(["m.safetensors", "--eval-corpus", "c.txt",
                               "--eval-blocks", "8",
                               "--eval-texts", "a.txt,b.txt"])
        with self.assertRaises(SystemExit):
            check_corpus_args(args)

    def test_check_corpus_args_zero_blocks(self):
        args = ppl_parse_args(["m.safetensors", "--eval-corpus", "c.txt",
                               "--eval-blocks", "0"])
        with self.assertRaises(SystemExit):
            check_corpus_args(args)


class TestQuantizedPerplexityEndToEnd(unittest.TestCase):
    """Gated: quantize the real GPT-2 124M and forward 16 tokens.

    The full per-scheme sweep (~40 s/forward) is a research-log artifact,
    not a unit test. This pins the plumbing end to end cheaply: quantized
    weights must still produce finite logits.
    """

    @classmethod
    def setUpClass(cls):
        cls.tensors = read_safetensors(GPT2_WEIGHTS)

    def test_quantized_model_forwards_finite(self):
        qw, bpw = quantize_model(self.tensors, "ternary_1step",
                                 group_size=128)
        self.assertAlmostEqual(bpw, float(np.log2(3)) + 16 / 128, places=9)
        logits = GPT2(qw).forward(list(range(16)))
        self.assertEqual(logits.shape, (16, 50257))
        self.assertTrue(np.all(np.isfinite(logits)))

    def test_perplexity_of_matches_model_method(self):
        ids = list(range(16))
        self.assertAlmostEqual(
            perplexity_of(self.tensors, ids),
            GPT2(self.tensors).perplexity(ids), places=9)


from src.quant_rnd.schemes import _lloyd_1d
from src.quant_rnd.fisher import (
    capture_linear_inputs,
    diag_fisher_weights,
    per_weight_importance,
)


class TestFisherWeightedLloyd(unittest.TestCase):
    """Diagonal-Fisher reweighting of the Lloyd centroid fit (OBQ slice 1).

    fisher.py collects per-input-channel activation energy d_j from an fp32
    forward pass; schemes.py::_lloyd_1d accepts it as per-element weights so
    high-energy channels pull the codebook toward themselves. Ungated tests
    pin the weighted-fit mechanics on synthetic data; gated tests pin the
    capture/energy plumbing on the real GPT-2 checkpoint.
    """

    def test_weighted_uniform_matches_unweighted(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal(256).astype(np.float32)
        a = _lloyd_1d(x, k=4, n_iter=20)
        b = _lloyd_1d(x, k=4, n_iter=20, w=np.ones(256))
        self.assertTrue(np.allclose(a, b, rtol=1e-6, atol=1e-9))

    def test_weighted_zero_weight_points_ignored(self):
        # One cluster + one far outlier with zero weight: the weighted
        # centroid must sit on the cluster, the unweighted one is dragged
        # toward the outlier.
        x = np.concatenate([np.linspace(-1, 1, 100),
                            [50.0]]).astype(np.float32)
        w = np.concatenate([np.ones(100), [0.0]])
        cw = _lloyd_1d(x, k=1, n_iter=20, w=w)
        cu = _lloyd_1d(x, k=1, n_iter=20)
        self.assertLess(abs(float(cw[0])), 0.05)
        self.assertGreater(abs(float(cu[0]) - float(cw[0])), 0.3)

    def test_weighted_rejects_negative(self):
        x = np.arange(16, dtype=np.float32)
        w = np.ones(16)
        w[3] = -1.0
        with self.assertRaises(ValueError):
            _lloyd_1d(x, k=2, w=w)

    def test_weighted_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            _lloyd_1d(np.arange(16, dtype=np.float32), k=2,
                      w=np.ones(8))

    def test_kmeans_q8_none_weight_is_default(self):
        rng = np.random.default_rng(7)
        w = rng.standard_normal(512).astype(np.float32)
        a = quantize_int2_kmeans_q8(w, group_size=128)
        b = quantize_int2_kmeans_q8(w, group_size=128, sample_weight=None)
        self.assertTrue(np.array_equal(a.codes, b.codes))
        self.assertTrue(np.array_equal(a.scales, b.scales))

    def test_kmeans_q8_weighted_changes_fit(self):
        # Deterministic (no random init): seeded weights must move the
        # codebook vs the unweighted fit.
        rng = np.random.default_rng(7)
        w = rng.standard_normal(512).astype(np.float32)
        sw = rng.uniform(0.1, 2.0, 512).astype(np.float32)
        a = quantize_int2_kmeans_q8(w, group_size=128)
        b = quantize_int2_kmeans_q8(w, group_size=128, sample_weight=sw)
        self.assertFalse(np.array_equal(a.codes, b.codes))
        self.assertFalse(np.array_equal(a.scales, b.scales))
        # Same storage format: bpw unchanged, decode still finite.
        self.assertAlmostEqual(a.bpw, b.bpw, places=12)
        self.assertTrue(np.all(np.isfinite(b.reconstruct())))

    def test_kmeans_q8_weighted_wrong_length(self):
        with self.assertRaises(ValueError):
            quantize_int2_kmeans_q8(np.zeros(128, dtype=np.float32),
                                    group_size=128,
                                    sample_weight=np.ones(64))

    def _fake_linear_dict(self):
        rng = np.random.default_rng(3)
        return {
            "h.0.attn.c_attn.weight": rng.standard_normal((8, 16)),
            "wte.weight": rng.standard_normal((32, 16)),
        }

    def test_quantize_model_passes_sample_weights(self):
        fake = self._fake_linear_dict()
        sw = {"h.0.attn.c_attn.weight": np.ones(8 * 16, dtype=np.float32)}
        out, bpw = quantize_model(fake, "int2_kmeans_q8", group_size=32,
                                  sample_weights=sw)
        self.assertAlmostEqual(bpw, 2.0 + 4 * 8 / 32 + 16 / 32, places=9)
        self.assertEqual(out["h.0.attn.c_attn.weight"].shape, (8, 16))
        # input dict not modified
        self.assertEqual(fake["h.0.attn.c_attn.weight"].shape, (8, 16))

    def test_quantize_model_ignores_weights_for_unsupported_scheme(self):
        fake = self._fake_linear_dict()
        sw = {"h.0.attn.c_attn.weight": np.ones(8 * 16, dtype=np.float32)}
        out, _ = quantize_model(fake, "ternary_uniform", group_size=32,
                                sample_weights=sw)
        plain, _ = quantize_model(fake, "ternary_uniform", group_size=32)
        self.assertTrue(np.array_equal(
            out["h.0.attn.c_attn.weight"], plain["h.0.attn.c_attn.weight"]))

    def test_quantize_model_unknown_weight_name_ignored(self):
        fake = self._fake_linear_dict()
        sw = {"nope.weight": np.ones(10, dtype=np.float32)}
        out, _ = quantize_model(fake, "int2_kmeans_q8", group_size=32,
                                sample_weights=sw)
        self.assertIn("h.0.attn.c_attn.weight", out)


@unittest.skipUnless(os.path.exists(GPT2_WEIGHTS), "gpt2 weights not present")
class TestFisherCaptureGated(unittest.TestCase):
    """Activation-capture and Fisher-energy plumbing on the real checkpoint."""

    @classmethod
    def setUpClass(cls):
        cls.tensors = read_safetensors(GPT2_WEIGHTS)
        cls.names = sorted(linear_weight_tensors(cls.tensors))

    def test_forward_capture_leaves_logits_unchanged(self):
        ids = list(range(16))
        model = GPT2(self.tensors)
        ref = model.forward(ids)
        cap: dict = {}
        got = model.forward(ids, capture=cap)
        self.assertTrue(np.array_equal(ref, got))
        self.assertEqual(set(cap), set(self.names))

    def test_capture_shapes_match_in_dims(self):
        ids = list(range(16))
        cap = capture_linear_inputs(self.tensors, ids)
        for name in self.names:
            in_dim = self.tensors[name].shape[0]
            self.assertEqual(cap[name].shape, (16, in_dim), name)

    def test_diag_fisher_nonnegative_and_sane(self):
        ids = list(range(16))
        cap = capture_linear_inputs(self.tensors, ids)
        energies = diag_fisher_weights(cap, self.names)
        self.assertEqual(set(energies), set(self.names))
        total = 0.0
        for name in self.names:
            e = energies[name]
            self.assertEqual(e.shape, (self.tensors[name].shape[0],), name)
            self.assertTrue(np.all(np.isfinite(e)), name)
            self.assertTrue(np.all(e >= 0), name)
            total += float(e.sum())
        self.assertGreater(total, 0.0)

    def test_per_weight_importance_alignment(self):
        ids = list(range(16))
        imp = per_weight_importance(self.tensors, ids)
        cap = capture_linear_inputs(self.tensors, ids)
        energies = diag_fisher_weights(cap, self.names)
        self.assertEqual(set(imp), set(self.names))
        name = self.names[0]
        w = self.tensors[name]
        out_dim = w.shape[1]
        self.assertEqual(imp[name].shape, (w.size,))
        e = energies[name]
        for j in range(w.shape[0]):
            seg = imp[name][j * out_dim:(j + 1) * out_dim]
            self.assertTrue(np.all(seg == e[j]), (name, j))


from src.quant_rnd.fisher import layer_fisher_trace
from src.quant_rnd.ppl import (
    check_mixed_args,
    count_blocks,
    mixed_scheme_label,
    parse_layer_schemes,
    quantize_model_per_layer,
    resolve_layer_schemes,
    run_mixed,
    sensitive_top_k,
)
from src.quant_rnd.realweights import layer_index_of


class TestLayerWiseMixedPrecision(unittest.TestCase):
    """Per-block scheme selector: parsing, sensitivity ranking, and the
    parameter-weighted bpw plumbing. Real-model numbers live in the
    research log; one gated test below pins the Fisher-trace plumbing
    on the real checkpoint."""

    @staticmethod
    def _fake_two_blocks():
        rng = np.random.default_rng(7)
        return {
            "h.0.mlp.c_fc.weight": rng.standard_normal((64, 32)).astype(np.float32),
            "h.0.attn.c_proj.weight": (0.1 * rng.standard_normal((32, 64))).astype(np.float32),
            "h.1.mlp.c_fc.weight": rng.standard_normal((16, 48)).astype(np.float32),
            # pass-through: embeddings, biases
            "wte.weight": rng.standard_normal((50, 64)).astype(np.float32),
            "h.0.mlp.c_fc.bias": rng.standard_normal((32,)).astype(np.float32),
        }

    # -- layer_index_of --

    def test_layer_index_of(self):
        self.assertEqual(layer_index_of("h.0.attn.c_attn.weight"), 0)
        self.assertEqual(layer_index_of("h.11.mlp.c_fc.weight"), 11)
        self.assertIsNone(layer_index_of("wte.weight"))
        self.assertIsNone(layer_index_of("ln_f.weight"))
        self.assertIsNone(layer_index_of("h.weight"))

    # -- parse_layer_schemes --

    def test_parse_layer_schemes_valid(self):
        a = parse_layer_schemes("0-5:int8_uniform,6-11:ternary_1step", 12)
        self.assertEqual(a, {i: "int8_uniform" for i in range(6)}
                         | {i: "ternary_1step" for i in range(6, 12)})
        b = parse_layer_schemes(" 0 : ternary_uniform , 1-2 : int8_uniform ",
                                3)
        self.assertEqual(b, {0: "ternary_uniform", 1: "int8_uniform",
                             2: "int8_uniform"})

    def test_parse_layer_schemes_gaps_overlaps_ranges(self):
        with self.assertRaises(ValueError):  # gap at block 5
            parse_layer_schemes("0-4:int8_uniform,6-11:ternary_1step", 12)
        with self.assertRaises(ValueError):  # block 5 twice
            parse_layer_schemes("0-5:int8_uniform,5-11:ternary_1step", 12)
        with self.assertRaises(ValueError):  # block 12 out of range
            parse_layer_schemes("0-12:ternary_1step", 12)
        with self.assertRaises(ValueError):  # inverted range
            parse_layer_schemes("5-2:ternary_1step,0-1:int8_uniform", 6)
        with self.assertRaises(ValueError):  # unknown scheme
            parse_layer_schemes("0-11:nope", 12)
        with self.assertRaises(ValueError):  # missing colon
            parse_layer_schemes("0-11", 12)
        with self.assertRaises(ValueError):  # empty spec covers nothing
            parse_layer_schemes("", 2)

    # -- sensitive_top_k --

    def test_sensitive_top_k_order_and_ties(self):
        trace = {0: 1.0, 1: 5.0, 2: 5.0, 3: 0.5}
        self.assertEqual(sensitive_top_k(trace, 2), [1, 2])
        self.assertEqual(sensitive_top_k(trace, 3), [1, 2, 0])

    def test_sensitive_top_k_bad_k(self):
        trace = {0: 1.0, 1: 2.0}
        for bad in (0, 3, -1, "2"):
            with self.assertRaises(ValueError):
                sensitive_top_k(trace, bad)

    # -- count_blocks --

    def test_count_blocks(self):
        self.assertEqual(count_blocks(self._fake_two_blocks()), 2)
        with self.assertRaises(KeyError):
            count_blocks({})

    # -- quantize_model_per_layer --

    def test_per_layer_quantize_matches_single_scheme(self):
        fake = self._fake_two_blocks()
        assignment = {0: "int8_uniform", 1: "ternary_1step"}
        out, avg_bpw = quantize_model_per_layer(fake, assignment,
                                                group_size=32)
        # per-tensor output is independent of the other blocks' schemes
        ref8, _ = quantize_model(fake, "int8_uniform", group_size=32)
        ref3, _ = quantize_model(fake, "ternary_1step", group_size=32)
        np.testing.assert_array_equal(out["h.0.mlp.c_fc.weight"],
                                      ref8["h.0.mlp.c_fc.weight"])
        np.testing.assert_array_equal(out["h.1.mlp.c_fc.weight"],
                                      ref3["h.1.mlp.c_fc.weight"])
        # pass-through tensors unchanged
        np.testing.assert_array_equal(out["wte.weight"], fake["wte.weight"])
        np.testing.assert_array_equal(out["h.0.mlp.c_fc.bias"],
                                      fake["h.0.mlp.c_fc.bias"])

    def test_per_layer_avg_bpw_is_param_weighted(self):
        fake = self._fake_two_blocks()
        assignment = {0: "int8_uniform", 1: "ternary_1step"}
        _, avg = quantize_model_per_layer(fake, assignment, group_size=32)
        p0 = fake["h.0.mlp.c_fc.weight"].size + fake["h.0.attn.c_proj.weight"].size
        p1 = fake["h.1.mlp.c_fc.weight"].size
        b0 = SCHEMES["int8_uniform"](
            np.zeros(p0, dtype=np.float32), group_size=32).bpw
        b1 = SCHEMES["ternary_1step"](
            np.zeros(p1, dtype=np.float32), group_size=32).bpw
        self.assertAlmostEqual(avg, (p0 * b0 + p1 * b1) / (p0 + p1),
                               places=9)

    def test_per_layer_unknown_scheme_raises(self):
        with self.assertRaises(KeyError):
            quantize_model_per_layer(self._fake_two_blocks(),
                                     {0: "nope", 1: "ternary_1step"},
                                     group_size=32)

    def test_per_layer_missing_block_raises(self):
        with self.assertRaises(KeyError):
            quantize_model_per_layer(self._fake_two_blocks(),
                                     {0: "ternary_1step"}, group_size=32)

    # -- mixed_scheme_label --

    def test_mixed_scheme_label_runs(self):
        a = {0: "int8_uniform", 1: "int8_uniform",
             2: "ternary_1step", 3: "ternary_1step"}
        self.assertEqual(mixed_scheme_label(a),
                         "mix[0-1:int8_uniform,2-3:ternary_1step]")
        b = {i: ("int8_uniform" if i < 5 else "ternary_1step")
             for i in range(12)}
        self.assertEqual(mixed_scheme_label(b),
                         "mix[0-4:int8_uniform,5-11:ternary_1step]")

    # -- check_mixed_args / CLI --

    def _base_args(self):
        return ppl_parse_args(["m.safetensors"])

    def test_cli_mixed_flags_parse(self):
        args = ppl_parse_args(["m.safetensors", "--per-layer-schemes",
                               "0-5:int8_uniform,6-11:ternary_1step"])
        self.assertEqual(args.per_layer_schemes,
                         "0-5:int8_uniform,6-11:ternary_1step")
        args = ppl_parse_args(["m.safetensors", "--sensitive-layers", "5",
                               "--sensitive-scheme", "int8_uniform",
                               "--base-scheme", "ternary_1step"])
        self.assertEqual(args.sensitive_layers, 5)
        self.assertEqual(args.sensitive_scheme, "int8_uniform")
        self.assertEqual(args.base_scheme, "ternary_1step")

    def test_check_mixed_args_defaults_ok(self):
        check_mixed_args(self._base_args())  # no flags: no-op

    def test_check_mixed_args_exclusions(self):
        base = ["m.safetensors"]
        # both modes at once
        args = ppl_parse_args(base + ["--per-layer-schemes", "0:int8_uniform",
                                       "--sensitive-layers", "1",
                                       "--sensitive-scheme", "int8_uniform",
                                       "--base-scheme", "ternary_1step"])
        with self.assertRaises(SystemExit):
            check_mixed_args(args)
        # with one-layer / only-names / obq
        for extra in (["--one-layer", "h.0.attn.c_attn.weight"],
                      ["--only-names", "wte.weight"],
                      ["--one-layer", "h.0.attn.c_attn.weight", "--obq"],
                      ["--obq-all"]):
            args = ppl_parse_args(base + ["--per-layer-schemes", "0:int8_uniform"] + extra)
            with self.assertRaises(SystemExit):
                check_mixed_args(args)
        # sensitive scheme flags without --sensitive-layers
        args = ppl_parse_args(base + ["--sensitive-scheme", "int8_uniform"])
        with self.assertRaises(SystemExit):
            check_mixed_args(args)
        # --sensitive-layers without both schemes
        args = ppl_parse_args(base + ["--sensitive-layers", "1"])
        with self.assertRaises(SystemExit):
            check_mixed_args(args)
        # --fisher and --quantize-embeddings are separate protocols
        args = ppl_parse_args(base + ["--per-layer-schemes", "0:int8_uniform",
                                       "--fisher"])
        with self.assertRaises(SystemExit):
            check_mixed_args(args)
        args = ppl_parse_args(base + ["--per-layer-schemes", "0:int8_uniform",
                                       "--quantize-embeddings"])
        with self.assertRaises(SystemExit):
            check_mixed_args(args)

    def test_check_mixed_args_unknown_sensitive_scheme(self):
        args = ppl_parse_args(["m.safetensors", "--sensitive-layers", "1",
                               "--sensitive-scheme", "nope",
                               "--base-scheme", "ternary_1step"])
        with self.assertRaises(ValueError):
            check_mixed_args(args)


class TestLayerFisherTraceGated(unittest.TestCase):
    """Fisher-trace plumbing on the real GPT-2 checkpoint (gated: ~1
    forward pass)."""

    @classmethod
    def setUpClass(cls):
        cls.tensors = read_safetensors(GPT2_WEIGHTS)

    def test_trace_covers_all_blocks(self):
        trace = layer_fisher_trace(self.tensors, list(range(16)))
        self.assertEqual(set(trace), set(range(12)))
        self.assertTrue(all(v > 0 for v in trace.values()))
        total = sum(trace.values())
        self.assertTrue(math.isfinite(total) and total > 0)


from src.quant_rnd import ppl as ppl_module
from src.quant_rnd.ppl import (
    block_linear_names,
    check_mechanism_args,
    leave_one_out_ppl,
    ordered_desc,
    run_sensitivity_mechanism,
    spearman_rho,
    trace_on_quantized,
)


class TestSensitivityMechanism(unittest.TestCase):
    """Activation-drift mechanism probe (ppl.py): ranking helpers and
    leave-one-out targeting mechanics on a fake weight dict; one gated
    end-to-end smoke on the real checkpoint below."""

    @staticmethod
    def _fake_two_block_dict():
        rng = np.random.default_rng(11)
        return {
            "h.0.attn.c_attn.weight":
                rng.standard_normal((32, 64)).astype(np.float32),
            "h.0.mlp.c_fc.weight":
                rng.standard_normal((64, 32)).astype(np.float32),
            "h.1.attn.c_attn.weight":
                rng.standard_normal((32, 64)).astype(np.float32),
            "h.1.mlp.c_fc.weight":
                rng.standard_normal((64, 32)).astype(np.float32),
            "wte.weight":
                rng.standard_normal((50, 64)).astype(np.float32),
            "h.0.attn.c_attn.bias":
                rng.standard_normal((64,)).astype(np.float32),
        }

    def test_ordered_desc_and_tie_break(self):
        self.assertEqual(ordered_desc({0: 1.0, 1: 3.0, 2: 2.0}),
                         [1, 2, 0])
        # ties break by lower index (deterministic)
        self.assertEqual(ordered_desc({2: 5.0, 0: 5.0, 1: 5.0}),
                         [0, 1, 2])

    def test_spearman_perfect_orders(self):
        self.assertAlmostEqual(
            spearman_rho([0, 1, 2, 3], [0, 1, 2, 3]), 1.0)
        self.assertAlmostEqual(
            spearman_rho([0, 1, 2, 3], [3, 2, 1, 0]), -1.0)

    def test_spearman_partial(self):
        # one adjacent swap in 4 elements -> rho 0.8 (hand-computed)
        self.assertAlmostEqual(
            spearman_rho([0, 1, 2, 3], [0, 1, 3, 2]), 0.8)

    def test_spearman_rejects_mismatched_or_degenerate(self):
        with self.assertRaises(ValueError):
            spearman_rho([0, 1, 2], [0, 1, 3])
        with self.assertRaises(ValueError):
            spearman_rho([0], [0])

    def test_block_linear_names_groups_and_excludes(self):
        groups = block_linear_names(self._fake_two_block_dict())
        self.assertEqual(set(groups), {0, 1})
        self.assertEqual(groups[0],
                         {"h.0.attn.c_attn.weight", "h.0.mlp.c_fc.weight"})
        self.assertEqual(groups[1],
                         {"h.1.attn.c_attn.weight", "h.1.mlp.c_fc.weight"})

    def test_block_linear_names_rejects_blockless(self):
        with self.assertRaises(KeyError):
            block_linear_names(
                {"wte.weight": np.zeros((4, 4), dtype=np.float32)})

    def test_leave_one_out_targets_exactly_one_block(self):
        fake = self._fake_two_block_dict()
        seen = []

        def fake_ppl(tensors, ids):
            changed = sorted(n for n in tensors
                             if not np.array_equal(tensors[n], fake[n]))
            seen.append(changed)
            return 100.0 + len(changed)

        orig = ppl_module.perplexity_of
        ppl_module.perplexity_of = fake_ppl
        try:
            rows = leave_one_out_ppl(fake, [0, 1, 2], "ternary_1step",
                                     group_size=32)
        finally:
            ppl_module.perplexity_of = orig
        self.assertEqual([r["block"] for r in rows], [0, 1])
        self.assertEqual(set(seen[0]),
                         {"h.0.attn.c_attn.weight", "h.0.mlp.c_fc.weight"})
        self.assertEqual(set(seen[1]),
                         {"h.1.attn.c_attn.weight", "h.1.mlp.c_fc.weight"})
        for r in rows:
            self.assertTrue(math.isfinite(r["ppl"]) and r["ppl"] > 0)
            self.assertTrue(math.isfinite(r["bpw"]) and r["bpw"] > 0)

    def test_leave_one_out_blocks_arg_unknown_scheme_and_block(self):
        fake = self._fake_two_block_dict()

        def fake_ppl(tensors, ids):
            return 42.0

        orig = ppl_module.perplexity_of
        ppl_module.perplexity_of = fake_ppl
        try:
            rows = leave_one_out_ppl(fake, [0], "ternary_1step",
                                     group_size=32, blocks=[1])
        finally:
            ppl_module.perplexity_of = orig
        self.assertEqual([r["block"] for r in rows], [1])
        with self.assertRaises(KeyError):
            leave_one_out_ppl(fake, [0], "no_such_scheme")
        with self.assertRaises(ValueError):
            leave_one_out_ppl(fake, [0], "ternary_1step", blocks=[7])

    def test_trace_on_quantized_uses_fisher_plumbing(self):
        import src.quant_rnd.fisher as fisher_mod
        fake = self._fake_two_block_dict()
        captured = {}

        def fake_trace(tensors, ids):
            captured["keys"] = set(tensors)
            return {0: 1.0, 1: 2.0}

        orig = fisher_mod.layer_fisher_trace
        fisher_mod.layer_fisher_trace = fake_trace
        try:
            out = trace_on_quantized(fake, [0, 1], "ternary_1step",
                                     group_size=32)
        finally:
            fisher_mod.layer_fisher_trace = orig
        self.assertEqual(out, {0: 1.0, 1: 2.0})
        # the trace ran on the fully-quantized weight dict (all keys
        # present), not on a per-block subset
        self.assertEqual(captured["keys"], set(fake))

    def test_check_mechanism_args(self):
        args = ppl_parse_args(["m.safetensors", "--sensitivity-mechanism",
                               "ternary_1step_ds"])
        check_mechanism_args(args)  # must not raise
        bad = ppl_parse_args(["m.safetensors", "--sensitivity-mechanism",
                              "ternary_1step_ds", "--fisher"])
        with self.assertRaises(SystemExit):
            check_mechanism_args(bad)
        bad2 = ppl_parse_args(["m.safetensors", "--sensitivity-mechanism",
                               "no_such_scheme"])
        with self.assertRaises(ValueError):
            check_mechanism_args(bad2)


@unittest.skipUnless(os.path.exists(GPT2_WEIGHTS), "gpt2 weights not present")
class TestSensitivityMechanismGated(unittest.TestCase):
    """Gated: the mechanism probe on the real GPT-2 checkpoint, 2 blocks
    x 16 tokens (~4 cheap forwards). Pins the plumbing end to end."""

    @classmethod
    def setUpClass(cls):
        cls.tensors = read_safetensors(GPT2_WEIGHTS)

    def test_mechanism_probe_two_blocks(self):
        ids = list(range(16))
        res = run_sensitivity_mechanism(self.tensors, ids, "ternary_1step",
                                        group_size=128, blocks=[0, 11])
        self.assertTrue(math.isfinite(res["fp32_ppl"])
                        and res["fp32_ppl"] > 0)
        self.assertEqual(set(res["trace_fp32"]), set(range(12)))
        self.assertEqual([r["block"] for r in res["leave_one_out"]],
                         [0, 11])
        self.assertEqual(set(res["trace_quantized"]), set(range(12)))
        for key in ("rho_damage", "rho_drift"):
            self.assertTrue(-1.0 <= res[key] <= 1.0)


def _reference_ternary_lloyd_fit(g, dual, n_iter=20, thresh_factor=1.0):
    """FROZEN oracle: verbatim copy of the pre-vectorization
    _ternary_lloyd_fit (2026-09-22), kept to pin the vectorized batch
    core bit-identical against the old per-group float64 loop. Do NOT
    "simplify" this to call the new code - that would make the pin
    circular. If a future numpy changes its reduction order, these tests
    fail loudly and the published anchor numbers get re-verified."""
    g = g.astype(np.float64)
    t = 0.5 * thresh_factor
    if dual:
        pos = g[g > 0]
        neg = g[g < 0]
        s_pos = float(pos.mean()) if pos.size else 0.0
        s_neg = float(-neg.mean()) if neg.size else 0.0
    else:
        s_pos = s_neg = float(np.mean(np.abs(g)))
    for _ in range(n_iter):
        a_pos = g > t * s_pos
        a_neg = g < -t * s_neg
        new_pos = float(g[a_pos].mean()) if a_pos.any() else s_pos
        new_neg = float(-g[a_neg].mean()) if a_neg.any() else s_neg
        if not dual:
            n_assigned = a_pos.sum() + a_neg.sum()
            if n_assigned:
                new = (a_pos.sum() * new_pos + a_neg.sum() * new_neg) / n_assigned
            else:
                new = s_pos
            new_pos = new_neg = new
        if new_pos == s_pos and new_neg == s_neg:
            break
        s_pos, s_neg = new_pos, new_neg
    s_pos = max(s_pos, 1e-12)
    s_neg = max(s_neg, 1e-12)
    codes = np.zeros(g.shape[0], dtype=np.int8)
    codes[g > t * s_pos] = 1
    codes[g < -t * s_neg] = -1
    return s_pos, s_neg, codes


class TestTernaryLloydVectorized(unittest.TestCase):
    """The vectorized ternary encoder must be bit-identical to the old
    per-group loop (roadmap: vectorize item) - every published anchor
    number (SQNR tables, ppl 2764/795.96, opcount zero-rates) was
    produced by the old path."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20260922)
        cls.tensors = [
            rng.standard_normal(4096).astype(np.float32),          # clean
            (rng.standard_normal(4096) + 0.5).astype(np.float32),  # skewed
            np.zeros(4096, dtype=np.float32),                      # all-zero groups
            np.abs(rng.standard_normal(4096)).astype(np.float32),  # one-sided
            rng.standard_normal(1000).astype(np.float32),         # ragged (padding)
            rng.standard_normal(37).astype(np.float32),           # single padded group
        ]

    def _reference_assembled(self, w, dual, n_iter, thresh_factor,
                             group_size):
        wp, n_groups, n = _groups(w, group_size=group_size)
        scales = np.zeros((n_groups, 2 if dual else 1), dtype=np.float32)
        codes = np.zeros(n_groups * group_size, dtype=np.int8)
        for gi in range(n_groups):
            sp, sn, c = _reference_ternary_lloyd_fit(
                wp[gi], dual=dual, n_iter=n_iter,
                thresh_factor=thresh_factor)
            scales[gi, 0] = np.float32(sp)
            if dual:
                scales[gi, 1] = np.float32(sn)
            codes[gi * group_size:(gi + 1) * group_size] = c
        return codes[:n], scales

    def test_batch_core_bit_identical(self):
        from src.quant_rnd.schemes import _ternary_lloyd_fit_batch
        for dual in (False, True):
            for n_iter in (1, 2, 20):
                for thresh_factor in (1.0, 1.2):
                    for gs in (64, 128, 256):
                        for w in self.tensors:
                            wp, n_groups, n = _groups(w, group_size=gs)
                            sp, sn, codes, _st = _ternary_lloyd_fit_batch(
                                wp, dual=dual, n_iter=n_iter,
                                thresh_factor=thresh_factor)
                            ref_codes, ref_scales = self._reference_assembled(
                                w, dual, n_iter, thresh_factor, gs)
                            tag = (f"dual={dual} n_iter={n_iter} "
                                   f"tf={thresh_factor} gs={gs}")
                            np.testing.assert_array_equal(
                                codes[:n], ref_codes,
                                err_msg=f"codes {tag}")
                            got = np.stack([sp, sn], axis=1)[:, :2 if dual else 1]
                            np.testing.assert_array_equal(
                                got.astype(np.float32), ref_scales,
                                err_msg=f"scales {tag}")

    def test_scheme_functions_bit_identical(self):
        # The four public scheme entry points route through the batch
        # core; pin their full QuantResults against the frozen reference.
        configs = [
            (quantize_ternary_lloyd, False, 20, 1.0),
            (quantize_ternary_lloyd_ds, True, 20, 1.0),
            (quantize_ternary_1step, False, 1, 1.0),
            (quantize_ternary_1step_ds, True, 1, 1.0),
        ]
        for fn, dual, n_iter, tf in configs:
            for w in self.tensors:
                q = fn(w)
                ref_codes, ref_scales = self._reference_assembled(
                    w, dual, n_iter, tf, 128)
                np.testing.assert_array_equal(q.codes, ref_codes,
                                              err_msg=fn.__name__)
                np.testing.assert_array_equal(q.scales, ref_scales,
                                              err_msg=fn.__name__)

    def test_sparse_variant_bit_identical(self):
        q = SCHEMES["ternary_1step_sp"](self.tensors[0])
        ref_codes, ref_scales = self._reference_assembled(
            self.tensors[0], False, 1, 1.2, 128)
        np.testing.assert_array_equal(q.codes, ref_codes)
        np.testing.assert_array_equal(q.scales, ref_scales)

    def test_single_group_wrapper_matches_batch(self):
        # _ternary_lloyd_fit is now a thin wrapper over the batch core:
        # same outputs as the batch on one group, and the history hook
        # used by diagnose.py still records the absmean init first.
        from src.quant_rnd.schemes import _ternary_lloyd_fit_batch
        for dual in (False, True):
            for tf in (1.0, 1.2):
                wp, _ng, _n = _groups(self.tensors[1], group_size=128)
                g = wp[3]
                h = []
                sp, sn, codes = _ternary_lloyd_fit(
                    g, dual=dual, thresh_factor=tf, history=h)
                bsp, bsn, bcodes, _st = _ternary_lloyd_fit_batch(
                    g.reshape(1, -1), dual=dual, thresh_factor=tf)
                self.assertEqual(sp, float(bsp[0]))
                self.assertEqual(sn, float(bsn[0]))
                np.testing.assert_array_equal(codes, bcodes)
                self.assertGreaterEqual(len(h), 1)
                g64 = g.astype(np.float64)
                if not dual:
                    self.assertAlmostEqual(h[0][0],
                                           float(np.mean(np.abs(g64))),
                                           places=9)
                    self.assertAlmostEqual(h[0][1], h[0][0], places=9)
                else:
                    # Dual init: per-side conditional means, not absmean.
                    pos = g64[g64 > 0]
                    neg = g64[g64 < 0]
                    self.assertAlmostEqual(h[0][0], float(pos.mean()),
                                           places=9)
                    self.assertAlmostEqual(h[0][1], float(-neg.mean()),
                                           places=9)


if __name__ == "__main__":
    unittest.main()
