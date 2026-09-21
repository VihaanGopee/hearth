"""Tests for the OBQ/GPTQ-style error-compensation slice (obq.py)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quant_rnd.obq import (
    hessian_weighted_sse,
    inverse_hessian,
    quantize_layer_obq,
)
from src.quant_rnd.schemes import _scale_overhead, quantize_int2_kmeans_q8


def _rng(seed=7):
    return np.random.default_rng(seed)


class TestInverseHessian(unittest.TestCase):
    def test_identity_on_full_rank(self):
        rng = _rng()
        X = rng.normal(size=(200, 32))
        Hinv = inverse_hessian(X, damp_frac=1e-6)
        H = X.T @ X + 1e-6 * np.mean(np.diag(X.T @ X)) * np.eye(32)
        np.testing.assert_allclose(H @ Hinv, np.eye(32), atol=1e-6)

    def test_symmetric(self):
        X = _rng().normal(size=(150, 24))
        Hinv = inverse_hessian(X)
        np.testing.assert_allclose(Hinv, Hinv.T, atol=1e-12)

    def test_rank_deficient_needs_damping(self):
        # T < d_in: undamped inverse does not exist; damped must be finite.
        X = _rng().normal(size=(10, 32))
        Hinv = inverse_hessian(X, damp_frac=0.01)
        self.assertTrue(np.all(np.isfinite(Hinv)))
        # Larger damping -> Hinv closer to (damp*I)^{-1}, i.e. smaller norm.
        Hinv_big = inverse_hessian(X, damp_frac=0.1)
        self.assertLess(np.linalg.norm(Hinv_big), np.linalg.norm(Hinv))

    def test_rejects_nonpositive_damping(self):
        X = _rng().normal(size=(50, 8))
        with self.assertRaises(ValueError):
            inverse_hessian(X, damp_frac=0.0)
        with self.assertRaises(ValueError):
            inverse_hessian(X, damp_frac=-0.5)

    def test_rejects_non_2d(self):
        with self.assertRaises(ValueError):
            inverse_hessian(np.zeros(16))

    def test_all_zero_activations_still_finite(self):
        Hinv = inverse_hessian(np.zeros((20, 8)), damp_frac=0.01)
        self.assertTrue(np.all(np.isfinite(Hinv)))


class TestQuantizeLayerObq(unittest.TestCase):
    def _small(self, seed=7):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(300, 64))
        W = rng.normal(scale=0.05, size=(64, 32))
        return W.astype(np.float32), X

    def test_single_block_equals_naive_per_column(self):
        # With one block there is no compensation: the result must be
        # bit-identical to per-column int2_kmeans_q8 quantization.
        W, X = self._small()
        Wq, _ = quantize_layer_obq(W, X, group_size=64, block_size=64)
        ref = np.stack(
            [quantize_int2_kmeans_q8(W[:, k], group_size=64).reconstruct()
             for k in range(W.shape[1])], axis=1)
        np.testing.assert_array_equal(Wq, ref)

    def test_compensation_reduces_hessian_objective(self):
        # The core mathematical claim: the greedy block update is the exact
        # minimizer of the quadratic given each quantized block, so the
        # OBQ objective must beat the no-compensation (naive) choice.
        W, X = self._small()
        H = X.T @ X
        Wq_obq, _ = quantize_layer_obq(W, X, group_size=16, block_size=16)
        Wq_naive, _ = quantize_layer_obq(W, X, group_size=16, block_size=64)
        obj_obq = hessian_weighted_sse(W, Wq_obq, H)
        obj_naive = hessian_weighted_sse(W, Wq_naive, H)
        self.assertLess(obj_obq, obj_naive)

    def test_block_update_formula_pin(self):
        # Manual check of the GPTQ compensation step on a tiny case.
        rng = np.random.default_rng(3)
        X = rng.normal(size=(60, 8))
        W = rng.normal(scale=0.05, size=(8, 4)).astype(np.float32)
        Hinv = inverse_hessian(X)
        Wq, _ = quantize_layer_obq(W, X, group_size=4, block_size=4)
        b = slice(0, 4)
        # Block 0 sees no compensation: bit-identical to naive fit.
        ref0 = np.stack(
            [quantize_int2_kmeans_q8(W[b, k], group_size=4).reconstruct()
             for k in range(4)], axis=1)
        np.testing.assert_array_equal(Wq[b, :], ref0)
        # The exact GPTQ update applied to the remaining rows...
        err = W[b, :].astype(np.float64) - ref0.astype(np.float64)
        E = np.linalg.solve(Hinv[b, b], err)
        compensated = W[4:, :].astype(np.float64) - Hinv[4:, b] @ E
        # ...and block 1 was quantized from those compensated rows.
        ref1 = np.stack(
            [quantize_int2_kmeans_q8(compensated[:, k],
                                    group_size=4).reconstruct()
             for k in range(4)], axis=1)
        np.testing.assert_array_equal(Wq[4:, :], ref1)

    def test_bpw_matches_kmeans_q8(self):
        # Identical bitrate formula to int2_kmeans_q8 by construction:
        # 2-bit payload + 4 int8 codebook entries + one fp16 scale/group.
        W, X = self._small()
        _, bpw = quantize_layer_obq(W, X, group_size=16, block_size=16)
        self.assertAlmostEqual(bpw, 2.0 + 4 * 8 / 16
                               + _scale_overhead(1, 16))
        # And at the reference config it is exactly the anchor bitrate.
        W128 = np.zeros((128, 8), dtype=np.float32)
        X128 = _rng().normal(size=(200, 128))
        _, bpw128 = quantize_layer_obq(W128, X128)
        self.assertAlmostEqual(bpw128, 2.375)

    def test_input_not_mutated(self):
        W, X = self._small()
        W0 = W.copy()
        quantize_layer_obq(W, X, group_size=16, block_size=16)
        np.testing.assert_array_equal(W, W0)

    def test_rejects_bad_shapes(self):
        W, X = self._small()
        with self.assertRaises(ValueError):
            quantize_layer_obq(W.ravel(), X)
        with self.assertRaises(ValueError):
            quantize_layer_obq(W, X[:, :32])  # channel mismatch

    def test_zero_column_stays_finite(self):
        W, X = self._small()
        W[:, 5] = 0.0
        Wq, _ = quantize_layer_obq(W, X, group_size=16, block_size=16)
        self.assertTrue(np.all(np.isfinite(Wq)))
        np.testing.assert_allclose(Wq[:, 5], 0.0, atol=1e-6)

    def test_output_shape_and_dtype(self):
        W, X = self._small()
        Wq, _ = quantize_layer_obq(W, X, group_size=16, block_size=16)
        self.assertEqual(Wq.shape, W.shape)
        self.assertEqual(Wq.dtype, np.float32)

    def test_block_size_must_align_with_groups(self):
        # block_size must be a multiple of group_size and divide d_in;
        # anything else is rejected rather than silently mis-accounted.
        W, X = self._small()
        with self.assertRaises(ValueError):
            quantize_layer_obq(W, X, group_size=16, block_size=24)
        with self.assertRaises(ValueError):
            quantize_layer_obq(W, X, group_size=16, block_size=48)


class TestHessianWeightedSse(unittest.TestCase):
    def test_zero_error(self):
        W = _rng().normal(size=(16, 8))
        H = np.eye(16)
        self.assertAlmostEqual(hessian_weighted_sse(W, W, H), 0.0)

    def test_matches_manual_quadratic(self):
        rng = _rng(5)
        W = rng.normal(size=(16, 8))
        Wq = rng.normal(size=(16, 8))
        X = rng.normal(size=(100, 16))
        H = X.T @ X
        D = W - Wq
        manual = sum(float(D[:, k] @ H @ D[:, k]) for k in range(8))
        self.assertAlmostEqual(hessian_weighted_sse(W, Wq, H), manual,
                               places=6)


if __name__ == "__main__":
    unittest.main()
