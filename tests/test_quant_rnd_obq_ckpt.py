"""Tests for the checkpointed OBQ runner (obq_ckpt.py)."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quant_rnd.obq import quantize_layers_obq
from src.quant_rnd.obq_ckpt import (
    assemble_full_model,
    check_params_compatible,
    quantize_missing,
    read_manifest,
    run_params,
    safe_layer_filename,
    write_layer_ckpt,
)


def _rng(seed=7):
    return np.random.default_rng(seed)


def _tiny_model():
    """Two synthetic linear layers (names pass linear_weight_tensors)
    plus a bias that must pass through untouched."""
    rng = _rng()
    tensors = {
        "h.0.attn.c_attn.weight": rng.normal(size=(32, 16)).astype(np.float32),
        "h.0.mlp.c_fc.weight": rng.normal(size=(32, 16)).astype(np.float32),
        "h.0.attn.c_attn.bias": rng.normal(size=(16,)).astype(np.float32),
    }
    linear = {n: t for n, t in tensors.items() if n.endswith(".weight")}
    inputs = {n: rng.normal(size=(48, 32)).astype(np.float64)
              for n in linear}
    return tensors, linear, inputs


def _params(group_size=16, damp_frac=0.01, ids=None):
    if ids is None:
        ids = np.arange(48, dtype=np.int64)
    return run_params(group_size, damp_frac, ids)


class TestSafeLayerFilename(unittest.TestCase):
    def test_dots_and_slashes_gone(self):
        f = safe_layer_filename("h.0/attn.c_attn.weight")
        self.assertNotIn("/", f)
        self.assertNotIn("\\", f)
        self.assertTrue(f.endswith(".npz"))

    def test_distinct_names_stay_distinct(self):
        a = safe_layer_filename("h.0.attn.c_attn.weight")
        b = safe_layer_filename("h.1.attn.c_attn.weight")
        self.assertNotEqual(a, b)

    def test_no_path_escape(self):
        f = safe_layer_filename("../../evil")
        self.assertEqual(os.path.basename(f), f)


class TestRunParams(unittest.TestCase):
    def test_deterministic(self):
        ids = np.arange(10)
        self.assertEqual(_params(ids=ids), _params(ids=ids))

    def test_token_sequence_matters(self):
        p1 = _params(ids=np.arange(10))
        p2 = _params(ids=np.arange(10) + 1)
        self.assertNotEqual(p1["text_sha"], p2["text_sha"])

    def test_hyperparams_recorded(self):
        p = run_params(128, 0.01, np.arange(5))
        self.assertEqual(p["group_size"], 128)
        self.assertEqual(p["damp_frac"], 0.01)
        self.assertEqual(p["block_size"], 128)
        self.assertEqual(p["n_tokens"], 5)


class TestManifestRoundTrip(unittest.TestCase):
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(read_manifest(d), {})

    def test_write_then_read(self):
        with tempfile.TemporaryDirectory() as d:
            params = _params()
            Wq = np.ones((32, 16), dtype=np.float32)
            write_layer_ckpt(d, "h.0.attn.c_attn.weight", Wq, params)
            m = read_manifest(d)
            self.assertIn("h.0.attn.c_attn.weight", m["layers"])
            self.assertEqual(m["params"]["group_size"], 16)
            # Second layer appends, first entry survives.
            write_layer_ckpt(d, "h.0.mlp.c_fc.weight", Wq * 2, params)
            m2 = read_manifest(d)
            self.assertEqual(len(m2["layers"]), 2)

    def test_stored_weight_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            params = _params()
            Wq = _rng().normal(size=(32, 16)).astype(np.float32)
            write_layer_ckpt(d, "h.0.attn.c_attn.weight", Wq, params)
            m = read_manifest(d)
            path = os.path.join(d, m["layers"]["h.0.attn.c_attn.weight"])
            with np.load(path) as z:
                np.testing.assert_array_equal(z["Wq"], Wq)


class TestCheckParamsCompatible(unittest.TestCase):
    def test_empty_manifest_ok(self):
        check_params_compatible({}, _params())

    def test_match_ok(self):
        with tempfile.TemporaryDirectory() as d:
            p = _params()
            write_layer_ckpt(d, "n", np.zeros((2, 2), np.float32), p)
            check_params_compatible(read_manifest(d), p)

    def test_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = _params()
            write_layer_ckpt(d, "n", np.zeros((2, 2), np.float32), p)
            m = read_manifest(d)
            for bad in (_params(group_size=32),
                        _params(damp_frac=0.02),
                        _params(ids=np.arange(9))):
                with self.assertRaises(ValueError):
                    check_params_compatible(m, bad)


class TestQuantizeMissing(unittest.TestCase):
    def test_resume_in_two_chunks(self):
        tensors, linear, inputs = _tiny_model()
        with tempfile.TemporaryDirectory() as d:
            params = _params()
            r1 = quantize_missing(linear, inputs, d, params, max_layers=1)
            self.assertEqual(r1["new"], 1)
            self.assertEqual(r1["done"], 1)
            self.assertEqual(r1["total"], 2)
            self.assertIsNotNone(r1["next"])
            r2 = quantize_missing(linear, inputs, d, params, max_layers=1)
            self.assertEqual(r2["new"], 1)
            self.assertEqual(r2["done"], 2)
            self.assertIsNone(r2["next"])
            # Third call: nothing left, reports completion.
            r3 = quantize_missing(linear, inputs, d, params)
            self.assertEqual(r3["new"], 0)
            self.assertEqual(r3["done"], 2)

    def test_max_layers_zero_is_status_probe(self):
        _, linear, inputs = _tiny_model()
        with tempfile.TemporaryDirectory() as d:
            r = quantize_missing(linear, inputs, d, _params(),
                                 max_layers=0)
            self.assertEqual(r["new"], 0)
            self.assertEqual(r["done"], 0)
            self.assertEqual(r3_next(r), "h.0.attn.c_attn.weight")
            self.assertEqual(read_manifest(d), {})

    def test_checkpointed_assembly_matches_in_memory(self):
        # The resumed run must assemble a dict bit-identical to an
        # in-memory quantize_layers_obq over the same layers.
        tensors, linear, inputs = _tiny_model()
        with tempfile.TemporaryDirectory() as d:
            params = _params()
            quantize_missing(linear, inputs, d, params, max_layers=1)
            quantize_missing(linear, inputs, d, params)
            assembled = assemble_full_model(tensors, d)
        ref, _ = quantize_layers_obq(linear, inputs, group_size=16,
                                     damp_frac=0.01, n_iter=20)
        for name in linear:
            np.testing.assert_array_equal(assembled[name], ref[name])
        # Bias passes through untouched.
        np.testing.assert_array_equal(
            assembled["h.0.attn.c_attn.bias"],
            tensors["h.0.attn.c_attn.bias"])

    def test_param_mismatch_on_resume_raises(self):
        _, linear, inputs = _tiny_model()
        with tempfile.TemporaryDirectory() as d:
            quantize_missing(linear, inputs, d, _params(), max_layers=1)
            with self.assertRaises(ValueError):
                quantize_missing(linear, inputs, d,
                                 _params(damp_frac=0.02), max_layers=1)

    def test_missing_inputs_raise(self):
        _, linear, inputs = _tiny_model()
        inputs = {k: v for k, v in inputs.items()
                  if k != "h.0.mlp.c_fc.weight"}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(KeyError):
                quantize_missing(linear, inputs, d, _params())

    def test_incomplete_assembly_raises(self):
        tensors, linear, inputs = _tiny_model()
        with tempfile.TemporaryDirectory() as d:
            quantize_missing(linear, inputs, d, _params(), max_layers=1)
            with self.assertRaises(KeyError):
                assemble_full_model(tensors, d)


def r3_next(r):
    return r["next"]


if __name__ == "__main__":
    unittest.main()
