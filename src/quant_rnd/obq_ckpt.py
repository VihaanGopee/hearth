"""Checkpointed full-model OBQ quantization (the session-timeout unblocker).

The 48-layer `ppl.py --obq-all` run cannot finish inside one ~25-min
session on this VM (two runs killed by the execution timeout on
2026-09-21), so this module checkpoints each layer's quantized weights
to a directory (under research/data/, gitignored) as soon as the layer
finishes quantizing. Later runs read the manifest, skip already-done
layers, and quantize the next chunk; when every layer is checkpointed,
the full fp32 weight dict is assembled and the final perplexity is
measured — the verdict against the 795.96 naive anchor (< ~400 keeps
the fidelity path alive).

On-disk format: <dir>/manifest.json plus one <sanitized>.npz per layer
holding the quantized float32 weight under key "Wq". The stored Wq is
the exact array obq.quantize_layers_obq returns, and quantization is
deterministic, so a resumed run assembles a dict that is bit-identical
to what an in-memory --obq-all would produce.

Hyperparameter guard: the manifest records group_size, damp_frac,
n_iter, and a hash of the calibration token sequence. A run whose
params differ from the manifest raises ValueError rather than mixing
incompatible checkpoints.
"""
import hashlib
import json
import os
import re

import numpy as np

from .obq import quantize_layer_obq

MANIFEST_NAME = "manifest.json"

# Pin the hyperparameters the checkpointed run fixes, so the manifest
# and the tests agree on what "same params" means.
N_ITER = 20  # quantize_layer_obq default; not currently CLI-exposed


def safe_layer_filename(name: str) -> str:
    """Map a tensor name (e.g. 'h.0.attn.c_attn.weight') to a flat .npz
    filename with no path components."""
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    return stem + ".npz"


def run_params(group_size: int, damp_frac: float,
               token_ids) -> dict:
    """The manifest-recorded parameter fingerprint of one checkpointed
    run. token_ids is the calibration/eval token sequence."""
    ids = np.asarray(token_ids, dtype=np.int64)
    return {
        "group_size": int(group_size),
        "damp_frac": float(damp_frac),
        "n_iter": N_ITER,
        "block_size": int(group_size),  # quantize_layer_obq default
        "text_sha": hashlib.sha1(ids.tobytes()).hexdigest(),
        "n_tokens": int(ids.size),
    }


def read_manifest(ckpt_dir: str) -> dict:
    """The stored manifest ({"params": ..., "layers": {name: file}}) or
    an empty dict when no manifest exists yet."""
    path = os.path.join(ckpt_dir, MANIFEST_NAME)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _write_manifest(ckpt_dir: str, manifest: dict) -> None:
    tmp = os.path.join(ckpt_dir, MANIFEST_NAME + ".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, os.path.join(ckpt_dir, MANIFEST_NAME))


def check_params_compatible(manifest: dict, params: dict) -> None:
    """Raise ValueError if an existing manifest was built with different
    run parameters (would silently mix incompatible checkpoints)."""
    if not manifest:
        return
    old = manifest.get("params", {})
    for key in ("group_size", "damp_frac", "n_iter", "block_size",
                "text_sha"):
        if old.get(key) != params.get(key):
            raise ValueError(
                f"checkpoint manifest param {key!r} mismatch: existing "
                f"{old.get(key)!r} vs requested {params.get(key)!r}; "
                f"use a fresh checkpoint directory")


def write_layer_ckpt(ckpt_dir: str, name: str, Wq: np.ndarray,
                     params: dict) -> None:
    """Checkpoint one layer's quantized weights and update the manifest
    (atomic write + rename, so a kill leaves no half-written file)."""
    os.makedirs(ckpt_dir, exist_ok=True)
    fname = safe_layer_filename(name)
    # np.savez appends ".npz" unless the name already ends with it, so
    # the temp name must keep the extension for the atomic rename below.
    tmp = os.path.join(ckpt_dir, fname + ".tmp.npz")
    np.savez(tmp, Wq=np.ascontiguousarray(Wq, dtype=np.float32))
    os.replace(tmp, os.path.join(ckpt_dir, fname))
    manifest = read_manifest(ckpt_dir)
    if not manifest:
        manifest = {"params": params, "layers": {}}
    else:
        check_params_compatible(manifest, params)
    manifest["layers"][name] = fname
    _write_manifest(ckpt_dir, manifest)


def quantize_missing(linear: dict, inputs: dict, ckpt_dir: str,
                     params: dict, max_layers: int | None = None) -> dict:
    """Quantize every linear layer not yet in the manifest, checkpointing
    each as it finishes; layers are processed in sorted name order so
    the resume point is deterministic.

    Returns {"new": n_newly_quantized, "done": n_total_done,
    "total": n_layers, "next": next_layer_name_or_None}.
    max_layers caps how many NEW layers this call quantizes (0 means a
    pure status probe). Input dicts are not modified.
    """
    check_params_compatible(read_manifest(ckpt_dir), params)
    targets = sorted(linear)
    manifest = read_manifest(ckpt_dir)
    todo = [n for n in targets if n not in manifest.get("layers", {})]
    if max_layers is not None:
        todo = todo[:max_layers]
    new = 0
    for name in todo:
        if name not in inputs:
            raise KeyError(f"no captured activations for {name!r}")
        Wq, _bpw = quantize_layer_obq(
            linear[name], inputs[name],
            group_size=params["group_size"],
            block_size=params["block_size"],
            damp_frac=params["damp_frac"], n_iter=params["n_iter"])
        write_layer_ckpt(ckpt_dir, name, Wq, params)
        new += 1
        manifest = read_manifest(ckpt_dir)
    remaining = [n for n in targets if n not in manifest.get("layers", {})]
    return {"new": new, "done": len(targets) - len(remaining),
            "total": len(targets),
            "next": remaining[0] if remaining else None}


def assemble_full_model(tensors: dict, ckpt_dir: str) -> dict:
    """Rebuild the {name: float32 ndarray} weight dict with every
    checkpointed layer substituted for its fp32 original — bit-identical
    to what quantize_model_obq returns. Raises KeyError if any linear
    layer is missing from the manifest (use quantize_missing first)."""
    from .realweights import linear_weight_tensors
    manifest = read_manifest(ckpt_dir)
    layers = manifest.get("layers", {})
    linear = set(linear_weight_tensors(tensors))
    missing = sorted(n for n in linear if n not in layers)
    if missing:
        raise KeyError(f"{len(missing)} linear layers not yet "
                       f"checkpointed (first: {missing[0]!r})")
    out = {n: np.ascontiguousarray(t, dtype=np.float32)
           for n, t in tensors.items()}
    for name in linear:
        path = os.path.join(ckpt_dir, layers[name])
        with np.load(path) as z:
            Wq = z["Wq"]
        out[name] = np.ascontiguousarray(Wq, dtype=np.float32).reshape(
            tensors[name].shape)
    return out
