"""Real-weight SQNR probe for the quant R&D prototypes.

Extends bench.py's synthetic SQNR comparison onto real model weights:
loads a small model's .safetensors with a dependency-free parser (NumPy
only - no torch/safetensors required), samples contiguous group-aligned
blocks from the Linear weight matrices, runs every scheme in SCHEMES, and
reports the SQNR ranking.

This answers a standing roadmap question: does the synthetic ranking
(Lloyd k-means > ternary_lloyd > ternary_outlier > dual_scale_ternary)
reproduce on real weight distributions? Synthetic tensors are i.i.d.
with planted outliers; real weights have per-channel structure (row/col
scale variance, real outliers), which can shift the ranking either way.

Honest scope: SQNR on real weights is still a proxy - it says nothing
about perplexity. It does remove the "synthetic distribution" caveat
from the scheme ranking, which is the step this probe exists to take.

Usage: python3 -m src.quant_rnd.realweights <model.safetensors>
       [--seed N] [--groups-per-matrix G] [--matrices REGEX]
"""
import argparse
import json
import re
import struct

import numpy as np

from .bench import print_table, sqnr_db
from .schemes import GROUP_SIZE, SCHEMES

# Safetensors dtypes this parser supports. BF16/I64/F64/U8 exist in the
# wild but NumPy can't hold BF16 without ml_dtypes, so we refuse them
# explicitly instead of silently mis-decoding.
_DTYPE_MAP = {
    "F32": np.float32,
    "F16": np.float16,
    "I8": np.int8,
    "I32": np.int32,
}


def read_safetensors(path: str) -> dict:
    """Parse a .safetensors file into {name: ndarray}, NumPy only.

    Format: 8-byte little-endian header length N, N bytes of JSON
    ({name: {"dtype", "shape", "data_offsets": [start, end]}}), then the
    raw little-endian tensor bytes. Tensors are returned as C-order
    arrays reshaped to their stored shape.
    """
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
        tensors = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype_name = meta["dtype"]
            if dtype_name not in _DTYPE_MAP:
                raise ValueError(
                    f"tensor {name!r}: dtype {dtype_name!r} not supported "
                    f"by the dependency-free parser (have {sorted(_DTYPE_MAP)})"
                )
            np_dtype = _DTYPE_MAP[dtype_name]
            start, end = meta["data_offsets"]
            f.seek(data_start + start)
            raw = f.read(end - start)
            arr = np.frombuffer(raw, dtype=np_dtype).reshape(meta["shape"])
            tensors[name] = np.ascontiguousarray(arr)
        return tensors


def linear_weight_tensors(tensors: dict) -> dict:
    """Keep the 2-D Linear weight matrices, sorted by name.

    Keeps entries named like "<layer>.weight" whose value is 2-D. This
    skips biases (".bias"), 1-D LayerNorm scales, and embeddings are kept
    out by the ndim==2 rule only if they are 2-D - embeddings ARE 2-D
    (vocab x hidden), so they are excluded by name instead: tensors whose
    name contains "embed" / "wte" / "wpe" are skipped. Rationale: the
    schemes target the matmul weights; embedding tables are a separate
    quantization story (usually kept at higher precision).
    """
    out = {}
    for name in sorted(tensors):
        if not name.endswith(".weight"):
            continue
        low = name.lower()
        if "embed" in low or "wte" in low or "wpe" in low:
            continue
        if tensors[name].ndim != 2:
            continue
        out[name] = tensors[name].astype(np.float32)
    return out


def sample_groups(tensor: np.ndarray, n_groups: int,
                  group_size: int = GROUP_SIZE,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample n_groups random contiguous group_size blocks, flattened.

    Groups are the unit the quantizers operate on, so sampling whole
    blocks (not scattered weights) keeps the probe faithful. The trailing
    partial block is never sampled. Deterministic given the rng.
    """
    rng = rng if rng is not None else np.random.default_rng()
    flat = tensor.ravel()
    n_blocks = flat.shape[0] // group_size
    if n_blocks == 0:
        raise ValueError("tensor smaller than one group")
    k = min(n_groups, n_blocks)
    blocks = rng.choice(n_blocks, size=k, replace=False)
    blocks.sort()
    return np.concatenate(
        [flat[b * group_size:(b + 1) * group_size] for b in blocks]
    ).astype(np.float32)


def rank_on_real_weights(tensors: dict, n_groups_per_matrix: int = 64,
                         group_size: int = GROUP_SIZE, seed: int = 7,
                         schemes: dict | None = None) -> list:
    """Run every scheme on sampled blocks of each matrix; mean SQNR.

    Each matrix contributes one sampled slice (n_groups_per_matrix
    groups); the reported SQNR is the mean over matrices, so a huge
    matrix doesn't dominate a tiny one. bpw is the scheme's stored value
    (group-size dependent); results are sorted by SQNR descending.
    """
    schemes = schemes if schemes is not None else SCHEMES
    rng = np.random.default_rng(seed)
    totals = {name: 0.0 for name in schemes}
    counts = {name: 0 for name in schemes}
    bpw = {}
    for _name, matrix in tensors.items():
        w = sample_groups(matrix, n_groups_per_matrix, group_size, rng)
        for name, fn in schemes.items():
            q = fn(w, group_size=group_size) if _takes_group_size(fn) else fn(w)
            totals[name] += sqnr_db(w, q.reconstruct())
            counts[name] += 1
            bpw[name] = q.bpw
    results = [
        {"scheme": name, "sqnr_db": totals[name] / counts[name],
         "bpw": bpw[name], "n_matrices": counts[name]}
        for name in schemes
    ]
    results.sort(key=lambda r: r["sqnr_db"], reverse=True)
    return results


def _takes_group_size(fn) -> bool:
    """All scheme functions take group_size except none currently do.

    Kept as a hook: rank_on_real_weights passes group_size through when
    the scheme signature supports it, so a future scheme with a different
    signature won't crash the probe.
    """
    import inspect

    return "group_size" in inspect.signature(fn).parameters


def print_report(results: list, n_groups_per_matrix: int) -> None:
    print(f"real-weight SQNR probe ({results[0]['n_matrices']} matrices, "
          f"{n_groups_per_matrix} groups/matrix)")
    print_table(results)


def parse_args(argv: list | None = None) -> argparse.Namespace:
    """Argument parser, factored out of main() so tests can call it."""
    ap = argparse.ArgumentParser(description="real-weight SQNR probe")
    ap.add_argument("safetensors", help="path to model.safetensors")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--groups-per-matrix", type=int, default=64)
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE,
                    help="quantizer group size (e.g. 64, 128, 256)")
    ap.add_argument("--matrices", type=str, default="",
                    help="regex filter on tensor names")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    tensors = read_safetensors(args.safetensors)
    mats = linear_weight_tensors(tensors)
    if args.matrices:
        rx = re.compile(args.matrices)
        mats = {n: t for n, t in mats.items() if rx.search(n)}
    print(f"{len(mats)} linear weight matrices selected "
          f"(group size {args.group_size})")
    results = rank_on_real_weights(mats,
                                   n_groups_per_matrix=args.groups_per_matrix,
                                   group_size=args.group_size,
                                   seed=args.seed)
    print_report(results, args.groups_per_matrix)


if __name__ == "__main__":
    main()
