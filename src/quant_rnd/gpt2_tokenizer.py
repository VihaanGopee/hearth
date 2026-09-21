"""Dependency-free GPT-2 byte-level BPE tokenizer.

Loads vocab.json + merges.txt (small files, ~1.5 MB, stored under
research/data/tokenizer/) and implements the HuggingFace GPT2Tokenizer
encoding convention with NumPy-free stdlib only.

Pre-tokenization uses an ASCII-safe rendering of the GPT-2 regex pattern
('s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|
\\s+(?!\\S)|\\s+). Python's `re` has no \\p{} classes, so \\p{L}/\\p{N}
are rendered as [a-zA-Z]/[0-9]; the two patterns are exactly equivalent
on ASCII input and degrade gracefully (never crash) on non-ASCII input
thanks to the byte-level fallback. The eval text used by the perplexity
harness is pure ASCII, so this is exact where it matters.

Decode is exact: token ids -> vocab strings -> bytes -> utf-8.
"""

import json
import os
import re

# ASCII-safe GPT-2 pre-tokenizer pattern (see module docstring).
_PRETOKEN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^a-zA-Z0-9\s]+"
    r"|\s+(?!\S)|\s+"
)


def bytes_to_unicode() -> dict:
    """Byte -> unicode char map (HF's bytes_to_unicode, verbatim logic).

    Maps the 256 byte values onto printable unicode code points so that
    BPE can operate on "characters" without ever producing whitespace or
    control characters in the merge table.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def _get_pairs(word: tuple) -> set:
    """Adjacent symbol pairs of a BPE word."""
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


def bpe(token: str, ranks: dict, cache: dict) -> str:
    """Apply BPE merges to one pre-token; returns space-joined symbols."""
    if token in cache:
        return cache[token]
    word = tuple(token)
    pairs = _get_pairs(word)
    if not pairs:
        cache[token] = token
        return token
    while True:
        bigram = min(pairs, key=lambda p: ranks.get(p, float("inf")))
        if bigram not in ranks:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
            except ValueError:
                new_word.extend(word[i:])
                break
            new_word.extend(word[i:j])
            i = j
            if i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = tuple(new_word)
        if len(word) == 1:
            break
        pairs = _get_pairs(word)
    out = " ".join(word)
    cache[token] = out
    return out


class GPT2Tokenizer:
    """Minimal GPT-2 encoder/decoder (HF GPT2Tokenizer convention)."""

    def __init__(self, tokenizer_dir: str):
        with open(os.path.join(tokenizer_dir, "vocab.json"), encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(os.path.join(tokenizer_dir, "merges.txt"), encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
        # First line is "#version: 0.2"; pairs follow as "a b".
        pairs = [tuple(ln.split(" ")) for ln in lines if ln and not ln.startswith("#")]
        self.bpe_ranks = {pair: i for i, pair in enumerate(pairs)}
        self.cache = {}

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def encode(self, text: str) -> list:
        """Encode text to a list of token ids (no BOS/EOS added)."""
        ids = []
        for pre in _PRETOKEN.findall(text):
            token = "".join(self.byte_encoder[b] for b in pre.encode("utf-8"))
            for symbol in bpe(token, self.bpe_ranks, self.cache).split(" "):
                ids.append(self.encoder[symbol])
        return ids

    def decode(self, ids: list) -> str:
        """Decode token ids back to text (exact round-trip on encode output)."""
        text = "".join(self.decoder[i] for i in ids)
        byte_vals = bytearray(self.byte_decoder[c] for c in text)
        return byte_vals.decode("utf-8", errors="replace")
