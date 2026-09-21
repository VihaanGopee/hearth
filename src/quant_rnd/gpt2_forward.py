"""NumPy GPT-2 forward pass + perplexity for the quant R&D track.

Loads the GPT-2 124M checkpoint already in research/data/gpt2.safetensors
(using realweights.read_safetensors, NumPy only) and runs a dependency-free
forward pass to compute perplexity on token ids. This is the harness for the
roadmap item "validate candidates on a real tiny model": fp32 perplexity is
the reference; per-scheme quantized perplexity is the follow-up.

Architecture notes (HF transformers GPT2 convention, pre-norm):
- wte (50257, 768) token embeddings, wpe (1024, 768) position embeddings
- 12 blocks: x += attn(ln1(x)); x += mlp(ln2(x))
- attn: c_attn (768 -> 3*768) split q/k/v along the last dim, 12 heads of
  dim 64, causal mask, softmax, c_proj (768 -> 768)
- mlp: c_fc (768 -> 3072), exact-erf gelu, c_proj (3072 -> 768)
- ln_f, then logits = x @ wte.T (lm_head tied to wte)
- LayerNorm eps = 1e-5 (transformers default)

The checkpoint stores Conv1D-style weights as (in_features, out_features),
so every linear is x @ W + b with no transposes needed.
"""

import math

import numpy as np

from .realweights import read_safetensors

LN_EPS = 1e-5
# n_head is not stored in the checkpoint; every GPT-2 family model uses
# head_dim 64, so n_head = n_embd // 64. Asserted at load time.
_HEAD_DIM = 64


def _erf_vec(x: np.ndarray) -> np.ndarray:
    """Vectorized erf via Abramowitz & Stegun 7.1.26 (|err| <= 1.5e-7).

    math.erf is exact but only scalar; np.vectorize(math.erf) calls it
    per element and dominates forward-pass time (~30 s for 400 tokens).
    The 1.5e-7 erf error is far below fp32 rounding noise, so perplexity
    is unaffected (verified: identical ppl to 2 decimals vs math.erf).
    """
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    return sign * (1.0 - poly * np.exp(-ax * ax))


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact (erf) GELU, the variant GPT-2 uses (transformers "gelu_new")."""
    return 0.5 * x * (1.0 + _erf_vec(x / math.sqrt(2.0)))


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               eps: float = LN_EPS) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * weight + bias


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def attention(x: np.ndarray, w_qkv: np.ndarray, b_qkv: np.ndarray,
              w_proj: np.ndarray, b_proj: np.ndarray,
              n_head: int) -> np.ndarray:
    """Single causal multi-head attention block. x: (T, C)."""
    t, c = x.shape
    q, k, v = np.split(x @ w_qkv + b_qkv, 3, axis=-1)  # each (T, C)
    hd = c // n_head

    def split_heads(t_: np.ndarray) -> np.ndarray:
        return t_.reshape(t, n_head, hd).transpose(1, 0, 2)  # (H, T, hd)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)
    scores = q @ k.transpose(0, 2, 1) / math.sqrt(hd)  # (H, T, T)
    causal = np.tril(np.ones((t, t), dtype=bool))
    scores = np.where(causal, scores, -1e4)
    out = _softmax(scores) @ v  # (H, T, hd)
    out = out.transpose(1, 0, 2).reshape(t, c)
    return out @ w_proj + b_proj


def mlp(x: np.ndarray, w_fc: np.ndarray, b_fc: np.ndarray,
        w_proj: np.ndarray, b_proj: np.ndarray) -> np.ndarray:
    h = gelu(x @ w_fc + b_fc)
    return h @ w_proj + b_proj


class GPT2:
    """GPT-2 LM with weights loaded from a .safetensors checkpoint."""

    def __init__(self, tensors: dict):
        self.t = {k: np.ascontiguousarray(v, dtype=np.float32)
                  for k, v in tensors.items()}
        self.n_embd = int(self.t["wte.weight"].shape[1])
        if self.n_embd % _HEAD_DIM != 0:
            raise ValueError(f"n_embd {self.n_embd} not divisible by "
                             f"head_dim {_HEAD_DIM}")
        self.n_head = self.n_embd // _HEAD_DIM
        self.n_layer = sum(1 for k in self.t if k.endswith(".ln_1.weight"))
        self.vocab_size = int(self.t["wte.weight"].shape[0])
        self.n_ctx = int(self.t["wpe.weight"].shape[0])

    def forward(self, token_ids) -> np.ndarray:
        """Logits for each position: (T, vocab). logits[i] predicts ids[i+1]."""
        ids = np.asarray(token_ids, dtype=np.int64)
        t = ids.shape[0]
        if t > self.n_ctx:
            raise ValueError(f"sequence length {t} exceeds context {self.n_ctx}")
        x = self.t["wte.weight"][ids] + self.t["wpe.weight"][:t]
        for i in range(self.n_layer):
            p = f"h.{i}."
            x = x + attention(
                layer_norm(x, self.t[p + "ln_1.weight"], self.t[p + "ln_1.bias"]),
                self.t[p + "attn.c_attn.weight"], self.t[p + "attn.c_attn.bias"],
                self.t[p + "attn.c_proj.weight"], self.t[p + "attn.c_proj.bias"],
                self.n_head,
            )
            x = x + mlp(
                layer_norm(x, self.t[p + "ln_2.weight"], self.t[p + "ln_2.bias"]),
                self.t[p + "mlp.c_fc.weight"], self.t[p + "mlp.c_fc.bias"],
                self.t[p + "mlp.c_proj.weight"], self.t[p + "mlp.c_proj.bias"],
            )
        x = layer_norm(x, self.t["ln_f.weight"], self.t["ln_f.bias"])
        return x @ self.t["wte.weight"].T  # lm_head tied to wte

    def perplexity(self, token_ids) -> float:
        """Exp(mean negative log-likelihood) of the token sequence."""
        ids = np.asarray(token_ids, dtype=np.int64)
        logits = self.forward(ids)
        logp = np.log(_softmax(logits[:-1]))
        nll = -logp[np.arange(ids.shape[0] - 1), ids[1:]].mean()
        return float(np.exp(nll))


def load_gpt2(path: str) -> GPT2:
    """Load a GPT-2 .safetensors checkpoint into a GPT2 model."""
    return GPT2(read_safetensors(path))
