
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Profiler is imported only in __main__ so the module stays lightweight when imported.


# -----------------------------
# LLaMA-style building blocks
# -----------------------------

@dataclass
class LlamaConfig:
    vocab_size: int
    max_seq_len: int

    dim: int = 512                 # hidden size
    n_layers: int = 6
    n_heads: int = 8               # query heads
    n_kv_heads: Optional[int] = None  # for GQA; if None, uses n_heads

    multiple_of: int = 256         # FFN hidden dim is rounded to multiple_of (LLaMA uses 256)
    ffn_dim_multiplier: Optional[float] = None  # optional multiplier like LLaMA-2
    rms_norm_eps: float = 1e-6

    rope_theta: float = 10000.0    # RoPE base (LLaMA uses 10000)
    dropout: float = 0.0           # LLaMA typically uses 0.0

    # Attention kernel selection
    use_sdpa: bool = True          # use torch.scaled_dot_product_attention when available
    sdpa_force_math: bool = False  # set True to force math SDPA (useful for debugging)
    debug: bool = False            # enable verbose debug prints (Q/K/V contents)

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        json_path: Optional[str] = None,
        vocab_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        use_reduced: bool = False,
    ) -> "LlamaConfig":
        """
        Load a preset model config from a JSON file and return a LlamaConfig instance.

        The JSON should contain entries with keys matching `name` and values:
          {"dim":..., "n_layers":..., "n_heads":..., "n_kv_heads":..., "ffn_dim":..., "vocab_sizes":[..], "max_seq_len":...}

        `ffn_dim` is converted to `ffn_dim_multiplier` relative to the LLaMA base hidden formula (8*dim/3).
        If `vocab_size` is provided it overrides the preset's first vocab option.
        """
        if json_path is None:
            json_path = Path(__file__).parent / "config" / "llama_configs.json"
        else:
            json_path = Path(json_path)

        with open(json_path, "r") as f:
            data = json.load(f)

        if name not in data:
            raise KeyError(f"Model config '{name}' not found in {json_path}")

        entry = data[name]
        # Optionally use the reduced testing subsection if requested and available
        if use_reduced and isinstance(entry, dict) and "reduced" in entry:
            # prefer reduced subsection for dimensions but keep top-level label if reduced has none
            reduced_entry = entry["reduced"]
            # capture labels
            reduced_label = reduced_entry.get("label") or entry.get("label")
            entry = reduced_entry
            # attach a note we used the reduced preset
            entry["_used_label"] = reduced_label
        dim = int(entry["dim"])
        n_layers = int(entry["n_layers"])
        n_heads = int(entry["n_heads"])
        n_kv_heads = int(entry.get("n_kv_heads", n_heads))
        ffn_dim = int(entry["ffn_dim"])
        vocab_options = entry.get("vocab_sizes", [32000])

        chosen_vocab = vocab_size if vocab_size is not None else int(vocab_options[0])
        ms = int(max_seq_len) if max_seq_len is not None else int(entry.get("max_seq_len", 4096))

        # Convert absolute ffn_dim to multiplier relative to base hidden (8*dim/3)
        base_hidden = int(8 * dim / 3)
        ffn_multiplier = float(ffn_dim) / float(base_hidden)

        cfg_obj = cls(
            vocab_size=chosen_vocab,
            max_seq_len=ms,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            ffn_dim_multiplier=ffn_multiplier,
        )

        # Attach optional human-readable label if present in JSON
        label_val = entry.get("label") or entry.get("_used_label")
        if label_val is not None:
            try:
                setattr(cfg_obj, "label", label_val)
            except Exception:
                pass
        # Attach optional full-form label if present in JSON
        label_full_val = entry.get("label_full")
        if label_full_val is not None:
            try:
                setattr(cfg_obj, "label_full", label_full_val)
            except Exception:
                pass

        return cfg_obj

class RMSNorm(nn.Module):
    """RMSNorm as used by LLaMA (no mean subtraction)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D]
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: [..., Dh] with Dh even
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Precomputes RoPE cos/sin tables up to max_seq_len for a given head_dim.
    Implements "original" RoPE used in LLaMA.
    """
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # [Dh/2]
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute for max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum("t,f->tf", t, self.inv_freq)  # [T, Dh/2]
        emb = torch.cat([freqs, freqs], dim=-1)            # [T, Dh]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)  # [T, Dh]
        self.register_buffer("sin_cached", emb.sin(), persistent=False)  # [T, Dh]

    def forward(self, xq: torch.Tensor, xk: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key.
        xq, xk: [B, H, T, Dh]
        start_pos: offset for cached decoding (KV-cache)
        """
        t = xq.shape[-2]
        cos = self.cos_cached[start_pos : start_pos + t]  # [T, Dh]
        sin = self.sin_cached[start_pos : start_pos + t]  # [T, Dh]

        # broadcast to [B, H, T, Dh]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        xq_out = (xq * cos) + (_rotate_half(xq) * sin)
        xk_out = (xk * cos) + (_rotate_half(xk) * sin)
        return xq_out, xk_out


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads for GQA.
    x: [B, H_kv, T, Dh] -> [B, H_kv * n_rep, T, Dh]
    """
    if n_rep == 1:
        return x
    b, h, t, dh = x.shape
    x = x[:, :, None, :, :]                 # [B, H_kv, 1, T, Dh]
    x = x.expand(b, h, n_rep, t, dh)         # [B, H_kv, n_rep, T, Dh]
    return x.reshape(b, h * n_rep, t, dh)    # [B, H, T, Dh]


class LlamaAttention(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads if cfg.n_kv_heads is not None else cfg.n_heads

        if cfg.dim % cfg.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        self.head_dim = cfg.dim // cfg.n_heads

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        self.q_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, cfg.max_seq_len, theta=cfg.rope_theta)
        self.dropout = nn.Dropout(cfg.dropout)

        # Precompute a full causal mask for the maximum context window.
        # Sliced in forward to avoid rebuilding it each call.
        self.register_buffer(
            "causal_mask",
            torch.ones((cfg.max_seq_len, cfg.max_seq_len), dtype=torch.bool).tril(),
            persistent=False,
        )
        # Internal flag to ensure Q/K/V debug print happens only once
        self._qkv_printed = False

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: [B, T, D]
        attn_mask: optional boolean mask
          - padding mask: [B, T_k] (True for valid keys)
          - full mask:    [B, T_q, T_k] (True for allowed attentions)
        start_pos: index offset for decoding with kv_cache
        kv_cache: optional dict with "k" and "v" tensors of shape [B, H_kv, T_cached, Dh]

        Returns:
          y: [B, T, D]
          new_cache: {"k": ..., "v": ...}
        """
        b, t, d = x.shape

        # Project
        q = self.q_proj(x)  # [B, T, D]
        k = self.k_proj(x)  # [B, T, H_kv*Dh]
        v = self.v_proj(x)  # [B, T, H_kv*Dh]

        # Print Q/K/V before reshape (debug mode only)
        if not getattr(self, "_qkv_printed", False):
            try:
                if getattr(self.cfg, "debug", False):
                    print("QKV BEFORE reshape shapes:", q.shape, k.shape, v.shape)
                    # show small samples to illustrate tensor contents without overwhelming the log
                    q_flat = q.reshape(-1)[:8].tolist()
                    k_flat = k.reshape(-1)[:8].tolist()
                    v_flat = v.reshape(-1)[:8].tolist()
                    print("Q sample (first 8 values):", q_flat)
                    print("K sample (first 8 values):", k_flat)
                    print("V sample (first 8 values):", v_flat)
            except Exception:
                pass

        # Shape to heads
        q = rearrange(q, "b t (h dh) -> b h t dh", h=self.n_heads)
        k = rearrange(k, "b t (h dh) -> b h t dh", h=self.n_kv_heads)
        v = rearrange(v, "b t (h dh) -> b h t dh", h=self.n_kv_heads)

        # Print Q/K/V after reshape (debug mode only)
        if not getattr(self, "_qkv_printed", False):
            try:
                if getattr(self.cfg, "debug", False):
                    print("QKV AFTER reshape shapes:", q.shape, k.shape, v.shape)
                    print("Q[0,0,0,:8]:", q[0, 0, 0, :8].tolist())
                    print("K[0,0,0,:8]:", k[0, 0, 0, :8].tolist())
                    print("V[0,0,0,:8]:", v[0, 0, 0, :8].tolist())
            except Exception:
                pass
            finally:
                self._qkv_printed = True

        # RoPE
        q, k = self.rotary(q, k, start_pos=start_pos)

        # KV cache concat (for decoding)
        if kv_cache is not None and ("k" in kv_cache) and ("v" in kv_cache):
            k = torch.cat([kv_cache["k"], k], dim=2)  # concat on T
            v = torch.cat([kv_cache["v"], v], dim=2)

        new_cache = {"k": k, "v": v}

        # GQA: repeat k/v heads to match q heads
        if self.n_kv_heads != self.n_heads:
            if self.n_heads % self.n_kv_heads != 0:
                raise ValueError("n_heads must be divisible by n_kv_heads for GQA")
            n_rep = self.n_heads // self.n_kv_heads
            k = _repeat_kv(k, n_rep)
            v = _repeat_kv(v, n_rep)

        # Attention
        # q: [B, H, Tq, Dh], k: [B, H, Tk, Dh]
        tq = q.shape[2]
        tk = k.shape[2]

        # Print matrix-multiply shapes once (debug mode only)
        if not getattr(self, "_qkv_printed", False) and getattr(self.cfg, "debug", False):
            print(f"MM Shapes: Q {q.shape}, K {k.shape}") 
        
        # Build an "allowed" boolean mask (True means allowed attention).
        # - For full-seq (start_pos=0, tq==tk), slice the cached causal mask.
        # - For KV-cache decoding (start_pos>0), the correct band is tril(diagonal=start_pos).
        if start_pos == 0 and tk == tq and tk <= self.causal_mask.shape[0]:
            causal_allow = self.causal_mask[:tq, :tk]  # [Tq, Tk]
        else:
            # With cache, queries correspond to positions [start_pos..start_pos+tq-1]
            # and keys correspond to [0..tk-1]. Allow if key_idx <= query_pos.
            causal_allow = torch.ones((tq, tk), device=x.device, dtype=torch.bool).tril(diagonal=start_pos)

        allow = causal_allow[None, None, :, :]  # [1, 1, Tq, Tk]

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                raise TypeError("attn_mask must be a boolean tensor")

            # Normalize to [B, 1, Tq, Tk] where True means valid/allowed.
            if attn_mask.dim() == 2:
                # [B, Tk] key padding mask
                attn_mask = attn_mask[:, None, None, :]
            elif attn_mask.dim() == 3:
                # [B, Tq, Tk]
                attn_mask = attn_mask[:, None, :, :]
            elif attn_mask.dim() != 4:
                raise ValueError("attn_mask must have shape [B,Tk], [B,Tq,Tk], or broadcastable to [B,1,Tq,Tk]")

            allow = allow & attn_mask

        # Prefer SDPA when available. Note: SDPA's boolean attn_mask uses True to MASK OUT positions,
        # so we pass the inverse of our "allow" mask.
        use_sdpa = self.cfg.use_sdpa and hasattr(F, "scaled_dot_product_attention")

        if use_sdpa:
            dropout_p = self.cfg.dropout if self.training else 0.0

            # Fast path: no padding mask and no cache offset. Let SDPA apply causal masking internally,
            # which avoids materializing a full [Tq,Tk] mask.
            if attn_mask is None and start_pos == 0 and tq == tk:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=True,
                )
            else:
                disallow = ~allow  # True => masked

                # Optional: force math SDPA for debugging/profiling (CUDA only).
                if self.cfg.sdpa_force_math and hasattr(torch.backends, "cuda"):
                    try:
                        torch.backends.cuda.enable_flash_sdp(False)
                        torch.backends.cuda.enable_mem_efficient_sdp(False)
                        torch.backends.cuda.enable_math_sdp(True)
                    except Exception:
                        pass

                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=disallow,
                    dropout_p=dropout_p,
                    is_causal=False,  # causal is already in the mask (supports start_pos)
                )  # [B, H, Tq, Dh]
        else:
            # Fallback explicit attention (useful for older PyTorch).
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, Tq, Tk]
            attn_logits = attn_logits.masked_fill(~allow, float("-inf"))
            attn = F.softmax(attn_logits, dim=-1)
            attn = self.dropout(attn)
            y = torch.matmul(attn, v)  # [B, H, Tq, Dh]

        y = rearrange(y, "b h t dh -> b t (h dh)")
        y = self.o_proj(y)
        y = self.dropout(y)
        return y, new_cache


class SwiGLU(nn.Module):
    """LLaMA MLP: down_proj(silu(gate_proj(x)) * up_proj(x))"""
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        hidden_dim = int(8 * cfg.dim / 3)  # LLaMA heuristic
        if cfg.ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim * cfg.ffn_dim_multiplier)
        hidden_dim = cfg.multiple_of * ((hidden_dim + cfg.multiple_of - 1) // cfg.multiple_of)

        self.gate_proj = nn.Linear(cfg.dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(cfg.dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, cfg.dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class LlamaBlock(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)
        self.attn = LlamaAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h, new_cache = self.attn(self.attn_norm(x), attn_mask=attn_mask, start_pos=start_pos, kv_cache=kv_cache)
        x = x + h
        x = x + self.mlp(self.ffn_norm(x))
        return x, new_cache


class LlamaModel(nn.Module):
    """
    Minimal LLaMA-style decoder-only Transformer with RoPE, RMSNorm, SwiGLU, and optional GQA + KV-cache.
    """
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([LlamaBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)

        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight  # weight tying

    @torch.no_grad()
    def init_kv_cache(self, batch_size: int, device: Optional[torch.device] = None) -> list:
        """
        Create an empty KV-cache list (one dict per layer).
        Keys/values will be created on first forward with cache.
        """
        device = device if device is not None else next(self.parameters()).device
        cache = [{"k": None, "v": None} for _ in range(self.cfg.n_layers)]
        return cache

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_cache: Optional[list] = None,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        input_ids: [B, T]
        attn_mask: optional boolean mask:
          - key padding mask: [B, T_total_keys] (True for valid keys)
          - full mask:        [B, T_query, T_total_keys]
        start_pos: offset for cached decoding (0 for full-seq training)
        kv_cache: optional list of per-layer caches [{"k":..., "v":...}, ...]
                  where k/v are [B, H_kv, T_cached, Dh]. For full-seq forward, pass None.

        Returns:
          logits: [B, T, V] (or hidden if return_hidden)
          kv_cache_out: updated cache list (if kv_cache was provided), else None
        """
        b, t = input_ids.shape
        if start_pos + t > self.cfg.max_seq_len:
            raise ValueError("start_pos + seq_len exceeds max_seq_len")

        x = self.tok_embeddings(input_ids)  # [B, T, D]

        kv_out = None
        if kv_cache is not None:
            if len(kv_cache) != self.cfg.n_layers:
                raise ValueError("kv_cache must have length n_layers")
            kv_out = []

        for i, layer in enumerate(self.layers):
            layer_cache = None
            if kv_cache is not None:
                # Normalize None->empty for concat logic
                k = kv_cache[i].get("k", None)
                v = kv_cache[i].get("v", None)
                layer_cache = {}
                if k is not None and v is not None:
                    layer_cache = {"k": k, "v": v}

            x, new_cache = layer(x, attn_mask=attn_mask, start_pos=start_pos, kv_cache=layer_cache)

            if kv_cache is not None:
                kv_out.append(new_cache)

        x = self.norm(x)

        if return_hidden:
            return x, kv_out

        logits = self.lm_head(x)
        return logits, kv_out


# ---- minimal usage example ----
if __name__ == "__main__":
    import argparse
    from torch.profiler import profile, ProfilerActivity

    parser = argparse.ArgumentParser(description="Minimal LLaMA-style model (RoPE/RMSNorm/SwiGLU) with optional SDPA")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto",
                        help="Execution device on Apple Silicon: 'mps' for Metal, 'cpu' for BLAS/Accelerate")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--preset", choices=["llama-3.2-1b", "llama-3.1-8b", "llama-3.1-70b"],
                        default="llama-3.2-1b",
                        help="Use a preset model configuration from config/llama_configs.json (default: llama-3.2-1b)")
    parser.add_argument("--no-reduced", action="store_true",
                        help="Do not use the preset's 'reduced' testing configuration (use full-size preset)")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-sdpa", action="store_true", help="Disable scaled_dot_product_attention path")
    parser.add_argument("--sdpa-math", action="store_true", help="(CUDA only) force math SDPA kernel")
    parser.add_argument("--profile", action="store_true", help="Run torch.profiler around a single forward pass")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup forwards before profiling")
    parser.add_argument("--use-pad-mask", action="store_true",
                        help="Pass an all-True padding mask (useful to test mask overhead). Default: no attn_mask")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() (recommended for CPU, experimental for MPS)")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints of Q/K/V contents")
    args = parser.parse_args()

    # Device selection (Apple Silicon)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
        device = torch.device("mps")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Helpful for CPU matmul quality/speed tradeoffs in newer PyTorch.
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if args.preset:
        cfg = LlamaConfig.from_pretrained(
            args.preset,
            vocab_size=args.vocab,
            max_seq_len=args.max_seq_len,
            use_reduced=not args.no_reduced,
        )
        # apply runtime overrides
        cfg.use_sdpa = not args.no_sdpa
        cfg.sdpa_force_math = args.sdpa_math
        cfg.dropout = args.dropout
        cfg.rope_theta = args.rope_theta
        # If kv-heads explicitly provided, override preset
        if args.kv_heads is not None:
            cfg.n_kv_heads = args.kv_heads
    else:
        cfg = LlamaConfig(
            vocab_size=args.vocab,
            max_seq_len=args.max_seq_len,
            dim=args.dim,
            n_layers=args.layers,
            n_heads=args.heads,
            n_kv_heads=args.kv_heads if args.kv_heads is not None else args.heads,
            dropout=args.dropout,
            rope_theta=args.rope_theta,
            use_sdpa=not args.no_sdpa,
            sdpa_force_math=args.sdpa_math,
        )

# apply debug flag from CLI (set after cfg is created)
    cfg.debug = bool(args.debug)

    # Print model summary
    model_name = getattr(cfg, "label", args.preset if args.preset else "custom")
    # compute FF size using the same logic as SwiGLU
    base_hidden = int(8 * cfg.dim / 3)
    if cfg.ffn_dim_multiplier is not None:
        ffn_hidden = int(base_hidden * cfg.ffn_dim_multiplier)
    else:
        ffn_hidden = base_hidden
    # round to multiple_of
    ffn_hidden = cfg.multiple_of * ((ffn_hidden + cfg.multiple_of - 1) // cfg.multiple_of)

    print("Models:")
    print(f"Name: {model_name}")
    # Prefer explicit full-form label if available
    if hasattr(cfg, "label_full"):
        print(cfg.label_full)
    else:
        # Full-form mapping
        full_map = {
            "D": "HiddenDim",
            "L": "NumLayers",
            "H": "QueryHeads",
            "Hkv": "KVHeads",
            "FF": "FFN",
            "V": "VocabSize",
        }
        print(
            f"Dimension: {full_map['D']}={cfg.dim}, {full_map['L']}={cfg.n_layers}, {full_map['H']}={cfg.n_heads}, {full_map['Hkv']}={cfg.n_kv_heads}, {full_map['FF']}={ffn_hidden}, {full_map['V']}={cfg.vocab_size}"
        )
    model = LlamaModel(cfg).to(device)

    if args.compile:
    	print("Compiling model with torch.compile()")
    	model = torch.compile(
        	model,
        	mode="max-autotune",     # best for CPU
        	fullgraph=False,         # safer with attention branches
    	)

    model.eval()

    B, T = args.batch, args.seq_len
    input_ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    attn_mask = None
    if args.use_pad_mask:
        attn_mask = torch.ones(B, T, dtype=torch.bool, device=device)

    # Warmup (important for MPS and for caching allocator)
    with torch.no_grad():
        for _ in range(max(args.warmup, 0)):
            _ = model(input_ids, attn_mask=attn_mask)

    if not args.profile:
        with torch.no_grad():
            logits, _ = model(input_ids, attn_mask=attn_mask)
        print(logits.shape)
    else:
        acts = [ProfilerActivity.CPU]
        if device.type == "mps":
            acts.append(ProfilerActivity.MPS)
        if torch.cuda.is_available():
            acts.append(ProfilerActivity.CUDA)

        with torch.no_grad():
            with profile(activities=acts, record_shapes=True, profile_memory=True, with_flops=True) as prof:
                logits, _ = model(input_ids, attn_mask=attn_mask)

        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=60))
        print(logits.shape)
