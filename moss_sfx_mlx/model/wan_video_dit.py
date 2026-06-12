"""MLX transpose of moss_soundeffect_v2/diffsynth/models/wan_video_dit.py.

Only the classes/functions the audio model uses are ported (DiTBlock and its
constituents). Structure, names, and op order match upstream 1:1; differences
are torch->mlx substitutions only, noted inline. Video-only classes (WanModel,
3D rope, converters) are intentionally absent — the audio model never touches
them.

RoPE representation: upstream precomputes complex64 `freqs_cis` and uses
`.real`/`.imag` inside `rope_apply`. MLX complex support is partial, so freqs
are carried as a (cos, sin) tuple of float32 arrays with identical values.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def flash_attention(q: mx.array, k: mx.array, v: mx.array, num_heads: int):
    # Upstream dispatches to flash-attn/sage/SDPA; all are math-equivalent to
    # SDPA. MLX has one fused path: mx.fast.scaled_dot_product_attention.
    b, s, _ = q.shape
    q = q.reshape(b, q.shape[1], num_heads, -1).transpose(0, 2, 1, 3)
    k = k.reshape(b, k.shape[1], num_heads, -1).transpose(0, 2, 1, 3)
    v = v.reshape(b, v.shape[1], num_heads, -1).transpose(0, 2, 1, 3)
    x = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)
    x = x.transpose(0, 2, 1, 3).reshape(b, s, -1)
    return x


def modulate(x: mx.array, shift: mx.array, scale: mx.array):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    # Upstream computes in float64; MLX GPU has no float64 so this runs fp32.
    # Parity impact measured at the time-embedding gate; revisit if it fails.
    half = dim // 2
    freqs = mx.power(10000, -mx.arange(half, dtype=mx.float32) / half)
    sinusoid = position.astype(mx.float32)[:, None] * freqs[None, :]
    x = mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=1)
    return x


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0, s: float = 1.0):
    # 1d rope precompute — float64 in numpy (matches upstream), stored as
    # float32 (cos, sin); upstream's complex64 carries the same precision.
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float64) / dim))
    pos = np.arange(end, dtype=np.float64) * s
    freqs = np.outer(pos, freqs)
    return mx.array(np.cos(freqs).astype(np.float32)), mx.array(np.sin(freqs).astype(np.float32))


def rope_apply(x, freqs, num_heads):
    # Interleaved even/odd pair rotation, fp32 compute, cast back — exactly
    # upstream's real-valued formulation (wan_video_dit.py:104-118).
    out_dtype = x.dtype
    b, s, _ = x.shape
    x = x.reshape(b, s, num_heads, -1).astype(mx.float32)
    x = x.reshape(*x.shape[:-1], -1, 2)                     # [b, s, n, d/2, 2]
    x_even, x_odd = x[..., 0], x[..., 1]
    cos, sin = freqs                                        # each [s, 1, d/2]
    x_out = mx.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
        axis=-1,
    ).reshape(b, s, -1)                                     # [b, s, n*d]
    return x_out.astype(out_dtype)


class RMSNorm(nn.Module):
    # torch.nn.RMSNorm(dim, eps) equivalent — note: applied over the FULL
    # model dim (1536) before the head split, not per-head.
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def __call__(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def __call__(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def __call__(self, x: mx.array, y: mx.array):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approx='precise'), nn.Linear(ffn_dim, dim))
        self.modulation = mx.random.normal((1, 6, dim)) / dim**0.5
        self.gate = GateModule()

    def __call__(self, x, context, t_mod, freqs):
        has_seq = t_mod.ndim == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(
            self.modulation.astype(t_mod.dtype) + t_mod, 6, axis=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(approx='none'),  # upstream nn.GELU() = exact erf
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = mx.zeros((1, 514, 1280))

    def __call__(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.astype(x.dtype)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size, eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = mx.random.normal((1, 2, dim)) / dim**0.5

    def __call__(self, x, t_mod):
        if t_mod.ndim == 3:
            shift, scale = mx.split(
                mx.expand_dims(self.modulation, 0).astype(t_mod.dtype) + mx.expand_dims(t_mod, 2), 2, axis=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            # t_mod is originally [B, C]; broadcasting works for B=1 but not for
            # B>1 against [1, 2, C], so reshape explicitly here.
            shift, scale = mx.split(
                self.modulation.astype(t_mod.dtype) + mx.expand_dims(t_mod, 1), 2, axis=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x
