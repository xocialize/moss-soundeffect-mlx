"""MLX transpose of moss_soundeffect_v2/diffsynth/models/wan_audio_dit.py.

Structure, names, and op order match upstream 1:1 (torch->mlx substitutions
only). Image/camera-control branches the SFX checkpoint never exercises
(`has_ref_conv`, `add_control_adapter`) raise NotImplementedError instead of
silently diverging.

NOTE on the inference path: upstream inference goes through
`model_fn_wan_video` (pipelines/wan_audio.py), not `WanAudioModel.forward` —
the two are op-identical for this config (no image/vace/ref branches), so this
forward IS the oracle-equivalent path. Parity-test against either.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .wan_video_dit import (
    DiTBlock,
    Head,
    MLP,
    precompute_freqs_cis,
    sinusoidal_embedding_1d,
)


def modulate(x: mx.array, shift: mx.array, scale: mx.array):
    # x is fp32 after layer norm
    return (x * (1 + scale) + shift).astype(shift.dtype)


def legacy_precompute_freqs_cis_1d(dim: int, end: int = 16384, theta: float = 10000.0, base_tps=4.0, target_tps=44100 / 2048):
    s = float(base_tps) / float(target_tps)
    # 1d rope precompute (oobleck VAE path — unused by the dac checkpoint)
    f_cos, f_sin = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta, s)
    no_cos, no_sin = precompute_freqs_cis(dim // 3, end, theta, s)
    # No positional encoding applied to the remaining dimensions
    # (upstream: torch.ones_like(complex) -> cos=1, sin=0).
    no_cos, no_sin = mx.ones_like(no_cos), mx.zeros_like(no_sin)
    return (f_cos, f_sin), (no_cos, no_sin), (no_cos, no_sin)


def _chunk3(cos, sin):
    # torch.chunk(3, dim=-1) sizes: ceil(n/3) for all but the last chunk.
    n = cos.shape[-1]
    c = math.ceil(n / 3)
    idx = [(0, c), (c, min(2 * c, n)), (min(2 * c, n), n)]
    return tuple((cos[..., a:b], sin[..., a:b]) for a, b in idx)


def precompute_freqs_cis_1d(dim: int, end: int = 16384, theta: float = 10000.0):
    cos, sin = precompute_freqs_cis(dim, end, theta)
    return _chunk3(cos, sin)


class WanAudioModel(nn.Module):

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size,
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        vae_type: str = "oobleck",
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = tuple(patch_size)
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.vae_type = vae_type
        # Upstream: nn.Conv1d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        # operating on (B, C, T). MLX conv1d is channel-last; patchify handles
        # the transpose. Weight layout (O, I, K) -> (O, K, I) at conversion.
        self.patch_embedding = nn.Conv1d(
            in_dim, dim, kernel_size=self.patch_size[0], stride=self.patch_size[0]
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approx='precise'),  # upstream GELU(approximate='tanh')
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = [
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ]
        self.head = Head(dim, out_dim, self.patch_size, eps)
        head_dim = dim // num_heads
        if vae_type == "oobleck":
            freqs = legacy_precompute_freqs_cis_1d(head_dim, base_tps=4.0, target_tps=44100 / 2048)
        elif vae_type == "dac":
            freqs = precompute_freqs_cis_1d(head_dim)
        else:
            raise ValueError(f"Invalid VAE type: {vae_type}")
        # Underscore prefix: non-parameter buffers in MLX (excluded from
        # parameters()/update()), mirroring persistent=False buffers upstream.
        self._freqs_cis_0 = freqs[0]
        self._freqs_cis_1 = freqs[1]
        self._freqs_cis_2 = freqs[2]

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            raise NotImplementedError("has_ref_conv is not used by MOSS-SoundEffect-v2.0")
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            raise NotImplementedError("add_control_adapter is not used by MOSS-SoundEffect-v2.0")
        self.control_adapter = None

    @property
    def freqs(self):
        # Backwards-compatible accessor: external code can still use self.freqs[i].
        return (self._freqs_cis_0, self._freqs_cis_1, self._freqs_cis_2)

    def patchify(self, x: mx.array, control_camera_latents_input=None):
        if control_camera_latents_input is not None:
            raise NotImplementedError("camera control is not used by MOSS-SoundEffect-v2.0")
        # x: (b, c, f) channel-first like upstream; MLX conv1d wants (b, f, c).
        x = self.patch_embedding(x.transpose(0, 2, 1))      # (b, f, dim)
        grid_size = (x.shape[1],)
        # Upstream rearranges conv output 'b c f -> b f c'; already (b, f, c) here.
        return x, grid_size  # x, grid_size: (f)

    def unpatchify(self, x: mx.array, grid_size):
        # rearrange 'b f (p c) -> b c (f p)'
        f = grid_size[0]
        p = self.patch_size[0]
        b = x.shape[0]
        x = x.reshape(b, f, p, -1)                          # (b, f, p, c)
        x = x.transpose(0, 3, 1, 2)                         # (b, c, f, p)
        return x.reshape(b, x.shape[1], f * p)              # (b, c, f*p)

    def __call__(self,
                 x: mx.array,
                 timestep: mx.array,
                 context: mx.array,
                 clip_feature=None,
                 y=None,
                 **kwargs,
                 ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).reshape(t.shape[0], 6, self.dim)
        context = self.text_embedding(context)

        if self.has_image_input:
            x = mx.concatenate([x, y], axis=1)  # (b, c_x + c_y, f)
            clip_embdding = self.img_emb(clip_feature)
            context = mx.concatenate([clip_embdding, context], axis=1)

        x, (f,) = self.patchify(x)

        # Upstream: cat 3 complex chunks -> (f, 1, head_dim/2). Carried here as
        # a (cos, sin) float pair with identical values.
        freqs = tuple(
            mx.concatenate([
                self.freqs[0][i][:f].reshape(f, -1),
                self.freqs[1][i][:f].reshape(f, -1),
                self.freqs[2][i][:f].reshape(f, -1),
            ], axis=-1).reshape(f, 1, -1)
            for i in range(2)
        )

        for block in self.blocks:
            x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f,))
        return x
