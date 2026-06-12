"""DiT parity: upstream torch WanAudioModel / DiTBlock vs MLX port.

Small-config with seeded random weights copied PT -> MLX (Stage-2 gates 2-3).
Production-scale parity with the real checkpoint runs separately once the
conversion recipe exists. MLX pinned to CPU to keep framework noise ~1e-7.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

UPSTREAM = Path("/Volumes/DEV_ARCHIVE/_ref/MOSS-TTS/moss_soundeffect_v2")
sys.path.insert(0, str(UPSTREAM))

import mlx.core as mx

mx.set_default_device(mx.cpu)

from diffsynth.models.wan_audio_dit import WanAudioModel as PTWanAudioModel  # noqa: E402
from diffsynth.models.wan_video_dit import (  # noqa: E402
    DiTBlock as PTDiTBlock,
    precompute_freqs_cis as pt_precompute_freqs_cis,
)

from moss_sfx_mlx.model.wan_audio_dit import WanAudioModel as MXWanAudioModel  # noqa: E402
from moss_sfx_mlx.model.wan_video_dit import DiTBlock as MXDiTBlock  # noqa: E402

from ._helpers import assert_parity, load_pt_state_into_mx, make_seeded_input  # noqa: E402

SMALL = dict(
    dim=256, in_dim=8, ffn_dim=512, out_dim=8, text_dim=64, freq_dim=32,
    eps=1e-6, patch_size=(1,), num_heads=4, num_layers=2,
    has_image_input=False, vae_type="dac",
)


def rename(key: str) -> str:
    # torch nn.Sequential indices -> MLX nn.Sequential 'layers' attribute.
    for prefix in ("text_embedding", "time_embedding", "time_projection"):
        key = key.replace(f"{prefix}.", f"{prefix}.layers.")
    if ".ffn." in key:
        key = key.replace(".ffn.", ".ffn.layers.")
    if key.startswith("ffn."):
        key = "ffn.layers." + key[len("ffn."):]
    if key.startswith("proj."):  # img MLP, unused here
        key = "proj.layers." + key[len("proj."):]
    return key


def test_dit_block_parity():
    torch.manual_seed(0)
    pt_block = PTDiTBlock(has_image_input=False, dim=256, num_heads=4, ffn_dim=512, eps=1e-6).eval()
    mx_block = MXDiTBlock(has_image_input=False, dim=256, num_heads=4, ffn_dim=512, eps=1e-6)
    load_pt_state_into_mx(mx_block, pt_block.state_dict(), rename=rename)

    f, head_dim = 96, 64
    x = make_seeded_input((2, f, 256), seed=1)
    ctx = make_seeded_input((2, 32, 256), seed=2)
    t_mod = make_seeded_input((2, 6, 256), seed=3)

    # freqs: upstream complex (f, 1, head_dim/2); MLX carries (cos, sin).
    freqs_cis = pt_precompute_freqs_cis(head_dim, end=16384)[:f].reshape(f, 1, -1)
    mx_freqs = (
        mx.array(freqs_cis.real.numpy().astype("float32")),
        mx.array(freqs_cis.imag.numpy().astype("float32")),
    )

    with torch.no_grad():
        pt_out = pt_block(
            torch.from_numpy(x), torch.from_numpy(ctx), torch.from_numpy(t_mod), freqs_cis
        )
    mx_out = mx_block(mx.array(x), mx.array(ctx), mx.array(t_mod), mx_freqs)
    assert_parity(pt_out, mx_out, threshold=1e-3, name="dit_block")


def test_wan_audio_model_parity_small():
    torch.manual_seed(0)
    pt_model = PTWanAudioModel(**SMALL).eval()
    mx_model = MXWanAudioModel(**SMALL)
    load_pt_state_into_mx(
        mx_model,
        pt_model.state_dict(),
        rename=rename,
        conv_keys={"patch_embedding.weight"},
        conv_ndim_by_key={"patch_embedding.weight": 1},
    )

    x = make_seeded_input((1, SMALL["in_dim"], 128), seed=10)   # (B, C, T)
    ctx = make_seeded_input((1, 24, SMALL["text_dim"]), seed=11)
    t = np.array([500.0], dtype="float32")

    with torch.no_grad():
        pt_out = pt_model(
            torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(ctx)
        )
    mx_out = mx_model(mx.array(x), mx.array(t), mx.array(ctx))
    assert_parity(pt_out, mx_out, threshold=1e-3, name="wan_audio_model_small")
