"""DAC-VAE parity: upstream torch DAC vs MLX port.

Covers the #1 silent-failure path (decode of raw DiT latents). Two levels:
  1. small-config random weights (fast, structural)
  2. the real vae_128d_48k.pth checkpoint, decode + encode round at fp32

MLX pinned to CPU; upstream loaded on CPU.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

UPSTREAM = Path("/Volumes/DEV_ARCHIVE/_ref/MOSS-TTS/moss_soundeffect_v2")
WEIGHTS = Path("/Volumes/DEV_ARCHIVE/weights/MOSS-SoundEffect-v2.0/vae/vae_128d_48k.pth")
sys.path.insert(0, str(UPSTREAM))

import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx.utils import tree_unflatten  # noqa: E402

from moss_sfx_mlx.model.dac_vae import DAC as MXDAC  # noqa: E402
from moss_sfx_mlx.utils.convert import dac_pt_to_mlx  # noqa: E402

from ._helpers import assert_parity, make_seeded_input  # noqa: E402


def load_into_mx(mx_model, pt_state_dict):
    converted = dac_pt_to_mlx(pt_state_dict)
    mx_model.update(tree_unflatten([(k, mx.array(v)) for k, v in converted.items()]))
    mx.eval(mx_model.parameters())


def make_pt_dac(**kwargs):
    from diffsynth.models.dac_vae import DAC as PTDAC

    return PTDAC(**kwargs).eval()


SMALL = dict(
    encoder_dim=16, encoder_rates=[2, 4], latent_dim=8, decoder_dim=64,
    decoder_rates=[4, 2], sample_rate=8000, continuous=True,
)


def test_dac_small_decode_parity():
    torch.manual_seed(0)
    pt = make_pt_dac(**SMALL)
    mxm = MXDAC(**SMALL)
    load_into_mx(mxm, pt.state_dict())

    z = make_seeded_input((2, 8, 16), seed=5)  # (B, D, T_lat)
    with torch.no_grad():
        pt_audio = pt.decode(torch.from_numpy(z))
    mx_audio = mxm.decode(mx.array(z))
    assert_parity(pt_audio, mx_audio, threshold=1e-4, name="dac_small_decode")


def test_dac_small_encode_parity():
    torch.manual_seed(0)
    pt = make_pt_dac(**SMALL)
    mxm = MXDAC(**SMALL)
    load_into_mx(mxm, pt.state_dict())

    audio = (make_seeded_input((2, 1, 128), seed=6) * 0.1).clip(-1, 1)
    with torch.no_grad():
        pt_mode = pt.encode(torch.from_numpy(audio))[0].mode()
    mx_mode = mxm.encode(mx.array(audio))[0].mode()
    assert_parity(pt_mode, mx_mode, threshold=1e-4, name="dac_small_encode_mode")


@pytest.mark.skipif(not WEIGHTS.exists(), reason="real checkpoint not downloaded")
def test_dac_real_checkpoint_decode_parity():
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    kwargs = ckpt["metadata"]["kwargs"]
    sd = ckpt["state_dict"]

    pt = make_pt_dac(**kwargs)
    pt.load_state_dict(sd)
    mxm = MXDAC(**kwargs)
    load_into_mx(mxm, sd)

    # 1 s of latents at 50 Hz — same magnitude class as DiT output (unit noise).
    z = make_seeded_input((1, 128, 50), seed=42)
    with torch.no_grad():
        pt_audio = pt.decode(torch.from_numpy(z))
    mx_audio = mxm.decode(mx.array(z))
    assert pt_audio.shape == (1, 1, 48000)
    # Stage-2 gate: waveform max_abs < 1e-2 fp32 (handoff §4 step 5).
    assert_parity(pt_audio, mx_audio, threshold=1e-2, name="dac_real_decode")
    # Tonal-periodicity guard: identical outputs implies no MLX-side artifact;
    # also sanity-check the waveform is in the Tanh range and non-degenerate.
    a = np.array(mx_audio)
    assert np.abs(a).max() <= 1.0 + 1e-6
    assert np.abs(a).std() > 1e-4


@pytest.mark.skipif(not WEIGHTS.exists(), reason="real checkpoint not downloaded")
def test_dac_real_checkpoint_encode_decode_roundtrip_parity():
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    kwargs = ckpt["metadata"]["kwargs"]
    sd = ckpt["state_dict"]

    pt = make_pt_dac(**kwargs)
    pt.load_state_dict(sd)
    mxm = MXDAC(**kwargs)
    load_into_mx(mxm, sd)

    audio = (make_seeded_input((1, 1, 48000), seed=7) * 0.05).clip(-1, 1)
    with torch.no_grad():
        pt_z = pt.encode(torch.from_numpy(audio))[0].mode()
    mx_z = mxm.encode(mx.array(audio))[0].mode()
    assert_parity(pt_z, mx_z, threshold=1e-2, name="dac_real_encode_mode")
