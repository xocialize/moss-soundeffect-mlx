"""End-to-end parity vs Stage-0 goldens (Stage-2 gates 4 + 6).

Run scripts/capture_goldens.py first (torch CPU, fp32). Every input is
injected from the golden fixtures — no RNG crosses the framework boundary.
MLX runs fp32 on the CPU stream; thresholds are the handoff §4 gates.
"""

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
WEIGHTS = Path("/Volumes/DEV_ARCHIVE/weights/MOSS-SoundEffect-v2.0")

import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx.utils import tree_unflatten  # noqa: E402

from moss_sfx_mlx.config import WanAudioModelConfig  # noqa: E402
from moss_sfx_mlx.model.wan_audio_dit import WanAudioModel  # noqa: E402
from moss_sfx_mlx.scheduler import FlowMatchScheduler  # noqa: E402

from ._helpers import assert_parity  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (FIXTURES / "golden_meta.json").exists(),
    reason="run scripts/capture_goldens.py first",
)


@pytest.fixture(scope="module")
def goldens():
    meta = json.loads((FIXTURES / "golden_meta.json").read_text())
    load = lambda n: np.load(FIXTURES / f"{n}.npy")
    return dict(
        meta=meta,
        context=load("golden_context"),
        context_nega=load("golden_context_nega"),
        noise=load("golden_noise"),
        final_latent=load("golden_final_latent"),
        audio=load("golden_audio"),
        velocities={t: load(f"golden_velocity_t{int(t)}") for t in meta["velocity_timesteps"]},
    )


@pytest.fixture(scope="module")
def dit_fp32():
    """MLX DiT loaded fp32 straight from the original checkpoint (parity master)."""
    from moss_sfx_mlx.utils.convert import rename_dit_key

    cfg = WanAudioModelConfig()
    model = WanAudioModel(
        dim=cfg.dim, in_dim=cfg.in_dim, ffn_dim=cfg.ffn_dim, out_dim=cfg.out_dim,
        text_dim=cfg.text_dim, freq_dim=cfg.freq_dim, eps=cfg.eps,
        patch_size=cfg.patch_size, num_heads=cfg.num_heads,
        num_layers=cfg.num_layers, has_image_input=cfg.has_image_input,
        vae_type=cfg.vae_type,
    )
    weights = mx.load(str(WEIGHTS / "transformer" / "diffusion_pytorch_model.safetensors"))
    converted = {}
    for k, v in weights.items():
        if k == "patch_embedding.weight":
            v = v.transpose(0, 2, 1)
        converted[rename_dit_key(k)] = v
    model.update(tree_unflatten(list(converted.items())))
    mx.eval(model.parameters())
    return model


def test_velocity_parity(goldens, dit_fp32):
    """Gate 3 against the golden capture (not just side-by-side PT)."""
    x = mx.array(goldens["noise"])
    ctx = mx.array(goldens["context"])
    for t_val, golden_v in goldens["velocities"].items():
        v = dit_fp32(x, mx.array([t_val], dtype=mx.float32), ctx)
        assert_parity(golden_v, v, threshold=1e-2, name=f"velocity_t{int(t_val)}")


def test_denoise_loop_parity(goldens, dit_fp32):
    """Gate 4: full loop (steps/cfg/shift from golden_meta), final latent < 1e-2."""
    meta = goldens["meta"]
    scheduler = FlowMatchScheduler(shift=meta["sigma_shift"], sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(meta["steps"], shift=meta["sigma_shift"])

    latents = mx.array(goldens["noise"])
    ctx = mx.array(goldens["context"])
    ctx_nega = mx.array(goldens["context_nega"])
    cfg = meta["cfg"]

    for i in range(len(scheduler.timesteps)):
        t = scheduler.timesteps[i].reshape(1)
        v_posi = dit_fp32(latents, t, ctx)
        v_nega = dit_fp32(latents, t, ctx_nega)
        v = v_nega + cfg * (v_posi - v_nega)
        latents = scheduler.step(v, scheduler.timesteps[i], latents)
        mx.eval(latents)

    assert_parity(goldens["final_latent"], latents, threshold=1e-2, name="final_latent")


def test_decode_golden_latent_parity(goldens):
    """Gate 5/6 seam: decode the GOLDEN final latent, compare waveforms."""
    from moss_sfx_mlx.config import DACConfig
    from moss_sfx_mlx.model.dac_vae import DAC

    vcfg = DACConfig()
    vae = DAC(
        encoder_dim=vcfg.encoder_dim, encoder_rates=vcfg.encoder_rates,
        latent_dim=vcfg.latent_dim, decoder_dim=vcfg.decoder_dim,
        decoder_rates=vcfg.decoder_rates, sample_rate=vcfg.sample_rate,
        continuous=vcfg.continuous,
    )
    weights = mx.load(str(WEIGHTS / "mlx" / "vae.safetensors"))
    vae.update(tree_unflatten(list(weights.items())))
    mx.eval(vae.parameters())

    audio = vae.decode(mx.array(goldens["final_latent"]))
    assert_parity(goldens["audio"], audio, threshold=1e-2, name="golden_audio")
