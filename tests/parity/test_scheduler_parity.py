"""FlowMatchScheduler parity: upstream torch reference vs MLX port.

Pure-math comparison — no weights needed. The upstream module is imported
straight from the MOSS-TTS clone; torch is an optional dev dependency.
Run with MLX pinned to CPU (done below) so framework noise stays ~1e-7.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

UPSTREAM = Path(__file__).resolve().parents[2].parent / "_ref" / "MOSS-TTS"
if not (UPSTREAM / "moss_soundeffect_v2").exists():
    UPSTREAM = Path("/Volumes/DEV_ARCHIVE/_ref/MOSS-TTS")
sys.path.insert(0, str(UPSTREAM / "moss_soundeffect_v2" / "diffsynth" / "schedulers"))

import mlx.core as mx

mx.set_default_device(mx.cpu)

from flow_match import FlowMatchScheduler as PTFlowMatchScheduler  # noqa: E402

from moss_sfx_mlx.scheduler import FlowMatchScheduler as MXFlowMatchScheduler  # noqa: E402

# Production config: shift=5.0, sigma_min=0.0, extra_one_step=True (wan_audio.py:124)
PROD = dict(shift=5.0, sigma_min=0.0, extra_one_step=True)


def make_pair(num_inference_steps=100, **kwargs):
    cfg = {**PROD, **kwargs}
    return (
        PTFlowMatchScheduler(num_inference_steps=num_inference_steps, **cfg),
        MXFlowMatchScheduler(num_inference_steps=num_inference_steps, **cfg),
    )


def test_sigma_schedule_parity():
    pt, mxs = make_pair(num_inference_steps=100)
    np.testing.assert_allclose(
        np.array(mxs.sigmas), pt.sigmas.numpy(), atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        np.array(mxs.timesteps), pt.timesteps.numpy(), atol=1e-3, rtol=0
    )


def test_sigma_schedule_parity_sigma_shift_override():
    # The pipeline calls set_timesteps(num_inference_steps, shift=sigma_shift)
    # at every __call__ — exercise the override path.
    pt, mxs = make_pair(num_inference_steps=100)
    pt.set_timesteps(50, shift=5.0)
    mxs.set_timesteps(50, shift=5.0)
    np.testing.assert_allclose(
        np.array(mxs.sigmas), pt.sigmas.numpy(), atol=1e-6, rtol=0
    )


def test_full_denoise_loop_parity():
    """Run the exact Euler loop the pipeline runs, with injected velocities."""
    rng = np.random.default_rng(42)
    shape = (1, 128, 1500)  # (B, latent_dim, T) at 30 s / 50 Hz
    sample_np = rng.standard_normal(shape).astype("float32")

    pt, mxs = make_pair(num_inference_steps=20)
    pt_sample = torch.from_numpy(sample_np)
    mx_sample = mx.array(sample_np)

    for i, t in enumerate(pt.timesteps):
        v_np = rng.standard_normal(shape).astype("float32") * 0.1
        pt_sample = pt.step(torch.from_numpy(v_np), pt.timesteps[i], pt_sample)
        mx_sample = mxs.step(mx.array(v_np), float(np.array(mxs.timesteps[i])), mx_sample)

    max_abs = float(np.max(np.abs(pt_sample.numpy() - np.array(mx_sample))))
    assert max_abs < 1e-4, f"denoise loop diverges: max_abs={max_abs}"


def test_add_noise_parity():
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((2, 128, 100)).astype("float32")
    eps = rng.standard_normal((2, 128, 100)).astype("float32")

    pt, mxs = make_pair(num_inference_steps=100)
    t = pt.timesteps[0:2]
    pt_out = pt.add_noise(torch.from_numpy(x0), torch.from_numpy(eps), t)
    mx_out = mxs.add_noise(mx.array(x0), mx.array(eps), mx.array(t.numpy()))
    np.testing.assert_allclose(np.array(mx_out), pt_out.numpy(), atol=1e-5, rtol=0)
