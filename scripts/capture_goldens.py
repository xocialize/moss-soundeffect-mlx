#!/usr/bin/env python
"""Stage-0 golden-tensor capture from the PyTorch reference (CPU, fp32).

Run with TORCHDYNAMO_DISABLE=1 (model_fn is torch.compile-decorated upstream).
Drives the upstream components directly (no facade bf16 autocast) so goldens
are fp32. Noise is generated in numpy and INJECTED — torch RNG is not
reproducible in MLX (handoff Stage 0).

Captures to tests/fixtures/:
  golden_context.npy      Qwen3 hidden states, padded 512 + zeroed tail (1)
  golden_noise.npy        initial latent noise (1, 128, 1500)            (2)
  golden_velocity_t*.npy  DiT velocity at fixed timesteps                (3)
  golden_final_latent.npy latent after the full denoise loop             (4)
  golden_audio.npy        VAE decode of (4), (1, 1, 1440000)             (5)
  golden.wav              perceptual A/B copy                            (6)
  golden_meta.json        config used (prompt, steps, cfg, shift, seed)

Usage:
  TORCHDYNAMO_DISABLE=1 nice -n 10 .venv/bin/python scripts/capture_goldens.py \
      [--steps 10] [--model-dir DIR]
"""

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = Path("/Volumes/DEV_ARCHIVE/_ref/MOSS-TTS/moss_soundeffect_v2")
sys.path.insert(0, str(UPSTREAM))

DEFAULT_DIR = "/Volumes/DEV_ARCHIVE/weights/MOSS-SoundEffect-v2.0"
PROMPT = "a heavy wooden door creaks open slowly"
SECONDS = 10.0
VELOCITY_TIMESTEPS = [1000.0, 500.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_DIR)
    ap.add_argument("--steps", type=int, default=10,
                    help="denoise steps for fixtures (MLX e2e test must match)")
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--shift", type=float, default=5.0)
    args = ap.parse_args()

    import torch

    from diffsynth.models.dac_vae import DAC
    from diffsynth.models.qwen3_text_encoder import Qwen3TextEncoder
    from diffsynth.models.wan_audio_dit import WanAudioModel
    from diffsynth.pipelines.wan_audio import _convert_hf_dit_state_dict
    from diffsynth.prompters import WanPrompter
    from diffsynth.schedulers.flow_match import FlowMatchScheduler
    from safetensors.torch import load_file

    model_dir = Path(args.model_dir)
    fixtures = REPO / "tests" / "fixtures"
    fixtures.mkdir(exist_ok=True)

    torch.set_grad_enabled(False)

    # (1) context — full prompter path (clean, pad 512, zero tail), fp32 encoder
    text_encoder = Qwen3TextEncoder(str(model_dir / "text_encoder"), torch_dtype=torch.float32)
    prompter = WanPrompter(tokenizer_path=str(model_dir / "tokenizer"))
    prompter.fetch_models(text_encoder)
    full_prompt = f"{PROMPT} duration: {SECONDS:.1f}s"
    context = prompter.encode_prompt(full_prompt, device="cpu").float()
    context_nega = prompter.encode_prompt("", device="cpu").float()
    np.save(fixtures / "golden_context.npy", context.numpy())
    np.save(fixtures / "golden_context_nega.npy", context_nega.numpy())
    del text_encoder, prompter
    print("context captured", context.shape)

    # (2) injected noise
    rng = np.random.default_rng(42)
    noise = rng.standard_normal((1, 128, 1500)).astype("float32")
    np.save(fixtures / "golden_noise.npy", noise)

    # DiT fp32
    with open(model_dir / "transformer" / "config.json") as f:
        dit_cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    dit = WanAudioModel(**{**dit_cfg, "patch_size": tuple(dit_cfg["patch_size"])}).eval()
    dit.load_state_dict(_convert_hf_dit_state_dict(
        load_file(str(model_dir / "transformer" / "diffusion_pytorch_model.safetensors"))))
    print("dit loaded")

    # (3) velocity at fixed timesteps
    x = torch.from_numpy(noise)
    for t_val in VELOCITY_TIMESTEPS:
        t = torch.tensor([t_val], dtype=torch.float32)
        v = dit(x, t, context)
        np.save(fixtures / f"golden_velocity_t{int(t_val)}.npy", v.numpy())
        print(f"velocity t={t_val} captured")

    # (4) full denoise loop — CFG exactly as upstream (two passes, fp32 combine)
    scheduler = FlowMatchScheduler(shift=args.shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(args.steps, shift=args.shift)
    latents = torch.from_numpy(noise)
    for i, timestep in enumerate(scheduler.timesteps):
        t = timestep.unsqueeze(0).to(torch.float32)
        v_posi = dit(latents, t, context)
        v_nega = dit(latents, t, context_nega)
        v = v_nega.float() + args.cfg * (v_posi.float() - v_nega.float())
        latents = scheduler.step(v, scheduler.timesteps[i], latents)
        print(f"step {i + 1}/{args.steps} done")
    np.save(fixtures / "golden_final_latent.npy", latents.numpy())
    del dit

    # (5) decode fp32
    vae = DAC.load(str(model_dir / "vae" / "vae_128d_48k.pth")).eval().float()
    audio = vae.decode(latents)
    np.save(fixtures / "golden_audio.npy", audio.numpy())
    print("audio decoded", audio.shape)

    # (6) wav
    pcm = (np.clip(audio[0].numpy(), -1, 1) * 32767).astype(np.int16)
    with wave.open(str(fixtures / "golden.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(pcm.T.tobytes())

    meta = dict(prompt=PROMPT, seconds=SECONDS, steps=args.steps, cfg=args.cfg,
                sigma_shift=args.shift, noise_seed=42,
                velocity_timesteps=VELOCITY_TIMESTEPS, dtype="float32",
                note="noise injected from numpy default_rng(42); context via full prompter path")
    (fixtures / "golden_meta.json").write_text(json.dumps(meta, indent=1))
    print("goldens complete")


if __name__ == "__main__":
    main()
