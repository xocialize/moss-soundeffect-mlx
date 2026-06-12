#!/usr/bin/env python
"""Generate a sound effect with the MLX pipeline.

Usage:
  .venv/bin/python scripts/generate.py --prompt "a heavy wooden door creaks open" \
      --seconds 5 --steps 100 --cfg 4.0 --seed 0 --out output/door.wav [--cpu]
"""

import argparse
import wave
from pathlib import Path

import numpy as np


def save_wav(path, audio_np, sample_rate):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[0])
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.T.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="output/sfx.wav")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import mlx.core as mx

    if args.cpu:
        mx.set_default_device(mx.cpu)

    from moss_sfx_mlx.pipeline_mlx import MossSoundEffectPipeline

    pipe = MossSoundEffectPipeline.from_pretrained(args.model_dir)
    audio = pipe(
        prompt=args.prompt,
        seconds=args.seconds,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg,
        seed=args.seed,
    )
    audio_np = np.array(audio[0]).astype(np.float32)
    save_wav(args.out, audio_np, pipe.sample_rate)
    peak = float(np.abs(audio_np).max())
    print(f"saved {args.out}  shape={audio_np.shape}  peak={peak:.3f}")
    if mx.metal.is_available() and not args.cpu:
        print(f"peak metal mem: {mx.get_peak_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
