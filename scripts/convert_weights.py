#!/usr/bin/env python
"""Stage-1 weight conversion: original HF snapshot -> MLX safetensors.

Writes <model_dir>/mlx/{dit,vae}.safetensors. The text encoder needs no
conversion (bf16 HF safetensors load directly via mx.load in the wrapper).

  dit.safetensors  bf16 (runtime default; parity tests load the original fp32)
  vae.safetensors  fp32 (upstream decodes under fp32 autocast; weight-norm fused)

THE Tier-3 trap: MLX is lazy — unevaluated arrays serialize as ZEROS with no
error. mx.eval() runs immediately before every save below. Do not remove.

Usage: .venv/bin/python scripts/convert_weights.py [--model-dir DIR] [--cpu]
"""

import argparse
from pathlib import Path

import mlx.core as mx

DEFAULT_DIR = "/Volumes/DEV_ARCHIVE/weights/MOSS-SoundEffect-v2.0"


def convert_dit(model_dir: Path, out_dir: Path):
    from moss_sfx_mlx.utils.convert import rename_dit_key

    src = model_dir / "transformer" / "diffusion_pytorch_model.safetensors"
    print(f"DiT: {src}")
    weights = mx.load(str(src))  # fp32 safetensors, framework-agnostic
    out = {}
    for k, v in weights.items():
        if k == "patch_embedding.weight":
            v = v.transpose(0, 2, 1)  # Conv1d (O, I, K) -> (O, K, I)
        out[rename_dit_key(k)] = v.astype(mx.bfloat16)
    mx.eval(list(out.values()))  # MUST precede save: lazy arrays save as zeros
    mx.save_safetensors(str(out_dir / "dit.safetensors"), out)
    print(f"  -> {out_dir / 'dit.safetensors'} ({len(out)} tensors, bf16)")


def convert_vae(model_dir: Path, out_dir: Path):
    import torch

    from moss_sfx_mlx.utils.convert import dac_pt_to_mlx

    src = model_dir / "vae" / "vae_128d_48k.pth"
    print(f"VAE: {src}")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    converted = dac_pt_to_mlx(ckpt["state_dict"])  # fuse weight-norm, rename, transpose
    out = {k: mx.array(v) for k, v in converted.items()}  # fp32
    mx.eval(list(out.values()))  # MUST precede save: lazy arrays save as zeros
    mx.save_safetensors(str(out_dir / "vae.safetensors"), out)
    print(f"  -> {out_dir / 'vae.safetensors'} ({len(out)} tensors, fp32)")
    print(f"  vae kwargs: {ckpt['metadata']['kwargs']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_DIR)
    ap.add_argument("--cpu", action="store_true", help="run on the CPU stream (GPU busy)")
    args = ap.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)

    model_dir = Path(args.model_dir)
    out_dir = model_dir / "mlx"
    out_dir.mkdir(exist_ok=True)

    convert_dit(model_dir, out_dir)
    convert_vae(model_dir, out_dir)

    # Verify nothing serialized as zeros (the lazy-save failure mode).
    for name in ("dit.safetensors", "vae.safetensors"):
        loaded = mx.load(str(out_dir / name))
        n_zero = sum(1 for v in loaded.values() if float(mx.abs(v).max()) == 0.0)
        nonzero_biases_expected = ("bias",)
        print(f"check {name}: {len(loaded)} tensors, {n_zero} all-zero "
              f"(expect 0 or only genuine zero-init {nonzero_biases_expected})")


if __name__ == "__main__":
    main()
