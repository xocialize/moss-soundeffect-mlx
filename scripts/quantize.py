#!/usr/bin/env python
"""Stage-4 int4 quantization of the DiT (handoff §4).

Scope: transformer-block Linear weights ONLY. Precision-sensitive projections
stay high-precision (skill guidance — lifting them raised Lens int4 cosine
0.9944 -> 0.9976): patch_embedding (conv), text/time embeddings, time
projection, head, all norms, modulation tables. VAE and text encoder are never
quantized.

Validation is per-pass cosine on an identical injected input vs the bf16
model (int4 gate ~0.99+), NOT output-PSNR — quantization legitimately moves
the denoise trajectory to a different-but-valid sample.

Usage: .venv/bin/python scripts/quantize.py [--group-size 64] [--bits 4]
"""

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten

from moss_sfx_mlx.config import WanAudioModelConfig
from moss_sfx_mlx.model.wan_audio_dit import WanAudioModel

DEFAULT_DIR = Path("/Volumes/DEV_ARCHIVE/weights/MOSS-SoundEffect-v2.0")


def build_dit(weights_path: Path, dtype=mx.bfloat16) -> WanAudioModel:
    cfg = WanAudioModelConfig()
    model = WanAudioModel(
        dim=cfg.dim, in_dim=cfg.in_dim, ffn_dim=cfg.ffn_dim, out_dim=cfg.out_dim,
        text_dim=cfg.text_dim, freq_dim=cfg.freq_dim, eps=cfg.eps,
        patch_size=cfg.patch_size, num_heads=cfg.num_heads,
        num_layers=cfg.num_layers, has_image_input=cfg.has_image_input,
        vae_type=cfg.vae_type,
    )
    weights = mx.load(str(weights_path))
    model.update(tree_unflatten([(k, v.astype(dtype)) for k, v in weights.items()]))
    mx.eval(model.parameters())
    return model


def quantize_dit(model: WanAudioModel, group_size: int, bits: int):
    def predicate(path: str, module: nn.Module) -> bool:
        # Linears inside transformer blocks only.
        return isinstance(module, nn.Linear) and path.startswith("blocks.")

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)
    mx.eval(model.parameters())


def per_pass_cosine(model_a, model_b, seed=42) -> float:
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((1, 128, 1500)).astype("float32")).astype(mx.bfloat16)
    ctx = mx.array(rng.standard_normal((1, 512, 2048)).astype("float32") * 2).astype(mx.bfloat16)
    t = mx.array([500.0])
    va = np.array(model_a(x, t, ctx).astype(mx.float32)).ravel()
    vb = np.array(model_b(x, t, ctx).astype(mx.float32)).ravel()
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4)
    args = ap.parse_args()

    src = args.model_dir / "mlx" / "dit.safetensors"
    ref = build_dit(src)
    quant = build_dit(src)
    quantize_dit(quant, args.group_size, args.bits)

    cos = per_pass_cosine(ref, quant)
    gate = 0.99 if args.bits == 4 else 0.9999
    print(f"per-pass cosine int{args.bits}/g{args.group_size} vs bf16: {cos:.6f} (gate ~{gate})")

    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(quant.parameters()))
    mx.eval(list(flat.values()))  # lazy arrays save as zeros — evaluate first
    out = args.model_dir / "mlx" / f"dit_int{args.bits}_g{args.group_size}.safetensors"
    mx.save_safetensors(
        str(out), flat,
        metadata={"quantization": f"int{args.bits} group_size={args.group_size} blocks-Linear-only",
                  "per_pass_cosine_vs_bf16": f"{cos:.6f}"})
    size_gb = out.stat().st_size / 1e9
    print(f"saved {out} ({size_gb:.2f} GB)")
    if cos < gate:
        print(f"WARNING: cosine {cos:.4f} below gate {gate} — consider int8/g128 fallback")


if __name__ == "__main__":
    main()
