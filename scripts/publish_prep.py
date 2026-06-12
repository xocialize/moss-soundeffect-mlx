#!/usr/bin/env python
"""Assemble the two mlx-community repos locally for review (NO upload).

Targets (user-confirmed):
  mlx-community/MOSS-SoundEffect-v2.0-bf16   DiT bf16 / VAE fp32 / Qwen3 bf16
  mlx-community/MOSS-SoundEffect-v2.0-4bit   DiT int4 g64 (blocks-Linear only)

Output: <model_dir>/publish/MOSS-SoundEffect-v2.0-{bf16,4bit}/ — loadable
as-is by MossSoundEffectPipeline.from_pretrained(<dir>); upload afterwards via
`hf upload mlx-community/<name> <dir>` once reviewed.

Layout per repo (no PyTorch weights — MLX only):
  README.md                 model card (mlx-community contract + Parity section)
  model_index.json          + mlx_quantization for the -4bit repo
  mlx/dit.safetensors       mlx/vae.safetensors
  transformer/config.json   vae/config.json   scheduler/scheduler_config.json
  text_encoder/             original bf16 safetensors + configs (no conversion needed)
  tokenizer/

Usage: .venv/bin/python scripts/publish_prep.py
"""

import json
import shutil
from pathlib import Path

SRC = Path("/Volumes/DEV_ARCHIVE/weights/MOSS-SoundEffect-v2.0")
OUT = SRC / "publish"

CARD = """---
library_name: mlx
license: apache-2.0
license_link: https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0/blob/main/README.md
pipeline_tag: text-to-audio
base_model: OpenMOSS-Team/MOSS-SoundEffect-v2.0
tags:
  - mlx
---

# mlx-community/MOSS-SoundEffect-v2.0-{suffix}

This model [mlx-community/MOSS-SoundEffect-v2.0-{suffix}](https://huggingface.co/mlx-community/MOSS-SoundEffect-v2.0-{suffix})
was converted to MLX format from
[OpenMOSS-Team/MOSS-SoundEffect-v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0)
— a text-to-sound-effect diffusion pipeline (foley / ambience / creature /
action audio, 48 kHz, up to 30 s) with a 1.3B Wan-style flow-matching DiT, a
continuous 128-d DAC VAE (50 Hz latents), and a frozen Qwen3-1.7B text encoder.

{precision_note}

## Use with mlx

```bash
pip install moss-sfx-mlx  # https://github.com/xocialize/moss-soundeffect-mlx
```

```python
from moss_sfx_mlx.pipeline_mlx import MossSoundEffectPipeline

pipe = MossSoundEffectPipeline.from_pretrained("mlx-community/MOSS-SoundEffect-v2.0-{suffix}")
audio = pipe(prompt="a heavy wooden door creaks open slowly",
             seconds=5, num_inference_steps=100, cfg_scale=4.0, seed=0)
# audio: (1, 1, samples) mx.array at 48 kHz
```

## Parity

Validated against the upstream PyTorch reference (fp32, CPU stream, per-module
and end-to-end golden tensors; full suite in the GitHub repo):

- End-to-end waveform vs PyTorch golden (10-step CFG denoise): max_abs < 1e-2 fp32
- Full-DiT velocity field at production scale (T=1500): max_abs < 1e-2 fp32
- DAC-VAE decode vs reference: max_abs < 1e-2 fp32 (no scale constant — the
  learned post_quant_conv is faithful)
- Qwen3 hidden states: cosine 1.0, max_abs 4.4e-4 (fp32 accumulation floor)
{quant_parity}
- 10-prompt perceptual A/B at 100 steps: passed human review (correct content,
  duration, no tonal artifacts)

## Performance (Apple M5 Max)

100 steps, cfg 4.0, full 30 s latent: {perf}

## License

Apache-2.0, matching the upstream model, code, and all components.
"""

VARIANTS = {
    "bf16": dict(
        dit_src="mlx/dit.safetensors",
        quant=None,
        precision_note=(
            "Precision: DiT bf16, DAC-VAE fp32 (the reference decodes under fp32 "
            "autocast), Qwen3 text encoder bf16."
        ),
        quant_parity="",
        perf="60 s wall clock, 14.2 GB peak memory.",
    ),
    "4bit": dict(
        dit_src="mlx/dit_int4_g64.safetensors",
        quant={"bits": 4, "group_size": 64,
               "scope": "transformer-block Linear weights only"},
        precision_note=(
            "Precision: DiT int4 (group_size 64, transformer-block Linears only — "
            "embeddings, time/text projections, head, and norms stay bf16), "
            "DAC-VAE fp32, Qwen3 text encoder bf16."
        ),
        quant_parity=(
            "- int4 DiT per-pass cosine vs bf16 on identical injected inputs: "
            "0.999425 (gate 0.99)\n"
        ),
        perf="45 s wall clock, 12.2 GB peak memory; DiT shrinks 2.83 GB -> 0.83 GB.",
    ),
}


def assemble(suffix: str, spec: dict):
    dest = OUT / f"MOSS-SoundEffect-v2.0-{suffix}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # Components
    (dest / "mlx").mkdir()
    shutil.copy2(SRC / spec["dit_src"], dest / "mlx" / "dit.safetensors")
    shutil.copy2(SRC / "mlx" / "vae.safetensors", dest / "mlx" / "vae.safetensors")
    for component in ("text_encoder", "tokenizer", "scheduler"):
        shutil.copytree(SRC / component, dest / component)
    for cfg in ("transformer/config.json", "vae/config.json"):
        (dest / cfg).parent.mkdir(exist_ok=True)
        shutil.copy2(SRC / cfg, dest / cfg)

    # model_index.json (+ quantization marker for -4bit)
    index = json.loads((SRC / "model_index.json").read_text())
    index["mlx_converted_from"] = "OpenMOSS-Team/MOSS-SoundEffect-v2.0"
    if spec["quant"]:
        index["mlx_quantization"] = spec["quant"]
    (dest / "model_index.json").write_text(json.dumps(index, indent=2))

    # Model card
    (dest / "README.md").write_text(CARD.format(
        suffix=suffix,
        precision_note=spec["precision_note"],
        quant_parity=spec["quant_parity"],
        perf=spec["perf"],
    ))

    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e9
    print(f"{dest}  ({size:.2f} GB)")


def main():
    for suffix, spec in VARIANTS.items():
        assemble(suffix, spec)
    print("\nReview, then upload with:")
    for suffix in VARIANTS:
        print(f"  hf upload mlx-community/MOSS-SoundEffect-v2.0-{suffix} "
              f"{OUT}/MOSS-SoundEffect-v2.0-{suffix} .")


if __name__ == "__main__":
    main()
