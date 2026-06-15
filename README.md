# moss-soundeffect-mlx

MLX (Python) port of [MOSS-SoundEffect-v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0)
(OpenMOSS) — text → sound effects (foley / ambience / creature / action), 48 kHz, ≤ 30 s clips —
for Apple Silicon. Apache-2.0, matching upstream.

**Status: COMPLETE — all handoff acceptance criteria closed.** Python oracle
parity-locked; weights live on mlx-community. The Swift port and the MLXEngine
capability package have been split into their own repos (see
[Swift packages](#swift-packages)); this repo remains the parity oracle —
golden fixtures, conversion tooling, upstream findings.

Weights:
[mlx-community/MOSS-SoundEffect-v2.0-bf16](https://huggingface.co/mlx-community/MOSS-SoundEffect-v2.0-bf16) ·
[mlx-community/MOSS-SoundEffect-v2.0-4bit](https://huggingface.co/mlx-community/MOSS-SoundEffect-v2.0-4bit)

## Install

```bash
pip install -e .            # package: moss-sfx-mlx (import as moss_sfx_mlx)
pip install -e ".[parity]"  # + torch / diffusers for the PyTorch↔MLX parity suite
```

## Usage

`from_pretrained` resolves a weights directory in order: explicit `model_dir`
arg → `MOSS_SFX_MLX_WEIGHTS_DIR` env var → Hub snapshot of the repo id. The
directory must contain the converted `mlx/{dit,vae}.safetensors` (the published
mlx-community repos already do).

```python
from moss_sfx_mlx.pipeline_mlx import MossSoundEffectPipeline
pipe = MossSoundEffectPipeline.from_pretrained("mlx-community/MOSS-SoundEffect-v2.0-bf16")
audio = pipe(prompt="a heavy wooden door creaks open slowly", seconds=5)
```

The published-`-4bit` repo records its DiT quantization in `model_index.json`,
so the caller needs no extra args; pass `dit_file=` / `quantization=(bits, group_size)`
to load a local int4 pass directly.

Parity locked (fp32, CPU stream, vs upstream PyTorch):
- `FlowMatchScheduler` — schedule + Euler loop + add_noise
- `WanAudioModel` DiT — block-level AND full 30-layer production scale (T=1500,
  real checkpoint, real Qwen3 conditioning), max_abs < 1e-2
- DAC-VAE — real-checkpoint decode + encode at < 1e-2 (weight-norm fused at conversion)
- Qwen3 text encoder — cosine 1.0, max_abs 4.4e-4 (fp32 floor given Qwen's
  ~1.2e4 massive activations; see test comment)

## Scripts

| Script | Purpose |
|---|---|
| `scripts/convert_weights.py` | HF snapshot → `<model_dir>/mlx/{dit,vae}.safetensors` (bf16 DiT / fp32 VAE) |
| `scripts/quantize.py` | int4 DiT pass (default g64, blocks-`Linear` only; `--group-size`/`--bits`) |
| `scripts/generate.py` | CLI inference (`--prompt --seconds --steps --cfg --seed --out [--cpu --model-dir]`) |
| `scripts/capture_goldens.py` | capture PyTorch reference goldens |
| `scripts/export_swift_fixtures.py` | bundle `.npy` goldens into one safetensors for the Swift test suite |
| `scripts/publish_prep.py` | assemble the two mlx-community repos locally for review (no upload) |

Example:

```bash
.venv/bin/python scripts/generate.py --prompt "a heavy wooden door creaks open" \
    --seconds 5 --steps 100 --cfg 4.0 --seed 0 --out output/door.wav [--cpu]
```

## Layout

- `docs/handoff.md` — authoritative port spec (stages, parity gates, risks)
- `docs/upstream-findings.md` — resolved upstream-source questions (DiT internals,
  50 Hz latent rate, no-VAE-scale-constant, CFG order, prompt padding)
- `moss_sfx_mlx/` — Python parity oracle (MLX): `pipeline_mlx.py`, `scheduler.py`,
  `prompter.py`, `model/{wan_audio_dit,wan_video_dit,dac_vae,qwen3_text_encoder}.py`
- `tests/parity/` — PyTorch↔MLX parity suite (`pip install -e ".[parity]"`)
- `tests/fixtures/` — golden tensors (`.npy` + `swift_goldens.safetensors`)

## Architecture

| Component | Class | Notes |
|---|---|---|
| DiT | `WanAudioModel` | 30 layers, dim 1536, 12 heads, flow-matching, ~1.3B |
| Codec | `DAC` (continuous VAE) | 128-d latents @ 50 Hz (hop 960), 48 kHz out |
| Text encoder | Qwen3-1.7B-Base | last-layer hidden states (2048-d) → cross-attn |
| Scheduler | `FlowMatchScheduler` | shift 5.0, sigma_min 0.0, extra_one_step |

The Python facade class is `MossSoundEffectPipeline`; the inner engine is
`WanAudioPipeline`. Local dev: set `MOSS_SFX_MLX_WEIGHTS_DIR` to a directory of
converted weights.

## Performance (Apple M5 Max, 128 GB)

100 inference steps, cfg_scale 4.0, full 30 s latent (output cropped to `seconds`):

| DiT precision | Wall clock | Per step (incl. 2× CFG) | Peak memory | DiT size |
|---|---|---|---|---|
| bf16 | 60 s | ~0.47 s steady-state | 14.2 GB | 2.83 GB |
| int4 g64 (blocks-Linear only) | 45 s | ~0.40 s | 12.2 GB | 0.83 GB |

int4 per-pass cosine vs bf16: **0.999425** (gate 0.99). Noise-decode periodicity
check clean (frame-boundary autocorr ~0.01). Perceptual A/B batch (10 prompts,
100 steps, `output/ab_batch/`): **human-reviewed, passed** (2026-06-11 — correct
content, correct duration, no tonal artifacts).

## Swift packages

The MLX-Swift port lives in its own SPM-consumable repo:
[xocialize/moss-soundeffect-mlx-swift](https://github.com/xocialize/moss-soundeffect-mlx-swift).
The MLXEngine `soundEffect` capability package wrapping it is
[xocialize/mlx-moss-soundeffect-swift](https://github.com/xocialize/mlx-moss-soundeffect-swift).
(The local `swift/` directory here is a build scratch dir only — it is not
tracked and not the deliverable.)
