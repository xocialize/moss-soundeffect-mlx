# moss-soundeffect-mlx

MLX / MLX-Swift port of [MOSS-SoundEffect-v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0)
(OpenMOSS) — text → sound effects (foley / ambience / creature / action), 48 kHz, ≤ 30 s clips —
for Apple Silicon. Apache-2.0, matching upstream.

**Status: PUBLISHED (Python pipeline) — Stage 2 parity-locked, GPU-validated,
perceptually reviewed, weights live on mlx-community. Stage 3 (Swift) nearly
complete: DiT/VAE/scheduler validated vs goldens; slow e2e + tokenizer pending.**

Weights:
[mlx-community/MOSS-SoundEffect-v2.0-bf16](https://huggingface.co/mlx-community/MOSS-SoundEffect-v2.0-bf16) ·
[mlx-community/MOSS-SoundEffect-v2.0-4bit](https://huggingface.co/mlx-community/MOSS-SoundEffect-v2.0-4bit)

```python
from moss_sfx_mlx.pipeline_mlx import MossSoundEffectPipeline
pipe = MossSoundEffectPipeline.from_pretrained("mlx-community/MOSS-SoundEffect-v2.0-bf16")
audio = pipe(prompt="a heavy wooden door creaks open slowly", seconds=5)
```

Parity locked (fp32, CPU stream, vs upstream PyTorch):
- `FlowMatchScheduler` — schedule + Euler loop + add_noise
- `WanAudioModel` DiT — block-level AND full 30-layer production scale (T=1500,
  real checkpoint, real Qwen3 conditioning), max_abs < 1e-2
- DAC-VAE — real-checkpoint decode + encode at < 1e-2 (weight-norm fused at conversion)
- Qwen3 text encoder — cosine 1.0, max_abs 4.4e-4 (fp32 floor given Qwen's
  ~1.2e4 massive activations; see test comment)

Weights converted: `<weights>/mlx/{dit,vae}.safetensors` via
[scripts/convert_weights.py](scripts/convert_weights.py) (bf16 DiT / fp32 VAE,
zero-tensor check passed). Generate with
[scripts/generate.py](scripts/generate.py); capture reference goldens with
[scripts/capture_goldens.py](scripts/capture_goldens.py).

- `docs/handoff.md` — authoritative port spec (stages, parity gates, risks)
- `docs/upstream-findings.md` — resolved upstream-source questions (DiT internals,
  50 Hz latent rate, no-VAE-scale-constant, CFG order, prompt padding)
- `moss_sfx_mlx/` — Python parity oracle (MLX)
- `swift/` — MLX-Swift deliverable (loads into MLXEngine) — not started
- `tests/parity/` — PyTorch↔MLX parity suite (`pip install -e ".[parity]"`)

## Architecture

| Component | Class | Notes |
|---|---|---|
| DiT | `WanAudioModel` | 30 layers, dim 1536, 12 heads, flow-matching, ~1.3B |
| Codec | `DAC` (continuous VAE) | 128-d latents @ 50 Hz (hop 960), 48 kHz out |
| Text encoder | Qwen3-1.7B-Base | last-layer hidden states (2048-d) → cross-attn |
| Scheduler | `FlowMatchScheduler` | shift 5.0, sigma_min 0.0, extra_one_step |

Local dev: set `MOSS_SFX_MLX_WEIGHTS_DIR` to a directory of converted weights.

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

## Swift package

The MLX-Swift port now lives in its own repo (SPM-consumable):
[xocialize/moss-soundeffect-mlx-swift](https://github.com/xocialize/moss-soundeffect-mlx-swift).
The MLXEngine capability package is
[xocialize/mlx-moss-soundeffect-swift](https://github.com/xocialize/mlx-moss-soundeffect-swift).
This repo remains the parity oracle: golden fixtures, conversion tooling, upstream findings.
