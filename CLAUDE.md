# CLAUDE.md — moss-soundeffect-mlx

Repo context for Claude. Read this first when opening this repo cold.

## Local machine layout (this box)

- Upstream reference clone (the oracle): `/Volumes/DEV_ARCHIVE/_ref/MOSS-TTS` (`moss_soundeffect_v2/`)
- Source PyTorch weights (full 11.2 GB HF snapshot): `/Volumes/DEV_ARCHIVE/weights/MOSS-SoundEffect-v2.0`
- Resolved §6 open items + extra pipeline facts: `docs/upstream-findings.md` (read before touching model code)
- DiT checkpoint key inventory (diffusers-style keys, all fp32): `docs/dit-checkpoint-keys.json`
- Python env: `.venv` (`pip install -e ".[parity]"` — torch is a dev-only dep)

## What this repo is

Greenfield **Tier-3 MLX-Swift port** of **MOSS-SoundEffect-v2.0** (OpenMOSS) — a text→sound-effect model (foley / ambience / creature / action audio from text captions, 48 kHz, ≤30 s clips). The Swift package is the deliverable; it loads into **MLXEngine** for anime-studio SFX generation.

Full coder handoff lives at `docs/handoff.md` — that is the authoritative spec. This file is the orientation layer.

## Invoke the mlx-porting skill

This is a model port. The `mlx-porting` skill governs all work here — routing, weight conversion, parity testing, the silent-failure checklist, and the MLX-Swift consumer idioms. Invoke it eagerly; MLX ports **fail silently**, so parity discipline is non-negotiable. This is a **Tier 3** port (multi-component pipeline), not Tier 1/2 — do not reach for `mlx_lm.convert`; use the `mlx-forge` recipe path.

## Architecture (4 components — all Apache-2.0 / MIT, clean GO)

| Component | Class | Key config | Weight |
|---|---|---|---|
| DiT backbone | `WanAudioModel` | 30 layers, dim 1536, 12 heads (hd 128), ffn 8960, in/out 128, text_dim 2048, eps 1e-6, flow-matching, ~1.3B | `diffusion_pytorch_model.safetensors` (5.66 GB) |
| Audio codec | `DAC` (continuous VAE) | latent_dim 128, 48 kHz, continuous (NOT RVQ) | `vae_128d_48k.pth` — pickled full object (1.49 GB) |
| Text encoder | `Qwen3ForCausalLM` (frozen) | Qwen3-1.7B-Base, hidden 2048, 28 layers, GQA 16/8 | 2× safetensors (~4.06 GB) |
| Scheduler | `FlowMatchScheduler` | shift 5.0, sigma_min 0.0, extra_one_step true, 1000 train steps | config |

Inference defaults: `num_inference_steps=100`, `cfg_scale=4.0`, `sigma_shift=5.0`. Output `(B,C,T)` waveform.

## The oracle (reference = ground truth, this is a transpose not a redesign)

- Weights/card: https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0
- Upstream code: https://github.com/OpenMOSS/MOSS-TTS/tree/main/moss_soundeffect_v2
- Match upstream class/method/file names 1:1 (`WanAudioModel`, `MossSoundEffectPipeline`, `FlowMatchScheduler`). A reader must be able to diff port-vs-upstream and see only PyTorch↔MLX op substitutions. Do **not** refactor, rename, or "clean up" during the port — match first, optimize later with a framework-constraint justification.

## Swift donors (study before writing — do not reinvent)

- DiT + flow-matching end-to-end in MLX-Swift: `lucasnewman/f5-tts-swift` (MIT)
- DACVAE codec for MLX-Swift: `Blaizzy/mlx-audio-swift` → `MLXAudioCodecs` (evaluate its DACVAE before porting the decoder from scratch)
- Qwen3 text encoder: `mlx-swift-examples` (extract hidden states; do not use the LM head)

## Build chain (do not skip the Python middle layer)

**PyTorch ref → parity-locked MLX-Python → validated MLX-Swift.** Swift numerics can't be diffed against PyTorch directly; the MLX-Python package exists to be the parity oracle the Swift side validates against. Stage gates in `docs/handoff.md` §4 — do not advance a stage until its fp32 parity gate is green.

## Conformance

C0–C13 is **deferred** (standalone port for now). Build per-module parity fixtures anyway — they feed directly into the conformance parity items when that wrap happens later.

## Top silent-failure risks (full list in handoff §5)

1. **DAC-VAE latent scale constant** — missing/wrong fixed scale before decode → plausible-looking garbage audio, no error. Highest risk.
2. **DAC decoder conv layout** — PyTorch `(B,C,T)` vs MLX channel-last; transpose every conv weight.
3. **Flow-match sigma-shift** — `shift=5.0` + `extra_one_step=true` is an off-by-one trap; match the sigma schedule exactly.
4. CFG batch-doubling order; cross-attn padding leak; snake activation + weight-norm in the DAC decoder.

Run `noise_decode_check` / `detect_checkerboard` (from `tests/parity/_helpers.py`) before shipping — layer parity passing does NOT mean e2e is clean.

## Conversion trap (Tier-3 only)

MLX is lazy — **unevaluated tensors serialize as zeros, no error.** Call `mx.eval(weight)` immediately before every `mx.save_safetensors`. The DAC-VAE ships as a pickled full `.pth` object — load it, extract `.state_dict()`, re-serialize clean. Quantize only DiT `Linear.weight`; keep VAE + text encoder at bf16. Lock fp16/bf16 parity before any int4.

## Swift consumer idioms (the three that waste a day each)

- Load: `MLX.loadArrays` → `ModuleParameters.unflattened` → `model.update(parameters:, verify: .noUnusedKeys)`. `.noUnusedKeys` turns a remap bug into a boot failure instead of a silent zero-init layer.
- Dotted-key remap: safetensors `block.0.weight` ↔ Swift property `block_0` (props can't contain dots) — remap before `unflattened`.
- GPU-state classes `@unchecked Sendable` + `NSLock`; everything `Float32`/`Float16`/`BFloat16` (`Float64` crashes the GPU).
- Symptom "loads fine, `.noUnusedKeys` passes, garbage *only on Swift*" → it's one of the above before it's a layer-translation bug.

## Build / env

- Swift: build via **`xcodebuild`** against the workspace. SwiftPM CLI is not used for anything touching MLX/Metal. (Non-GPU test lane needing `swift test` → metallib-rename escape hatch in skill `repo-layout.md`.)
- Weight hosting (user-confirmed): code at `github.com/xocialize/moss-soundeffect-mlx`; weights published to **mlx-community** with weight-style naming — `mlx-community/MOSS-SoundEffect-v2.0-bf16` and `mlx-community/MOSS-SoundEffect-v2.0-4bit`. (Dustin owns both `xocialize` and `xocialize-code` orgs — neither hosts these weights; don't "correct" the targets from skill/handoff text, ask him.) Repo names follow the **mlx-community weight-style grammar** (preserve upstream case + quant suffix, never GGUF suffixes): `xocialize/MOSS-SoundEffect-v2.0-bf16` and `xocialize/MOSS-SoundEffect-v2.0-4bit` (int4 g64, DiT blocks-Linear only; VAE fp32 in both). Model card: `library_name: mlx`, `pipeline_tag: text-to-audio`, `base_model: OpenMOSS-Team/MOSS-SoundEffect-v2.0`, Apache-2.0, and an audio **Parity section** (e2e golden max_abs, int4 per-pass cosine). `MOSS_SFX_MLX_WEIGHTS_DIR` env var overrides for local dev.
- License: publish **Apache-2.0** (matches upstream + all deps).

## Resolve from upstream source before finalizing (handoff §6 — do in Stage 0)

DiT internals (RoPE flavor, QK-norm, AdaLN variant, FFN activation, pre/post-norm) · DAC-VAE frame rate in Hz (gates latent-length math, not in any config — read off the VAE object) · which Qwen3 hidden layer feeds cross-attn · the VAE latent scale constant · VAE weight provenance under Apache-2.0. All inferred from Wan2.1 lineage during assessment — verify against the actual modeling file, don't trust the inference.
