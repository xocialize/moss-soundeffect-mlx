# Upstream source findings — §6 open items resolved

Resolved 2026-06-11 by reading `OpenMOSS/MOSS-TTS @ main` (`moss_soundeffect_v2/`),
local clone at `/Volumes/DEV_ARCHIVE/_ref/MOSS-TTS`. File:line refs are into that clone.

## 1. DiT internals (`diffsynth/models/wan_audio_dit.py` + `wan_video_dit.py`)

| Question | Answer | Source |
|---|---|---|
| RoPE flavor | **Interleaved even/odd pairs** (`x[..., d/2, 2]`, rotate `(even, odd)` pairs), complex-polar freqs precomputed in float64. NOT half-split. fp32 compute inside `rope_apply`, cast back to input dtype. | `wan_video_dit.py:104-118` |
| RoPE freqs (dac path) | `precompute_freqs_cis_1d(head_dim=128, end=16384, theta=10000)` over the **full head_dim**, then `.chunk(3)` and re-concatenated in forward — net effect: plain 1-D RoPE over all 128 dims. The 3-way chunk is a vestige of the 3D-video lineage. | `wan_audio_dit.py:42-44, 230-234` |
| QK-norm | **Yes — RMSNorm over the FULL model dim (1536), applied to q/k BEFORE the head split**, eps=1e-6, learned weight. Uses `torch.nn.RMSNorm` (fp32 upcast internally). Not per-head — porting it per-head would be a silent bug. | `wan_video_dit.py:152-166` |
| AdaLN variant | `x * (1 + scale) + shift`. Per-block learned `modulation` table `(1, 6, dim)` **added to** `t_mod` then chunked 6-way: shift/scale/gate for MSA, shift/scale/gate for FFN. Head has its own `(1, 2, dim)` table (shift, scale — in that chunk order). | `wan_video_dit.py:235-251`, `Head:274-290` |
| FFN activation | `Linear → GELU(approximate='tanh') → Linear`. **No SwiGLU/GEGLU.** | `wan_video_dit.py:230-231` |
| Pre/post-norm | Pre-norm. `norm1`/`norm2` = LayerNorm **without affine** (modulated by AdaLN), `norm3` = LayerNorm **with affine** (feeds cross-attn). | `wan_video_dit.py:227-229` |
| Block order | `x = x + gate_msa * self_attn(modulate(norm1(x)))` → `x = x + cross_attn(norm3(x), ctx)` (un-gated, un-modulated) → `x = x + gate_mlp * ffn(modulate(norm2(x)))` | `wan_video_dit.py:235-251` |
| Cross-attn | Same RMSNorm-q/k as self-attn, **no RoPE**, no mask. `has_image_input=False` for this model. | `wan_video_dit.py:171-207` |
| Text embedding | `Linear(2048→1536) → GELU(tanh) → Linear(1536→1536)` | `wan_audio_dit.py:144-148` |
| Time embedding | `sinusoidal_embedding_1d(freq_dim=256, t)` computed in **float64** upstream (use fp32 in MLX; parity-check this), then `Linear→SiLU→Linear`, then `time_projection = SiLU→Linear(dim→6*dim)`. Timestep path runs under fp32 autocast in the reference pipeline. | `wan_audio_dit.py:18-22, 149-155`; `wan_audio.py:755-758` |
| Patch embed | `nn.Conv1d(128, 1536, kernel_size=1, stride=1)` (patch_size=[1] → effectively a pointwise linear). Unpatchify = `rearrange('b f (p c) -> b c (f p)')` with p=1. | `wan_audio_dit.py:141-143, 191-206` |
| Attention | Plain SDPA, non-causal, no mask, heads stacked (NOT interleaved QKV — separate q/k/v Linears). `head_dim = 1536/12 = 128`. | `wan_video_dit.py:36-73` |

**Note:** the inference path does NOT call `WanAudioModel.forward` — it uses the standalone
`model_fn_wan_video(...)` in `pipelines/wan_audio.py:684-861`, which replicates the same op
order (and is the thing wrapped in `torch.compile`). The two are equivalent for our config
(no image/vace/reference branches). Port `WanAudioModel.forward` and parity-test against
`model_fn_wan_video` outputs.

## 2. DAC-VAE latent frame rate (`diffsynth/models/dac_vae.py`)

- `hop_length = prod(encoder_rates)` (`dac_vae.py:833`). Latent length `T = num_samples // hop_length`
  (`wan_audio.py:458`), with `num_samples` rounded DOWN to a hop multiple (`wan_audio.py:181-186`).
- **Confirmed from the shipped checkpoint** (pickletools disassembly of `vae_128d_48k.pth`,
  2026-06-11 — NOT the class defaults):
  - `encoder_rates = [2, 3, 4, 5, 8]` → `hop_length = 960` → frame rate = 48000/960 = **50 Hz**
  - `decoder_rates = [8, 5, 4, 3, 2]` (5 upsample blocks)
  - `encoder_dim = 128`, `decoder_dim = 2048`, `latent_dim = 128`, `sample_rate = 48000`, `continuous = True`
  - 30 s × 48 kHz = 1,440,000 samples → latent `T = 1500`.
- Decoder output passes through final `nn.Tanh()` — waveform bounded [-1, 1] (`Decoder:792-796`).

## 3. Which Qwen3 hidden layer feeds cross-attn

**Final layer**: `outputs.hidden_states[-1]` (`qwen3_text_encoder.py:40`). Loaded via
`AutoModelForCausalLM` but the LM head is never used. bf16. Frozen, `use_cache=False`.

## 4. VAE latent scale constant

**There is no fixed scale constant.** The continuous DAC variant replaces it with learned convs:
- encode: `z = quant_conv(encoder(x))` → `DiagonalGaussianDistribution(z)`; inference uses `.mode()` (= mean) (`dac_vae.py:922-930`, `wan_audio.py:493-497`)
- decode: `audio = decoder(post_quant_conv(z))` — raw DiT latents go straight in (`dac_vae.py:932-955`, `wan_audio.py:425-429`)

The latents are NOT normalized/scaled anywhere in the pipeline. ⚠️ **Adding an SD-style
`scaling_factor` would itself be the silent-failure bug.** The `post_quant_conv` (1×1 Conv1d,
128→128) and `quant_conv` (1×1, 128→256) must not be dropped.
VAE decode runs under **fp32 autocast** upstream (`wan_audio.py:425`) — keep the MLX VAE fp32/bf16-checked.

## 5. DAC-VAE weight provenance

Code is MIT-lineage (Descript DAC, `dac_vae.py` docstrings); MOSS-TTS repo README states
Apache-2.0 for the family incl. retrained VAE weights. Close the loop with a one-line README
citation at publish time (Stage 4). No gate.

## Additional pipeline facts (not in §6 but load-bearing)

| Fact | Detail | Source |
|---|---|---|
| Duration conditioning | Prompt suffix `" duration: {seconds:.1f}s"` appended (training convention, `append_duration_suffix=True`). | `pipeline_moss_soundeffect.py:198-205` |
| Fixed-size denoise | Always denoises the FULL `max_inference_seconds=30` s latent; output waveform is cropped to `seconds`. Latent T does not vary with `seconds`. | `pipeline_moss_soundeffect.py:171-223` |
| CFG | **Two separate forward passes** (cond first, then uncond; `cfg_merge=False` default — no batch-doubling). Combined in fp32: `nega + cfg_scale * (posi - nega)`. Skipped entirely when `cfg_scale == 1.0`. | `wan_audio.py:398-407` |
| Negative prompt | Default `""` (empty string), still tokenized/padded to 512. **The resulting context is ALL ZEROS**: Qwen3's tokenizer adds no special tokens, so "" has 0 valid tokens and the prompter zeroes every position — the CFG unconditional pass runs on zero conditioning. Verified in golden capture. | `pipeline_moss_soundeffect.py:164`, `wan_prompter.py:108-110` |
| Scheduler ctor | `FlowMatchScheduler(shift=5.0 (from scheduler_config.json), sigma_min=0.0, extra_one_step=True)`. NB class defaults differ (shift=3.0, sigma_min=0.003/1.002) — the pipeline overrides are what matter. | `wan_audio.py:124` |
| Sigma schedule | `sigmas = linspace(1.0, 0.0, steps+1)[:-1]`, then `sigmas = 5*s/(1+4*s)`; `timesteps = sigmas*1000`. Euler step: `x += v*(σ_next − σ)`, last step σ_next=0. `step()` looks up the timestep by nearest-match (`argmin |timesteps - t|`), not by index. | `flow_match.py:34-85` |
| Velocity convention | Model predicts `noise − x₀`; `x_t = (1−σ)·x₀ + σ·noise`. | `flow_match.py:97-116` |
| Tokenizer | Qwen3 tokenizer, `padding='max_length'`, `truncation=True`, `max_length=512` (`text_len=512`), whitespace clean (ftfy + html unescape + collapse). | `wan_prompter.py:35-93` |
| Padding handling | Attention mask passed to Qwen3; then **embeddings at pad positions are zeroed** (`prompt_emb[i, v:] = 0`). Cross-attn itself is mask-free over all 512 positions. Replicate the zeroing exactly. | `wan_prompter.py:98-110` |
| Noise init | `(B, 128, num_samples // hop_length)`, `generate_noise` with seed on CPU (`rand_device="cpu"`). For parity: inject golden noise — torch RNG is not reproducible in MLX. | `wan_audio.py:449-460` |
| Autocast | Pipeline runs under bf16 autocast; timestep embedding + CFG combine + VAE decode forced fp32. | `pipeline_moss_soundeffect.py:209`, `wan_audio.py:425,755` |
| Weight key mapping | HF-exported DiT checkpoint uses diffusers-style keys; `_convert_hf_dit_state_dict` (`wan_audio.py:51-114`) maps them to native names (`attn1→self_attn`, `ffn.net.0.proj→ffn.0`, `scale_shift_table→modulation`, …). The conversion recipe must apply the same mapping. |
| DAC weight norm | Every VAE conv is `weight_norm`-wrapped (`weight_g`/`weight_v`). Fuse at conversion: `w = g * v / ‖v‖` (dim=0 for Conv1d, per weight_norm default... verify dims for ConvTranspose1d). Snake: `x + (α+1e-9)⁻¹·sin²(αx)`, α shape `(1, C, 1)`. | `dac_vae.py:307-333` |

## Swift donor evaluation (Stage 3, 2026-06-11)

- **mlx-audio-swift `MLXAudioCodecs/DACVAE`**: evaluated per handoff — it is the
  *watermarked* DACVAE variant (extra ELU watermark upsample path `block_3`,
  `wmStride`), NOT our plain continuous VAE (224 tensors, no watermark branch).
  Custom port in `swift/Sources/.../DACVAEDecoder.swift` was the right call; their
  `DACVAEWNConvTranspose1d` validates the custom-ConvTranspose-module pattern.
- **f5-tts-swift**: used as the Module/@ModuleInfo/Package.swift idiom reference.
- **Qwen3 (mlx-swift-examples)**: deferred — Swift text encoder is behind a
  `TextEncoding` protocol; parity tests inject golden contexts.
- **Swift validation is GPU-free**: CPU stream needs no metallib and is
  bitwise-stable vs the Python oracle (mlx-swift skill, porting.md §1/§3).
