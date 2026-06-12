"""MLX transpose of the MOSS-SoundEffect inference path.

Mirrors pipeline_moss_soundeffect.py (MossSoundEffectPipeline) +
diffsynth/pipelines/wan_audio.py (WanAudioPipeline) for the T2A path the SFX
model uses. Video-lineage branches (image/vace/camera/teacache/sliding-window)
are not ported — the SFX checkpoint never exercises them.

Faithful behaviors (docs/upstream-findings.md):
  * prompt suffix " duration: {seconds:.1f}s"; always denoises the FULL
    max_inference_seconds latent and crops the waveform.
  * CFG: two separate forwards, combined fp32: nega + cfg * (posi - nega);
    skipped when cfg_scale == 1.0.
  * timestep path fp32; VAE decode fp32.
  * NO latent scale constant anywhere.

RNG note: mx.random is not torch-compatible. Same seed != same audio as CUDA
reference. Inject `latents` for parity work.
"""

import json
from pathlib import Path
from typing import List, Optional, Union

import mlx.core as mx
from tqdm import tqdm

from .model.dac_vae import DAC
from .model.qwen3_text_encoder import Qwen3TextEncoder
from .model.wan_audio_dit import WanAudioModel
from .prompter import WanPrompter
from .scheduler import FlowMatchScheduler
from .utils.weights import resolve_weights_dir


class WanAudioPipeline:

    def __init__(self, dit, vae, text_encoder, prompter, scheduler,
                 dtype=mx.bfloat16):
        self.dit = dit
        self.vae = vae
        self.text_encoder = text_encoder
        self.prompter = prompter
        self.scheduler = scheduler
        self.dtype = dtype
        self.audio_latent_dim = 128
        self.num_samples_division_factor = vae.hop_length

    def check_resize_num_channels_num_samples(self, num_channels, num_samples):
        if num_samples % self.num_samples_division_factor != 0:
            num_samples = num_samples // self.num_samples_division_factor * self.num_samples_division_factor
        return num_channels, num_samples

    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = "",
        seed: Optional[int] = None,
        num_samples=44100 * 10,
        num_channels=1,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        latents: Optional[mx.array] = None,
        progress_bar_cmd=tqdm,
    ):
        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)

        batch_size = len(prompt) if isinstance(prompt, (list, tuple)) else 1
        if batch_size > 1 and not isinstance(negative_prompt, (list, tuple)):
            negative_prompt = [negative_prompt] * batch_size

        _, num_samples = self.check_resize_num_channels_num_samples(num_channels, num_samples)

        context_posi = self.prompter.encode_prompt(prompt).astype(self.dtype)
        context_nega = self.prompter.encode_prompt(negative_prompt).astype(self.dtype)

        if latents is None:
            if seed is not None:
                mx.random.seed(seed)
            latents = mx.random.normal(
                (batch_size, self.audio_latent_dim, num_samples // self.num_samples_division_factor)
            ).astype(self.dtype)
        else:
            latents = latents.astype(self.dtype)

        for progress_id in progress_bar_cmd(range(len(self.scheduler.timesteps))):
            timestep = self.scheduler.timesteps[progress_id].reshape(1)

            noise_pred_posi = self.dit(latents, timestep, context_posi)
            if cfg_scale != 1.0:
                noise_pred_nega = self.dit(latents, timestep, context_nega)
                # CFG combined in fp32, exactly like upstream (wan_audio.py:403-405).
                noise_pred = noise_pred_nega.astype(mx.float32) + cfg_scale * (
                    noise_pred_posi.astype(mx.float32) - noise_pred_nega.astype(mx.float32)
                )
            else:
                noise_pred = noise_pred_posi

            latents = self.scheduler.step(
                noise_pred, self.scheduler.timesteps[progress_id], latents.astype(mx.float32)
            ).astype(self.dtype)
            mx.eval(latents)  # keep Metal command buffers bounded

        # Decode at fp32 (upstream decodes under fp32 autocast).
        audio = self.vae.decode(latents.astype(mx.float32))
        mx.eval(audio)
        return audio


class MossSoundEffectPipeline:
    """diffusers-style facade matching upstream MossSoundEffectPipeline."""

    def __init__(self, engine: WanAudioPipeline, sample_rate: int = 48000,
                 max_inference_seconds: int = 30):
        self.engine = engine
        self.sample_rate = int(sample_rate)
        self.max_inference_seconds = int(max_inference_seconds)

    @classmethod
    def from_pretrained(cls, model_dir=None, dtype=mx.bfloat16,
                        dit_file="dit.safetensors", quantization=None):
        """Load from the original HF snapshot dir + its `mlx/` converted subdir.

        Run scripts/convert_weights.py once to produce mlx/dit.safetensors and
        mlx/vae.safetensors next to the original weights. For an int4 DiT pass
        e.g. dit_file="dit_int4_g64.safetensors", quantization=(4, 64)
        (bits, group_size — produced by scripts/quantize.py).
        """
        import mlx.nn as mlx_nn
        from mlx.utils import tree_unflatten

        from .config import DACConfig, WanAudioModelConfig

        model_dir = Path(resolve_weights_dir(model_dir))
        mlx_dir = model_dir / "mlx"
        if not (mlx_dir / dit_file).exists():
            raise FileNotFoundError(
                f"{mlx_dir / dit_file} not found — run scripts/convert_weights.py first"
            )

        with open(model_dir / "model_index.json") as f:
            index = json.load(f)

        # Published -4bit repos record their DiT quantization in
        # model_index.json so callers need no extra args.
        if quantization is None and "mlx_quantization" in index:
            q = index["mlx_quantization"]
            quantization = (int(q["bits"]), int(q["group_size"]))

        cfg = WanAudioModelConfig()
        dit = WanAudioModel(
            dim=cfg.dim, in_dim=cfg.in_dim, ffn_dim=cfg.ffn_dim, out_dim=cfg.out_dim,
            text_dim=cfg.text_dim, freq_dim=cfg.freq_dim, eps=cfg.eps,
            patch_size=cfg.patch_size, num_heads=cfg.num_heads,
            num_layers=cfg.num_layers, has_image_input=cfg.has_image_input,
            vae_type=cfg.vae_type,
        )
        if quantization is not None:
            bits, group_size = quantization
            mlx_nn.quantize(
                dit, group_size=group_size, bits=bits,
                class_predicate=lambda path, m: isinstance(m, mlx_nn.Linear) and path.startswith("blocks."),
            )
        dit_weights = mx.load(str(mlx_dir / dit_file))
        # Quantized tensors (packed uint32 / scales) must keep their dtype.
        dit.update(tree_unflatten([
            (k, v if v.dtype in (mx.uint32,) or quantization is not None else v.astype(dtype))
            for k, v in dit_weights.items()
        ]))

        vcfg = DACConfig()
        vae = DAC(
            encoder_dim=vcfg.encoder_dim, encoder_rates=vcfg.encoder_rates,
            latent_dim=vcfg.latent_dim, decoder_dim=vcfg.decoder_dim,
            decoder_rates=vcfg.decoder_rates, sample_rate=vcfg.sample_rate,
            continuous=vcfg.continuous,
        )
        vae_weights = mx.load(str(mlx_dir / "vae.safetensors"))
        # VAE stays fp32 — upstream decodes under fp32 autocast.
        vae.update(tree_unflatten([(k, v.astype(mx.float32)) for k, v in vae_weights.items()]))

        text_encoder = Qwen3TextEncoder(model_dir / "text_encoder", dtype=dtype)
        prompter = WanPrompter(tokenizer_path=str(model_dir / "tokenizer"))
        prompter.fetch_models(text_encoder)

        scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        engine = WanAudioPipeline(dit, vae, text_encoder, prompter, scheduler, dtype=dtype)
        mx.eval(dit.parameters(), vae.parameters())

        return cls(
            engine=engine,
            sample_rate=int(index.get("sample_rate", 48000)),
            max_inference_seconds=int(index.get("max_inference_seconds", 30)),
        )

    def __call__(
        self,
        prompt: Union[str, List[str]],
        seconds: float = 10.0,
        num_inference_steps: int = 100,
        cfg_scale: float = 4.0,
        sigma_shift: float = 5.0,
        seed: int = 0,
        negative_prompt: str = "",
        append_duration_suffix: bool = True,
        num_channels: int = 1,
        max_inference_seconds: Optional[int] = None,
        latents: Optional[mx.array] = None,
        progress_bar_cmd=tqdm,
    ):
        seconds = round(float(seconds), 1)
        if seconds <= 0:
            raise ValueError(f"seconds must be > 0, got {seconds}")
        full_seconds = int(max_inference_seconds or self.max_inference_seconds)
        if seconds > full_seconds:
            raise ValueError(
                f"seconds={seconds} exceeds max_inference_seconds={full_seconds}"
            )

        def _format(p: str) -> str:
            p = p.strip()
            return f"{p} duration: {seconds:.1f}s" if append_duration_suffix else p

        if isinstance(prompt, (list, tuple)):
            prompts = [_format(p) for p in prompt]
        else:
            prompts = [_format(prompt)]

        num_samples_full = self.sample_rate * full_seconds
        audio = self.engine(
            prompt=prompts if len(prompts) > 1 else prompts[0],
            negative_prompt=negative_prompt,
            seed=int(seed),
            cfg_scale=float(cfg_scale),
            sigma_shift=float(sigma_shift),
            num_inference_steps=int(num_inference_steps),
            num_samples=num_samples_full,
            num_channels=int(num_channels),
            latents=latents,
            progress_bar_cmd=progress_bar_cmd,
        )

        output_samples = int(self.sample_rate * seconds)
        return audio[:, :, :output_samples]
