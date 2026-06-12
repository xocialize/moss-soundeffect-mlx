"""Configs mirroring the shipped OpenMOSS-Team/MOSS-SoundEffect-v2.0 repo verbatim.

Field names and defaults match the upstream JSON files / pickled VAE kwargs 1:1.
Do not rename fields or "clean up" defaults — the reference is the oracle.
Sources (verified 2026-06-11, see docs/upstream-findings.md):
  - transformer/config.json
  - scheduler/scheduler_config.json
  - model_index.json
  - vae/vae_128d_48k.pth pickled ctor kwargs
"""

from dataclasses import dataclass, field


@dataclass
class WanAudioModelConfig:
    # transformer/config.json
    in_dim: int = 128
    out_dim: int = 128
    text_dim: int = 2048
    freq_dim: int = 256
    eps: float = 1e-6
    patch_size: tuple = (1,)
    has_image_input: bool = False
    vae_type: str = "dac"
    dim: int = 1536
    ffn_dim: int = 8960
    num_heads: int = 12
    num_layers: int = 30


@dataclass
class FlowMatchSchedulerConfig:
    # scheduler/scheduler_config.json — NB these override the upstream class
    # defaults (shift=3.0, sigma_min=0.003/1.002); the JSON values are correct.
    shift: float = 5.0
    sigma_min: float = 0.0
    extra_one_step: bool = True
    num_train_timesteps: int = 1000


@dataclass
class DACConfig:
    # Pickled ctor kwargs inside vae/vae_128d_48k.pth (NOT the dac-package
    # class defaults). hop_length = prod(encoder_rates) = 960 -> 50 Hz latents.
    encoder_dim: int = 128
    encoder_rates: list = field(default_factory=lambda: [2, 3, 4, 5, 8])
    latent_dim: int = 128
    decoder_dim: int = 2048
    decoder_rates: list = field(default_factory=lambda: [8, 5, 4, 3, 2])
    sample_rate: int = 48000
    continuous: bool = True

    @property
    def hop_length(self) -> int:
        n = 1
        for r in self.encoder_rates:
            n *= r
        return n


@dataclass
class Qwen3TextEncoderConfig:
    # text_encoder/config.json (Qwen3-1.7B-Base)
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 40960
    # rope_scaling: null, use_sliding_window: false — plain RoPE, no YaRN.
    tie_word_embeddings: bool = True
    attention_bias: bool = False


@dataclass
class PipelineConfig:
    # model_index.json
    dit_variant: str = "1.3B"
    sample_rate: int = 48000
    max_inference_seconds: int = 30
    vae_type: str = "dac"
    text_encoder_type: str = "qwen3"
    # WanPrompter
    text_len: int = 512
    # MossSoundEffectPipeline.__call__ defaults
    num_inference_steps: int = 100
    cfg_scale: float = 4.0
    sigma_shift: float = 5.0
