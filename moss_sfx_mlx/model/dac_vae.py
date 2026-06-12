"""MLX transpose of moss_soundeffect_v2/diffsynth/models/dac_vae.py.

Continuous-VAE paths only — the shipped vae_128d_48k.pth has continuous=True;
the RVQ branch (VectorQuantize / ResidualVectorQuantize) and the audiotools
CodecMixin (compress/decompress) are not used by the SFX pipeline and are
intentionally absent. Everything else matches upstream structure and op order
1:1 with two framework-constraint deviations, both invisible at module
boundaries:

  * Layout: MLX convs are channel-last, so tensors flow (B, T, C) internally;
    `encode`/`decode`/`__call__` keep upstream's (B, C, T) interface and
    transpose at the boundary. Snake's alpha is stored (1, 1, C) instead of
    (1, C, 1); conversion transposes it.
  * Weight norm: upstream wraps every conv in torch weight_norm (weight_g /
    weight_v). MLX layers hold the fused weight (g * v / ||v||) — fuse at
    conversion (inference-only, mathematically identical).
"""

import math
from typing import List, Union

import mlx.core as mx
import mlx.nn as nn


def WNConv1d(*args, **kwargs):
    # weight_norm is fused into the weight at conversion time.
    return nn.Conv1d(*args, **kwargs)


def WNConvTranspose1d(*args, **kwargs):
    return nn.ConvTranspose1d(*args, **kwargs)


def snake(x, alpha):
    # Upstream reshapes (B, C, -1) around the op — a no-op for 1-d audio.
    return x + mx.reciprocal(alpha + 1e-9) * mx.power(mx.sin(alpha * x), 2)


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # (1, 1, C) for channel-last; upstream is (1, C, 1).
        self.alpha = mx.ones((1, 1, channels))

    def __call__(self, x):
        return snake(x, self.alpha)


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        # Channel split — axis -1 here (channel-last); upstream chunks dim=1.
        self.mean, self.logvar = mx.split(parameters, 2, axis=-1)
        self.logvar = mx.clip(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = mx.exp(0.5 * self.logvar)
        self.var = mx.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = mx.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * mx.random.normal(self.mean.shape)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return mx.array([0.0])
        if other is None:
            return 0.5 * mx.mean(
                mx.power(self.mean, 2) + self.var - 1.0 - self.logvar,
                axis=[1, 2],
            )
        return 0.5 * mx.mean(
            mx.power(self.mean - other.mean, 2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            axis=[1, 2],
        )

    def mode(self):
        return self.mean


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def __call__(self, x):
        y = self.block(x)
        # Time axis is 1 (channel-last); upstream crops axis -1.
        pad = (x.shape[1] - y.shape[1]) // 2
        if pad > 0:
            x = x[:, pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def __call__(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        self.block = nn.Sequential(*block)
        self.enc_dim = d_model

    def __call__(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def __call__(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def __call__(self, x):
        return self.model(x)


class DAC(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        continuous: bool = False,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate
        self.continuous = continuous

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = math.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        if not continuous:
            raise NotImplementedError(
                "RVQ (continuous=False) is not used by MOSS-SoundEffect-v2.0; "
                "the shipped vae_128d_48k.pth is continuous=True."
            )
        self.quant_conv = nn.Conv1d(latent_dim, 2 * latent_dim, 1)
        self.post_quant_conv = nn.Conv1d(latent_dim, latent_dim, 1)

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )
        self.sample_rate = sample_rate

    def preprocess(self, audio_data, sample_rate):
        # audio_data: (B, 1, T) channel-first like upstream.
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = mx.pad(audio_data, [(0, 0), (0, 0), (0, right_pad)])

        return audio_data

    def encode(self, audio_data: mx.array):
        """audio_data: (B, 1, T) -> DiagonalGaussianDistribution over (B, D, T_lat).

        Returns the same 5-tuple as upstream's continuous branch:
        (posterior, codes, latents, commitment_loss, codebook_loss).
        """
        z = self.encoder(audio_data.transpose(0, 2, 1))     # (B, T_lat, D)
        z = self.quant_conv(z)                              # (B, T_lat, 2D)
        z = z.transpose(0, 2, 1)                            # (B, 2D, T_lat) — upstream layout
        # DiagonalGaussianDistribution here splits the channel axis; feed it
        # channel-last to keep its internals simple, then expose (B, D, T).
        posterior = DiagonalGaussianDistribution(z.transpose(0, 2, 1))
        posterior.mean = posterior.mean.transpose(0, 2, 1)
        posterior.logvar = posterior.logvar.transpose(0, 2, 1)
        posterior.std = posterior.std.transpose(0, 2, 1)
        posterior.var = posterior.var.transpose(0, 2, 1)
        codes, latents, commitment_loss, codebook_loss = None, None, 0, 0
        return posterior, codes, latents, commitment_loss, codebook_loss

    def decode(self, z: mx.array):
        """z: (B, D, T_lat) raw DiT latents -> (B, 1, T) waveform in [-1, 1].

        NO scale constant — post_quant_conv is the learned equivalent
        (docs/upstream-findings.md §4). Do not add one.
        """
        z = z.transpose(0, 2, 1)                            # (B, T_lat, D)
        z = self.post_quant_conv(z)
        audio = self.decoder(z)                             # (B, T, 1)
        return audio.transpose(0, 2, 1)                     # (B, 1, T)

    def __call__(self, audio_data: mx.array, sample_rate: int = None):
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        posterior, _, _, _, _ = self.encode(audio_data)
        z = posterior.sample()
        x = self.decode(z)
        kl_loss = posterior.kl().mean()
        return {
            "audio": x[..., :length],
            "z": z,
            "kl_loss": kl_loss,
        }
