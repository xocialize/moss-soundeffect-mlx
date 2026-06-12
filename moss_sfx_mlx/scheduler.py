"""FlowMatchScheduler — MLX transpose of
moss_soundeffect_v2/diffsynth/schedulers/flow_match.py (upstream oracle).

Structure, method names, and math match upstream 1:1; only torch -> mlx op
substitutions. Inference-relevant paths only differ where noted.
The pipeline constructs this as
FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
(class defaults intentionally mirror upstream's, which differ — see config.py).
"""

import math

import mlx.core as mx


def _linspace(start, stop, num):
    # torch.linspace inclusive of both endpoints.
    if num == 1:
        return mx.array([start], dtype=mx.float32)
    step = (stop - start) / (num - 1)
    return start + step * mx.arange(num, dtype=mx.float32)


class FlowMatchScheduler:

    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None, dynamic_shift_len=None):
        if shift is not None:
            self.shift = shift
        # sigma is the noise strength at each step.
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = _linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            # Linear schedule from high noise to low noise.
            self.sigmas = _linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = self.sigmas[::-1]
        if self.exponential_shift:
            mu = self.calculate_shift(dynamic_shift_len) if dynamic_shift_len is not None else self.exponential_shift_mu
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            # Classic flow-match shift formula.
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            # BSMNTW weighting — training-only path, kept for 1:1 diffability.
            x = self.timesteps
            y = mx.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        # Upstream looks the timestep up by nearest match, not by index.
        timestep_id = int(mx.argmin(mx.abs(self.timesteps - timestep)))
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            # Last step: jump straight to the boundary.
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(self, timestep, sample, sample_stablized):
        timestep_id = int(mx.argmin(mx.abs(self.timesteps - timestep)))
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output

    def add_noise(self, original_samples, noise, timestep):
        timestep = mx.array(timestep) if not isinstance(timestep, mx.array) else timestep
        timestep = mx.reshape(timestep, (-1,))
        # [B, len_timesteps] distance matrix -> nearest timestep id per sample.
        dists = mx.abs(self.timesteps[None, :] - timestep[:, None])
        timestep_ids = mx.argmin(dists, axis=1)
        sigmas = self.sigmas[timestep_ids].reshape(-1, 1, 1)
        # x_t = (1 - sigma) * x_0 + sigma * eps
        sample = (1 - sigmas) * original_samples + sigmas * noise
        return sample

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):
        timestep = mx.reshape(mx.array(timestep), (-1,))
        dists = mx.abs(self.timesteps[None, :] - timestep[:, None])
        timestep_ids = mx.argmin(dists, axis=1)
        weights = self.linear_timesteps_weights[timestep_ids]
        return weights

    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu
