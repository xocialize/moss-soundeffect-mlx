"""PyTorch -> MLX state-dict conversion for the DAC VAE and the DiT.

Used by the parity tests and the Stage-1 conversion recipe. Input is the raw
torch state_dict from vae_128d_48k.pth (old-style weight_norm: weight_g /
weight_v pairs); output is a flat {key: np.ndarray} dict matching the MLX
module tree in moss_sfx_mlx/model/dac_vae.py.

Transforms applied:
  1. weight-norm fusion: w = g * v / ||v||  (norm over all dims except 0,
     torch weight_norm dim=0 default — same rule for Conv1d and ConvTranspose1d)
  2. container rename: torch Sequential indices -> MLX `.layers.` attribute
  3. layout transpose:
       Conv1d          (O, I, K) -> (O, K, I)
       ConvTranspose1d (I, O, K) -> (O, K, I)
       Snake alpha     (1, C, 1) -> (1, 1, C)

ConvTranspose detection is structural: the only ConvTranspose1d in the model
is the upsampler at `decoder.model.<i>.block.1` (DecoderBlock position 1).
"""

import re

import numpy as np

_CONVT_RE = re.compile(r"^decoder\.model\.\d+\.block\.1\.weight$")


def fuse_weight_norm(state_dict: dict) -> dict:
    """Fuse weight_g/weight_v pairs into plain `weight` (numpy, fp32)."""
    out = {}
    for k, v in state_dict.items():
        arr = np.asarray(v.detach().cpu().float().numpy() if hasattr(v, "detach") else v)
        if k.endswith(".weight_g"):
            continue
        if k.endswith(".weight_v"):
            base = k[: -len(".weight_v")]
            g = state_dict[base + ".weight_g"]
            g = np.asarray(g.detach().cpu().float().numpy() if hasattr(g, "detach") else g)
            norm = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1).reshape(
                (-1,) + (1,) * (arr.ndim - 1)
            )
            out[base + ".weight"] = g * arr / norm
        else:
            out[k] = arr
    return out


def rename_dac_key(key: str) -> str:
    key = key.replace(".block.", ".block.layers.")
    key = key.replace(".model.", ".model.layers.")
    return key


def dac_pt_to_mlx(state_dict: dict) -> dict:
    """Full torch->MLX conversion. Returns {mlx_key: np.ndarray}."""
    fused = fuse_weight_norm(state_dict)
    out = {}
    for k, arr in fused.items():
        if k.endswith(".alpha"):
            arr = arr.transpose(0, 2, 1)            # (1, C, 1) -> (1, 1, C)
        elif k.endswith(".weight") and arr.ndim == 3:
            if _CONVT_RE.match(k):
                arr = arr.transpose(1, 2, 0)        # ConvT (I, O, K) -> (O, K, I)
            else:
                arr = arr.transpose(0, 2, 1)        # Conv (O, I, K) -> (O, K, I)
        out[rename_dac_key(k)] = np.ascontiguousarray(arr)
    return out


# ---------------------------------------------------------------------------
# DiT (WanAudioModel)
#
# The HF checkpoint uses diffusers-style keys. Upstream maps them to native
# names in _convert_hf_dit_state_dict (pipelines/wan_audio.py:51-114); we
# apply the same mapping plus the MLX Sequential `.layers.` insert and the
# patch-embedding conv transpose.
# ---------------------------------------------------------------------------

_DIT_GLOBAL_RENAME = {
    "condition_embedder.text_embedder.linear_1": "text_embedding.layers.0",
    "condition_embedder.text_embedder.linear_2": "text_embedding.layers.2",
    "condition_embedder.time_embedder.linear_1": "time_embedding.layers.0",
    "condition_embedder.time_embedder.linear_2": "time_embedding.layers.2",
    "condition_embedder.time_proj": "time_projection.layers.1",
    "proj_out": "head.head",
    "patch_embedding": "patch_embedding",
}

_DIT_BLOCK_RENAME = {
    "attn1.norm_q": "self_attn.norm_q",
    "attn1.norm_k": "self_attn.norm_k",
    "attn1.to_q": "self_attn.q",
    "attn1.to_k": "self_attn.k",
    "attn1.to_v": "self_attn.v",
    "attn1.to_out.0": "self_attn.o",
    "attn2.norm_q": "cross_attn.norm_q",
    "attn2.norm_k": "cross_attn.norm_k",
    "attn2.to_q": "cross_attn.q",
    "attn2.to_k": "cross_attn.k",
    "attn2.to_v": "cross_attn.v",
    "attn2.to_out.0": "cross_attn.o",
    "ffn.net.0.proj": "ffn.layers.0",
    "ffn.net.2": "ffn.layers.2",
    "norm2": "norm3",
}


def rename_dit_key(key: str) -> str:
    if key == "scale_shift_table":
        return "head.modulation"
    for old, new in _DIT_GLOBAL_RENAME.items():
        if key.startswith(old + "."):
            return new + key[len(old):]
    if key.startswith("blocks."):
        _, idx, suffix = key.split(".", 2)
        if suffix == "scale_shift_table":
            return f"blocks.{idx}.modulation"
        for old, new in _DIT_BLOCK_RENAME.items():
            if suffix.startswith(old + "."):
                return f"blocks.{idx}.{new}{suffix[len(old):]}"
    return key


def dit_hf_to_mlx(state_dict: dict) -> dict:
    """diffusers-keyed DiT checkpoint -> {mlx_key: np.ndarray}."""
    out = {}
    for k, v in state_dict.items():
        arr = np.asarray(v.detach().cpu().float().numpy() if hasattr(v, "detach") else v)
        if k == "patch_embedding.weight":
            arr = arr.transpose(0, 2, 1)  # Conv1d (O, I, K) -> (O, K, I)
        out[rename_dit_key(k)] = np.ascontiguousarray(arr)
    return out
