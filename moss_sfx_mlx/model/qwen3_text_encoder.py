"""MLX transpose of moss_soundeffect_v2/diffsynth/models/qwen3_text_encoder.py.

Wraps Qwen3 (decoder-only) as a text encoder; reuses mlx-lm's qwen3
implementation per the handoff (do not reinvent). Returns last-layer hidden
states — in HF transformers `hidden_states[-1]` is the post-final-norm output,
which is exactly what `Qwen3Model.__call__` (mlx-lm) returns.

Attention-mask note: upstream passes the pad mask to the HF model. With
right-padding and causal attention, pad positions cannot influence real-token
rows, and the prompter zeroes pad-position embeddings afterwards anyway
(wan_prompter.py:108-110) — so a plain causal mask is equivalent post-zeroing.
The parity test asserts this on the actually-fed (zeroed) embeddings.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3


class Qwen3TextEncoder(nn.Module):
    """Interface matches upstream: forward(ids, mask) -> [B, L, hidden] last-layer states."""

    def __init__(self, model_path, dtype=mx.bfloat16):
        super().__init__()
        model_path = Path(model_path)
        with open(model_path / "config.json") as f:
            config = json.load(f)
        args = qwen3.ModelArgs.from_dict(config)
        self.model = qwen3.Model(args)

        weights = {}
        for shard in sorted(model_path.glob("model*.safetensors")):
            weights.update(mx.load(str(shard)))
        if hasattr(self.model, "sanitize"):
            weights = self.model.sanitize(weights)
        weights = {k: v.astype(dtype) for k, v in weights.items()}
        self.model.load_weights(list(weights.items()))
        self.model.eval()
        self.dim = config["hidden_size"]  # 2048 for Qwen3-1.7B

    def __call__(self, ids, mask=None):
        """
        Args:
            ids:  [batch, seq_len] token ids
            mask: accepted for interface parity; see module docstring.
        Returns:
            hidden_states: [batch, seq_len, dim] last-layer hidden states
        """
        # Qwen3Model applies the final norm — equivalent to HF hidden_states[-1].
        return self.model.model(ids)
