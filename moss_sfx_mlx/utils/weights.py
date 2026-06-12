"""Weight locating / loading helpers.

Resolution order for the weights directory:
  1. explicit `weights_dir` argument
  2. MOSS_SFX_MLX_WEIGHTS_DIR environment variable
  3. huggingface_hub snapshot download (converted-MLX repo, Stage 4)

For local dev on this machine the source checkpoint lives at
/Volumes/DEV_ARCHIVE/weights/MOSS-SoundEffect-v2.0 (upstream PyTorch layout)
and converted MLX weights will live alongside it after Stage 1.
"""

import os
from pathlib import Path

import mlx.core as mx

ENV_VAR = "MOSS_SFX_MLX_WEIGHTS_DIR"
# Stage-4 publish targets: mlx-community, weight-style naming (preserve
# upstream case + official quant suffix):
#   mlx-community/MOSS-SoundEffect-v2.0-bf16  (DiT bf16, VAE fp32, Qwen3 bf16)
#   mlx-community/MOSS-SoundEffect-v2.0-4bit  (DiT int4 g64 blocks-Linear only)
DEFAULT_REPO_ID = "mlx-community/MOSS-SoundEffect-v2.0-bf16"


def resolve_weights_dir(weights_dir=None, repo_id: str = DEFAULT_REPO_ID) -> Path:
    if weights_dir is not None:
        return Path(weights_dir)
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id))


def load_split_safetensors(directory, prefix: str) -> dict:
    """Load all `{prefix}*.safetensors` shards in `directory` into one dict."""
    directory = Path(directory)
    shards = sorted(directory.glob(f"{prefix}*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no '{prefix}*.safetensors' in {directory}")
    weights = {}
    for shard in shards:
        weights.update(mx.load(str(shard)))
    return weights
