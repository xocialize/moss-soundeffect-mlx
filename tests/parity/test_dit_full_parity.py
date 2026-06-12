"""Full-DiT parity at PRODUCTION scale with the real checkpoint.

Stage-2 gate 3 (handoff §4): velocity field at a fixed timestep,
max_abs < 1e-2 fp32. Conditioning uses real Qwen3 embeddings (cached as a
fixture on first run) — random-context parity can miss bugs that only show
at trained magnitudes. ~14 GB transient (both DiTs fp32), CPU-pinned.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

UPSTREAM = Path("/Volumes/DEV_ARCHIVE/_ref/MOSS-TTS/moss_soundeffect_v2")
WEIGHTS = Path("/Volumes/DEV_ARCHIVE/weights/MOSS-SoundEffect-v2.0")
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
sys.path.insert(0, str(UPSTREAM))

import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx.utils import tree_unflatten  # noqa: E402

from moss_sfx_mlx.config import WanAudioModelConfig  # noqa: E402
from moss_sfx_mlx.model.wan_audio_dit import WanAudioModel as MXWanAudioModel  # noqa: E402
from moss_sfx_mlx.utils.convert import dit_hf_to_mlx  # noqa: E402

from ._helpers import assert_parity, make_seeded_input  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (WEIGHTS / "transformer").exists(), reason="weights not downloaded"
)

PROMPT = "a heavy wooden door creaks open slowly duration: 10.0s"
T_LAT = 1500  # 30 s x 50 Hz — production latent length


def get_context_fixture() -> np.ndarray:
    """Real Qwen3 embeddings for PROMPT, padded to 512 + zeroed tail (fp32)."""
    path = FIXTURES / "qwen3_context_door_creak.npy"
    if path.exists():
        return np.load(path)
    import transformers

    from moss_sfx_mlx.model.qwen3_text_encoder import Qwen3TextEncoder

    tok = transformers.AutoTokenizer.from_pretrained(str(WEIGHTS / "tokenizer"))
    enc = tok([PROMPT], return_tensors="np", padding="max_length",
              truncation=True, max_length=512, add_special_tokens=True)
    encoder = Qwen3TextEncoder(WEIGHTS / "text_encoder", dtype=mx.float32)
    emb = np.array(encoder(mx.array(enc.input_ids))).astype("float32")
    emb[0, int(enc.attention_mask.sum()):] = 0
    FIXTURES.mkdir(exist_ok=True)
    np.save(path, emb)
    return emb


def test_dit_full_parity_real_weights():
    from safetensors.torch import load_file

    from diffsynth.models.wan_audio_dit import WanAudioModel as PTWanAudioModel
    from diffsynth.pipelines.wan_audio import _convert_hf_dit_state_dict

    context = get_context_fixture()

    cfg = WanAudioModelConfig()
    cfg_kwargs = dict(
        dim=cfg.dim, in_dim=cfg.in_dim, ffn_dim=cfg.ffn_dim, out_dim=cfg.out_dim,
        text_dim=cfg.text_dim, freq_dim=cfg.freq_dim, eps=cfg.eps,
        patch_size=cfg.patch_size, num_heads=cfg.num_heads,
        num_layers=cfg.num_layers, has_image_input=cfg.has_image_input,
        vae_type=cfg.vae_type,
    )

    hf_sd = load_file(str(WEIGHTS / "transformer" / "diffusion_pytorch_model.safetensors"))

    pt_model = PTWanAudioModel(**cfg_kwargs).eval()
    load_result = pt_model.load_state_dict(_convert_hf_dit_state_dict(hf_sd))
    assert not load_result.missing_keys and not load_result.unexpected_keys

    mx_model = MXWanAudioModel(**cfg_kwargs)
    converted = dit_hf_to_mlx(hf_sd)
    del hf_sd
    mx_model.update(tree_unflatten([(k, mx.array(v)) for k, v in converted.items()]))
    del converted
    mx.eval(mx_model.parameters())

    x = make_seeded_input((1, 128, T_LAT), seed=42)
    t = np.array([1000.0], dtype="float32")  # first denoise step (sigma=1.0)

    with torch.no_grad():
        pt_out = pt_model(
            torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(context)
        )
    mx_out = mx_model(mx.array(x), mx.array(t), mx.array(context))
    assert pt_out.shape == (1, 128, T_LAT)
    assert_parity(pt_out, mx_out, threshold=1e-2, name="dit_full_t1000")
