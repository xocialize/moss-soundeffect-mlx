#!/usr/bin/env python
"""Bundle the Stage-0 golden .npy fixtures into one safetensors file for the
Swift test suite (MLX.loadArrays reads safetensors; it does not read .npy).

Usage: .venv/bin/python scripts/export_swift_fixtures.py
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

mx.set_default_device(mx.cpu)

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

NAMES = [
    "golden_context",
    "golden_context_nega",
    "golden_noise",
    "golden_final_latent",
    "golden_audio",
    "golden_velocity_t1000",
    "golden_velocity_t500",
]


def add_token_fixtures(out):
    """Tokenize the golden prompt so Swift tests don't need a tokenizer."""
    import transformers

    meta = json.loads((FIXTURES / "golden_meta.json").read_text())
    full_prompt = f"{meta['prompt']} duration: {meta['seconds']:.1f}s"
    tok = transformers.AutoTokenizer.from_pretrained(
        "/Volumes/DEV_ARCHIVE/weights/MOSS-SoundEffect-v2.0/tokenizer")
    enc = tok([full_prompt], return_tensors="np", padding="max_length",
              truncation=True, max_length=512, add_special_tokens=True)
    out["golden_ids"] = mx.array(enc.input_ids.astype("int32"))
    out["golden_mask"] = mx.array(enc.attention_mask.astype("int32"))


def main():
    out = {}
    for name in NAMES:
        arr = np.load(FIXTURES / f"{name}.npy")
        out[name] = mx.array(arr)
    add_token_fixtures(out)
    meta = json.loads((FIXTURES / "golden_meta.json").read_text())
    mx.eval(list(out.values()))  # lazy arrays save as zeros — evaluate first
    dest = FIXTURES / "swift_goldens.safetensors"
    mx.save_safetensors(
        str(dest), out,
        metadata={k: str(v) for k, v in meta.items()},
    )
    loaded = mx.load(str(dest))
    # golden_context_nega is legitimately all-zero: empty negative prompt ->
    # 0 valid tokens (Qwen3 adds no BOS) -> prompter zeroes every position.
    assert all(
        float(mx.abs(loaded[n]).max()) > 0
        for n in NAMES if n != "golden_context_nega"
    )
    print(f"wrote {dest} ({len(out)} tensors)")


if __name__ == "__main__":
    main()
