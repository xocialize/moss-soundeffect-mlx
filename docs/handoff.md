# MOSS-SoundEffect-v2.0 → MLX-Swift Port — Coder Handoff

**Target repo:** `github.com/xocialize/moss-soundeffect-mlx` (new, standalone)
**Tier:** 3 (multi-component pipeline — text encoder + DiT + VAE + scheduler)
**Effort:** Medium, ~2–4 focused weeks
**Conformance:** C0–C13 deferred. Build standalone; conformance wrap is a later workstream. (Parity gates below are the *technical* acceptance bar regardless.)
**License posture:** GO — all components Apache-2.0 / MIT (see §1.4). Publish port under Apache-2.0.
**Use case:** anime-studio SFX generation (environmental/foley/creature/action audio from text captions, 48 kHz, ≤30 s clips).

---

## 0. Orientation — read before writing code

This is a **transpose, not a redesign**. The PyTorch reference is the oracle; every config value, schedule, and op order is matched exactly. Deviations ("cleaner defaults", "optimized schedule") are how silent audio-gen bugs get in. Preserve isomorphic structure with upstream: same file names, same class names, same forward-pass call order, so `model.py` (upstream) diffs 1:1 against `model_mlx.py` (port) showing only PyTorch↔MLX op substitutions.

**Two-language structure.** Python `-mlx` package (the parity oracle + weight conversion) and the Swift package (the deliverable that loads into MLXEngine) live in one repo. The Python side exists *to validate the Swift side*: you cannot parity-test Swift against PyTorch directly, so the chain is **PyTorch ref → MLX-Python (parity-locked) → MLX-Swift (validated against the MLX-Python golden outputs)**. Skipping the Python middle layer means debugging Swift numerics blind. Don't.

### Reference sources (the oracle)
- Model card / weights: `https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0`
- Upstream code: `https://github.com/OpenMOSS/MOSS-TTS/tree/main/moss_soundeffect_v2`
- Pipeline class to match: `MossSoundEffectPipeline` (diffusers 0.32.0)
- DiT backbone class: `WanAudioModel` (Wan2.1 DiT lineage)
- Scheduler: `FlowMatchScheduler` (shift=5.0, sigma_min=0.0, extra_one_step=true, num_train_timesteps=1000)

### Swift donors (study before writing — do not reinvent)
- **DiT + flow-matching + vocoder, end-to-end in MLX-Swift:** `github.com/lucasnewman/f5-tts-swift` (MIT). Closest structural analog; the DiT block, RoPE, AdaLN time-conditioning, and the Euler flow-matching loop are all here.
- **DACVAE codec for MLX-Swift:** `github.com/Blaizzy/mlx-audio-swift` → `MLXAudioCodecs` module (README lists SNAC, Encodec, Vocos, Mimi, **DACVAE**). Evaluate this *first* for the decoder before porting from scratch.
- **Qwen3 text encoder:** `mlx-swift-examples` (Qwen3 already implemented). Reuse the transformer, extract per-token hidden states (do **not** use the LM head).

### Skill + helpers
- The `mlx-porting` skill (`/mnt/skills/user/mlx-porting/SKILL.md`) governs this port. Key reference files: `parity-testing.md`, `repo-layout.md`, `weight-conversion.md`, `common-pitfalls.md`.
- Copy `scripts/parity_helpers.py` from the skill into `tests/parity/_helpers.py` — provides `make_seeded_input`, `pt_to_mx`, `mx_to_np`, `transpose_pt_conv`, `load_pt_state_into_mx`, `assert_parity`, `tensor_stats`, `detect_checkerboard`, `noise_decode_check`. Don't rewrite these.

---

## 1. The model (verified from repo config)

### 1.1 Task
Text caption → non-speech audio (foley, ambience, creature, action, short percussive/musical). Bilingual EN/ZH prompts. 48 kHz output. Duration controllable via `seconds` (≤30). No audio-to-audio / no reference-audio path. Only continuous control is duration.

### 1.2 Components & weights
| Component | Class | Shape / config | Weight file | Size |
|---|---|---|---|---|
| **DiT backbone** | `WanAudioModel` | `num_layers=30`, `dim=1536`, `num_heads=12` (head_dim 128), `ffn_dim=8960`, `in_dim=out_dim=128`, `text_dim=2048`, `patch_size=[1]`, `eps=1e-6`. ~1.3B params, flow-matching objective. | `diffusion_pytorch_model.safetensors` | 5.66 GB |
| **Audio codec** | `DAC` (continuous-latent VAE) | `latent_dim=128`, `sample_rate=48000`. Continuous latent (NOT discrete RVQ). | `vae_128d_48k.pth` (pickled full object) | 1.49 GB |
| **Text encoder** | `Qwen3ForCausalLM` (frozen) | hidden_size=2048, 28 layers, 16 heads / 8 KV (GQA), vocab 151936, base `Qwen/Qwen3-1.7B-Base`. 2048-d hidden states feed DiT cross-attn (= `text_dim`). | 2× safetensors shards | ~4.06 GB |
| **Scheduler** | `FlowMatchScheduler` | shift=5.0, sigma_min=0.0, extra_one_step=true, num_train_timesteps=1000 | (config) | — |

Total repo ~11.2 GB.

### 1.3 Inference call (correct usage — the HF auto-snippet on the model page is wrong boilerplate)
```python
pipe = MossSoundEffectPipeline.from_pretrained(
    "OpenMOSS-Team/MOSS-SoundEffect-v2.0", torch_dtype=torch.bfloat16, device="cuda")
audio = pipe(prompt=..., seconds=10, num_inference_steps=100, cfg_scale=4.0)
```
Sampling defaults: `num_inference_steps=100`, `cfg_scale=4.0`, `sigma_shift=5.0`. Output waveform `(B, C, T)`.

> Upstream wraps the DiT in `torch.compile` + Triton CUDA Graph. Irrelevant to MLX, but when you run the **reference** to capture golden tensors, set `TORCHDYNAMO_DISABLE=1` to avoid compile errors on the capture box.

### 1.4 Licensing — GO (all permissive, no gate)
- MOSS-SoundEffect-v2.0 weights: **Apache-2.0** (model card + MOSS-TTS README).
- MOSS-TTS code: **Apache-2.0**.
- DAC codec lineage (Descript Audio Codec): **MIT**. MOSS DAC-VAE is a retrained continuous variant; covered by the family Apache-2.0 statement.
- Qwen3-1.7B text encoder: **Apache-2.0**.
- **No commercial gate, no non-commercial/research-only clause.** Publish the port Apache-2.0.

---

## 2. Existing-port reality (why this is a from-scratch port)

- `mlx-community/MOSS-SoundEffect-MLX-4bit` exists but is a **mirror of the v1 8B autoregressive RVQ model**, MLX-**Python**, weights-only. **Does not transfer** to v2's continuous-latent DiT. Ignore it except as a curiosity.
- `Blaizzy/mlx-audio` supports several MOSS TTS models but **not** MOSS-SoundEffect (v1 or v2). Issue #536 requests SFX/music-gen generally; not implemented.
- **No MLX-Swift port of v2.0 exists.** This is greenfield.

---

## 3. Repo layout (Tier-3 monorepo + Swift mirror)

```
moss-soundeffect-mlx/
├── README.md
├── LICENSE                          # Apache-2.0
├── pyproject.toml                   # mlx, mlx-arsenal, huggingface-hub, safetensors; [parity]=torch,diffusers
├── moss_sfx_mlx/                    # ── PYTHON parity oracle ──
│   ├── __init__.py
│   ├── pipeline_mlx.py              # MossSoundEffectPipeline analog, from_pretrained()
│   ├── model/
│   │   ├── wan_audio_model.py       # WanAudioModel (DiT) — match upstream class/method names
│   │   ├── attention.py             # self-attn + cross-attn blocks
│   │   ├── dac_vae.py               # continuous DAC-VAE decoder
│   │   └── text_encoder.py          # Qwen3 hidden-state extractor
│   ├── config.py                    # dataclasses mirroring source config.json verbatim
│   ├── scheduler.py                 # FlowMatchScheduler (shift=5.0, extra_one_step=true)
│   └── utils/weights.py             # split-safetensors load, HF download, MOSS_SFX_MLX_WEIGHTS_DIR override
├── tests/
│   ├── parity/                      # PT↔MLX-Python, torch = optional dep
│   │   ├── _helpers.py              # copied from skill scripts/parity_helpers.py
│   │   ├── test_text_encoder_parity.py
│   │   ├── test_dit_block_parity.py
│   │   ├── test_dit_full_parity.py
│   │   ├── test_vae_decode_parity.py
│   │   └── test_pipeline_parity.py
│   ├── smoke/                       # shapes/config/e2e no-numeric
│   └── fixtures/                    # golden npy captured from PyTorch ref
└── swift/                           # ── SWIFT deliverable (MLXEngine consumer) ──
    ├── MossSoundEffect.xcworkspace  # Xcode is the build tool (NOT swift build/test CLI)
    ├── Package.swift                # dependency manifest only
    └── Sources/MossSoundEffectMLX/
        ├── WanAudioModel.swift
        ├── Attention.swift
        ├── DACVAEDecoder.swift
        ├── Qwen3TextEncoder.swift
        ├── FlowMatchScheduler.swift
        └── Pipeline.swift
```

**Weight hosting:** multi-component pipeline → weights go to `xocialize` HF (or `xocialize-code`) directly, NOT mlx-community (no clean single-`model_type` slot). Swift side loads from that repo. `MOSS_SFX_MLX_WEIGHTS_DIR` env var overrides for local dev.

**Swift build:** `xcodebuild` against the workspace. SwiftPM CLI is not used for anything touching MLX/Metal. (If a non-GPU test lane needs `swift test`, see the metallib-rename escape hatch in skill `repo-layout.md` → "SPM CLI cannot compile Metal shaders".)

---

## 4. Staged plan

### Stage 0 — Capture golden tensors from the PyTorch reference (0.5–1 day)
On a CUDA (or CPU) box with the upstream repo installed, `TORCHDYNAMO_DISABLE=1`, fixed seed. Capture and commit to `tests/fixtures/` as `.npy`:
1. A golden prompt → Qwen3 **hidden states** `(1, L, 2048)` (last hidden layer; confirm which layer upstream feeds to cross-attn — likely final, verify).
2. A fixed initial noise latent `(1, 128, T)` for a known `seconds` (compute T from the VAE frame rate — see §5 open item).
3. DiT **velocity output** at a fixed timestep given (1)+(2).
4. Final latent after the full denoising loop.
5. VAE **decode** of (4) → waveform `(1, C, T_audio)`.
6. The final audio as `.wav` for perceptual A/B.

These bypass the `mx.random` vs `torch` RNG incompatibility — every downstream parity test injects these exact bytes on both sides.

### Stage 1 — Weight conversion (1–2 days)
Tier-3 = `mlx-forge` recipe (or a hand-written convert script following the recipe pattern). Split safetensors per component (`dit`, `vae`, `text_encoder`) so each loads/quantizes independently.

**THE conversion trap (Tier-3 only):** MLX is lazy — **unevaluated tensors serialize as zeros with no error.** Call `mx.eval(weight)` (or the `_materialize` helper) immediately before every `mx.save_safetensors`. `mlx_lm.convert` does this internally; your custom Tier-3 script does NOT get it for free.

Specifics:
- **DAC-VAE is a pickled full `.pth` object**, not a state_dict. Load it, extract `.state_dict()`, re-serialize clean. (Decoder-only is needed for T2A — but convert the whole thing, prune later.)
- Conv weights: PyTorch `(O, I, *K)` → MLX `(O, *K, I)`. Handle in the recipe `transform` step (`mlx_forge.transpose.transpose_conv` is generic). The DAC decoder is conv-heavy — this matters.
- Quantize **only** transformer `Linear.weight` in the DiT. Keep VAE and text-encoder at bf16. Lock fp16/bf16 parity **before** any int4.
- bf16 default for the DiT and VAE; capture fp32 reference for parity (parity tests run fp32 to isolate framework drift).

### Stage 2 — MLX-Python port + parity lock (1–2 weeks)
Build the Python `-mlx` modules and parity-test each against the Stage-0 goldens. **Do not advance a stage until its parity gate is green.** Build order (cheapest/most-certain first):

1. **Qwen3 text encoder** (lowest risk). Port/reuse from mlx-swift-examples' Qwen3. Gate: hidden-state cosine ≈ 1.0, `max_abs < 1e-4` fp32 vs golden (1).
2. **DiT block** (one layer). Self-attn + RoPE + cross-attn + AdaLN + FFN + residual. Gate: `max_abs < 1e-3` fp32.
3. **Full DiT** (30 layers). Velocity field at fixed timestep. Gate: `max_abs < 1e-2` fp32 vs golden (3).
4. **FlowMatchScheduler + denoising loop**. Inject golden noise (2), run 100 steps, CFG=4.0, shift=5.0. Gate: final latent matches golden (4), `max_abs < 1e-2`.
5. **DAC-VAE decoder** (critical path — see §5). Inject golden latent (4), decode. Gate: waveform `max_abs < 1e-2` fp32 vs golden (5), **and** `noise_decode_check` shows no periodic tonal pattern.
6. **End-to-end**. Full pipe on the golden prompt+seed. Gate: PSNR/spectral match vs golden (6) + perceptual listen (no tonal tails, correct duration).

Parity harness pattern (fp32, torch as optional dep, `pytest.importorskip`):
```python
torch = pytest.importorskip("torch")
x = make_seeded_input(shape, seed=42)          # numpy, injected both sides
pt_out = pt_module(torch.from_numpy(x)).detach().numpy()
mx_out = mx_to_np(mx_module(mx.array(x)))
assert_parity(pt_out, mx_out, max_abs=1e-4)
```

### Stage 3 — MLX-Swift port (1 week, overlaps Stage 2)
Translate the parity-locked Python modules to Swift, validating each against the **MLX-Python golden outputs** (not PyTorch directly). Lean on f5-tts-swift for DiT/scheduler and mlx-audio-swift's DACVAE for the decoder.

Swift consumer idioms (skill `repo-layout.md` → "MLX-Swift consumer idioms"):
- **Weight load three-step:** `MLX.loadArrays(url:)` → `ModuleParameters.unflattened(loaded)` → `model.update(parameters:, verify: .noUnusedKeys)`. The `.noUnusedKeys` is what turns a remap bug into a boot-time failure instead of a silently zero-initialized layer.
- **Dotted-key remap:** safetensors `block.0.weight` ↔ Swift property `block_0` (Swift props can't contain dots). Remap at load before `unflattened` (regex in the skill). Without it `.noUnusedKeys` either fails loudly (good) or no-ops a layer (bad).
- **GPU-state classes `@unchecked Sendable` + `NSLock`** around mutable `MLXArray`/kernel-handle state. MLX-Swift arrays aren't `Sendable`-marked upstream.
- **`Float64` → GPU crash.** Keep everything `Float32`/`Float16`/`BFloat16`. Watch `Float * MLXArray` ambiguity.
- Symptom "loads fine, `.noUnusedKeys` passes, but output is garbage *only on Swift*" → it's one of the three above before it's a layer-translation bug.

### Stage 4 — Quantize + publish (2–3 days)
- int4 DiT Linears only (`nn.quantize(group_size=64, bits=4)`), `group_size=128, bits=8` fallback. Re-run parity at relaxed int4 thresholds (`max_abs < 5e-2` vs fp16 ref).
- Publish weights to `xocialize` HF (Apache-2.0). README per skill convention: honest Features (what works / PSNR), Requirements (Apple Silicon, macOS, RAM), Quick Start, Performance table (M-series peak mem + wallclock), Known Limitations, Citation, License.

---

## 5. Silent-failure surface (audio-gen fails silently — read this twice)

Ranked by risk. Layer parity passing does **not** mean end-to-end is correct; add the noise-path smoke test.

1. **DAC-VAE latent scaling/normalization (HIGHEST RISK).** Continuous VAE latents are typically multiplied by a fixed scale constant before decode. Wrong/missing constant → plausible-looking but garbage audio, no error thrown. Find the exact constant in the upstream VAE wrapper; verify decode parity from the golden latent before trusting anything.
2. **Conv channel/layout mismatch in the DAC decoder.** PyTorch `(B,C,T)` conv1d vs MLX channel-last. Mishandled → noise. The decoder is conv-heavy; transpose every conv weight in the recipe.
3. **Flow-match sigma-shift schedule mismatch.** `shift=5.0` + `extra_one_step=true` is an off-by-one trap. The sigma schedule must match upstream exactly; an off-by-one yields subtly-wrong-then-divergent output.
4. **CFG batch-doubling order.** cond/uncond concatenation order must match upstream; reversed → inverted guidance.
5. **Text cross-attention masking/padding.** Padded prompt tokens leaking into cross-attn → conditioning drift.
6. **DAC decoder snake activation + weight-norm reconstruction.** Snake (`x + sin²(αx)/α`) and weight-normalized convs must be reconstructed exactly; wrong α handling → tonal artifacts.
7. **Checkerboard/tonal periodicity.** If output has periodic tonal regions at a stride: suspect `mx.tile` vs `mx.repeat`, upsample axis order, or scheduler fp32 leaking into a bf16 DiT. Run `detect_checkerboard` / `noise_decode_check` from the helpers before shipping.

**bf16 caution for narrow nets:** if any sub-net is small (<~100M) and bf16 parity drops more than a few dB vs fp16, ship the fp16 preset for that component and document it. (Qwen3 + 1.3B DiT are large enough to be fine; the VAE decoder is the one to spot-check.)

---

## 6. Open items to resolve from upstream source (do these in Stage 0)

These were inferred from class lineage + minimal config during assessment and must be **confirmed against `moss_soundeffect_v2/` modeling code** before finalizing:

1. **DiT internals.** RoPE flavor (traditional vs default), presence of QK-norm (RMSNorm on Q/K), AdaLN variant (additive vs `x*(1+scale)+shift`), FFN activation (SwiGLU vs GEGLU vs GELU), pre-norm vs post-norm. Inferred from Wan2.1 lineage — verify each against the actual `WanAudioModel` forward.
2. **DAC-VAE latent frame rate (Hz).** Not in any config. Needed to compute latent length T from `seconds`. Read from the VAE object or derive from the decoder stride stack.
3. **Which Qwen3 hidden layer** feeds cross-attn (final vs penultimate). Capture the exact layer in Stage 0 golden (1).
4. **VAE latent scale constant** (see §5.1). Find it explicitly.
5. **DAC-VAE weight provenance** — one-line confirmation the retrained continuous VAE weights are covered by the family Apache-2.0 (they are per the blanket statement; just close the loop).

When in doubt, match the reference exactly and ask before deviating "to fix" an artifact — deviations almost always hide bugs.

---

## 7. Acceptance criteria (definition of done)

- [ ] All Stage-2 parity gates green (text-encoder, DiT block, full DiT, scheduler, VAE decode, e2e) at the fp32 thresholds in §4.
- [ ] `noise_decode_check` clean (no periodic tonal pattern) at production config.
- [ ] Swift output matches MLX-Python golden outputs (per-module + e2e).
- [ ] End-to-end perceptual A/B vs PyTorch on ≥10 prompts at fixed seed — correct content, correct duration (≤ `seconds`), no tonal tails.
- [ ] int4 DiT parity within `max_abs < 5e-2` vs fp16; fp16 fallback documented if needed.
- [ ] Weights published Apache-2.0 to `xocialize` HF; README per skill convention with M-series perf table.
- [ ] Swift package builds via `xcodebuild`, loads with `verify: .noUnusedKeys` passing, runs in MLXEngine harness.
- [ ] (Deferred) C0–C13 conformance wrap — separate workstream; per-module fixtures from this port feed directly into the conformance parity items.

---

## 8. One-paragraph summary for the queue card

Greenfield Tier-3 MLX-Swift port of MOSS-SoundEffect-v2.0 (text→SFX, 48 kHz, ≤30 s) into `xocialize/moss-soundeffect-mlx`, for anime-studio foley/ambience generation. Four components — Qwen3-1.7B text encoder, 1.3B Wan-style DiT (flow-matching), continuous 128-d DAC-VAE decoder, FlowMatchScheduler — all Apache-2.0/MIT (clean GO). No usable existing port (the mlx-community artifact is the unrelated v1 RVQ model). Every component has a Swift donor (f5-tts-swift for DiT+scheduler, mlx-audio-swift for DACVAE, mlx-swift-examples for Qwen3), so effort is Medium (~2–4 wk). Build chain is PyTorch ref → parity-locked MLX-Python → validated MLX-Swift; the DAC-VAE decoder (latent scaling + conv layout) is the critical silent-failure path. Conformance (C0–C13) deferred to a follow-on. Five upstream-source confirmations (DiT internals, VAE frame rate, cross-attn hidden layer, latent scale constant, VAE weight provenance) to close in Stage 0.
