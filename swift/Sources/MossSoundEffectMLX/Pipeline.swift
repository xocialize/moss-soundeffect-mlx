// Inference pipeline — Swift transpose of moss_sfx_mlx/pipeline_mlx.py
// (parity-locked oracle). Faithful behaviors:
//   * CFG: two separate forwards, fp32 combine nega + cfg * (posi - nega)
//   * fixed full-length latent; waveform cropped by the caller
//   * VAE decode fp32
//   * empty negative prompt == all-zero context (Qwen3 emits no BOS for "")
//
// Text encoding is behind a protocol until the Qwen3 Swift wrapper lands;
// parity tests inject golden contexts directly.

import Foundation
import MLX
import MLXNN

public enum WeightLoading {
    /// diffusers-style DiT checkpoint key -> our module key.
    /// Mirrors moss_sfx_mlx/utils/convert.py rename_dit_key.
    static let globalRename: [(String, String)] = [
        ("condition_embedder.text_embedder.linear_1", "text_embedding.layers.0"),
        ("condition_embedder.text_embedder.linear_2", "text_embedding.layers.2"),
        ("condition_embedder.time_embedder.linear_1", "time_embedding.layers.0"),
        ("condition_embedder.time_embedder.linear_2", "time_embedding.layers.2"),
        ("condition_embedder.time_proj", "time_projection.layers.1"),
        ("proj_out", "head.head"),
    ]

    static let blockRename: [(String, String)] = [
        ("attn1.norm_q", "self_attn.norm_q"),
        ("attn1.norm_k", "self_attn.norm_k"),
        ("attn1.to_q", "self_attn.q"),
        ("attn1.to_k", "self_attn.k"),
        ("attn1.to_v", "self_attn.v"),
        ("attn1.to_out.0", "self_attn.o"),
        ("attn2.norm_q", "cross_attn.norm_q"),
        ("attn2.norm_k", "cross_attn.norm_k"),
        ("attn2.to_q", "cross_attn.q"),
        ("attn2.to_k", "cross_attn.k"),
        ("attn2.to_v", "cross_attn.v"),
        ("attn2.to_out.0", "cross_attn.o"),
        ("ffn.net.0.proj", "ffn.layers.0"),
        ("ffn.net.2", "ffn.layers.2"),
        ("norm2", "norm3"),
    ]

    public static func renameDiTKey(_ key: String) -> String {
        if key == "scale_shift_table" { return "head.modulation" }
        for (old, new) in globalRename where key.hasPrefix(old + ".") {
            return new + key.dropFirst(old.count)
        }
        if key.hasPrefix("blocks.") {
            let parts = key.split(separator: ".", maxSplits: 2).map(String.init)
            if parts.count == 3 {
                let (idx, suffix) = (parts[1], parts[2])
                if suffix == "scale_shift_table" { return "blocks.\(idx).modulation" }
                for (old, new) in blockRename where suffix.hasPrefix(old + ".") {
                    return "blocks.\(idx).\(new)\(suffix.dropFirst(old.count))"
                }
            }
        }
        return key
    }

    /// Load the ORIGINAL fp32 diffusers checkpoint (parity master) into a DiT.
    public static func loadDiTFromOriginal(_ model: WanAudioModel, url: URL) throws {
        let raw = try loadArrays(url: url)
        var weights = [String: MLXArray]()
        for (k, v) in raw {
            var arr = v
            if k == "patch_embedding.weight" {
                arr = arr.transposed(0, 2, 1)  // Conv1d (O, I, K) -> (O, K, I)
            }
            weights[renameDiTKey(k)] = arr
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .noUnusedKeys)
        eval(model)
    }

    /// Load converted safetensors (already in MLX layout + our key names).
    public static func loadConverted(_ model: Module, url: URL, dtype: DType? = nil) throws {
        var weights = try loadArrays(url: url)
        if let dtype {
            weights = weights.mapValues { $0.asType(dtype) }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .noUnusedKeys)
        eval(model)
    }
}

public final class MossSoundEffectPipeline {
    public let dit: WanAudioModel
    public let vae: DAC
    public let scheduler: FlowMatchScheduler
    public let sampleRate = 48000
    public let maxInferenceSeconds = 30
    /// Tokenization ships in-package (swift-transformers) — MLXEngine has no
    /// internal tokenizer. Optional so parity tests can inject golden contexts.
    public var prompter: WanPrompter?

    public init(dit: WanAudioModel, vae: DAC, prompter: WanPrompter? = nil) {
        self.dit = dit
        self.vae = vae
        self.prompter = prompter
        self.scheduler = FlowMatchScheduler(shift: 5.0, sigmaMin: 0.0, extraOneStep: true)
    }

    /// Full text -> audio path, mirroring the Python facade: appends the
    /// trained " duration: X.Xs" suffix, denoises the FULL 30 s latent, and
    /// crops the waveform to `seconds`.
    public func generate(
        prompt: String,
        seconds: Double = 10.0,
        negativePrompt: String = "",
        numInferenceSteps: Int = 100,
        cfgScale: Float = 4.0,
        sigmaShift: Float = 5.0,
        seed: UInt64 = 0,
        appendDurationSuffix: Bool = true
    ) throws -> MLXArray {
        guard let prompter else {
            throw NSError(
                domain: "MossSoundEffectMLX", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "pipeline has no prompter — construct with WanPrompter"])
        }
        let secondsRounded = (seconds * 10).rounded() / 10
        precondition(secondsRounded > 0 && secondsRounded <= Double(maxInferenceSeconds))
        let fullPrompt = appendDurationSuffix
            ? "\(prompt.trimmingCharacters(in: .whitespacesAndNewlines)) duration: \(String(format: "%.1f", secondsRounded))s"
            : prompt
        let context = prompter.encodePrompt([fullPrompt])
        let contextNega = prompter.encodePrompt([negativePrompt])

        let latentLength = sampleRate * maxInferenceSeconds / vae.config.hopLength
        MLXRandom.seed(seed)  // NB: not torch/python-RNG compatible
        let noise = MLXRandom.normal([1, vae.config.latentDim, latentLength])

        let audio = generate(
            context: context, contextNega: contextNega, noise: noise,
            numInferenceSteps: numInferenceSteps, cfgScale: cfgScale, sigmaShift: sigmaShift)
        let outputSamples = Int(Double(sampleRate) * secondsRounded)
        return audio[0..., 0..., 0 ..< outputSamples]
    }

    /// Denoise injected noise under injected (already-encoded) contexts and
    /// decode. Mirrors the oracle WanAudioPipeline.__call__ T2A path.
    public func generate(
        context: MLXArray,
        contextNega: MLXArray,
        noise: MLXArray,
        numInferenceSteps: Int = 100,
        cfgScale: Float = 4.0,
        sigmaShift: Float = 5.0
    ) -> MLXArray {
        scheduler.setTimesteps(numInferenceSteps, shift: sigmaShift)

        var latents = noise
        for i in 0 ..< scheduler.timesteps.count {
            let t = scheduler.timesteps[i]
            let timestep = MLXArray([t])

            let vPosi = dit(latents, timestep: timestep, context: context)
            var v = vPosi
            if cfgScale != 1.0 {
                let vNega = dit(latents, timestep: timestep, context: contextNega)
                // fp32 combine, exactly like upstream.
                v = vNega.asType(.float32) + cfgScale * (vPosi.asType(.float32) - vNega.asType(.float32))
            }
            latents = scheduler.step(v, timestep: t, sample: latents.asType(.float32))
                .asType(noise.dtype)
            eval(latents)  // keep command buffers bounded
        }

        // Decode at fp32 (upstream decodes under fp32 autocast).
        let audio = vae.decode(latents.asType(.float32))
        eval(audio)
        return audio
    }
}
