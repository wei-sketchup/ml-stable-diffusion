// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Accelerate
import CoreGraphics
import CoreML
import Foundation
import NaturalLanguage

/// Schedulers compatible with StableDiffusionPipeline
public enum StableDiffusionScheduler {
    /// Scheduler that uses a pseudo-linear multi-step (PLMS) method
    case pndmScheduler
    /// Scheduler that uses a second order DPM-Solver++ algorithm
    case dpmSolverMultistepScheduler
}

/// RNG compatible with StableDiffusionPipeline
public enum StableDiffusionRNG {
    /// RNG that matches numpy implementation
    case numpyRNG
    /// RNG that matches PyTorch CPU implementation.
    case torchRNG
    /// RNG that matches PyTorch CUDA implementation.
    case nvidiaRNG
    /// Use MLTensor.
    case coreml
}

public enum PipelineError: String, Swift.Error {
    case missingUnetInputs
    case startingImageProvidedWithoutEncoder
    case startingText2ImgWithoutTextEncoder
    case unsupportedOSVersion
}

public protocol StableDiffusionPipelineProtocol: ResourceManaging {
    var canSafetyCheck: Bool { get }

    func generateImages(
        configuration config: PipelineConfiguration,
        progressHandler: (PipelineProgress) -> Bool
    ) async throws -> [CGImage?]

    func decodeToImages(
        _ latents: [MLTensor],
        configuration config: PipelineConfiguration
    ) async throws -> [CGImage?]
}

public extension StableDiffusionPipelineProtocol {
    var canSafetyCheck: Bool { false }
}

/// A pipeline used to generate image samples from text input using stable diffusion
///
/// This implementation matches:
/// [Hugging Face Diffusers Pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)
public struct StableDiffusionPipeline: StableDiffusionPipelineProtocol {
    /// Model to generate embeddings for tokenized input text
    var textEncoder: TextEncoderModel

    /// Model used to predict noise residuals given an input, diffusion time step, and conditional embedding
    var unet: Unet

    /// Model used to generate final image from latent diffusion process
    var decoder: Decoder

    /// Model used to latent space for image2image, and soon, in-painting
    var encoder: Encoder?

    /// Optional model for checking safety of generated image
    var safetyChecker: SafetyChecker? = nil

    /// Optional model used before Unet to control generated images by additonal inputs
    var controlNet: ControlNet? = nil

    /// Reports whether this pipeline can perform safety checks
    public var canSafetyCheck: Bool {
        safetyChecker != nil
    }

    /// Option to reduce memory during image generation
    ///
    /// If true, the pipeline will lazily load TextEncoder, Unet, Decoder, and SafetyChecker
    /// when needed and aggressively unload their resources after
    ///
    /// This will increase latency in favor of reducing memory
    var reduceMemory: Bool = false

    /// Option to use system multilingual NLContextualEmbedding as encoder
    var useMultilingualTextEncoder: Bool = false

    /// Optional natural language script to use for the text encoder.
    var script: Script? = nil

    /// Creates a pipeline using the specified models and tokenizer
    ///
    /// - Parameters:
    ///   - textEncoder: Model for encoding tokenized text
    ///   - unet: Model for noise prediction on latent samples
    ///   - decoder: Model for decoding latent sample to image
    ///   - controlNet: Optional model to control generated images by additonal inputs
    ///   - safetyChecker: Optional model for checking safety of generated images
    ///   - reduceMemory: Option to enable reduced memory mode
    /// - Returns: Pipeline ready for image generation
    public init(
        textEncoder: TextEncoderModel,
        unet: Unet,
        decoder: Decoder,
        encoder: Encoder?,
        controlNet: ControlNet? = nil,
        safetyChecker: SafetyChecker? = nil,
        reduceMemory: Bool = false
    ) {
        self.textEncoder = textEncoder
        self.unet = unet
        self.decoder = decoder
        self.encoder = encoder
        self.controlNet = controlNet
        self.safetyChecker = safetyChecker
        self.reduceMemory = reduceMemory
    }

    /// Creates a pipeline using the specified models and tokenizer
    ///
    /// - Parameters:
    ///   - textEncoder: Model for encoding tokenized text
    ///   - unet: Model for noise prediction on latent samples
    ///   - decoder: Model for decoding latent sample to image
    ///   - controlNet: Optional model to control generated images by additonal inputs
    ///   - safetyChecker: Optional model for checking safety of generated images
    ///   - reduceMemory: Option to enable reduced memory mode
    ///   - useMultilingualTextEncoder: Option to use system multilingual NLContextualEmbedding as encoder
    ///   - script: Optional natural language script to use for the text encoder.
    /// - Returns: Pipeline ready for image generation
    public init(
        textEncoder: TextEncoderModel,
        unet: Unet,
        decoder: Decoder,
        encoder: Encoder?,
        controlNet: ControlNet? = nil,
        safetyChecker: SafetyChecker? = nil,
        reduceMemory: Bool = false,
        useMultilingualTextEncoder: Bool = false,
        script: Script? = nil
    ) {
        self.textEncoder = textEncoder
        self.unet = unet
        self.decoder = decoder
        self.encoder = encoder
        self.controlNet = controlNet
        self.safetyChecker = safetyChecker
        self.reduceMemory = reduceMemory
        self.useMultilingualTextEncoder = useMultilingualTextEncoder
        self.script = script
    }

    /// Load required resources for this pipeline
    ///
    /// If reducedMemory is true this will instead call prewarmResources instead
    /// and let the pipeline lazily load resources as needed
    public func loadResources() async throws {
        if reduceMemory {
            try await prewarmResources()
        } else {
            try await withThrowingTaskGroup(of: Void.self) { group in
                group.addTask { try await unet.loadResources() }
                group.addTask { try await textEncoder.loadResources() }
                group.addTask { try await decoder.loadResources() }
                group.addTask { try await encoder?.loadResources() }
                group.addTask { try await controlNet?.loadResources() }
                group.addTask { try await safetyChecker?.loadResources() }
                try await group.waitForAll()
            }
        }
    }

    /// Unload the underlying resources to free up memory
    public func unloadResources() async {
        await withTaskGroup(of: Void.self) { group in
            group.addTask { await unet.unloadResources() }
            group.addTask { await textEncoder.unloadResources() }
            group.addTask { await decoder.unloadResources() }
            group.addTask { await encoder?.unloadResources() }
            group.addTask { await controlNet?.unloadResources() }
            group.addTask { await safetyChecker?.unloadResources() }
            await group.waitForAll()
        }
    }

    // Prewarm resources one at a time
    public func prewarmResources() async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { try await unet.prewarmResources() }
            group.addTask { try await textEncoder.prewarmResources() }
            group.addTask { try await decoder.prewarmResources() }
            group.addTask { try await encoder?.prewarmResources() }
            group.addTask { try await controlNet?.prewarmResources() }
            group.addTask { try await safetyChecker?.prewarmResources() }
            try await group.waitForAll()
        }
    }

    /// Image generation using stable diffusion
    /// - Parameters:
    ///   - configuration: Image generation configuration
    ///   - progressHandler: Callback to perform after each step, stops on receiving false response
    /// - Returns: An array of `imageCount` optional images.
    ///            The images will be nil if safety checks were performed and found the result to be un-safe
    public func generateImages(
        configuration config: Configuration,
        progressHandler: (Progress) -> Bool = { _ in true }
    ) async throws -> [CGImage?] {
        // Encode the input prompt and negative prompt
        let promptEmbedding = try await textEncoder.encode(config.prompt)
        let negativePromptEmbedding = try await textEncoder.encode(config.negativePrompt)

        if reduceMemory {
            await textEncoder.unloadResources()
        }

        // Convert to Unet hidden state representation
        // Concatenate the prompt and negative prompt embeddings
        let concatEmbedding = MLTensor(
            concatenating: [negativePromptEmbedding, promptEmbedding],
            alongAxis: 0
        )

        let hiddenStates = useMultilingualTextEncoder ? concatEmbedding : toHiddenStates(concatEmbedding)

        /// Setup schedulers
        var schedulers = [Scheduler]()
        schedulers.reserveCapacity(config.imageCount)
        for _ in 0..<config.imageCount {
            switch config.schedulerType {
            case .pndmScheduler:
                let scheduler = PNDMScheduler(stepCount: config.stepCount)
                schedulers.append(scheduler)
            case .dpmSolverMultistepScheduler:
                let scheduler = await DPMSolverMultistepScheduler(
                    stepCount: config.stepCount, timeStepSpacing: config.schedulerTimestepSpacing)
                schedulers.append(scheduler)
            }
        }

        // Generate random latent samples from specified seed
        var latents = try await generateLatentSamples(configuration: config, scheduler: schedulers[0])

        // Store denoised latents from scheduler to pass into decoder
        var denoisedLatents = latents.map { $0.cast(to: Float.self) }

        if reduceMemory {
            await encoder?.unloadResources()
        }
        let timestepStrength: Float? = config.mode == .imageToImage ? config.strength : nil

        // Convert cgImage for ControlNet into a tensor
        let controlNetConds = config.controlNetInputs.map { cgImage in
            guard let rgbTensor = cgImage.planarRGBTensor() else {
                fatalError("Failed to create RGB tensor.")
            }
            return rgbTensor.tiled(multiples: [2, 1, 1, 1])
        }

        // De-noising loop
        let timeSteps: [Int] = schedulers[0].calculateTimesteps(strength: timestepStrength)
        for (step,t) in timeSteps.enumerated() {

            // Expand the latents for classifier-free guidance
            // and input to the Unet noise prediction model
            let latentUnetInput = latents.map {
                MLTensor(concatenating: [$0, $0], alongAxis: 0)
            }

            // Before Unet, execute controlNet and add the output into Unet inputs
            let additionalResiduals = try await controlNet?.execute(
                latents: latentUnetInput,
                timeStep: t,
                hiddenStates: hiddenStates,
                images: controlNetConds,
                weights: config.controlNetWeights
            )

            // Predict noise residuals from latent samples
            // and current time step conditioned on hidden states
            var noise = try await unet.predictNoise(
                latents: latentUnetInput,
                timeStep: t,
                hiddenStates: hiddenStates,
                additionalResiduals: additionalResiduals
            )

            noise = performGuidance(noise, config.guidanceScale)

            // Have the scheduler compute the previous (t-1) latent
            // sample given the predicted noise and current sample
            for i in 0..<config.imageCount {
                latents[i] = schedulers[i].step(
                    output: noise[i],
                    timeStep: t,
                    sample: latents[i]
                )
                denoisedLatents[i] = schedulers[i].modelOutputs.last ?? latents[i]
            }

            let currentLatentSamples = config.useDenoisedIntermediates ? denoisedLatents : latents
            var currentPreviewImages: [CGImage?] = []
            if config.previewFrequency > 0 && step.isMultiple(of: config.previewFrequency) {
                currentPreviewImages = try await decodeToImages(currentLatentSamples, configuration: config)
            }

            // Report progress
            let progress = Progress(
                pipeline: self,
                prompt: config.prompt,
                step: step,
                stepCount: timeSteps.count,
                currentLatentSamples: currentLatentSamples,
                currentImages: currentPreviewImages,
                configuration: config
            )
            if !progressHandler(progress) {
                // Stop if requested by handler
                return []
            }
        }

        if reduceMemory {
            await controlNet?.unloadResources()
            await unet.unloadResources()
        }

        // Decode the latent samples to images
        return try await decodeToImages(denoisedLatents, configuration: config)
    }

    func generateLatentSamples(
        configuration config: Configuration,
        scheduler: Scheduler
    ) async throws -> [MLTensor] {
        var sampleShape = await unet.latentSampleShape
        sampleShape[0] = 1

        let stdev = scheduler.initNoiseSigma
        var random = randomSource(from: config.rngType, seed: config.seed)
        let samples = (0..<config.imageCount).map { _ in
            random.normalTensor(sampleShape, mean: 0.0, stdev: stdev)
        }
        if let image = config.startingImage, config.mode == .imageToImage {
            guard let encoder else {
                throw PipelineError.startingImageProvidedWithoutEncoder
            }
            let latent = try await encoder.encode(
                image,
                scaleFactor: config.encoderScaleFactor,
                random: &random
            )
            return scheduler.addNoise(originalSample: latent, noise: samples, strength: config.strength)
        }
        return samples
    }

    public func decodeToImages(
        _ latents: [MLTensor],
        configuration config: Configuration
    ) async throws -> [CGImage?] {
        let images = try await decoder.decode(latents, scaleFactor: config.decoderScaleFactor)
        if reduceMemory {
            await decoder.unloadResources()
        }

        // If safety is disabled return what was decoded
        if config.disableSafety {
            return images
        }

        // If there is no safety checker return what was decoded
        guard let safetyChecker = safetyChecker else {
            return images
        }

        // Otherwise change images which are not safe to nil
        var safeImages = [CGImage?]()
        for image in images {
            let safeImage = try await safetyChecker.isSafe(image) ? image : nil
            safeImages.append(safeImage)
        }

        if reduceMemory {
            await safetyChecker.unloadResources()
        }

        return safeImages
    }

}

/// Sampling progress details
public struct PipelineProgress {
    public let pipeline: StableDiffusionPipelineProtocol
    public let prompt: String
    public let step: Int
    public let stepCount: Int
    public let currentLatentSamples: [MLTensor]
    public let currentImages: [CGImage?]
    public let configuration: PipelineConfiguration
    public var isSafetyEnabled: Bool {
        pipeline.canSafetyCheck && !configuration.disableSafety
    }
}

public extension StableDiffusionPipeline {
    /// Sampling progress details
    typealias Progress = PipelineProgress
}

// Helper functions

extension StableDiffusionPipelineProtocol {
    func randomSource(from rng: StableDiffusionRNG, seed: UInt32) -> RandomSource {
        switch rng {
        case .numpyRNG: NumPyRandomSource(seed: seed)
        case .torchRNG: TorchRandomSource(seed: seed)
        case .nvidiaRNG: NvRandomSource(seed: seed)
        case .coreml: MLTensorRandomSource(seed: seed)
        }
    }

    /// Transpose `(0, 2, 1)` and expand at `2`.
    func toHiddenStates(_ embedding: MLTensor) -> MLTensor {
        embedding.transposed(permutation: [0, 2, 1]).expandingShape(at: 2)
    }

    func performGuidance(_ noise: [MLTensor], _ guidanceScale: Float) -> [MLTensor] {
        noise.map { performGuidance($0, guidanceScale) }
    }

    func performGuidance(_ noise: MLTensor, _ guidanceScale: Float) -> MLTensor {
        let splitNoise = noise.split(count: 2)
        let (predictedNoiseUncond, predictedNoiseCond) = (splitNoise[0], splitNoise[1])
        return predictedNoiseUncond + guidanceScale * (predictedNoiseCond - predictedNoiseUncond)
    }
}
