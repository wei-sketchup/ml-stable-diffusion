// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2023 Apple Inc. All Rights Reserved.

import Accelerate
import CoreGraphics
import CoreML
import Foundation
import NaturalLanguage


/// A pipeline used to generate image samples from text input using stable diffusion XL
///
/// This implementation matches:
/// [Hugging Face Diffusers XL Pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py)
public struct StableDiffusionXLPipeline: StableDiffusionPipelineProtocol {
    public typealias Configuration = PipelineConfiguration
    public typealias Progress = PipelineProgress

    /// Model to generate embeddings for tokenized input text
    var textEncoder: TextEncoderXLModel?
    var textEncoder2: TextEncoderXLModel

    /// Model used to predict noise residuals given an input, diffusion time step, and conditional embedding
    var unet: Unet

    /// Model used to refine the image, if present
    var unetRefiner: Unet?

    /// Model used to generate final image from latent diffusion process
    var decoder: Decoder

    /// Model used to latent space for image2image, and soon, in-painting
    var encoder: Encoder?

    /// Option to reduce memory during image generation
    ///
    /// If true, the pipeline will lazily load TextEncoder, Unet, Decoder, and SafetyChecker
    /// when needed and aggressively unload their resources after
    ///
    /// This will increase latency in favor of reducing memory
    var reduceMemory: Bool = false

    /// Creates a pipeline using the specified models and tokenizer
    ///
    /// - Parameters:
    ///   - textEncoder: Model for encoding tokenized text
    ///   - textEncoder2: Second text encoding model
    ///   - unet: Model for noise prediction on latent samples
    ///   - decoder: Model for decoding latent sample to image
    ///   - reduceMemory: Option to enable reduced memory mode
    /// - Returns: Pipeline ready for image generation
    public init(
        textEncoder: TextEncoderXLModel?,
        textEncoder2: TextEncoderXLModel,
        unet: Unet,
        unetRefiner: Unet?,
        decoder: Decoder,
        encoder: Encoder?,
        reduceMemory: Bool = false
    ) {
        self.textEncoder = textEncoder
        self.textEncoder2 = textEncoder2
        self.unet = unet
        self.unetRefiner = unetRefiner
        self.decoder = decoder
        self.encoder = encoder
        self.reduceMemory = reduceMemory
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
                group.addTask { try await textEncoder2.loadResources() }
                group.addTask { try await unet.loadResources() }
                group.addTask { try await decoder.loadResources() }
                group.addTask { try await textEncoder?.loadResources() }
                // Only prewarm refiner unet on load so it's unloaded until needed
                group.addTask { try await unetRefiner?.prewarmResources() }
                group.addTask { try await encoder?.loadResources() }
                try await group.waitForAll()
            }
        }
    }

    /// Unload the underlying resources to free up memory
    public func unloadResources() async {
        await withTaskGroup(of: Void.self) { group in
            group.addTask { await textEncoder2.unloadResources() }
            group.addTask { await unet.unloadResources() }
            group.addTask { await decoder.unloadResources() }
            group.addTask { await textEncoder?.unloadResources() }
            group.addTask { await unetRefiner?.unloadResources() }
            group.addTask { await encoder?.unloadResources() }
            await group.waitForAll()
        }
    }

    /// Prewarm resources one at a time
    public func prewarmResources() async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { try await textEncoder2.prewarmResources() }
            group.addTask { try await unet.prewarmResources() }
            group.addTask { try await decoder.prewarmResources() }
            group.addTask { try await textEncoder?.prewarmResources() }
            // Only prewarm refiner unet on load so it's unloaded until needed
            group.addTask { try await unetRefiner?.prewarmResources() }
            group.addTask { try await encoder?.prewarmResources() }
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
        // Determine input type of Unet
        // SDXL Refiner has a latentTimeIdShape of [2, 5]
        // SDXL Base has either [12] or [2, 6]
        let isRefiner = await unet.latentTimeIdShape.last == 5

        // Setup geometry conditioning for base/refiner inputs
        var baseInput: ModelInputs?
        var refinerInput: ModelInputs?

        // Check if the first textEncoder is available, which is required for base models
        if textEncoder != nil {
            baseInput = try await generateConditioning(using: config, forRefiner: isRefiner)
        }

        // Check if the refiner unet exists, or if the current unet is a refiner
        if unetRefiner != nil || isRefiner {
            refinerInput = try await generateConditioning(using: config, forRefiner: true)
        }

        if reduceMemory {
            await textEncoder?.unloadResources()
            await textEncoder2.unloadResources()
        }

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

        // Store current model
        var unetModel = unet
        var currentInput = baseInput ?? refinerInput

        var unetHiddenStates = currentInput?.hiddenStates
        var unetPooledStates = currentInput?.pooledStates
        var unetGeometryConditioning = currentInput?.geometryConditioning

        let timeSteps: [Int] = schedulers[0].calculateTimesteps(strength: timestepStrength)

        // Calculate which step to swap to refiner
        let refinerStartStep = Int(Float(timeSteps.count) * config.refinerStart)

        // De-noising loop
        for (step,t) in timeSteps.enumerated() {
            // Expand the latents for classifier-free guidance
            // and input to the Unet noise prediction model
            let latentUnetInput = latents.map {
                MLTensor(concatenating: [$0, $0], alongAxis: 0)
            }

            // Switch to refiner if specified
            if let refiner = unetRefiner, step == refinerStartStep {
                await unet.unloadResources()

                unetModel = refiner
                currentInput = refinerInput
                unetHiddenStates = currentInput?.hiddenStates
                unetPooledStates = currentInput?.pooledStates
                unetGeometryConditioning = currentInput?.geometryConditioning
            }

            guard let hiddenStates = unetHiddenStates,
                  let pooledStates = unetPooledStates,
                  let geometryConditioning = unetGeometryConditioning else {
                throw PipelineError.missingUnetInputs
            }

            // Predict noise residuals from latent samples
            // and current time step conditioned on hidden states
            var noise = try await unetModel.predictNoise(
                latents: latentUnetInput,
                timeStep: t,
                hiddenStates: hiddenStates,
                pooledStates: pooledStates,
                geometryConditioning: geometryConditioning
            )
            noise = performGuidance(noise, config.guidanceScale)

            if step.isMultiple(of: 1) {
                _ = await noise[0].shapedArray(of: Float.self)
            }

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

        // Unload resources
        if reduceMemory {
            await unet.unloadResources()
        }
        await unetRefiner?.unloadResources()


        // Decode the latent samples to images
        return try await decodeToImages(denoisedLatents, configuration: config)
    }

    func encodePrompt(
        _ prompt: String,
        forRefiner: Bool = false
    ) async throws -> (MLTensor, MLTensor) {
        if forRefiner {
            // Refiner only takes textEncoder2 embeddings
            // [1, 77, 1280]
            return try await textEncoder2.encode(prompt)
        } else {
            guard let encoder = textEncoder else {
                throw PipelineError.startingText2ImgWithoutTextEncoder
            }
            let (embeds1, _) = try await encoder.encode(prompt)
            let (embeds2, pooled) = try await textEncoder2.encode(prompt)

            // Base needs concatenated embeddings
            // [1, 77, 768], [1, 77, 1280] -> [1, 77, 2048]
            let embeds = MLTensor(concatenating: [embeds1, embeds2], alongAxis: 2)
            return (embeds, pooled)
        }
    }

    func generateConditioning(
        using config: Configuration,
        forRefiner: Bool = false
    ) async throws -> ModelInputs {
        // Encode the input prompt and negative prompt
        let (promptEmbedding, pooled) = try await encodePrompt(config.prompt, forRefiner: forRefiner)
        let (negativePromptEmbedding, negativePooled) = try await encodePrompt(config.negativePrompt, forRefiner: forRefiner)

        // Convert to Unet hidden state representation
        // Concatenate the prompt and negative prompt embeddings
        let hiddenStates = toHiddenStates(MLTensor(concatenating: [negativePromptEmbedding, promptEmbedding], alongAxis: 0))
        let pooledStates = MLTensor(concatenating: [negativePooled, pooled], alongAxis: 0)

        // Inline helper functions for geometry creation
        func refinerGeometry() -> MLTensor {
            let negativeGeometry = MLTensor(
                shape: [1, 5],
                scalars: [
                    config.originalSize, config.originalSize,
                    config.cropsCoordsTopLeft, config.cropsCoordsTopLeft,
                    config.negativeAestheticScore
                ]
            )
            let positiveGeometry = MLTensor(
                shape: [1, 5],
                scalars: [
                    config.originalSize, config.originalSize,
                    config.cropsCoordsTopLeft, config.cropsCoordsTopLeft,
                    config.aestheticScore
                ]
            )
            return MLTensor(concatenating: [negativeGeometry, positiveGeometry], alongAxis: 0)
        }

        func baseGeometry() async -> MLTensor {
            let geometry = await MLTensor(
                // TODO: This checks if the time_ids input is looking for [12] or [2, 6]
                // Remove once model input shapes are ubiquitous
                shape: unet.latentTimeIdShape.count > 1 ? [1, 6] : [6],
                scalars: [
                    config.originalSize, config.originalSize,
                    config.cropsCoordsTopLeft, config.cropsCoordsTopLeft,
                    config.targetSize, config.targetSize
                ]
            )
            return MLTensor(concatenating: [geometry, geometry], alongAxis: 0)
        }

        let geometry = forRefiner ? refinerGeometry() : await baseGeometry()

        return ModelInputs(hiddenStates: hiddenStates, pooledStates: pooledStates, geometryConditioning: geometry)
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
        let result = try await decoder.decode(latents, scaleFactor: config.decoderScaleFactor)
        if reduceMemory {
            await decoder.unloadResources()
        }
        return result
    }

    struct ModelInputs {
        var hiddenStates: MLTensor
        var pooledStates: MLTensor
        var geometryConditioning: MLTensor
    }
}
