// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

/// U-Net noise prediction model for stable diffusion
public struct Unet: ResourceManaging {

    /// Model used to predict noise residuals given an input, diffusion time step, and conditional embedding
    ///
    /// It can be in the form of a single model or multiple stages
    let models: [ManagedMLModel]

    /// Creates a U-Net noise prediction model
    ///
    /// - Parameters:
    ///   - url: Location of single U-Net  compiled Core ML model.
    ///   - configuration: Configuration to be used when the model is loaded.
    ///   - functionName: The function name the U-net model will use. The default function is used if no
    ///     function name is specified.
    /// - Returns: U-net model that will lazily load its required resources when needed or requested
    public init(
        modelAt url: URL,
        configuration: MLModelConfiguration,
        functionName: String? = nil
    ) {
        var configuration = configuration
        if let functionName {
            configuration = configuration.copy() as! MLModelConfiguration
            configuration.functionName = functionName
        }
        models = [ManagedMLModel(modelAt: url, configuration: configuration)]
    }

    /// Creates a U-Net noise prediction model
    ///
    /// - Parameters:
    ///   - urls: Location of chunked U-Net via urls to each compiled chunk.
    ///   - configuration: Configuration to be used when the model is loaded.
    ///   - functionName: The function name the U-net model will use. The default function is used if no
    ///     function name is specified.
    /// - Returns: U-net model that will lazily load its required resources when needed or requested
    public init(
        chunksAt urls: [URL],
        configuration: MLModelConfiguration,
        functionName: String? = nil
    ) {
        var configuration = configuration
        if let functionName {
            configuration = configuration.copy() as! MLModelConfiguration
            configuration.functionName = functionName
        }
        self.models = urls.map { ManagedMLModel(modelAt: $0, configuration: configuration) }
    }

    /// Load resources.
    public func loadResources() async throws {
        try await models.loadResources()
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() async {
        await models.unloadResources()
    }

    /// Pre-warm resources
    public func prewarmResources() async throws {
        // Override default to pre-warm each model
        // TODO: Verify that this is called.
        try await models.prewarmResources()
    }

    /// The expected shape of the models latent sample input
    public var latentSampleShape: [Int] {
        get async {
            guard let shape = try? await models.firstShape(forInputNamed: InputNames.sample) else {
                fatalError("Failed to get input shape for input named `\(InputNames.sample)`.")
            }
            return shape
        }
    }

    /// The expected shape of the geometry conditioning
    public var latentTimeIdShape: [Int] {
        get async {
            guard let shape = try? await models.firstShape(forInputNamed: InputNames.timeIds) else {
                fatalError("Failed to get input shape for input named `\(InputNames.timeIds)`.")
            }
            return shape
        }
    }

    /// Batch prediction noise from latent samples
    ///
    /// - Parameters:
    ///   - latents: Batch of latent samples in an array
    ///   - timeStep: Current diffusion timestep
    ///   - hiddenStates: Hidden state to condition on
    /// - Returns: Array of predicted noise residuals
    func predictNoise(
        latents: [MLTensor],
        timeStep: Int,
        hiddenStates: MLTensor,
        additionalResiduals: [[String: MLTensor]]? = nil
    ) async throws -> [MLTensor] {
        // Match time step batch dimension to the model / latent samples
        let t = MLTensor(shape: [2], scalars:[Float(timeStep), Float(timeStep)])

        // Dispatch all inputs and return the results.
        var noise = [MLTensor]()
        for (index, latent) in latents.enumerated() {
            var inputs = [
                InputNames.sample : latent,
                InputNames.timestep : t,
                InputNames.encoderHiddenStates: hiddenStates
            ]
            if let residuals = additionalResiduals?[index] {
                for (k, v) in residuals {
                    inputs[k] = v
                }
            }
            // Make predictions
            let outputs = try await predictions(from: inputs)
            guard let result = outputs.values.first else {
                fatalError("Missing output(s) \(#file) \(#line)")
            }
            noise.append(result.cast(to: Float.self))
        }
        return noise
    }

    /// Batch prediction noise from latent samples, for Stable Diffusion XL
    ///
    /// - Parameters:
    ///   - latents: Batch of latent samples in an array
    ///   - timeStep: Current diffusion timestep
    ///   - hiddenStates: Hidden state to condition on
    ///   - pooledStates: Additional text states to condition on
    ///   - geometryConditioning: Condition on image geometry
    /// - Returns: Array of predicted noise residuals
    func predictNoise(
        latents: [MLTensor],
        timeStep: Int,
        hiddenStates: MLTensor,
        pooledStates: MLTensor,
        geometryConditioning: MLTensor
    ) async throws -> [MLTensor] {
        // Match time step batch dimension to the model / latent samples
        let t = MLTensor(shape: [2], scalars: [Float(timeStep), Float(timeStep)])

        // Dispatch all inputs and return the results.
        var noise = [MLTensor]()
        for latent in latents {
            let inputs = [
                InputNames.sample : latent,
                InputNames.timestep : t,
                InputNames.encoderHiddenStates: hiddenStates,
                InputNames.textEmbeddings: pooledStates,
                InputNames.timeIds: geometryConditioning
            ]
            // Make predictions
            let outputs = try await predictions(from: inputs)
            guard let result = outputs.values.first else {
                fatalError("Missing output(s) \(#file) \(#line)")
            }
            noise.append(result.cast(to: Float.self))
        }
        return noise
    }

    func predictions(from inputs: [String: MLTensor]) async throws -> [String: MLTensor] {
        var results = try await models.first!.perform { model in
            try await model.prediction(from: inputs)
        }
        guard models.count > 1 else {
            return results
        }
        // Manual pipeline batch prediction
        for stage in models.dropFirst() {
            // Combine the original inputs with the outputs of the last stage
            let next = results.merging(inputs) { out, _ in out }
            results = try await stage.perform { model in
                try await model.prediction(from: next)
            }
        }
        return results
    }
}

extension Unet {
    enum InputNames {
        static let sample = "sample"
        static let timestep = "timestep"
        static let encoderHiddenStates = "encoder_hidden_states"
        static let textEmbeddings = "text_embeds"
        static let timeIds = "time_ids"
    }
}
