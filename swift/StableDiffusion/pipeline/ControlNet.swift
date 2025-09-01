// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import CoreML
import Accelerate

public struct ControlNet: ResourceManaging {

    let models: [ManagedMLModel]

    public init(modelAt urls: [URL], configuration: MLModelConfiguration) {
        models = urls.map { ManagedMLModel(modelAt: $0, configuration: configuration) }
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
        try await models.prewarmResources()
    }

    var inputImageDescriptions: [MLFeatureDescription] {
        get async throws {
            try await models.featureDescriptions(forInputNamed: "controlnet_cond") ?? []
        }
    }

    /// The expected shape of the models image input
    public var inputImageShapes: [[Int]] {
        get async throws {
            try await models.shapes(forInputNamed: "controlnet_cond") ?? []
        }
    }

    /// Calculate additional inputs for Unet to generate intended image following provided images
    ///
    /// - Parameters:
    ///   - latents: Batch of latent samples in an array
    ///   - timeStep: Current diffusion timestep
    ///   - hiddenStates: Hidden state to condition on
    ///   - images: Images for each ControlNet
    ///   - weights: Weights applied to the images.
    /// - Returns: Array of predicted noise residuals
    func execute(
        latents: [MLTensor],
        timeStep: Int,
        hiddenStates: MLTensor,
        images: [MLTensor],
        weights: [Float16],
    ) async throws -> [[String: MLTensor]] {
        // Match time step batch dimension to the model / latent samples
        let t = MLTensor(shape: [2], scalars: [Float(timeStep), Float(timeStep)])

        var results = [[String: MLTensor]]()
        results.reserveCapacity(models.count)

        for (modelIndex, model) in models.enumerated() {
            for (latentIndex, latent) in latents.enumerated() {
                let inputs = [
                    "sample": latent,
                    "timestep": t,
                    "encoder_hidden_states": hiddenStates,
                    "controlnet_cond": images[modelIndex]
                ]
                var outputs = try await model.perform { model in
                    try await model.prediction(from: inputs)
                }

                for key in outputs.keys {
                    outputs[key] = outputs[key]! * weights[modelIndex]
                }

                if modelIndex == 0 {
                    results.append(outputs)
                } else {
                    for outputName in outputs.keys {
                        guard
                            let lhs = results[latentIndex][outputName],
                            let rhs = outputs[outputName]
                        else {
                            fatalError("Output mismatch detected at \(#file):\(#line)")
                        }
                        results[latentIndex][outputName] = lhs + rhs
                    }
                }
            }
        }
        return results
    }
}
