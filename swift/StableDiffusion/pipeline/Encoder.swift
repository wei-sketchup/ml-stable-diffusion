// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import CoreML

/// A encoder model which produces latent samples from RGB images
public struct Encoder: ResourceManaging {

    public enum Error: String, Swift.Error {
        case failedToCreateTensorFromImage
        case sampleInputShapeNotCorrect
        case missingOutput
    }

    /// VAE encoder model + post math and adding noise from schedular
    let model: ManagedMLModel

    /// Create encoder from Core ML model
    ///
    /// - Parameters:
    ///     - url: Location of compiled VAE encoder Core ML model
    ///     - configuration: configuration to be used when the model is loaded
    /// - Returns: An encoder that will lazily load its required resources when needed or requested
    public init(modelAt url: URL, configuration: MLModelConfiguration) {
        model = ManagedMLModel(modelAt: url, configuration: configuration)
    }

    /// Ensure the model has been loaded into memory
    public func loadResources() async throws {
        try await model.loadResources()
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() async {
       await model.unloadResources()
    }

    /// Encode image into latent sample
    ///
    ///  - Parameters:
    ///    - image: Input image
    ///    - scaleFactor: scalar multiplier on latents before encoding image
    ///    - random: Random number generator.
    ///  - Returns: The encoded latent space as a MLTensor.
    public func encode(
        _ image: CGImage,
        scaleFactor: Float32,
        random: inout some RandomSource
    ) async throws -> MLTensor {
        guard
            let inputName = try await model.firstInputName(),
            let inputShape = try await model.firstInputShape()
        else {
            fatalError("Failed to retrieve input name and/or input shape, \(#file) \(#line)")
        }
        guard let imageTensor = image.planarRGBTensor(minValue: -1.0, maxValue: 1.0) else {
            throw Error.failedToCreateTensorFromImage
        }
        guard imageTensor.shape == inputShape else {
            // TODO: Implement auto resizing and croping
            throw Error.sampleInputShapeNotCorrect
        }
        let outputs = try await model.perform { model in
            try await model.prediction(from: [inputName: imageTensor])
        }
        assert(outputs.count == 1)
        guard let output = outputs.values.first?.cast(to: Float.self) else {
            throw Error.missingOutput
        }

        // DiagonalGaussianDistribution
        let mean = output[0][0..<4]
        let logvar = output[0][4..<8].clamped(to: -30...20)
        let std = (0.5 * logvar).exp()
        let latent = random.normalTensor(logvar.shape) * mean + std

        // Reference pipeline scales the latent after encoding
        let latentScaled = (latent * scaleFactor).expandingShape(at: 0)

        return latentScaled
    }
}
