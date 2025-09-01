// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

/// A decoder model which produces RGB images from latent samples
public struct Decoder: ResourceManaging {
    /// VAE decoder model
    var model: ManagedMLModel

    /// Create decoder from Core ML model
    ///
    /// - Parameters:
    ///     - url: Location of compiled VAE decoder Core ML model
    ///     - configuration: configuration to be used when the model is loaded
    /// - Returns: A decoder that will lazily load its required resources when needed or requested
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

    /// Batch decode latent samples into images
    ///
    ///  - Parameters:
    ///    - latents: Batch of latent samples to decode
    ///    - scaleFactor: scalar divisor on latents before decoding
    ///  - Returns: decoded images
    public func decode(_ latents: [MLTensor], scaleFactor: Float32) async throws -> [CGImage] {
        guard let inputName = try await model.firstInputName() else {
            fatalError("Failed to retrieve input name, \(#file) \(#line)")
        }
        var decodedLatents = [MLTensor]()
        decodedLatents.reserveCapacity(latents.count)
        for latent in latents {
            // Reference pipeline scales the latent samples before decoding
            let scaledLatent = latent / scaleFactor
            let outputs = try await model.perform { model in
                try await model.prediction(from: [inputName: scaledLatent])
            }
            guard let output = outputs.values.first else {
                fatalError("Missing output(s) \(#file) \(#line)")
            }
            decodedLatents.append(output)
        }
        // Transform the outputs to CGImages
        var images = [CGImage]()
        images.reserveCapacity(decodedLatents.count)
        for decodedLatent in decodedLatents {
            let image = try await CGImage.fromTensor(decodedLatent)
            images.append(image)
        }
        return images
    }
}
