// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2023 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

public protocol TextEncoderXLModel: ResourceManaging {
    typealias TextEncoderXLOutput = (hiddenEmbeddings: MLTensor, pooledOutputs: MLTensor)
    func encode(_ text: String) async throws -> TextEncoderXLOutput
}

///  A model for encoding text, suitable for SDXL
public struct TextEncoderXL: TextEncoderXLModel {

    /// Text tokenizer
    let tokenizer: BPETokenizer

    /// Embedding model
    let model: ManagedMLModel

    /// Creates text encoder which embeds a tokenized string
    ///
    /// - Parameters:
    ///   - tokenizer: Tokenizer for input text
    ///   - url: Location of compiled text encoding  Core ML model
    ///   - configuration: configuration to be used when the model is loaded
    /// - Returns: A text encoder that will lazily load its required resources when needed or requested
    public init(tokenizer: BPETokenizer, modelAt url: URL, configuration: MLModelConfiguration) {
        self.tokenizer = tokenizer
        self.model = ManagedMLModel(modelAt: url, configuration: configuration)
    }

    /// Ensure the model has been loaded into memory
    public func loadResources() async throws {
        try await model.loadResources()
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() async {
       await model.unloadResources()
    }

    /// Encode input text/string
    ///
    ///  - Parameters:
    ///     - text: Input text to be tokenized and then embedded
    ///  - Returns: Embedding representing the input text
    public func encode(_ text: String) async throws -> TextEncoderXLOutput {
        guard let inputShape = try await model.firstInputShape() else {
            fatalError("Failed to obtain input shape \(#file) \(#line)")
        }
        // Get models expected input length
        let inputLength = inputShape.last!

        // Tokenize, padding to the expected length
        var (tokens, ids) = tokenizer.tokenize(input: text, minCount: inputLength)

        // Truncate if necessary
        if ids.count > inputLength {
            tokens = tokens.dropLast(tokens.count - inputLength)
            ids = ids.dropLast(ids.count - inputLength)
            let truncated = tokenizer.decode(tokens: tokens)
            print("Needed to truncate input '\(text)' to '\(truncated)'")
        }

        // Use the model to generate the embedding
        return try await encode(ids: ids)
    }

    func encode(ids: [Int]) async throws -> TextEncoderXLOutput {
        guard
            let inputName = try await model.firstInputName(),
            let inputShape = try await model.firstInputShape()
        else {
            fatalError("Failed to obtain input name and/or shape \(#file) \(#line)")
        }

        let floatIds = ids.map { Float32($0) }
        let inputTensor = MLTensor(shape: inputShape, scalars: floatIds)

        let outputs = try await model.perform { model in
            try await model.prediction(from: [inputName: inputTensor])
        }

        guard
            let embeddingFeature = outputs["hidden_embeds"],
            let pooledFeature = outputs["pooled_outputs"]
        else {
            fatalError("Missing output(s) \(#file) \(#line)")
        }
        return (embeddingFeature, pooledFeature)
    }
}
