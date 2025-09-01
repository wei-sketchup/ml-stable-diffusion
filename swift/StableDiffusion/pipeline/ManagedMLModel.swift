// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import CoreML

/// A class to manage and gate access to a Core ML model
///
/// It will automatically load a model into memory when needed or requested
/// It allows one to request to unload the model from memory
actor ManagedMLModel: ResourceManaging {
    /// The location of the model
    var modelURL: URL

    /// The configuration to be used when the model is loaded
    var configuration: MLModelConfiguration

    /// The loaded model (when loaded)
    var loadedModel: MLModel?

    /// Create a managed model given its location and desired loaded configuration
    ///
    /// - Parameters:
    ///     - url: The location of the model
    ///     - configuration: The configuration to be used when the model is loaded/used
    /// - Returns: A managed model that has not been loaded
    public init(modelAt url: URL, configuration: MLModelConfiguration) {
        self.modelURL = url
        self.configuration = configuration
        self.loadedModel = nil
    }

    /// Instantiation and load model into memory
    public func loadResources() async throws {
        try loadModel()
    }

    /// Unload the model if it was loaded
    public func unloadResources() async {
        loadedModel = nil
    }

    /// Perform an operation with the managed model via a supplied closure.
    ///  The model will be loaded and supplied to the closure and should only be
    ///  used within the closure to ensure all resource management is synchronized
    ///
    /// - Parameters:
    ///     - body: Closure which performs and action on a loaded model
    /// - Returns: The result of the closure
    /// - Throws: An error if the model cannot be loaded or if the closure throws
    public func perform<R>(_ body: (MLModel) async throws -> R) async throws -> R {
        try autoreleasepool {
            try loadModel()
        }
        return try await body(loadedModel!)
    }

    private func loadModel() throws {
        if loadedModel == nil {
            loadedModel = try MLModel(contentsOf: modelURL, configuration: configuration)
        }
    }
}

extension ManagedMLModel {
    /// Returns the shape for the input matching the given name.
    ///
    /// - Parameter name: The input name.
    /// - Returns: The matching shape or `nil` if no match is found.
    func shape(forInputNamed name: String) async throws -> [Int]? {
        try await featureDescription(forInputNamed: name)?.multiArrayConstraint?.shape.map { $0.intValue }
    }

    /// Returns the description for the input matching the given name.
    ///
    /// - Parameter name: The input name.
    /// - Returns: The matching description or `nil` if no match is found.
    func featureDescription(forInputNamed name: String) async throws -> MLFeatureDescription? {
        try await perform { model in
            model.modelDescription.inputDescriptionsByName.first(where: { $0.key == name })?.value
        }
    }

    /// Returns the first input feature description.
    func firstInputFeatureDescription() async throws -> MLFeatureDescription? {
        try await perform { model in
            model.modelDescription.inputDescriptionsByName.first?.value
        }
    }

    /// Returns the name of the first input.
    func firstInputName() async throws -> String? {
        try await firstInputFeatureDescription()?.name
    }

    /// Returns the shape of the first input.
    func firstInputShape() async throws -> [Int]? {
        try await firstInputFeatureDescription()?.multiArrayConstraint?.shape.map { $0.intValue }
    }
}

extension Array where Element == ManagedMLModel {
    /// Returns all shapes from the models whose inputs match the given name.
    ///
    /// - Parameter name: The input name.
    /// - Returns: The matching shapes or `nil` if no match is found.
    func shapes(forInputNamed name: String) async throws -> [[Int]]? {
        let shapes = try await featureDescriptions(forInputNamed: name)?.compactMap { description in
            description.multiArrayConstraint?.shape.map { $0.intValue }
        }
        guard let shapes, shapes.count == count else {
            return nil
        }
        return shapes
    }

    /// Returns all descriptions from the models whose inputs match the given name.
    ///
    /// - Parameter name: The input name.
    /// - Returns: The matching description or `nil` if no match is found.
    func featureDescriptions(forInputNamed name: String) async throws -> [MLFeatureDescription]? {
        var descriptions = [MLFeatureDescription]()
        for model in self {
            if let description = try await model.perform({ model in
                model.modelDescription.inputDescriptionsByName.first(where: { $0.key == name })?.value
            }) {
                descriptions.append(description)
            }
        }
        guard descriptions.count == count else {
            return nil
        }
        return descriptions
    }

    /// Returns the shape of the first model's input matching the given name.
    ///
    /// - Parameter name: The input name.
    /// - Returns: The first matching shape or `nil` if no match is found.
    func firstShape(forInputNamed name: String) async throws -> [Int]? {
        try await firstFeatureDescription(forInputNamed: name)?.multiArrayConstraint?.shape.map { $0.intValue }
    }

    /// Returns the description of the first model's input matching the given name.
    ///
    /// - Parameter name: The input name.
    /// - Returns: The first matching description or `nil` if no match is found.
    func firstFeatureDescription(forInputNamed name: String) async throws -> MLFeatureDescription? {
        try await first?.perform { model in
            model.modelDescription.inputDescriptionsByName.first(where: { $0.key == name })?.value
        }
    }
}
