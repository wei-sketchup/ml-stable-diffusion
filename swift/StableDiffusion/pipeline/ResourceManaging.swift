// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

/// Protocol for managing internal resources
public protocol ResourceManaging {

    /// Request resources to be loaded and ready if possible
    func loadResources() async throws

    /// Request resources are unloaded / remove from memory if possible
    func unloadResources() async
}

extension ResourceManaging {
    /// Request resources are pre-warmed by loading and unloading
    func prewarmResources() async throws {
        try await loadResources()
        await unloadResources()
    }
}

extension Array where Element: ResourceManaging {
    /// Request resources to be loaded and ready if possible.
    func loadResources() async throws {
        for model in self {
            try await model.loadResources()
        }
    }

    /// Request resources are unloaded / remove from memory if possible.
    func unloadResources() async {
        for model in self {
            await model.unloadResources()
        }
    }

    /// Request resources are pre-warmed by loading and unloading each model.
    func prewarmResources() async throws {
        for model in self {
            try await model.prewarmResources()
        }
    }
}
