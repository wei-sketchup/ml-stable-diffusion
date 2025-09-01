// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.

import CoreML

struct MLTensorRandomSource: RandomSource {
    let seed: UInt64

    init(seed: UInt32) {
        self.seed = UInt64(seed)
    }

    func normalTensor(_ shape: [Int], mean: Float, stdev: Float) -> MLTensor {
        MLTensor(randomNormal: shape, mean: mean, standardDeviation: stdev, seed: seed)
    }
}
