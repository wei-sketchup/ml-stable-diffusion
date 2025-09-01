import CoreML

public protocol RandomSource {
    /// Generate a tensor with scalars from a normal distribution with given mean and standard deviation.
    mutating func normalTensor(_ shape: [Int], mean: Float, stdev: Float) -> MLTensor
}

extension RandomSource {
    mutating func normalTensor(_ shape: [Int]) -> MLTensor {
        normalTensor(shape, mean: 0.0, stdev: 1.0)
    }
}
