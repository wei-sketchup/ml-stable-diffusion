// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Accelerate
//@_spi(MLTensor) import CoreML
import CoreML
import CoreGraphics

extension CGImage {
    public enum CGImageConversionError: String, Error {
        case wrongNumberOfChannels
        case creatingCGImageFailed
    }

    public static func fromTensor(_ tensor: MLTensor) async throws -> CGImage {
        // Expecting the given tensor to have the shape [N,C,H,W], where C == 3
        let channelCount = tensor.shape[1]
        guard channelCount == 3 else {
            throw CGImageConversionError.wrongNumberOfChannels
        }

        let height = tensor.shape[2]
        let width = tensor.shape[3]

        // Map values from -1.0 to 1.0 to a range of 0 to 255
        var normalizedTensor = ((tensor + 1.0) * 255.0) / 2.0
        normalizedTensor = normalizedTensor.clamped(to: 0...255)
        // Add the alpha channel
        let alpha = MLTensor(repeating: 255, shape: [1, 1, height ,width], scalarType: Float.self)
        // Concatenate the color channels with the alpha along the channel dimension
        var rgb = MLTensor(concatenating: [normalizedTensor, alpha], alongAxis: 1)
        // Transpose from [N, C, H, W] to [N, H, W, C] (and remove the batch dimension)
        rgb = rgb.transposed(permutation: [0, 2, 3, 1]).squeezingShape(at: 0)
        // Using SPI (requires using Swift from internal toolchain)
//        let image = await rgb.withUnsafeBytes { buffer in
//            CGContext(
//                data: UnsafeMutableRawBufferPointer(mutating: buffer).baseAddress!,
//                width: width,
//                height: height,
//                bitsPerComponent: 8,
//                bytesPerRow: MemoryLayout<UInt8>.stride * Int(width) * 4,
//                space: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB(),
//                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
//            )?.makeImage()
//        }
//        guard let image else {
//            throw CGImageConversionError.creatingCGImageFailed
//        }
        let rgbArray = await rgb.shapedArray(of: Float.self)
        // Cast to UInt8
        var rgb8bitArray = Array<UInt8>(repeating: 0, count: rgbArray.scalarCount)
        rgb8bitArray.withUnsafeMutableBufferPointer { destPointer in
            rgbArray.changingLayout(to: .firstMajorContiguous)
                .withUnsafeShapedBufferPointer { srcPointer, srcShape, srcStride in
                    vDSP.convertElements(of: srcPointer, to: &destPointer, rounding: .towardNearestInteger)
                }
        }
        guard let image = CGContext(
            data: &rgb8bitArray,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: MemoryLayout<UInt8>.stride * Int(width) * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )?.makeImage() else {
            throw CGImageConversionError.creatingCGImageFailed
        }
        return image
    }

    /// Returns a 32-bit floating point tensor representation of the image with the layout `1x3xHxW`.
    public func planarRGBTensor(minValue: Float = 0.0, maxValue: Float = 1.0) -> MLTensor? {
        guard let rgbaData else {
            return nil
        }
        var image = MLTensor(
            shape: [1, Int(height), Int(width), 4],
            data: rgbaData,
            scalarType: UInt8.self
        ).cast(to: Float.self)
        image = image[nil, nil, nil, 0..<3]
        image = image.transposed(permutation: [0, 3, 1, 2])
        image /= 255.0

        if minValue == 0.0, maxValue == 1.0 {
            return image
        }
        return image * (maxValue - minValue) + minValue
    }
}

extension CGImage {
    fileprivate var rgbaData: Data? {
        var data = Data(repeating: 0, count: width * height * 4)
        data.withUnsafeMutableBytes { pointer in
            let context = CGContext(
                data: pointer.baseAddress!,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: MemoryLayout<UInt8>.stride * width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            context?.draw(
                self,
                in: CGRect(origin: .zero, size: CGSize(width: width, height: height)))
        }
        return data
    }
}
