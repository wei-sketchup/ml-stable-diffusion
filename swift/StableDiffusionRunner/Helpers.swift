import Foundation
import UniformTypeIdentifiers
import CoreGraphics
import Cocoa
import CoreImage

func convertImageToCGImage(imageURL: URL) throws -> CGImage {
    let imageData = try Data(contentsOf: imageURL)
    guard
        let nsImage = NSImage(data: imageData),
        let loadedImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil)
    else {
        throw RunError.resources("Image not available \(imageURL)")
    }
    return loadedImage
}

func saveImages(
    _ images: [CGImage?],
    outputPath: String,
    imageNameFn: (Int, Int?) -> String,
    step: Int? = nil,
    logNames: Bool = false
) throws -> Int {
    let url = URL(filePath: outputPath)
    var saved = 0
    for i in 0 ..< images.count {

        guard let image = images[i] else {
            if logNames {
                log("Image \(i) failed safety check and was not saved")
            }
            continue
        }

        let name = imageNameFn(i, step)
        let fileURL = url.appending(path:name)

        guard let dest = CGImageDestinationCreateWithURL(fileURL as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw RunError.saving("Failed to create destination for \(fileURL)")
        }
        CGImageDestinationAddImage(dest, image, nil)
        if !CGImageDestinationFinalize(dest) {
            throw RunError.saving("Failed to save \(fileURL)")
        }
        if logNames {
            log("Saved \(name)\n")
        }
        saved += 1
    }
    return saved
}

func log(_ str: String, term: String = "") {
    print(str, terminator: term)
}
