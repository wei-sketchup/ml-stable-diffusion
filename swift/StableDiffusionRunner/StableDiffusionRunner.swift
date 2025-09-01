// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import ArgumentParser
import CoreGraphics
import CoreML
import Foundation
import StableDiffusion
import UniformTypeIdentifiers
import Cocoa
import CoreImage
//import NaturalLanguage

@main struct StableDiffusionRunner: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Run stable diffusion 1.5 to generate images guided by a text prompt",
        version: "0.1"
    )

    @Argument(help: "Input string prompt")
    var prompt: String

    @Option(help: "Input string negative prompt")
    var negativePrompt: String = ""

    @Option(
        help: ArgumentHelp(
            "Path to stable diffusion resources.",
            discussion: "The resource directory should contain\n" +
                " - *compiled* models: {TextEncoder,Unet,VAEDecoder}.mlmodelc\n" +
                " - tokenizer info: vocab.json, merges.txt",
            valueName: "directory-path"
        )
    )
    var resourcePath: String = "./"

    @Option(help: "Path to starting image.")
    var image: String? = nil

    @Option(help: "Strength for image2image.")
    var strength: Float = 0.5

    @Option(help: "Number of images to sample / generate")
    var imageCount: Int = 1

    @Option(help: "Number of diffusion steps to perform, increase to improve quality")
    var stepCount: Int = 25

    @Option(
        help: ArgumentHelp(
            "How often to save samples at intermediate steps",
            discussion: "Set to 0 to only save the final sample"
        )
    )
    var saveEvery: Int = 0

    @Option(help: "Output path")
    var outputPath: String = "./"

    @Option(help: "Random seed")
    var seed: UInt32 = UInt32.random(in: 0...UInt32.max)

    @Option(help: "Controls the influence of the text prompt on sampling process (0=random images)")
    var guidanceScale: Float = 7.5

    @Option(help: "Compute units to load model with {all,cpuOnly,cpuAndGPU,cpuAndNeuralEngine}")
    var computeUnits: ComputeUnits = .cpuAndGPU

    @Option(help: "Scheduler to use, one of {pndm, dpmpp}")
    var scheduler: SchedulerOption = .pndm

    @Option(help: "Random number generator to use, one of {numpy, torch, nvidia, coreml}")
    var rng: RNGOption = .coreml

    @Option(
        parsing: .upToNextOption,
        help: "ControlNet models used in image generation (enter file names in Resources/controlnet without extension)"
    )
    var controlnet: [String] = []

    @Option(
        parsing: .upToNextOption,
        help: "image for each controlNet model (corresponding to the same order as --controlnet)"
    )
    var controlnetInputs: [String] = []

    @Option(
        parsing: .upToNextOption,
        help: "weight (or scale) for each controlNet model (corresponding to the same order as --controlnet)"
    )
    var controlnetWeights: [Float16] = []

    mutating func run() async throws {
        guard FileManager.default.fileExists(atPath: resourcePath) else {
            throw RunError.resources("Resource path does not exist \(resourcePath)")
        }

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits.asMLComputeUnits
        config.allowLowPrecisionAccumulationOnGPU = true
        let resourceURL = URL(filePath: resourcePath)

        log("Loading resources and creating pipeline\n")
        log("(Note: This can take a while the first time using these resources)\n")
        let scaleFactor: Float32 = 0.18215
        let pipeline = try StableDiffusionPipeline(
            resourcesAt: resourceURL,
            controlNet: controlnet,
            configuration: config
        )

        try await pipeline.loadResources()

        let startingImage: CGImage? = switch image {
        case .some(let image): try convertImageToCGImage(imageURL: URL(filePath: image))
        case .none: nil
        }

        // convert image for ControlNet into CGImage when controlNet available
        let controlNetInputs = try controlnetInputs.compactMap { imagePath in
            let imageURL = URL(filePath: imagePath)
            return try convertImageToCGImage(imageURL: imageURL)
        }

        log("Sampling ...\n")
        let sampleTimer = SampleTimer()
        sampleTimer.start()

        var pipelineConfig = StableDiffusionPipeline.Configuration(prompt: prompt)

        assert(
            controlnet.count == controlnetInputs.count &&
            controlnetInputs.count == controlNetInputs.count &&
            controlnet.count == controlnetWeights.count
        )

        pipelineConfig.negativePrompt = negativePrompt
        pipelineConfig.startingImage = startingImage
        pipelineConfig.strength = strength
        pipelineConfig.imageCount = imageCount
        pipelineConfig.stepCount = stepCount
        pipelineConfig.seed = seed
        pipelineConfig.controlNetInputs = controlNetInputs
        pipelineConfig.controlNetWeights = controlnetWeights
        pipelineConfig.guidanceScale = guidanceScale
        pipelineConfig.schedulerType = scheduler.stableDiffusionScheduler
        pipelineConfig.rngType = rng.stableDiffusionRNG
        pipelineConfig.useDenoisedIntermediates = true
        pipelineConfig.encoderScaleFactor = scaleFactor
        pipelineConfig.decoderScaleFactor = scaleFactor
        pipelineConfig.previewFrequency = saveEvery

        let images = try await withMLTensorComputePolicy(.cpuOnly) {
            try await pipeline.generateImages(configuration: pipelineConfig) { progress in
                sampleTimer.stop()
                handleProgress(progress,sampleTimer)
                if progress.stepCount != progress.step {
                    sampleTimer.start()
                }
                return true
            }
        }

        _ = try saveImages(
            images,
            outputPath: outputPath,
            imageNameFn: makeImageNameFn(),
            logNames: true
        )
    }

    func handleProgress(
        _ progress: StableDiffusionPipeline.Progress,
        _ sampleTimer: SampleTimer
    ) {
        log("\u{1B}[1A\u{1B}[K")
        log("Step \(progress.step) of \(progress.stepCount) ")
        log(" [")
        log(String(format: "mean: %.2f, ", 1.0/sampleTimer.mean))
        log(String(format: "median: %.2f, ", 1.0/sampleTimer.median))
        log(String(format: "last %.2f", 1.0/sampleTimer.allSamples.last!))
        log("] step/sec")

        if saveEvery > 0, progress.step % saveEvery == 0 {
            let currentImages = progress.currentImages.compactMap { $0 }
            if !currentImages.isEmpty {
                let saveCount = (try? saveImages(
                    currentImages,
                    outputPath: outputPath,
                    imageNameFn: makeImageNameFn(),
                    step: progress.step
                )) ?? 0
                log(" saved \(saveCount) image\(saveCount != 1 ? "s" : "")")
            }
        }
        log("\n")
    }

    func makeImageNameFn() -> ((Int, Int?) -> String) {
        func imageNameFactory(_ sample: Int, _ step: Int? = nil) -> String {
            let fileCharLimit = 75
            var name = prompt.prefix(fileCharLimit).replacingOccurrences(of: " ", with: "_")
            if imageCount != 1 {
                name += ".\(sample)"
            }

            if image != nil {
                name += ".str\(Int(strength * 100))"
            }

            name += ".\(seed)"

            if let step = step {
                name += ".\(step)"
            } else {
                name += ".final"
            }
            name += ".png"
            return name
        }

        return imageNameFactory
    }
}

enum RunError: Error {
    case resources(String)
    case saving(String)
    case unsupported(String)
}

enum ComputeUnits: String, ExpressibleByArgument, CaseIterable {
    case all, cpuAndGPU, cpuOnly, cpuAndNeuralEngine
    var asMLComputeUnits: MLComputeUnits {
        switch self {
        case .all: return .all
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuOnly: return .cpuOnly
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        }
    }
}

enum SchedulerOption: String, ExpressibleByArgument {
    case pndm, dpmpp
    var stableDiffusionScheduler: StableDiffusionScheduler {
        switch self {
        case .pndm: return .pndmScheduler
        case .dpmpp: return .dpmSolverMultistepScheduler
        }
    }
}

enum RNGOption: String, ExpressibleByArgument {
    case numpy, torch, nvidia, coreml
    var stableDiffusionRNG: StableDiffusionRNG {
        switch self {
        case .numpy: .numpyRNG
        case .torch: .torchRNG
        case .nvidia: .nvidiaRNG
        case .coreml: .coreml
        }
    }
}

extension Script: ExpressibleByArgument {}

extension Float16: @retroactive _SendableMetatype {}
extension Float16: @retroactive ExpressibleByArgument {}
