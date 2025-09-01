// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "stable-diffusion",
    platforms: [
        .macOS("15.0"),
        .iOS("18.0"),
    ],
    products: [
        .library(
            name: "StableDiffusionLib",
            targets: ["StableDiffusionLib"]),
        .executable(
            name: "StableDiffusionSample",
            targets: ["StableDiffusionCLI"]),
        .executable(
            name: "StableDiffusionRunner",
            targets: ["StableDiffusionRunner"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.2.3")
    ],
    targets: [
        .target(
            name: "StableDiffusionLib",
            dependencies: [],
            path: "swift/StableDiffusion"),
        .executableTarget(
            name: "StableDiffusionCLI",
            dependencies: [
                "StableDiffusionLib",
                .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "swift/StableDiffusionCLI"),
        .executableTarget(
            name: "StableDiffusionRunner",
            dependencies: [
                "StableDiffusionLib",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ],
            path: "swift/StableDiffusionRunner"),
        .testTarget(
            name: "StableDiffusionTests",
            dependencies: ["StableDiffusionLib"],
            path: "swift/StableDiffusionTests",
            resources: [
                .copy("Resources/vocab.json"),
                .copy("Resources/merges.txt")
            ]),
    ]
)
