// swift-tools-version:5.9
//
// Root forwarder so Xcode's "Add Package Dependencies → URL" flow works
// against this repo. Upstream livekit/livekit-wakeword keeps the actual
// Swift package at swift/Package.swift because the repo also contains
// Python and Rust crates; Xcode only looks at the repo root, so without
// this file the remote SPM resolve fails with "Package.swift doesn't
// exist in file system."
//
// The package code, sources, and resources still live under swift/ —
// this manifest just points the SwiftPM target at them.
import PackageDescription

let package = Package(
    name: "LiveKitWakeWord",
    platforms: [.iOS(.v16), .macOS(.v14)],
    products: [
        .library(name: "LiveKitWakeWord", targets: ["LiveKitWakeWord"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/microsoft/onnxruntime-swift-package-manager",
            from: "1.20.0"
        ),
    ],
    targets: [
        .target(
            name: "LiveKitWakeWord",
            dependencies: [
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ],
            path: "swift/Sources/LiveKitWakeWord",
            resources: [
                .copy("Resources/melspectrogram.onnx"),
                .copy("Resources/embedding_model.onnx"),
            ]
        ),
    ]
)
