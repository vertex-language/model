// The 'model' repository: running models. Families (model/llama, ...),
// the kits they share, weight formats and generation. See README.md.
import PackageDescription

let package = Package(
    name: "model",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .library(name: "model/gguf", targets: ["gguf"]),
        .executable(name: "test-gguf", targets: ["test_gguf"]),
        .executable(name: "test-quant", targets: ["test_quant"]),
    ],
    targets: [
        // GGUF: llama.cpp's weight format, metadata and tensors, mapped.
        .target(
            name: "gguf",
            path: "gguf"
        ),
        .executableTarget(
            name: "test_gguf",
            dependencies: ["gguf"],
            path: "tests/gguf",
            exclude: ["golden"]
        ),
        // A quantized model's tensors decoded on every device, against
        // llama.cpp's dequantizer.
        .executableTarget(
            name: "test_quant",
            dependencies: ["gguf"],
            path: "tests/quant",
            exclude: ["golden"]
        ),
    ]
)
