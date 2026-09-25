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
        .library(name: "model/llama", targets: ["llama"]),
        .executable(name: "test-gguf", targets: ["test_gguf"]),
        .executable(name: "test-quant", targets: ["test_quant"]),
        .executable(name: "test-llama", targets: ["test_llama"]),
    ],
    targets: [
        // GGUF: llama.cpp's weight format, metadata and tensors, mapped.
        .target(
            name: "gguf",
            path: "gguf"
        ),
        // The Llama family: Llama 1 and 2 and what shares their
        // architecture, run from GGUF.
        .target(
            name: "llama",
            dependencies: ["gguf"],
            path: "llama"
        ),
        .executableTarget(
            name: "test_llama",
            dependencies: ["llama"],
            path: "tests/llama",
            exclude: ["golden"]
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
