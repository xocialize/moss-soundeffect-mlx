// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MossSoundEffectMLX",
    platforms: [.macOS(.v14), .iOS(.v16)],
    products: [
        .library(
            name: "MossSoundEffectMLX",
            targets: ["MossSoundEffectMLX"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.0")
    ],
    targets: [
        .target(
            name: "MossSoundEffectMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "Sources/MossSoundEffectMLX"
        ),
        .testTarget(
            name: "MossSoundEffectMLXTests",
            dependencies: ["MossSoundEffectMLX"],
            path: "Tests/MossSoundEffectMLXTests"
        ),
    ]
)
