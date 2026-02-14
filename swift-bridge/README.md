# Swift-to-Rust CoreML Bridge

This directory contains the Swift code that wraps Apple's CoreML framework for depth estimation, making it callable from Rust via FFI.

## Architecture

```
Rust (src/depth_coreml.rs)
    ↕ FFI (extern "C")
Swift (CoreMLDepth.swift)
    ↕ Apple Frameworks
CoreML + Neural Engine
```

## Files

- **CoreMLDepth.swift** — Swift wrapper providing C-compatible API for CoreML
- **README.md** — This file

## API

### coreml_load_model

```swift
@_cdecl("coreml_load_model")
public func loadModel(_ pathPtr: UnsafePointer<CChar>) -> UnsafeMutableRawPointer?
```

Loads and compiles a CoreML model from an `.mlpackage` file.

**Returns**: Opaque pointer to MLModel, or NULL on error

### coreml_infer_depth

```swift
@_cdecl("coreml_infer_depth")
public func inferDepth(
    _ modelPtr: UnsafeMutableRawPointer,
    _ rgbData: UnsafePointer<Float>,
    _ width: Int32,
    _ height: Int32,
    _ outputPtr: UnsafeMutablePointer<Float>
) -> Int32
```

Runs depth estimation inference.

**Input**:
- `modelPtr`: Model from `coreml_load_model`
- `rgbData`: NCHW Float32 array [1, 3, H, W]
- `width`, `height`: Input dimensions
- `outputPtr`: Buffer for output [H * W floats]

**Returns**: 0 on success, negative error code on failure

### coreml_unload_model

```swift
@_cdecl("coreml_unload_model")
public func unloadModel(_ modelPtr: UnsafeMutableRawPointer)
```

Releases the model.

## Data Types

### Input
- Format: NCHW (batch, channels, height, width)
- Type: Float32
- Shape: [1, 3, 518, 518]
- Normalized: ImageNet mean/std

### Output
- Format: HW (height, width)  
- Type: Float16 (model) → Float32 (converted for Rust)
- Shape: [518, 518]
- Range: Raw depth values (not normalized)

## Compilation

This Swift code is compiled by `build.rs` when building the Rust crate:

```bash
swiftc -emit-library -static -module-name CoreMLDepth -O \
    swift-bridge/CoreMLDepth.swift -o libCoreMLDepth.a
```

Linked frameworks:
- CoreML.framework
- CoreVideo.framework
- Accelerate.framework
- Foundation.framework

## Error Handling

Error codes returned by `coreml_infer_depth`:
- `0`: Success
- `-1`: Failed to create input MLMultiArray
- `-2`: Inference failed
- `-3`: Failed to extract depth output
- `-4`: General error (see stdout for details)

## Performance Notes

### First Load
- Compiles `.mlpackage` to `.mlmodelc` (~7.5s)
- Cached in `/var/folders/.../T/`

### Subsequent Loads
- Uses cached `.mlmodelc`
- Fast (~10ms)

### Inference
- Base model: ~128ms on M4 Pro
- Uses Apple Neural Engine + GPU + CPU
- Configured via `config.computeUnits = .all`

## Testing

Test Swift code directly:

```swift
import CoreML

let path = "checkpoints/DepthAnythingV2BaseF16.mlpackage"
let url = URL(fileURLWithPath: path)

let compiledURL = try MLModel.compileModel(at: url)
let config = MLModelConfiguration()
config.computeUnits = .all
let model = try MLModel(contentsOf: compiledURL, configuration: config)

print("Model loaded: \(model.modelDescription)")
```

## Debugging

Enable CoreML logging:

```bash
export COREML_VERBOSE=1
cargo run --example photo_coreml
```

## Requirements

- macOS 15+ (for Float16 support)
- Xcode Command Line Tools
- Swift 5.0+
- Apple Silicon (for Neural Engine)

## References

- [CoreML API Docs](https://developer.apple.com/documentation/coreml)
- [MLModel Class](https://developer.apple.com/documentation/coreml/mlmodel)
- [Swift C Interop](https://developer.apple.com/documentation/swift/c-interoperability)
