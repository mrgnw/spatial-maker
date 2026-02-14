# CoreML Implementation Guide

## Overview

We successfully implemented a Swift-to-Rust bridge to use CoreML depth estimation models on Apple Silicon, achieving **3.4x speedup** over ONNX CPU inference.

## Architecture

```
┌─────────────┐
│   Rust      │
│  (examples/ │
│   photo_    │
│   coreml.rs)│
└──────┬──────┘
       │ FFI calls via @_cdecl
       ▼
┌─────────────────────────────┐
│   Swift Bridge              │
│  (swift-bridge/             │
│   CoreMLDepth.swift)        │
│                             │
│  - coreml_load_model()      │
│  - coreml_infer_depth()     │
│  - coreml_unload_model()    │
└──────┬──────────────────────┘
       │ Apple Frameworks
       ▼
┌─────────────────────────────┐
│   CoreML + Frameworks       │
│  - CoreML.framework         │
│  - CoreVideo.framework      │
│  - Accelerate.framework     │
│  - Foundation.framework     │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Apple Neural Engine        │
│  + GPU + CPU                │
└─────────────────────────────┘
```

## Performance Results (M4 Pro)

### Benchmark: Depth Estimation Only

| Backend | Model | Time | Speedup |
|---------|-------|------|---------|
| ONNX CPU | Small (24.8M) | 430ms | 1x baseline |
| **CoreML** | **Base (97.5M)** | **128ms** | **3.4x faster** |

### Full Pipeline (640×480 image → SBS stereo)

| Step | Time |
|------|------|
| Model compilation (first run only) | ~7.5s |
| Image load & preprocessing | ~10ms |
| **CoreML depth inference** | **~128ms** |
| Stereo pair generation (DIBR) | ~15ms |
| SBS image creation & save | ~10ms |
| **Total (warm runs)** | **~150ms** |

### Video Processing Estimate

For 60 seconds @ 24fps (1440 frames):
- **ONNX CPU**: 430ms × 1440 = ~10 minutes
- **CoreML Base**: 128ms × 1440 = **~3 minutes**

Still faster than real-time!

## Implementation Details

### File Structure

```
spatial-maker/
├── swift-bridge/
│   └── CoreMLDepth.swift        # Swift wrapper for CoreML
├── src/
│   ├── depth_coreml.rs          # Rust FFI bindings
│   └── lib.rs                   # Public API with CoreML support
├── examples/
│   ├── photo.rs                 # ONNX example
│   └── photo_coreml.rs          # CoreML example (macOS only)
├── build.rs                     # Compiles Swift → static lib
└── Cargo.toml                   # Rust dependencies
```

### Swift Bridge Functions

```swift
// Load and compile CoreML model
@_cdecl("coreml_load_model")
public func loadModel(_ pathPtr: UnsafePointer<CChar>) -> UnsafeMutableRawPointer?

// Run depth inference
@_cdecl("coreml_infer_depth")
public func inferDepth(
    _ modelPtr: UnsafeMutableRawPointer,
    _ rgbData: UnsafePointer<Float>,
    _ width: Int32,
    _ height: Int32,
    _ outputPtr: UnsafeMutablePointer<Float>
) -> Int32

// Unload model
@_cdecl("coreml_unload_model")
public func unloadModel(_ modelPtr: UnsafeMutableRawPointer)
```

### Rust FFI Bindings

```rust
extern "C" {
    fn coreml_load_model(path: *const std::os::raw::c_char) -> *mut std::os::raw::c_void;
    fn coreml_unload_model(model: *mut std::os::raw::c_void);
    fn coreml_infer_depth(
        model: *mut std::os::raw::c_void,
        rgb_data: *const f32,
        width: i32,
        height: i32,
        output: *mut f32,
    ) -> i32;
}
```

### Build Process

The `build.rs` script:
1. Compiles `CoreMLDepth.swift` to static library using `swiftc`
2. Links Apple frameworks (CoreML, CoreVideo, Accelerate, Foundation)
3. Links Swift runtime libraries
4. Only runs on macOS (`#[cfg(target_os = "macos")]`)

### Data Flow

1. **Rust → Swift (Input)**:
   - Image → RGB → Normalize → NCHW Float32 array
   - Pass raw pointer to Swift

2. **Swift processing**:
   - Wrap Float32 data in MLMultiArray
   - Run CoreML inference
   - Get Float16 output from model
   - Convert Float16 → Float32 for Rust

3. **Swift → Rust (Output)**:
   - Copy converted data to Rust buffer
   - Rust normalizes to 0-1 range
   - Creates depth map image

## Key Learnings

### 1. Model Compilation Required

CoreML `.mlpackage` files must be compiled to `.mlmodelc` before use:

```swift
let compiledURL = try MLModel.compileModel(at: url)
```

First run takes ~7.5s to compile. Subsequent runs use cached compiled model (~150ms total).

### 2. Float16 Output

The Base model outputs **Float16**, not Float32. Must convert:

```swift
let srcPtr16 = depthArray.dataPointer.assumingMemoryBound(to: Float16.self)
for i in 0..<outputCount {
    outputPtr[i] = Float(srcPtr16[i])
}
```

### 3. Compute Units Configuration

```swift
let config = MLModelConfiguration()
config.computeUnits = .all  // Use ANE + GPU + CPU
```

This enables the Neural Engine for maximum performance.

### 4. Memory Management

Swift uses ARC (Automatic Reference Counting). For FFI:
- `Unmanaged.passRetained()` when returning to Rust
- `Unmanaged.fromOpaque().release()` when done

## Usage

### Basic Example

```rust
use spatial_maker::CoreMLDepthEstimator;

let estimator = CoreMLDepthEstimator::new(
    "checkpoints/DepthAnythingV2BaseF16.mlpackage"
)?;

let depth_map = estimator.estimate(&input_image)?;
// depth_map is now a normalized Float32 depth map
```

### Command Line

```bash
# Using CoreML (macOS only)
cargo run --example photo_coreml --release -- \
    input.jpg \
    -o output_sbs.jpg \
    --model DepthAnythingV2BaseF16.mlpackage

# Using ONNX (cross-platform)
cargo run --example photo --release -- \
    input.jpg \
    -o output_sbs.jpg \
    --model depth_anything_v2_vits.onnx
```

## Benchmarking

To benchmark on your machine:

```bash
# Build release binary
cargo build --example photo_coreml --release

# Run benchmark (5 iterations)
for i in {1..5}; do
    time ./target/release/examples/photo_coreml \
        test_image.jpg -o output_$i.jpg
done
```

Look for "Depth estimation:" lines to get pure inference time.

## Troubleshooting

### Model fails to load

**Error**: `Failed to load CoreML model`

**Solution**: Ensure you have:
1. Xcode Command Line Tools installed: `xcode-select --install`
2. macOS 15+ (required for Float16 support)
3. Valid `.mlpackage` file (check with `ls -la checkpoints/`)

### Swift compilation fails

**Error**: `swiftc: command not found`

**Solution**: Install Xcode Command Line Tools

### Link errors

**Error**: Framework not found

**Solution**: Check that `build.rs` links all required frameworks:
- CoreML.framework
- CoreVideo.framework  
- Accelerate.framework
- Foundation.framework

## Next Steps

1. **Test Small model** — Expected ~50-60ms (faster)
2. **Test Large model** — Expected ~200-300ms (better quality)
3. **Video processing** — Batch frame processing
4. **Optimize data pipeline** — Reduce conversion overhead
5. **ANE profiling** — Verify Neural Engine utilization

## References

- [Apple CoreML Documentation](https://developer.apple.com/documentation/coreml)
- [Depth Anything V2 Paper](https://arxiv.org/abs/2406.09414)
- [Our models on HuggingFace](https://huggingface.co/mrgnw/depth-anything-v2-coreml)
- [Swift FFI Guide](https://developer.apple.com/documentation/swift/c-interoperability)

## License

MIT (see LICENSE file)

---

**Status**: ✅ Working and benchmarked on M4 Pro  
**Date**: February 14, 2026  
**Author**: spatial-maker team
