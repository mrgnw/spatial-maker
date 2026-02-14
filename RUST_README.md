# spatial-maker

Convert 2D images and videos to stereoscopic 3D spatial content for Apple Vision Pro using AI depth estimation.

[![Crates.io](https://img.shields.io/crates/v/spatial-maker.svg)](https://crates.io/crates/spatial-maker)
[![Documentation](https://docs.rs/spatial-maker/badge.svg)](https://docs.rs/spatial-maker)

## Features

- **Fast depth estimation** — CoreML on Apple Silicon (128ms/frame on M4 Pro) with ONNX fallback
- **High-quality stereo generation** — Depth-Image Based Rendering (DIBR) with hole filling
- **Photo & video support** — Process single images or full videos with progress callbacks
- **Multi-format input** — JPEG, PNG, AVIF, JPEG XL, HEIC via native decoders or ffmpeg
- **MV-HEVC output** — Side-by-side, top-and-bottom, or separate stereo pairs with optional MV-HEVC packaging

## 🚧 massively under construction 🧱

## Quick Start

### Install the CLI

```sh
cargo install spatial-maker
```

for the latest code (untested)

```sh
cargo install --git https://github.com/mrgnw/spatial-maker --force
```



### Convert a Video

```bash
INPUT_VIDEO=~/Movies/my-video.mp4
MODEL=b  # s (small), b (base), or l (large)

spatial-maker video "$INPUT_VIDEO" -o "$OUTPUT_VIDEO" --model "$MODEL"
```

### Convert a Photo

```bash
INPUT_PHOTO=~/Pictures/photo.jpg
MODEL=b
MAX_DISPARITY=30  # Higher = more 3D depth

spatial-maker photo "$INPUT_PHOTO"
  --model "$MODEL" \
  --max-disparity "$MAX_DISPARITY"
```

### Use as a Library

Add to your `Cargo.toml`:

```toml
[dependencies]
spatial-maker = "0.1"
```

Process a photo:

```rust
use spatial_maker::{process_photo, SpatialConfig, OutputOptions};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input_photo = Path::new("input.jpg");
    let output_photo = Path::new("output_sbs.jpg");
    
    let config = SpatialConfig::default(); // Uses Small model (48MB, auto-downloaded)
    let output_opts = OutputOptions::default(); // Side-by-side JPEG

    process_photo(input_photo, output_photo, config, output_opts).await?;

    Ok(())
}
```

Process a video:

```rust
use spatial_maker::{process_video, SpatialConfig, VideoProgress};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input_video = Path::new("input.mp4");
    let output_video = Path::new("output_sbs.mp4");
    let config = SpatialConfig::default();
    
    process_video(
        input_video,
        output_video,
        config,
        Some(Box::new(|progress: VideoProgress| {
            println!("{}: {:.1}%", progress.stage, progress.percent);
        })),
    ).await?;

    Ok(())
}
```

## Model Sizes

Models are auto-downloaded from [HuggingFace](https://huggingface.co/mrgnw/depth-anything-v2-coreml) on first use:

| Size | Download | Quality | Speed (M4 Pro) | License |
|------|----------|---------|----------------|---------|
| **Small** (default) | 48 MB | Good | ~85ms/frame | Apache-2.0 ✅ |
| **Base** | 186 MB | Better | ~128ms/frame | CC-BY-NC-4.0 ⚠️ |
| **Large** | 638 MB | Best | ~190ms/frame | CC-BY-NC-4.0 ⚠️ |

⚠️ Base and Large models are **non-commercial only** (CC-BY-NC-4.0). Small is Apache-2.0 (commercial OK).

Select a model (CLI):

```bash
MODEL=b  # s (small, 48MB), b (base, 186MB), l (large, 638MB)
INPUT=~/Movies/video.mp4
OUTPUT=~/Desktop/spatial.mp4

spatial-maker video "$INPUT" -o "$OUTPUT" --model "$MODEL"
```

Select a model (library):

```rust
let model_size = "b".to_string(); // "s", "b", or "l"
let max_disparity = 30;

let config = SpatialConfig {
    encoder_size: model_size,
    max_disparity,
    ..Default::default()
};
```

## Feature Flags

```toml
[dependencies]
spatial-maker = { version = "0.1", features = ["onnx"] }
```

| Feature | Default | Description |
|---------|---------|-------------|
| `coreml` | ✅ | CoreML via Swift FFI (macOS only, ~3x faster than ONNX CPU) |
| `onnx` | ❌ | ONNX Runtime fallback (cross-platform) |
| `avif` | ❌ | Native AVIF decoder (requires system libdav1d) |
| `jxl` | ❌ | Native JPEG XL decoder (pure Rust via jxl-oxide) |
| `heic` | ❌ | Native HEIC decoder (requires system libheif) |

Formats not enabled via feature flags fall back to ffmpeg conversion (slower but works for everything).

## How It Works

1. **Depth Estimation**: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) via CoreML (default) or ONNX
2. **Preprocessing**: Resize to 518×518, normalize with ImageNet stats, NCHW tensor format
3. **Stereo Generation**: DIBR shifts pixels based on depth; expanding-ring hole filling for disocclusions
4. **Video Pipeline**: `ffmpeg` frame extraction → depth+stereo (parallelized) → `ffmpeg` encoding with model caching

## API Overview

```rust
// High-level API (most common)
pub async fn process_photo(input, output, config, output_options) -> SpatialResult<()>
pub async fn process_video(input, output, config, progress_cb) -> SpatialResult<()>

// Low-level API (custom pipelines)
pub struct CoreMLDepthEstimator; // macOS CoreML backend
pub struct OnnxDepthEstimator;   // ONNX Runtime backend
pub fn generate_stereo_pair(image, depth, max_disparity) -> SpatialResult<(DynamicImage, DynamicImage)>
pub fn create_sbs_image(left, right) -> DynamicImage
```

See [docs.rs/spatial-maker](https://docs.rs/spatial-maker) for full API documentation.

## Requirements

### For CoreML (default, macOS only):
- macOS 13.0+ (Ventura or later)
- Apple Silicon recommended (falls back to CPU on Intel)
- Xcode Command Line Tools (for Swift compilation during build)

### For ONNX (optional, cross-platform):
- Any OS (Linux, Windows, macOS)
- Binaries auto-downloaded via `ort` crate

### For Video:
- `ffmpeg` in PATH — [https://ffmpeg.org](https://ffmpeg.org)
- `spatial` CLI for MV-HEVC output — [https://blog.mikeswanson.com/spatial](https://blog.mikeswanson.com/spatial)

```bash
brew install ffmpeg spatial
```

## License

MIT

Models: Small (Apache-2.0), Base/Large (CC-BY-NC-4.0). See [HuggingFace repo](https://huggingface.co/mrgnw/depth-anything-v2-coreml) for details.
