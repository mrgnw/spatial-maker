# spatial-maker (Rust)

Convert 2D images and videos to stereoscopic 3D spatial content for Apple Vision Pro using AI depth estimation.

This is a Rust library that uses **Depth Anything V2** via ONNX Runtime to estimate depth from 2D images and create side-by-side (SBS) stereo pairs for 3D viewing.

## Features

- ✅ **Fast depth estimation** using ONNX Runtime with CoreML acceleration on Apple Silicon
- ✅ **High quality stereo generation** using Depth-Image Based Rendering (DIBR)
- ✅ **Photo support** — convert single images to SBS stereo
- 🚧 **Video support** — coming soon (per-frame processing with optional temporal smoothing)
- 🚧 **Video Depth Anything integration** — temporally consistent depth for flicker-free video

## Quick Start

### Download the Model

Download the Depth Anything V2 Small ONNX model:

```bash
mkdir -p ~/.spatial-maker/checkpoints
curl -L -o ~/.spatial-maker/checkpoints/depth_anything_v2_vits.onnx \
  "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits.onnx"
```

### Usage (Photo Example)

```bash
cargo run --example photo --release -- input.jpg -o output_sbs.jpg --max-disparity 30
```

Or use as a library:

```rust
use spatial_maker::process_photo;
use image::open;

fn main() -> anyhow::Result<()> {
    let input = open("photo.jpg")?;
    let sbs = process_photo(
        input,
        "~/.spatial-maker/checkpoints/depth_anything_v2_vits.onnx",
        30.0  // max disparity in pixels
    )?;
    sbs.save("photo_sbs.jpg")?;
    Ok(())
}
```

## How It Works

1. **Depth Estimation**: Uses Depth Anything V2 (24.8M params) via ONNX Runtime to estimate per-pixel depth
2. **Preprocessing**: Resizes image to 518x518, normalizes with ImageNet mean/std
3. **Stereo Generation**: Uses DIBR to create left/right eye views by shifting pixels based on depth
4. **Output**: Horizontally stacks left and right views into a side-by-side image

## Model Info

- **Model**: [Depth Anything V2 Small](https://github.com/DepthAnything/Depth-Anything-V2)
- **ONNX Export**: [fabio-sim/Depth-Anything-ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX)
- **License**: Apache-2.0
- **Size**: 95 MB
- **Speed**: ~13ms on RTX 4080, faster on Apple Silicon with CoreML

## Roadmap

- [x] Photo support (SBS output)
- [x] ONNX model integration
- [x] Model checkpoint discovery
- [ ] CoreML execution provider testing
- [ ] Video frame processing
- [ ] Video Depth Anything integration (temporal consistency)
- [ ] Integration with Frame app (replace Python subprocess)

## Development

Built with:
- [ort](https://github.com/pykeio/ort) — ONNX Runtime bindings for Rust
- [image](https://github.com/image-rs/image) — Image encoding/decoding
- [ndarray](https://github.com/rust-ndarray/ndarray) — N-dimensional arrays

See [research/rust-depth.md](research/rust-depth.md) for detailed design decisions and model comparisons.

## License

MIT
