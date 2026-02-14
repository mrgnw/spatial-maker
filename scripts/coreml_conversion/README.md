# CoreML Conversion Scripts

This directory contains scripts to convert Depth Anything V2 PyTorch models to CoreML format for optimized inference on Apple Silicon.

## Quick Start

```bash
# 1. Download PyTorch checkpoints (if not already present)
cd checkpoints
curl -L -o depth_anything_v2_vitb.pth \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth"
curl -L -o depth_anything_v2_vitl.pth \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"

# 2. Convert to CoreML
cd ..
uv run --with coremltools --with torch --with jkp-depth-anything-v2 \
  python scripts/coreml_conversion/convert_to_coreml.py
```

## What Gets Created

The script generates CoreML models in `checkpoints/`:

- `DepthAnythingV2BaseF16.mlpackage` (186 MB) — Base model, Float16
- `DepthAnythingV2LargeF16.mlpackage` (638 MB) — Large model, Float16

These can be used directly with CoreML on macOS 15+ / iOS 18+.

## Technical Details

### Conversion Process

1. Loads PyTorch checkpoint (`.pth`)
2. Traces model with `torch.jit.trace` using dummy input
3. Converts to CoreML using `coremltools.convert()`
4. Sets precision to Float16 for optimal ANE performance
5. Adds metadata (author, license, descriptions)
6. Saves as `.mlpackage` format

### Model Configuration

**Base (vitb):**
- Encoder: ViT-B
- Features: 128
- Out channels: [96, 192, 384, 768]
- Parameters: 97.5M
- License: Apache-2.0

**Large (vitl):**
- Encoder: ViT-L
- Features: 256
- Out channels: [256, 512, 1024, 1024]
- Parameters: 335.3M
- License: CC-BY-NC-4.0

### Requirements

- macOS (CoreML conversion requires macOS)
- Python 3.10+
- `coremltools` — Apple's CoreML conversion library
- `torch` — PyTorch
- `jkp-depth-anything-v2` — Depth Anything V2 model architecture

All dependencies are automatically managed by `uv run`.

## Why CoreML?

CoreML models on Apple Silicon:
- Use the **Neural Engine** (ANE) for massive acceleration
- Are **10-20x faster** than ONNX CPU inference
- Are **pre-compiled** — no runtime compilation overhead
- Support Float16 precision natively on ANE

Expected performance on M4 Pro:
- Base: ~60-90ms per frame (~11-16 fps)
- Large: ~200-300ms per frame (~3-5 fps)

Compare to ONNX CPU:
- Small: 430ms
- Base: ~1-2s (estimated)
