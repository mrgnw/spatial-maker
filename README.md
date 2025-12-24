# Spatial Video Maker

Convert 2D videos to stereoscopic 3D spatial videos for viewing on VR headsets (Apple Vision Pro / Meta Quest) using local, on-device processing with Apple Silicon Metal acceleration.

## Pipeline Overview

```
Input 2D Video
    ↓
[Downscale to 1080p @ 24fps]
    ↓
[Generate depth maps using Depth Anything V2]
    ↓
[Create side-by-side (SBS) stereoscopic video]
    ↓
[Convert to MV-HEVC spatial video using `spatial` CLI]
    ↓
MV-HEVC Spatial Video (Vision Pro/Quest compatible)
```

## Quick Start

### Basic Usage

```bash
# Run the full pipeline
python pipeline.py input_video.mp4

# Specify output path
python pipeline.py input_video.mp4 -o output_spatial.mov
```

### Options

```bash
# Use a larger depth model for better quality (vits, vitb, vitl)
python pipeline.py input_video.mp4 --encoder vitl

# Adjust 3D intensity (20-50 recommended for 1080p)
python pipeline.py input_video.mp4 --max-disparity 40

# Process only the first 10 seconds (useful for testing)
python pipeline.py input_video.mp4 --duration 10

# Keep intermediate files for debugging
python pipeline.py input_video.mp4 --keep-intermediate

# Skip downscaling if input is already 1080p @ 24fps
python pipeline.py input_video.mp4 --skip-downscale
```

## Requirements

### Dependencies

```bash
# Install Python dependencies
uv sync
```

### External Tools

- **ffmpeg** with VideoToolbox support (for hardware-accelerated encoding)
- **[spatial CLI](https://blog.mikeswanson.com/spatial_docs/)** for MV-HEVC spatial video creation

Install spatial CLI:
```bash
brew install spatial
```

### Depth Model Checkpoints

Download Depth Anything V2 checkpoints to `Depth-Anything-V2/checkpoints/`:
- `depth_anything_v2_vits.pth` (small, fastest)
- `depth_anything_v2_vitb.pth` (base, balanced)
- `depth_anything_v2_vitl.pth` (large, best quality)

## Individual Scripts

### Depth Estimation

```bash
# Generate depth maps for a video
python -c "
from depth_estimators import DepthAnythingV2Estimator
estimator = DepthAnythingV2Estimator(encoder='vits')
result = estimator.process_video('input.mp4', 'output/depth_frames')
"
```

### Stereo Conversion

```bash
# Convert 2D video + depth frames to SBS stereoscopic
python scripts/stereo_converter.py input.mp4 output/depth_frames output_sbs.mp4 --max-disparity 30
```

### Downscaling

```bash
# Downscale to 1080p @ 24fps
python scripts/downscale_video.py input.mp4 output_1080p.mp4
```

## Parameters Guide

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--encoder` | Depth model size (vits/vitb/vitl) | `vits` for speed, `vitl` for quality |
| `--max-disparity` | 3D intensity in pixels | 20-50 for 1080p |
| `--duration` | Limit processing time | Use for testing |

### Disparity Guidelines

- **20-30**: Subtle 3D effect, comfortable for long viewing
- **30-40**: Moderate 3D effect, good balance
- **40-50**: Strong 3D effect, more immersive but may cause discomfort

## Output Formats

- **SBS (Side-by-Side)**: Half-width format, compatible with most VR players
- **MV-HEVC**: Apple's spatial video format for Vision Pro

## Project Structure

```
spatial-maker/
├── pipeline.py              # End-to-end conversion pipeline
├── depth_estimators/
│   ├── depth_anything_v2.py # Depth Anything V2 implementation
│   └── apple_depth_pro.py   # Apple Depth Pro implementation
├── scripts/
│   ├── stereo_converter.py  # SBS stereoscopic conversion
│   ├── downscale_video.py   # Video downscaling utility
│   └── stitch_video.py      # Frame stitching utility
├── Depth-Anything-V2/       # Depth Anything V2 submodule
│   └── checkpoints/         # Model checkpoints
├── samples/                 # Sample videos
└── output/                  # Generated outputs
```
