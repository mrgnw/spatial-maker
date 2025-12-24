# Spatial Video Maker

Convert 2D videos to stereoscopic 3D spatial videos for viewing on VR headsets (Apple Vision Pro / Meta Quest) using local, on-device processing with Apple Silicon Metal acceleration.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourname/spatial-maker.git
cd spatial-maker

# Install globally (adds spatial-maker to your PATH)
uv tool install .

# Or for development (run with: uv run spatial-maker)
uv sync
```

To update after making changes:
```bash
uv tool install . --force
```

To uninstall:
```bash
uv tool uninstall spatial-maker
```

### Requirements

- Python 3.10+
- macOS with Apple Silicon (for Metal acceleration)
- [ffmpeg](https://ffmpeg.org/) with VideoToolbox support
- [spatial CLI](https://blog.mikeswanson.com/spatial_docs/) for MV-HEVC output

```bash
# Install ffmpeg and spatial CLI
brew install ffmpeg spatial
```

### Model Checkpoints

Download Depth Anything V2 checkpoints to `checkpoints/`:

```bash
mkdir -p checkpoints
# Small (fastest, 24.8M params)
curl -L -o checkpoints/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
# Base (balanced, 97.5M params)
curl -L -o checkpoints/depth_anything_v2_vitb.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
# Large (best quality, 335.3M params)
curl -L -o checkpoints/depth_anything_v2_vitl.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
```

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

### Single File

```bash
# Using the CLI (if installed globally)
spatial-maker video.mp4

# Or with uv run
uv run spatial-maker video.mp4

# Specify output path
spatial-maker video.mp4 -o output_spatial.mov
```

### Batch Processing (Folder)

```bash
# Process all videos in a folder
spatial-maker /path/to/videos/

# Output goes to /path/to/videos/spatial/
# Supports: .mp4, .mov, .avi, .mkv, .m4v, .webm, .wmv, .flv
```

### Options

```bash
# Use a larger depth model for better quality (vits, vitb, vitl)
spatial-maker video.mp4 --encoder vitl

# Adjust 3D intensity (20-50 recommended for 1080p)
spatial-maker video.mp4 --max-disparity 40

# Process only the first 10 seconds (useful for testing)
spatial-maker video.mp4 --duration 10

# Keep intermediate files for debugging
spatial-maker video.mp4 --keep-intermediate

# Skip downscaling if input is already 1080p @ 24fps
spatial-maker video.mp4 --skip-downscale
```

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
├── pipeline.py              # End-to-end conversion pipeline (CLI entry point)
├── depth_estimators/
│   ├── depth_anything_v2.py # Depth Anything V2 implementation
│   └── apple_depth_pro.py   # Apple Depth Pro implementation
├── scripts/
│   ├── stereo_converter.py  # SBS stereoscopic conversion
│   ├── downscale_video.py   # Video downscaling utility
│   └── stitch_video.py      # Frame stitching utility
├── checkpoints/             # Model checkpoints (download separately)
├── samples/                 # Sample videos
└── output/                  # Generated outputs
```
