# Spatial Video Maker

Convert 2D videos to stereoscopic 3D spatial videos for Apple Vision Pro using AI depth estimation. Runs locally on macOS with Apple Silicon Metal acceleration.

## Installation

```bash
uv tool install spatial-maker
```

### Requirements

- macOS with Apple Silicon (for Metal acceleration)
- [ffmpeg](https://ffmpeg.org/) with VideoToolbox support
- [spatial CLI](https://blog.mikeswanson.com/spatial_docs/) for MV-HEVC output

```bash
brew install ffmpeg spatial
```

### Model Checkpoints

Download Depth Anything V2 checkpoints:

```bash
mkdir -p ~/.spatial-maker/checkpoints

# Small (fastest, 24.8M params)
curl -L -o ~/.spatial-maker/checkpoints/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth

# Base (balanced, 97.5M params)
curl -L -o ~/.spatial-maker/checkpoints/depth_anything_v2_vitb.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth

# Large (best quality, 335.3M params)
curl -L -o ~/.spatial-maker/checkpoints/depth_anything_v2_vitl.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
```

You only need the model sizes you plan to use. `vits` (small) is a good default.

## Usage

```bash
# Single file
spatial-maker video.mp4

# Specify output path
spatial-maker video.mp4 -o output_spatial.mov

# Process all videos in a folder
spatial-maker /path/to/videos/
```

### Options

```bash
# Depth model size: vits (fast), vitb (balanced), vitl (best)
spatial-maker video.mp4 --encoder vitl

# 3D intensity in pixels (20-50 recommended)
spatial-maker video.mp4 --max-disparity 40

# Process only first 10 seconds (for testing)
spatial-maker video.mp4 --duration 10

# Skip downscaling (if input is already 1080p@24fps)
spatial-maker video.mp4 --skip-downscale

# Keep intermediate files
spatial-maker video.mp4 --keep-intermediate

# JSON progress output (for GUI integration)
spatial-maker video.mp4 --json-progress
```

## Pipeline

```
Input 2D Video
    |
[Downscale to 1080p @ 24fps]
    |
[Depth estimation + stereo pair creation (Depth Anything V2 + DIBR)]
    |
[Audio extraction + muxing]
    |
[MV-HEVC spatial video via spatial CLI]
    |
Spatial Video (.mov)
```

## Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--encoder` | Depth model size (vits/vitb/vitl) | `vits` for speed, `vitl` for quality |
| `--max-disparity` | 3D intensity in pixels | 20-50 for 1080p |
| `--duration` | Limit processing time (seconds) | Use for testing |
| `--skip-downscale` | Keep original resolution | Only if already 1080p |
| `--json-progress` | Emit JSON progress to stdout | For GUI integration |

### 3D Intensity Guide

- **20-30**: Subtle, comfortable for long viewing
- **30-40**: Moderate, good balance
- **40-50**: Strong, more immersive

## Development

```bash
git clone https://github.com/mrgnw/spatial-maker.git
cd spatial-maker
uv sync
uv run spatial-maker video.mp4
```

## Acknowledgments

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) by Yang et al.
- [spatial](https://blog.mikeswanson.com/spatial_docs/) by Mike Swanson

## License

MIT
