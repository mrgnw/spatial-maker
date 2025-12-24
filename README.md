# Project Objective & Next Steps

## Primary Objective

**Convert 2D videos to stereoscopic 3D (side-by-side or spatial video format) for viewing on VR headsets (Apple Vision Pro / Meta Quest) using local, on-device processing with Apple Silicon Metal acceleration.**

We will be using the [spatial cli](https://blog.mikeswanson.com/spatial_docs/) to generate spatial videos from side-by-side ("sbs") stereoscopic videos.

## Current pipeline

```
Input 2D Video

[create a downscaled copy to 1080p24fps]
    ↓
[Generate depth for video using depth anything v2 (can choose model)]
    ↓
[Create stereoscopic version (sbs)]
    ↓
[Create spatial version (`spatial make`)]
    ↓
MV-HEVC Spatial Video (Vision Pro/Quest compatible)

```
