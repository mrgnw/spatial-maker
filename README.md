# Project Objective & Next Steps

## Primary Objective

**Convert 2D videos to stereoscopic 3D (side-by-side or spatial video format) for viewing on VR headsets (Apple Vision Pro / Meta Quest) using local, on-device processing with Apple Silicon Metal acceleration.**

We will be using the [spatial cli](https://blog.mikeswanson.com/spatial_docs/) to generate spatial videos from side-by-side ("sbs") stereoscopic videos.

## Current Status: investigating solutions

We are investigating better tools for faster depth estimation. If we can output a depth map or side-by-side video, we can then use that to output a spatial video.

We tried using DA3 (Depth Anything V3) integration with Apple Metal (MPS) support - but it was way too slow (10 seconds for 3 frames = ~3.3 seconds/frame)

We want to do a minimal test on 2-3 solutions to see what would be fastest. We will start with examples in the samples/1080p24fps folder. (If nothing is there yet, we can use ffmpeg to convert samples/4k/barcelona-lights.MOV to 1080p24fps)

### implementation preferences

If we end up making a script that involves using a model, it would be nice to pull from huggingface it will be easy to manage the models and potentially switch to a different model (in the future - the model choice can be hardcoded).

Many solutions seem to utilize python. If we use python, use UV [to manage the dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/)


### Better Tools to Investigate

We want to create a sample for some of these and compare the time it takes to generate them, and save the output so we can compare the quality.

#### 1. Apple Depth Pro ⭐ (Highest Priority)
**Why:** Made by Apple, designed for speed, metric depth in <1 second

- **Speed:** 0.3 seconds for 2.25MP image on GPU
- **Quality:** High-resolution, sharp boundaries
- **Open Source:** [GitHub - apple/ml-depth-pro](https://github.com/apple/ml-depth-pro)
- **Model:** [Hugging Face - apple/DepthPro](https://huggingface.co/apple/DepthPro)

**Sources:**
- [Apple ML Research - Depth Pro](https://machinelearning.apple.com/research/depth-pro)
- [GitHub Repository](https://github.com/apple/ml-depth-pro)
- [VentureBeat Article](https://venturebeat.com/ai/apple-releases-depth-pro-an-ai-model-that-rewrites-the-rules-of-3d-vision)
### 2. Video Depth Anything (CVPR 2025)
**Why:** Specifically designed for consistent depth in long videos

- **Speed:** Faster than diffusion models
- **Consistency:** Temporal consistency for video
- **Open Source:** [GitHub - DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)

### 3. Other projects to consider after testing the above

- Projects that support HD and perform well in [these benchmarks](https://raw.githubusercontent.com/AIVFI/Video-Depth-Estimation-Rankings-and-2D-to-3D-Video-Conversion-Rankings/refs/heads/main/README.md)
- [2D-to-3D-Video-Converter](https://github.com/SreeHarhsa/2D-to-3D-Video-Converter-AI-Powered-Depth-Stereo-Effect) - GPU acceleration, batch processing


## Anticipated Output Pipeline

```
Input 2D Video 

[create a downscaled copy to 1080p24fps, at least for now. Ideally this step takes full advantage of apple silicon as well and retains quality (aside fitting the dimensions we set)]
    ↓
[Generate depth for video*]
    ↓
[Create stereoscopic version (sbs)]
    ↓
[Create spatial version (`spatial make`)]
    ↓
MV-HEVC Spatial Video (Vision Pro/Quest compatible)

*ideally we can generate depth for video directly as a video, but if we need to split into frames, get the depth, then stitch them back together, that is an option. It's just not likely to be as efficient, so we aren't prioritizing solutions that require that.
```

## Apple's Documentation on Spatial Video

We are going to use the spatial cli for the final conversion, so don't read these docs unless I specifically ask you to. But if it helps understand the final format, this is Apple's documentation:
- [Converting side-by-side 3D video to multiview HEVC and spatial video](https://developer.apple.com/documentation/AVFoundation/converting-side-by-side-3d-video-to-multiview-hevc-and-spatial-video)
- [https://developer.apple.com/documentation/imageio/creating-spatial-photos-and-videos-with-spatial-metadata](https://developer.apple.com/documentation/imageio/creating-spatial-photos-and-videos-with-spatial-metadata)
- [Writing-spatial-photos](https://developer.apple.com/documentation/ImageIO/writing-spatial-photos)


# Instructions added after starting

- I would like to keep our code clean and modular - each approach can be its own module, and the benchmark script will run and time each approach. So for now we'll have our apple depth module, and the benchmark module will import that and benchmark it.
- our benchmark can specify a number of seconds to sample, defaulting to 3 seconds
