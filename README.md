# spatial-maker

Turn ordinary photos and videos into spatial 3D for Apple Vision Pro. It estimates a depth map for each frame, renders a second eye with depth-image-based rendering (DIBR) and writes the result as a spatial HEIC photo or an MV-HEVC video.

The work in this repository is the media pipeline: stereo rendering and hole filling, edge refinement, temporal smoothing of video depth, and native HEIC and MV-HEVC writers in Swift. Depth comes from pretrained, off-the-shelf models that run locally through Apple's CoreML. This project uses those models; it does not train, tune or serve them.

## Who it is for

People with a Vision Pro (or any spatial-photo viewer) who want their existing photos and clips in 3D without a cloud service, and Rust developers who want a library for 2D-to-stereo conversion.

## Built with

- Rust: CLI, library, photo and video pipeline
- Swift bridge: CoreML, ImageIO (stereo HEIC), AVFoundation and VideoToolbox (MV-HEVC)
- Pretrained depth models: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), Depth Anything 3, Video Depth Anything
- ffmpeg for frame extraction and audio

## Requirements

- macOS 14+ on Apple Silicon
- Xcode Command Line Tools (the build compiles the Swift bridge)
- Rust 1.75+
- ffmpeg, for video

```sh
xcode-select --install
brew install ffmpeg rust
```

## Install and run

```sh
git clone https://github.com/mrgnw/spatial-maker.git
cd spatial-maker
cargo build --release
```

Set the input, then run the line unchanged. Models download on first use (about 48 MB for the default video model, 638 MB for the default photo model).

```sh
INPUT=$HOME/Pictures/photo.jpg
./target/release/spatial-maker "$INPUT"
```

```sh
INPUT=$HOME/Movies/clip.mp4
OUTPUT=$HOME/Movies/clip_spatial.mov
./target/release/spatial-maker "$INPUT" -o "$OUTPUT"
```

Videos use `-m s` by default. `-m vda` (steadier video depth) needs a model that is not publicly downloadable yet; see Models and licenses.

Run the tests:

```sh
cargo test
```

## Options

| Flag | Meaning |
|-|-|
| `-m s`, `b`, `l`, `da3`, `vda` | Depth model. Photos default to `da3`, videos to `s` |
| `--max-disparity` | 3D strength in pixels. Default is 1.2% of image width |
| `--output-types` | `spatial` (default), `sbs`, `tab`, `sep`, `depth` |
| `--heic-quality` | Spatial photo quality, default 80 |
| `--temporal-alpha` | Video depth smoothing, default 0.7 |

Full list:

```sh
./target/release/spatial-maker --help
```

## Library

```toml
[dependencies]
spatial-maker = { git = "https://github.com/mrgnw/spatial-maker" }
```

See `examples/` for runnable photo and stereo-from-depth programs.

## Models and licenses

spatial-maker is MIT. The depth models have their own terms, taken from the upstream model cards and repositories ([Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2#license), [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything#license), [DA3MONO-LARGE](https://huggingface.co/depth-anything/DA3MONO-LARGE)):

| Model | License |
|-|-|
| Depth Anything V2 Small (`s`) | Apache-2.0 |
| Depth Anything V2 Base and Large (`b`, `l`) | CC BY-NC 4.0, non-commercial |
| Depth Anything 3 Mono Large (`da3`) | Apache-2.0 |
| Video Depth Anything Small (`vda`) | Apache-2.0 |

The MIT license covers this repository's code only. It does not relicense any model weights.

Where the weights come from:

- This repository contains no model weights, and the release tarballs contain only the `spatial-maker` binary.
- Weights are downloaded at runtime, on first use, by the tool running on your machine.
- The downloaded files are CoreML conversions that the author hosts on Hugging Face: [mrgnw/depth-anything-v2-coreml](https://huggingface.co/mrgnw/depth-anything-v2-coreml) (V2 Small, Base, Large) and [mrgnw/depth-anything-v3-coreml](https://huggingface.co/mrgnw/depth-anything-v3-coreml) (DA3 Mono Large). They are redistributed there, outside this repository. The conversion scripts are in `scripts/coreml_conversion/`.
- The `vda` download points at `mrgnw/video-depth-anything-coreml`, which is not publicly available yet, so `-m vda` only works if you already have that model locally.
- Base and Large are non-commercial: check that your use fits CC BY-NC 4.0 before choosing `-m b` or `-m l`.

## Roadmap

See [ROADMAP.md](ROADMAP.md) and the [open issues](https://github.com/mrgnw/spatial-maker/issues).

## Acknowledgments

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Mike Swanson's spatial](https://blog.mikeswanson.com/spatial_docs/) for MV-HEVC groundwork

## License

[MIT](LICENSE)
