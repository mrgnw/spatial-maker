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

Set the input, then run the line unchanged. Models download on first use.

```sh
INPUT=$HOME/Pictures/photo.jpg
./target/release/spatial-maker "$INPUT"
```

```sh
INPUT=$HOME/Movies/clip.mp4
OUTPUT=$HOME/Movies/clip_spatial.mov
./target/release/spatial-maker "$INPUT" -o "$OUTPUT" -m vda
```

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

spatial-maker is MIT. The depth models have their own terms:

| Model | License |
|-|-|
| Depth Anything V2 Small (`s`) | Apache-2.0 |
| Depth Anything V2 Base and Large (`b`, `l`) | CC BY-NC 4.0, non-commercial |
| Depth Anything 3 Mono Large (`da3`) | Apache-2.0 |
| Video Depth Anything Small (`vda`) | Apache-2.0 |

## Roadmap

See [ROADMAP.md](ROADMAP.md) and the [open issues](https://github.com/mrgnw/spatial-maker/issues).

## Acknowledgments

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Mike Swanson's spatial](https://blog.mikeswanson.com/spatial_docs/) for MV-HEVC groundwork

## License

[MIT](LICENSE)
