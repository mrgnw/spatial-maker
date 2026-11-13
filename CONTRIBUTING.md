# Contributing

Start with an [open issue](https://github.com/mrgnw/spatial-maker/issues), especially one marked [good first issue](https://github.com/mrgnw/spatial-maker/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22). For a larger change, describe the proposed approach in an issue first.

## Build and test

Use an Apple Silicon Mac with macOS 14+, Rust 1.88+, Xcode Command Line Tools, and ffmpeg for video. See the [README](README.md#requirements) for setup.

```sh
git clone https://github.com/mrgnw/spatial-maker.git
cd spatial-maker
```

```sh
cargo build --release
```

```sh
cargo test
```

CI runs the release build and tests on `macos-latest` because the default build links the Swift bridge and Apple frameworks. The unit tests do not need downloaded model weights. A green CI run does not verify conversion quality or playback on a Vision Pro.

## Pull requests

Keep each pull request focused. Use tabs, follow the surrounding Rust and Swift style, and add a regression test for a behavior change. Include the commands you ran and their results. For output changes, include a small before/after preview and the model and options used.

Share only media you have permission to redistribute. Credit its source and license, remove GPS metadata, and keep model weights and large conversion outputs out of Git. Do not attach private photos to issues.

Code contributions are covered by the project's [MIT license](LICENSE). Model weights retain their [separate licenses](README.md#models-and-licenses).
