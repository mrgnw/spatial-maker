# Roadmap

spatial-maker is a Rust and Swift tool that converts photos and videos to spatial 3D for Apple Vision Pro. Work is tracked in [GitHub issues](https://github.com/mrgnw/spatial-maker/issues); each one below is scoped to a few hours and has acceptance criteria.

## Quality of the 3D output

- [#1](https://github.com/mrgnw/spatial-maker/issues/1) Fill for holes wider than the existing vertical smoothing handles
- [#8](https://github.com/mrgnw/spatial-maker/issues/8) Property tests for the stereo renderer
- [#10](https://github.com/mrgnw/spatial-maker/issues/10) Verify command that reads back stereo HEIC and MV-HEVC output

## Testing and CI

- [#4](https://github.com/mrgnw/spatial-maker/issues/4) rustfmt and clippy gates
- [#6](https://github.com/mrgnw/spatial-maker/issues/6) Injectable depth source so the pipeline runs without a model
- [#7](https://github.com/mrgnw/spatial-maker/issues/7) End-to-end video test on a generated clip
- [#14](https://github.com/mrgnw/spatial-maker/issues/14) Build and test the library on Linux

## Usability and releases

- [#5](https://github.com/mrgnw/spatial-maker/issues/5) Release workflow with a macOS tarball, crates.io refresh
- [#9](https://github.com/mrgnw/spatial-maker/issues/9) Directory input and skipping finished outputs
- [#11](https://github.com/mrgnw/spatial-maker/issues/11) Checksummed, resumable model downloads

## Docs

- [#12](https://github.com/mrgnw/spatial-maker/issues/12) One architecture page, remove stale notes
- [#13](https://github.com/mrgnw/spatial-maker/issues/13) README gallery on freely licensed photos

## Known unknowns

- Nothing has been checked on a Vision Pro yet; [#2](https://github.com/mrgnw/spatial-maker/issues/2) tracks the open items left after v0.8.0 (Vision Pro check, the `vda` model download, fur edges).
