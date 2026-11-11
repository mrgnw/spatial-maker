# Roadmap

spatial-maker is a Rust and Swift tool that converts photos and videos to spatial 3D for Apple Vision Pro. Work is tracked in [GitHub issues](https://github.com/mrgnw/spatial-maker/issues); each one below is scoped to a few hours and has acceptance criteria.

## Quality of the 3D output

- [#23](https://github.com/mrgnw/spatial-maker/issues/23) Vertically coherent fill for wide disocclusion holes
- [#42](https://github.com/mrgnw/spatial-maker/issues/42) Property tests for the stereo renderer
- [#44](https://github.com/mrgnw/spatial-maker/issues/44) Verify command that reads back stereo HEIC and MV-HEVC output

## Testing and CI

- [#37](https://github.com/mrgnw/spatial-maker/issues/37) Build and test on macOS for every push and pull request
- [#38](https://github.com/mrgnw/spatial-maker/issues/38) rustfmt and clippy gates
- [#40](https://github.com/mrgnw/spatial-maker/issues/40) Injectable depth source so the pipeline runs without a model
- [#41](https://github.com/mrgnw/spatial-maker/issues/41) End-to-end video test on a generated clip
- [#48](https://github.com/mrgnw/spatial-maker/issues/48) Build and test the library on Linux

## Usability and releases

- [#39](https://github.com/mrgnw/spatial-maker/issues/39) Release workflow with a macOS tarball, crates.io refresh
- [#43](https://github.com/mrgnw/spatial-maker/issues/43) Directory input and skipping finished outputs
- [#45](https://github.com/mrgnw/spatial-maker/issues/45) Checksummed, resumable model downloads

## Docs

- [#46](https://github.com/mrgnw/spatial-maker/issues/46) One architecture page, remove stale notes
- [#47](https://github.com/mrgnw/spatial-maker/issues/47) README gallery on freely licensed photos

## Known unknowns

- Nothing has been checked on a Vision Pro yet; [#31](https://github.com/mrgnw/spatial-maker/issues/31) tracks what shipped in v0.7.0 and what is still open.
