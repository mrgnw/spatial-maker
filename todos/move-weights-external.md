---
status: planned
branch: plan/move-weights-external
---

# Move model weights to /Volumes/r0

Internal disk is 97% full. Move all spatial-maker weights to `/Volumes/r0/models/spatial-maker/` and leave symlinks behind. Fail with a clear error when r0 is not mounted; never re-download onto the internal disk.

Surveyed 2026-09-25. Code refs are against local `main` at `cbc1c4b` (3 commits ahead of `origin/main`, not pushed).

## What is on disk now

| Path | Size | Used by |
|-|-|-|
| `~/.spatial-maker/checkpoints/DepthAnythingV2{Small,Base,Large}F16.mlpackage` | 47M + 186M + 638M | Rust CLI (`-m s/b/l`), `analysis/depth_bench.py` |
| `~/.spatial-maker/checkpoints/DepthAnythingV3Mono.mlpackage` | 638M | Rust CLI (`-m da3`) |
| `~/.spatial-maker/checkpoints/da3mono-coreml/` (mlpackage + README) | 638M | `depth_bench.py` only. Byte-identical duplicate of the one above. |
| `~/.spatial-maker/checkpoints/distill-any-depth/small/model.safetensors` | 95M | nothing. Byte-identical to repo `distill_any_depth_vits.safetensors`. |
| repo `checkpoints/depth_anything_v2_vit{s,b,l}.pth` | 95M + 372M + 1.2G | `scripts/coreml_conversion/convert_to_coreml.py` (conversion source) |
| repo `checkpoints/distill_any_depth_vits.safetensors` | 95M | `convert_to_coreml.py` |
| repo `checkpoints/DistillAnyDepthSmallF16.mlpackage` | 47M | `depth_bench.py` |
| repo `checkpoints/DepthAnythingV2SmallF16.mlpackage` | 47M | Rust dev-path fallback only. Byte-identical to the home copy. |
| repo `checkpoints/depth_pro.pt` | 1.8G | nothing. No reference in code, scripts, or justfile. |
| `~/.cache/huggingface` | 46M | nothing in this repo. Leave it. |

Stale: `depth_pro.pt`, `da3mono-coreml/`, `distill-any-depth/`, repo `DepthAnythingV2SmallF16.mlpackage`. About 2.5G of the 5.9G total.

## How the code finds weights

Rust (`src/model.rs`):

- `get_checkpoint_dir()`: `$SPATIAL_MAKER_CHECKPOINTS`, else `~/.spatial-maker/checkpoints`.
- `find_model()`: exact filename in that dir, then a fuzzy name match in `./checkpoints` (cwd-relative) and `~/.spatial-maker/checkpoints`.
- `ensure_model_exists()`: if `find_model` fails, `create_dir_all(checkpoint_dir)` and download from HuggingFace (up to 638M per model). Called by `src/main.rs`, `src/lib.rs`, `src/video.rs` on every run.

Python:

- `analysis/depth_bench.py`: hardcoded `~/.spatial-maker/checkpoints` and repo `checkpoints/`.
- `scripts/coreml_conversion/convert_to_coreml.py`: reads and writes repo `checkpoints/`. Never downloads; prints URLs if a source is missing.
- `justfile` upload recipe: reads repo `checkpoints/*.tar.gz`.

Nothing uses the HuggingFace cache, `HF_HOME`, or `TORCH_HOME`.

## Mechanism: symlinks plus one guard

Symlink both directories to r0. No env var.

- `~/.spatial-maker/checkpoints` -> `/Volumes/r0/models/spatial-maker/checkpoints`
- repo `checkpoints` -> `/Volumes/r0/models/spatial-maker/source`

Why symlinks over `SPATIAL_MAKER_CHECKPOINTS`: the env var only reaches the Rust binary, and only in shells that export it (paseo agents, `just`, launchd would each need it). Python hardcodes the paths. A symlink covers the installed binary, dev builds, and both Python scripts with zero config.

What happens when r0 is unmounted today: the symlink dangles, `find_model` fails, `ensure_model_exists` calls `create_dir_all` on the dangling link, which errors with `Failed to create checkpoint directory: File exists`. No download happens, but the message is misleading. `/Volumes` is root-owned, so nothing can be created under `/Volumes/r0` on the internal disk.

Code change (one guard in `ensure_model_exists`, before `create_dir_all`):

```rust
if checkpoint_dir.is_symlink() && !checkpoint_dir.exists() {
	return Err(SpatialError::ConfigError(format!(
		"Checkpoint dir {} points to {}, which is missing. Is the drive mounted?",
		checkpoint_dir.display(),
		std::fs::read_link(&checkpoint_dir).unwrap_or_default().display(),
	)));
}
```

Also, when `SPATIAL_MAKER_CHECKPOINTS` is set and the dir does not exist, return the same kind of error instead of creating it. An explicit path that is missing is a config error, not a first run.

Python: `depth_bench.py` gets a one-line check at startup that `HOME_CKPT.exists()` and exits with the same message. `convert_to_coreml.py` calls `checkpoint_dir.mkdir(exist_ok=True)`, which raises `FileExistsError` on a dangling symlink; that is loud enough, leave it.

## Drive and layout

`/Volumes/r0`: 325G free, fixed USB, HFS+ RAID0, mounted since boot along with all other externals (uptime 11 days). It already holds `/Volumes/r0/models/gguf` (trxcc) and `/Volumes/r0/hf-cache`, so it is the existing home for model files. RAID0 has no redundancy; acceptable because every file here can be re-downloaded or re-converted.

`/Volumes/wd` (37G free, PCIe) is faster but nearly full and holds the MLX models. The others have 43-66G and are media drives.

```
/Volumes/r0/models/spatial-maker/
	checkpoints/          <- ~/.spatial-maker/checkpoints (runtime CoreML models)
		DepthAnythingV2SmallF16.mlpackage
		DepthAnythingV2BaseF16.mlpackage
		DepthAnythingV2LargeF16.mlpackage
		DepthAnythingV3Mono.mlpackage
		DistillAnyDepthSmallF16.mlpackage
	source/               <- repo checkpoints/ (PyTorch sources, conversion output, README)
		depth_anything_v2_vit{s,b,l}.pth
		distill_any_depth_vits.safetensors
		README.md
```

`DistillAnyDepthSmallF16.mlpackage` moves into `checkpoints/` so all runtime models sit in one dir.

## Code and config changes

1. `.gitignore`: change `checkpoints/` to `/checkpoints`. A trailing slash does not match a symlink, so git would show the link as untracked.
2. `Cargo.toml` `exclude`: `"checkpoints/"` -> `"checkpoints"` for the same reason.
3. `src/model.rs`: the two guards above.
4. `analysis/depth_bench.py`: `distill-small` -> `HOME_CKPT / 'DistillAnyDepthSmallF16.mlpackage'`; `da3mono-large` -> `HOME_CKPT / 'DepthAnythingV3Mono.mlpackage'`; add the mount check.
5. `scripts/coreml_conversion/convert_to_coreml.py`: no change. It writes new mlpackages to `source/`; copy to `checkpoints/` by hand, as today.
6. `checkpoints/README.md` is untracked today (whole dir ignored). It moves with `source/`.

## Migration steps

Run when no agent is benchmarking (the "faster encoding" agent reads these models).

```sh
DEST=/Volumes/r0/models/spatial-maker
REPO=~/dev/spatial-maker
```

```sh
mkdir -p "$DEST/checkpoints" "$DEST/source"
rsync -a --exclude da3mono-coreml --exclude distill-any-depth ~/.spatial-maker/checkpoints/ "$DEST/checkpoints/"
rsync -a "$REPO/checkpoints/DistillAnyDepthSmallF16.mlpackage" "$DEST/checkpoints/"
rsync -a --exclude '*.mlpackage' --exclude depth_pro.pt "$REPO/checkpoints/" "$DEST/source/"
diff -rq ~/.spatial-maker/checkpoints/DepthAnythingV3Mono.mlpackage "$DEST/checkpoints/DepthAnythingV3Mono.mlpackage"
```

After verifying the copies, replace each dir with a symlink:

```sh
mv ~/.spatial-maker/checkpoints ~/.spatial-maker/checkpoints.old
ln -s "$DEST/checkpoints" ~/.spatial-maker/checkpoints
mv "$REPO/checkpoints" "$REPO/checkpoints.old"
ln -s "$DEST/source" "$REPO/checkpoints"
```

Verify, then delete the `.old` dirs (frees ~5.9G):

- `spatial-maker -m s` and `-m da3` on `test_baseline.jpg` load without downloading.
- `uv run analysis/depth_bench.py` finds all five models.
- Unmount check: `SPATIAL_MAKER_CHECKPOINTS=/Volumes/nope/x spatial-maker -m s test_baseline.jpg` fails with the new message and writes nothing.

Decide separately: `depth_pro.pt` (1.8G, unused). Delete, or keep in `source/` if Depth Pro is a future candidate.

## Out of scope

- `target/` (441M): build outputs; `build-dir` is already in the cargo cache. `cargo clean` reclaims it any time.
- `spatial-maker-rust/` (455M): separate nested repo, gitignored. 454M of it is `examples/outputs`, not weights. Delete or move that outputs dir on its own.
- `~/.cache/huggingface` (46M): not used by this repo.
