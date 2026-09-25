---
status: done
branch: plan/move-weights-external
---

# Move model weights to /Volumes/r0

Done 2026-09-25. Weights live in r0's shared per-format model dirs, next to `gguf/` (trxcc), `mlx/`, and ollama's `blobs/` + `manifests/`.

| r0 dir | Contents | Reached through |
|-|-|-|
| `/Volumes/r0/models/coreml/` | `DepthAnythingV2{Small,Base,Large}F16`, `DepthAnythingV3Mono`, `DistillAnyDepthSmallF16` mlpackages + READMEs | `~/.spatial-maker/checkpoints` symlink |
| `/Volumes/r0/models/pytorch/` | `depth_anything_v2_vit{s,b,l}.pth`, `distill_any_depth_vits.safetensors`, `depth_pro.pt` | repo `checkpoints` symlink |

Why symlinks, not `SPATIAL_MAKER_CHECKPOINTS`: the env var only reaches the Rust binary, and only in shells that export it. Symlinks cover the installed binary, dev builds, `analysis/depth_bench.py`, and `convert_to_coreml.py`.

Unmounted r0: `ensure_model_exists` returns "Checkpoint dir ... links to ..., which is missing. Is the drive mounted?" and never downloads. The bench script prints `SKIP <model>: <path> missing`.

Dropped byte-identical duplicates: `da3mono-coreml/` (DA3 mlpackage), `distill-any-depth/small/model.safetensors`, repo `DepthAnythingV2SmallF16.mlpackage`.

Verified: all copies byte-compared before deleting the originals; `depth_bench.py` loads and runs all five CoreML models from r0.

## Open

- `depth_pro.pt` (1.8G) is unused by any code. Kept on r0.
- `spatial-maker-rust/examples/outputs` (454M) and `target/` (441M) are still on the internal disk. Not weights.
