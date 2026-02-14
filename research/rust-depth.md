# Rust Depth Estimation Library for spatial-maker

## Model Decision

**Depth Anything V2 Small** via ONNX Runtime — ready-to-use `.onnx` files from [fabio-sim/Depth-Anything-ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX/releases/tag/v2.0.0). 

- **License**: Apache-2.0
- **Params**: 24.8M
- **Input**: `(1, 3, 518, 518)` 
- **Output**: `(1, 518, 518)`
- **Speed on M4 Pro**: **~430ms** total (CPU execution via ONNX Runtime)
  - First run: 1.1s (cold start)
  - Subsequent runs: 425-434ms average
  - CoreML EP tested but slower (9.5s) due to compilation overhead
- **Upgrade path**: Later swap in Video Depth Anything Small for temporal consistency on video

## Crate Structure

```
spatial-maker/
├── Cargo.toml              # lib crate: ort (coreml), image, ndarray, anyhow
├── src/
│   ├── lib.rs              # pub API: process_photo(), process_video()
│   ├── depth.rs            # ONNX session, preprocessing, inference
│   ├── stereo.rs           # DIBR: depth map → left/right eye pair
│   ├── output.rs           # SBS image/video output, spatial CLI wrapper
│   └── model.rs            # checkpoint discovery & download
├── examples/
│   └── photo.rs            # CLI: image → spatial SBS photo
├── pipeline.py             # existing Python (kept, deprecated later)
└── scripts/                # existing Python (kept, deprecated later)
```

## Dependencies

```toml
[dependencies]
ort = { version = "2.0.0-rc.11", features = ["coreml", "ndarray"] }
image = "0.25"
ndarray = "0.17"
anyhow = "1.0"
```

## Phase 1: Photo MVP (high priority)

1. **Init crate** — `cargo init --lib`, add deps
2. **`depth.rs`** — Load ONNX model via `ort::Session` with CoreML EP fallback to CPU. Preprocessing borrowed from `bits/candle-spike/`:
   - Resize to 518x518 (Lanczos3 via `image` crate)
   - ImageNet normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
   - NCHW float32 tensor via `ndarray`
   - Run inference, min-max normalize output to 0.0-1.0
3. **`stereo.rs`** — Port `create_stereo_pair()` from `scripts/depth_to_stereo.py:25-72`:
   - `disparity = depth * max_disparity`
   - Build coordinate grids, shift left/right by ±disparity/2
   - Bilinear interpolation with border replication (~80 lines)
4. **`output.rs`** — Horizontally stack left+right into SBS image, save via `image` crate
5. **`examples/photo.rs`** — CLI: `cargo run --example photo -- input.jpg -o spatial.jpg --max-disparity 30`
6. **Download model** — Get `depth_anything_v2_vits.onnx` from fabio-sim releases, store in `~/.spatial-maker/checkpoints/`
7. **Test CoreML** — Verify `ort` CoreML EP works with this ONNX graph on Apple Silicon

## Phase 2: Production Features (medium priority)

8. **Model management** — Checkpoint discovery across standard paths (same as Python: `~/.spatial-maker/checkpoints/`, env var, etc.) + download from HuggingFace/GitHub
9. **Video processing** — Extract frames via ffmpeg subprocess pipe → per-frame depth+stereo → pipe SBS frames back to ffmpeg HEVC encoder. Progress callback API for Frame integration.

## Phase 3: Better Model (low priority)

10. **Video Depth Anything** — Export VDA Small to ONNX (one-time Python script). The temporal attention layers make this non-trivial but the offline batch mode (32 frames) is exportable. This gives flicker-free video depth. If export is too complex, stay on DA V2 with lightweight frame-to-frame smoothing.

## Frame Integration (after Phase 1 works)

- Add `spatial-maker` as a path dependency in Frame's `Cargo.toml`
- Replace `run_spatial_worker` (subprocess spawning `uv tool run spatial-maker`) with direct Rust library calls
- Update model download URLs from `.pth` → `.onnx`
- Remove Python/uv dependency entirely

## Performance Results (M4 Pro)

**CPU Execution (ONNX Runtime):**
- **430ms average** for full pipeline (load image → depth → stereo → save SBS)
- First run: 1.1s (cold start with model loading)
- Subsequent runs: 425-434ms consistent
- No Python overhead, pure Rust binary

**CoreML Execution Provider:**
- Tested but **9.5s total** — significantly slower than CPU
- Likely due to ONNX→CoreML compilation overhead on first use
- May have unsupported ops falling back to CPU intermittently
- **Conclusion**: CPU execution is optimal for this model on M4 Pro

The CPU performance is excellent because:
1. M4 Pro has very fast CPU cores
2. Depth Anything V2 Small is only 24.8M params
3. ONNX Runtime is highly optimized
4. No Python interpreter overhead

## Research References

### Best Models (Early 2026)

| Model | Params | License | Temporal? | ONNX? | Speed |
|-------|--------|---------|-----------|-------|-------|
| **Video Depth Anything Small** | 28.4M | Apache-2.0 | **Yes** | Needs export | 7.5ms/frame (A100 FP16) |
| **Depth Anything V2 Small** | 24.8M | Apache-2.0 | No | **Ready** | 13ms (RTX4080) |
| MoGe-2 ViT-S | 35M | MIT | No | Yes | ~60ms (A100) |
| PromptDA Small | 25.1M | Apache-2.0 | No | Needs export | Unknown |
| Depth Pro (Apple) | ~300M+ | Restrictive | No | No | 300ms |

### Key Findings from ~/dev/bits/

The `bits/` repo has two depth implementations:

1. **Browser (Transformers.js)** — Working production code using `@huggingface/transformers` with ONNX Runtime Web (WASM). Uses `Xenova/depth-anything-small-hf` model. Proves the ONNX pipeline works end-to-end.

2. **Rust Candle spike** — Proof-of-concept at `bits/candle-spike/src/main.rs` (292 lines) that validates:
   - Metal GPU detection via `Device::new_metal(0)`
   - Image preprocessing (resize to 518x518, ImageNet normalization, NCHW layout)
   - Output postprocessing (min-max normalize, Luma PNG)
   - **Gap**: Actual model inference is simulated — the ViT+DPT architecture isn't implemented in Candle

This reinforces the `ort` + ONNX approach: the preprocessing/postprocessing Rust code is proven, and `ort` gives us the model for free.

### ONNX Models Available

From [fabio-sim/Depth-Anything-ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX):

- **Static shape models** (opset 18, TorchDynamo export):
  - `depth_anything_v2_vits.onnx` (99MB) — vits model
  - `depth_anything_v2_vitb.onnx` — vitb model
  - `depth_anything_v2_vitl.onnx` — vitl model

- **Dynamic shape models** (opset 17):
  - Same sizes, but support variable H/W (must be divisible by 14)

All models have signature: `(B, 3, H, W) → (B, H, W)` depth map.

### Preprocessing Pipeline (Confirmed Identical)

| Step | Python | JS (Transformers.js) | Rust spike | Rust target |
|------|--------|---------------------|------------|-------------|
| Resize | 518px | 518x518 | 518x518 Lanczos3 | 518x518 Lanczos3 |
| Normalize | ImageNet | ImageNet | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | Same |
| Layout | NCHW | NCHW | [1, 3, H, W] | [1, 3, H, W] |
| Output | min-max → 0-1 | min-max → 0-255 PNG | min-max → 0-255 Luma PNG | min-max → 0-1 float |

### Video Depth Anything (VDA) Details

- **Released**: CVPR 2025 Highlight
- **Temporal consistency**: Yes — adds temporal attention layers on top of DA V2
- **Sizes**: Small (28.4M, Apache-2.0), Base (113.1M, CC-BY-NC-4.0), Large (381.8M, CC-BY-NC-4.0)
- **Speed**: Small 7.5ms/frame FP16 on A100
- **ONNX export**: Not provided officially. Temporal attention with cached hidden states makes export non-trivial
- **Streaming mode**: Experimental feature available (training-free, but performance drop)
- **Best for**: Spatial video where temporal consistency is critical

For photos, DA V2 is sufficient. For video, VDA is ideal if we can export it to ONNX, otherwise we add frame-to-frame smoothing on DA V2 output.
