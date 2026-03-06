# Spatial Settings Panel Overhaul

Improve spatial settings panel: model downloading, unified start button, eliminate redundant downscaling.

## Status: COMPLETE

All tasks done. `svelte-check` passes with 0 errors. `cargo check` passes.

## Decisions Made

- **Downscale integration:** Two-pass approach. Frame GUI does ffmpeg conversion first (user's resolution/codec/trim settings), then spatial-maker processes the output with `--skip-downscale`.
- **FPS:** Keep 24fps hardcoded for spatial (Vision Pro preference).
- **Model download:** Rust/Tauri command with reqwest streaming + progress events to the UI.

## Tasks

### 1. Rust: Add reqwest dependency and model check/download Tauri commands
- [x] Add `reqwest` (with `stream` feature) and `futures-util` to Cargo.toml
- [x] Add `ModelDownloadProgressPayload`, `ModelDownloadCompletePayload`, `ModelDownloadErrorPayload` to `spatial/types.rs`
- [x] Add `enabled: bool` field to `SpatialConfig` Rust struct
- [x] Add `check_spatial_models` command (checks `~/.spatial-maker/checkpoints/` for each model)
- [x] Add `download_spatial_model` command (streams from HuggingFace, emits progress events)
- Files changed:
  - `frame/src-tauri/Cargo.toml`
  - `frame/src-tauri/src/spatial/types.rs`
  - `frame/src-tauri/src/spatial/commands.rs`

### 2. Rust: Register new commands in lib.rs
- [x] Add `check_spatial_models` and `download_spatial_model` to invoke_handler
- File: `frame/src-tauri/src/lib.rs`

### 3. Svelte: Add 'enabled' field to SpatialConfig type + service layer
- [x] Add `enabled: boolean` to `SpatialConfig` interface in `types.ts` (default: `false`)
- [x] Default `skipDownscale` to `true` (no longer user-facing, always skipped)
- [x] Add `checkSpatialModels()` and `downloadSpatialModel()` to spatial service
- [x] Add `setupModelDownloadListeners()` for progress/complete/error events
- Files:
  - `frame/src/lib/types.ts`
  - `frame/src/lib/services/spatial.ts`

### 4. Svelte: Overhaul SpatialTab UI
- [x] Add "Enable spatial encoding" checkbox at the top (prominent)
- [x] Gray out rest of settings when disabled (opacity + pointer-events-none)
- [x] Model selector: show download status per model (check icon / arrow icon / progress %)
- [x] Clicking unavailable model triggers download with inline progress bar
- [x] Fix spacing issues (tighter gaps, consistent sizing)
- [x] Remove "Skip downscale" checkbox (always skipped now)
- [x] Remove old info box about manual installation
- File: `frame/src/lib/components/settings/tabs/SpatialTab.svelte`

### 5. Svelte: Remove Spatial button from titlebar
- [x] Remove `onStartSpatial` prop and the Spatial button from all 3 titlebars
- [x] Remove `onStartSpatial` from Titlebar.svelte wrapper
- [x] Remove `IconGlasses` import from titlebars
- [x] Clean up +page.svelte (remove onStartSpatial binding)
- Files:
  - `frame/src/lib/components/layout/titlebar/MacosTitlebar.svelte`
  - `frame/src/lib/components/layout/titlebar/WindowsTitlebar.svelte`
  - `frame/src/lib/components/layout/titlebar/LinuxTitlebar.svelte`
  - `frame/src/lib/components/layout/Titlebar.svelte`
  - `frame/src/routes/+page.svelte`

### 6. Svelte: Wire Start button to chain spatial after conversion
- [x] Add `onConversionCompleted` callback to `ConversionCallbacks` interface
- [x] In `useConversionQueue.svelte.ts` `onCompleted` handler: call callback instead of marking COMPLETED directly
- [x] Add `queueSpatialForFile(id, filePath)` method to spatial queue
- [x] In `+page.svelte`: wire callback to check `spatialQueue.config.enabled` and chain spatial
- Files:
  - `frame/src/lib/features/conversion/useConversionQueue.svelte.ts`
  - `frame/src/lib/features/spatial/useSpatialQueue.svelte.ts`
  - `frame/src/routes/+page.svelte`

### 7. Rust: Always pass --skip-downscale when invoked from Frame
- [x] In `spatial/worker.rs`, always add `--skip-downscale` flag
- [x] Removed conditional `if task.config.skip_downscale` check
- [x] Updated progress mapping (depth_stereo 0-85% instead of 10-85%)
- File: `frame/src-tauri/src/spatial/worker.rs`

### 8. Build and verify compilation
- [x] `cargo check` passes (1 pre-existing dead_code warning for TaskNotFound)
- [x] `svelte-check` passes with 0 errors, 0 warnings

## Architecture Notes

### Flow (after changes)
```
User clicks Start
  -> Frame ffmpeg conversion (resolution, codec, trim, etc.)
  -> conversion-completed event (with outputPath)
  -> if spatialConfig.enabled:
     -> queue_spatial(id, outputPath, spatialConfig) with --skip-downscale
     -> spatial-maker: depth+stereo -> audio mux -> MV-HEVC
     -> spatial-completed event
  -> Mark file as COMPLETED
```

### Model URLs
- Small (vits, ~99MB): https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
- Base (vitb, ~390MB): https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
- Large (vitl, ~1.3GB): https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
