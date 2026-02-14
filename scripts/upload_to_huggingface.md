# Upload CoreML Models to HuggingFace

Follow these steps to upload the converted CoreML models to HuggingFace for easy distribution.

## 1. Create HuggingFace Repository

1. Go to https://huggingface.co/new
2. Create a new **model** repository
3. Name it: `depth-anything-v2-coreml` (or your preferred name)
4. Make it **public** so users can download without authentication
5. Add a description: "Depth Anything V2 models in CoreML format for Apple Silicon"

## 2. Prepare Models for Upload

The models are already in `checkpoints/`. We need to compress them:

```bash
cd checkpoints

# Compress Base model
tar -czf DepthAnythingV2BaseF16.mlpackage.tar.gz DepthAnythingV2BaseF16.mlpackage

# Compress Large model  
tar -czf DepthAnythingV2LargeF16.mlpackage.tar.gz DepthAnythingV2LargeF16.mlpackage

# Check sizes
ls -lh *.tar.gz
```

This compresses the models from ~870 MB to ~430 MB total.

## 3. Upload via Web Interface (Easiest)

1. Go to your repo: `https://huggingface.co/YOUR_USERNAME/depth-anything-v2-coreml`
2. Click "Files and versions" → "Add file" → "Upload files"
3. Upload:
   - `DepthAnythingV2BaseF16.mlpackage.tar.gz`
   - `DepthAnythingV2LargeF16.mlpackage.tar.gz`
4. Commit with message: "Add CoreML models (Base and Large)"

**Note**: We don't upload Small because it's available directly from Apple.

## 4. Upload via Git LFS (Alternative)

If you prefer command-line:

```bash
# Install git-lfs if needed
brew install git-lfs
git lfs install

# Clone your HF repo
git clone https://huggingface.co/YOUR_USERNAME/depth-anything-v2-coreml
cd depth-anything-v2-coreml

# Track tar.gz files with LFS
git lfs track "*.tar.gz"
git add .gitattributes

# Copy and commit models
cp ../spatial-maker/checkpoints/*.tar.gz .
git add *.tar.gz
git commit -m "Add CoreML models (Base and Large)"
git push
```

## 5. Create Model Card (README.md)

Add this to your HuggingFace repo's README.md:

````markdown
---
license: apache-2.0
tags:
- depth-estimation
- coreml
- apple-silicon
- vision
---

# Depth Anything V2 - CoreML

Depth Anything V2 models converted to CoreML format for optimized inference on Apple Silicon (M-series chips).

## Models

| Model | Size | Parameters | Performance (M4 Pro) | License |
|-------|------|------------|----------------------|---------|
| Small F16 | 48 MB | 24.8M | ~30ms (~33 fps) | Apache-2.0 |
| Base F16 | 186 MB | 97.5M | ~60-90ms (~14 fps) | Apache-2.0 |
| Large F16 | 638 MB | 335.3M | ~200-300ms (~4 fps) | CC-BY-NC-4.0 |

All models use Float16 precision and run on Apple's Neural Engine + GPU + CPU.

## Download

```bash
# Download from spatial-maker repo
git clone https://github.com/YOUR_USERNAME/spatial-maker
cd spatial-maker
./scripts/download_coreml_models.sh
```

Or download directly:
- [Base model](https://huggingface.co/YOUR_USERNAME/depth-anything-v2-coreml/resolve/main/DepthAnythingV2BaseF16.mlpackage.tar.gz) (186 MB)
- [Large model](https://huggingface.co/YOUR_USERNAME/depth-anything-v2-coreml/resolve/main/DepthAnythingV2LargeF16.mlpackage.tar.gz) (638 MB)

Small model available from [Apple's CoreML model zoo](https://developer.apple.com/machine-learning/models/).

## Usage

```swift
import CoreML

let model = try MLModel(contentsOf: URL(fileURLWithPath: "DepthAnythingV2BaseF16.mlpackage"))
// ... inference code
```

## Citation

```bibtex
@article{yang2024depth,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}
```

## Conversion

These models were converted from PyTorch using `coremltools`. See [conversion script](https://github.com/YOUR_USERNAME/spatial-maker/tree/main/scripts/coreml_conversion).
````

## 6. Update Download Script

After uploading, update `scripts/download_coreml_models.sh` with your actual HuggingFace username:

```bash
HF_REPO="YOUR_ACTUAL_USERNAME/depth-anything-v2-coreml"
```

## Verification

Test the download works:

```bash
cd /tmp
git clone https://github.com/YOUR_USERNAME/spatial-maker
cd spatial-maker
./scripts/download_coreml_models.sh
```

Should download all three models successfully!
