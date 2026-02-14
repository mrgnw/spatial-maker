#!/bin/bash
set -e

# Download CoreML Depth Anything V2 models for Apple Silicon
# These are hosted on HuggingFace for easy distribution

CHECKPOINT_DIR="checkpoints"
HF_REPO="mrgnw/depth-anything-v2-coreml"
BASE_URL="https://huggingface.co/${HF_REPO}/resolve/main"

mkdir -p "$CHECKPOINT_DIR"
cd "$CHECKPOINT_DIR"

echo "Downloading Depth Anything V2 CoreML models..."
echo "Source: https://huggingface.co/${HF_REPO}"
echo ""

# Small (Apple's official, fast reference)
if [ ! -d "DepthAnythingV2SmallF16.mlpackage" ]; then
    echo "📥 Downloading Small model (48 MB)..."
    curl -L -o DepthAnythingV2SmallF16.mlpackage.zip \
        "https://ml-assets.apple.com/coreml/models/Image/DepthEstimation/DepthAnything/DepthAnythingV2SmallF16.mlpackage.zip"
    unzip -q DepthAnythingV2SmallF16.mlpackage.zip
    rm DepthAnythingV2SmallF16.mlpackage.zip
    echo "✓ Small model ready"
else
    echo "✓ Small model already exists"
fi

# Base (converted, sweet spot for quality/speed)
if [ ! -d "DepthAnythingV2BaseF16.mlpackage" ]; then
    echo "📥 Downloading Base model (186 MB)..."
    curl -L -o DepthAnythingV2BaseF16.mlpackage.tar.gz \
        "${BASE_URL}/DepthAnythingV2BaseF16.mlpackage.tar.gz"
    tar -xzf DepthAnythingV2BaseF16.mlpackage.tar.gz
    rm DepthAnythingV2BaseF16.mlpackage.tar.gz
    echo "✓ Base model ready"
else
    echo "✓ Base model already exists"
fi

# Large (converted, maximum quality)
if [ ! -d "DepthAnythingV2LargeF16.mlpackage" ]; then
    echo "📥 Downloading Large model (638 MB)..."
    curl -L -o DepthAnythingV2LargeF16.mlpackage.tar.gz \
        "${BASE_URL}/DepthAnythingV2LargeF16.mlpackage.tar.gz"
    tar -xzf DepthAnythingV2LargeF16.mlpackage.tar.gz
    rm DepthAnythingV2LargeF16.mlpackage.tar.gz
    echo "✓ Large model ready"
else
    echo "✓ Large model already exists"
fi

echo ""
echo "✅ All CoreML models ready!"
echo ""
echo "Models:"
echo "  - Small:  48 MB  (~30ms,    30+ fps on M4 Pro)"
echo "  - Base:   186 MB (~60-90ms, 11-16 fps on M4 Pro)"
echo "  - Large:  638 MB (~200-300ms, 3-5 fps on M4 Pro)"
echo ""
echo "These models use the Apple Neural Engine for fast inference."
