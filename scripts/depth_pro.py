#!/usr/bin/env python3
import time
import json
import sys
from pathlib import Path
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
import numpy as np


def load_model():
    model_path = hf_hub_download(repo_id="apple/DepthPro", filename="depth_pro.pt")
    model = torch.jit.load(model_path)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()
    return model, device


def estimate_depth(image_path: str, output_path: str = None):
    if output_path is None:
        p = Path(image_path)
        output_path = f"{p.stem}_depth.png"

    model, device = load_model()

    img = Image.open(image_path).convert('RGB')
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    start = time.time()
    with torch.no_grad():
        depth = model(img_tensor)
    inference_time = time.time() - start

    depth_np = depth.squeeze().cpu().numpy()
    depth_normalized = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255).astype(np.uint8)
    Image.fromarray(depth_normalized).save(output_path)

    stats = {
        "input": image_path,
        "output": output_path,
        "inference_time": round(inference_time, 3),
        "device": str(device)
    }

    return stats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python depth_pro.py <input_image> [output_depth]")
        sys.exit(1)

    output = sys.argv[2] if len(sys.argv) > 2 else None
    stats = estimate_depth(sys.argv[1], output)

    print(json.dumps(stats, indent=2))
