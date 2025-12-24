import time
import subprocess
import os
import sys
from pathlib import Path
import tempfile
import torch
import numpy as np
from PIL import Image
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2


class DepthAnythingV2Estimator:
    def __init__(self, encoder='vits'):
        self.model = None
        self.device = None
        self.encoder = encoder

    def load_model(self):
        if self.model is not None:
            return

        num_threads = max(1, os.cpu_count() - 4)
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        checkpoint_path = Path(__file__).parent.parent / "Depth-Anything-V2" / "checkpoints" / f"depth_anything_v2_{self.encoder}.pth"

        self.model = DepthAnythingV2(**model_configs[self.encoder])
        self.model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'))

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device).eval()

    def process_video(self, video_path: str, output_dir: str = "output", max_frames: int = 8):
        warmup_start = time.time()
        self.load_model()
        warmup_time = time.time() - warmup_start
        print(f"  Model load time: {warmup_time:.3f}s")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frames = []
        for i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        print(f"  Warming up model with first frame...")
        warmup_start = time.time()
        with torch.no_grad():
            _ = self.model.infer_image(frames[0], 518)
        warmup_inference = time.time() - warmup_start
        print(f"  First inference (warmup): {warmup_inference:.3f}s")

        frame_times = []

        for i, frame in enumerate(frames):
            print(f"  Processing frame {i+1}/{len(frames)}...")

            start = time.time()
            with torch.no_grad():
                depth = self.model.infer_image(frame, 518)
            elapsed = time.time() - start
            frame_times.append(elapsed)
            print(f"    Inference: {elapsed:.3f}s")

            depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
            Image.fromarray(depth_normalized).save(output_dir / f"depth_frame_{i+1:04d}.png")

        total_time = sum(frame_times)
        avg_time = total_time / len(frame_times)
        fps = 1 / avg_time if avg_time > 0 else 0
        return {
            "frames_processed": len(frame_times),
            "total_time": total_time,
            "avg_time_per_frame": avg_time,
            "fps": round(fps, 2),
            "warmup_time": warmup_time,
            "warmup_inference": warmup_inference,
            "device": str(self.device)
        }
