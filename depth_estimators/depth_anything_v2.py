import time
import subprocess
import os
import sys
import psutil
from pathlib import Path
import tempfile
import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

from depth_anything_v2.dpt import DepthAnythingV2
from scripts.stitch_video import stitch_frames_to_hevc

_threads_configured = False


class DepthAnythingV2Estimator:
    def __init__(self, encoder='vits', num_threads=None):
        self.model = None
        self.device = None
        self.encoder = encoder
        self.num_threads = num_threads if num_threads is not None else max(1, os.cpu_count() - 4)

    def load_model(self):
        global _threads_configured
        if self.model is not None:
            return

        if not _threads_configured:
            torch.set_num_threads(self.num_threads)
            torch.set_num_interop_threads(self.num_threads)
            _threads_configured = True

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        checkpoint_name = f"depth_anything_v2_{self.encoder}.pth"
        checkpoint_path = self._find_checkpoint(checkpoint_name)

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = DepthAnythingV2(**model_configs[self.encoder])
        # Load weights directly to target device to avoid MPS/CPU tensor mismatch
        self.model.load_state_dict(torch.load(str(checkpoint_path), map_location=self.device))
        self.model = self.model.to(self.device).eval()

    def _find_checkpoint(self, checkpoint_name: str) -> Path:
        """
        Find checkpoint file in multiple possible locations.

        Searches in order:
        1. Next to the script's source directory (for development)
        2. User's home directory ~/.spatial-maker/checkpoints/
        3. XDG data directory
        4. Current working directory ./checkpoints/
        """
        search_paths = [
            # Development: relative to source file
            Path(__file__).parent.parent / "checkpoints" / checkpoint_name,
            # User home directory
            Path.home() / ".spatial-maker" / "checkpoints" / checkpoint_name,
            # XDG data home
            Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
            / "spatial-maker" / "checkpoints" / checkpoint_name,
            # Current working directory
            Path.cwd() / "checkpoints" / checkpoint_name,
        ]

        # Also check if SPATIAL_MAKER_CHECKPOINTS env var is set
        env_checkpoint_dir = os.environ.get("SPATIAL_MAKER_CHECKPOINTS")
        if env_checkpoint_dir:
            search_paths.insert(0, Path(env_checkpoint_dir) / checkpoint_name)

        for path in search_paths:
            if path.exists():
                return path

        # Not found - provide helpful error message
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_name}' not found.\n"
            f"Searched locations:\n"
            + "\n".join(f"  - {p}" for p in search_paths)
            + "\n\nPlease download the checkpoint and place it in one of these locations:\n"
            f"  mkdir -p ~/.spatial-maker/checkpoints\n"
            f"  # Then copy {checkpoint_name} there"
        )

    def process_video(self, video_path: str, output_dir: str = "output", max_frames: int = None):
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024

        warmup_start = time.time()
        self.load_model()
        warmup_time = time.time() - warmup_start

        mem_after_load = process.memory_info().rss / 1024 / 1024
        print(f"  Model load time: {warmup_time:.3f}s")
        print(f"  Memory after load: {mem_after_load:.1f} MB (delta: +{mem_after_load - mem_before:.1f} MB)")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frames = []
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and i >= max_frames:
                break
            frames.append(frame)
            i += 1
        cap.release()

        warmup_start = time.time()
        with torch.no_grad():
            _ = self.model.infer_image(frames[0], 518)
        warmup_inference = time.time() - warmup_start
        print(f"  First inference (warmup): {warmup_inference:.3f}s")

        frame_times = []

        for i, frame in enumerate(tqdm(frames, desc="  Processing frames", unit="frame")):
            start = time.time()
            with torch.no_grad():
                depth = self.model.infer_image(frame, 518)
            elapsed = time.time() - start
            frame_times.append(elapsed)

            depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
            Image.fromarray(depth_normalized).save(output_dir / f"depth_frame_{i+1:04d}.png")

        total_time = sum(frame_times)
        avg_time = total_time / len(frame_times)
        fps = 1 / avg_time if avg_time > 0 else 0

        mem_peak = process.memory_info().rss / 1024 / 1024

        print(f"  Stitching frames to HEVC video...")
        # Create video path adjacent to output_dir with same name
        output_dir_name = output_dir.name
        video_path = output_dir.parent / f"{output_dir_name}.mp4"
        stitch_frames_to_hevc(str(output_dir), str(video_path), fps=24)

        return {
            "frames_processed": len(frame_times),
            "total_time": total_time,
            "avg_time_per_frame": avg_time,
            "fps": round(fps, 2),
            "warmup_time": warmup_time,
            "warmup_inference": warmup_inference,
            "memory_mb": round(mem_peak, 1),
            "device": str(self.device),
            "video_path": str(video_path)
        }
