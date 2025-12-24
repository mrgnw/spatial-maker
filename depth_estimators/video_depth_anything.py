import time
import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "Video-Depth-Anything"))
from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames


class VideoDepthAnythingEstimator:
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

        checkpoint_path = Path(__file__).parent.parent / "Video-Depth-Anything" / "checkpoints" / f"video_depth_anything_{self.encoder}.pth"

        self.model = VideoDepthAnything(**model_configs[self.encoder], metric=False)
        self.model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'), strict=True)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device).eval()

    def process_video(self, video_path: str, output_dir: str = "output", max_frames: int = 3):
        warmup_start = time.time()
        self.load_model()
        warmup_time = time.time() - warmup_start
        print(f"  Model load time: {warmup_time:.3f}s")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Reading video frames...")
        frames, fps = read_video_frames(video_path, process_length=max_frames, target_fps=-1, max_res=1280)

        print(f"  Warming up model with first frame...")
        warmup_start = time.time()
        with torch.no_grad():
            _ = self.model.infer_video_depth(frames[:1], fps, input_size=518, device=self.device.type, fp32=False)
        warmup_inference = time.time() - warmup_start
        print(f"  First inference (warmup): {warmup_inference:.3f}s")

        print(f"  Running depth inference on {len(frames)} frames...")
        start = time.time()
        with torch.no_grad():
            depths, _ = self.model.infer_video_depth(frames, fps, input_size=518, device=self.device.type, fp32=False)
        total_time = time.time() - start

        print(f"  Saving depth maps...")
        d_min, d_max = depths.min(), depths.max()
        print(f"  Depth range: {d_min:.3f} to {d_max:.3f}")
        for i, depth in enumerate(depths):
            depth_normalized = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            Image.fromarray(depth_normalized).save(output_dir / f"depth_frame_{i+1:04d}.png")

        return {
            "frames_processed": len(frames),
            "total_time": total_time,
            "avg_time_per_frame": total_time / len(frames),
            "warmup_time": warmup_time,
            "warmup_inference": warmup_inference,
            "device": str(self.device)
        }
