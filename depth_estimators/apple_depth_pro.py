import time
import subprocess
import os
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor
import torch
import depth_pro
import numpy as np
from PIL import Image


class AppleDepthPro:
    def __init__(self):
        self.model = None
        self.transform = None
        self.device = None

    def load_model(self):
        if self.model is not None:
            return

        num_threads = max(1, os.cpu_count() - 4)
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)

        self.model, self.transform = depth_pro.create_model_and_transforms()
        self.model.eval()

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

    def process_video(self, video_path: str, output_dir: str = "output", max_frames: int = 8):
        warmup_start = time.time()
        self.load_model()
        warmup_time = time.time() - warmup_start
        print(f"  Model load time: {warmup_time:.3f}s")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / "frames"
            frames_dir.mkdir()

            cmd = ["ffmpeg", "-i", video_path, "-frames:v", str(max_frames), f"{frames_dir}/frame_%04d.png"]
            subprocess.run(cmd, capture_output=True, check=True)

            frames = sorted(frames_dir.glob("frame_*.png"))

            print(f"  Warming up model with first frame...")
            first_frame = frames[0]
            image, _, f_px = depth_pro.load_rgb(str(first_frame))
            image = self.transform(image).to(self.device)
            warmup_start = time.time()
            with torch.no_grad():
                _ = self.model.infer(image, f_px=f_px)
            warmup_inference = time.time() - warmup_start
            print(f"  First inference (warmup): {warmup_inference:.3f}s")

            frame_times = []

            for i, frame in enumerate(frames, 1):
                print(f"  Processing frame {i}/{len(frames)}...")
                output = output_dir / f"depth_{frame.name}"

                image, _, f_px = depth_pro.load_rgb(str(frame))
                image = self.transform(image).to(self.device)

                start = time.time()
                with torch.no_grad():
                    prediction = self.model.infer(image, f_px=f_px)
                elapsed = time.time() - start
                frame_times.append(elapsed)
                print(f"    Inference: {elapsed:.3f}s")

                depth = prediction["depth"].cpu().numpy()
                depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
                Image.fromarray(depth_normalized).save(output)

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
