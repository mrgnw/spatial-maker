import time
import subprocess
from pathlib import Path
import tempfile
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

        self.model, self.transform = depth_pro.create_model_and_transforms()
        self.model.eval()

        self.device = torch.device("cpu")

    def process_video(self, video_path: str, output_dir: str):
        self.load_model()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / "frames"
            frames_dir.mkdir()

            cmd = ["ffmpeg", "-i", video_path, f"{frames_dir}/frame_%04d.png"]
            subprocess.run(cmd, capture_output=True, check=True)

            frames = sorted(frames_dir.glob("frame_*.png"))
            frame_times = []

            for frame in frames:
                output = output_dir / f"depth_{frame.name}"

                image, _, f_px = depth_pro.load_rgb(str(frame))
                image = self.transform(image)

                start = time.time()
                with torch.no_grad():
                    prediction = self.model.infer(image, f_px=f_px)
                frame_times.append(time.time() - start)

                depth = prediction["depth"].cpu().numpy()
                depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
                Image.fromarray(depth_normalized).save(output)

            total_time = sum(frame_times)
            return {
                "frames_processed": len(frame_times),
                "total_time": total_time,
                "avg_time_per_frame": total_time / len(frame_times),
                "device": str(self.device)
            }
