#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


def downscale_to_1080p24(input_path: str, output_path: str = None, duration: float = None):
    if output_path is None:
        p = Path(input_path)
        output_path = f"{p.stem}_1080p{p.suffix}"

    cmd = ["ffmpeg", "-y", "-i", input_path]

    if duration is not None:
        cmd.extend(["-t", str(duration)])

    cmd.extend([
        "-vf", "scale='min(1920,iw)':'min(1080,ih)':force_original_aspect_ratio=decrease",
        "-r", "24",
        "-c:v", "h264_videotoolbox",
        "-b:v", "8M",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path
    ])
    subprocess.run(cmd, check=True)
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python downscale_video.py <input> [output]")
        sys.exit(1)

    output = sys.argv[2] if len(sys.argv) > 2 else None
    result = downscale_to_1080p24(sys.argv[1], output)
    print(f"Created: {result}")
