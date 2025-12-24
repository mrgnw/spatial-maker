#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


def downscale_to_1080p24(input_path: str, output_path: str = None, duration: float = None):
    input_p = Path(input_path)

    cached = Path("samples/1080p24fps") / f"{input_p.stem}_1080p{input_p.suffix}"

    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", "scale='min(1920,iw)':'min(1080,ih)':force_original_aspect_ratio=decrease",
            "-r", "24",
            "-c:v", "h264_videotoolbox",
            "-b:v", "8M",
            "-c:a", "aac",
            "-b:a", "192k",
            str(cached)
        ]
        subprocess.run(cmd, check=True)

    if duration is not None:
        if output_path is None:
            raise ValueError("output_path required when duration is specified")
        cmd = [
            "ffmpeg", "-y", "-i", str(cached),
            "-t", str(duration),
            "-c", "copy",
            output_path
        ]
        subprocess.run(cmd, check=True)
        return output_path

    if output_path is None:
        return str(cached)

    Path(cached).replace(output_path)
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python downscale_video.py <input> [output]")
        sys.exit(1)

    output = sys.argv[2] if len(sys.argv) > 2 else None
    result = downscale_to_1080p24(sys.argv[1], output)
    print(f"Created: {result}")
