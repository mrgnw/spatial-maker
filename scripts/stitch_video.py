import subprocess
from pathlib import Path


def stitch_frames_to_hevc(frames_dir: str, output_path: str, fps: int = 24):
    """
    Stitch depth map frames into an HEVC video using ffmpeg with VideoToolbox hardware acceleration.

    Args:
        frames_dir: Directory containing depth_frame_*.png files
        output_path: Path for output HEVC video
        fps: Frame rate (default 24)
    """
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find frame pattern - supports both depth_frame_*.png and depth_*.png
    if list(frames_dir.glob("depth_frame_*.png")):
        pattern = frames_dir / "depth_frame_%04d.png"
    elif list(frames_dir.glob("depth_*.png")):
        pattern = frames_dir / "depth_%04d.png"
    else:
        raise ValueError(f"No depth frames found in {frames_dir}")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(pattern),
        "-c:v", "hevc_videotoolbox",
        "-b:v", "8M",
        "-tag:v", "hvc1",
        str(output_path)
    ]

    subprocess.run(cmd, check=True, capture_output=True)
    return str(output_path)
