"""
Stereo Converter - Create stereoscopic 3D videos from 2D video + depth maps.

Uses Depth-Image Based Rendering (DIBR) to synthesize left and right eye views.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import subprocess
import shutil


def create_stereo_pair(
    rgb_frame: np.ndarray,
    depth_frame: np.ndarray,
    max_disparity: int = 30,
) -> tuple:
    """
    Create left and right eye views from RGB frame and depth map.

    Args:
        rgb_frame: Original RGB image (H, W, 3)
        depth_frame: Normalized depth map (H, W), values 0-255
        max_disparity: Maximum pixel shift for closest objects (controls 3D intensity)

    Returns:
        (left_eye, right_eye) tuple of images
    """
    h, w = rgb_frame.shape[:2]

    # Normalize depth to 0-1
    if depth_frame.max() > 1:
        depth_normalized = depth_frame.astype(np.float32) / 255.0
    else:
        depth_normalized = depth_frame.astype(np.float32)

    # Calculate disparity map (closer = higher value = more shift)
    disparity = (depth_normalized * max_disparity).astype(np.float32)

    # Create coordinate grids
    x_coords = np.arange(w, dtype=np.float32)
    y_coords = np.arange(h, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # Shift for left eye (shift right for closer objects)
    x_left = x_grid + disparity / 2
    # Shift for right eye (shift left for closer objects)
    x_right = x_grid - disparity / 2

    # Remap using cv2 (handles interpolation and boundary)
    left_eye = cv2.remap(
        rgb_frame,
        x_left,
        y_grid,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    right_eye = cv2.remap(
        rgb_frame,
        x_right,
        y_grid,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return left_eye, right_eye


def create_sbs_frame(
    left_eye: np.ndarray, right_eye: np.ndarray, half_width: bool = True
) -> np.ndarray:
    """
    Create side-by-side stereoscopic frame.

    Args:
        left_eye: Left eye view
        right_eye: Right eye view
        half_width: If True, compress each eye to half width (standard SBS format)

    Returns:
        Side-by-side frame (left | right)
    """
    if half_width:
        h, w = left_eye.shape[:2]
        left_half = cv2.resize(left_eye, (w // 2, h))
        right_half = cv2.resize(right_eye, (w // 2, h))
        return np.hstack([left_half, right_half])
    else:
        return np.hstack([left_eye, right_eye])


def convert_video_to_sbs(
    rgb_video_path: str,
    depth_frames_dir: str,
    output_path: str,
    max_disparity: int = 30,
    fps: int = 24,
    half_width: bool = True,
):
    """
    Convert 2D video + depth frames to side-by-side stereoscopic video.

    Args:
        rgb_video_path: Path to original RGB video
        depth_frames_dir: Directory containing depth_frame_*.png files
        output_path: Output path for SBS video
        max_disparity: Maximum pixel shift (controls 3D intensity, 20-50 for 1080p)
        fps: Output frame rate
        half_width: If True, create half-width SBS (maintains original resolution)

    Returns:
        Path to the output SBS video
    """
    depth_dir = Path(depth_frames_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load depth frames
    depth_files = sorted(depth_dir.glob("depth_frame_*.png"))
    if not depth_files:
        depth_files = sorted(depth_dir.glob("depth_*.png"))
    if not depth_files:
        raise ValueError(f"No depth frames found in {depth_dir}")

    # Open video
    cap = cv2.VideoCapture(str(rgb_video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {rgb_video_path}")

    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(depth_files))

    # Get dimensions from first frame
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame from video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    h, w = first_frame.shape[:2]

    # Create temporary frame directory for ffmpeg
    temp_dir = output_path.parent / f"{output_path.stem}_sbs_frames"
    temp_dir.mkdir(exist_ok=True)

    try:
        for i in tqdm(range(total_frames), desc="Creating stereo pairs"):
            ret, rgb_frame = cap.read()
            if not ret:
                break

            # Load corresponding depth frame
            depth_frame = cv2.imread(str(depth_files[i]), cv2.IMREAD_GRAYSCALE)
            if depth_frame is None:
                raise ValueError(f"Could not load depth frame: {depth_files[i]}")

            # Resize depth to match RGB if needed
            if depth_frame.shape[:2] != rgb_frame.shape[:2]:
                depth_frame = cv2.resize(depth_frame, (w, h))

            # Create stereo pair
            left_eye, right_eye = create_stereo_pair(
                rgb_frame, depth_frame, max_disparity
            )

            # Create SBS frame
            sbs_frame = create_sbs_frame(left_eye, right_eye, half_width)

            # Save frame
            cv2.imwrite(str(temp_dir / f"sbs_{i+1:04d}.png"), sbs_frame)

        cap.release()

        # Encode to HEVC using VideoToolbox hardware acceleration
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(temp_dir / "sbs_%04d.png"),
            "-c:v",
            "hevc_videotoolbox",
            "-b:v",
            "12M",
            "-tag:v",
            "hvc1",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    finally:
        # Cleanup temp frames
        shutil.rmtree(temp_dir, ignore_errors=True)

    return str(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert 2D video + depth to SBS stereoscopic video"
    )
    parser.add_argument("rgb_video", help="Path to input RGB video")
    parser.add_argument("depth_dir", help="Directory containing depth frames")
    parser.add_argument("output", help="Output path for SBS video")
    parser.add_argument(
        "--max-disparity",
        type=int,
        default=30,
        help="Maximum pixel shift for 3D effect (default: 30)",
    )
    parser.add_argument("--fps", type=int, default=24, help="Output FPS (default: 24)")
    parser.add_argument(
        "--full-width",
        action="store_true",
        help="Output full width SBS (double width) instead of half-width",
    )

    args = parser.parse_args()

    result = convert_video_to_sbs(
        args.rgb_video,
        args.depth_dir,
        args.output,
        max_disparity=args.max_disparity,
        fps=args.fps,
        half_width=not args.full_width,
    )
    print(f"Created SBS video: {result}")
