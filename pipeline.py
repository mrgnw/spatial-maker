#!/usr/bin/env python3
"""
End-to-end pipeline for converting 2D videos to spatial 3D videos.

Usage:
    spatial-maker video.mp4
    spatial-maker /path/to/folder
    spatial-maker video.mp4 --encoder l --max-disparity 40
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil

from scripts.downscale_video import downscale_to_1080p24
from scripts.depth_to_stereo import process_video_to_sbs

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".wmv", ".flv"}
ENCODER_MAP = {"s": "vits", "m": "vitb", "l": "vitl"}


def check_spatial_cli():
    """Check if the spatial CLI is available."""
    try:
        result = subprocess.run(["spatial", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def find_videos_in_folder(folder: Path) -> list[Path]:
    """Find all video files in a folder (non-recursive)."""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(folder.glob(f"*{ext}"))
        videos.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(videos))


def extract_audio(video_path: str, output_path: str) -> bool:
    """Extract audio from video to AAC file. Returns True if audio exists."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "aac",
        "-b:a", "192k",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and Path(output_path).exists()


def mux_audio_to_video(video_path: str, audio_path: str, output_path: str) -> str:
    """Mux audio into video file."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # If muxing fails, return original video without audio
        return video_path
    return output_path


def run_pipeline(
    input_video: str,
    output_path: str = None,
    encoder: str = "vits",
    max_disparity: int = 40,
    keep_intermediate: bool = False,
    skip_downscale: bool = False,
    duration: float = None,
):
    """Run the full 2D to spatial video conversion pipeline."""
    input_path = Path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    if output_path is None:
        output_path = Path("output") / f"{input_path.stem}_spatial.mov"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_path.parent / f"{output_path.stem}_work"
    work_dir.mkdir(exist_ok=True)

    try:
        # Stage 1: Downscale
        if skip_downscale:
            prepared_video = str(input_path)
        else:
            print("↓ Downscaling to 1080p24...")
            prepared_video = downscale_to_1080p24(
                str(input_path),
                output_path=str(work_dir / f"{input_path.stem}_1080p.mp4") if duration else None,
                duration=duration,
            )

        # Stage 2: Depth + SBS
        print(f"◐ Generating depth & stereo (encoder={encoder[-1]}, disparity={max_disparity})...")
        sbs_video = work_dir / f"{input_path.stem}_sbs.mp4"
        max_frames = int(duration * 24) if duration else None
        sbs_result = process_video_to_sbs(
            prepared_video,
            str(sbs_video),
            encoder=encoder,
            max_disparity=max_disparity,
            fps=24,
            max_frames=max_frames,
        )

        # Stage 2.5: Add audio from original video to SBS
        audio_file = work_dir / f"{input_path.stem}_audio.aac"
        if extract_audio(str(input_path), str(audio_file)):
            sbs_with_audio = work_dir / f"{input_path.stem}_sbs_audio.mp4"
            sbs_video = mux_audio_to_video(str(sbs_video), str(audio_file), str(sbs_with_audio))
        else:
            sbs_video = str(sbs_video)

        # Stage 3: Spatial video
        print("◉ Creating spatial video...")
        if not check_spatial_cli():
            print("⚠ 'spatial' CLI not found. Install: brew install spatial")
            print(f"  SBS video: {sbs_result['output_path']}")
            return sbs_result['output_path']

        cmd = [
            "spatial", "make",
            "-i", sbs_video if isinstance(sbs_video, str) else str(sbs_video),
            "-f", "sbs",
            "-o", str(output_path),
            "-q", "0.9",
            "-y",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"spatial CLI failed: {result.stderr}")

        return str(output_path)

    finally:
        if not keep_intermediate and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


def run_batch(
    folder: Path,
    encoder: str = "vits",
    max_disparity: int = 40,
    keep_intermediate: bool = False,
    skip_downscale: bool = False,
    duration: float = None,
):
    """Process all videos in a folder."""
    videos = find_videos_in_folder(folder)

    if not videos:
        print(f"No video files found in {folder}")
        return []

    output_folder = folder / "spatial"
    output_folder.mkdir(exist_ok=True)

    print(f"\n◎ Processing {len(videos)} videos → {output_folder}\n")

    results = []
    failed = []

    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name}")
        output_path = output_folder / f"{video.stem}_spatial.mov"

        try:
            result = run_pipeline(
                str(video),
                output_path=str(output_path),
                encoder=encoder,
                max_disparity=max_disparity,
                keep_intermediate=keep_intermediate,
                skip_downscale=skip_downscale,
                duration=duration,
            )
            results.append(result)
            print(f"    ✓ Done\n")
        except Exception as e:
            failed.append((video, str(e)))
            print(f"    ✗ Failed: {e}\n")

    # Summary
    print(f"{'─' * 40}")
    print(f"✓ {len(results)}/{len(videos)} completed")
    if failed:
        for video, error in failed:
            print(f"✗ {video.name}: {error}")
    print(f"→ {output_folder}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert 2D video to spatial 3D video for VR headsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    spatial-maker video.mp4
    spatial-maker /path/to/videos/
    spatial-maker video.mp4 --encoder l
    spatial-maker video.mp4 --duration 10
        """,
    )
    parser.add_argument("input", help="Input video file or folder")
    parser.add_argument("-o", "--output", help="Output path (default: output/name_spatial.mov)")
    parser.add_argument(
        "--encoder", choices=["s", "m", "l"], default="s",
        help="Model size: s=small, m=medium, l=large (default: s)",
    )
    parser.add_argument(
        "--max-disparity", type=int, default=40,
        help="3D intensity, 20-50 for 1080p (default: 40)",
    )
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate files")
    parser.add_argument("--skip-downscale", action="store_true", help="Skip 1080p24 downscale")
    parser.add_argument("--duration", type=float, help="Process only first N seconds")

    args = parser.parse_args()
    input_path = Path(args.input)
    encoder = ENCODER_MAP[args.encoder]

    try:
        if input_path.is_dir():
            if args.output:
                print("Note: --output ignored for folders, using <folder>/spatial/")
            results = run_batch(
                input_path,
                encoder=encoder,
                max_disparity=args.max_disparity,
                keep_intermediate=args.keep_intermediate,
                skip_downscale=args.skip_downscale,
                duration=args.duration,
            )
            if not results:
                sys.exit(1)
        else:
            print(f"\n◎ {input_path.name}\n")
            result = run_pipeline(
                args.input,
                output_path=args.output,
                encoder=encoder,
                max_disparity=args.max_disparity,
                keep_intermediate=args.keep_intermediate,
                skip_downscale=args.skip_downscale,
                duration=args.duration,
            )
            print(f"\n✓ {result}\n")
    except Exception as e:
        print(f"\n✗ {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
