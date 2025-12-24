#!/usr/bin/env python3
"""
End-to-end pipeline for converting 2D videos to spatial 3D videos.

Pipeline stages:
1. Downscale input video to 1080p @ 24fps
2. Generate depth maps using Depth Anything V2
3. Create side-by-side stereoscopic video
4. Convert to MV-HEVC spatial video using spatial CLI

Usage:
    python pipeline.py input_video.mp4 -o output_spatial.mov
    python pipeline.py input_video.mp4 --encoder vitb --max-disparity 40
    python pipeline.py /path/to/folder  # Process all videos in folder
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil

from scripts.downscale_video import downscale_to_1080p24
from scripts.depth_to_stereo import process_video_to_sbs

# Video formats to process when given a folder
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".wmv", ".flv"}


def check_spatial_cli():
    """Check if the spatial CLI is available."""
    try:
        result = subprocess.run(
            ["spatial", "--version"],
            capture_output=True,
            text=True,
        )
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


def run_pipeline(
    input_video: str,
    output_path: str = None,
    encoder: str = "vits",
    max_disparity: int = 30,
    keep_intermediate: bool = False,
    skip_downscale: bool = False,
    duration: float = None,
    quiet: bool = False,
):
    """
    Run the full 2D to spatial video conversion pipeline.

    Args:
        input_video: Path to input 2D video
        output_path: Path for output spatial video (default: input_spatial.mov)
        encoder: Depth Anything V2 encoder (vits, vitb, vitl)
        max_disparity: Maximum pixel disparity for 3D effect (20-50 for 1080p)
        keep_intermediate: Keep intermediate files (depth frames, SBS video)
        skip_downscale: Skip downscaling if input is already 1080p24
        duration: Optional duration limit in seconds
        quiet: Suppress stage headers (for batch processing)

    Returns:
        Path to the output spatial video
    """
    input_path = Path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    # Set up output paths
    if output_path is None:
        output_path = Path("output") / f"{input_path.stem}_spatial.mov"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create working directory for intermediate files
    work_dir = output_path.parent / f"{output_path.stem}_work"
    work_dir.mkdir(exist_ok=True)

    def print_stage(msg):
        if not quiet:
            print(msg)

    try:
        # Stage 1: Downscale to 1080p @ 24fps
        print_stage("\n" + "=" * 60)
        print_stage("Stage 1: Preparing input video (1080p @ 24fps)")
        print_stage("=" * 60)

        if skip_downscale:
            prepared_video = str(input_path)
            print_stage(f"  Skipping downscale, using: {prepared_video}")
        else:
            prepared_video = downscale_to_1080p24(
                str(input_path),
                output_path=str(work_dir / f"{input_path.stem}_1080p.mp4") if duration else None,
                duration=duration,
            )
            print_stage(f"  Prepared video: {prepared_video}")

        # Stage 2: Generate depth and create SBS stereoscopic video
        print_stage("\n" + "=" * 60)
        print_stage(f"Stage 2: Depth estimation + SBS stereo (encoder: {encoder}, disparity: {max_disparity})")
        print_stage("=" * 60)

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
        print_stage(f"  Processed {sbs_result['frames_processed']} frames")
        print_stage(f"  SBS video: {sbs_result['output_path']}")

        # Stage 3: Convert to spatial video using spatial CLI
        print_stage("\n" + "=" * 60)
        print_stage("Stage 3: Converting to MV-HEVC spatial video")
        print_stage("=" * 60)

        if not check_spatial_cli():
            print("  WARNING: 'spatial' CLI not found!")
            print("  Install from: https://blog.mikeswanson.com/spatial")
            print(f"  SBS video ready at: {sbs_result}")
            print("  Run manually: spatial make --sbs <input> -o <output>")
            final_output = sbs_result
        else:
            cmd = [
                "spatial",
                "make",
                "-i",
                str(sbs_video),
                "-f",
                "sbs",
                "-o",
                str(output_path),
                "-q",
                "0.9",  # High quality (0.0-1.0), ~48 Mbps output
                "-y",
            ]
            print_stage(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  Error: {result.stderr}")
                raise RuntimeError("spatial CLI failed")
            print_stage(f"  Spatial video: {output_path}")
            final_output = str(output_path)

        print_stage("\n" + "=" * 60)
        print_stage("Pipeline complete!")
        print_stage("=" * 60)
        print_stage(f"  Output: {final_output}")

        return final_output

    finally:
        # Cleanup intermediate files unless requested to keep them
        if not keep_intermediate and work_dir.exists():
            print_stage(f"\nCleaning up intermediate files in {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)


def run_batch(
    folder: Path,
    encoder: str = "vits",
    max_disparity: int = 30,
    keep_intermediate: bool = False,
    skip_downscale: bool = False,
    duration: float = None,
):
    """
    Process all videos in a folder.

    Args:
        folder: Path to folder containing videos
        encoder: Depth Anything V2 encoder (vits, vitb, vitl)
        max_disparity: Maximum pixel disparity for 3D effect
        keep_intermediate: Keep intermediate files
        skip_downscale: Skip downscaling
        duration: Optional duration limit in seconds

    Returns:
        List of output paths
    """
    videos = find_videos_in_folder(folder)

    if not videos:
        print(f"No video files found in {folder}")
        print(f"Supported formats: {', '.join(sorted(VIDEO_EXTENSIONS))}")
        return []

    # Create output folder
    output_folder = folder / "spatial"
    output_folder.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Batch processing: {len(videos)} videos")
    print(f"Output folder: {output_folder}")
    print(f"{'=' * 60}\n")

    results = []
    failed = []

    for i, video in enumerate(videos, 1):
        print(f"\n{'#' * 60}")
        print(f"# [{i}/{len(videos)}] Processing: {video.name}")
        print(f"{'#' * 60}")

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
                quiet=False,
            )
            results.append(result)
            print(f"\n✓ [{i}/{len(videos)}] Completed: {video.name}")
        except Exception as e:
            failed.append((video, str(e)))
            print(f"\n✗ [{i}/{len(videos)}] Failed: {video.name} - {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Batch processing complete!")
    print(f"{'=' * 60}")
    print(f"  Successful: {len(results)}/{len(videos)}")
    if failed:
        print(f"  Failed: {len(failed)}/{len(videos)}")
        for video, error in failed:
            print(f"    - {video.name}: {error}")
    print(f"  Output folder: {output_folder}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert 2D video to spatial 3D video for VR headsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single file
    python pipeline.py video.mp4

    # Process all videos in a folder
    python pipeline.py /path/to/videos/

    # Use larger encoder for better depth quality
    python pipeline.py video.mp4 --encoder vitl

    # Adjust 3D intensity
    python pipeline.py video.mp4 --max-disparity 40

    # Process only first 10 seconds
    python pipeline.py video.mp4 --duration 10

    # Keep intermediate files for debugging
    python pipeline.py video.mp4 --keep-intermediate
        """,
    )
    parser.add_argument(
        "input",
        help="Input 2D video file or folder containing videos",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output spatial video path (default: output/input_spatial.mov, or input_folder/spatial/ for folders)",
    )
    parser.add_argument(
        "--encoder",
        choices=["vits", "vitb", "vitl"],
        default="vits",
        help="Depth Anything V2 encoder size (default: vits)",
    )
    parser.add_argument(
        "--max-disparity",
        type=int,
        default=30,
        help="Maximum pixel disparity for 3D effect, 20-50 for 1080p (default: 30)",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate files (depth frames, SBS video)",
    )
    parser.add_argument(
        "--skip-downscale",
        action="store_true",
        help="Skip downscaling (use if input is already 1080p @ 24fps)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Process only the first N seconds of the video",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    try:
        if input_path.is_dir():
            # Batch mode - process all videos in folder
            if args.output:
                print("Warning: -o/--output is ignored in folder mode. Output goes to <folder>/spatial/")
            results = run_batch(
                input_path,
                encoder=args.encoder,
                max_disparity=args.max_disparity,
                keep_intermediate=args.keep_intermediate,
                skip_downscale=args.skip_downscale,
                duration=args.duration,
            )
            if not results:
                sys.exit(1)
        else:
            # Single file mode
            result = run_pipeline(
                args.input,
                output_path=args.output,
                encoder=args.encoder,
                max_disparity=args.max_disparity,
                keep_intermediate=args.keep_intermediate,
                skip_downscale=args.skip_downscale,
                duration=args.duration,
            )
            print(f"\nSuccess! Output: {result}")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
