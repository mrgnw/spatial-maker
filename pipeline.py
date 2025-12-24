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
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil

from scripts.downscale_video import downscale_to_1080p24
from scripts.depth_to_stereo import process_video_to_sbs


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


def run_pipeline(
    input_video: str,
    output_path: str = None,
    encoder: str = "vits",
    max_disparity: int = 30,
    keep_intermediate: bool = False,
    skip_downscale: bool = False,
    duration: float = None,
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

    try:
        # Stage 1: Downscale to 1080p @ 24fps
        print("\n" + "=" * 60)
        print("Stage 1: Preparing input video (1080p @ 24fps)")
        print("=" * 60)

        if skip_downscale:
            prepared_video = str(input_path)
            print(f"  Skipping downscale, using: {prepared_video}")
        else:
            prepared_video = downscale_to_1080p24(
                str(input_path),
                output_path=str(work_dir / f"{input_path.stem}_1080p.mp4") if duration else None,
                duration=duration,
            )
            print(f"  Prepared video: {prepared_video}")

        # Stage 2: Generate depth and create SBS stereoscopic video
        print("\n" + "=" * 60)
        print(f"Stage 2: Depth estimation + SBS stereo (encoder: {encoder}, disparity: {max_disparity})")
        print("=" * 60)

        sbs_video = work_dir / f"{input_path.stem}_sbs.mp4"
        sbs_result = process_video_to_sbs(
            prepared_video,
            str(sbs_video),
            encoder=encoder,
            max_disparity=max_disparity,
            fps=24,
        )
        print(f"  Processed {sbs_result['frames_processed']} frames")
        print(f"  SBS video: {sbs_result['output_path']}")

        # Stage 3: Convert to spatial video using spatial CLI
        print("\n" + "=" * 60)
        print("Stage 3: Converting to MV-HEVC spatial video")
        print("=" * 60)

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
                "-y",
            ]
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  Error: {result.stderr}")
                raise RuntimeError("spatial CLI failed")
            print(f"  Spatial video: {output_path}")
            final_output = str(output_path)

        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print("=" * 60)
        print(f"  Output: {final_output}")

        return final_output

    finally:
        # Cleanup intermediate files unless requested to keep them
        if not keep_intermediate and work_dir.exists():
            print(f"\nCleaning up intermediate files in {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Convert 2D video to spatial 3D video for VR headsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (uses vits encoder, default settings)
    python pipeline.py video.mp4

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
    parser.add_argument("input", help="Input 2D video file")
    parser.add_argument(
        "-o", "--output", help="Output spatial video path (default: input_spatial.mov)"
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

    try:
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
