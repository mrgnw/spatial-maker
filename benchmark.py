#!/usr/bin/env python3
import json
import csv
import sys
from pathlib import Path
import tempfile
from depth_estimators import AppleDepthPro
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from downscale_video import downscale_to_1080p24


def save_results(results: dict, output_dir: str = "benchmarks"):
    Path(output_dir).mkdir(exist_ok=True)

    json_path = Path(output_dir) / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    tsv_path = Path(output_dir) / "results.tsv"
    with open(tsv_path, "w", newline='') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["model", "frames", "total_time", "avg_time", "fps"])
        for model_name, stats in results.items():
            writer.writerow([
                model_name,
                stats["frames_processed"],
                stats["total_time"],
                stats["avg_time_per_frame"],
                round(1 / stats["avg_time_per_frame"], 2)
            ])

    print(f"\nResults saved to {output_dir}/")


def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <video_path> [sample_seconds]")
        sys.exit(1)

    video_path = sys.argv[1]
    sample_seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    output_base = Path("output")
    output_base.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        prepared_video = Path(tmpdir) / "sample.mp4"

        print(f"Downscaling to 1080p24fps and trimming to {sample_seconds}s...")
        downscale_to_1080p24(video_path, str(prepared_video), duration=sample_seconds)

        results = {}

        print("\nBenchmarking Apple Depth Pro...")
        depth_dir = output_base / "apple_depth_pro"
        estimator = AppleDepthPro()
        results["apple_depth_pro"] = estimator.process_video(str(prepared_video), str(depth_dir))

        print("\n" + "="*60)
        for model_name, stats in results.items():
            print(f"{model_name}:")
            print(f"  Frames: {stats['frames_processed']}")
            print(f"  Total: {stats['total_time']:.3f}s")
            print(f"  Avg/frame: {stats['avg_time_per_frame']:.3f}s")
            print(f"  FPS: {1/stats['avg_time_per_frame']:.2f}")
        print("="*60)

        save_results(results)


if __name__ == "__main__":
    main()
