#!/usr/bin/env python3
import sys
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm


def create_ass_subtitle(title: str, duration: float, position: str = "bottom_center") -> str:
    """Create an ASS subtitle entry with positioning."""
    # ASS format: Dialogue: layer, start, end, style, name, marginL, marginR, marginV, effect, text
    # Position codes: \pos(x,y) for absolute positioning, or use alignment
    # Alignment: 1=bottom-left, 2=bottom-center, 3=bottom-right, etc.

    start_time = "00:00:00.00"
    end_time = format_timestamp(duration)

    # Use alignment 2 for bottom-center
    dialogue = f"Dialogue: 0,{start_time},{end_time},TitleStyle,,0,0,0,,{title}"
    return dialogue


def format_timestamp(seconds: float) -> str:
    """Convert seconds to ASS timestamp format (HH:MM:SS.CC)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def create_ass_file(title: str, duration: float) -> str:
    """Create a complete ASS subtitle file with proper formatting."""
    ass_content = """[Script Info]
Title: Comparison Video
ScriptType: v4.00+
Collisions: Normal
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TitleStyle,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,3,2,2,10,10,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    start_time = "00:00:00.00"
    end_time = format_timestamp(duration)

    dialogue = f"Dialogue: 0,{start_time},{end_time},TitleStyle,,0,0,0,,{title}"
    ass_content += dialogue

    return ass_content


def get_video_title(video_path: str) -> str:
    """Extract a clean title from video path."""
    path = Path(video_path)
    return path.stem


def get_short_title(full_title: str) -> str:
    """Convert full title to short compact name for subtitles."""
    # Map long names to short names
    short_names = {
        'depth_anything_v2_small': 'dav2_small',
        'depth_anything_v2_base': 'dav2_base',
        'depth_anything_v2_large': 'dav2_large',
        'apple_depth_pro': 'adp',
    }

    # Check for exact matches first
    if full_title in short_names:
        return short_names[full_title]

    # Check for partial matches (in case of old bench names)
    for long_name, short_name in short_names.items():
        if long_name in full_title:
            return short_name

    # Return original if no mapping found
    return full_title


def get_video_duration(video_path: str) -> float:
    """Get video duration using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1:noprint_wrappers=1",
                video_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        raise ValueError(f"Failed to get duration of {video_path}: {e}")


def compare_videos(video_paths: list, output_path: str = "comparison.mp4"):
    """Create a side-by-side comparison video with subtitles."""

    if not video_paths:
        raise ValueError("No video paths provided")

    # Resolve video paths
    resolved_paths = []
    for video_path in video_paths:
        # Handle both with and without extension
        if not video_path.endswith(('.mp4', '.MOV', '.mov', '.avi', '.mkv')):
            test_path = video_path + '.mp4'
            if Path(test_path).exists():
                video_path = test_path
            elif not Path(video_path).exists():
                print(f"Warning: Video not found: {video_path}")
                continue

        if not Path(video_path).exists():
            print(f"Warning: Video not found: {video_path}")
            continue

        resolved_paths.append(video_path)

    if not resolved_paths:
        raise ValueError("No valid videos found")

    print(f"Creating comparison video from {len(resolved_paths)} videos...")

    # Get video information
    videos_info = []
    max_duration = 0

    for video_path in resolved_paths:
        try:
            duration = get_video_duration(video_path)
            title = get_video_title(video_path)
            videos_info.append({
                'path': video_path,
                'title': title,
                'duration': duration
            })
            max_duration = max(max_duration, duration)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue

    if not videos_info:
        raise ValueError("No valid videos could be processed")

    # Create subtitle files for each video
    subtitle_files = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for i, video_info in enumerate(videos_info):
            short_title = get_short_title(video_info['title'])
            ass_content = create_ass_file(short_title, max_duration)
            sub_file = tmpdir / f"subtitle_{i}.ass"
            sub_file.write_text(ass_content)
            subtitle_files.append(str(sub_file))

        # Build ffmpeg complex filter for side-by-side layout with subtitles
        inputs = []
        filter_parts = []

        # Add video inputs
        for i, video_info in enumerate(videos_info):
            inputs.extend(["-i", video_info['path']])

        # Build the filter graph
        # Scale videos and add subtitles
        scaled_filters = []
        subtitle_filters = []

        for i in range(len(videos_info)):
            # Scale to 720p height, round width to even number for codec compatibility
            scaled_filters.append(f"[{i}]scale=-1:720,pad=ceil(iw/2)*2:ceil(ih/2)*2[v{i}]")
            # Add subtitle overlay using absolute path
            sub_path = subtitle_files[i].replace("\\", "/").replace("'", "\\'")  # Escape for ffmpeg filter
            subtitle_filters.append(f"[v{i}]subtitles='{sub_path}'[v{i}_sub]")

        # Concatenate all videos horizontally
        concat_inputs = "".join([f"[v{i}_sub]" for i in range(len(videos_info))])
        concat_filter = f"{concat_inputs}hstack=inputs={len(videos_info)}"

        filter_graph = ";".join(scaled_filters + subtitle_filters) + f";{concat_filter}"

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex", filter_graph,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            output_path
        ]

        print(f"Output: {output_path}")
        print("Encoding comparison video...")

        try:
            subprocess.run(cmd, check=True)
            print(f"Comparison video saved to: {output_path}")
        except subprocess.CalledProcessError as e:
            raise ValueError(f"FFmpeg encoding failed: {e}")


def generate_default_output_name(video_paths: list) -> str:
    """Generate a readable default output filename from input video names."""
    titles = [get_video_title(path) for path in video_paths]
    # Join titles with underscores, limit length
    combined = "_vs_".join(titles)
    if len(combined) > 60:
        # If too long, abbreviate
        combined = "_vs_".join([t[:10] for t in titles])
    return f"{combined}_comparison.mp4"


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run compare <video1> <video2> [video3] ... [--output output.mp4]")
        print("Example: uv run compare dav2_base.mp4 dav2_large dav2_small")
        sys.exit(1)

    # Parse arguments
    video_paths = []
    output_path = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--output" and i + 1 < len(args):
            output_path = args[i + 1]
            i += 2
        elif not args[i].startswith("--"):
            video_paths.append(args[i])
            i += 1
        else:
            i += 1

    # Generate default output name if not specified
    if output_path is None:
        output_path = generate_default_output_name(video_paths)

    try:
        compare_videos(video_paths, output_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
