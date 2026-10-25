#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
Flicker of depth videos written by `spatial-maker --output-types depth`: mean per-pixel change between
consecutive frames, top half only (same measure as flicker_check.py). Lower = more stable.

Usage: uv run analysis/video_flicker.py <depth.mov>...
"""

import json
import subprocess
import sys

import numpy as np


def frame_size(path):
	probe = subprocess.run(
		[
			'ffprobe',
			'-v',
			'error',
			'-select_streams',
			'v:0',
			'-show_entries',
			'stream=width,height',
			'-of',
			'json',
			path,
		],
		capture_output=True,
		check=True,
	)
	stream = json.loads(probe.stdout)['streams'][0]
	return stream['width'], stream['height']


def gray_frames(path):
	width, height = frame_size(path)
	raw = subprocess.run(
		['ffmpeg', '-v', 'error', '-i', path, '-f', 'rawvideo', '-pix_fmt', 'gray', '-'],
		capture_output=True,
		check=True,
	).stdout
	frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width).astype(np.float32)
	assert len(frames) >= 2, f'{path}: need 2+ frames'
	return frames


print('| depth video | frames | mean frame-to-frame delta (top half, 0-255) |')
print('|-|-|-|')
for path in sys.argv[1:]:
	frames = gray_frames(path)
	top = frames[:, : frames.shape[1] // 2]
	print(f'| {path.rsplit("/", 1)[-1]} | {len(frames)} | {np.abs(np.diff(top, axis=0)).mean():.2f} |')
