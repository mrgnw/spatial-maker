#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pillow"]
# ///
"""Mean per-pixel change between consecutive depth frames, static top-half only. Lower = more stable."""

from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).parent / 'out'
MODELS = ['da2-small', 'da2-base', 'da2-large', 'distill-small', 'da3mono-large']

print('| model | mean frame-to-frame delta (top half, 0-255) |')
print('|---|---|')
for model in MODELS:
	frames = sorted((OUT / model).glob('frame_*.png'))
	assert len(frames) >= 2, f'{model}: need 2+ frames'
	maps = [np.asarray(Image.open(f), dtype=np.float32) for f in frames]
	h = maps[0].shape[0] // 2
	deltas = [np.abs(b[:h] - a[:h]).mean() for a, b in zip(maps, maps[1:])]
	print(f'| {model} | {np.mean(deltas):.2f} |')
