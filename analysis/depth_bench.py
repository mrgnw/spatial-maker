#!/usr/bin/env -S uv run --python 3.12 --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "coremltools==9.0",
#     "numpy<2",
#     "pillow",
# ]
# ///
"""
Benchmark CoreML depth models on the M4 Pro: warm ms/frame + depth map PNGs for A/B.
Usage: uv run analysis/depth_bench.py
"""

import os
import statistics
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image

HOME_CKPT = Path.home() / '.spatial-maker/checkpoints'
REPO = Path(__file__).parent.parent
OUT = Path(os.environ.get('DEPTH_BENCH_OUT', REPO / 'analysis/out'))
CANDIDATES = Path(os.environ.get('DEPTH_BENCH_CANDIDATES', HOME_CKPT))

MODELS = {
	'da2-small': HOME_CKPT / 'DepthAnythingV2SmallF16.mlpackage',
	'da2-base': HOME_CKPT / 'DepthAnythingV2BaseF16.mlpackage',
	'da2-large': HOME_CKPT / 'DepthAnythingV2LargeF16.mlpackage',
	'distill-small': HOME_CKPT / 'DistillAnyDepthSmallF16.mlpackage',
	'da3mono-large': HOME_CKPT / 'DepthAnythingV3Mono.mlpackage',
	'da3mono-large-new': CANDIDATES / 'DepthAnythingV3Mono.mlpackage',
	'da3metric-large': CANDIDATES / 'DepthAnythingV3MetricLarge.mlpackage',
	'da3-small': CANDIDATES / 'DepthAnythingV3Small.mlpackage',
	'da3-base': CANDIDATES / 'DepthAnythingV3Base.mlpackage',
}

INVERT_OUTPUT = {n for n in MODELS if n.startswith('da3')}

TEST_IMAGES = [REPO / 'test_baseline.jpg'] + sorted((OUT / 'frames').glob('frame_*.jpg'))

TIMING_ITERS = 10
WARMUP_ITERS = 3


def input_spec(model):
	spec = model.get_spec()
	inp = spec.description.input[0]
	assert inp.type.HasField('imageType'), f'expected image input, got {inp.type}'
	return inp.name, inp.type.imageType.width, inp.type.imageType.height


def depth_from_prediction(prediction):
	arr = next(iter(prediction.values()))
	return np.asarray(arr, dtype=np.float32).squeeze()


def normalized_png(depth, invert, out_path):
	lo, hi = depth.min(), depth.max()
	norm = (depth - lo) / (hi - lo) if hi > lo else np.zeros_like(depth)
	if invert:
		norm = 1.0 - norm
	Image.fromarray((norm * 255).astype(np.uint8)).save(out_path)


def bench_model(name, path):
	if not path.exists():
		print(f'SKIP {name}: {path} missing')
		return None

	t0 = time.perf_counter()
	model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL)
	inp_name, width, height = input_spec(model)
	load_s = time.perf_counter() - t0

	images = {p: Image.open(p).convert('RGB').resize((width, height), Image.LANCZOS) for p in TEST_IMAGES}
	timing_img = images[TEST_IMAGES[0]]

	for _ in range(WARMUP_ITERS):
		model.predict({inp_name: timing_img})

	times = []
	for _ in range(TIMING_ITERS):
		t0 = time.perf_counter()
		model.predict({inp_name: timing_img})
		times.append((time.perf_counter() - t0) * 1000)

	model_out = OUT / name
	model_out.mkdir(parents=True, exist_ok=True)
	for p, img in images.items():
		depth = depth_from_prediction(model.predict({inp_name: img}))
		normalized_png(depth, name in INVERT_OUTPUT, model_out / f'{p.stem}.png')

	return {
		'name': name,
		'input': f'{width}x{height}',
		'load_s': load_s,
		'median_ms': statistics.median(times),
		'min_ms': min(times),
		'max_ms': max(times),
	}


def main():
	selected = sys.argv[1:] or list(MODELS)
	results = [r for r in (bench_model(n, MODELS[n]) for n in selected) if r]

	print('\n| model | input | load (s) | median ms | min | max |')
	print('|---|---|---|---|---|---|')
	for r in results:
		print(
			f'| {r["name"]} | {r["input"]} | {r["load_s"]:.1f} | '
			f'{r["median_ms"]:.0f} | {r["min_ms"]:.0f} | {r["max_ms"]:.0f} |'
		)
	print(f'\ndepth maps: {OUT}/<model>/*.png')


if __name__ == '__main__':
	main()
