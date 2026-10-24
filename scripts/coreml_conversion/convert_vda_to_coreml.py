#!/usr/bin/env -S uv run --python 3.12 --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "coremltools==9.0",
#     "numpy<2",
#     "torch==2.7.0",
#     "torchvision==0.22.0",
#     "opencv-python-headless",
#     "huggingface_hub",
#     "easydict",
#     "einops",
#     "pillow",
#     "tqdm",
# ]
# ///
"""
Convert Video Depth Anything Small (Apache-2.0) to a streaming CoreML model.

One call = one frame. The temporal cache is explicit: 8 inputs `cache{i}` of shape (31, D, C) hold the
hidden states of the 31 frames in the window, 8 outputs `state{i}` of shape (1, D, C) are this frame's
hidden states. The caller keeps the window (see StreamWindow below and swift-bridge/CoreMLDepth.swift).

The model code comes from a checkout of https://github.com/DepthAnything/Video-Depth-Anything.

Usage:
	VDA_SRC=/Volumes/wd/cache/spatial-maker/src/vda VDA_FRAMES=/Volumes/wd/cache/spatial-maker/upgrades/frames \
		uv run scripts/coreml_conversion/convert_vda_to_coreml.py
"""

import os
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image

VDA_SRC = Path(os.environ['VDA_SRC'])
sys.path.insert(0, str(VDA_SRC))

from video_depth_anything.video_depth_stream import VideoDepthAnything  # noqa: E402

INPUT_SIZE = 518
PATCH = INPUT_SIZE // 14
WINDOW = 31
LAYERS = [2, 5, 8, 11]
OUTPUT_NAME = 'VideoDepthAnythingSmall.mlpackage'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VdaStream(nn.Module):
	def __init__(self, vda):
		super().__init__()
		self.vda = vda
		self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
		self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

	def forward(self, image, c0, c1, c2, c3, c4, c5, c6, c7):
		x = (image - self.mean) / self.std
		features = self.vda.pretrained.get_intermediate_layers(x, LAYERS, return_class_token=True)
		cache = [c.permute(1, 0, 2) for c in (c0, c1, c2, c3, c4, c5, c6, c7)]
		depth, states = self.vda.head(features, PATCH, PATCH, 1, cached_hidden_state_list=cache)
		depth = F.relu(
			F.interpolate(depth, size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=True)
		)
		return (depth.reshape(1, INPUT_SIZE, INPUT_SIZE), *[s.permute(1, 0, 2) for s in states])


class StreamWindow:
	"""Mirror of VideoDepthAnything.infer_video_depth_one's cache bookkeeping (frame 0 is kept forever)."""

	def __init__(self):
		self.frames = []
		self.next_id = 0

	def window(self):
		return self.frames[0:2] + self.frames[-(WINDOW - 2) :]

	def push(self, states):
		if not self.frames:
			self.frames = [states] * (WINDOW + 1)
		else:
			self.frames.append(states)
		if self.next_id + WINDOW + 1 > 2 * (WINDOW + 1 - 10) - 1 - 2 + 1:
			del self.frames[1]
		self.next_id += 1


def load_vda():
	vda = VideoDepthAnything(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
	checkpoint = hf_hub_download('depth-anything/Video-Depth-Anything-Small', 'video_depth_anything_vits.pth')
	vda.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
	return vda.eval()


def state_shapes(model):
	image = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
	x = (image - model.mean) / model.std
	features = model.vda.pretrained.get_intermediate_layers(x, LAYERS, return_class_token=True)
	with torch.no_grad():
		_, states = model.vda.head(features, PATCH, PATCH, 1)
	return [tuple(s.permute(1, 0, 2).shape) for s in states]


def convert(model, shapes):
	example = (torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE), *[torch.rand(WINDOW, d, c) for _, d, c in shapes])
	with torch.no_grad():
		traced = torch.jit.trace(model, example)
	cache_inputs = [
		ct.TensorType(name=f'cache{i}', shape=(WINDOW, d, c), dtype=np.float16)
		for i, (_, d, c) in enumerate(shapes)
	]
	mlmodel = ct.convert(
		traced,
		inputs=[
			ct.ImageType(
				name='image',
				shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
				color_layout=ct.colorlayout.RGB,
				scale=1.0 / 255.0,
			),
			*cache_inputs,
		],
		outputs=[
			ct.TensorType(name='depth', dtype=np.float16),
			*[ct.TensorType(name=f'state{i}', dtype=np.float16) for i in range(len(shapes))],
		],
		compute_precision=ct.precision.FLOAT16,
		minimum_deployment_target=ct.target.macOS14,
		compute_units=ct.ComputeUnit.CPU_AND_GPU,
	)
	mlmodel.short_description = (
		'Video Depth Anything Small, streaming (one frame per call, explicit temporal cache)'
	)
	mlmodel.author = 'ByteDance (converted to CoreML by spatial-maker)'
	mlmodel.license = 'Apache-2.0'
	return mlmodel


def coreml_stream(mlmodel, images, shapes):
	window = StreamWindow()
	depths = []
	for index, img in enumerate(images):
		if index == 0:
			zeros = {f'cache{i}': np.zeros((WINDOW, d, c), np.float16) for i, (_, d, c) in enumerate(shapes)}
			primed = mlmodel.predict({'image': img, **zeros})
			cache = {f'cache{i}': np.repeat(primed[f'state{i}'], WINDOW, axis=0) for i in range(len(shapes))}
		else:
			frames = window.window()
			cache = {f'cache{i}': np.concatenate([f[i] for f in frames], axis=0) for i in range(len(shapes))}
		out = mlmodel.predict({'image': img, **cache})
		window.push([out[f'state{i}'] for i in range(len(shapes))])
		depths.append(np.asarray(out['depth'], np.float32).squeeze())
	return depths


def torch_stream(vda, images):
	ref = VideoDepthAnything(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
	ref.load_state_dict(vda.state_dict())
	ref.eval()
	return [
		ref.infer_video_depth_one(np.asarray(img), input_size=INPUT_SIZE, device='cpu', fp32=True)
		for img in images
	]


def check_fidelity(vda, mlmodel, shapes):
	frames = sorted(Path(os.environ['VDA_FRAMES']).glob('frame_*.jpg'))
	assert len(frames) >= 2, 'VDA_FRAMES needs frame_*.jpg'
	images = [Image.open(f).convert('RGB').resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS) for f in frames]
	got = coreml_stream(mlmodel, images, shapes)
	ref = torch_stream(vda, images)
	pearson = [np.corrcoef(r.ravel(), g.ravel())[0, 1] for r, g in zip(ref, got)]
	print(
		f'fidelity vs torch streaming: frame0 {pearson[0]:.5f}, min {min(pearson):.5f}, mean {np.mean(pearson):.5f}'
	)
	assert min(pearson[1:]) > 0.99, f'CoreML streaming diverges from torch (min pearson {min(pearson):.4f})'


def main():
	output_dir = Path(os.environ.get('VDA_OUTPUT_DIR', Path.home() / '.spatial-maker/checkpoints'))
	vda = load_vda()
	model = VdaStream(vda).eval()
	shapes = state_shapes(model)
	print(f'state shapes: {shapes}')
	mlmodel = convert(model, shapes)
	check_fidelity(vda, mlmodel, shapes)
	output_path = output_dir / OUTPUT_NAME
	mlmodel.save(str(output_path))
	size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / (1024 * 1024)
	print(f'saved {output_path} ({size_mb:.0f} MB)')


if __name__ == '__main__':
	main()
