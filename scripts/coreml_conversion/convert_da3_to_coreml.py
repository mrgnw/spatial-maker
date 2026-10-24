#!/usr/bin/env -S uv run --python 3.12 --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "coremltools==9.0",
#     "numpy<2",
#     "torch==2.7.0",
#     "safetensors",
#     "huggingface_hub",
#     "addict",
#     "omegaconf",
#     "einops",
#     "tqdm",
#     "pillow",
# ]
# ///
"""
Convert Depth Anything 3 checkpoints to CoreML (504x504 RGB Image input, FP16 'depth' output).

The model code comes from a checkout of https://github.com/ByteDance-Seed/Depth-Anything-3
(its package pulls xformers/open3d/gsplat, which the depth path does not need).

Usage:
	DA3_SRC=/Volumes/wd/cache/spatial-maker/src/da3 uv run scripts/coreml_conversion/convert_da3_to_coreml.py [mono-large] [metric-large] [small] [base]
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
from safetensors.torch import load_file

sys.path.insert(0, str(Path(os.environ['DA3_SRC']) / 'src'))

from depth_anything_3.cfg import create_object, load_config  # noqa: E402
from depth_anything_3.model import dpt, dualdpt  # noqa: E402
from depth_anything_3.model.utils import head_utils  # noqa: E402
from depth_anything_3.registry import MODEL_REGISTRY  # noqa: E402

MODEL_CONFIGS = {
	'mono-large': {
		'preset': 'da3mono-large',
		'hf_repo': 'depth-anything/DA3MONO-LARGE',
		'output': 'DepthAnythingV3Mono.mlpackage',
		'license': 'Apache-2.0',
	},
	'metric-large': {
		'preset': 'da3metric-large',
		'hf_repo': 'depth-anything/DA3METRIC-LARGE',
		'output': 'DepthAnythingV3MetricLarge.mlpackage',
		'license': 'Apache-2.0',
	},
	'small': {
		'preset': 'da3-small',
		'hf_repo': 'depth-anything/DA3-SMALL',
		'output': 'DepthAnythingV3Small.mlpackage',
		'license': 'Apache-2.0',
	},
	'base': {
		'preset': 'da3-base',
		'hf_repo': 'depth-anything/DA3-BASE',
		'output': 'DepthAnythingV3Base.mlpackage',
		'license': 'Apache-2.0',
	},
}

INPUT_SIZE = 504
PATCH_SIZE = 14

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

REPO = Path(__file__).parent.parent.parent
FIDELITY_IMAGE = REPO / 'test_baseline.jpg'


class Da3Depth(nn.Module):
	def __init__(self, net):
		super().__init__()
		self.net = net
		self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
		self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

	def ref_cam_token(self):
		vit = self.net.backbone.pretrained
		return vit.camera_token[:, :1] if hasattr(vit, 'camera_token') else None

	def forward(self, x):
		x = ((x - self.mean) / self.std).unsqueeze(1)
		feats, _ = self.net.backbone(
			x, cam_token=self.ref_cam_token(), export_feat_layers=[], ref_view_strategy='first'
		)
		out = self.net.head(feats, INPUT_SIZE, INPUT_SIZE, patch_start_idx=0)
		return out['depth'].reshape(1, INPUT_SIZE, INPUT_SIZE)


def freeze_pos_embed(vit):
	grid = INPUT_SIZE // PATCH_SIZE
	pos_embed = vit.pos_embed.detach().float()
	side = int((pos_embed.shape[1] - 1) ** 0.5)
	patch = pos_embed[:, 1:].reshape(1, side, side, -1).permute(0, 3, 1, 2)
	patch = F.interpolate(patch, size=(grid, grid), mode='bicubic', antialias=False)
	patch = patch.permute(0, 2, 3, 1).reshape(1, grid * grid, -1)
	frozen = torch.cat([pos_embed[:, :1], patch], dim=1)
	vit.interpolate_pos_encoding = lambda x, w, h: frozen.to(x.dtype)
	if getattr(vit, 'position_getter', None) is not None:
		ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='ij')
		positions = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1)[None]
		vit.position_getter = lambda batch, h, w, device: positions.expand(batch, -1, -1).clone()


def cache_uv_grids():
	cache = {}

	def cached_uv_grid(width, height, aspect_ratio=None, dtype=None, device=None):
		key = (int(width), int(height), aspect_ratio, dtype)
		if key not in cache:
			cache[key] = head_utils.create_uv_grid(*key, device=device)
		return cache[key]

	dpt.create_uv_grid = cached_uv_grid
	dualdpt.create_uv_grid = cached_uv_grid


def load_net(config):
	net = create_object(load_config(MODEL_REGISTRY[config['preset']]))
	weights = load_file(hf_hub_download(config['hf_repo'], 'model.safetensors'))
	weights = {k.removeprefix('model.'): v for k, v in weights.items()}
	missing, unexpected = net.load_state_dict(weights, strict=False)
	depth_missing = [k for k in missing if '_aux' not in k]
	assert not depth_missing, f'missing weights: {depth_missing[:5]}'
	print(f'missing ray-head weights: {len(missing)}')
	print(f'unused weights (camera/gs heads): {len(unexpected)}')
	net.eval()
	freeze_pos_embed(net.backbone.pretrained)
	return net


def fidelity_input():
	img = Image.open(FIDELITY_IMAGE).convert('RGB').resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)
	tensor = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
	return img, tensor


def check_fidelity(model, mlmodel):
	img, tensor = fidelity_input()
	with torch.no_grad():
		ref = model(tensor).numpy().ravel()
	got = np.asarray(mlmodel.predict({'image': img})['depth'], dtype=np.float32).ravel()
	pearson = np.corrcoef(ref, got)[0, 1]
	norm = lambda d: (d - d.min()) / (d.max() - d.min())
	mae = np.abs(norm(ref) - norm(got)).mean()
	print(f'fidelity vs FP32 torch: pearson {pearson:.5f}, normalized MAE {mae * 100:.2f}%')
	assert pearson > 0.99, f'CoreML output diverges from torch (pearson {pearson:.4f})'


def convert_model(key, output_dir):
	config = MODEL_CONFIGS[key]
	output_path = output_dir / config['output']
	print(f'\n=== {key}: {config["hf_repo"]} -> {output_path}')

	cache_uv_grids()
	model = Da3Depth(load_net(config)).eval()
	example = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
	with torch.no_grad():
		model(example)
		traced = torch.jit.trace(model, example)

	mlmodel = ct.convert(
		traced,
		inputs=[
			ct.ImageType(
				name='image',
				shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
				color_layout=ct.colorlayout.RGB,
				scale=1.0 / 255.0,
			)
		],
		outputs=[ct.TensorType(name='depth', dtype=np.float16)],
		compute_precision=ct.precision.FLOAT16,
		minimum_deployment_target=ct.target.macOS14,
	)
	mlmodel.short_description = f'Depth Anything 3 ({config["hf_repo"]}), relative depth, larger = farther'
	mlmodel.author = 'ByteDance Seed (converted to CoreML by spatial-maker)'
	mlmodel.license = config['license']
	mlmodel.input_description['image'] = f'RGB image ({INPUT_SIZE}x{INPUT_SIZE})'
	mlmodel.output_description['depth'] = f'Depth map (1x{INPUT_SIZE}x{INPUT_SIZE})'

	check_fidelity(model, mlmodel)
	mlmodel.save(str(output_path))
	size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / (1024 * 1024)
	print(f'saved {output_path} ({size_mb:.0f} MB)')


def main():
	output_dir = Path(os.environ.get('DA3_OUTPUT_DIR', Path.home() / '.spatial-maker/checkpoints'))
	keys = sys.argv[1:] or list(MODEL_CONFIGS)
	unknown = [k for k in keys if k not in MODEL_CONFIGS]
	assert not unknown, f'unknown models {unknown}; options: {", ".join(MODEL_CONFIGS)}'
	for key in keys:
		convert_model(key, output_dir)


if __name__ == '__main__':
	main()
