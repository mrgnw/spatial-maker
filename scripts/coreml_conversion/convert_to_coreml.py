#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coremltools==7.2",
#     "numpy<2",
#     "torch==2.7.0",
#     "jkp-depth-anything-v2",
# ]
# ///
"""
Convert Depth Anything V2 models (Small, Base, Large) to CoreML format
with uniform Image input (518x518 BGR) for Apple Silicon inference.

ImageNet normalization is baked into the model so the runtime just sends
raw pixel data via CVPixelBuffer — CoreML handles the /255 scaling.

Usage:
	uv run scripts/coreml_conversion/convert_to_coreml.py [vits] [vitb] [vitl]
"""

import sys
from pathlib import Path

import coremltools as ct
import torch
import torch.nn as nn
from depth_anything_v2.dpt import DepthAnythingV2

MODEL_CONFIGS = {
	'vits': {
		'encoder': 'vits',
		'features': 64,
		'out_channels': [48, 96, 192, 384],
		'checkpoint': 'depth_anything_v2_vits.pth',
		'output': 'DepthAnythingV2SmallF16.mlpackage',
		'hf_repo': 'depth-anything/Depth-Anything-V2-Small',
		'license': 'Apache-2.0',
	},
	'vitb': {
		'encoder': 'vitb',
		'features': 128,
		'out_channels': [96, 192, 384, 768],
		'checkpoint': 'depth_anything_v2_vitb.pth',
		'output': 'DepthAnythingV2BaseF16.mlpackage',
		'hf_repo': 'depth-anything/Depth-Anything-V2-Base',
		'license': 'Apache-2.0',
	},
	'vitl': {
		'encoder': 'vitl',
		'features': 256,
		'out_channels': [256, 512, 1024, 1024],
		'checkpoint': 'depth_anything_v2_vitl.pth',
		'output': 'DepthAnythingV2LargeF16.mlpackage',
		'hf_repo': 'depth-anything/Depth-Anything-V2-Large',
		'license': 'CC-BY-NC-4.0',
	},
}

INPUT_SIZE = 518

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class NormalizedDepthModel(nn.Module):
	def __init__(self, model):
		super().__init__()
		self.model = model
		self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
		self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

	def forward(self, x):
		x = (x - self.mean) / self.std
		return self.model(x)


def convert_model(model_key: str, checkpoint_dir: Path, output_dir: Path):
	config = MODEL_CONFIGS[model_key]
	checkpoint_path = checkpoint_dir / config['checkpoint']
	output_path = output_dir / config['output']

	print(f'\n{"=" * 60}')
	print(f'Converting {model_key.upper()} to CoreML')
	print(f'{"=" * 60}')

	if not checkpoint_path.exists():
		print(f'Checkpoint not found: {checkpoint_path}')
		print(f'Download:')
		print(f'  curl -L -o {checkpoint_path} \\')
		print(f'    https://huggingface.co/{config["hf_repo"]}/resolve/main/{config["checkpoint"]}')
		return False

	print(f'Loading {checkpoint_path}')
	base_model = DepthAnythingV2(
		encoder=config['encoder'],
		features=config['features'],
		out_channels=config['out_channels'],
	)
	base_model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'))
	base_model.eval()

	model = NormalizedDepthModel(base_model)
	model.eval()

	print(f'Tracing with input (1, 3, {INPUT_SIZE}, {INPUT_SIZE})')
	example_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
	with torch.no_grad():
		traced_model = torch.jit.trace(model, example_input)

	print('Converting to CoreML (Float16, Image input)')
	mlmodel = ct.convert(
		traced_model,
		inputs=[
			ct.ImageType(
				name='image',
				shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
				color_layout=ct.colorlayout.BGR,
				scale=1.0 / 255.0,
				bias=0.0,
			)
		],
		outputs=[ct.TensorType(name='depth')],
		compute_precision=ct.precision.FLOAT16,
		minimum_deployment_target=ct.target.macOS14,
	)

	mlmodel.short_description = f'Depth Anything V2 {model_key.upper()} - Monocular depth estimation'
	mlmodel.author = 'Depth Anything Team (converted to CoreML)'
	mlmodel.license = config['license']
	mlmodel.version = '2.1'
	mlmodel.input_description['image'] = f'BGR image ({INPUT_SIZE}x{INPUT_SIZE})'
	mlmodel.output_description['depth'] = f'Depth map ({INPUT_SIZE}x{INPUT_SIZE})'

	print(f'Saving to {output_path}')
	mlmodel.save(str(output_path))

	size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / (1024 * 1024)
	print(f'Saved: {size_mb:.1f} MB')
	return True


def main():
	repo_root = Path(__file__).parent.parent.parent
	checkpoint_dir = repo_root / 'checkpoints'
	output_dir = checkpoint_dir
	checkpoint_dir.mkdir(exist_ok=True)

	models_to_convert = sys.argv[1:] if len(sys.argv) > 1 else list(MODEL_CONFIGS.keys())

	for key in models_to_convert:
		if key not in MODEL_CONFIGS:
			print(f'Unknown model: {key}. Options: {", ".join(MODEL_CONFIGS.keys())}')
			sys.exit(1)

	success = {}
	for key in models_to_convert:
		success[key] = convert_model(key, checkpoint_dir, output_dir)

	print(f'\n{"=" * 60}')
	for name, ok in success.items():
		print(f'  {name:6} {"OK" if ok else "FAILED"}')

	if not all(success.values()):
		sys.exit(1)


if __name__ == '__main__':
	main()
