#!/usr/bin/env python3
"""
Convert Depth Anything V2 models (Base and Large) to CoreML format.

This script converts PyTorch Depth Anything V2 models to Apple's CoreML format
for optimized inference on Apple Silicon (M-series chips) using the Neural Engine.

Requirements:
    uv run --with coremltools --with torch --with jkp-depth-anything-v2 \
        python scripts/coreml_conversion/convert_to_coreml.py

Models will be saved to: checkpoints/
"""

import sys
from pathlib import Path

import coremltools as ct
import torch
from depth_anything_v2.dpt import DepthAnythingV2

# Model configurations
MODEL_CONFIGS = {
	'vitb': {
		'encoder': 'vitb',
		'features': 128,
		'out_channels': [96, 192, 384, 768],
		'checkpoint': 'depth_anything_v2_vitb.pth',
		'output': 'DepthAnythingV2BaseF16.mlpackage',
	},
	'vitl': {
		'encoder': 'vitl',
		'features': 256,
		'out_channels': [256, 512, 1024, 1024],
		'checkpoint': 'depth_anything_v2_vitl.pth',
		'output': 'DepthAnythingV2LargeF16.mlpackage',
	},
}

INPUT_SIZE = 518


def convert_model(model_key: str, checkpoint_dir: Path, output_dir: Path):
	"""Convert a single model to CoreML format."""
	config = MODEL_CONFIGS[model_key]
	checkpoint_path = checkpoint_dir / config['checkpoint']
	output_path = output_dir / config['output']

	print(f'\n{"=" * 60}')
	print(f'Converting {model_key.upper()} model to CoreML')
	print(f'{"=" * 60}')

	# Check if checkpoint exists
	if not checkpoint_path.exists():
		print(f'❌ Checkpoint not found: {checkpoint_path}')
		print(f'Please download from:')
		print(
			f'   https://huggingface.co/depth-anything/Depth-Anything-V2-{model_key.capitalize()}/resolve/main/{config["checkpoint"]}'
		)
		return False

	print(f'✓ Loading PyTorch model from {checkpoint_path}')
	model = DepthAnythingV2(
		encoder=config['encoder'],
		features=config['features'],
		out_channels=config['out_channels'],
	)
	model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'))
	model.eval()

	print(f'✓ Tracing model with input shape (1, 3, {INPUT_SIZE}, {INPUT_SIZE})')
	example_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

	with torch.no_grad():
		traced_model = torch.jit.trace(model, example_input)

	print('✓ Converting to CoreML (Float16 precision)')
	print('  This may take a few minutes...')

	mlmodel = ct.convert(
		traced_model,
		inputs=[
			ct.TensorType(
				name='image',
				shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
				dtype=float,
			)
		],
		outputs=[ct.TensorType(name='depth')],
		compute_precision=ct.precision.FLOAT16,
		minimum_deployment_target=ct.target.macOS15,
	)

	# Add metadata
	mlmodel.short_description = (
		f'Depth Anything V2 {model_key.upper()} - Monocular depth estimation'
	)
	mlmodel.author = 'Depth Anything Team (converted to CoreML)'
	mlmodel.license = 'CC-BY-NC-4.0'  # Base and Large are both non-commercial
	mlmodel.version = '2.0'

	mlmodel.input_description['image'] = 'RGB input image (518x518)'
	mlmodel.output_description['depth'] = 'Depth map (518x518)'

	print(f'✓ Saving to {output_path}')
	mlmodel.save(str(output_path))

	# Get file size
	size_mb = output_path.stat().st_size / (1024 * 1024)
	print(f'✓ Model saved: {size_mb:.1f} MB')
	print(f'✓ Ready to use with CoreML on Apple Silicon!')

	return True


def main():
	# Setup paths
	repo_root = Path(__file__).parent.parent.parent
	checkpoint_dir = repo_root / 'checkpoints'
	output_dir = checkpoint_dir

	checkpoint_dir.mkdir(exist_ok=True)

	print('Depth Anything V2 → CoreML Conversion')
	print('=' * 60)
	print(f'Checkpoint directory: {checkpoint_dir}')
	print(f'Output directory: {output_dir}')

	# Convert both models
	success = {
		'Base': convert_model('vitb', checkpoint_dir, output_dir),
		'Large': convert_model('vitl', checkpoint_dir, output_dir),
	}

	# Summary
	print(f'\n{"=" * 60}')
	print('Conversion Summary')
	print(f'{"=" * 60}')

	for name, ok in success.items():
		status = '✓ Success' if ok else '✗ Failed'
		print(f'{name:10} {status}')

	if all(success.values()):
		print('\n✓ All conversions completed successfully!')
		print('\nYou can now use these models with CoreML on Apple Silicon.')
		print('Expected performance on M4 Pro:')
		print('  - Base:  ~60-90ms per frame')
		print('  - Large: ~200-300ms per frame')
	else:
		print('\n✗ Some conversions failed. See errors above.')
		sys.exit(1)


if __name__ == '__main__':
	main()
