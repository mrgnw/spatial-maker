"""
Combined depth estimation and stereo conversion - pipes directly to ffmpeg without intermediate frames.

This module combines depth estimation (Depth Anything V2) with stereo conversion (DIBR)
into a single streaming pipeline that avoids writing intermediate depth frames to disk.
"""

import json
import subprocess
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from depth_anything_v2.dpt import DepthAnythingV2

_threads_configured = False
_processor_cache = {}


def create_stereo_pair(
	rgb_frame: np.ndarray,
	depth_frame: np.ndarray,
	max_disparity: int = 30,
) -> tuple:
	"""
	Create left and right eye views from RGB frame and depth map.

	Args:
	    rgb_frame: Original RGB image (H, W, 3)
	    depth_frame: Normalized depth map (H, W), values 0-1
	    max_disparity: Maximum pixel shift for closest objects

	Returns:
	    (left_eye, right_eye) tuple of images
	"""
	h, w = rgb_frame.shape[:2]

	# Calculate disparity map (closer = higher value = more shift)
	disparity = (depth_frame * max_disparity).astype(np.float32)

	# Create coordinate grids
	x_coords = np.arange(w, dtype=np.float32)
	y_coords = np.arange(h, dtype=np.float32)
	x_grid, y_grid = np.meshgrid(x_coords, y_coords)

	# Shift for left eye (shift right for closer objects)
	x_left = x_grid + disparity / 2
	# Shift for right eye (shift left for closer objects)
	x_right = x_grid - disparity / 2

	# Remap using cv2 (handles interpolation and boundary)
	left_eye = cv2.remap(
		rgb_frame,
		x_left,
		y_grid,
		interpolation=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_REPLICATE,
	)
	right_eye = cv2.remap(
		rgb_frame,
		x_right,
		y_grid,
		interpolation=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_REPLICATE,
	)

	return left_eye, right_eye


class DepthToStereoProcessor:
	"""
	Combined depth estimation and stereo conversion processor.

	Streams RGB frames through depth estimation and stereo conversion,
	piping output directly to ffmpeg without intermediate files.
	"""

	def __init__(
		self,
		encoder: str = 'vits',
		num_threads: int = None,
		json_progress: bool = False,
	):
		self.encoder = encoder
		self.num_threads = (
			num_threads if num_threads is not None else max(1, os.cpu_count() - 4)
		)
		self.model = None
		self.device = None
		self.json_progress = json_progress

	def load_model(self):
		"""Load the Depth Anything V2 model."""
		global _threads_configured
		if self.model is not None:
			return

		if not _threads_configured:
			torch.set_num_threads(self.num_threads)
			torch.set_num_interop_threads(self.num_threads)
			_threads_configured = True

		model_configs = {
			'vits': {
				'encoder': 'vits',
				'features': 64,
				'out_channels': [48, 96, 192, 384],
			},
			'vitb': {
				'encoder': 'vitb',
				'features': 128,
				'out_channels': [96, 192, 384, 768],
			},
			'vitl': {
				'encoder': 'vitl',
				'features': 256,
				'out_channels': [256, 512, 1024, 1024],
			},
		}

		checkpoint_name = f'depth_anything_v2_{self.encoder}.pth'
		checkpoint_path = self._find_checkpoint(checkpoint_name)

		if torch.backends.mps.is_available():
			self.device = torch.device('mps')
		elif torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')

		self.model = DepthAnythingV2(**model_configs[self.encoder])
		# Load weights directly to target device to avoid MPS/CPU tensor mismatch
		self.model.load_state_dict(
			torch.load(str(checkpoint_path), map_location=self.device)
		)
		self.model = self.model.to(self.device).eval()

	def _find_checkpoint(self, checkpoint_name: str) -> Path:
		"""
		Find checkpoint file in multiple possible locations.

		Searches in order:
		1. Next to the script's source directory (for development)
		2. User's home directory ~/.spatial-maker/checkpoints/
		3. XDG data directory
		4. Current working directory ./checkpoints/
		"""
		search_paths = [
			# Development: relative to source file
			Path(__file__).parent.parent / 'checkpoints' / checkpoint_name,
			# User home directory
			Path.home() / '.spatial-maker' / 'checkpoints' / checkpoint_name,
			# XDG data home
			Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share'))
			/ 'spatial-maker'
			/ 'checkpoints'
			/ checkpoint_name,
			# Current working directory
			Path.cwd() / 'checkpoints' / checkpoint_name,
		]

		# Also check if SPATIAL_MAKER_CHECKPOINTS env var is set
		env_checkpoint_dir = os.environ.get('SPATIAL_MAKER_CHECKPOINTS')
		if env_checkpoint_dir:
			search_paths.insert(0, Path(env_checkpoint_dir) / checkpoint_name)

		for path in search_paths:
			if path.exists():
				return path

		# Not found - provide helpful error message
		raise FileNotFoundError(
			f"Checkpoint '{checkpoint_name}' not found.\n"
			f'Searched locations:\n'
			+ '\n'.join(f'  - {p}' for p in search_paths)
			+ '\n\nPlease download the checkpoint and place it in one of these locations:\n'
			f'  mkdir -p ~/.spatial-maker/checkpoints\n'
			f'  # Then copy {checkpoint_name} there'
		)

	def process_video(
		self,
		video_path: str,
		output_path: str,
		max_disparity: int = 30,
		fps: int = 24,
		max_frames: int = None,
	) -> dict:
		"""
		Process video: generate depth and create SBS stereo, piping directly to ffmpeg.

		Args:
		    video_path: Path to input RGB video
		    output_path: Path for output SBS video
		    max_disparity: Maximum pixel disparity for 3D effect
		    fps: Output frame rate
		    max_frames: Optional limit on frames to process

		Returns:
		    Dict with processing statistics
		"""
		self.load_model()

		# Open input video
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			raise ValueError(f'Could not open video: {video_path}')

		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		if max_frames is not None:
			total_frames = min(total_frames, max_frames)

		# Get dimensions
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

		# Output is full-width SBS (double width)
		out_width = width * 2
		out_height = height

		output_path = Path(output_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)

		# Start ffmpeg process to receive piped frames
		ffmpeg_cmd = [
			'ffmpeg',
			'-y',
			'-f',
			'rawvideo',
			'-vcodec',
			'rawvideo',
			'-s',
			f'{out_width}x{out_height}',
			'-pix_fmt',
			'bgr24',
			'-r',
			str(fps),
			'-i',
			'-',  # Read from stdin
			'-c:v',
			'hevc_videotoolbox',
			'-q:v',
			'65',  # Quality-based encoding (0-100, higher = better)
			'-tag:v',
			'hvc1',
			str(output_path),
		]

		ffmpeg_proc = subprocess.Popen(
			ffmpeg_cmd,
			stdin=subprocess.PIPE,
			stdout=subprocess.DEVNULL,
			stderr=subprocess.DEVNULL,
		)

		# Warmup inference
		ret, first_frame = cap.read()
		if not ret:
			raise ValueError('Could not read first frame')
		cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

		with torch.no_grad():
			_ = self.model.infer_image(first_frame, 518)

		frames_processed = 0

		progress_iter = range(total_frames)
		if not self.json_progress:
			progress_iter = tqdm(progress_iter, desc='  ', unit='frame', ncols=60)

		try:
			for _ in progress_iter:
				ret, rgb_frame = cap.read()
				if not ret:
					break

				with torch.no_grad():
					depth = self.model.infer_image(rgb_frame, 518)

				depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())

				left_eye, right_eye = create_stereo_pair(
					rgb_frame, depth_normalized, max_disparity
				)

				sbs_frame = np.hstack([left_eye, right_eye])

				ffmpeg_proc.stdin.write(sbs_frame.tobytes())
				frames_processed += 1

				if self.json_progress:
					pct = (
						(frames_processed / total_frames * 100)
						if total_frames > 0
						else 0
					)
					print(
						json.dumps(
							{
								'event': 'progress',
								'stage': 'depth_stereo',
								'frame': frames_processed,
								'total': total_frames,
								'pct': round(pct, 1),
							}
						),
						flush=True,
					)

				if max_frames is not None and frames_processed >= max_frames:
					break

		finally:
			cap.release()
			ffmpeg_proc.stdin.close()
			ffmpeg_proc.wait()

		return {
			'frames_processed': frames_processed,
			'output_path': str(output_path),
			'dimensions': f'{out_width}x{out_height}',
		}


def process_video_to_sbs(
	video_path: str,
	output_path: str,
	encoder: str = 'vits',
	max_disparity: int = 30,
	fps: int = 24,
	max_frames: int = None,
	json_progress: bool = False,
) -> dict:
	"""
	Convenience function to process a video to SBS stereo.

	Uses a cached processor instance to reuse the model across multiple calls
	(important for batch processing).

	Args:
	    video_path: Path to input RGB video
	    output_path: Path for output SBS video
	    encoder: Depth Anything V2 encoder (vits, vitb, vitl)
	    max_disparity: Maximum pixel disparity for 3D effect
	    fps: Output frame rate
	    max_frames: Optional limit on frames to process

	Returns:
	    Dict with processing statistics
	"""
	global _processor_cache

	cache_key = (encoder, json_progress)
	if cache_key not in _processor_cache:
		_processor_cache[cache_key] = DepthToStereoProcessor(
			encoder=encoder, json_progress=json_progress
		)

	processor = _processor_cache[cache_key]
	return processor.process_video(
		video_path,
		output_path,
		max_disparity=max_disparity,
		fps=fps,
		max_frames=max_frames,
	)


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(
		description='Convert 2D video to SBS stereo using depth estimation'
	)
	parser.add_argument('input', help='Input video file')
	parser.add_argument('output', help='Output SBS video file')
	parser.add_argument(
		'--encoder',
		choices=['vits', 'vitb', 'vitl'],
		default='vits',
		help='Depth model encoder (default: vits)',
	)
	parser.add_argument(
		'--max-disparity',
		type=int,
		default=30,
		help='Maximum pixel disparity (default: 30)',
	)
	parser.add_argument(
		'--fps',
		type=int,
		default=24,
		help='Output FPS (default: 24)',
	)
	parser.add_argument(
		'--max-frames',
		type=int,
		help='Maximum frames to process',
	)

	args = parser.parse_args()

	result = process_video_to_sbs(
		args.input,
		args.output,
		encoder=args.encoder,
		max_disparity=args.max_disparity,
		fps=args.fps,
		max_frames=args.max_frames,
	)
	print(f'Done! Output: {result["output_path"]}')
