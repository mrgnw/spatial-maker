use crate::NormalizeMode;
use ndarray::Array2;

pub struct DepthProcessor {
	prev_depth: Option<Array2<f32>>,
	ema_min: f32,
	ema_max: f32,
	global_min: f32,
	global_max: f32,
	temporal_alpha: f32,
	bilateral_sigma_space: f32,
	bilateral_sigma_color: f32,
	depth_blur_sigma: f32,
	normalize_mode: NormalizeMode,
	frame_index: u32,
}

impl DepthProcessor {
	pub fn new(
		temporal_alpha: f32,
		bilateral_sigma_space: f32,
		bilateral_sigma_color: f32,
		depth_blur_sigma: f32,
		normalize_mode: NormalizeMode,
	) -> Self {
		Self {
			prev_depth: None,
			ema_min: 0.0,
			ema_max: 0.0,
			global_min: f32::INFINITY,
			global_max: f32::NEG_INFINITY,
			temporal_alpha,
			bilateral_sigma_space,
			bilateral_sigma_color,
			depth_blur_sigma,
			normalize_mode,
			frame_index: 0,
		}
	}

	pub fn set_global_range(&mut self, min: f32, max: f32) {
		self.global_min = min;
		self.global_max = max;
	}

	pub fn update_global_range(&mut self, raw_depth: &Array2<f32>) {
		let min = raw_depth.iter().copied().fold(f32::INFINITY, f32::min);
		let max = raw_depth.iter().copied().fold(f32::NEG_INFINITY, f32::max);
		self.global_min = self.global_min.min(min);
		self.global_max = self.global_max.max(max);
	}

	pub fn process(&mut self, raw_depth: Array2<f32>) -> Array2<f32> {
		let mut depth = self.normalize(raw_depth);

		if self.bilateral_sigma_space > 0.0 {
			depth = bilateral_filter(&depth, self.bilateral_sigma_space, self.bilateral_sigma_color);
		}

		if self.depth_blur_sigma > 0.0 {
			depth = gaussian_blur(&depth, self.depth_blur_sigma);
		}

		if self.temporal_alpha > 0.0 && self.temporal_alpha < 1.0 {
			if let Some(ref prev) = self.prev_depth {
				if prev.dim() == depth.dim() {
					let alpha = self.temporal_alpha;
					depth.zip_mut_with(prev, |curr, &prev_val| {
						*curr = alpha * *curr + (1.0 - alpha) * prev_val;
					});
				}
			}
			self.prev_depth = Some(depth.clone());
		}

		self.frame_index += 1;
		depth
	}

	fn normalize(&mut self, raw: Array2<f32>) -> Array2<f32> {
		match self.normalize_mode {
			NormalizeMode::PerFrame => normalize_minmax(raw),
			NormalizeMode::RunningEMA => {
				let min = raw.iter().copied().fold(f32::INFINITY, f32::min);
				let max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);

				let adapt_rate = 0.05;
				if self.frame_index == 0 {
					self.ema_min = min;
					self.ema_max = max;
				} else {
					self.ema_min = self.ema_min + adapt_rate * (min - self.ema_min);
					self.ema_max = self.ema_max + adapt_rate * (max - self.ema_max);
				}

				let range = self.ema_max - self.ema_min;
				if range > 1e-6 {
					raw.mapv(|v| ((v - self.ema_min) / range).clamp(0.0, 1.0))
				} else {
					raw.mapv(|_| 0.5)
				}
			}
			NormalizeMode::Global => {
				let range = self.global_max - self.global_min;
				if range > 1e-6 {
					raw.mapv(|v| ((v - self.global_min) / range).clamp(0.0, 1.0))
				} else {
					raw.mapv(|_| 0.5)
				}
			}
		}
	}
}

fn normalize_minmax(mut depth: Array2<f32>) -> Array2<f32> {
	let min = depth.iter().copied().fold(f32::INFINITY, f32::min);
	let max = depth.iter().copied().fold(f32::NEG_INFINITY, f32::max);
	let range = max - min;
	if range > 1e-6 {
		depth.mapv_inplace(|v| (v - min) / range);
	}
	depth
}

pub fn bilateral_filter(depth: &Array2<f32>, sigma_space: f32, sigma_color: f32) -> Array2<f32> {
	let (h, w) = depth.dim();
	let mut out = Array2::zeros((h, w));
	let radius = (sigma_space * 2.0).ceil() as i32;
	let space_coeff = -0.5 / (sigma_space * sigma_space);
	let color_coeff = -0.5 / (sigma_color * sigma_color);

	for y in 0..h {
		for x in 0..w {
			let center = depth[[y, x]];
			let mut sum = 0.0f32;
			let mut weight_sum = 0.0f32;

			let y0 = (y as i32 - radius).max(0) as usize;
			let y1 = (y as i32 + radius).min(h as i32 - 1) as usize;
			let x0 = (x as i32 - radius).max(0) as usize;
			let x1 = (x as i32 + radius).min(w as i32 - 1) as usize;

			for ny in y0..=y1 {
				for nx in x0..=x1 {
					let dy = ny as f32 - y as f32;
					let dx = nx as f32 - x as f32;
					let spatial_dist = dx * dx + dy * dy;
					let val = depth[[ny, nx]];
					let color_dist = (val - center) * (val - center);

					let weight = (spatial_dist * space_coeff + color_dist * color_coeff).exp();
					sum += val * weight;
					weight_sum += weight;
				}
			}

			out[[y, x]] = if weight_sum > 0.0 { sum / weight_sum } else { center };
		}
	}

	out
}

pub fn gaussian_blur(depth: &Array2<f32>, sigma: f32) -> Array2<f32> {
	let radius = (sigma * 3.0).ceil() as i32;
	let kernel_size = (2 * radius + 1) as usize;
	let mut kernel = vec![0.0f32; kernel_size];
	let coeff = -0.5 / (sigma * sigma);

	for i in 0..kernel_size {
		let d = i as f32 - radius as f32;
		kernel[i] = (d * d * coeff).exp();
	}
	let ksum: f32 = kernel.iter().sum();
	for v in &mut kernel {
		*v /= ksum;
	}

	let (h, w) = depth.dim();

	let mut temp = Array2::zeros((h, w));
	for y in 0..h {
		for x in 0..w {
			let mut sum = 0.0f32;
			for i in 0..kernel_size {
				let nx = (x as i32 + i as i32 - radius).clamp(0, w as i32 - 1) as usize;
				sum += depth[[y, nx]] * kernel[i];
			}
			temp[[y, x]] = sum;
		}
	}

	let mut out = Array2::zeros((h, w));
	for y in 0..h {
		for x in 0..w {
			let mut sum = 0.0f32;
			for i in 0..kernel_size {
				let ny = (y as i32 + i as i32 - radius).clamp(0, h as i32 - 1) as usize;
				sum += temp[[ny, x]] * kernel[i];
			}
			out[[y, x]] = sum;
		}
	}

	out
}
