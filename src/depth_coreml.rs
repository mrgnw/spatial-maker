use crate::error::{SpatialError, SpatialResult};
use image::{DynamicImage, ImageBuffer, Luma};
use ndarray::Array2;
use std::ffi::CString;

const INPUT_SIZE: u32 = 518;
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

extern "C" {
	fn coreml_load_model(path: *const std::os::raw::c_char) -> *mut std::os::raw::c_void;
	fn coreml_unload_model(model: *mut std::os::raw::c_void);
	fn coreml_infer_depth(
		model: *mut std::os::raw::c_void,
		rgb_data: *const f32,
		width: i32,
		height: i32,
		output: *mut f32,
	) -> i32;
}

pub struct CoreMLDepthEstimator {
	model: *mut std::os::raw::c_void,
}

impl CoreMLDepthEstimator {
	pub fn new(model_path: &str) -> SpatialResult<Self> {
		let c_path = CString::new(model_path)
			.map_err(|e| SpatialError::ModelError(format!("Invalid model path: {}", e)))?;

		let model = unsafe { coreml_load_model(c_path.as_ptr()) };

		if model.is_null() {
			return Err(SpatialError::ModelError(format!(
				"Failed to load CoreML model: {}",
				model_path
			)));
		}

		tracing::info!("CoreML model loaded: {}", model_path);

		Ok(Self { model })
	}

	pub fn estimate_raw(&self, image: &DynamicImage) -> SpatialResult<ImageBuffer<Luma<f32>, Vec<f32>>> {
		let (orig_width, orig_height) = (image.width(), image.height());

		let resized = image.resize_exact(
			INPUT_SIZE,
			INPUT_SIZE,
			image::imageops::FilterType::Lanczos3,
		);

		let rgb = resized.to_rgb8();
		let mut input_data = Vec::with_capacity(3 * INPUT_SIZE as usize * INPUT_SIZE as usize);

		for c in 0..3 {
			for pixel in rgb.pixels() {
				let normalized = (pixel[c] as f32 / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
				input_data.push(normalized);
			}
		}

		let output_size = (INPUT_SIZE * INPUT_SIZE) as usize;
		let mut output_data = vec![0.0f32; output_size];

		let result = unsafe {
			coreml_infer_depth(
				self.model,
				input_data.as_ptr(),
				INPUT_SIZE as i32,
				INPUT_SIZE as i32,
				output_data.as_mut_ptr(),
			)
		};

		if result != 0 {
			return Err(SpatialError::ModelError(format!(
				"CoreML inference failed with error code: {}",
				result
			)));
		}

		let min_val = output_data.iter().copied().fold(f32::INFINITY, f32::min);
		let max_val = output_data
			.iter()
			.copied()
			.fold(f32::NEG_INFINITY, f32::max);
		let range = max_val - min_val;

		if range > 1e-6 {
			for v in &mut output_data {
				*v = (*v - min_val) / range;
			}
		}

		let depth_image = ImageBuffer::from_fn(INPUT_SIZE, INPUT_SIZE, |x, y| {
			let idx = (y * INPUT_SIZE + x) as usize;
			Luma([output_data[idx]])
		});

		let resized_depth = image::imageops::resize(
			&depth_image,
			orig_width,
			orig_height,
			image::imageops::FilterType::Lanczos3,
		);

		Ok(resized_depth)
	}

	pub fn estimate(&self, image: &DynamicImage) -> SpatialResult<Array2<f32>> {
		let depth_image = self.estimate_raw(image)?;
		let (width, height) = depth_image.dimensions();
		let data: Vec<f32> = depth_image.pixels().map(|p| p[0]).collect();
		let depth_2d = Array2::from_shape_vec((height as usize, width as usize), data)
			.map_err(|e| SpatialError::TensorError(format!("Failed to reshape depth: {}", e)))?;
		Ok(depth_2d)
	}
}

impl Drop for CoreMLDepthEstimator {
	fn drop(&mut self) {
		unsafe {
			coreml_unload_model(self.model);
		}
	}
}

unsafe impl Send for CoreMLDepthEstimator {}
unsafe impl Sync for CoreMLDepthEstimator {}
