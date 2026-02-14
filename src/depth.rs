#[cfg(feature = "onnx")]
use crate::error::{SpatialError, SpatialResult};
#[cfg(feature = "onnx")]
use image::DynamicImage;
#[cfg(feature = "onnx")]
use ndarray::Array2;
#[cfg(feature = "onnx")]
use ort::session::{builder::GraphOptimizationLevel, Session};

#[cfg(feature = "onnx")]
const INPUT_SIZE: u32 = 518;
#[cfg(feature = "onnx")]
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
#[cfg(feature = "onnx")]
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[cfg(feature = "onnx")]
pub struct OnnxDepthEstimator {
	session: Session,
}

#[cfg(feature = "onnx")]
impl OnnxDepthEstimator {
	pub fn new(model_path: &str) -> SpatialResult<Self> {
		let session = Session::builder()
			.map_err(|e| SpatialError::ModelError(format!("Failed to create session: {}", e)))?
			.with_optimization_level(GraphOptimizationLevel::Level3)
			.map_err(|e| SpatialError::ModelError(format!("Failed to set opt level: {}", e)))?
			.with_intra_threads(4)
			.map_err(|e| SpatialError::ModelError(format!("Failed to set threads: {}", e)))?
			.commit_from_file(model_path)
			.map_err(|e| SpatialError::ModelError(format!("Failed to load ONNX model: {}", e)))?;

		Ok(Self { session })
	}

	pub fn estimate(&mut self, image: &DynamicImage) -> SpatialResult<Array2<f32>> {
		let (orig_width, orig_height) = (image.width(), image.height());
		let size = INPUT_SIZE as usize;

		let resized = image.resize_exact(
			INPUT_SIZE,
			INPUT_SIZE,
			image::imageops::FilterType::Lanczos3,
		);

		let rgb = resized.to_rgb8();
		let mut input_data = vec![0.0f32; 1 * 3 * size * size];

		for (i, pixel) in rgb.pixels().enumerate() {
			for c in 0..3 {
				let normalized = (pixel[c] as f32 / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
				input_data[c * size * size + i] = normalized;
			}
		}

		let input_value = ort::value::Value::from_array(([1usize, 3, size, size], input_data))
			.map_err(|e| SpatialError::TensorError(format!("Failed to create input: {}", e)))?;

		let outputs = self.session.run(ort::inputs![input_value])
			.map_err(|e| SpatialError::ModelError(format!("Inference failed: {}", e)))?;

		let (shape, data) = outputs[0].try_extract_tensor::<f32>()
			.map_err(|e| SpatialError::TensorError(format!("Failed to extract output: {}", e)))?;

		let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
		let h = dims[1];
		let w = dims[2];

		let depth_data: Vec<f32> = data.to_vec();

		let min_val = depth_data.iter().copied().fold(f32::INFINITY, f32::min);
		let max_val = depth_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
		let range = max_val - min_val;

		let normalized: Vec<f32> = if range > 1e-6 {
			depth_data.iter().map(|&v| (v - min_val) / range).collect()
		} else {
			vec![0.5; depth_data.len()]
		};

		let depth_image = image::ImageBuffer::from_fn(w as u32, h as u32, |x, y| {
			image::Luma([normalized[y as usize * w + x as usize]])
		});

		let resized_depth = image::imageops::resize(
			&depth_image,
			orig_width,
			orig_height,
			image::imageops::FilterType::Lanczos3,
		);

		let data: Vec<f32> = resized_depth.pixels().map(|p| p[0]).collect();
		Array2::from_shape_vec((orig_height as usize, orig_width as usize), data)
			.map_err(|e| SpatialError::TensorError(format!("Failed to reshape depth: {}", e)))
	}
}
