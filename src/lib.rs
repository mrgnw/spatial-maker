pub mod depth;
pub mod error;
pub mod image_loader;
pub mod model;
pub mod output;
pub mod stereo;
pub mod video;

#[cfg(all(target_os = "macos", feature = "coreml"))]
pub mod depth_coreml;

pub use error::{SpatialError, SpatialResult};
pub use image_loader::load_image;
pub use model::{find_model, get_checkpoint_dir, model_exists};
pub use output::{create_sbs_image, save_stereo_image, ImageEncoding, MVHEVCConfig, OutputFormat, OutputOptions};
pub use stereo::generate_stereo_pair;
pub use video::{get_video_metadata, process_video, ProgressCallback, VideoMetadata, VideoProgress};

#[cfg(all(target_os = "macos", feature = "coreml"))]
pub use depth_coreml::CoreMLDepthEstimator;

#[cfg(feature = "onnx")]
pub use depth::OnnxDepthEstimator;

use std::path::Path;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SpatialConfig {
	pub encoder_size: String,
	pub max_disparity: u32,
	pub target_depth_size: u32,
}

pub type StereoOutputFormat = OutputFormat;

impl Default for SpatialConfig {
	fn default() -> Self {
		Self {
			encoder_size: "s".to_string(),
			max_disparity: 30,
			target_depth_size: 518,
		}
	}
}

pub async fn process_photo(
	input_path: &Path,
	output_path: &Path,
	config: SpatialConfig,
	output_options: OutputOptions,
) -> SpatialResult<()> {
	let input_image = load_image(input_path).await?;

	model::ensure_model_exists::<fn(u64, u64)>(&config.encoder_size, None).await?;

	#[cfg(all(target_os = "macos", feature = "coreml"))]
	let depth_map = {
		let model_path = model::find_model(&config.encoder_size)?;
		let model_str = model_path.to_str().ok_or_else(|| {
			SpatialError::ModelError("Invalid model path encoding".to_string())
		})?;
		let estimator = CoreMLDepthEstimator::new(model_str)?;
		estimator.estimate(&input_image)?
	};

	#[cfg(not(all(target_os = "macos", feature = "coreml")))]
	let depth_map = {
		#[cfg(feature = "onnx")]
		{
			let model_path = model::find_model(&config.encoder_size)?;
			let estimator = OnnxDepthEstimator::new(model_path.to_str().unwrap())?;
			estimator.estimate(&input_image)?
		}
		#[cfg(not(feature = "onnx"))]
		{
			return Err(SpatialError::ConfigError(
				"No depth backend enabled. Enable 'coreml' (macOS) or 'onnx' feature.".to_string(),
			));
		}
	};

	let (left, right) = generate_stereo_pair(&input_image, &depth_map, config.max_disparity)?;

	save_stereo_image(&left, &right, output_path, output_options)?;

	Ok(())
}

pub async fn process_video_sbs(
	input_path: &Path,
	output_path: &Path,
	config: SpatialConfig,
	progress_cb: Option<ProgressCallback>,
) -> SpatialResult<()> {
	video::process_video(input_path, output_path, config, progress_cb).await
}
