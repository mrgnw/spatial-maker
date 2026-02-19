pub mod depth;
pub mod depth_filter;
pub mod error;
pub mod image_loader;
pub mod model;
pub mod output;
pub mod stereo;
pub mod video;

#[cfg(all(target_os = "macos", feature = "coreml"))]
pub mod depth_coreml;

pub use depth_filter::DepthProcessor;
pub use error::{SpatialError, SpatialResult};
pub use image_loader::load_image;
pub use model::{find_model, get_checkpoint_dir, model_exists};
pub use output::{
	create_sbs_image, save_stereo_image,
	DepthFormat, ImageEncoding, MVHEVCConfig, OutputFormat, OutputOptions, OutputType,
	depth_formats, needs_depth, needs_stereo, parse_output_types, save_depth_map, stereo_types,
};
pub use stereo::generate_stereo_pair;
pub use video::{get_video_metadata, process_video, ProgressCallback, VideoMetadata, VideoProgress};

#[cfg(all(target_os = "macos", feature = "coreml"))]
pub use depth_coreml::CoreMLDepthEstimator;

#[cfg(feature = "onnx")]
pub use depth::OnnxDepthEstimator;

use std::path::Path;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum NormalizeMode {
	PerFrame,
	RunningEMA,
	Global,
}

impl Default for NormalizeMode {
	fn default() -> Self {
		Self::RunningEMA
	}
}

impl std::fmt::Display for NormalizeMode {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::PerFrame => write!(f, "per-frame"),
			Self::RunningEMA => write!(f, "running"),
			Self::Global => write!(f, "global"),
		}
	}
}

impl std::str::FromStr for NormalizeMode {
	type Err = String;
	fn from_str(s: &str) -> Result<Self, Self::Err> {
		match s.to_lowercase().as_str() {
			"per-frame" | "perframe" | "frame" => Ok(Self::PerFrame),
			"running" | "ema" | "running-ema" => Ok(Self::RunningEMA),
			"global" | "two-pass" | "twopass" => Ok(Self::Global),
			_ => Err(format!("Unknown normalize mode: '{}'. Use: per-frame, running, global", s)),
		}
	}
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SpatialConfig {
	pub encoder_size: String,
	pub max_disparity: u32,
	pub target_depth_size: u32,
	pub temporal_alpha: f32,
	pub bilateral_sigma_space: f32,
	pub bilateral_sigma_color: f32,
	pub depth_blur_sigma: f32,
	pub normalize_mode: NormalizeMode,
}

pub type StereoOutputFormat = OutputFormat;

impl Default for SpatialConfig {
	fn default() -> Self {
		Self {
			encoder_size: "s".to_string(),
			max_disparity: 30,
			target_depth_size: 518,
			temporal_alpha: 0.7,
			bilateral_sigma_space: 5.0,
			bilateral_sigma_color: 0.1,
			depth_blur_sigma: 1.5,
			normalize_mode: NormalizeMode::RunningEMA,
		}
	}
}

pub struct ProcessPhotoOutput {
	pub depth_paths: Vec<std::path::PathBuf>,
	pub stereo_paths: Vec<std::path::PathBuf>,
}

pub async fn process_photo(
	input_path: &Path,
	output_base_path: &Path,
	config: SpatialConfig,
	output_types: &[OutputType],
	output_options: OutputOptions,
) -> SpatialResult<ProcessPhotoOutput> {
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

	let mut result = ProcessPhotoOutput {
		depth_paths: Vec::new(),
		stereo_paths: Vec::new(),
	};

	if needs_depth(output_types) {
		let parent = output_base_path.parent().unwrap_or_else(|| Path::new("."));
		let stem = output_base_path.file_stem().and_then(|s| s.to_str()).unwrap_or("output");

		for fmt in depth_formats(output_types) {
			let filename = format!("{}-depth{}.{}", stem, fmt.suffix(), fmt.extension());
			let depth_path = parent.join(&filename);
			save_depth_map(&depth_map, &depth_path, fmt)?;
			result.depth_paths.push(depth_path.clone());
		}
	}

	if needs_stereo(output_types) {
		let (left, right) = generate_stereo_pair(&input_image, &depth_map, config.max_disparity)?;
		let src_ext = input_path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
		let stereo_ext = match src_ext.as_str() {
			"heic" | "heif" | "avif" | "jxl" => "jpg",
			"" => "jpg",
			other => other,
		};
		let parent = output_base_path.parent().unwrap_or_else(|| Path::new("."));
		let stem = output_base_path.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
		let stereo_path = parent.join(format!("{}-spatial.{}", stem, stereo_ext));
		save_stereo_image(&left, &right, &stereo_path, output_options)?;
		result.stereo_paths.push(stereo_path);
	}

	Ok(result)
}

pub async fn process_video_sbs(
	input_path: &Path,
	output_path: &Path,
	config: SpatialConfig,
	progress_cb: Option<ProgressCallback>,
) -> SpatialResult<()> {
	video::process_video(input_path, output_path, config, &[OutputType::Spatial], progress_cb).await
}
