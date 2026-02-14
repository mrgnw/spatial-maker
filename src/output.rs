use crate::error::{SpatialError, SpatialResult};
use image::DynamicImage;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
	SideBySide,
	TopAndBottom,
	Separate,
}

impl OutputFormat {
	pub fn name(&self) -> &'static str {
		match self {
			OutputFormat::SideBySide => "side-by-side",
			OutputFormat::TopAndBottom => "top-and-bottom",
			OutputFormat::Separate => "separate",
		}
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageEncoding {
	Jpeg { quality: u8 },
	Png,
}

impl ImageEncoding {
	pub fn extension(&self) -> &'static str {
		match self {
			ImageEncoding::Jpeg { .. } => "jpg",
			ImageEncoding::Png => "png",
		}
	}

	pub fn from_path<P: AsRef<Path>>(path: P) -> Self {
		let ext = path
			.as_ref()
			.extension()
			.and_then(|e| e.to_str())
			.unwrap_or("")
			.to_lowercase();

		match ext.as_str() {
			"png" => ImageEncoding::Png,
			_ => ImageEncoding::Jpeg { quality: 95 },
		}
	}
}

#[derive(Clone, Debug)]
pub struct MVHEVCConfig {
	pub spatial_cli_path: Option<PathBuf>,
	pub enabled: bool,
	pub quality: u8,
	pub keep_intermediate: bool,
}

impl Default for MVHEVCConfig {
	fn default() -> Self {
		Self {
			spatial_cli_path: None,
			enabled: false,
			quality: 95,
			keep_intermediate: false,
		}
	}
}

#[derive(Clone, Debug)]
pub struct OutputOptions {
	pub layout: OutputFormat,
	pub image_format: ImageEncoding,
	pub mvhevc: Option<MVHEVCConfig>,
}

impl Default for OutputOptions {
	fn default() -> Self {
		Self {
			layout: OutputFormat::SideBySide,
			image_format: ImageEncoding::Jpeg { quality: 95 },
			mvhevc: None,
		}
	}
}

pub fn create_sbs_image(left: &DynamicImage, right: &DynamicImage) -> DynamicImage {
	let left_width = left.width();
	let left_height = left.height();

	let combined_width = left_width + right.width();
	let mut combined = DynamicImage::new_rgb8(combined_width, left_height);

	image::imageops::overlay(&mut combined, left, 0, 0);
	image::imageops::overlay(&mut combined, right, left_width as i64, 0);

	combined
}

pub fn save_stereo_image(
	left: &DynamicImage,
	right: &DynamicImage,
	output_path: impl AsRef<Path>,
	options: OutputOptions,
) -> SpatialResult<()> {
	let output_path = output_path.as_ref();

	if let Some(parent) = output_path.parent() {
		std::fs::create_dir_all(parent).map_err(|e| {
			SpatialError::ImageError(format!("Failed to create output directory: {}", e))
		})?;
	}

	match options.layout {
		OutputFormat::SideBySide => {
			save_side_by_side(left, right, output_path, options.image_format)?;
		}
		OutputFormat::TopAndBottom => {
			save_top_and_bottom(left, right, output_path, options.image_format)?;
		}
		OutputFormat::Separate => {
			save_separate(left, right, output_path, options.image_format)?;
		}
	}

	if let Some(mvhevc_config) = options.mvhevc {
		if mvhevc_config.enabled {
			encode_mvhevc(output_path, &mvhevc_config)?;
			if !mvhevc_config.keep_intermediate {
				let _ = std::fs::remove_file(output_path);
			}
		}
	}

	Ok(())
}

fn save_side_by_side(
	left: &DynamicImage,
	right: &DynamicImage,
	output_path: &Path,
	encoding: ImageEncoding,
) -> SpatialResult<()> {
	if left.height() != right.height() {
		return Err(SpatialError::ImageError(format!(
			"Left and right images must have the same height: {} != {}",
			left.height(),
			right.height()
		)));
	}

	let combined = create_sbs_image(left, right);
	save_image(&combined, output_path, encoding)
}

fn save_top_and_bottom(
	left: &DynamicImage,
	right: &DynamicImage,
	output_path: &Path,
	encoding: ImageEncoding,
) -> SpatialResult<()> {
	if left.width() != right.width() {
		return Err(SpatialError::ImageError(format!(
			"Left and right images must have the same width: {} != {}",
			left.width(),
			right.width()
		)));
	}

	let combined_height = left.height() + right.height();
	let mut combined = DynamicImage::new_rgb8(left.width(), combined_height);

	image::imageops::overlay(&mut combined, left, 0, 0);
	image::imageops::overlay(&mut combined, right, 0, left.height() as i64);

	save_image(&combined, output_path, encoding)
}

fn save_separate(
	left: &DynamicImage,
	right: &DynamicImage,
	output_path: &Path,
	encoding: ImageEncoding,
) -> SpatialResult<()> {
	let stem = output_path
		.file_stem()
		.and_then(|s| s.to_str())
		.ok_or_else(|| SpatialError::ImageError("Invalid output path".to_string()))?;

	let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
	let ext = encoding.extension();

	let left_path = parent.join(format!("{}_L.{}", stem, ext));
	let right_path = parent.join(format!("{}_R.{}", stem, ext));

	save_image(left, &left_path, encoding)?;
	save_image(right, &right_path, encoding)?;

	Ok(())
}

fn save_image(image: &DynamicImage, path: &Path, encoding: ImageEncoding) -> SpatialResult<()> {
	match encoding {
		ImageEncoding::Jpeg { quality } => {
			let rgb_image = image.to_rgb8();
			let file = std::fs::File::create(path).map_err(|e| {
				SpatialError::ImageError(format!("Failed to create output file: {}", e))
			})?;

			let mut jpeg_encoder =
				image::codecs::jpeg::JpegEncoder::new_with_quality(file, quality);
			jpeg_encoder
				.encode(
					rgb_image.as_ref(),
					rgb_image.width(),
					rgb_image.height(),
					image::ExtendedColorType::Rgb8,
				)
				.map_err(|e| SpatialError::ImageError(format!("Failed to encode JPEG: {}", e)))?;
		}
		ImageEncoding::Png => {
			image
				.save(path)
				.map_err(|e| SpatialError::ImageError(format!("Failed to save PNG: {}", e)))?;
		}
	}

	Ok(())
}

fn encode_mvhevc(stereo_path: &Path, config: &MVHEVCConfig) -> SpatialResult<()> {
	let spatial_path = config
		.spatial_cli_path
		.as_ref()
		.map(|p| p.as_path())
		.unwrap_or_else(|| Path::new("spatial"));

	let hevc_path = stereo_path.with_extension("heic");

	let format = if stereo_path.to_string_lossy().contains("top-bottom")
		|| stereo_path.to_string_lossy().contains("_tb_")
	{
		"hou"
	} else {
		"sbs"
	};

	let quality_normalized = (config.quality as f32 / 100.0).clamp(0.0, 1.0);

	let mut cmd = Command::new(spatial_path);
	cmd.arg("make")
		.arg("--input")
		.arg(stereo_path)
		.arg("--output")
		.arg(&hevc_path)
		.arg("--format")
		.arg(format)
		.arg("--quality")
		.arg(quality_normalized.to_string())
		.arg("--overwrite");

	let output = cmd.output().map_err(|e| {
		SpatialError::ImageError(format!(
			"Failed to run `spatial` CLI: {}. Ensure the `spatial` tool is installed and in PATH.",
			e
		))
	})?;

	if !output.status.success() {
		let stderr = String::from_utf8_lossy(&output.stderr);
		return Err(SpatialError::ImageError(format!(
			"MV-HEVC encoding failed: {}",
			stderr
		)));
	}

	Ok(())
}
