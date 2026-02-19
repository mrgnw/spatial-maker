use crate::error::{SpatialError, SpatialResult};
use image::DynamicImage;
use ndarray::Array2;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthFormat {
	Avif,
	Png,
	Png16,
}

impl DepthFormat {
	pub fn extension(&self) -> &'static str {
		match self {
			DepthFormat::Avif => "avif",
			DepthFormat::Png => "png",
			DepthFormat::Png16 => "png",
		}
	}

	pub fn suffix(&self) -> &'static str {
		match self {
			DepthFormat::Avif => "",
			DepthFormat::Png => "",
			DepthFormat::Png16 => "-16bit",
		}
	}
}

pub const DEFAULT_DEPTH_FORMAT: DepthFormat = DepthFormat::Avif;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputType {
	Depth(Vec<DepthFormat>),
	SideBySide,
	TopAndBottom,
	Separate,
	Spatial,
}

pub fn needs_depth(types: &[OutputType]) -> bool {
	types.iter().any(|t| matches!(t, OutputType::Depth(_)))
}

pub fn needs_stereo(types: &[OutputType]) -> bool {
	types.iter().any(|t| matches!(t, OutputType::SideBySide | OutputType::TopAndBottom | OutputType::Separate | OutputType::Spatial))
}

pub fn depth_formats(types: &[OutputType]) -> Vec<DepthFormat> {
	types.iter().filter_map(|t| {
		if let OutputType::Depth(fmts) = t { Some(fmts.clone()) } else { None }
	}).flatten().collect()
}

pub fn stereo_types(types: &[OutputType]) -> Vec<&OutputType> {
	types.iter().filter(|t| matches!(t, OutputType::SideBySide | OutputType::TopAndBottom | OutputType::Separate | OutputType::Spatial)).collect()
}

fn is_depth_format(s: &str) -> bool {
	matches!(s, "avif" | "png" | "png16")
}

fn is_stereo_type(s: &str) -> bool {
	matches!(s, "sbs" | "tab" | "sep" | "spatial")
}

fn parse_depth_format(s: &str) -> Result<DepthFormat, String> {
	match s {
		"avif" => Ok(DepthFormat::Avif),
		"png" => Ok(DepthFormat::Png),
		"png16" => Ok(DepthFormat::Png16),
		_ => Err(format!("Unknown depth format: '{}'. Use: avif, png, png16", s)),
	}
}

fn parse_stereo_type(s: &str) -> Result<OutputType, String> {
	match s {
		"sbs" => Ok(OutputType::SideBySide),
		"tab" => Ok(OutputType::TopAndBottom),
		"sep" => Ok(OutputType::Separate),
		"spatial" => Ok(OutputType::Spatial),
		_ => Err(format!("Unknown output type: '{}'", s)),
	}
}

pub fn parse_output_types(s: &str) -> Result<Vec<OutputType>, String> {
	let parts: Vec<&str> = s.split(',').map(|p| p.trim()).filter(|p| !p.is_empty()).collect();
	let mut types = Vec::new();
	let mut depth_fmts = Vec::new();
	let mut has_depth = false;

	for part in &parts {
		if *part == "depth" {
			has_depth = true;
			continue;
		}

		if let Some(after_colon) = part.strip_prefix("depth:") {
			has_depth = true;
			depth_fmts.push(parse_depth_format(after_colon)?);
			continue;
		}

		if has_depth && is_depth_format(part) {
			depth_fmts.push(parse_depth_format(part)?);
			continue;
		}

		if is_stereo_type(part) {
			types.push(parse_stereo_type(part)?);
		} else if is_depth_format(part) {
			return Err(format!(
				"'{}' must be specified as a depth sub-format: depth:{}", part, part
			));
		} else {
			return Err(format!("Unknown output type: '{}'", part));
		}
	}

	if has_depth {
		if depth_fmts.is_empty() {
			depth_fmts.push(DEFAULT_DEPTH_FORMAT);
		}
		types.insert(0, OutputType::Depth(depth_fmts));
	}

	if types.is_empty() {
		return Err("No output types specified".to_string());
	}

	Ok(types)
}

// --- Depth map saving ---

fn normalize_depth(depth: &Array2<f32>) -> (f32, f32) {
	let mut min_val = f32::INFINITY;
	let mut max_val = f32::NEG_INFINITY;
	for &v in depth.iter() {
		if v < min_val { min_val = v; }
		if v > max_val { max_val = v; }
	}
	(min_val, max_val)
}

pub fn save_depth_png8(depth: &Array2<f32>, path: &Path) -> SpatialResult<()> {
	let (h, w) = depth.dim();
	let (min_val, max_val) = normalize_depth(depth);
	let range = max_val - min_val;

	let pixels: Vec<u8> = depth.iter().map(|&v| {
		if range > 1e-6 {
			((v - min_val) / range * 255.0).round() as u8
		} else {
			128u8
		}
	}).collect();

	let img = image::GrayImage::from_raw(w as u32, h as u32, pixels)
		.ok_or_else(|| SpatialError::ImageError("Failed to create grayscale image".to_string()))?;

	img.save(path)
		.map_err(|e| SpatialError::ImageError(format!("Failed to save depth PNG: {}", e)))?;

	Ok(())
}

pub fn save_depth_png16(depth: &Array2<f32>, path: &Path) -> SpatialResult<()> {
	let (h, w) = depth.dim();
	let (min_val, max_val) = normalize_depth(depth);
	let range = max_val - min_val;

	let pixels: Vec<u16> = depth.iter().map(|&v| {
		if range > 1e-6 {
			((v - min_val) / range * 65535.0).round() as u16
		} else {
			32768u16
		}
	}).collect();

	let file = std::fs::File::create(path)
		.map_err(|e| SpatialError::ImageError(format!("Failed to create output file: {}", e)))?;
	let writer = std::io::BufWriter::new(file);

	let encoder = image::codecs::png::PngEncoder::new(writer);
	use image::ImageEncoder;

	let byte_data: Vec<u8> = pixels.iter().flat_map(|&v| v.to_be_bytes()).collect();

	encoder.write_image(
		&byte_data,
		w as u32,
		h as u32,
		image::ExtendedColorType::L16,
	).map_err(|e| SpatialError::ImageError(format!("Failed to encode 16-bit PNG: {}", e)))?;

	Ok(())
}

pub fn save_depth_avif(depth: &Array2<f32>, path: &Path) -> SpatialResult<()> {
	let (h, w) = depth.dim();
	let (min_val, max_val) = normalize_depth(depth);
	let range = max_val - min_val;

	let pixels: Vec<u8> = depth.iter().map(|&v| {
		if range > 1e-6 {
			((v - min_val) / range * 255.0).round() as u8
		} else {
			128u8
		}
	}).collect();

	let rgb_pixels: Vec<u8> = pixels.iter().flat_map(|&v| [v, v, v]).collect();

	let path_str = path.to_str()
		.ok_or_else(|| SpatialError::ImageError("Invalid output path".to_string()))?;

	let mut child = Command::new("ffmpeg")
		.args([
			"-f", "rawvideo",
			"-pix_fmt", "rgb24",
			"-s", &format!("{}x{}", w, h),
			"-i", "-",
			"-frames:v", "1",
			"-c:v", "libsvtav1",
			"-crf", "23",
			"-y",
			path_str,
		])
		.stdin(std::process::Stdio::piped())
		.stdout(std::process::Stdio::null())
		.stderr(std::process::Stdio::piped())
		.spawn()
		.map_err(|e| SpatialError::Other(format!("Failed to spawn ffmpeg for AVIF encoding: {}", e)))?;

	if let Some(mut stdin) = child.stdin.take() {
		use std::io::Write;
		stdin.write_all(&rgb_pixels)
			.map_err(|e| SpatialError::IoError(format!("Failed to write depth data to ffmpeg: {}", e)))?;
	}

	let output = child.wait_with_output()
		.map_err(|e| SpatialError::Other(format!("ffmpeg AVIF encoding failed: {}", e)))?;

	if !output.status.success() {
		let stderr = String::from_utf8_lossy(&output.stderr);
		return Err(SpatialError::ImageError(format!("ffmpeg AVIF encoding failed: {}", stderr)));
	}

	Ok(())
}

pub fn save_depth_map(depth: &Array2<f32>, path: &Path, format: DepthFormat) -> SpatialResult<()> {
	if let Some(parent) = path.parent() {
		std::fs::create_dir_all(parent).map_err(|e| {
			SpatialError::ImageError(format!("Failed to create output directory: {}", e))
		})?;
	}

	match format {
		DepthFormat::Avif => save_depth_avif(depth, path)?,
		DepthFormat::Png => save_depth_png8(depth, path)?,
		DepthFormat::Png16 => save_depth_png16(depth, path)?,
	}

	Ok(())
}

// --- Existing stereo output ---

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

pub fn encode_mvhevc(stereo_path: &Path, config: &MVHEVCConfig) -> SpatialResult<()> {
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
