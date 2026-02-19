use crate::error::{SpatialError, SpatialResult};
use image::DynamicImage;
use std::path::Path;
use std::process::Command;

pub async fn load_image(path: impl AsRef<Path>) -> SpatialResult<DynamicImage> {
	let path = path.as_ref();

	if !path.exists() {
		return Err(SpatialError::ImageError(format!(
			"Image file not found: {:?}",
			path
		)));
	}

	let extension = path
		.extension()
		.and_then(|ext| ext.to_str())
		.map(|s| s.to_lowercase())
		.ok_or_else(|| SpatialError::ImageError(format!("File has no extension: {:?}", path)))?;

	match extension.as_str() {
		"avif" => load_avif(path).await,
		"jxl" => load_jxl(path).await,
		"heic" | "heif" => load_heic(path).await,
		"jpg" | "jpeg" | "png" | "gif" | "bmp" | "tiff" | "tif" | "webp" => load_standard(path),
		_ => Err(SpatialError::ImageError(format!(
			"Unsupported image format: .{}",
			extension
		))),
	}
}

fn load_standard(path: impl AsRef<Path>) -> SpatialResult<DynamicImage> {
	let path = path.as_ref();
	let img = image::open(path)
		.map_err(|e| SpatialError::ImageError(format!("Failed to load image {:?}: {}", path, e)))?;
	Ok(img)
}

async fn load_avif(path: &Path) -> SpatialResult<DynamicImage> {
	#[cfg(feature = "avif")]
	{
		match image::open(path) {
			Ok(img) => return Ok(img),
			Err(e) => {
				tracing::warn!("Native AVIF decoder failed: {}, falling back to ffmpeg", e);
			}
		}
	}
	load_with_ffmpeg(path, "avif").await
}

async fn load_jxl(path: &Path) -> SpatialResult<DynamicImage> {
	#[cfg(feature = "jxl")]
	{
		match load_jxl_native(path) {
			Ok(img) => return Ok(img),
			Err(e) => {
				tracing::warn!("Native JXL decoder failed: {}, falling back to ffmpeg", e);
			}
		}
	}
	load_with_ffmpeg(path, "jxl").await
}

async fn load_heic(path: &Path) -> SpatialResult<DynamicImage> {
	#[cfg(feature = "heic")]
	{
		match load_heic_native(path) {
			Ok(img) => return Ok(img),
			Err(e) => {
				tracing::warn!("Native HEIC decoder failed: {}, falling back to ffmpeg", e);
			}
		}
	}
	load_with_ffmpeg(path, "heic").await
}

#[cfg(feature = "jxl")]
fn load_jxl_native(path: &Path) -> SpatialResult<DynamicImage> {
	use jxl_oxide::JxlImage;

	let data = std::fs::read(path)
		.map_err(|e| SpatialError::IoError(format!("Failed to read JXL file: {}", e)))?;

	let jxl_image = JxlImage::builder()
		.read(&data[..])
		.map_err(|e| SpatialError::ImageError(format!("JXL decode failed: {:?}", e)))?;

	let width = jxl_image.width();
	let height = jxl_image.height();

	let render = jxl_image
		.render_frame(0)
		.map_err(|e| SpatialError::ImageError(format!("JXL render failed: {:?}", e)))?;

	let planar = render.image_planar();
	if planar.is_empty() {
		return Err(SpatialError::ImageError(
			"JXL image has no color channels".to_string(),
		));
	}

	let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);
	for y in 0..height {
		for x in 0..width {
			let idx = (y * width + x) as usize;
			let r = (planar[0].buf()[idx] * 255.0).clamp(0.0, 255.0) as u8;
			let g = if planar.len() > 1 {
				(planar[1].buf()[idx] * 255.0).clamp(0.0, 255.0) as u8
			} else {
				r
			};
			let b = if planar.len() > 2 {
				(planar[2].buf()[idx] * 255.0).clamp(0.0, 255.0) as u8
			} else {
				r
			};
			rgb_data.push(r);
			rgb_data.push(g);
			rgb_data.push(b);
		}
	}

	let img_buffer = image::RgbImage::from_raw(width, height, rgb_data).ok_or_else(|| {
		SpatialError::ImageError("Failed to create image buffer from JXL data".to_string())
	})?;

	Ok(DynamicImage::ImageRgb8(img_buffer))
}

#[cfg(feature = "heic")]
fn load_heic_native(path: &Path) -> SpatialResult<DynamicImage> {
	use libheif_rs::{ColorSpace, HeifContext, LibHeif, RgbChroma};

	let lib_heif = LibHeif::new();
	let ctx = HeifContext::read_from_file(
		path.to_str()
			.ok_or_else(|| SpatialError::IoError("Invalid path encoding".to_string()))?,
	)
	.map_err(|e| SpatialError::ImageError(format!("Failed to load HEIC file: {:?}", e)))?;

	let handle = ctx.primary_image_handle().map_err(|e| {
		SpatialError::ImageError(format!("Failed to get HEIC image handle: {:?}", e))
	})?;

	let width = handle.width();
	let height = handle.height();

	let image = lib_heif
		.decode(&handle, ColorSpace::Rgb(RgbChroma::Rgb), None)
		.map_err(|e| SpatialError::ImageError(format!("HEIC decode failed: {:?}", e)))?;

	let planes = image.planes();
	let interleaved = planes.interleaved.ok_or_else(|| {
		SpatialError::ImageError("No interleaved plane in HEIC image".to_string())
	})?;

	let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);
	for y in 0..height {
		let row_start = (y * interleaved.stride as u32) as usize;
		let row_end = row_start + (width * 3) as usize;
		rgb_data.extend_from_slice(&interleaved.data[row_start..row_end]);
	}

	let img_buffer = image::RgbImage::from_raw(width, height, rgb_data).ok_or_else(|| {
		SpatialError::ImageError("Failed to create image buffer from HEIC data".to_string())
	})?;

	Ok(DynamicImage::ImageRgb8(img_buffer))
}

async fn load_with_ffmpeg(path: &Path, format: &str) -> SpatialResult<DynamicImage> {
	if !is_ffmpeg_available() {
		return Err(SpatialError::ImageError(format!(
			"{} format requires ffmpeg for conversion (not installed or not in PATH)",
			format.to_uppercase()
		)));
	}

	let temp_dir = std::env::temp_dir();
	let temp_filename = format!(
		"spatial_maker_convert_{}_{}.jpg",
		format,
		std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.unwrap_or_default()
			.as_millis()
	);
	let temp_path = temp_dir.join(temp_filename);

	let input_str = path
		.to_str()
		.ok_or_else(|| SpatialError::IoError("Invalid input path".to_string()))?;
	let output_str = temp_path
		.to_str()
		.ok_or_else(|| SpatialError::IoError("Invalid output path".to_string()))?;

	let output = Command::new("ffmpeg")
		.args(&["-i", input_str, "-q:v", "2", "-y", output_str])
		.output()
		.map_err(|e| SpatialError::IoError(format!("Failed to run ffmpeg: {}", e)))?;

	if !output.status.success() {
		let stderr = String::from_utf8_lossy(&output.stderr);
		return Err(SpatialError::ImageError(format!(
			"ffmpeg conversion failed for {} format:\n{}",
			format.to_uppercase(),
			stderr
		)));
	}

	let img = image::open(&temp_path).map_err(|e| {
		let _ = std::fs::remove_file(&temp_path);
		SpatialError::ImageError(format!("Failed to load converted image: {}", e))
	})?;

	let _ = std::fs::remove_file(&temp_path);

	Ok(img)
}

fn is_ffmpeg_available() -> bool {
	Command::new("ffmpeg")
		.arg("-version")
		.output()
		.map(|output| output.status.success())
		.unwrap_or(false)
}
