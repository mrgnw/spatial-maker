use crate::error::{SpatialError, SpatialResult};
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

pub fn get_checkpoint_dir() -> SpatialResult<PathBuf> {
	if let Ok(custom_dir) = std::env::var("SPATIAL_MAKER_CHECKPOINTS") {
		Ok(PathBuf::from(custom_dir))
	} else {
		let home = dirs::home_dir().ok_or_else(|| {
			SpatialError::ConfigError("Could not determine home directory".to_string())
		})?;
		Ok(home.join(".spatial-maker").join("checkpoints"))
	}
}

#[derive(Clone, Debug)]
pub struct ModelMetadata {
	pub name: String,
	pub filename: String,
	pub url: String,
	pub size_mb: u32,
}

impl ModelMetadata {
	pub fn coreml(encoder_size: &str) -> SpatialResult<Self> {
		match encoder_size {
			"s" | "small" => Ok(ModelMetadata {
				name: "depth-anything-v2-small".to_string(),
				filename: "DepthAnythingV2SmallF16.mlpackage".to_string(),
				url: "https://huggingface.co/mrgnw/depth-anything-v2-coreml/resolve/main/DepthAnythingV2SmallF16.mlpackage.tar.gz".to_string(),
				size_mb: 48,
			}),
			"b" | "base" => Ok(ModelMetadata {
				name: "depth-anything-v2-base".to_string(),
				filename: "DepthAnythingV2BaseF16.mlpackage".to_string(),
				url: "https://huggingface.co/mrgnw/depth-anything-v2-coreml/resolve/main/DepthAnythingV2BaseF16.mlpackage.tar.gz".to_string(),
				size_mb: 186,
			}),
			"l" | "large" => Ok(ModelMetadata {
				name: "depth-anything-v2-large".to_string(),
				filename: "DepthAnythingV2LargeF16.mlpackage".to_string(),
				url: "https://huggingface.co/mrgnw/depth-anything-v2-coreml/resolve/main/DepthAnythingV2LargeF16.mlpackage.tar.gz".to_string(),
				size_mb: 638,
			}),
			other => Err(SpatialError::ConfigError(
				format!("Unknown encoder size: '{}'. Use 's', 'b', or 'l'", other)
			)),
		}
	}

	#[cfg(feature = "onnx")]
	pub fn onnx(encoder_size: &str) -> SpatialResult<Self> {
		match encoder_size {
			"s" | "small" => Ok(ModelMetadata {
				name: "depth-anything-v2-small".to_string(),
				filename: "depth_anything_v2_small.onnx".to_string(),
				url: "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx".to_string(),
				size_mb: 99,
			}),
			"b" | "base" => Ok(ModelMetadata {
				name: "depth-anything-v2-base".to_string(),
				filename: "depth_anything_v2_base.onnx".to_string(),
				url: "https://huggingface.co/onnx-community/depth-anything-v2-base/resolve/main/onnx/model.onnx".to_string(),
				size_mb: 380,
			}),
			"l" | "large" => Ok(ModelMetadata {
				name: "depth-anything-v2-large".to_string(),
				filename: "depth_anything_v2_large.onnx".to_string(),
				url: "https://huggingface.co/onnx-community/depth-anything-v2-large/resolve/main/onnx/model.onnx".to_string(),
				size_mb: 1300,
			}),
			other => Err(SpatialError::ConfigError(
				format!("Unknown encoder size: '{}'. Use 's', 'b', or 'l'", other)
			)),
		}
	}
}

pub fn find_model(encoder_size: &str) -> SpatialResult<PathBuf> {
	let checkpoint_dir = get_checkpoint_dir()?;

	#[cfg(all(target_os = "macos", feature = "coreml"))]
	{
		let meta = ModelMetadata::coreml(encoder_size)?;
		let model_path = checkpoint_dir.join(&meta.filename);
		if model_path.exists() {
			return Ok(model_path);
		}
	}

	#[cfg(feature = "onnx")]
	{
		let meta = ModelMetadata::onnx(encoder_size)?;
		let model_path = checkpoint_dir.join(&meta.filename);
		if model_path.exists() {
			return Ok(model_path);
		}
	}

	// Also check development paths
	let dev_paths = [
		PathBuf::from("checkpoints"),
		dirs::home_dir()
			.unwrap_or_default()
			.join(".spatial-maker")
			.join("checkpoints"),
	];

	for dir in &dev_paths {
		if dir.exists() {
			if let Ok(entries) = std::fs::read_dir(dir) {
				for entry in entries.flatten() {
					let name = entry.file_name().to_string_lossy().to_string();
					if name.contains("DepthAnything") || name.contains("depth_anything") {
						let lower_size = encoder_size.to_lowercase();
						let name_lower = name.to_lowercase();
						let matches = match lower_size.as_str() {
							"s" | "small" => name_lower.contains("small"),
							"b" | "base" => name_lower.contains("base"),
							"l" | "large" => name_lower.contains("large"),
							_ => false,
						};
						if matches {
							return Ok(entry.path());
						}
					}
				}
			}
		}
	}

	Err(SpatialError::ModelError(format!(
		"Model not found for encoder size '{}'. Run download first.",
		encoder_size
	)))
}

pub fn model_exists(encoder_size: &str) -> bool {
	find_model(encoder_size).is_ok()
}

pub async fn ensure_model_exists<F>(
	encoder_size: &str,
	progress_fn: Option<F>,
) -> SpatialResult<PathBuf>
where
	F: FnMut(u64, u64),
{
	if let Ok(path) = find_model(encoder_size) {
		return Ok(path);
	}

	let checkpoint_dir = get_checkpoint_dir()?;
	tokio::fs::create_dir_all(&checkpoint_dir)
		.await
		.map_err(|e| {
			SpatialError::IoError(format!("Failed to create checkpoint directory: {}", e))
		})?;

	#[cfg(all(target_os = "macos", feature = "coreml"))]
	{
		let meta = ModelMetadata::coreml(encoder_size)?;
		let model_path = checkpoint_dir.join(&meta.filename);
		download_model(&meta, &model_path, progress_fn).await?;
		return Ok(model_path);
	}

	#[cfg(all(feature = "onnx", not(all(target_os = "macos", feature = "coreml"))))]
	{
		let meta = ModelMetadata::onnx(encoder_size)?;
		let model_path = checkpoint_dir.join(&meta.filename);
		download_model(&meta, &model_path, progress_fn).await?;
		return Ok(model_path);
	}

	#[cfg(not(any(all(target_os = "macos", feature = "coreml"), feature = "onnx")))]
	{
		let _ = progress_fn;
		Err(SpatialError::ConfigError(
			"No depth backend enabled. Enable 'coreml' (macOS) or 'onnx' feature.".to_string(),
		))
	}
}

async fn download_model<F>(
	metadata: &ModelMetadata,
	destination: &Path,
	mut progress_fn: Option<F>,
) -> SpatialResult<()>
where
	F: FnMut(u64, u64),
{
	tracing::info!("Downloading model: {} from {}", metadata.name, metadata.url);

	let response = reqwest::get(&metadata.url)
		.await
		.map_err(|e| SpatialError::Other(format!("Failed to download model: {}", e)))?;

	let total_bytes = response
		.content_length()
		.unwrap_or(metadata.size_mb as u64 * 1_000_000);

	let is_tar_gz = metadata.url.ends_with(".tar.gz");

	if is_tar_gz {
		let temp_path = destination.with_extension("tar.gz");
		let mut file = tokio::fs::File::create(&temp_path)
			.await
			.map_err(|e| SpatialError::IoError(format!("Failed to create file: {}", e)))?;

		let mut downloaded = 0u64;
		let mut stream = response.bytes_stream();
		use futures_util::StreamExt;

		while let Some(chunk) = stream.next().await {
			let chunk = chunk.map_err(|e| SpatialError::Other(format!("Download interrupted: {}", e)))?;
			file.write_all(&chunk)
				.await
				.map_err(|e| SpatialError::IoError(format!("Failed to write to file: {}", e)))?;
			downloaded += chunk.len() as u64;
			if let Some(ref mut f) = progress_fn {
				f(downloaded, total_bytes);
			}
		}
		drop(file);

		let parent = destination
			.parent()
			.ok_or_else(|| SpatialError::IoError("Invalid destination path".to_string()))?;

		let output = std::process::Command::new("tar")
			.args(&["xzf"])
			.arg(&temp_path)
			.arg("-C")
			.arg(parent)
			.output()
			.map_err(|e| SpatialError::IoError(format!("Failed to extract tar.gz: {}", e)))?;

		if !output.status.success() {
			let stderr = String::from_utf8_lossy(&output.stderr);
			return Err(SpatialError::IoError(format!("tar extraction failed: {}", stderr)));
		}

		let _ = tokio::fs::remove_file(&temp_path).await;
	} else {
		let mut file = tokio::fs::File::create(destination)
			.await
			.map_err(|e| SpatialError::IoError(format!("Failed to create file: {}", e)))?;

		let mut downloaded = 0u64;
		let mut stream = response.bytes_stream();
		use futures_util::StreamExt;

		while let Some(chunk) = stream.next().await {
			let chunk = chunk.map_err(|e| SpatialError::Other(format!("Download interrupted: {}", e)))?;
			file.write_all(&chunk)
				.await
				.map_err(|e| SpatialError::IoError(format!("Failed to write to file: {}", e)))?;
			downloaded += chunk.len() as u64;
			if let Some(ref mut f) = progress_fn {
				f(downloaded, total_bytes);
			}
		}
	}

	tracing::info!("Model downloaded: {:?}", destination);
	Ok(())
}
