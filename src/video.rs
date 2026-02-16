use crate::error::{SpatialError, SpatialResult};
use crate::stereo::generate_stereo_pair;
use crate::SpatialConfig;
use image::{DynamicImage, ImageBuffer, RgbImage};
use std::path::Path;
use std::process::Stdio;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::sync::mpsc;

#[derive(Clone, Debug)]
pub struct VideoProgress {
	pub current_frame: u32,
	pub total_frames: u32,
	pub stage: String,
	pub percent: f64,
}

impl VideoProgress {
	pub fn new(current_frame: u32, total_frames: u32, stage: String) -> Self {
		let percent = if total_frames > 0 {
			(current_frame as f64 / total_frames as f64 * 100.0).min(100.0)
		} else {
			0.0
		};
		Self {
			current_frame,
			total_frames,
			stage,
			percent,
		}
	}
}

#[derive(Clone, Debug)]
pub struct VideoMetadata {
	pub width: u32,
	pub height: u32,
	pub fps: f64,
	pub total_frames: u32,
	pub duration: f64,
	pub has_audio: bool,
}

pub type ProgressCallback = Box<dyn Fn(VideoProgress) + Send + Sync>;

pub async fn get_video_metadata(input_path: &Path) -> SpatialResult<VideoMetadata> {
	let input_str = input_path
		.to_str()
		.ok_or_else(|| SpatialError::Other("Invalid input path encoding".to_string()))?;

	let output = Command::new("ffprobe")
		.args([
			"-v", "error",
			"-select_streams", "v:0",
			"-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
			"-show_entries", "format=duration",
			"-of", "json",
			input_str,
		])
		.output()
		.await
		.map_err(|e| {
			SpatialError::Other(format!(
				"Failed to run ffprobe (is ffmpeg installed?): {}",
				e
			))
		})?;

	if !output.status.success() {
		let stderr = String::from_utf8_lossy(&output.stderr);
		return Err(SpatialError::Other(format!("ffprobe failed: {}", stderr)));
	}

	let stdout = String::from_utf8_lossy(&output.stdout);
	let json: serde_json::Value = serde_json::from_str(&stdout)
		.map_err(|e| SpatialError::Other(format!("Failed to parse ffprobe JSON: {}", e)))?;

	let stream = json["streams"]
		.as_array()
		.and_then(|s| s.first())
		.ok_or_else(|| SpatialError::Other("No video stream found".to_string()))?;

	let width = stream["width"]
		.as_u64()
		.ok_or_else(|| SpatialError::Other("Failed to parse width".to_string()))? as u32;
	let height = stream["height"]
		.as_u64()
		.ok_or_else(|| SpatialError::Other("Failed to parse height".to_string()))? as u32;

	let fps = stream["r_frame_rate"]
		.as_str()
		.map(|s| {
			if let Some((num, den)) = s.split_once('/') {
				let n: f64 = num.parse().unwrap_or(30.0);
				let d: f64 = den.parse().unwrap_or(1.0);
				n / d
			} else {
				s.parse().unwrap_or(30.0)
			}
		})
		.unwrap_or(30.0);

	let duration = stream["duration"]
		.as_str()
		.and_then(|s| s.parse::<f64>().ok())
		.or_else(|| {
			json["format"]["duration"]
				.as_str()
				.and_then(|s| s.parse::<f64>().ok())
		})
		.unwrap_or(0.0);

	let total_frames = stream["nb_frames"]
		.as_str()
		.and_then(|s| s.parse::<u32>().ok())
		.unwrap_or_else(|| (duration * fps).round() as u32);

	let audio_output = Command::new("ffprobe")
		.args([
			"-v", "error",
			"-select_streams", "a:0",
			"-show_entries", "stream=codec_type",
			"-of", "csv=p=0",
			input_str,
		])
		.output()
		.await
		.map_err(|e| SpatialError::Other(format!("Failed to check audio: {}", e)))?;

	let has_audio = String::from_utf8_lossy(&audio_output.stdout)
		.trim()
		.contains("audio");

	Ok(VideoMetadata {
		width,
		height,
		fps,
		total_frames,
		duration,
		has_audio,
	})
}

async fn extract_frames(
	input_path: &Path,
	metadata: &VideoMetadata,
) -> SpatialResult<mpsc::Receiver<Vec<u8>>> {
	let (tx, rx) = mpsc::channel::<Vec<u8>>(10);

	let width = metadata.width;
	let height = metadata.height;
	let frame_size = (width * height * 3) as usize;

	let input_path = input_path.to_path_buf();

	tokio::spawn(async move {
		let mut child = Command::new("ffmpeg")
			.args([
				"-i",
				input_path.to_str().unwrap(),
				"-f",
				"rawvideo",
				"-pix_fmt",
				"rgb24",
				"-vsync",
				"0",
				"-",
			])
			.stdout(Stdio::piped())
			.stderr(Stdio::null())
			.spawn()
			.expect("Failed to spawn ffmpeg");

		let stdout = child.stdout.take().expect("Failed to capture stdout");
		let mut reader = tokio::io::BufReader::new(stdout);
		let mut frame_buffer = vec![0u8; frame_size];

		loop {
			match reader.read_exact(&mut frame_buffer).await {
				Ok(_) => {
					if tx.send(frame_buffer.clone()).await.is_err() {
						break;
					}
				}
				Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
				Err(_) => break,
			}
		}

		let _ = child.wait().await;
	});

	Ok(rx)
}

fn frame_to_image(data: &[u8], width: u32, height: u32) -> SpatialResult<DynamicImage> {
	let rgb_image = RgbImage::from_raw(width, height, data.to_vec()).ok_or_else(|| {
		SpatialError::ImageError(format!(
			"Failed to create image from frame data ({}x{})",
			width, height
		))
	})?;
	Ok(DynamicImage::ImageRgb8(rgb_image))
}

async fn encode_stereo_video(
	output_path: std::path::PathBuf,
	metadata: VideoMetadata,
	mut rx: mpsc::Receiver<(DynamicImage, DynamicImage)>,
) -> SpatialResult<()> {
	let width = metadata.width;
	let height = metadata.height;
	let fps = metadata.fps;

	let output_width = width * 2;
	let output_height = height;

	let mut child = Command::new("ffmpeg")
		.args([
			"-f",
			"rawvideo",
			"-pix_fmt",
			"rgb24",
			"-s",
			&format!("{}x{}", output_width, output_height),
			"-r",
			&format!("{}", fps),
			"-i",
			"-",
			"-c:v",
			"libx264",
			"-preset",
			"medium",
			"-crf",
			"23",
			"-pix_fmt",
			"yuv420p",
			"-y",
			output_path.to_str().unwrap(),
		])
		.stdin(Stdio::piped())
		.stdout(Stdio::null())
		.stderr(Stdio::null())
		.spawn()
		.map_err(|e| SpatialError::Other(format!("Failed to spawn ffmpeg encoder: {}", e)))?;

	let mut stdin = child.stdin.take().expect("Failed to capture stdin");

	while let Some((left, right)) = rx.recv().await {
		let mut sbs_image = ImageBuffer::new(output_width, output_height);

		let left_rgb = left.to_rgb8();
		for y in 0..height {
			for x in 0..width {
				let pixel = left_rgb.get_pixel(x, y);
				sbs_image.put_pixel(x, y, *pixel);
			}
		}

		let right_rgb = right.to_rgb8();
		for y in 0..height {
			for x in 0..width {
				let pixel = right_rgb.get_pixel(x, y);
				sbs_image.put_pixel(width + x, y, *pixel);
			}
		}

		stdin
			.write_all(&sbs_image.into_raw())
			.await
			.map_err(|e| SpatialError::IoError(format!("Failed to write frame: {}", e)))?;
	}

	drop(stdin);

	let status = child
		.wait()
		.await
		.map_err(|e| SpatialError::Other(format!("ffmpeg encoding failed: {}", e)))?;

	if !status.success() {
		return Err(SpatialError::Other(
			"ffmpeg encoding exited with error".to_string(),
		));
	}

	Ok(())
}

pub async fn process_video(
	input_path: &Path,
	output_path: &Path,
	config: SpatialConfig,
	progress_cb: Option<ProgressCallback>,
) -> SpatialResult<()> {
	if !input_path.exists() {
		return Err(SpatialError::IoError(format!(
			"Input file not found: {:?}",
			input_path
		)));
	}

	let metadata = get_video_metadata(input_path).await?;

	crate::model::ensure_model_exists::<fn(u64, u64)>(&config.encoder_size, None).await?;

	#[cfg(all(target_os = "macos", feature = "coreml"))]
	let estimator = {
		let model_path = crate::model::find_model(&config.encoder_size)?;
		let model_str = model_path.to_str().ok_or_else(|| {
			SpatialError::ModelError("Invalid model path encoding".to_string())
		})?;
		std::sync::Arc::new(crate::depth_coreml::CoreMLDepthEstimator::new(model_str)?)
	};

	let mut frame_rx = extract_frames(input_path, &metadata).await?;

	let (processed_tx, processed_rx) = mpsc::channel::<(DynamicImage, DynamicImage)>(10);

	let encode_handle = tokio::spawn(encode_stereo_video(
		output_path.to_path_buf(),
		metadata.clone(),
		processed_rx,
	));

	let mut frame_count = 0u32;
	let total_frames = metadata.total_frames;

	if let Some(ref cb) = progress_cb {
		cb(VideoProgress::new(0, total_frames, "extracting".to_string()));
	}

	while let Some(frame_data) = frame_rx.recv().await {
		let frame = frame_to_image(&frame_data, metadata.width, metadata.height)?;

		frame_count += 1;
		if let Some(ref cb) = progress_cb {
			if frame_count % 10 == 0 || frame_count == total_frames {
				cb(VideoProgress::new(
					frame_count,
					total_frames,
					"processing".to_string(),
				));
			}
		}

		#[cfg(all(target_os = "macos", feature = "coreml"))]
		let depth_map = estimator.estimate(&frame)?;

		#[cfg(not(all(target_os = "macos", feature = "coreml")))]
		let depth_map = {
			#[cfg(feature = "onnx")]
			{
				// For ONNX, we'd need to cache the estimator too
				// For now, this is a placeholder - ONNX video is not the primary path
				let model_path = crate::model::find_model(&config.encoder_size)?;
				let est = crate::depth::OnnxDepthEstimator::new(model_path.to_str().unwrap())?;
				est.estimate(&frame)?
			}
			#[cfg(not(feature = "onnx"))]
			{
				return Err(SpatialError::ConfigError(
					"No depth backend enabled. Enable 'coreml' or 'onnx' feature.".to_string(),
				));
			}
		};

		let (left, right) = generate_stereo_pair(&frame, &depth_map, config.max_disparity)?;

		if processed_tx.send((left, right)).await.is_err() {
			return Err(SpatialError::Other(
				"Encoder stopped unexpectedly".to_string(),
			));
		}
	}

	drop(processed_tx);

	if let Some(ref cb) = progress_cb {
		cb(VideoProgress::new(
			total_frames,
			total_frames,
			"encoding".to_string(),
		));
	}

	encode_handle
		.await
		.map_err(|e| SpatialError::Other(format!("Encoding task failed: {}", e)))??;

	if let Some(ref cb) = progress_cb {
		cb(VideoProgress::new(
			total_frames,
			total_frames,
			"complete".to_string(),
		));
	}

	Ok(())
}
