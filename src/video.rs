use crate::depth_filter::DepthProcessor;
use crate::error::{SpatialError, SpatialResult};
use crate::output::{needs_depth, needs_stereo, OutputType};
use crate::stereo::{generate_stereo_pair, EdgeRefinement};
use crate::{NormalizeMode, SpatialConfig};
use image::{DynamicImage, RgbImage};
use ndarray::Array2;
use std::path::Path;
use std::process::Stdio;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{ChildStderr, Command};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

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

const STDERR_TAIL_BYTES: usize = 4096;

fn collect_stderr_tail(mut stderr: ChildStderr) -> JoinHandle<String> {
	tokio::spawn(async move {
		let mut tail = Vec::with_capacity(STDERR_TAIL_BYTES * 2);
		let mut chunk = [0u8; 4096];
		while let Ok(n) = stderr.read(&mut chunk).await {
			if n == 0 {
				break;
			}
			tail.extend_from_slice(&chunk[..n]);
			if tail.len() > STDERR_TAIL_BYTES {
				tail.drain(..tail.len() - STDERR_TAIL_BYTES);
			}
		}
		String::from_utf8_lossy(&tail).trim().to_string()
	})
}

async fn stderr_text(stderr: JoinHandle<String>) -> String {
	stderr.await.unwrap_or_default()
}

type FrameReader = (mpsc::Receiver<Vec<u8>>, JoinHandle<SpatialResult<u32>>);

async fn extract_frames(input_path: &Path, metadata: &VideoMetadata) -> SpatialResult<FrameReader> {
	let (tx, rx) = mpsc::channel::<Vec<u8>>(10);

	let frame_size = (metadata.width * metadata.height * 3) as usize;
	let vf_scale = format!("scale={}:{}", metadata.width, metadata.height);
	let input_str = input_path
		.to_str()
		.ok_or_else(|| SpatialError::IoError(format!("Invalid input path: {:?}", input_path)))?;

	let mut child = Command::new("ffmpeg")
		.args([
			"-v", "error",
			"-nostats",
			"-i", input_str,
			"-vf", &vf_scale,
			"-f", "rawvideo",
			"-pix_fmt", "rgb24",
			"-fps_mode", "passthrough",
			"-",
		])
		.stdin(Stdio::null())
		.stdout(Stdio::piped())
		.stderr(Stdio::piped())
		.spawn()
		.map_err(|e| SpatialError::Other(format!("Failed to spawn ffmpeg frame extractor: {}", e)))?;

	let stdout = child.stdout.take().expect("stdout is piped");
	let stderr = collect_stderr_tail(child.stderr.take().expect("stderr is piped"));

	let handle = tokio::spawn(async move {
		let mut reader = tokio::io::BufReader::new(stdout);
		let mut frame_buffer = vec![0u8; frame_size];
		let mut frames = 0u32;

		loop {
			match reader.read_exact(&mut frame_buffer).await {
				Ok(_) => {
					if tx.send(frame_buffer.clone()).await.is_err() {
						let _ = child.kill().await;
						return Ok(frames);
					}
					frames += 1;
				}
				Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
				Err(e) => {
					let _ = child.kill().await;
					return Err(SpatialError::IoError(format!("Failed to read ffmpeg frames: {}", e)));
				}
			}
		}

		let status = child
			.wait()
			.await
			.map_err(|e| SpatialError::Other(format!("ffmpeg frame extraction failed: {}", e)))?;
		if !status.success() {
			return Err(SpatialError::Other(format!(
				"ffmpeg frame extraction exited with {}: {}",
				status,
				stderr_text(stderr).await
			)));
		}
		if frames == 0 {
			return Err(SpatialError::Other(format!(
				"ffmpeg extracted no frames: {}",
				stderr_text(stderr).await
			)));
		}
		Ok(frames)
	});

	Ok((rx, handle))
}

async fn finish_extraction(handle: JoinHandle<SpatialResult<u32>>) -> SpatialResult<u32> {
	handle
		.await
		.map_err(|e| SpatialError::Other(format!("Frame extraction task failed: {}", e)))?
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

#[cfg(not(all(target_os = "macos", feature = "coreml")))]
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
		.stderr(Stdio::piped())
		.spawn()
		.map_err(|e| SpatialError::Other(format!("Failed to spawn ffmpeg encoder: {}", e)))?;

	let mut stdin = child.stdin.take().expect("Failed to capture stdin");
	let stderr = collect_stderr_tail(child.stderr.take().expect("stderr is piped"));
	let mut write_error = None;

	while let Some((left, right)) = rx.recv().await {
		let mut sbs_image = image::ImageBuffer::new(output_width, output_height);

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

		if let Err(e) = stdin.write_all(&sbs_image.into_raw()).await {
			write_error = Some(SpatialError::IoError(format!("Failed to write frame: {}", e)));
			break;
		}
	}

	drop(stdin);

	let status = child
		.wait()
		.await
		.map_err(|e| SpatialError::Other(format!("ffmpeg encoding failed: {}", e)))?;

	if !status.success() {
		return Err(SpatialError::Other(format!(
			"ffmpeg encoding exited with {}: {}",
			status,
			stderr_text(stderr).await
		)));
	}
	if let Some(e) = write_error {
		return Err(e);
	}

	Ok(())
}

async fn encode_depth_video(
	output_path: std::path::PathBuf,
	metadata: VideoMetadata,
	mut rx: mpsc::Receiver<Array2<f32>>,
) -> SpatialResult<()> {
	let width = metadata.width;
	let height = metadata.height;
	let fps = metadata.fps;

	let mut child = Command::new("ffmpeg")
		.args([
			"-f", "rawvideo",
			"-pix_fmt", "gray",
			"-s", &format!("{}x{}", width, height),
			"-r", &format!("{}", fps),
			"-i", "-",
			"-c:v", "libsvtav1",
			"-crf", "23",
			"-pix_fmt", "yuv420p",
			"-y",
			output_path.to_str().unwrap(),
		])
		.stdin(Stdio::piped())
		.stdout(Stdio::null())
		.stderr(Stdio::piped())
		.spawn()
		.map_err(|e| SpatialError::Other(format!("Failed to spawn ffmpeg depth encoder: {}", e)))?;

	let mut stdin = child.stdin.take().expect("Failed to capture stdin");
	let stderr = collect_stderr_tail(child.stderr.take().expect("stderr is piped"));
	let mut write_error = None;

	while let Some(depth) = rx.recv().await {
		let mut min_val = f32::INFINITY;
		let mut max_val = f32::NEG_INFINITY;
		for &v in depth.iter() {
			if v < min_val { min_val = v; }
			if v > max_val { max_val = v; }
		}
		let range = max_val - min_val;

		let pixels: Vec<u8> = depth.iter().map(|&v| {
			if range > 1e-6 {
				((v - min_val) / range * 255.0).round() as u8
			} else {
				128u8
			}
		}).collect();

		if let Err(e) = stdin.write_all(&pixels).await {
			write_error = Some(SpatialError::IoError(format!("Failed to write depth frame: {}", e)));
			break;
		}
	}

	drop(stdin);

	let status = child
		.wait()
		.await
		.map_err(|e| SpatialError::Other(format!("ffmpeg depth encoding failed: {}", e)))?;

	if !status.success() {
		return Err(SpatialError::Other(format!(
			"ffmpeg depth encoding exited with {}: {}",
			status,
			stderr_text(stderr).await
		)));
	}
	if let Some(e) = write_error {
		return Err(e);
	}

	Ok(())
}

#[cfg(all(target_os = "macos", feature = "coreml"))]
fn encode_spatial_video(
	output_path: std::path::PathBuf,
	input_path: std::path::PathBuf,
	metadata: VideoMetadata,
	max_disparity: u32,
	mut rx: mpsc::Receiver<(DynamicImage, DynamicImage)>,
) -> SpatialResult<()> {
	use crate::output::comfort_disparity_adjustment;
	use crate::spatial_video::{SpatialMetadata, SpatialVideoWriter};

	let audio_source = metadata.has_audio.then_some(input_path.as_path());
	let mut writer = SpatialVideoWriter::new(
		&output_path,
		audio_source,
		metadata.width,
		metadata.height,
		metadata.fps,
		&SpatialMetadata {
			disparity_adjustment: comfort_disparity_adjustment(max_disparity, metadata.width),
			..SpatialMetadata::default()
		},
	)?;
	while let Some((left, right)) = rx.blocking_recv() {
		writer.append(&left, &right)?;
	}
	writer.finish()
}

pub async fn process_video(
	input_path: &Path,
	output_path: &Path,
	config: SpatialConfig,
	output_types: &[OutputType],
	progress_cb: Option<ProgressCallback>,
	force: bool,
) -> SpatialResult<()> {
	if !input_path.exists() {
		return Err(SpatialError::IoError(format!(
			"Input file not found: {:?}",
			input_path
		)));
	}

	let do_depth = needs_depth(output_types);
	let do_stereo = needs_stereo(output_types);

	let mut metadata = get_video_metadata(input_path).await?;
	metadata.width = metadata.width & !1;
	metadata.height = metadata.height & !1;
	let stereo_output = {
		let stem = output_path.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
		let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
		parent.join(format!("{}-spatial.mov", stem))
	};

	crate::model::ensure_model_exists::<fn(u64, u64)>(&config.encoder_size, None).await?;

	#[cfg(all(target_os = "macos", feature = "coreml"))]
	let estimator = {
		let model_path = crate::model::find_model(&config.encoder_size)?;
		let model_str = model_path.to_str().ok_or_else(|| {
			SpatialError::ModelError("Invalid model path encoding".to_string())
		})?;
		let meta = crate::model::ModelMetadata::coreml(&config.encoder_size)?;
		std::sync::Arc::new(crate::depth_coreml::CoreMLDepthEstimator::new(
			model_str,
			meta.input_size,
			meta.inverted_depth,
		)?)
	};

	let mut depth_processor = DepthProcessor::new(
		config.temporal_alpha,
		config.bilateral_sigma_space,
		config.bilateral_sigma_color,
		config.depth_blur_sigma,
		config.normalize_mode.clone(),
	);

	let total_frames = metadata.total_frames;

	if matches!(config.normalize_mode, NormalizeMode::Global) {
		if let Some(ref cb) = progress_cb {
			cb(VideoProgress::new(0, total_frames, "scanning".to_string()));
		}

		let (mut scan_rx, scan_handle) = extract_frames(input_path, &metadata).await?;
		let mut scan_count = 0u32;
		while let Some(frame_data) = scan_rx.recv().await {
			let frame = frame_to_image(&frame_data, metadata.width, metadata.height)?;
			scan_count += 1;

			#[cfg(all(target_os = "macos", feature = "coreml"))]
			{
				let raw = estimator.estimate_unnormalized(&frame)?;
				depth_processor.update_global_range(&raw);
			}

			#[cfg(not(all(target_os = "macos", feature = "coreml")))]
			{
				let _ = frame;
			}

			if let Some(ref cb) = progress_cb {
				if scan_count % 10 == 0 || scan_count == total_frames {
					cb(VideoProgress::new(
						scan_count,
						total_frames,
						"scanning".to_string(),
					));
				}
			}
		}
		drop(scan_rx);
		finish_extraction(scan_handle).await?;
	}

	#[cfg(all(target_os = "macos", feature = "coreml"))]
	estimator.reset_stream();

	let (mut frame_rx, frame_handle) = extract_frames(input_path, &metadata).await?;

	let stereo_tx_opt;
	let stereo_handle;

	if do_stereo {
		let (tx, rx) = mpsc::channel::<(DynamicImage, DynamicImage)>(10);
		stereo_tx_opt = Some(tx);
		#[cfg(all(target_os = "macos", feature = "coreml"))]
		let handle = {
			let (output, input, metadata) = (stereo_output.clone(), input_path.to_path_buf(), metadata.clone());
			let max_disparity = config.max_disparity_px(metadata.width);
			tokio::task::spawn_blocking(move || encode_spatial_video(output, input, metadata, max_disparity, rx))
		};
		#[cfg(not(all(target_os = "macos", feature = "coreml")))]
		let handle = tokio::spawn(encode_stereo_video(stereo_output.clone(), metadata.clone(), rx));
		stereo_handle = Some(handle);
	} else {
		stereo_tx_opt = None;
		stereo_handle = None;
	}

	let depth_tx_opt;
	let depth_handle;

	let _encode_depth = if do_depth {
		let depth_path = {
			let stem = output_path.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
			let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
			parent.join(format!("{}-depth.mp4", stem))
		};

		if !force && depth_path.exists() {
			depth_tx_opt = None;
			depth_handle = None;
			false
		} else {
			let (tx, rx) = mpsc::channel::<Array2<f32>>(10);
			depth_tx_opt = Some(tx);
			depth_handle = Some(tokio::spawn(encode_depth_video(
				depth_path,
				metadata.clone(),
				rx,
			)));
			true
		}
	} else {
		depth_tx_opt = None;
		depth_handle = None;
		false
	};

	let mut frame_count = 0u32;

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
		let depth_map = {
			let raw = estimator.estimate_unnormalized(&frame)?;
			depth_processor.process(raw)
		};

		#[cfg(not(all(target_os = "macos", feature = "coreml")))]
		let depth_map = {
			#[cfg(feature = "onnx")]
			{
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

		if let Some(ref depth_tx) = depth_tx_opt {
			if depth_tx.send(depth_map.clone()).await.is_err() {
				break;
			}
		}

		if let Some(ref stereo_tx) = stereo_tx_opt {
			let (left, right) = generate_stereo_pair(&frame, &depth_map, config.max_disparity_px(frame.width()), EdgeRefinement::for_video())?;
			if stereo_tx.send((left, right)).await.is_err() {
				break;
			}
		}
	}

	drop(frame_rx);
	let extraction = finish_extraction(frame_handle).await;
	drop(stereo_tx_opt);
	drop(depth_tx_opt);
	extraction?;

	if let Some(ref cb) = progress_cb {
		cb(VideoProgress::new(
			total_frames,
			total_frames,
			"encoding".to_string(),
		));
	}

	if let Some(handle) = stereo_handle {
		handle
			.await
			.map_err(|e| SpatialError::Other(format!("Stereo encoding task failed: {}", e)))??;
	}

	if let Some(handle) = depth_handle {
		handle
			.await
			.map_err(|e| SpatialError::Other(format!("Depth encoding task failed: {}", e)))??;
	}


	if let Some(ref cb) = progress_cb {
		cb(VideoProgress::new(
			total_frames,
			total_frames,
			"complete".to_string(),
		));
	}

	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[tokio::test]
	async fn extract_frames_reports_ffmpeg_failure() {
		let dir = tempfile::tempdir().unwrap();
		let input = dir.path().join("not-a-video.mp4");
		std::fs::write(&input, b"not a video").unwrap();
		let metadata = VideoMetadata {
			width: 16,
			height: 16,
			fps: 30.0,
			total_frames: 1,
			duration: 1.0,
			has_audio: false,
		};

		let (mut rx, handle) = extract_frames(&input, &metadata).await.unwrap();
		assert!(rx.recv().await.is_none());
		let err = finish_extraction(handle).await.unwrap_err().to_string();
		assert!(err.contains("ffmpeg frame extraction exited"), "{}", err);
		assert!(err.contains("not-a-video.mp4"), "{}", err);
	}
}
