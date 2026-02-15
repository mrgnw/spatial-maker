use clap::Parser;
use spatial_maker::{
	process_photo, process_video, ImageEncoding, MVHEVCConfig, OutputFormat, OutputOptions,
	SpatialConfig, VideoProgress,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "spatial-maker")]
#[command(about = "Convert 2D images and videos to stereoscopic 3D spatial content")]
#[command(version)]
#[command(long_version = concat!(
	env!("CARGO_PKG_VERSION"),
	"\nRepository: ",
	env!("CARGO_PKG_REPOSITORY")
))]
struct Cli {
	/// Input image or video files
	#[arg(required = true)]
	inputs: Vec<PathBuf>,

	/// Output file (only valid with a single input)
	#[arg(short, long)]
	output: Option<PathBuf>,

	/// Model size: s (small, 48MB), b (base, 186MB), l (large, 638MB)
	#[arg(short, long, default_value = "s")]
	model: String,

	/// Maximum disparity in pixels (higher = more 3D depth)
	#[arg(long, default_value = "30")]
	max_disparity: u32,

	/// Output format for photos: sbs (side-by-side), tab (top-and-bottom), sep (separate L/R)
	#[arg(long, default_value = "sbs")]
	format: String,

	/// JPEG quality for photos (1-100)
	#[arg(long, default_value = "95")]
	quality: u8,

	/// Enable MV-HEVC packaging for photos (requires 'spatial' CLI in PATH)
	#[arg(long)]
	mvhevc: bool,
}

enum MediaType {
	Photo,
	Video,
}

fn detect_media_type(path: &PathBuf) -> MediaType {
	let ext = path
		.extension()
		.and_then(|s| s.to_str())
		.unwrap_or("")
		.to_lowercase();

	match ext.as_str() {
		"mp4" | "mov" | "avi" | "mkv" | "m4v" | "webm" | "flv" | "wmv" | "mpg" | "mpeg" => {
			MediaType::Video
		}
		_ => MediaType::Photo,
	}
}

fn generate_output_path(input: &PathBuf, media_type: &MediaType) -> PathBuf {
	let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
	let src_ext = input
		.extension()
		.and_then(|s| s.to_str())
		.unwrap_or("")
		.to_lowercase();

	let extension = match media_type {
		MediaType::Video => match src_ext.as_str() {
			"" => "mp4",
			other => other,
		},
		MediaType::Photo => match src_ext.as_str() {
			"heic" | "heif" | "avif" | "jxl" => "jpg",
			"" => "jpg",
			_ => &src_ext,
		},
	};

	let parent = input.parent().unwrap_or_else(|| std::path::Path::new("."));
	parent.join(format!("{}-spatial.{}", stem, extension))
}

async fn process_single(
	input: &PathBuf,
	output: PathBuf,
	config: SpatialConfig,
	cli: &Cli,
) -> Result<(), Box<dyn std::error::Error>> {
	let media_type = detect_media_type(input);

	match media_type {
		MediaType::Photo => {
			let layout = match cli.format.as_str() {
				"sbs" => OutputFormat::SideBySide,
				"tab" => OutputFormat::TopAndBottom,
				"sep" => OutputFormat::Separate,
				_ => {
					eprintln!("Invalid format '{}'. Use: sbs, tab, or sep", cli.format);
					std::process::exit(1);
				}
			};

			let output_options = OutputOptions {
				layout,
				image_format: ImageEncoding::Jpeg {
					quality: cli.quality,
				},
				mvhevc: if cli.mvhevc {
					Some(MVHEVCConfig {
						spatial_cli_path: None,
						enabled: true,
						quality: cli.quality,
						keep_intermediate: false,
					})
				} else {
					None
				},
			};

			eprintln!("Processing photo: {:?}", input);
			process_photo(input, &output, config, output_options).await?;
			eprintln!("Saved to: {:?}", output);
		}
		MediaType::Video => {
			eprintln!("Processing video: {:?}", input);
			let start = std::time::Instant::now();

			process_video(
				input,
				&output,
				config,
				Some(Box::new(|progress: VideoProgress| {
					eprint!(
						"\r[{}] Frame {}/{} ({:.1}%)",
						progress.stage, progress.current_frame, progress.total_frames, progress.percent
					);
				})),
			)
			.await?;

			eprintln!();
			eprintln!("Saved to: {:?}", output);
			eprintln!("Total time: {:.1}s", start.elapsed().as_secs_f64());
		}
	}

	Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let cli = Cli::parse();

	if cli.output.is_some() && cli.inputs.len() > 1 {
		eprintln!("--output cannot be used with multiple inputs");
		std::process::exit(1);
	}

	let config = SpatialConfig {
		encoder_size: cli.model.clone(),
		max_disparity: cli.max_disparity,
		target_depth_size: 518,
	};

	let total = cli.inputs.len();
	let mut errors = Vec::new();

	for (i, input) in cli.inputs.iter().enumerate() {
		if total > 1 {
			eprintln!("[{}/{}]", i + 1, total);
		}

		let media_type = detect_media_type(input);
		let output = cli
			.output
			.clone()
			.unwrap_or_else(|| generate_output_path(input, &media_type));

		if let Err(e) = process_single(input, output, config.clone(), &cli).await {
			eprintln!("Error processing {:?}: {}", input, e);
			errors.push((input.clone(), e));
		}
	}

	if !errors.is_empty() {
		eprintln!("\n{}/{} files failed", errors.len(), total);
		std::process::exit(1);
	}

	Ok(())
}
