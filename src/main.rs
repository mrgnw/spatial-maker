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
	/// Input image or video file
	input: PathBuf,

	/// Output file (defaults to input path with -spatial suffix)
	#[arg(short, long)]
	output: Option<PathBuf>,

	/// Model size: s (small, 48MB), b (base, 186MB), l (large, 638GB)
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
	let extension = input.extension().and_then(|s| s.to_str()).unwrap_or(match media_type {
		MediaType::Video => "mp4",
		MediaType::Photo => "jpg",
	});

	let parent = input.parent().unwrap_or_else(|| std::path::Path::new("."));
	parent.join(format!("{}-spatial.{}", stem, extension))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let cli = Cli::parse();

	let media_type = detect_media_type(&cli.input);
	let output = cli
		.output
		.unwrap_or_else(|| generate_output_path(&cli.input, &media_type));

	let config = SpatialConfig {
		encoder_size: cli.model.clone(),
		max_disparity: cli.max_disparity,
		target_depth_size: 518,
	};

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

			eprintln!("Processing photo: {:?}", cli.input);
			eprintln!(
				"Model: {}, Max disparity: {}, Format: {}",
				config.encoder_size, cli.max_disparity, cli.format
			);

			process_photo(&cli.input, &output, config, output_options).await?;

			eprintln!("✓ Saved to: {:?}", output);
		}
		MediaType::Video => {
			eprintln!("Processing video: {:?}", cli.input);
			eprintln!(
				"Model: {}, Max disparity: {}",
				config.encoder_size, cli.max_disparity
			);

			let start = std::time::Instant::now();

			process_video(
				&cli.input,
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
			eprintln!("✓ Saved to: {:?}", output);
			eprintln!("Total time: {:.1}s", start.elapsed().as_secs_f64());
		}
	}

	Ok(())
}
