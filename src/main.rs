use clap::{Parser, Subcommand};
use spatial_maker::{
	process_photo, process_video, ImageEncoding, MVHEVCConfig, OutputFormat, OutputOptions,
	SpatialConfig, VideoProgress,
};
use std::path::PathBuf;

fn generate_output_path(input: &PathBuf, media_type: &str) -> PathBuf {
	let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
	let extension = input.extension().and_then(|s| s.to_str()).unwrap_or(
		if media_type == "video" { "mp4" } else { "jpg" }
	);
	
	let parent = input.parent().unwrap_or_else(|| std::path::Path::new("."));
	parent.join(format!("{}-spatial.{}", stem, extension))
}

#[derive(Parser)]
#[command(name = "spatial-maker")]
#[command(about = "Convert 2D images and videos to stereoscopic 3D spatial content")]
#[command(version)]
struct Cli {
	#[command(subcommand)]
	command: Commands,
}

#[derive(Subcommand)]
enum Commands {
	/// Convert a photo to stereoscopic 3D
	Photo {
		/// Input image file
		input: PathBuf,

		/// Output image file (defaults to input path with -spatial suffix)
		#[arg(short, long)]
		output: Option<PathBuf>,

		/// Model size: s (small, 48MB), b (base, 186MB), l (large, 638MB)
		#[arg(short, long, default_value = "s")]
		model: String,

		/// Maximum disparity in pixels (higher = more 3D depth)
		#[arg(long, default_value = "30")]
		max_disparity: u32,

		/// Output format: sbs (side-by-side), tab (top-and-bottom), sep (separate L/R files)
		#[arg(long, default_value = "sbs")]
		format: String,

		/// JPEG quality (1-100)
		#[arg(long, default_value = "95")]
		quality: u8,

		/// Enable MV-HEVC packaging (requires 'spatial' CLI in PATH)
		#[arg(long)]
		mvhevc: bool,
	},
	/// Convert a video to stereoscopic 3D
	Video {
		/// Input video file
		input: PathBuf,

		/// Output video file (defaults to input path with -spatial suffix)
		#[arg(short, long)]
		output: Option<PathBuf>,

		/// Model size: s (small, 48MB), b (base, 186MB), l (large, 638MB)
		#[arg(short, long, default_value = "s")]
		model: String,

		/// Maximum disparity in pixels (higher = more 3D depth)
		#[arg(long, default_value = "30")]
		max_disparity: u32,
	},
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let cli = Cli::parse();

	match cli.command {
		Commands::Photo {
			input,
			output,
			model,
			max_disparity,
			format,
			quality,
			mvhevc,
		} => {
			let output = output.unwrap_or_else(|| generate_output_path(&input, "photo"));

			let config = SpatialConfig {
				encoder_size: model,
				max_disparity,
				target_depth_size: 518,
			};

			let layout = match format.as_str() {
				"sbs" => OutputFormat::SideBySide,
				"tab" => OutputFormat::TopAndBottom,
				"sep" => OutputFormat::Separate,
				_ => {
					eprintln!("Invalid format '{}'. Use: sbs, tab, or sep", format);
					std::process::exit(1);
				}
			};

			let output_options = OutputOptions {
				layout,
				image_format: ImageEncoding::Jpeg { quality },
				mvhevc: if mvhevc {
					Some(MVHEVCConfig {
						spatial_cli_path: None,
						enabled: true,
						quality,
						keep_intermediate: false,
					})
				} else {
					None
				},
			};

			eprintln!("Processing photo: {:?}", input);
			eprintln!("Model: {}, Max disparity: {}, Format: {}", config.encoder_size, max_disparity, format);

			process_photo(&input, &output, config, output_options).await?;

			eprintln!("✓ Saved to: {:?}", output);
		}
		Commands::Video {
			input,
			output,
			model,
			max_disparity,
		} => {
			let output = output.unwrap_or_else(|| generate_output_path(&input, "video"));

			let config = SpatialConfig {
				encoder_size: model,
				max_disparity,
				target_depth_size: 518,
			};

			eprintln!("Processing video: {:?}", input);
			eprintln!("Model: {}, Max disparity: {}", config.encoder_size, max_disparity);

			let start = std::time::Instant::now();

			process_video(
				&input,
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
