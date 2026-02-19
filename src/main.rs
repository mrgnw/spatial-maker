use clap::{Parser, Subcommand};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use spatial_maker::{
	process_photo, process_video, ImageEncoding, MVHEVCConfig, NormalizeMode, OutputFormat,
	OutputOptions, OutputType, SpatialConfig, VideoProgress,
	needs_stereo, parse_output_types,
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
	#[command(subcommand)]
	command: Option<Commands>,

	/// Input image or video files
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

	/// Output types (comma-separated): depth, depth:avif,png,png16, sbs, tab, sep, spatial
	#[arg(long, default_value = "spatial")]
	output_types: String,

	/// JPEG quality for photos (1-100)
	#[arg(long, default_value = "95")]
	quality: u8,

	/// Temporal EMA blend factor for video depth (0=off, 1=no smoothing, default 0.7)
	#[arg(long, default_value = "0.7")]
	temporal_alpha: f32,

	/// Bilateral filter spatial sigma (0=off, default 5.0)
	#[arg(long, default_value = "5.0")]
	bilateral_sigma: f32,

	/// Bilateral filter range sigma (default 0.1)
	#[arg(long, default_value = "0.1")]
	bilateral_range: f32,

	/// Gaussian blur sigma for depth edge softening (0=off, default 1.5)
	#[arg(long, default_value = "1.5")]
	depth_blur: f32,

	/// Depth normalization mode for video: running (default), per-frame, global (two-pass)
	#[arg(long, default_value = "running")]
	normalize: String,
}

#[derive(Subcommand)]
enum Commands {
	/// Update spatial-maker to the latest release
	#[command(name = "self")]
	Self_ {
		#[command(subcommand)]
		action: SelfAction,
	},
}

#[derive(Subcommand)]
enum SelfAction {
	/// Download and install the latest release
	Update,
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

fn generate_output_base(input: &PathBuf, model: &str) -> PathBuf {
	let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
	let parent = input.parent().unwrap_or_else(|| std::path::Path::new("."));
	parent.join(format!("{}-{}", stem, model))
}



fn model_display_name(encoder_size: &str) -> (&str, u32) {
	match encoder_size {
		"s" | "small" => ("small", 48),
		"b" | "base" => ("base", 186),
		"l" | "large" => ("large", 638),
		_ => (encoder_size, 0),
	}
}

async fn process_single(
	input: &PathBuf,
	output: PathBuf,
	config: SpatialConfig,
	output_types: &[OutputType],
	cli: &Cli,
) -> Result<(), Box<dyn std::error::Error>> {
	let media_type = detect_media_type(input);

	match media_type {
		MediaType::Photo => {
			let has_stereo = needs_stereo(output_types);

			let layout = if has_stereo {
				let stereo = spatial_maker::stereo_types(output_types);
				match stereo.first() {
					Some(OutputType::TopAndBottom) => OutputFormat::TopAndBottom,
					Some(OutputType::Separate) => OutputFormat::Separate,
					_ => OutputFormat::SideBySide,
				}
			} else {
				OutputFormat::SideBySide
			};

			let has_spatial = output_types.iter().any(|t| matches!(t, OutputType::Spatial));

			let output_options = OutputOptions {
				layout,
				image_format: ImageEncoding::Jpeg {
					quality: cli.quality,
				},
				mvhevc: if has_spatial {
					Some(MVHEVCConfig {
						spatial_cli_path: None,
						enabled: true,
						quality: cli.quality,
						keep_intermediate: has_stereo && output_types.iter().any(|t| matches!(t, OutputType::SideBySide | OutputType::TopAndBottom | OutputType::Separate)),
					})
				} else {
					None
				},
			};

			let filename = input.file_name().and_then(|s| s.to_str()).unwrap_or("?");
			eprintln!("{} {}", style("🖼").cyan(), style(filename).bold());

			let spinner = ProgressBar::new_spinner();
			spinner.set_style(
				ProgressStyle::default_spinner()
					.template("{spinner:.cyan} {msg}")
					.unwrap()
			);
			spinner.set_message("Processing...");
			spinner.enable_steady_tick(std::time::Duration::from_millis(80));

			let result = process_photo(input, &output, config.clone(), output_types, output_options).await?;

			spinner.finish_and_clear();
			let (model_name, model_mb) = model_display_name(&cli.model);
			eprintln!(
				"{} {} / {} MB / depth-anything-v2-{}",
				style("✔").green().bold(),
				style("done").green(),
				model_mb,
				model_name,
			);

			for path in &result.depth_paths {
				let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
				eprintln!("{} {}", style("→").dim(), style(name).dim());
			}
			for path in &result.stereo_paths {
				let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
				eprintln!("{} {}", style("→").dim(), style(name).dim());
			}
		}
		MediaType::Video => {
			let filename = input.file_name().and_then(|s| s.to_str()).unwrap_or("?");
			eprintln!("{} {}", style("🎥").cyan(), style(filename).bold());

			let (model_name, model_mb) = model_display_name(&cli.model);
			let model_info = format!("model loaded / {} MB / depth-anything-v2-{}", model_mb, model_name);

			let spinner = ProgressBar::new_spinner();
			spinner.set_style(
				ProgressStyle::default_spinner()
					.template("{spinner:.cyan} {msg}")
					.unwrap()
			);
			spinner.set_message("Loading model...");
			spinner.enable_steady_tick(std::time::Duration::from_millis(80));

			let start = std::time::Instant::now();

			let pb = ProgressBar::new(100);
			pb.set_style(
				ProgressStyle::default_bar()
					.template("{msg} {bar:20.cyan/blue} {pos:>3}% {prefix}")
					.unwrap()
					.progress_chars("━╸─")
			);

			let pb_inner = pb.clone();
			let start_clone = start.clone();
			let model_loaded = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
			let model_loaded_clone = model_loaded.clone();
			let spinner_clone = spinner.clone();

			process_video(
				input,
				&output,
				config,
				output_types,
				Some(Box::new(move |progress: VideoProgress| {
					if !model_loaded_clone.load(std::sync::atomic::Ordering::Relaxed) {
						model_loaded_clone.store(true, std::sync::atomic::Ordering::Relaxed);
						spinner_clone.finish_and_clear();
						eprintln!(
							"{} {}",
							style("✔").green().bold(),
							model_info,
						);
					}

					let elapsed = start_clone.elapsed().as_secs_f64();
					let fps = if elapsed > 0.1 && progress.current_frame > 0 {
						progress.current_frame as f64 / elapsed
					} else {
						0.0
					};

				match progress.stage.as_str() {
					"scanning" => {
						pb_inner.set_message(format!("{}", style("scanning depths").yellow()));
						let pct = if progress.total_frames > 0 {
							(progress.current_frame as f64 / progress.total_frames as f64 * 100.0) as u64
						} else {
							0
						};
						pb_inner.set_position(pct);
						pb_inner.set_prefix(format!("{}/{}", progress.current_frame, progress.total_frames));
					}
					"extracting" => {
						pb_inner.set_message(format!("{}", style("loading").yellow()));
						pb_inner.set_position(0);
					}
						"processing" => {
							let eta_secs = if fps > 0.1 {
								let remaining = progress.total_frames.saturating_sub(progress.current_frame);
								remaining as f64 / fps
							} else {
								0.0
							};

							let eta_str = if eta_secs > 60.0 {
								format!("{}m {:02}s", eta_secs as u64 / 60, eta_secs as u64 % 60)
							} else {
								format!("{:.0}s", eta_secs)
							};

							pb_inner.set_message(format!(
								"{:>5.1} fps",
								fps,
							));
							pb_inner.set_prefix(format!(
								"{}/{} eta {}",
								progress.current_frame,
								progress.total_frames,
								eta_str,
							));
							pb_inner.set_position(progress.percent as u64);
						}
						"encoding" => {
							pb_inner.set_message(format!("{}", style("encoding").yellow()));
							pb_inner.set_position(100);
							pb_inner.set_prefix(String::new());
						}
						"packaging" => {
							pb_inner.set_message(format!("{}", style("packaging MV-HEVC").yellow()));
						}
						"complete" => {
							pb_inner.finish_and_clear();
						}
						_ => {}
					}
				})),
			)
			.await?;

			pb.finish_and_clear();

			let elapsed = start.elapsed().as_secs_f64();
			let out_name = output.file_name().and_then(|s| s.to_str()).unwrap_or("?");
			eprintln!(
				"{} {} ({:.1}s)",
				style("→").green().bold(),
				style(out_name).white(),
				elapsed,
			);
		}
	}

	Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let cli = Cli::parse();

	if let Some(Commands::Self_ { action: SelfAction::Update }) = cli.command {
		return self_update().await;
	}

	if cli.inputs.is_empty() {
		eprintln!("No input files provided. Usage: spatial-maker <files...>");
		eprintln!("Run 'spatial-maker --help' for more information.");
		std::process::exit(1);
	}

	if cli.output.is_some() && cli.inputs.len() > 1 {
		eprintln!("--output cannot be used with multiple inputs");
		std::process::exit(1);
	}

	let output_types = parse_output_types(&cli.output_types).unwrap_or_else(|e| {
		eprintln!("Invalid --output-types: {}", e);
		std::process::exit(1);
	});

	let normalize_mode: NormalizeMode = cli.normalize.parse().unwrap_or_else(|e| {
		eprintln!("{}", e);
		std::process::exit(1);
	});

	let config = SpatialConfig {
		encoder_size: cli.model.clone(),
		max_disparity: cli.max_disparity,
		target_depth_size: 518,
		temporal_alpha: cli.temporal_alpha,
		bilateral_sigma_space: cli.bilateral_sigma,
		bilateral_sigma_color: cli.bilateral_range,
		depth_blur_sigma: cli.depth_blur,
		normalize_mode,
	};

	let total = cli.inputs.len();
	let mut errors = Vec::new();

	for (i, input) in cli.inputs.iter().enumerate() {
		if total > 1 {
			eprintln!(
				"{} {}/{}",
				style("─").dim(),
				style(i + 1).bold(),
				style(total).dim(),
			);
		}

		let output = cli
			.output
			.clone()
			.unwrap_or_else(|| generate_output_base(input, &cli.model));

		if let Err(e) = process_single(input, output, config.clone(), &output_types, &cli).await {
			eprintln!("{} {:?}: {}", style("✗").red().bold(), input, e);
			errors.push((input.clone(), e));
		}
	}

	if !errors.is_empty() {
		eprintln!(
			"\n{} {}/{} files failed",
			style("✗").red().bold(),
			errors.len(),
			total,
		);
		std::process::exit(1);
	}

	Ok(())
}

async fn self_update() -> Result<(), Box<dyn std::error::Error>> {
	let current_version = env!("CARGO_PKG_VERSION");
	let repo = "mrgnw/spatial-maker";

	eprintln!("Current version: v{}", current_version);
	eprintln!("Checking for updates...");

	let client = reqwest::Client::new();

	let release: serde_json::Value = client
		.get(format!("https://api.github.com/repos/{}/releases/latest", repo))
		.header("User-Agent", "spatial-maker")
		.send()
		.await?
		.json()
		.await?;

	let latest_tag = release["tag_name"]
		.as_str()
		.ok_or("Could not find latest release tag")?;
	let latest_version = latest_tag.trim_start_matches('v');

	if !is_newer_version(current_version, latest_version) {
		eprintln!("Already up to date (v{})", current_version);
		return Ok(());
	}

	eprintln!("New version available: v{} -> v{}", current_version, latest_version);

	let target = if cfg!(target_arch = "aarch64") {
		"aarch64-apple-darwin"
	} else if cfg!(target_arch = "x86_64") {
		"x86_64-apple-darwin"
	} else {
		return Err("Unsupported architecture".into());
	};

	let asset_name = format!("spatial-maker-{}-{}.tar.gz", latest_tag, target);

	let assets = release["assets"]
		.as_array()
		.ok_or("No assets in release")?;

	let download_url = assets
		.iter()
		.find(|a| a["name"].as_str() == Some(&asset_name))
		.and_then(|a| a["browser_download_url"].as_str())
		.ok_or_else(|| format!("No release asset found for {}", target))?;

	let pb = ProgressBar::new_spinner();
	pb.set_style(
		ProgressStyle::default_spinner()
			.template("{spinner:.cyan} {msg}")
			.unwrap()
	);
	pb.set_message(format!("Downloading {}...", asset_name));
	pb.enable_steady_tick(std::time::Duration::from_millis(80));

	let response = client
		.get(download_url)
		.header("User-Agent", "spatial-maker")
		.send()
		.await?;

	let bytes = response.bytes().await?;

	pb.set_message("Extracting...");

	let decoder = flate2::read::GzDecoder::new(&bytes[..]);
	let mut archive = tar::Archive::new(decoder);

	let temp_dir = std::env::temp_dir().join("spatial-maker-update");
	let _ = std::fs::remove_dir_all(&temp_dir);
	std::fs::create_dir_all(&temp_dir)?;

	archive.unpack(&temp_dir)?;

	let new_binary = temp_dir.join("spatial-maker");
	if !new_binary.exists() {
		pb.finish_and_clear();
		return Err("Binary not found in release archive".into());
	}

	let current_exe = std::env::current_exe()?;
	let install_path = if is_writable(&current_exe) {
		current_exe.clone()
	} else {
		let local_bin = dirs::home_dir()
			.ok_or("Could not determine home directory")?
			.join(".local/bin");
		std::fs::create_dir_all(&local_bin)?;
		let alt_path = local_bin.join("spatial-maker");
		eprintln!(
			"Cannot write to {} (try sudo), installing to {}",
			current_exe.display(),
			alt_path.display()
		);
		alt_path
	};

	let staging = install_path.with_extension("new");
	std::fs::copy(&new_binary, &staging)?;

	#[cfg(unix)]
	{
		use std::os::unix::fs::PermissionsExt;
		std::fs::set_permissions(&staging, std::fs::Permissions::from_mode(0o755))?;
	}

	std::fs::rename(&staging, &install_path)?;

	let _ = std::fs::remove_dir_all(&temp_dir);

	pb.finish_and_clear();
	eprintln!(
		"{} Updated to {} at {}",
		style("✔").green().bold(),
		style(format!("v{}", latest_version)).green(),
		install_path.display(),
	);

	if install_path != current_exe {
		eprintln!("Make sure {} is in your PATH", install_path.parent().unwrap().display());
	}

	Ok(())
}

fn is_newer_version(current: &str, latest: &str) -> bool {
	let parse = |v: &str| -> Vec<u32> {
		v.split('.').filter_map(|s| s.parse().ok()).collect()
	};
	let c = parse(current);
	let l = parse(latest);
	l > c
}

fn is_writable(path: &PathBuf) -> bool {
	if path.exists() {
		std::fs::OpenOptions::new()
			.write(true)
			.open(path)
			.is_ok()
	} else {
		path.parent()
			.map(|p| {
				let test = p.join(".spatial-maker-write-test");
				let ok = std::fs::File::create(&test).is_ok();
				let _ = std::fs::remove_file(&test);
				ok
			})
			.unwrap_or(false)
	}
}
