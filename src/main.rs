use clap::{Parser, Subcommand};
use spatial_maker::{
	process_video, ImageEncoding, SpatialPhotoConfig, NormalizeMode, OutputFormat,
	OutputOptions, OutputType, SpatialConfig, VideoProgress,
	needs_stereo, parse_output_types,
	tui::{self, AppState, FileStatus, MediaType},
	load_image, model, generate_stereo_pair_with_progress,
	comfort_disparity_adjustment, needs_depth, depth_formats, save_depth_map, load_depth_map, save_stereo_image, spatial_photo_path,
	CoreMLDepthEstimator,
};
use std::path::PathBuf;
use std::time::Instant;
use std::path::Path;
use tokio::sync::mpsc;

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

	/// Model: s (small, 48MB), b (base, 186MB, non-commercial), l (large, 638MB, non-commercial), da3 (v3 mono large, 638MB),
	/// vda (video depth anything small, temporally stable, 56MB). Default: da3 for photos, s for videos
	#[arg(short, long)]
	model: Option<String>,

	/// Maximum disparity in pixels (higher = more 3D depth). Default: 1.2% of image width
	#[arg(long)]
	max_disparity: Option<u32>,

	/// Output types (comma-separated): depth, depth:avif,png,png16, sbs, tab, sep, spatial
	#[arg(long, default_value = "spatial")]
	output_types: String,

	/// JPEG quality for photos (1-100)
	#[arg(long, default_value = "95", value_parser = clap::value_parser!(u8).range(1..=100))]
	quality: u8,

	/// HEIC quality for spatial photos (1-100)
	#[arg(long, default_value_t = spatial_maker::DEFAULT_HEIC_QUALITY, value_parser = clap::value_parser!(u8).range(1..=100))]
	heic_quality: u8,

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

	/// Force regeneration of depth maps even if they already exist
	#[arg(short, long)]
	force: bool,
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

const DEFAULT_PHOTO_MODEL: &str = "da3";
const DEFAULT_VIDEO_MODEL: &str = "s";

fn model_for(requested: &Option<String>, media_type: &MediaType) -> String {
	match (requested, media_type) {
		(Some(model), _) => model.clone(),
		(None, MediaType::Photo) => DEFAULT_PHOTO_MODEL.to_string(),
		(None, MediaType::Video) => DEFAULT_VIDEO_MODEL.to_string(),
	}
}

fn generate_output_base(input: &PathBuf, model: &str) -> PathBuf {
	let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
	let parent = input.parent().unwrap_or_else(|| std::path::Path::new("."));
	parent.join(format!("{}-{}", stem, model))
}

enum TuiEvent {
	FileStarted(usize),
	StageUpdate { index: usize, stage: String, progress: f64 },
	FileDone { index: usize, outputs: Vec<String>, duration: std::time::Duration },
	FileError { index: usize, error: String },
	VideoProgress { index: usize, progress: VideoProgress, fps: f64, eta: String },
	AllDone,
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

	let first_media_type = cli.inputs.first().map(detect_media_type).unwrap_or(MediaType::Photo);
	let header_model = model_for(&cli.model, &first_media_type);

	let config = SpatialConfig {
		encoder_size: header_model.clone(),
		max_disparity: cli.max_disparity,
		target_depth_size: 518,
		temporal_alpha: cli.temporal_alpha,
		bilateral_sigma_space: cli.bilateral_sigma,
		bilateral_sigma_color: cli.bilateral_range,
		depth_blur_sigma: cli.depth_blur,
		normalize_mode,
	};

	let model_meta = model::ModelMetadata::coreml(&header_model).unwrap_or_else(|e| {
		eprintln!("{}", e);
		std::process::exit(1);
	});

	let filenames: Vec<(String, MediaType)> = cli
		.inputs
		.iter()
		.map(|p| {
			let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string();
			let mt = detect_media_type(p);
			(name, mt)
		})
		.collect();

	let mut state = AppState::new(filenames, &model_meta.name, model_meta.size_mb);
	let mut terminal = tui::init_terminal()?;

	let (tx, mut rx) = mpsc::unbounded_channel::<TuiEvent>();

	let inputs_owned: Vec<PathBuf> = cli.inputs.clone();
	let output_opt = cli.output.clone();
	let requested_model = cli.model.clone();
	let quality = cli.quality;
	let heic_quality = cli.heic_quality;
	let force = cli.force;
	let output_types_owned = output_types.clone();
	let config_owned = config.clone();

	tokio::spawn(async move {
		for (i, input) in inputs_owned.iter().enumerate() {
			let _ = tx.send(TuiEvent::FileStarted(i));

			let model = model_for(&requested_model, &detect_media_type(input));
			let output = output_opt
				.clone()
				.unwrap_or_else(|| generate_output_base(input, &model));

			let file_start = Instant::now();

			let result = process_file(
				&tx,
				i,
				input,
				output,
				SpatialConfig {
					encoder_size: model,
					..config_owned.clone()
				},
				&output_types_owned,
				quality,
				heic_quality,
				force,
			)
			.await;

			let duration = file_start.elapsed();

			match result {
				Ok(outputs) => {
					let _ = tx.send(TuiEvent::FileDone { index: i, outputs, duration });
				}
				Err(e) => {
					let _ = tx.send(TuiEvent::FileError {
						index: i,
						error: e.to_string(),
					});
				}
			}
		}
		let _ = tx.send(TuiEvent::AllDone);
	});

	let mut tick_interval = tokio::time::interval(std::time::Duration::from_millis(100));
	let mut done = false;

	loop {
		tokio::select! {
			_ = tick_interval.tick() => {
				tui::render_frame(&mut terminal, &state)?;
			}
			event = rx.recv() => {
				match event {
				Some(TuiEvent::FileStarted(i)) => {
					state.mark_processing(i);
				}
				Some(TuiEvent::StageUpdate { index, stage, progress }) => {
					state.update_stage(index, stage, progress);
				}
					Some(TuiEvent::FileDone { index, outputs, duration }) => {
						state.mark_done(index, outputs, duration);
						let file_state = state.files[index].clone();
						tui::insert_completed_line(&mut terminal, &file_state, index, &state)?;
					}
					Some(TuiEvent::FileError { index, error }) => {
						state.mark_error(index, error);
						let file_state = state.files[index].clone();
						tui::insert_completed_line(&mut terminal, &file_state, index, &state)?;
					}
					Some(TuiEvent::VideoProgress { index, progress, fps, eta }) => {
						state.update_video_progress(index, &progress, fps, eta);
					}
					Some(TuiEvent::AllDone) | None => {
						done = true;
					}
				}
				tui::render_frame(&mut terminal, &state)?;
				if done {
					break;
				}
			}
		}
	}

	tui::restore_terminal();

	let error_count = state
		.files
		.iter()
		.filter(|f| matches!(f.status, FileStatus::Error(_)))
		.count();

	if error_count > 0 {
		eprintln!(
			"\n{}/{} files failed",
			error_count,
			state.total,
		);
		std::process::exit(1);
	}

	Ok(())
}

async fn process_file(
	tx: &mpsc::UnboundedSender<TuiEvent>,
	index: usize,
	input: &PathBuf,
	output: PathBuf,
	config: SpatialConfig,
	output_types: &[OutputType],
	quality: u8,
	heic_quality: u8,
	force: bool,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
	let media_type = detect_media_type(input);

	match media_type {
		MediaType::Photo => {
			let parent = output.parent().unwrap_or_else(|| Path::new("."));
			let stem = output.file_stem().and_then(|s| s.to_str()).unwrap_or("output");

			let do_depth = needs_depth(output_types);
			let do_stereo = needs_stereo(output_types);

			let depth_paths: Vec<(std::path::PathBuf, spatial_maker::DepthFormat)> = if do_depth {
				depth_formats(output_types)
					.into_iter()
					.map(|fmt| {
						let filename = format!("{}-depth{}.{}", stem, fmt.suffix(), fmt.extension());
						(parent.join(&filename), fmt)
					})
					.collect()
			} else {
				Vec::new()
			};

			let all_depth_exist = !depth_paths.is_empty() && depth_paths.iter().all(|(p, _)| p.exists());
			let skip_estimation = all_depth_exist && !force;

			let mut outputs = Vec::new();

			let depth_map = if skip_estimation {
				let _ = tx.send(TuiEvent::StageUpdate {
					index,
					stage: "depth cached".to_string(),
					progress: 1.0,
				});

				for (p, _) in &depth_paths {
					if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
						outputs.push(name.to_string());
					}
				}

				if do_stereo {
					let best = depth_paths.iter()
						.find(|(_, fmt)| matches!(fmt, spatial_maker::DepthFormat::Png16))
						.or_else(|| depth_paths.iter().find(|(_, fmt)| matches!(fmt, spatial_maker::DepthFormat::Png)))
						.or_else(|| depth_paths.first())
						.map(|(p, _)| p);
					match best {
						Some(p) => Some(load_depth_map(p)?),
						None => None,
					}
				} else {
					None
				}
			} else {
				let _ = tx.send(TuiEvent::StageUpdate {
					index,
					stage: "loading".to_string(),
					progress: 0.0,
				});
				let input_image_for_depth = load_image(input).await?;

				let _ = tx.send(TuiEvent::StageUpdate {
					index,
					stage: "loading model".to_string(),
					progress: 0.0,
				});
				model::ensure_model_exists::<fn(u64, u64)>(&config.encoder_size, None).await?;
				let model_path = model::find_model(&config.encoder_size)?;
				let model_str = model_path.to_str().ok_or("Invalid model path encoding")?;
				let meta = model::ModelMetadata::coreml(&config.encoder_size)?;
				let estimator =
					CoreMLDepthEstimator::new(model_str, meta.input_size, meta.inverted_depth)?;

				let _ = tx.send(TuiEvent::StageUpdate {
					index,
					stage: "estimating depth".to_string(),
					progress: 0.0,
				});
				let dm = estimator.estimate(&input_image_for_depth)?;

				if do_depth {
					let _ = tx.send(TuiEvent::StageUpdate {
						index,
						stage: "saving depth".to_string(),
						progress: 0.0,
					});

					for (depth_path, fmt) in &depth_paths {
						save_depth_map(&dm, depth_path, *fmt)?;
						if let Some(name) = depth_path.file_name().and_then(|s| s.to_str()) {
							outputs.push(name.to_string());
						}
					}
				}

				Some(dm)
			};

			if do_stereo {
				let dm = depth_map.as_ref().ok_or("Depth map required for stereo but not available")?;
				let input_image = load_image(input).await?;

				let _ = tx.send(TuiEvent::StageUpdate {
					index,
					stage: "generating stereo".to_string(),
					progress: 0.0,
				});

				let tx_clone = tx.clone();
				let max_disparity = config.max_disparity_px(input_image.width());
				let (left, right) = generate_stereo_pair_with_progress(
					&input_image,
					dm,
					max_disparity,
					spatial_maker::stereo::EdgeRefinement::for_photo(),
					Some(move |progress| {
						let _ = tx_clone.send(TuiEvent::StageUpdate {
							index,
							stage: "generating stereo".to_string(),
							progress,
						});
					}),
				)?;

				let _ = tx.send(TuiEvent::StageUpdate {
					index,
					stage: "saving".to_string(),
					progress: 0.0,
				});

				let stereo = spatial_maker::stereo_types(output_types);
				let layout = match stereo.first() {
					Some(OutputType::TopAndBottom) => OutputFormat::TopAndBottom,
					Some(OutputType::Separate) => OutputFormat::Separate,
					_ => OutputFormat::SideBySide,
				};

				let has_spatial = output_types.iter().any(|t| matches!(t, OutputType::Spatial));
				let has_layout = output_types.iter().any(|t| matches!(t, OutputType::SideBySide | OutputType::TopAndBottom | OutputType::Separate));

				let output_options = OutputOptions {
					layout,
					image_format: ImageEncoding::Jpeg { quality },
					spatial_photo: has_spatial.then_some(SpatialPhotoConfig {
						quality: heic_quality,
						keep_layout_image: has_layout,
						disparity_adjustment: comfort_disparity_adjustment(max_disparity, input_image.width()),
					}),
				};

				let src_ext = input.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
				let stereo_ext = match src_ext.as_str() {
					"heic" | "heif" | "avif" | "jxl" => "jpg",
					"" => "jpg",
					other => other,
				};
				let parent = output.parent().unwrap_or_else(|| Path::new("."));
				let stem = output.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
				let stereo_path = parent.join(format!("{}-spatial.{}", stem, stereo_ext));
				save_stereo_image(&left, &right, &stereo_path, output_options)?;

				let mut written = Vec::new();
				if has_layout {
					written.push(stereo_path.clone());
				}
				if has_spatial {
					written.push(spatial_photo_path(&stereo_path));
				}
				for path in written {
					if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
						outputs.push(name.to_string());
					}
				}
			}

			Ok(outputs)
		}
		MediaType::Video => {
			let start = Instant::now();
			let tx_clone = tx.clone();

			process_video(
				input,
				&output,
				config,
				output_types,
				Some(Box::new(move |progress: VideoProgress| {
					let elapsed = start.elapsed().as_secs_f64();
					let fps = if elapsed > 0.1 && progress.current_frame > 0 {
						progress.current_frame as f64 / elapsed
					} else {
						0.0
					};

					let eta = if fps > 0.1 {
						let remaining = progress.total_frames.saturating_sub(progress.current_frame);
						let eta_secs = remaining as f64 / fps;
						if eta_secs > 60.0 {
							format!("{}m{:02}s", eta_secs as u64 / 60, eta_secs as u64 % 60)
						} else {
							format!("{:.0}s", eta_secs)
						}
					} else {
						String::new()
					};

					let _ = tx_clone.send(TuiEvent::VideoProgress {
						index,
						progress,
						fps,
						eta,
					});
				})),
				force,
			)
			.await?;

			let out_name = output
				.file_name()
				.and_then(|s| s.to_str())
				.unwrap_or("?")
				.to_string();

			Ok(vec![out_name])
		}
	}
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

	eprintln!("Downloading {}...", asset_name);

	let response = client
		.get(download_url)
		.header("User-Agent", "spatial-maker")
		.send()
		.await?;

	let bytes = response.bytes().await?;

	eprintln!("Extracting...");

	let decoder = flate2::read::GzDecoder::new(&bytes[..]);
	let mut archive = tar::Archive::new(decoder);

	let temp_dir = std::env::temp_dir().join("spatial-maker-update");
	let _ = std::fs::remove_dir_all(&temp_dir);
	std::fs::create_dir_all(&temp_dir)?;

	archive.unpack(&temp_dir)?;

	let new_binary = temp_dir.join("spatial-maker");
	if !new_binary.exists() {
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

	eprintln!(
		"Updated to v{} at {}",
		latest_version,
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
