use clap::{Parser, Subcommand};
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

fn generate_output_path(input: &PathBuf, media_type: &MediaType) -> PathBuf {
	let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
	let src_ext = input
		.extension()
		.and_then(|s| s.to_str())
		.unwrap_or("")
		.to_lowercase();

	let extension = match media_type {
		MediaType::Video => "mov",
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

async fn self_update() -> Result<(), Box<dyn std::error::Error>> {
	let current_version = env!("CARGO_PKG_VERSION");
	let repo = "mrgnw/spatial-maker";

	eprintln!("Current version: v{}", current_version);
	eprintln!("Checking for updates...");

	let client = reqwest::Client::new();

	// Get latest release from GitHub API
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

	// Determine current arch
	let target = if cfg!(target_arch = "aarch64") {
		"aarch64-apple-darwin"
	} else if cfg!(target_arch = "x86_64") {
		"x86_64-apple-darwin"
	} else {
		return Err("Unsupported architecture".into());
	};

	let asset_name = format!("spatial-maker-{}-{}.tar.gz", latest_tag, target);

	// Find the download URL from release assets
	let assets = release["assets"]
		.as_array()
		.ok_or("No assets in release")?;

	let download_url = assets
		.iter()
		.find(|a| a["name"].as_str() == Some(&asset_name))
		.and_then(|a| a["browser_download_url"].as_str())
		.ok_or_else(|| format!("No release asset found for {}", target))?;

	eprintln!("Downloading {}...", asset_name);

	// Download to a temp file
	let response = client
		.get(download_url)
		.header("User-Agent", "spatial-maker")
		.send()
		.await?;

	let bytes = response.bytes().await?;

	// Extract the binary from the tarball
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

	// Determine where to install
	let current_exe = std::env::current_exe()?;
	let install_path = if is_writable(&current_exe) {
		current_exe.clone()
	} else {
		// Try ~/.local/bin as a user-writable alternative
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

	// Replace the binary atomically: copy new -> rename over old
	let staging = install_path.with_extension("new");
	std::fs::copy(&new_binary, &staging)?;

	#[cfg(unix)]
	{
		use std::os::unix::fs::PermissionsExt;
		std::fs::set_permissions(&staging, std::fs::Permissions::from_mode(0o755))?;
	}

	std::fs::rename(&staging, &install_path)?;

	// Cleanup
	let _ = std::fs::remove_dir_all(&temp_dir);

	eprintln!("Updated to v{} at {}", latest_version, install_path.display());

	// Remind about PATH if installed to ~/.local/bin
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
		// Try opening for write to test permissions
		std::fs::OpenOptions::new()
			.write(true)
			.open(path)
			.is_ok()
	} else {
		// Check if parent directory is writable
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
