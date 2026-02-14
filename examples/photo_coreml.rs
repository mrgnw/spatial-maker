use anyhow::Result;
use clap::Parser;
use spatial_maker::{create_stereo_pair, model, output, CoreMLDepthEstimator};

#[derive(Parser)]
#[command(name = "spatial-maker-coreml")]
#[command(about = "Convert a 2D photo to side-by-side stereo using CoreML (macOS only)", long_about = None)]
struct Args {
    /// Input image file
    input: String,

    /// Output image file
    #[arg(short, long)]
    output: String,

    /// Maximum pixel disparity for 3D effect (default: 30)
    #[arg(long, default_value = "30.0")]
    max_disparity: f32,

    /// Model checkpoint name (default: DepthAnythingV2BaseF16.mlpackage)
    #[arg(long, default_value = "DepthAnythingV2BaseF16.mlpackage")]
    model: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("🍎 Using CoreML backend (Apple Neural Engine + GPU + CPU)");
    println!("Loading image: {}", args.input);
    let input_image = image::open(&args.input)?;

    println!("Finding model checkpoint: {}", args.model);
    let model_path = model::find_checkpoint(&args.model)?;
    println!("Using model: {}", model_path.display());

    println!("Initializing CoreML depth estimator...");
    let estimator = CoreMLDepthEstimator::new(model_path.to_str().unwrap())?;

    println!("Estimating depth (this should be ~60-90ms on M4 Pro)...");
    let start = std::time::Instant::now();
    let depth_map = estimator.estimate(&input_image)?;
    let depth_time = start.elapsed();
    println!("✓ Depth estimation: {:?}", depth_time);

    println!("Creating stereo pair...");
    let (left, right) = create_stereo_pair(&input_image, &depth_map, args.max_disparity)?;

    println!("Creating SBS image...");
    let sbs = output::create_sbs_image(left, right);

    println!("Saving to: {}", args.output);
    sbs.save(&args.output)?;

    let total_time = start.elapsed();
    println!("\n✅ Done!");
    println!("Total time: {:?}", total_time);
    println!("Depth time: {:?}", depth_time);

    Ok(())
}
