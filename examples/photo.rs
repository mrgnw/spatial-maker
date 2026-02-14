use anyhow::Result;
use clap::Parser;
use spatial_maker::{create_stereo_pair, model, output, DepthEstimator};

#[derive(Parser)]
#[command(name = "spatial-maker-photo")]
#[command(about = "Convert a 2D photo to side-by-side stereo", long_about = None)]
struct Args {
    /// Input image file
    input: String,

    /// Output image file
    #[arg(short, long)]
    output: String,

    /// Maximum pixel disparity for 3D effect (default: 30)
    #[arg(long, default_value = "30.0")]
    max_disparity: f32,

    /// Model checkpoint name (default: depth_anything_v2_vits.onnx)
    #[arg(long, default_value = "depth_anything_v2_vits.onnx")]
    model: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading image: {}", args.input);
    let input_image = image::open(&args.input)?;

    println!("Finding model checkpoint: {}", args.model);
    let model_path = model::find_checkpoint(&args.model)?;
    println!("Using model: {}", model_path.display());

    println!("Initializing depth estimator...");
    let mut estimator = DepthEstimator::new(model_path.to_str().unwrap())?;

    println!("Estimating depth...");
    let depth_map = estimator.estimate(&input_image)?;

    println!("Creating stereo pair...");
    let (left, right) = create_stereo_pair(&input_image, &depth_map, args.max_disparity)?;

    println!("Creating SBS image...");
    let sbs = output::create_sbs_image(left, right);

    println!("Saving to: {}", args.output);
    sbs.save(&args.output)?;

    println!("Done!");
    Ok(())
}
