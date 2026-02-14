use clap::Parser;
use spatial_maker::{generate_stereo_pair, model, output, OnnxDepthEstimator};

#[derive(Parser)]
#[command(name = "spatial-maker-onnx")]
#[command(about = "Convert a 2D photo to side-by-side stereo using ONNX Runtime")]
struct Args {
	input: String,

	#[arg(short, long)]
	output: String,

	#[arg(long, default_value = "30")]
	max_disparity: u32,

	#[arg(long, default_value = "s")]
	encoder_size: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let args = Args::parse();

	let input_image = image::open(&args.input)?;

	let model_path = model::find_model(&args.encoder_size)?;

	let estimator = OnnxDepthEstimator::new(model_path.to_str().unwrap())?;

	let start = std::time::Instant::now();
	let depth_map = estimator.estimate(&input_image)?;
	let depth_time = start.elapsed();
	eprintln!("Depth estimation: {:?}", depth_time);

	let (left, right) = generate_stereo_pair(&input_image, &depth_map, args.max_disparity)?;

	let sbs = output::create_sbs_image(&left, &right);
	sbs.save(&args.output)?;

	eprintln!("Total time: {:?}", start.elapsed());

	Ok(())
}
