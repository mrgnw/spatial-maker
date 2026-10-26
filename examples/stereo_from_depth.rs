use spatial_maker::output::load_depth_map;
use spatial_maker::stereo::{generate_stereo_pair_with_refinement, EdgeRefinement};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 && args.len() != 7 {
        return Err(format!(
            "usage: {} IMAGE DEPTH_PNG16 MAX_DISPARITY RIGHT_EYE_OUT [RADIUS_FRACTION EPSILON]",
            args[0]
        )
        .into());
    }
    let image = image::open(PathBuf::from(&args[1]))?;
    let depth = load_depth_map(&PathBuf::from(&args[2]))?;
    let max_disparity: u32 = args[3].parse()?;
    let refinement = if args.len() == 7 {
        let radius_fraction: f32 = args[5].parse()?;
        let epsilon: f32 = args[6].parse()?;
        (radius_fraction > 0.0).then_some(EdgeRefinement {
            radius_fraction,
            epsilon,
        })
    } else {
        Some(EdgeRefinement::default())
    };

    let start = std::time::Instant::now();
    let (_, right) =
        generate_stereo_pair_with_refinement(&image, &depth, max_disparity, refinement)?;
    println!("stereo {:.0} ms", start.elapsed().as_secs_f64() * 1000.0);

    right.save(&args[4])?;
    Ok(())
}
