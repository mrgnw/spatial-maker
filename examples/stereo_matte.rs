use image::GrayImage;
use spatial_maker::matte::{foreground_matte, generate_stereo_pair_layered};
use spatial_maker::output::load_depth_map;
use spatial_maker::stereo::{generate_stereo_pair_with_refinement, EdgeRefinement};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if !(5..=7).contains(&args.len()) {
        return Err(format!(
            "usage: {} IMAGE DEPTH_PNG16 MAX_DISPARITY RIGHT_EYE_OUT [MATTE_OUT [MATTE_GAIN]]",
            args[0]
        )
        .into());
    }
    let image = image::open(PathBuf::from(&args[1]))?;
    let depth = load_depth_map(&PathBuf::from(&args[2]))?;
    let max_disparity: u32 = args[3].parse()?;
    let matte_gain: f32 = args.get(6).map_or(Ok(1.0), |g| g.parse())?;
    let refinement = Some(EdgeRefinement::default());

    let start = Instant::now();
    let mut matte = foreground_matte(&image.to_rgb8())?;
    println!("vision {:.0} ms", start.elapsed().as_secs_f64() * 1000.0);
    if let Some(alpha) = &mut matte {
        for a in alpha.iter_mut() {
            *a = ((*a - 0.5) * matte_gain + 0.5).clamp(0.0, 1.0);
        }
    }

    let start = Instant::now();
    let (_, right) = match &matte {
        Some(alpha) => {
            generate_stereo_pair_layered(&image, &depth, max_disparity, refinement, alpha)?
        }
        None => {
            println!("no subject, plain warp");
            generate_stereo_pair_with_refinement(&image, &depth, max_disparity, refinement)?
        }
    };
    println!("stereo {:.0} ms", start.elapsed().as_secs_f64() * 1000.0);
    right.save(&args[4])?;

    if let (Some(alpha), Some(path)) = (&matte, args.get(5)) {
        let pixels = alpha.iter().map(|a| (a * 255.0).round() as u8).collect();
        GrayImage::from_raw(image.width(), image.height(), pixels)
            .ok_or("matte size")?
            .save(path)?;
    }
    Ok(())
}
