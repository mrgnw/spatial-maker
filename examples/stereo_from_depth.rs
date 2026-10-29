use image::{ImageBuffer, Luma};
use spatial_maker::depth_refine::{
    dilate_foreground, median_depth_edges, median_stride, refine_depth_edges, DEFAULT_EPSILON,
    DEFAULT_RADIUS_FRACTION,
};
use spatial_maker::output::load_depth_map;
use spatial_maker::stereo::generate_stereo_pair_with_refinement;
use std::path::PathBuf;
use std::time::Instant;

// RADIUS_FRACTION 0 skips the guided filter. MEDIAN_SIGMA_COLOR 0 skips the weighted median.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if ![5, 7, 10, 11].contains(&args.len()) {
        return Err(format!(
            "usage: {} IMAGE DEPTH_PNG16 MAX_DISPARITY RIGHT_EYE_OUT \
             [RADIUS_FRACTION EPSILON [MEDIAN_SIGMA_COLOR MEDIAN_PASSES DILATE_PX [STRIDE_SCALE]]]",
            args[0]
        )
        .into());
    }
    let image = image::open(PathBuf::from(&args[1]))?;
    let depth = load_depth_map(&PathBuf::from(&args[2]))?;
    let max_disparity: u32 = args[3].parse()?;
    let number = |i: usize, default: f32| -> Result<f32, Box<dyn std::error::Error>> {
        Ok(args
            .get(i)
            .map(|a| a.parse())
            .transpose()?
            .unwrap_or(default))
    };
    let radius_fraction = number(5, DEFAULT_RADIUS_FRACTION)?;
    let epsilon = number(6, DEFAULT_EPSILON)?;
    let sigma_color = number(7, 0.0)?;
    let passes = number(8, 0.0)? as usize;
    let dilate = number(9, 0.0)? as usize;
    let stride_scale = number(10, 1.0)?;

    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    let (depth_height, depth_width) = depth.dim();
    let depth = ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(
        depth_width as u32,
        depth_height as u32,
        depth.iter().copied().collect(),
    )
    .expect("depth buffer matches its dimensions");
    let mut depth = image::imageops::resize(
        &depth,
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    )
    .into_raw();

    let start = Instant::now();
    if radius_fraction > 0.0 {
        depth = refine_depth_edges(&rgb, &depth, radius_fraction, epsilon);
    }
    let guided_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    if sigma_color > 0.0 {
        let stride = (median_stride(width, height) as f32 * stride_scale)
            .round()
            .max(1.0) as usize;
        depth = median_depth_edges(&rgb, &depth, stride, sigma_color, passes);
    }
    let median_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    if dilate > 0 {
        depth = dilate_foreground(&depth, width, height, dilate);
    }
    let dilate_ms = start.elapsed().as_secs_f64() * 1000.0;

    let depth = ndarray::Array2::from_shape_vec((height, width), depth)?;
    let start = Instant::now();
    let (_, right) = generate_stereo_pair_with_refinement(&image, &depth, max_disparity, None)?;
    println!(
        "{}x{} guided {:.0} ms, median {:.0} ms, dilate {:.0} ms, warp {:.0} ms",
        width,
        height,
        guided_ms,
        median_ms,
        dilate_ms,
        start.elapsed().as_secs_f64() * 1000.0
    );

    right.save(&args[4])?;
    Ok(())
}
