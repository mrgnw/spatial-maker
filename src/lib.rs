pub mod depth;
pub mod model;
pub mod output;
pub mod stereo;

#[cfg(target_os = "macos")]
pub mod depth_coreml;

pub use depth::DepthEstimator;
pub use stereo::create_stereo_pair;

#[cfg(target_os = "macos")]
pub use depth_coreml::CoreMLDepthEstimator;

use anyhow::Result;
use image::DynamicImage;

/// Process a photo to create a side-by-side stereo image (ONNX backend)
pub fn process_photo(
    input_image: DynamicImage,
    model_path: &str,
    max_disparity: f32,
) -> Result<DynamicImage> {
    let mut estimator = DepthEstimator::new(model_path)?;
    let depth_map = estimator.estimate(&input_image)?;
    let (left, right) = create_stereo_pair(&input_image, &depth_map, max_disparity)?;
    Ok(output::create_sbs_image(left, right))
}

/// Process a photo to create a side-by-side stereo image (CoreML backend, macOS only)
#[cfg(target_os = "macos")]
pub fn process_photo_coreml(
    input_image: DynamicImage,
    model_path: &str,
    max_disparity: f32,
) -> Result<DynamicImage> {
    let estimator = CoreMLDepthEstimator::new(model_path)?;
    let depth_map = estimator.estimate(&input_image)?;
    let (left, right) = create_stereo_pair(&input_image, &depth_map, max_disparity)?;
    Ok(output::create_sbs_image(left, right))
}
