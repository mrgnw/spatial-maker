use anyhow::Result;
use image::{DynamicImage, ImageBuffer, Luma, Rgb, RgbImage};

/// Create left and right eye views from RGB frame and depth map using DIBR
///
/// This ports the `create_stereo_pair` function from depth_to_stereo.py
pub fn create_stereo_pair(
    rgb_frame: &DynamicImage,
    depth_frame: &ImageBuffer<Luma<f32>, Vec<f32>>,
    max_disparity: f32,
) -> Result<(RgbImage, RgbImage)> {
    let rgb = rgb_frame.to_rgb8();
    let (width, height) = rgb.dimensions();

    // Create disparity map (closer = higher value = more shift)
    let disparity: Vec<f32> = depth_frame.pixels().map(|p| p[0] * max_disparity).collect();

    let left_eye = remap_image(&rgb, &disparity, width, height, 0.5)?;
    let right_eye = remap_image(&rgb, &disparity, width, height, -0.5)?;

    Ok((left_eye, right_eye))
}

/// Remap an image based on disparity values
///
/// This is equivalent to cv2.remap with INTER_LINEAR and BORDER_REPLICATE
fn remap_image(
    source: &RgbImage,
    disparity: &[f32],
    width: u32,
    height: u32,
    shift_factor: f32,
) -> Result<RgbImage> {
    let mut output = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let shift = disparity[idx] * shift_factor;
            let src_x = x as f32 + shift;

            // Bilinear interpolation
            let pixel = sample_bilinear(source, src_x, y as f32);
            output.put_pixel(x, y, pixel);
        }
    }

    Ok(output)
}

/// Sample a pixel using bilinear interpolation
fn sample_bilinear(image: &RgbImage, x: f32, y: f32) -> Rgb<u8> {
    let (width, height) = image.dimensions();

    // Clamp to image bounds (BORDER_REPLICATE)
    let x_clamped = x.clamp(0.0, (width - 1) as f32);
    let y_clamped = y.clamp(0.0, (height - 1) as f32);

    let x0 = x_clamped.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y0 = y_clamped.floor() as u32;
    let y1 = (y0 + 1).min(height - 1);

    let fx = x_clamped - x0 as f32;
    let fy = y_clamped - y0 as f32;

    let p00 = image.get_pixel(x0, y0);
    let p10 = image.get_pixel(x1, y0);
    let p01 = image.get_pixel(x0, y1);
    let p11 = image.get_pixel(x1, y1);

    let mut result = [0u8; 3];
    for c in 0..3 {
        let v00 = p00[c] as f32;
        let v10 = p10[c] as f32;
        let v01 = p01[c] as f32;
        let v11 = p11[c] as f32;

        let v0 = v00 * (1.0 - fx) + v10 * fx;
        let v1 = v01 * (1.0 - fx) + v11 * fx;
        let v = v0 * (1.0 - fy) + v1 * fy;

        result[c] = v.round().clamp(0.0, 255.0) as u8;
    }

    Rgb(result)
}
