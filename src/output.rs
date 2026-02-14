use image::{DynamicImage, ImageBuffer, RgbImage};

/// Create a side-by-side stereo image by horizontally stacking left and right views
pub fn create_sbs_image(left: RgbImage, right: RgbImage) -> DynamicImage {
    let (width, height) = left.dimensions();
    let mut sbs = ImageBuffer::new(width * 2, height);

    // Copy left eye to left half
    for y in 0..height {
        for x in 0..width {
            sbs.put_pixel(x, y, *left.get_pixel(x, y));
        }
    }

    // Copy right eye to right half
    for y in 0..height {
        for x in 0..width {
            sbs.put_pixel(width + x, y, *right.get_pixel(x, y));
        }
    }

    DynamicImage::ImageRgb8(sbs)
}
