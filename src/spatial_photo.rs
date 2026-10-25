use crate::error::{SpatialError, SpatialResult};
use crate::spatial_video::{path_to_cstring, SpatialMetadata};
use image::DynamicImage;
use std::os::raw::c_char;
use std::path::Path;

extern "C" {
    fn spatial_photo_write(
        output_path: *const c_char,
        left: *const u8,
        right: *const u8,
        width: i32,
        height: i32,
        quality: f64,
        baseline_micrometers: u32,
        horizontal_fov_millidegrees: u32,
        disparity_adjustment: i32,
    ) -> i32;
}

pub fn write_spatial_photo(
    output_path: &Path,
    left: &DynamicImage,
    right: &DynamicImage,
    quality: u8,
    metadata: &SpatialMetadata,
) -> SpatialResult<()> {
    assert!((1..=100).contains(&quality));
    assert!((-1.0..=1.0).contains(&metadata.disparity_adjustment));
    let left = left.to_rgb8();
    let right = right.to_rgb8();
    if left.dimensions() != right.dimensions() {
        return Err(SpatialError::ImageError(format!(
            "Left and right images must have the same size: {:?} != {:?}",
            left.dimensions(),
            right.dimensions()
        )));
    }
    let (width, height) = left.dimensions();
    let output = path_to_cstring(output_path)?;
    let status = unsafe {
        spatial_photo_write(
            output.as_ptr(),
            left.as_ptr(),
            right.as_ptr(),
            width as i32,
            height as i32,
            quality as f64 / 100.0,
            (metadata.baseline_mm * 1000.0).round() as u32,
            (metadata.horizontal_fov_degrees * 1000.0).round() as u32,
            (metadata.disparity_adjustment * 10000.0).round() as i32,
        )
    };
    if status != 0 {
        return Err(SpatialError::ImageError(format!(
            "Spatial HEIC write failed ({}) for {}",
            status,
            output_path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contains(haystack: &[u8], needle: &[u8]) -> bool {
        haystack.windows(needle.len()).any(|w| w == needle)
    }

    #[test]
    fn writes_heic_with_stereo_pair_group() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pair.heic");
        let left = DynamicImage::new_rgb8(64, 32);
        let right = DynamicImage::new_rgb8(64, 32);

        write_spatial_photo(&path, &left, &right, 90, &SpatialMetadata::default()).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert!(contains(&bytes[..32], b"ftypheic"));
        assert!(contains(&bytes, b"grpl"));
        assert!(contains(&bytes, b"ster"));
    }

    #[test]
    fn rejects_mismatched_eye_sizes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pair.heic");
        let left = DynamicImage::new_rgb8(64, 32);
        let right = DynamicImage::new_rgb8(32, 32);

        let err = write_spatial_photo(&path, &left, &right, 90, &SpatialMetadata::default());
        assert!(err.is_err());
        assert!(!path.exists());
    }
}
