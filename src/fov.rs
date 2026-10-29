use exif::{In, Reader, Tag, Value};
use std::path::Path;

pub const DEFAULT_HORIZONTAL_FOV_DEGREES: f64 = 65.0;
const MIN_HORIZONTAL_FOV_DEGREES: f64 = 40.0;
const MAX_HORIZONTAL_FOV_DEGREES: f64 = 85.0;
const FULL_FRAME_DIAGONAL_MM: f64 = 43.27;

pub fn horizontal_fov_from_focal_length_35mm(
    focal_length_35mm: f64,
    width: u32,
    height: u32,
) -> f64 {
    assert!(focal_length_35mm > 0.0);
    assert!(width > 0 && height > 0);
    let (w, h) = (width as f64, height as f64);
    let sensor_width_35mm = FULL_FRAME_DIAGONAL_MM * w / w.hypot(h);
    let degrees = 2.0
        * (sensor_width_35mm / (2.0 * focal_length_35mm))
            .atan()
            .to_degrees();
    degrees.clamp(MIN_HORIZONTAL_FOV_DEGREES, MAX_HORIZONTAL_FOV_DEGREES)
}

pub fn horizontal_fov_for_photo(path: &Path, width: u32, height: u32) -> f64 {
    read_focal_length_35mm(path).map_or(DEFAULT_HORIZONTAL_FOV_DEGREES, |f35| {
        horizontal_fov_from_focal_length_35mm(f35, width, height)
    })
}

fn read_focal_length_35mm(path: &Path) -> Option<f64> {
    let file = std::fs::File::open(path).ok()?;
    let exif = Reader::new()
        .read_from_container(&mut std::io::BufReader::new(file))
        .ok()?;
    let field = exif.get_field(Tag::FocalLengthIn35mmFilm, In::PRIMARY)?;
    let Value::Short(values) = &field.value else {
        return None;
    };
    values.first().filter(|&&mm| mm > 0).map(|&mm| mm as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iphone_main_camera_is_about_67_degrees() {
        let fov = horizontal_fov_from_focal_length_35mm(26.0, 4032, 3024);
        assert!((fov - 67.3).abs() < 0.1, "{fov}");
    }

    #[test]
    fn portrait_orientation_narrows_horizontal_fov() {
        let landscape = horizontal_fov_from_focal_length_35mm(26.0, 4032, 3024);
        let portrait = horizontal_fov_from_focal_length_35mm(26.0, 3024, 4032);
        assert!(portrait < landscape);
    }

    #[test]
    fn extreme_focal_lengths_are_clamped() {
        assert_eq!(
            horizontal_fov_from_focal_length_35mm(12.0, 4000, 3000),
            MAX_HORIZONTAL_FOV_DEGREES
        );
        assert_eq!(
            horizontal_fov_from_focal_length_35mm(200.0, 4000, 3000),
            MIN_HORIZONTAL_FOV_DEGREES
        );
    }

    #[test]
    fn missing_file_falls_back() {
        let fov = horizontal_fov_for_photo(Path::new("missing.jpg"), 4000, 3000);
        assert_eq!(fov, DEFAULT_HORIZONTAL_FOV_DEGREES);
    }

    #[test]
    fn photo_without_exif_falls_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("plain.png");
        image::DynamicImage::new_rgb8(8, 8).save(&path).unwrap();
        assert_eq!(
            horizontal_fov_for_photo(&path, 8, 8),
            DEFAULT_HORIZONTAL_FOV_DEGREES
        );
    }
}
