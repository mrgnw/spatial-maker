use crate::depth_refine::{box_mean, multiply, refine_depth_edges};
use crate::error::{SpatialError, SpatialResult};
use crate::stereo::{
    depth_at_image_size, downsample_row, rasterize_row, sample_bilinear, shade_row, EdgeRefinement,
    HOLE, SUPERSAMPLE, TEAR_DISPARITY,
};
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};
use ndarray::Array2;
use rayon::prelude::*;

const MIN_ALPHA: f32 = 1.0 / 255.0;
const INTERIOR_ALPHA: f32 = 0.9;
const EDGE_RADIUS_FRACTION: f32 = 1.0 / 128.0;

#[cfg(all(feature = "coreml", target_os = "macos"))]
extern "C" {
    fn vision_foreground_matte(rgb: *const u8, width: i32, height: i32, output: *mut f32) -> i32;
}

// Soft matte of the photo's foreground subjects from Apple Vision, or None when it finds none.
#[cfg(all(feature = "coreml", target_os = "macos"))]
pub fn foreground_matte(image: &RgbImage) -> SpatialResult<Option<Vec<f32>>> {
    let (width, height) = (image.width() as usize, image.height() as usize);
    let mut alpha = vec![0.0f32; width * height];
    let instances = unsafe {
        vision_foreground_matte(
            image.as_raw().as_ptr(),
            width as i32,
            height as i32,
            alpha.as_mut_ptr(),
        )
    };
    match instances {
        0 => Ok(None),
        n if n > 0 => Ok(Some(alpha)),
        code => Err(SpatialError::ModelError(format!(
            "Vision foreground matte failed with code {}",
            code
        ))),
    }
}

// Two-layer right eye. The background layer is the photo with the matted subject cut out and
// the gap filled from the background; the subject layer is shifted by its own disparity and
// composited over it with its soft alpha, so hair and fur keep partial coverage instead of being
// torn at a depth step.
pub fn generate_stereo_pair_layered(
    image: &DynamicImage,
    depth: &Array2<f32>,
    max_disparity: u32,
    refinement: Option<EdgeRefinement>,
    alpha: &[f32],
) -> SpatialResult<(DynamicImage, DynamicImage)> {
    let src = image.to_rgb8();
    let (width, height) = (src.width() as usize, src.height() as usize);
    if width == 0 || height == 0 {
        return Err(SpatialError::ImageError(
            "Cannot make a stereo pair from an empty image".into(),
        ));
    }
    assert_eq!(alpha.len(), width * height);

    let depth = depth_at_image_size(depth, width, height);
    let depth = match refinement {
        Some(r) => refine_depth_edges(&src, &depth, r.radius_fraction, r.epsilon),
        None => depth,
    };
    let radius = ((width.max(height) as f32 * EDGE_RADIUS_FRACTION).round() as usize).max(1);
    let background_depth: Vec<f32> = depth
        .iter()
        .zip(alpha)
        .map(|(&d, &a)| if a < MIN_ALPHA { d } else { f32::NAN })
        .collect();
    let subject_depth = subject_depth(&depth, alpha, width, height, radius);
    let subject = estimate_foreground(&src, alpha, radius);

    let mut right = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width as u32, height as u32);
    let row_bytes = width * 3;
    right
        .as_mut()
        .par_chunks_mut(row_bytes)
        .enumerate()
        .for_each(|(y, out_row)| {
            let row = y * width..(y + 1) * width;
            render_layered_row(
                &src.as_raw()[y * row_bytes..(y + 1) * row_bytes],
                &subject.as_raw()[y * row_bytes..(y + 1) * row_bytes],
                &alpha[row.clone()],
                &background_depth[row.clone()],
                &subject_depth[row],
                max_disparity as f32,
                out_row,
            );
        });

    Ok((image.clone(), DynamicImage::ImageRgb8(right)))
}

// A depth map can't follow single hairs, so the soft edge of the matte takes the depth of the
// subject's solid interior nearby; it then moves with the head instead of with the background.
fn subject_depth(
    depth: &[f32],
    alpha: &[f32],
    width: usize,
    height: usize,
    radius: usize,
) -> Vec<f32> {
    let interior: Vec<f32> = alpha
        .iter()
        .map(|&a| if a >= INTERIOR_ALPHA { 1.0 } else { 0.0 })
        .collect();
    let interior_share = box_mean(&interior, width, height, radius);
    let interior_depth = box_mean(&multiply(depth, &interior), width, height, radius);
    (0..width * height)
        .into_par_iter()
        .map(|i| {
            if alpha[i] < MIN_ALPHA {
                f32::NAN
            } else if alpha[i] >= INTERIOR_ALPHA || interior_share[i] < 1e-3 {
                depth[i]
            } else {
                interior_depth[i] / interior_share[i]
            }
        })
        .collect()
}

// Edge pixels mix subject and old background. Blur-fusion foreground estimation (Forte & Pitié
// 2021) recovers the subject's own colour there, so the old background doesn't travel with it.
fn estimate_foreground(image: &RgbImage, alpha: &[f32], radius: usize) -> RgbImage {
    let (width, height) = (image.width() as usize, image.height() as usize);
    let channels: Vec<Vec<f32>> = (0..3)
        .map(|c| image.pixels().map(|p| p[c] as f32 / 255.0).collect())
        .collect();
    let (foreground, background) = blur_fusion(
        &channels, &channels, &channels, alpha, width, height, radius,
    );
    let (foreground, _) = blur_fusion(&channels, &foreground, &background, alpha, width, height, 3);
    RgbImage::from_fn(width as u32, height as u32, |x, y| {
        let i = y as usize * width + x as usize;
        Rgb([0, 1, 2].map(|c| (foreground[c][i] * 255.0).round() as u8))
    })
}

fn blur_fusion(
    image: &[Vec<f32>],
    foreground: &[Vec<f32>],
    background: &[Vec<f32>],
    alpha: &[f32],
    width: usize,
    height: usize,
    radius: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let blurred_alpha = box_mean(alpha, width, height, radius);
    let beta: Vec<f32> = alpha.iter().map(|a| 1.0 - a).collect();
    let mut new_foreground = Vec::with_capacity(3);
    let mut blurred_background = Vec::with_capacity(3);
    for c in 0..3 {
        let f_sum = box_mean(&multiply(&foreground[c], alpha), width, height, radius);
        let b_sum = box_mean(&multiply(&background[c], &beta), width, height, radius);
        let f_blur: Vec<f32> = (0..alpha.len())
            .map(|i| f_sum[i] / (blurred_alpha[i] + 1e-5))
            .collect();
        let b_blur: Vec<f32> = (0..alpha.len())
            .map(|i| b_sum[i] / (1.0 - blurred_alpha[i] + 1e-5))
            .collect();
        let f: Vec<f32> = (0..alpha.len())
            .map(|i| {
                let a = alpha[i];
                let residual = image[c][i] - a * f_blur[i] - (1.0 - a) * b_blur[i];
                (f_blur[i] + a * residual).clamp(0.0, 1.0)
            })
            .collect();
        new_foreground.push(f);
        blurred_background.push(b_blur);
    }
    (new_foreground, blurred_background)
}

fn render_layered_row(
    src: &[u8],
    subject: &[u8],
    alpha: &[f32],
    background_depth: &[f32],
    subject_depth: &[f32],
    max_disparity: f32,
    out: &mut [u8],
) {
    let (background_disparity, source_x) = rasterize_row(background_depth, max_disparity);
    let background = shade_row(src, &background_disparity, &source_x);
    let (mut disparity, mut source_x) = rasterize_row(subject_depth, max_disparity);
    fill_subject_tears(alpha, &mut disparity, &mut source_x);

    let composite: Vec<[f32; 3]> = background
        .iter()
        .enumerate()
        .map(|(j, back)| {
            if disparity[j] == HOLE || background_disparity[j] > disparity[j] + TEAR_DISPARITY {
                return *back;
            }
            let front = sample_bilinear(subject, source_x[j]);
            let a = sample_linear(alpha, source_x[j]);
            [0, 1, 2].map(|c| front[c] * a + back[c] * (1.0 - a))
        })
        .collect();
    downsample_row(&composite, out);
}

// A gap in the subject layer is either background seen between parts of the subject (an arm and
// the body) or a tear at a depth step inside the subject. Only a tear has subject pixels all the
// way across in the source; fill those from the farther side so the background doesn't show
// through the subject.
fn fill_subject_tears(alpha: &[f32], disparity: &mut [f32], source_x: &mut [f32]) {
    let samples = disparity.len();
    let mut j = 0;
    while j < samples {
        if disparity[j] != HOLE {
            j += 1;
            continue;
        }
        let end = (j..samples)
            .find(|&i| disparity[i] != HOLE)
            .unwrap_or(samples);
        if j > 0 && end < samples {
            let (left, right) = (j - 1, end);
            let (a, b) = (source_x[left] as usize, source_x[right] as usize);
            let covered = alpha[a.min(b)..=a.max(b)].iter().all(|&v| v >= MIN_ALPHA);
            if covered {
                let far = if disparity[left] < disparity[right] {
                    left
                } else {
                    right
                };
                for k in j..end {
                    let distance = (k as f32 - far as f32).abs() / SUPERSAMPLE as f32;
                    disparity[k] = disparity[far];
                    source_x[k] = if far == right {
                        source_x[far] + distance
                    } else {
                        source_x[far] - distance
                    };
                }
            }
        }
        j = end;
    }
}

fn sample_linear(row: &[f32], x: f32) -> f32 {
    let x = x.clamp(0.0, (row.len() - 1) as f32);
    let x0 = x.floor() as usize;
    let x1 = (x0 + 1).min(row.len() - 1);
    row[x0] + (row[x1] - row[x0]) * (x - x0 as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUBJECT: [u8; 3] = [250, 20, 20];
    const BACKGROUND: [u8; 3] = [20, 20, 250];

    // Background at depth 0 with a subject at depth 1 over columns 40..60, whose last column is
    // a half-covered hair pixel: half subject colour, half background, alpha 0.5.
    fn hair_scene() -> (Vec<u8>, Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let width = 100;
        let mut src = Vec::new();
        let mut subject = Vec::new();
        let mut alpha = Vec::new();
        for x in 0..width {
            let a = match x {
                40..=58 => 1.0,
                59 => 0.5,
                _ => 0.0,
            };
            let mixed = [0, 1, 2]
                .map(|c| (SUBJECT[c] as f32 * a + BACKGROUND[c] as f32 * (1.0 - a)).round() as u8);
            src.extend_from_slice(&mixed);
            subject.extend_from_slice(if a > 0.0 { &SUBJECT } else { &BACKGROUND });
            alpha.push(a);
        }
        let background_depth = alpha
            .iter()
            .map(|&a| if a < MIN_ALPHA { 0.0 } else { f32::NAN })
            .collect();
        let subject_depth = alpha
            .iter()
            .map(|&a| if a < MIN_ALPHA { f32::NAN } else { 1.0 })
            .collect();
        (src, subject, alpha, background_depth, subject_depth)
    }

    #[test]
    fn soft_edge_moves_with_subject_and_keeps_partial_coverage() {
        let (src, subject, alpha, background_depth, subject_depth) = hair_scene();
        let mut out = vec![0u8; src.len()];
        render_layered_row(
            &src,
            &subject,
            &alpha,
            &background_depth,
            &subject_depth,
            10.0,
            &mut out,
        );

        let pixel = |x: usize| [out[x * 3], out[x * 3 + 1], out[x * 3 + 2]];
        assert_eq!(pixel(45), SUBJECT);
        assert_eq!(
            pixel(49),
            [135, 20, 135],
            "hair pixel shifted by its subject's disparity"
        );
        for x in 50..70 {
            assert_eq!(pixel(x), BACKGROUND, "column {} should show background", x);
        }
    }

    #[test]
    fn gap_between_subject_parts_shows_background() {
        let mut alpha = vec![0.0f32; 20];
        alpha[2..6].fill(1.0);
        alpha[10..14].fill(1.0);
        let depth: Vec<f32> = alpha
            .iter()
            .map(|&a| if a > 0.0 { 1.0 } else { f32::NAN })
            .collect();
        let (mut disparity, mut source_x) = rasterize_row(&depth, 3.0);
        let holes_before = disparity.iter().filter(|&&d| d == HOLE).count();
        fill_subject_tears(&alpha, &mut disparity, &mut source_x);
        assert_eq!(
            disparity.iter().filter(|&&d| d == HOLE).count(),
            holes_before
        );
    }

    #[test]
    fn tear_inside_subject_is_filled_from_the_farther_part() {
        let alpha = vec![1.0f32; 20];
        let depth: Vec<f32> = (0..20).map(|x| if x < 10 { 1.0 } else { 0.0 }).collect();
        let (mut disparity, mut source_x) = rasterize_row(&depth, 3.0);
        assert!(disparity.iter().any(|&d| d == HOLE));
        fill_subject_tears(&alpha, &mut disparity, &mut source_x);
        assert!(disparity.iter().all(|&d| d != HOLE));
    }

    #[test]
    fn unmatted_pixels_nearer_than_the_subject_stay_in_front() {
        const NEAR: [u8; 3] = [20, 250, 20];
        let width = 100;
        let alpha: Vec<f32> = (0..width)
            .map(|x| if (40..60).contains(&x) { 1.0 } else { 0.0 })
            .collect();
        let near = |x: usize| (60..70).contains(&x);
        let src: Vec<u8> = (0..width)
            .flat_map(|x| match x {
                40..=59 => SUBJECT,
                _ if near(x) => NEAR,
                _ => BACKGROUND,
            })
            .collect();
        let background_depth: Vec<f32> = (0..width)
            .map(|x| match x {
                40..=59 => f32::NAN,
                _ if near(x) => 1.0,
                _ => 0.0,
            })
            .collect();
        let subject_depth: Vec<f32> = alpha
            .iter()
            .map(|&a| if a > 0.0 { 0.2 } else { f32::NAN })
            .collect();
        let mut out = vec![0u8; src.len()];
        render_layered_row(
            &src,
            &src,
            &alpha,
            &background_depth,
            &subject_depth,
            10.0,
            &mut out,
        );

        let pixel = |x: usize| [out[x * 3], out[x * 3 + 1], out[x * 3 + 2]];
        assert_eq!(pixel(45), SUBJECT);
        assert_eq!(
            pixel(53),
            NEAR,
            "nearer background pixel hidden by the subject"
        );
    }
}
