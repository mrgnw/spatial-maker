use crate::depth_refine::{refine_depth_edges, DEFAULT_EPSILON, DEFAULT_RADIUS_FRACTION};
use crate::error::{SpatialError, SpatialResult};
use image::{DynamicImage, ImageBuffer, Luma, Rgb};
use ndarray::Array2;
use rayon::prelude::*;

const HOLE: f32 = f32::NEG_INFINITY;

pub fn generate_stereo_pair(
    image: &DynamicImage,
    depth: &Array2<f32>,
    max_disparity: u32,
) -> SpatialResult<(DynamicImage, DynamicImage)> {
    generate_stereo_pair_with_progress(image, depth, max_disparity, None::<fn(f64)>)
}

pub fn generate_stereo_pair_with_progress<F>(
    image: &DynamicImage,
    depth: &Array2<f32>,
    max_disparity: u32,
    mut progress_callback: Option<F>,
) -> SpatialResult<(DynamicImage, DynamicImage)>
where
    F: FnMut(f64),
{
    let pair = generate_stereo_pair_with_refinement(
        image,
        depth,
        max_disparity,
        Some(EdgeRefinement::default()),
    )?;
    if let Some(ref mut cb) = progress_callback {
        cb(100.0);
    }
    Ok(pair)
}

#[derive(Clone, Copy, Debug)]
pub struct EdgeRefinement {
    pub radius_fraction: f32,
    pub epsilon: f32,
}

impl Default for EdgeRefinement {
    fn default() -> Self {
        Self {
            radius_fraction: DEFAULT_RADIUS_FRACTION,
            epsilon: DEFAULT_EPSILON,
        }
    }
}

pub fn generate_stereo_pair_with_refinement(
    image: &DynamicImage,
    depth: &Array2<f32>,
    max_disparity: u32,
    refinement: Option<EdgeRefinement>,
) -> SpatialResult<(DynamicImage, DynamicImage)> {
    let src = image.to_rgb8();
    let (width, height) = (src.width() as usize, src.height() as usize);
    if width == 0 || height == 0 {
        return Err(SpatialError::ImageError(
            "Cannot make a stereo pair from an empty image".into(),
        ));
    }

    let depth = depth_at_image_size(depth, width, height);
    let depth = match refinement {
        Some(r) => refine_depth_edges(&src, &depth, r.radius_fraction, r.epsilon),
        None => depth,
    };
    let mut right = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width as u32, height as u32);
    let row_bytes = width * 3;

    right
        .as_mut()
        .par_chunks_mut(row_bytes)
        .zip(src.as_raw().par_chunks(row_bytes))
        .zip(depth.par_chunks(width))
        .for_each(|((out_row, src_row), depth_row)| {
            render_right_row(src_row, depth_row, max_disparity as f32, out_row);
        });

    Ok((image.clone(), DynamicImage::ImageRgb8(right)))
}

fn depth_at_image_size(depth: &Array2<f32>, width: usize, height: usize) -> Vec<f32> {
    let (depth_height, depth_width) = depth.dim();
    let values: Vec<f32> = depth.iter().copied().collect();
    if depth_width == width && depth_height == height {
        return values;
    }
    let buffer = ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(
        depth_width as u32,
        depth_height as u32,
        values,
    )
    .expect("depth buffer matches its dimensions");
    image::imageops::resize(
        &buffer,
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    )
    .into_raw()
}

// Right eye from the left eye. Near pixels (depth 1) shift left by up to max_disparity, so the
// background they uncover sits to their right.
fn render_right_row(src: &[u8], depth: &[f32], max_disparity: f32, out: &mut [u8]) {
    let width = depth.len();
    assert_eq!(src.len(), width * 3);
    assert_eq!(out.len(), width * 3);

    let target_disparity = warp_disparity(depth, max_disparity);

    let mut x = 0;
    while x < width {
        if target_disparity[x] != HOLE {
            let pixel = sample_bilinear(src, x as f32 + target_disparity[x]);
            out[x * 3..x * 3 + 3].copy_from_slice(&pixel);
            x += 1;
            continue;
        }
        let hole_end = (x..width)
            .find(|&i| target_disparity[i] != HOLE)
            .unwrap_or(width);
        fill_hole(src, &target_disparity, x, hole_end, out);
        x = hole_end;
    }
}

// Forward-warp each source pixel's disparity to its right-eye position, keeping the nearest.
// Splatting onto both neighbouring columns closes the 1 px cracks left by stretched surfaces.
fn warp_disparity(depth: &[f32], max_disparity: f32) -> Vec<f32> {
    let width = depth.len();
    let mut target = vec![HOLE; width];
    for (x, &d) in depth.iter().enumerate() {
        let disparity = d.clamp(0.0, 1.0) * max_disparity;
        let base = (x as f32 - disparity).floor() as i64;
        for t in [base, base + 1] {
            if t >= 0 && (t as usize) < width && disparity > target[t as usize] {
                target[t as usize] = disparity;
            }
        }
    }
    target
}

// Holes are background the left eye never saw. Mirror the background texture next to the hole
// into it instead of repeating the foreground edge.
fn fill_hole(src: &[u8], target_disparity: &[f32], start: usize, end: usize, out: &mut [u8]) {
    let width = target_disparity.len();
    let left = start.checked_sub(1);
    let right = (end < width).then_some(end);
    let background = match (left, right) {
        (Some(l), Some(r)) if target_disparity[l] < target_disparity[r] => l,
        (_, Some(r)) => r,
        (Some(l), None) => l,
        (None, None) => return,
    };
    let anchor = background as f32 + target_disparity[background];
    for x in start..end {
        let distance = (x as f32 - background as f32).abs();
        let source_x = if background >= end {
            anchor + distance
        } else {
            anchor - distance
        };
        let pixel = sample_bilinear(src, source_x);
        out[x * 3..x * 3 + 3].copy_from_slice(&pixel);
    }
}

fn sample_bilinear(row: &[u8], x: f32) -> [u8; 3] {
    let width = row.len() / 3;
    let x = x.clamp(0.0, (width - 1) as f32);
    let x0 = x.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let t = x - x0 as f32;
    let mut pixel = [0u8; 3];
    for c in 0..3 {
        let a = row[x0 * 3 + c] as f32;
        let b = row[x1 * 3 + c] as f32;
        pixel[c] = (a + (b - a) * t).round() as u8;
    }
    pixel
}

#[cfg(test)]
mod tests {
    use super::*;

    const FOREGROUND: [u8; 3] = [250, 20, 20];
    const BACKGROUND: [u8; 3] = [20, 20, 250];

    fn block_scene(width: usize) -> (Vec<u8>, Vec<f32>) {
        let mut src = Vec::with_capacity(width * 3);
        let mut depth = Vec::with_capacity(width);
        for x in 0..width {
            let near = (40..60).contains(&x);
            src.extend_from_slice(if near { &FOREGROUND } else { &BACKGROUND });
            depth.push(if near { 1.0 } else { 0.0 });
        }
        (src, depth)
    }

    #[test]
    fn flat_depth_leaves_row_unchanged() {
        let (src, _) = block_scene(100);
        let mut out = vec![0u8; src.len()];
        render_right_row(&src, &vec![0.0; 100], 10.0, &mut out);
        assert_eq!(out, src);
    }

    #[test]
    fn uncovered_background_is_not_filled_with_foreground() {
        let (src, depth) = block_scene(100);
        let mut out = vec![0u8; src.len()];
        render_right_row(&src, &depth, 10.0, &mut out);

        let pixel = |x: usize| [out[x * 3], out[x * 3 + 1], out[x * 3 + 2]];
        assert_eq!(pixel(35), FOREGROUND);
        assert_eq!(pixel(45), FOREGROUND);
        for x in 52..60 {
            assert_eq!(pixel(x), BACKGROUND, "column {} should show background", x);
        }
    }

    #[test]
    fn pair_keeps_image_size() {
        let image = DynamicImage::new_rgb8(64, 32);
        let depth = Array2::<f32>::from_elem((16, 32), 0.5);
        let (left, right) = generate_stereo_pair(&image, &depth, 8).unwrap();
        assert_eq!((left.width(), left.height()), (64, 32));
        assert_eq!((right.width(), right.height()), (64, 32));
    }
}
