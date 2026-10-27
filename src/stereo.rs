use crate::depth_refine::{refine_depth_edges, DEFAULT_EPSILON, DEFAULT_RADIUS_FRACTION};
use crate::error::{SpatialError, SpatialResult};
use image::{DynamicImage, ImageBuffer, Luma, Rgb};
use ndarray::Array2;
use rayon::prelude::*;

const HOLE: f32 = f32::NEG_INFINITY;
const SUPERSAMPLE: usize = 4;
const TEAR_DISPARITY: f32 = 1.0;

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
//
// The row is rendered at SUPERSAMPLE x horizontal resolution and averaged back down, so a
// silhouette that lands between pixels gives that pixel partial coverage instead of a whole-pixel
// step. That is what removes the staircase along slanted edges.
fn render_right_row(src: &[u8], depth: &[f32], max_disparity: f32, out: &mut [u8]) {
    let width = depth.len();
    assert_eq!(src.len(), width * 3);
    assert_eq!(out.len(), width * 3);

    let (disparity, source_x) = rasterize_row(depth, max_disparity);
    let colors = shade_row(src, &disparity, &source_x);

    for (x, pixel) in out.chunks_exact_mut(3).enumerate() {
        let samples = &colors[x * SUPERSAMPLE..(x + 1) * SUPERSAMPLE];
        for c in 0..3 {
            let sum: f32 = samples.iter().map(|s| s[c]).sum();
            pixel[c] = (sum / SUPERSAMPLE as f32).round() as u8;
        }
    }
}

// Each source pixel covers one pixel of width around its shifted centre. When its right
// neighbour is on the same surface (disparity within TEAR_DISPARITY), the footprint stretches to
// meet it so slopes don't crack; across a larger jump it tears and leaves a hole. Overlaps keep
// the nearest surface.
fn rasterize_row(depth: &[f32], max_disparity: f32) -> (Vec<f32>, Vec<f32>) {
    let width = depth.len();
    let samples = width * SUPERSAMPLE;
    let scale = SUPERSAMPLE as f32;
    let mut disparity = vec![HOLE; samples];
    let mut source_x = vec![0.0f32; samples];

    let disparity_at = |x: usize| depth[x].clamp(0.0, 1.0) * max_disparity;
    for x in 0..width {
        let d = disparity_at(x);
        let centre = (x as f32 + 0.5 - d) * scale;
        let start = centre - scale * 0.5;
        let mut end = centre + scale * 0.5;
        if x + 1 < width {
            let next = disparity_at(x + 1);
            if (next - d).abs() < TEAR_DISPARITY {
                end = end.max((x as f32 + 1.5 - next) * scale - scale * 0.5);
            }
        }
        let first = (start - 0.5).ceil().max(0.0) as usize;
        let last = ((end - 0.5).ceil() as i64).min(samples as i64);
        for j in first..last.max(first as i64) as usize {
            if d > disparity[j] {
                disparity[j] = d;
                source_x[j] = x as f32;
            }
        }
    }
    (disparity, source_x)
}

fn shade_row(src: &[u8], disparity: &[f32], source_x: &[f32]) -> Vec<[f32; 3]> {
    let samples = disparity.len();
    let mut colors = vec![[0.0f32; 3]; samples];
    let mut j = 0;
    while j < samples {
        if disparity[j] != HOLE {
            colors[j] = sample_bilinear(src, source_x[j]);
            j += 1;
            continue;
        }
        let end = (j..samples)
            .find(|&i| disparity[i] != HOLE)
            .unwrap_or(samples);
        fill_hole(src, disparity, source_x, j, end, &mut colors);
        j = end;
    }
    colors
}

// Holes are background the left eye never saw. Mirror the background texture next to the hole
// into it instead of repeating the foreground edge.
fn fill_hole(
    src: &[u8],
    disparity: &[f32],
    source_x: &[f32],
    start: usize,
    end: usize,
    colors: &mut [[f32; 3]],
) {
    let samples = disparity.len();
    let left = start.checked_sub(1);
    let right = (end < samples).then_some(end);
    let background = match (left, right) {
        (Some(l), Some(r)) if disparity[l] < disparity[r] => l,
        (_, Some(r)) => r,
        (Some(l), None) => l,
        (None, None) => return,
    };
    let anchor = source_x[background];
    for j in start..end {
        let distance = (j as f32 - background as f32).abs() / SUPERSAMPLE as f32;
        let x = if background >= end {
            anchor + distance
        } else {
            anchor - distance
        };
        colors[j] = sample_bilinear(src, x);
    }
}

fn sample_bilinear(row: &[u8], x: f32) -> [f32; 3] {
    let width = row.len() / 3;
    let x = x.clamp(0.0, (width - 1) as f32);
    let x0 = x.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let t = x - x0 as f32;
    let mut pixel = [0.0f32; 3];
    for c in 0..3 {
        let a = row[x0 * 3 + c] as f32;
        let b = row[x1 * 3 + c] as f32;
        pixel[c] = a + (b - a) * t;
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
    fn half_pixel_silhouette_blends_edge_pixels() {
        let (src, depth) = block_scene(100);
        let depth: Vec<f32> = depth.iter().map(|d| d * 0.5).collect();
        let mut out = vec![0u8; src.len()];
        render_right_row(&src, &depth, 5.0, &mut out);

        for x in [37, 57] {
            let red = out[x * 3];
            assert!(
                red > BACKGROUND[0] + 50 && red < FOREGROUND[0] - 50,
                "column {} should be a foreground/background blend, got {:?}",
                x,
                &out[x * 3..x * 3 + 3]
            );
        }
        assert_eq!(&out[45 * 3..45 * 3 + 3], &FOREGROUND);
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
