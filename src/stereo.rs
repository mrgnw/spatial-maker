use crate::depth_refine::{
    dilate_foreground, median_depth_edges, median_stride, refine_depth_edges, DEFAULT_EPSILON,
    DEFAULT_RADIUS_FRACTION, MEDIAN_SIGMA_COLOR,
};
use crate::error::{SpatialError, SpatialResult};
use image::{DynamicImage, ImageBuffer, Luma, Rgb};
use ndarray::Array2;
use rayon::prelude::*;

const HOLE: f32 = f32::NEG_INFINITY;
const SUPERSAMPLE: usize = 4;
const TEAR_DISPARITY: f32 = 1.0;
const HOLE_SMOOTH_RADIUS: usize = 4;

pub fn generate_stereo_pair(
    image: &DynamicImage,
    depth: &Array2<f32>,
    max_disparity: u32,
    refinement: EdgeRefinement,
) -> SpatialResult<(DynamicImage, DynamicImage)> {
    generate_stereo_pair_with_refinement(image, depth, max_disparity, Some(refinement))
}

pub fn generate_stereo_pair_with_progress<F>(
    image: &DynamicImage,
    depth: &Array2<f32>,
    max_disparity: u32,
    refinement: EdgeRefinement,
    mut progress_callback: Option<F>,
) -> SpatialResult<(DynamicImage, DynamicImage)>
where
    F: FnMut(f64),
{
    let pair = generate_stereo_pair(image, depth, max_disparity, refinement)?;
    if let Some(ref mut cb) = progress_callback {
        cb(100.0);
    }
    Ok(pair)
}

// How depth edges are moved onto the photo's edges before warping.
#[derive(Clone, Copy, Debug)]
pub enum EdgeRefinement {
    // Guided filter: cheap enough per video frame, but leaves edges as short ramps, so
    // background lines next to thin foreground can bend.
    Guided {
        radius_fraction: f32,
        epsilon: f32,
    },
    // Joint weighted median on the raw depth, then foreground dilation: edges become steps on
    // the photo's edges. ~5x the guided filter's cost, so photos only.
    Median {
        sigma_color: f32,
        passes: usize,
        dilate_px: usize,
    },
}

impl EdgeRefinement {
    pub fn for_photo() -> Self {
        Self::Median {
            sigma_color: MEDIAN_SIGMA_COLOR,
            passes: 2,
            dilate_px: 1,
        }
    }

    pub fn for_video() -> Self {
        Self::Guided {
            radius_fraction: DEFAULT_RADIUS_FRACTION,
            epsilon: DEFAULT_EPSILON,
        }
    }

    fn apply(self, src: &image::RgbImage, depth: &[f32]) -> Vec<f32> {
        let (width, height) = (src.width() as usize, src.height() as usize);
        match self {
            Self::Guided {
                radius_fraction,
                epsilon,
            } => refine_depth_edges(src, depth, radius_fraction, epsilon),
            Self::Median {
                sigma_color,
                passes,
                dilate_px,
            } => {
                let stride = median_stride(width, height);
                let median = median_depth_edges(src, depth, stride, sigma_color, passes);
                dilate_foreground(&median, width, height, dilate_px)
            }
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
        Some(r) => r.apply(&src, &depth),
        None => depth,
    };
    let mut right = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width as u32, height as u32);
    let row_bytes = width * 3;
    let mut holes = vec![0u8; width * height];

    right
        .as_mut()
        .par_chunks_mut(row_bytes)
        .zip(holes.par_chunks_mut(width))
        .zip(src.as_raw().par_chunks(row_bytes))
        .zip(depth.par_chunks(width))
        .for_each(|(((out_row, hole_row), src_row), depth_row)| {
            render_right_row(src_row, depth_row, max_disparity as f32, out_row, hole_row);
        });
    smooth_holes_vertically(right.as_mut(), &holes, width, HOLE_SMOOTH_RADIUS);

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
//
// `holes` gets, per pixel, how many of its SUPERSAMPLE samples were filled in rather than seen.
fn render_right_row(
    src: &[u8],
    depth: &[f32],
    max_disparity: f32,
    out: &mut [u8],
    holes: &mut [u8],
) {
    let width = depth.len();
    assert_eq!(src.len(), width * 3);
    assert_eq!(out.len(), width * 3);
    assert_eq!(holes.len(), width);

    let (disparity, source_x) = rasterize_row(depth, max_disparity);
    let colors = shade_row(src, &disparity, &source_x);

    for (x, pixel) in out.chunks_exact_mut(3).enumerate() {
        holes[x] = disparity[x * SUPERSAMPLE..(x + 1) * SUPERSAMPLE]
            .iter()
            .filter(|&&d| d == HOLE)
            .count() as u8;
        let samples = &colors[x * SUPERSAMPLE..(x + 1) * SUPERSAMPLE];
        for c in 0..3 {
            let sum: f32 = samples.iter().map(|s| s[c]).sum();
            pixel[c] = (sum / SUPERSAMPLE as f32).round() as u8;
        }
    }
}

// Rows fill a hole independently, so the mirrored texture changes from row to row. Average each
// fully filled pixel with the filled pixels above and below it. Pixels that show any real source
// stay untouched, so edges outside holes keep their sharpness.
fn smooth_holes_vertically(rgb: &mut [u8], holes: &[u8], width: usize, radius: usize) {
    let height = holes.len() / width;
    let source = rgb.to_vec();
    let is_hole = |x: usize, y: usize| holes[y * width + x] == SUPERSAMPLE as u8;
    rgb.par_chunks_mut(width * 3)
        .enumerate()
        .for_each(|(y, out_row)| {
            for x in (0..width).filter(|&x| is_hole(x, y)) {
                let mut sum = [0u32; 3];
                let mut count = 0u32;
                for yy in y.saturating_sub(radius)..(y + radius + 1).min(height) {
                    if !is_hole(x, yy) {
                        continue;
                    }
                    for c in 0..3 {
                        sum[c] += source[(yy * width + x) * 3 + c] as u32;
                    }
                    count += 1;
                }
                for c in 0..3 {
                    out_row[x * 3 + c] = ((sum[c] + count / 2) / count) as u8;
                }
            }
        });
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
        render_right_row(&src, &vec![0.0; 100], 10.0, &mut out, &mut vec![0; 100]);
        assert_eq!(out, src);
    }

    #[test]
    fn uncovered_background_is_not_filled_with_foreground() {
        let (src, depth) = block_scene(100);
        let mut out = vec![0u8; src.len()];
        render_right_row(&src, &depth, 10.0, &mut out, &mut vec![0; 100]);

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
        render_right_row(&src, &depth, 5.0, &mut out, &mut vec![0; 100]);

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
    fn wide_hole_fill_varies_smoothly_between_rows() {
        let (width, height) = (80usize, 40usize);
        let mut image = DynamicImage::new_rgb8(width as u32, height as u32).to_rgb8();
        let mut depth = Array2::<f32>::zeros((height, width));
        for y in 0..height {
            for x in 0..width {
                let near = (30..40).contains(&x);
                let noise = ((x * 7919 + y * 104729) % 97) as u8 * 2;
                let value = if near { 250 } else { 20 + noise };
                image.put_pixel(x as u32, y as u32, Rgb([value; 3]));
                depth[[y, x]] = if near { 1.0 } else { 0.0 };
            }
        }

        let (_, right) = generate_stereo_pair_with_refinement(
            &DynamicImage::ImageRgb8(image.clone()),
            &depth,
            10,
            None,
        )
        .unwrap();
        let right = right.to_rgb8();

        let row_to_row = |pixels: &dyn Fn(usize, usize) -> u8| -> f32 {
            let mut total = 0.0;
            let mut count = 0.0;
            for y in 10..height - 11 {
                for x in 32..38 {
                    total += (pixels(x, y) as f32 - pixels(x, y + 1) as f32).abs();
                    count += 1.0;
                }
            }
            total / count
        };
        let mut unsmoothed = vec![0u8; width * 3 * height];
        for y in 0..height {
            let depth_row: Vec<f32> = (0..width).map(|x| depth[[y, x]]).collect();
            let src_row = &image.as_raw()[y * width * 3..(y + 1) * width * 3];
            let out_row = &mut unsmoothed[y * width * 3..(y + 1) * width * 3];
            render_right_row(src_row, &depth_row, 10.0, out_row, &mut vec![0; width]);
        }

        let before = row_to_row(&|x, y| unsmoothed[(y * width + x) * 3]);
        let after = row_to_row(&|x, y| right.get_pixel(x as u32, y as u32)[0]);
        assert!(
            after < before / 3.0,
            "fill should vary less between rows, before {} after {}",
            before,
            after
        );

        for y in 0..height {
            for x in (0..28).chain(42..width) {
                assert_eq!(
                    right.get_pixel(x as u32, y as u32)[0],
                    unsmoothed[(y * width + x) * 3],
                    "pixel ({}, {}) is outside the hole and must not change",
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn pair_keeps_image_size() {
        let image = DynamicImage::new_rgb8(64, 32);
        let depth = Array2::<f32>::from_elem((16, 32), 0.5);
        let (left, right) =
            generate_stereo_pair(&image, &depth, 8, EdgeRefinement::for_video()).unwrap();
        assert_eq!((left.width(), left.height()), (64, 32));
        assert_eq!((right.width(), right.height()), (64, 32));
    }
}
