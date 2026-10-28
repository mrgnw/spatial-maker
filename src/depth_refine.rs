use image::RgbImage;
use rayon::prelude::*;

pub const DEFAULT_RADIUS_FRACTION: f32 = 1.0 / 256.0;
pub const DEFAULT_EPSILON: f32 = 3e-2;

// Depth models run at ~518 px, so upscaled depth edges follow a coarse grid and the warped
// silhouette comes out stair-stepped. A guided filter (He et al.) with the photo as guide snaps
// depth edges onto the photo's own edges at full resolution.
pub fn refine_depth_edges(
    image: &RgbImage,
    depth: &[f32],
    radius_fraction: f32,
    epsilon: f32,
) -> Vec<f32> {
    let (width, height) = (image.width() as usize, image.height() as usize);
    assert_eq!(depth.len(), width * height);
    assert!(radius_fraction > 0.0 && epsilon > 0.0);
    let radius = ((width.max(height) as f32 * radius_fraction).round() as usize).max(1);

    let guide: Vec<f32> = image
        .pixels()
        .map(|p| (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0)
        .collect();

    let mean_i = box_mean(&guide, width, height, radius);
    let mean_p = box_mean(depth, width, height, radius);
    let mean_ip = box_mean(&multiply(&guide, depth), width, height, radius);
    let mean_ii = box_mean(&multiply(&guide, &guide), width, height, radius);

    let (a, b): (Vec<f32>, Vec<f32>) = (0..width * height)
        .into_par_iter()
        .map(|i| {
            let covariance = mean_ip[i] - mean_i[i] * mean_p[i];
            let variance = mean_ii[i] - mean_i[i] * mean_i[i];
            let a = covariance / (variance + epsilon);
            (a, mean_p[i] - a * mean_i[i])
        })
        .unzip();

    let mean_a = box_mean(&a, width, height, radius);
    let mean_b = box_mean(&b, width, height, radius);
    (0..width * height)
        .into_par_iter()
        .map(|i| (mean_a[i] * guide[i] + mean_b[i]).clamp(0.0, 1.0))
        .collect()
}

pub const MEDIAN_TAP_RADIUS: usize = 3;
pub const MEDIAN_SIGMA_TAPS: f32 = 4.0;
pub const MEDIAN_SIGMA_COLOR: f32 = 0.1;
pub const MEDIAN_REFERENCE_LONG_SIDE: f32 = 1024.0;

// The guided filter turns a depth edge into a ramp, so background next to thin foreground gets
// in-between depth and its lines bend in the other eye. A joint weighted median (Shih et al. 2020)
// only ever outputs depth values from its window, weighted towards pixels of the same colour, so a
// ramp collapses to the two plateaus and the step lands on the photo's edge. The 7x7 taps are
// spread `stride` pixels apart so the window covers the same part of the scene at any resolution.
// One pass shortens a ramp by about half the window; a second pass finishes it.
pub fn median_depth_edges(
    image: &RgbImage,
    depth: &[f32],
    stride: usize,
    sigma_color: f32,
    passes: usize,
) -> Vec<f32> {
    let (width, height) = (image.width() as usize, image.height() as usize);
    assert_eq!(depth.len(), width * height);
    assert!(stride >= 1 && sigma_color > 0.0 && passes >= 1);

    let colors: Vec<[f32; 3]> = image
        .pixels()
        .map(|p| p.0.map(|c| c as f32 / 255.0))
        .collect();
    let side = 2 * MEDIAN_TAP_RADIUS + 1;
    let taps: Vec<(isize, isize, f32)> = (0..side * side)
        .map(|i| {
            let dy = (i / side) as isize - MEDIAN_TAP_RADIUS as isize;
            let dx = (i % side) as isize - MEDIAN_TAP_RADIUS as isize;
            let taps_squared = (dx * dx + dy * dy) as f32;
            let weight = (-taps_squared / (2.0 * MEDIAN_SIGMA_TAPS * MEDIAN_SIGMA_TAPS)).exp();
            (dx * stride as isize, dy * stride as isize, weight)
        })
        .collect();
    let falloff = -1.0 / (2.0 * sigma_color * sigma_color);

    let mut current = depth.to_vec();
    for _ in 0..passes {
        current = median_pass(&colors, &current, width, height, &taps, falloff);
    }
    current
}

fn median_pass(
    colors: &[[f32; 3]],
    depth: &[f32],
    width: usize,
    height: usize,
    taps: &[(isize, isize, f32)],
    falloff: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; width * height];
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, out_row)| {
            let mut samples = Vec::with_capacity(taps.len());
            for (x, out) in out_row.iter_mut().enumerate() {
                let centre = colors[y * width + x];
                samples.clear();
                for &(dx, dy, spatial) in taps {
                    let (sx, sy) = (x as isize + dx, y as isize + dy);
                    if sx < 0 || sy < 0 || sx >= width as isize || sy >= height as isize {
                        continue;
                    }
                    let i = sy as usize * width + sx as usize;
                    let distance_squared: f32 =
                        (0..3).map(|c| (colors[i][c] - centre[c]).powi(2)).sum();
                    samples.push((depth[i], spatial * (distance_squared * falloff).exp()));
                }
                *out = weighted_median(&mut samples);
            }
        });
    out
}

pub fn median_stride(width: usize, height: usize) -> usize {
    ((width.max(height) as f32 / MEDIAN_REFERENCE_LONG_SIDE).round() as usize).max(1)
}

fn weighted_median(samples: &mut [(f32, f32)]) -> f32 {
    assert!(!samples.is_empty());
    samples.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let half = samples.iter().map(|s| s.1).sum::<f32>() * 0.5;
    let mut cumulative = 0.0;
    for &(value, weight) in samples.iter() {
        cumulative += weight;
        if cumulative >= half {
            return value;
        }
    }
    samples[samples.len() - 1].0
}

// Mixed pixels on a silhouette carry foreground colour. Growing near depth by a pixel or two
// makes them move with the foreground instead of smearing onto the background (iw3
// edge_dilation).
pub fn dilate_foreground(depth: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    assert_eq!(depth.len(), width * height);
    let mut current = depth.to_vec();
    for _ in 0..radius {
        let previous = current;
        current = (0..width * height)
            .into_par_iter()
            .map(|i| {
                let (x, y) = (i % width, i / width);
                let mut near = previous[i];
                for ny in y.saturating_sub(1)..(y + 2).min(height) {
                    for nx in x.saturating_sub(1)..(x + 2).min(width) {
                        near = near.max(previous[ny * width + nx]);
                    }
                }
                near
            })
            .collect();
    }
    current
}

fn multiply(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.par_iter().zip(b.par_iter()).map(|(x, y)| x * y).collect()
}

fn box_mean(values: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let rows = box_mean_rows(values, width, radius);
    let columns = box_mean_rows(&transpose(&rows, width, height), height, radius);
    transpose(&columns, height, width)
}

fn box_mean_rows(values: &[f32], width: usize, radius: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; values.len()];
    out.par_chunks_mut(width)
        .zip(values.par_chunks(width))
        .for_each(|(out_row, row)| {
            let mut prefix = vec![0.0f64; width + 1];
            for (x, &v) in row.iter().enumerate() {
                prefix[x + 1] = prefix[x] + v as f64;
            }
            for (x, out) in out_row.iter_mut().enumerate() {
                let lo = x.saturating_sub(radius);
                let hi = (x + radius + 1).min(width);
                *out = ((prefix[hi] - prefix[lo]) / (hi - lo) as f64) as f32;
            }
        });
    out
}

fn transpose(values: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; values.len()];
    out.par_chunks_mut(height)
        .enumerate()
        .for_each(|(x, column)| {
            for (y, v) in column.iter_mut().enumerate() {
                *v = values[y * width + x];
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_mean_of_constant_is_constant() {
        let values = vec![0.25f32; 7 * 5];
        let mean = box_mean(&values, 7, 5, 2);
        assert!(mean.iter().all(|v| (v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn depth_edge_moves_onto_image_edge() {
        let (width, height) = (64usize, 16usize);
        let image = RgbImage::from_fn(width as u32, height as u32, |x, _| {
            if x < 32 {
                image::Rgb([255, 255, 255])
            } else {
                image::Rgb([0, 0, 0])
            }
        });
        let depth: Vec<f32> = (0..width * height)
            .map(|i| ((i % width) as f32 - 24.0).clamp(0.0, 16.0) / 16.0)
            .map(|d| 1.0 - d)
            .collect();

        let refined = refine_depth_edges(&image, &depth, 4.0 / 64.0, 1e-4);

        let row = &refined[8 * width..9 * width];
        let original_step = 1.0 / 16.0;
        let step_at_edge = row[31] - row[32];
        let step_beside_edge = row[29] - row[30];
        assert!(
            step_at_edge > 4.0 * original_step && step_at_edge > 4.0 * step_beside_edge,
            "edge not sharpened at the image edge: {:?}",
            &row[28..36]
        );
    }

    fn white_left_black_right(width: usize, height: usize, edge: usize) -> RgbImage {
        RgbImage::from_fn(width as u32, height as u32, |x, _| {
            if (x as usize) < edge {
                image::Rgb([255, 255, 255])
            } else {
                image::Rgb([0, 0, 0])
            }
        })
    }

    #[test]
    fn median_turns_ramp_into_step_on_image_edge() {
        let (width, height) = (64usize, 16usize);
        let image = white_left_black_right(width, height, 32);
        let ramp: Vec<f32> = (0..width * height)
            .map(|i| 1.0 - ((i % width) as f32 - 28.0).clamp(0.0, 8.0) / 8.0)
            .collect();

        let filtered = median_depth_edges(&image, &ramp, 2, 0.1, 2);

        let row = &filtered[8 * width..9 * width];
        assert!(
            row[..32].iter().all(|&d| d > 0.5),
            "near side: {:?}",
            &row[24..32]
        );
        assert!(
            row[32..].iter().all(|&d| d < 0.5),
            "far side: {:?}",
            &row[32..40]
        );
        assert_eq!(
            row[31] - row[32],
            row[30] - row[33],
            "edge is not a single step"
        );
    }

    #[test]
    fn median_keeps_thin_foreground_bar() {
        let (width, height) = (48usize, 16usize);
        let image = RgbImage::from_fn(width as u32, height as u32, |x, _| {
            if (20..23).contains(&x) {
                image::Rgb([255, 255, 255])
            } else {
                image::Rgb([0, 0, 0])
            }
        });
        let depth: Vec<f32> = (0..width * height)
            .map(|i| {
                if (20..23).contains(&(i % width)) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        let filtered = median_depth_edges(&image, &depth, 2, 0.1, 2);

        assert_eq!(filtered, depth);
    }

    #[test]
    fn dilation_grows_near_depth_by_radius() {
        let (width, height) = (9usize, 9usize);
        let mut depth = vec![0.0f32; width * height];
        depth[4 * width + 4] = 1.0;

        let dilated = dilate_foreground(&depth, width, height, 2);

        let near: Vec<usize> = (0..width)
            .filter(|&x| dilated[4 * width + x] == 1.0)
            .collect();
        assert_eq!(near, vec![2, 3, 4, 5, 6]);
    }
}
