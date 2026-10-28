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

pub(crate) fn multiply(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.par_iter().zip(b.par_iter()).map(|(x, y)| x * y).collect()
}

pub(crate) fn box_mean(values: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
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
}
