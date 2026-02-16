use crate::error::SpatialResult;
use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::Array2;

pub fn generate_stereo_pair(
	image: &DynamicImage,
	depth: &Array2<f32>,
	max_disparity: u32,
) -> SpatialResult<(DynamicImage, DynamicImage)> {
	let img_rgb = image.to_rgb8();
	let width = img_rgb.width() as usize;
	let height = img_rgb.height() as usize;

	let mut right_rgb: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(width as u32, height as u32);
	let mut depth_buffer = vec![f32::NEG_INFINITY; width * height];
	let mut filled = vec![false; width * height];

	for y in 0..height {
		for x in 0..width {
			let depth_val = get_depth_at(depth, x, y, width, height);
			let disparity = (depth_val * max_disparity as f32).round() as i32;
			let x_right = x as i32 - disparity;

			if x_right >= 0 && x_right < width as i32 {
				let idx = y * width + x_right as usize;
				if depth_val > depth_buffer[idx] {
					depth_buffer[idx] = depth_val;
					filled[idx] = true;
					if let Some(pixel) = img_rgb.get_pixel_checked(x as u32, y as u32) {
						right_rgb.put_pixel(x_right as u32, y as u32, *pixel);
					}
				}
			}
		}
	}

	fill_disocclusions(&mut right_rgb, &filled, width, height);

	let left_image = image.clone();
	let right_image = DynamicImage::ImageRgb8(right_rgb);

	Ok((left_image, right_image))
}

fn get_depth_at(
	depth: &Array2<f32>,
	x: usize,
	y: usize,
	img_width: usize,
	img_height: usize,
) -> f32 {
	let (depth_height, depth_width) = depth.dim();

	if depth_height == img_height && depth_width == img_width {
		depth[[y, x]]
	} else {
		let scaled_x = (x as f32 * depth_width as f32 / img_width as f32)
			.min(depth_width as f32 - 1.0) as usize;
		let scaled_y = (y as f32 * depth_height as f32 / img_height as f32)
			.min(depth_height as f32 - 1.0) as usize;

		if scaled_y < depth_height && scaled_x < depth_width {
			depth[[scaled_y, scaled_x]]
		} else {
			0.5
		}
	}
}

fn fill_disocclusions(
	image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
	filled: &[bool],
	width: usize,
	height: usize,
) {
	let original = image.clone();

	for y in 0..height {
		for x in 0..width {
			if filled[y * width + x] {
				continue;
			}
			// scan left to find nearest filled pixel on same row
			let mut left_pixel = None;
			for lx in (0..x).rev() {
				if filled[y * width + lx] {
					left_pixel = Some(*original.get_pixel(lx as u32, y as u32));
					break;
				}
			}
			// scan right
			let mut right_pixel = None;
			for rx in (x + 1)..width {
				if filled[y * width + rx] {
					right_pixel = Some(*original.get_pixel(rx as u32, y as u32));
					break;
				}
			}

			let fill = match (left_pixel, right_pixel) {
				(Some(l), Some(_)) => l, // prefer left (background side in DIBR)
				(Some(l), None) => l,
				(None, Some(r)) => r,
				(None, None) => continue,
			};
			image.put_pixel(x as u32, y as u32, fill);
		}
	}
}
