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

	let bg = Rgb([64u8, 64u8, 64u8]);
	for pixel in right_rgb.pixels_mut() {
		*pixel = bg;
	}

	for y in 0..height {
		for x in 0..width {
			let depth_val = get_depth_at(depth, x, y, width, height);
			let disparity = (depth_val * max_disparity as f32).round() as i32;
			let x_right = x as i32 - disparity;

			if x_right >= 0 && x_right < width as i32 {
				if let Some(pixel) = img_rgb.get_pixel_checked(x as u32, y as u32) {
					right_rgb.put_pixel(x_right as u32, y as u32, *pixel);
				}
			}
		}
	}

	fill_disocclusions(&mut right_rgb);

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

fn fill_disocclusions(image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>) {
	let width = image.width() as usize;
	let height = image.height() as usize;
	let bg = Rgb([64u8, 64u8, 64u8]);

	let original = image.clone();

	for y in 0..height {
		for x in 0..width {
			let pixel = original.get_pixel(x as u32, y as u32);
			if pixel[0] == bg[0] && pixel[1] == bg[1] && pixel[2] == bg[2] {
				if let Some(nearest) = find_nearest_valid(&original, x, y, bg) {
					image.put_pixel(x as u32, y as u32, nearest);
				}
			}
		}
	}
}

fn find_nearest_valid(
	image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
	cx: usize,
	cy: usize,
	bg: Rgb<u8>,
) -> Option<Rgb<u8>> {
	let width = image.width() as usize;
	let height = image.height() as usize;

	for radius in 1..=20 {
		for dy in -(radius as i32)..=(radius as i32) {
			for dx in -(radius as i32)..=(radius as i32) {
				if dx.abs() != radius as i32 && dy.abs() != radius as i32 {
					continue;
				}
				let nx = (cx as i32 + dx) as usize;
				let ny = (cy as i32 + dy) as usize;
				if nx < width && ny < height {
					let pixel = image.get_pixel(nx as u32, ny as u32);
					if pixel[0] != bg[0] || pixel[1] != bg[1] || pixel[2] != bg[2] {
						return Some(*pixel);
					}
				}
			}
		}
	}

	None
}
