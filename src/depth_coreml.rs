use crate::error::{SpatialError, SpatialResult};
use image::{DynamicImage, ImageBuffer, Luma};
use ndarray::Array2;
use std::ffi::CString;

extern "C" {
    fn coreml_load_model(path: *const std::os::raw::c_char) -> *mut std::os::raw::c_void;
    fn coreml_unload_model(model: *mut std::os::raw::c_void);
    fn coreml_infer_depth(
        model: *mut std::os::raw::c_void,
        rgb_data: *const u8,
        width: i32,
        height: i32,
        output: *mut f32,
    ) -> i32;
}

pub struct CoreMLDepthEstimator {
    model: *mut std::os::raw::c_void,
    input_size: u32,
    inverted_depth: bool,
}

impl CoreMLDepthEstimator {
    pub fn new(model_path: &str, input_size: u32, inverted_depth: bool) -> SpatialResult<Self> {
        assert!(input_size > 0);
        let c_path = CString::new(model_path)
            .map_err(|e| SpatialError::ModelError(format!("Invalid model path: {}", e)))?;

        let model = unsafe { coreml_load_model(c_path.as_ptr()) };

        if model.is_null() {
            return Err(SpatialError::ModelError(format!(
                "Failed to load CoreML model: {}",
                model_path
            )));
        }

        tracing::info!("CoreML model loaded: {}", model_path);

        Ok(Self {
            model,
            input_size,
            inverted_depth,
        })
    }

    fn infer_raw(&self, image: &DynamicImage) -> SpatialResult<Vec<f32>> {
        let resized = image.resize_exact(
            self.input_size,
            self.input_size,
            image::imageops::FilterType::Lanczos3,
        );

        let rgb = resized.to_rgb8();
        let input_data: Vec<u8> = rgb.as_raw().to_vec();

        let output_size = (self.input_size * self.input_size) as usize;
        let mut output_data = vec![0.0f32; output_size];

        let result = unsafe {
            coreml_infer_depth(
                self.model,
                input_data.as_ptr(),
                self.input_size as i32,
                self.input_size as i32,
                output_data.as_mut_ptr(),
            )
        };

        if result != 0 {
            return Err(SpatialError::ModelError(format!(
                "CoreML inference failed with error code: {}",
                result
            )));
        }

        // Depth-convention models (DA3): negate so higher = nearer, matching
        // the disparity convention the rest of the pipeline assumes.
        if self.inverted_depth {
            for v in &mut output_data {
                *v = -*v;
            }
        }

        Ok(output_data)
    }

    pub fn estimate_unnormalized(&self, image: &DynamicImage) -> SpatialResult<Array2<f32>> {
        let (orig_width, orig_height) = (image.width(), image.height());
        let output_data = self.infer_raw(image)?;

        let size = self.input_size;
        let depth_image = ImageBuffer::from_fn(size, size, |x, y| {
            let idx = (y * size + x) as usize;
            Luma([output_data[idx]])
        });

        let resized_depth = image::imageops::resize(
            &depth_image,
            orig_width,
            orig_height,
            image::imageops::FilterType::Lanczos3,
        );

        let (w, h) = resized_depth.dimensions();
        let data: Vec<f32> = resized_depth.pixels().map(|p| p[0]).collect();
        Array2::from_shape_vec((h as usize, w as usize), data)
            .map_err(|e| SpatialError::TensorError(format!("Failed to reshape depth: {}", e)))
    }

    pub fn estimate_raw(
        &self,
        image: &DynamicImage,
    ) -> SpatialResult<ImageBuffer<Luma<f32>, Vec<f32>>> {
        let (orig_width, orig_height) = (image.width(), image.height());
        let mut output_data = self.infer_raw(image)?;

        let min_val = output_data.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = output_data
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;

        if range > 1e-6 {
            for v in &mut output_data {
                *v = (*v - min_val) / range;
            }
        }

        let size = self.input_size;
        let depth_image = ImageBuffer::from_fn(size, size, |x, y| {
            let idx = (y * size + x) as usize;
            Luma([output_data[idx]])
        });

        let resized_depth = image::imageops::resize(
            &depth_image,
            orig_width,
            orig_height,
            image::imageops::FilterType::Lanczos3,
        );

        Ok(resized_depth)
    }

    pub fn estimate(&self, image: &DynamicImage) -> SpatialResult<Array2<f32>> {
        let depth_image = self.estimate_raw(image)?;
        let (width, height) = depth_image.dimensions();
        let data: Vec<f32> = depth_image.pixels().map(|p| p[0]).collect();
        let depth_2d = Array2::from_shape_vec((height as usize, width as usize), data)
            .map_err(|e| SpatialError::TensorError(format!("Failed to reshape depth: {}", e)))?;
        Ok(depth_2d)
    }
}

impl Drop for CoreMLDepthEstimator {
    fn drop(&mut self) {
        unsafe {
            coreml_unload_model(self.model);
        }
    }
}

unsafe impl Send for CoreMLDepthEstimator {}
unsafe impl Sync for CoreMLDepthEstimator {}
