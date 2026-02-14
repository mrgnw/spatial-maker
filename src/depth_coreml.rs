use anyhow::Result;
use image::{DynamicImage, ImageBuffer, Luma};
use std::ffi::CString;

const INPUT_SIZE: u32 = 518;
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

// FFI declarations for Swift CoreML bridge
extern "C" {
    fn coreml_load_model(path: *const std::os::raw::c_char) -> *mut std::os::raw::c_void;
    fn coreml_unload_model(model: *mut std::os::raw::c_void);
    fn coreml_infer_depth(
        model: *mut std::os::raw::c_void,
        rgb_data: *const f32,
        width: i32,
        height: i32,
        output: *mut f32,
    ) -> i32;
}

pub struct CoreMLDepthEstimator {
    model: *mut std::os::raw::c_void,
}

impl CoreMLDepthEstimator {
    pub fn new(model_path: &str) -> Result<Self> {
        let c_path = CString::new(model_path)?;

        let model = unsafe { coreml_load_model(c_path.as_ptr()) };

        if model.is_null() {
            anyhow::bail!("Failed to load CoreML model: {}", model_path);
        }

        println!("✓ CoreML model loaded: {}", model_path);

        Ok(Self { model })
    }

    /// Estimate depth from an image, returning a normalized depth map
    pub fn estimate(&self, image: &DynamicImage) -> Result<ImageBuffer<Luma<f32>, Vec<f32>>> {
        let (orig_width, orig_height) = (image.width(), image.height());

        // Preprocess: resize to 518x518
        let resized = image.resize_exact(
            INPUT_SIZE,
            INPUT_SIZE,
            image::imageops::FilterType::Lanczos3,
        );

        // Convert to RGB and normalize to float32 NCHW layout
        let rgb = resized.to_rgb8();
        let mut input_data = Vec::with_capacity(3 * INPUT_SIZE as usize * INPUT_SIZE as usize);

        // NCHW layout: all R channel, then G, then B
        for c in 0..3 {
            for pixel in rgb.pixels() {
                let normalized = (pixel[c] as f32 / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                input_data.push(normalized);
            }
        }

        // Allocate output buffer
        let output_size = (INPUT_SIZE * INPUT_SIZE) as usize;
        let mut output_data = vec![0.0f32; output_size];

        // Run inference via Swift CoreML
        let result = unsafe {
            coreml_infer_depth(
                self.model,
                input_data.as_ptr(),
                INPUT_SIZE as i32,
                INPUT_SIZE as i32,
                output_data.as_mut_ptr(),
            )
        };

        if result != 0 {
            anyhow::bail!("CoreML inference failed with error code: {}", result);
        }

        // Normalize depth to 0-1 range
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

        // Create depth image
        let depth_image = ImageBuffer::from_fn(INPUT_SIZE, INPUT_SIZE, |x, y| {
            let idx = (y * INPUT_SIZE + x) as usize;
            Luma([output_data[idx]])
        });

        // Resize back to original dimensions
        let resized_depth = image::imageops::resize(
            &depth_image,
            orig_width,
            orig_height,
            image::imageops::FilterType::Lanczos3,
        );

        Ok(resized_depth)
    }
}

impl Drop for CoreMLDepthEstimator {
    fn drop(&mut self) {
        unsafe {
            coreml_unload_model(self.model);
        }
    }
}

// Make it Send + Sync for multithreading (CoreML models are thread-safe)
unsafe impl Send for CoreMLDepthEstimator {}
unsafe impl Sync for CoreMLDepthEstimator {}
