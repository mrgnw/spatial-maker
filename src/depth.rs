use anyhow::{Context, Result};
use image::{DynamicImage, ImageBuffer, Luma};
use ndarray::{Array, Array4, Axis};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};

const INPUT_SIZE: u32 = 518;
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct DepthEstimator {
    session: Session,
}

impl DepthEstimator {
    pub fn new(model_path: &str) -> Result<Self> {
        // Note: CoreML EP tested but is slower than CPU on M4 Pro (9.5s vs 566ms)
        // Likely due to compilation overhead or unsupported ops. CPU is fast enough.
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")?;

        Ok(Self { session })
    }

    /// Estimate depth from an image, returning a normalized depth map
    pub fn estimate(&mut self, image: &DynamicImage) -> Result<ImageBuffer<Luma<f32>, Vec<f32>>> {
        let (orig_width, orig_height) = (image.width(), image.height());

        // Preprocess: resize to 518x518
        let resized = image.resize_exact(
            INPUT_SIZE,
            INPUT_SIZE,
            image::imageops::FilterType::Lanczos3,
        );

        // Convert to RGB and create NCHW tensor
        let rgb = resized.to_rgb8();
        let mut input_tensor =
            Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));

        for (y, row) in rgb.enumerate_rows() {
            for (x, _, pixel) in row {
                for c in 0..3 {
                    let normalized = (pixel[c] as f32 / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                    input_tensor[[0, c, y as usize, x as usize]] = normalized;
                }
            }
        }

        // Run inference
        let input_value = Value::from_array(input_tensor)?;
        let outputs = self.session.run(ort::inputs![input_value])?;

        // Extract the depth output tensor - returns (shape, data)
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        // Reconstruct as ndarray - shape should be [1, H, W]
        let depth_3d = Array::from_shape_vec((dims[0], dims[1], dims[2]), data.to_vec())?;

        // Extract to [H, W]
        let depth_2d = depth_3d.index_axis(Axis(0), 0).to_owned();

        // Normalize to 0-1 range
        let min_val = depth_2d.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = depth_2d.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;

        let normalized = if range > 1e-6 {
            depth_2d.mapv(|v| (v - min_val) / range)
        } else {
            Array::zeros(depth_2d.dim())
        };

        // Resize back to original dimensions
        let depth_image = ImageBuffer::from_fn(INPUT_SIZE, INPUT_SIZE, |x, y| {
            Luma([normalized[[y as usize, x as usize]]])
        });

        let resized_depth = image::imageops::resize(
            &depth_image,
            orig_width,
            orig_height,
            image::imageops::FilterType::Lanczos3,
        );

        Ok(resized_depth)
    }
}
