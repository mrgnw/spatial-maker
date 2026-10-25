use crate::error::{SpatialError, SpatialResult};
use image::DynamicImage;
use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::path::Path;

extern "C" {
    fn spatial_writer_create(
        output_path: *const c_char,
        audio_source_path: *const c_char,
        width: i32,
        height: i32,
        fps: f64,
        baseline_micrometers: u32,
        horizontal_fov_millidegrees: u32,
        disparity_adjustment: i32,
    ) -> *mut c_void;
    fn spatial_writer_append(writer: *mut c_void, left: *const u8, right: *const u8) -> i32;
    fn spatial_writer_finish(writer: *mut c_void) -> i32;
}

pub struct SpatialMetadata {
    pub baseline_mm: f64,
    pub horizontal_fov_degrees: f64,
    pub disparity_adjustment: f64,
}

impl Default for SpatialMetadata {
    fn default() -> Self {
        Self {
            baseline_mm: 65.0,
            horizontal_fov_degrees: 90.0,
            disparity_adjustment: 0.0,
        }
    }
}

pub struct SpatialVideoWriter {
    writer: *mut c_void,
    width: u32,
    height: u32,
}

impl SpatialVideoWriter {
    pub fn new(
        output_path: &Path,
        audio_source: Option<&Path>,
        width: u32,
        height: u32,
        fps: f64,
        metadata: &SpatialMetadata,
    ) -> SpatialResult<Self> {
        assert!(width > 0 && height > 0 && fps > 0.0);
        assert!((-1.0..=1.0).contains(&metadata.disparity_adjustment));
        let output = path_to_cstring(output_path)?;
        let audio = audio_source.map(path_to_cstring).transpose()?;
        let writer = unsafe {
            spatial_writer_create(
                output.as_ptr(),
                audio.as_ref().map_or(std::ptr::null(), |a| a.as_ptr()),
                width as i32,
                height as i32,
                fps,
                (metadata.baseline_mm * 1000.0).round() as u32,
                (metadata.horizontal_fov_degrees * 1000.0).round() as u32,
                (metadata.disparity_adjustment * 10000.0).round() as i32,
            )
        };
        if writer.is_null() {
            return Err(SpatialError::Other(format!(
                "Failed to start MV-HEVC writer for {}",
                output_path.display()
            )));
        }
        Ok(Self {
            writer,
            width,
            height,
        })
    }

    pub fn append(&mut self, left: &DynamicImage, right: &DynamicImage) -> SpatialResult<()> {
        let left = left.to_rgb8();
        let right = right.to_rgb8();
        assert_eq!(left.dimensions(), (self.width, self.height));
        assert_eq!(right.dimensions(), (self.width, self.height));
        let status = unsafe { spatial_writer_append(self.writer, left.as_ptr(), right.as_ptr()) };
        if status != 0 {
            return Err(SpatialError::Other(format!(
                "MV-HEVC frame append failed ({})",
                status
            )));
        }
        Ok(())
    }

    pub fn finish(self) -> SpatialResult<()> {
        let writer = self.writer;
        std::mem::forget(self);
        let status = unsafe { spatial_writer_finish(writer) };
        if status != 0 {
            return Err(SpatialError::Other(format!(
                "MV-HEVC finish failed ({})",
                status
            )));
        }
        Ok(())
    }
}

impl Drop for SpatialVideoWriter {
    fn drop(&mut self) {
        unsafe {
            spatial_writer_finish(self.writer);
        }
    }
}

unsafe impl Send for SpatialVideoWriter {}

fn path_to_cstring(path: &Path) -> SpatialResult<CString> {
    let s = path
        .to_str()
        .ok_or_else(|| SpatialError::Other(format!("Invalid path encoding: {}", path.display())))?;
    CString::new(s).map_err(|e| SpatialError::Other(format!("Invalid path: {}", e)))
}
