use std::fmt;

pub type SpatialResult<T> = Result<T, SpatialError>;

#[derive(Debug)]
pub enum SpatialError {
	ModelError(String),
	ImageError(String),
	TensorError(String),
	IoError(String),
	ConfigError(String),
	Other(String),
}

impl fmt::Display for SpatialError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			SpatialError::ModelError(msg) => write!(f, "Model error: {}", msg),
			SpatialError::ImageError(msg) => write!(f, "Image error: {}", msg),
			SpatialError::TensorError(msg) => write!(f, "Tensor error: {}", msg),
			SpatialError::IoError(msg) => write!(f, "I/O error: {}", msg),
			SpatialError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
			SpatialError::Other(msg) => write!(f, "Error: {}", msg),
		}
	}
}

impl std::error::Error for SpatialError {}

impl From<std::io::Error> for SpatialError {
	fn from(e: std::io::Error) -> Self {
		SpatialError::IoError(e.to_string())
	}
}

impl From<image::ImageError> for SpatialError {
	fn from(e: image::ImageError) -> Self {
		SpatialError::ImageError(e.to_string())
	}
}
