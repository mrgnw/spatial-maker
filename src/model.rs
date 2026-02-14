use anyhow::Result;
use std::path::PathBuf;

/// Find the checkpoint file for a given model name
///
/// Searches in order:
/// 1. Relative to source directory (for development)
/// 2. User's home directory ~/.spatial-maker/checkpoints/
/// 3. XDG data directory
/// 4. Current working directory ./checkpoints/
/// 5. SPATIAL_MAKER_CHECKPOINTS env var
pub fn find_checkpoint(checkpoint_name: &str) -> Result<PathBuf> {
    let search_paths = vec![
        // Development: relative to source file
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("checkpoints")
            .join(checkpoint_name),
        // User home directory
        dirs::home_dir()
            .unwrap_or_default()
            .join(".spatial-maker")
            .join("checkpoints")
            .join(checkpoint_name),
        // XDG data home
        dirs::data_dir()
            .unwrap_or_default()
            .join("spatial-maker")
            .join("checkpoints")
            .join(checkpoint_name),
        // Current working directory
        PathBuf::from("checkpoints").join(checkpoint_name),
    ];

    // Check env var first
    if let Ok(env_dir) = std::env::var("SPATIAL_MAKER_CHECKPOINTS") {
        let env_path = PathBuf::from(env_dir).join(checkpoint_name);
        if env_path.exists() {
            return Ok(env_path);
        }
    }

    for path in &search_paths {
        if path.exists() {
            return Ok(path.clone());
        }
    }

    anyhow::bail!(
        "Checkpoint '{}' not found.\nSearched locations:\n{}",
        checkpoint_name,
        search_paths
            .iter()
            .map(|p| format!("  - {}", p.display()))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
