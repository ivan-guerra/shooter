use serde::Deserialize;
use url::Url;

#[derive(Debug, Clone, Deserialize)]
pub struct Camera {
    pub stream_url: Url,
    pub horizontal_fov: f64,
    pub vertical_fov: f64,
    pub azimuth_offset: f64,
    pub elevation_offset: f64,
}

#[derive(Debug, Deserialize)]
pub struct ShooterConfig {
    pub camera: Camera,
}

impl ShooterConfig {
    pub fn new(config_path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(config_path)?;
        let config: ShooterConfig = toml::from_str(&contents)?;
        Ok(config)
    }
}
