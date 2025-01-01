//! Configuration management for the shooter system.
//!
//! This module provides structures for managing configuration settings for:
//! - Camera parameters including field of view and orientation offsets
//! - YOLO object detection model settings including model paths and detection thresholds
//!
//! Configuration is loaded from TOML files and deserialized into strongly-typed structures
//! using serde.
use serde::Deserialize;
use url::Url;

/// Configuration for a camera source
#[derive(Debug, Clone, Deserialize)]
pub struct Camera {
    /// URL of the video stream
    pub stream_url: Url,
    /// Horizontal field of view in degrees
    pub horizontal_fov: f64,
    /// Vertical field of view in degrees
    pub vertical_fov: f64,
    /// Azimuth offset in degrees from true north
    pub azimuth_offset: f64,
    /// Elevation offset in degrees from horizontal
    pub elevation_offset: f64,
}

/// Configuration settings for YOLO (You Only Look Once) object detection model
#[derive(Debug, Clone, Deserialize)]
pub struct Yolo {
    /// Path to the neural network model configuration file
    pub model_cfg: std::path::PathBuf,
    /// Path to the pre-trained model weights file
    pub model_weights: std::path::PathBuf,
    /// Input size (width and height) for the neural network in pixels
    pub input_size: i32,
    /// Scale factor for normalizing pixel values (typically 1/255)
    pub scale_factor: f64,
    /// Minimum confidence threshold for object detection
    pub confidence_threshold: f32,
    /// Confidence threshold used in non-maximum suppression
    pub nms_confidence_threshold: f32,
    /// Intersection over Union (IoU) threshold for non-maximum suppression
    pub nms_threshold: f32,
    /// Minimum score threshold for keeping detections
    pub score_threshold: f32,
    /// Maximum number of detections to return (0 means no limit)
    pub top_k: i32,
}

impl Default for Yolo {
    fn default() -> Self {
        Self {
            model_cfg: std::path::PathBuf::from("models/yolov4-tiny.cfg"),
            model_weights: std::path::PathBuf::from("models/yolov4-tiny.weights"),
            input_size: 416,
            scale_factor: 1.0 / 255.0,
            confidence_threshold: 0.5,
            nms_confidence_threshold: 0.5,
            nms_threshold: 0.45,
            score_threshold: 0.5,
            top_k: 0,
        }
    }
}

/// Configuration structure for the shooter component
/// Contains settings for both camera and YOLO detection
#[derive(Debug, Clone, Deserialize)]
pub struct ShooterConfig {
    /// Camera configuration settings
    pub camera: Camera,
    /// YOLO object detection configuration settings
    pub yolo: Yolo,
}

impl ShooterConfig {
    /// Creates a new ShooterConfig instance by reading from a TOML configuration file
    pub fn new(config_path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(config_path)?;
        let config: ShooterConfig = toml::from_str(&contents)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use testdir::testdir;

    #[test]
    fn yolo_default_values() {
        let default_yolo = Yolo::default();
        assert_eq!(default_yolo.input_size, 416);
        assert_eq!(default_yolo.scale_factor, 1.0 / 255.0);
        assert_eq!(default_yolo.confidence_threshold, 0.5);
        assert_eq!(default_yolo.nms_confidence_threshold, 0.5);
        assert_eq!(default_yolo.nms_threshold, 0.45);
        assert_eq!(default_yolo.score_threshold, 0.5);
        assert_eq!(default_yolo.top_k, 0);
    }

    #[test]
    fn shooter_config_valid_load() -> Result<(), Box<dyn std::error::Error>> {
        let dir = testdir!();
        let config_path = dir.join("config.toml");

        let config_content = r#"
            [camera]
            stream_url = "rtsp://example.com/stream"
            horizontal_fov = 90.0
            vertical_fov = 60.0
            azimuth_offset = 0.0
            elevation_offset = -15.0

            [yolo]
            model_cfg = "models/custom.cfg"
            model_weights = "models/custom.weights"
            input_size = 416
            scale_factor = 0.00392156862745098
            confidence_threshold = 0.5
            nms_confidence_threshold = 0.5
            nms_threshold = 0.45
            score_threshold = 0.5
            top_k = 100
        "#;

        fs::write(&config_path, config_content)?;

        let config = ShooterConfig::new(&config_path)?;

        assert_eq!(
            config.camera.stream_url.as_str(),
            "rtsp://example.com/stream"
        );
        assert_eq!(config.camera.horizontal_fov, 90.0);
        assert_eq!(config.camera.vertical_fov, 60.0);
        assert_eq!(config.camera.azimuth_offset, 0.0);
        assert_eq!(config.camera.elevation_offset, -15.0);

        assert_eq!(config.yolo.input_size, 416);
        assert_eq!(config.yolo.scale_factor, 0.00392156862745098);
        assert_eq!(config.yolo.confidence_threshold, 0.5);
        assert_eq!(config.yolo.top_k, 100);

        Ok(())
    }

    #[test]
    fn shooter_config_invalid_toml() {
        let dir = testdir!();
        let config_path = dir.join("invalid.toml");

        fs::write(&config_path, "invalid toml content").unwrap();

        let result = ShooterConfig::new(&config_path);
        assert!(result.is_err());
    }

    #[test]
    fn shooter_config_nonexistent_file() {
        let dir = testdir!();
        let config_path = dir.join("nonexistent.toml");

        let result = ShooterConfig::new(&config_path);
        assert!(result.is_err());
    }
}
