//! Shared Data Types for Turret Control System
//!
//! This module defines the common data structures and types used for communication
//! between the turret control client and server. It includes:
//!
//! - Command and telemetry data structures
//! - Configuration types for camera, YOLO detection, and network settings
//! - Geometric primitives for target tracking
//!
//! The types in this module are serializable using serde and are designed to be
//! used across the network boundary between client and server components.
//!
//! # Key Types
//!
//! - [`TurretCmd`]: Commands for controlling turret position and firing
//! - [`TurretGunTelemetry`]: Feedback data from the turret system
//! - [`ShooterConfig`]: Configuration settings for the entire system
//! - [`Camera`]: Settings for video input processing
//! - [`Yolo`]: Configuration for ML-based target detection
use serde::{Deserialize, Serialize};
use url::Url;

/// Represents a request from the client to the server for turret control commands.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct TurretCmdRequest {
    /// Unique identifier for the request to track command/response pairs
    pub request_id: u32,
}

/// Represents a command to control the turret's position and firing state.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct TurretCmd {
    /// Horizontal angle of the turret in degrees
    /// - Positive values rotate clockwise
    /// - Range: 0 to 360 degrees
    pub azimuth: f64,
    /// Vertical angle of the turret in degrees
    /// - Positive values move upward
    /// - Range: -10 to 90 degrees
    pub elevation: f64,
    /// Indicates whether the gun should fire
    /// - `true`: Trigger a shot
    /// - `false`: Hold fire
    pub fire: bool,
}

/// Configuration for a client connection to the turret control server.
#[derive(Debug, Clone, Deserialize)]
pub struct Client {
    /// The address of the server in the format "host:port"
    pub server_addr: String,
}

/// Represents a rectangular shape with position and dimensions
#[derive(Serialize, Deserialize, Debug)]
pub struct Rect {
    /// X-coordinate of the rectangle's top-left corner
    pub x: i32,
    /// Y-coordinate of the rectangle's top-left corner
    pub y: i32,
    /// Width of the rectangle in pixels
    pub width: i32,
    /// Height of the rectangle in pixels
    pub height: i32,
}

impl Rect {
    pub fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

/// Represents telemetry data for a turret gun system
#[derive(Serialize, Deserialize, Debug)]
pub struct TurretGunTelemetry {
    /// Horizontal angle of the turret in degrees
    pub azimuth: f64,
    /// Vertical angle of the turret in degrees
    pub elevation: f64,
    /// Indicates whether the gun has fired
    pub has_fired: bool,
    /// Four points defining the corners of the bounding box
    pub bounding_box: Rect,
    /// Width of the image in pixels
    pub img_width: i32,
    /// Height of the image in pixels
    pub img_height: i32,
}

impl TurretGunTelemetry {
    pub fn new(
        azimuth: f64,
        elevation: f64,
        has_fired: bool,
        bounding_box: Rect,
        img_width: i32,
        img_height: i32,
    ) -> Self {
        Self {
            azimuth,
            elevation,
            has_fired,
            bounding_box,
            img_width,
            img_height,
        }
    }
}

/// Configuration for a camera source
#[derive(Debug, Clone, Deserialize)]
pub struct Camera {
    /// URL of the video stream
    pub stream_url: Url,
    /// The number of frames per second
    pub frame_rate: u64,
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
            model_cfg: std::path::PathBuf::from("../models/yolov4-tiny.cfg"),
            model_weights: std::path::PathBuf::from("../models/yolov4-tiny.weights"),
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

/// Telemetry configuration structure containing network addresses for sending and receiving data
#[derive(Debug, Clone, Deserialize)]
pub struct Telemetry {
    /// Network address for sending telemetry data (e.g., 127.0.0.1:8000)
    pub send_addr: String,
    /// Network address for receiving telemetry data (e.g., 127.0.0.1:8000)
    pub recv_addr: String,
}

/// Configuration for the shooter application
#[derive(Debug, Clone, Deserialize)]
pub struct ShooterConfig {
    /// Camera configuration settings
    pub camera: Camera,
    /// YOLO model configuration settings
    pub yolo: Yolo,
    /// Telemetry configuration settings
    pub telemetry: Telemetry,
    pub client: Client,
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
            [client]
            server_addr = "127.0.0.1:8000"
            [camera]
            stream_url = "rtsp://example.com/stream"
            frame_rate = 10
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

            [telemetry]
            send_addr = "192.168.1.16:5000"
            recv_addr = "192.168.1.128:5000"
        "#;

        fs::write(&config_path, config_content)?;

        let config = ShooterConfig::new(&config_path)?;

        assert_eq!(config.client.server_addr.as_str(), "127.0.0.1:8000");
        assert_eq!(
            config.camera.stream_url.as_str(),
            "rtsp://example.com/stream"
        );
        assert_eq!(config.camera.frame_rate, 10);
        assert_eq!(config.camera.horizontal_fov, 90.0);
        assert_eq!(config.camera.vertical_fov, 60.0);
        assert_eq!(config.camera.azimuth_offset, 0.0);
        assert_eq!(config.camera.elevation_offset, -15.0);

        assert_eq!(config.yolo.input_size, 416);
        assert_eq!(config.yolo.scale_factor, 0.00392156862745098);
        assert_eq!(config.yolo.confidence_threshold, 0.5);
        assert_eq!(config.yolo.top_k, 100);

        assert_eq!(config.telemetry.send_addr, "192.168.1.16:5000");
        assert_eq!(config.telemetry.recv_addr, "192.168.1.128:5000");

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
