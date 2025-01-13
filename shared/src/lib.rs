//! Shared data structures and configuration types for the turret control system.
//!
//! This module contains the core types used for communication between client and server
//! components, including turret control commands and configuration structures for
//! cameras, object detection, and network settings.
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

impl TurretCmd {
    /// Creates a new `TurretCmd` instance with the specified azimuth, elevation, and fire state.
    pub fn new(azimuth: f64, elevation: f64, fire: bool) -> Self {
        Self {
            azimuth,
            elevation,
            fire,
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

/// Configuration for a client connection to the turret control server.
#[derive(Debug, Clone, Deserialize)]
pub struct ClientParams {
    /// The address of the server in the format "host:port"
    pub server_addr: String,
}

/// Server configuration parameters
#[derive(Debug, Clone, Deserialize)]
pub struct ServerParams {
    /// Port number on which the server will listen
    pub port: u16,
    /// Camera configuration settings
    pub camera: Camera,
    /// YOLO model configuration settings
    pub yolo: Yolo,
}

/// Configuration for the shooter application
#[derive(Debug, Clone, Deserialize)]
pub struct ShooterParams {
    pub server: ServerParams,
    pub client: ClientParams,
}

impl ShooterParams {
    /// Creates a new ShooterConfig instance by reading from a TOML configuration file
    pub fn new(config_path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(config_path)?;
        let config: ShooterParams = toml::from_str(&contents)?;
        Ok(config)
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use std::fs;
//     use testdir::testdir;
//
//     #[test]
//     fn yolo_default_values() {
//         let default_yolo = Yolo::default();
//         assert_eq!(default_yolo.input_size, 416);
//         assert_eq!(default_yolo.scale_factor, 1.0 / 255.0);
//         assert_eq!(default_yolo.confidence_threshold, 0.5);
//         assert_eq!(default_yolo.nms_confidence_threshold, 0.5);
//         assert_eq!(default_yolo.nms_threshold, 0.45);
//         assert_eq!(default_yolo.score_threshold, 0.5);
//         assert_eq!(default_yolo.top_k, 0);
//     }
//
//     #[test]
//     fn shooter_config_valid_load() -> Result<(), Box<dyn std::error::Error>> {
//         let dir = testdir!();
//         let config_path = dir.join("config.toml");
//
//         let config_content = r#"
//             [client]
//             server_addr = "127.0.0.1:8000"
//             [camera]
//             stream_url = "rtsp://example.com/stream"
//             frame_rate = 10
//             horizontal_fov = 90.0
//             vertical_fov = 60.0
//             azimuth_offset = 0.0
//             elevation_offset = -15.0
//
//             [yolo]
//             model_cfg = "models/custom.cfg"
//             model_weights = "models/custom.weights"
//             input_size = 416
//             scale_factor = 0.00392156862745098
//             confidence_threshold = 0.5
//             nms_confidence_threshold = 0.5
//             nms_threshold = 0.45
//             score_threshold = 0.5
//             top_k = 100
//
//             [telemetry]
//             send_addr = "192.168.1.16:5000"
//             recv_addr = "192.168.1.128:5000"
//         "#;
//
//         fs::write(&config_path, config_content)?;
//
//         let config = ShooterParams::new(&config_path)?;
//
//         assert_eq!(config.client.server_addr.as_str(), "127.0.0.1:8000");
//         assert_eq!(
//             config.camera.stream_url.as_str(),
//             "rtsp://example.com/stream"
//         );
//         assert_eq!(config.camera.frame_rate, 10);
//         assert_eq!(config.camera.horizontal_fov, 90.0);
//         assert_eq!(config.camera.vertical_fov, 60.0);
//         assert_eq!(config.camera.azimuth_offset, 0.0);
//         assert_eq!(config.camera.elevation_offset, -15.0);
//
//         assert_eq!(config.yolo.input_size, 416);
//         assert_eq!(config.yolo.scale_factor, 0.00392156862745098);
//         assert_eq!(config.yolo.confidence_threshold, 0.5);
//         assert_eq!(config.yolo.top_k, 100);
//
//         assert_eq!(config.telemetry.send_addr, "192.168.1.16:5000");
//         assert_eq!(config.telemetry.recv_addr, "192.168.1.128:5000");
//
//         Ok(())
//     }
//
//     #[test]
//     fn shooter_config_invalid_toml() {
//         let dir = testdir!();
//         let config_path = dir.join("invalid.toml");
//
//         fs::write(&config_path, "invalid toml content").unwrap();
//
//         let result = ShooterParams::new(&config_path);
//         assert!(result.is_err());
//     }
//
//     #[test]
//     fn shooter_config_nonexistent_file() {
//         let dir = testdir!();
//         let config_path = dir.join("nonexistent.toml");
//
//         let result = ShooterParams::new(&config_path);
//         assert!(result.is_err());
//     }
// }
