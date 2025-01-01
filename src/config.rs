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

#[derive(Debug, Clone, Deserialize)]
pub struct ShooterConfig {
    pub camera: Camera,
    pub yolo: Yolo,
}

impl ShooterConfig {
    pub fn new(config_path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(config_path)?;
        let config: ShooterConfig = toml::from_str(&contents)?;
        Ok(config)
    }
}
