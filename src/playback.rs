//! Video playback and processing module
//!
//! This module provides functionality for:
//! * Video capture and playback from files or streams
//! * Real-time frame processing and display
//! * Human detection using YOLOv4-tiny model
use crate::detection::DarknetModel;
use minifb::{Key, Window, WindowOptions};
use opencv::imgproc;
use opencv::prelude::*;
use opencv::videoio;

/// A structure for handling video playback operations
///
/// # Fields
/// * `cam` - VideoCapture instance for accessing video frames
/// * `width` - Width of the video frame in pixels
/// * `height` - Height of the video frame in pixels
pub struct VideoPlayer {
    cam: videoio::VideoCapture,
    width: usize,
    height: usize,
}

impl VideoPlayer {
    pub fn new(video_stream_url: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let cam = videoio::VideoCapture::from_file(video_stream_url, videoio::CAP_ANY)?;
        if !cam.is_opened()? {
            return Err("unable to open video stream".into());
        }

        let width = cam.get(videoio::CAP_PROP_FRAME_WIDTH)? as usize;
        let height = cam.get(videoio::CAP_PROP_FRAME_HEIGHT)? as usize;

        Ok(Self { cam, width, height })
    }
}

/// Converts an OpenCV Mat to a buffer compatible with minifb window
///
/// # Arguments
///
/// * `mat` - The source OpenCV Mat in RGB format
/// * `buffer` - The destination buffer to store RGBA pixels
/// * `width` - The expected width of the image
/// * `height` - The expected height of the image
///
/// # Returns
///
/// * `Result<(), Box<dyn std::error::Error>>` - Ok if successful, Err if dimensions mismatch
///
/// # Errors
///
/// Returns an error if:
/// * The Mat dimensions don't match the provided width and height
/// * Failed to access Mat data bytes
fn mat_to_minifb_buffer(
    mat: &Mat,
    buffer: &mut [u32],
    width: usize,
    height: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Ensure the Mat has the expected dimensions
    if mat.rows() as usize != height || mat.cols() as usize != width {
        return Err("Mat dimensions do not match window dimensions".into());
    }

    // Iterate over each pixel and pack it into an RGBA u32
    for (i, pixel) in mat.data_bytes()?.chunks(3).enumerate() {
        let r = pixel[0] as u32;
        let g = pixel[1] as u32;
        let b = pixel[2] as u32;
        buffer[i] = (255 << 24) | (r << 16) | (g << 8) | b; // RGBA format with alpha = 255
    }

    Ok(())
}

/// Captures and processes video frames to detect humans using YOLOv4-tiny model
///
/// # Arguments
/// * `player` - Mutable reference to a VideoPlayer instance that provides the video feed
///
/// # Returns
/// * `Result<(), Box<dyn std::error::Error>>` - Ok(()) on successful execution, or an Error if something fails
pub fn capture_humans(player: &mut VideoPlayer) -> Result<(), Box<dyn std::error::Error>> {
    let mut window = Window::new(
        "Shooter",
        player.width,
        player.height,
        WindowOptions::default(),
    )?;
    let mut frame = Mat::default();
    let mut buffer: Vec<u32> = vec![0; player.width * player.height]; // Buffer for minifb (u32 RGBA)
    let mut model = DarknetModel::new(
        std::path::Path::new("models/yolov4-tiny.cfg"),
        std::path::Path::new("models/yolov4-tiny.weights"),
    )?;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if player.cam.read(&mut frame)? && !frame.empty() {
            // Detect humans in the frame
            let boxes = model.find_humans(&frame)?;
            model.draw_bounding_boxes(&mut frame, &boxes)?;

            // Convert to RGB format (OpenCV uses BGR by default)
            let mut rgb_frame = Mat::default();
            imgproc::cvt_color(&frame, &mut rgb_frame, imgproc::COLOR_BGR2RGB, 0)?;

            // Convert the Mat to a buffer for minifb
            mat_to_minifb_buffer(&rgb_frame, &mut buffer, player.width, player.height)?;

            // Update the window with the buffer
            window.update_with_buffer(&buffer, player.width, player.height)?;
        }
    }

    Ok(())
}
