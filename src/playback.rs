//! Video playback and processing module for the shooter system.
//!
//! This module provides functionality for:
//! - Reading and displaying video streams
//! - Processing frames for human detection using YOLOv4-tiny
//! - Visual overlay of detection results including bounding boxes and targeting information
//! - Real-time display using the minifb window system
use crate::config::ShooterConfig;
use crate::detection::DarknetModel;
use crate::targeting;
use minifb::{Key, Window, WindowOptions};
use opencv::{
    core::{Mat, Scalar},
    imgproc,
    prelude::*,
    videoio,
};

/// A structure representing a video player that handles video playback and processing.
pub struct VideoPlayer {
    /// OpenCV video capture device for reading video frames
    dev: videoio::VideoCapture,
    /// Configuration settings for the shooter system
    configs: ShooterConfig,
    /// Width of the video frame in pixels
    width: usize,
    /// Height of the video frame in pixels
    height: usize,
}

impl VideoPlayer {
    pub fn new(configs: &ShooterConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let dev =
            videoio::VideoCapture::from_file(configs.camera.stream_url.as_str(), videoio::CAP_ANY)?;
        if !dev.is_opened()? {
            return Err("unable to open video stream".into());
        }

        let width = dev.get(videoio::CAP_PROP_FRAME_WIDTH)? as usize;
        let height = dev.get(videoio::CAP_PROP_FRAME_HEIGHT)? as usize;

        Ok(Self {
            dev,
            configs: configs.clone(),
            width,
            height,
        })
    }
}

/// Converts an OpenCV Mat to a buffer compatible with minifb window
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

/// Draws green bounding boxes on the input image.
fn draw_bounding_boxes(
    input_image: &mut opencv::core::Mat,
    boxes: &[opencv::core::Rect],
) -> Result<(), opencv::Error> {
    for bbox in boxes {
        imgproc::rectangle(
            input_image,
            *bbox,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            8,
            0,
        )?;
    }

    Ok(())
}

/// Draws text on the input image at the specified position.
fn draw_text(
    input_image: &mut opencv::core::Mat,
    text: &str,
    position: opencv::core::Point,
) -> Result<(), opencv::Error> {
    imgproc::put_text(
        input_image,
        text,
        position,
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        1,
        8,
        false,
    )?;

    Ok(())
}

/// Draws a red dot (circle) on the input image at the specified point.
fn draw_dot(
    input_image: &mut opencv::core::Mat,
    point: opencv::core::Point,
) -> Result<(), opencv::Error> {
    imgproc::circle(
        input_image,
        point,
        5,
        Scalar::new(0.0, 0.0, 255.0, 0.0),
        -1,
        8,
        0,
    )?;

    Ok(())
}

/// Captures and processes video frames to detect humans using YOLOv4-tiny model
pub fn capture_humans(player: &mut VideoPlayer) -> Result<(), Box<dyn std::error::Error>> {
    let mut window = Window::new(
        "Shooter",
        player.width,
        player.height,
        WindowOptions::default(),
    )?;
    let mut frame = Mat::default();
    let mut buffer: Vec<u32> = vec![0; player.width * player.height]; // Buffer for minifb (u32 RGBA)
    let mut model = DarknetModel::new(&player.configs.yolo)?;
    let text_pos = opencv::core::Point::new(10, 20);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if player.dev.read(&mut frame)? && !frame.empty() {
            // Detect humans in the frame
            let boxes = model.find_humans(&frame)?;
            for b in &boxes {
                // Calculate and display the azimuth and elevation angles
                let target_pos = targeting::get_target_position(
                    b,
                    (
                        player.configs.yolo.input_size,
                        player.configs.yolo.input_size,
                    ),
                    &player.configs.camera,
                );
                draw_text(
                    &mut frame,
                    &format!(
                        "az: {:.2} el: {:.2}",
                        target_pos.azimuth, target_pos.elevation
                    ),
                    text_pos,
                )?;

                // Draw a dot at the center of the bounding box
                let box_center = targeting::get_center_of_rect(b);
                draw_dot(
                    &mut frame,
                    opencv::core::Point::new(box_center.0, box_center.1),
                )?;
            }
            // Draw bounding boxes around detected humans
            draw_bounding_boxes(&mut frame, &boxes)?;

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
