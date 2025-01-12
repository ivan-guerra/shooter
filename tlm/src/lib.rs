//! Turret Live Monitor (TLM) visualization module
//!
//! This module provides functionality for real-time visualization of turret telemetry data,
//! including camera feed display, target position rendering, and debugging information.
//! It uses OpenCV for image processing and minifb for window display.
//!
//! Main features:
//! - Real-time camera feed display
//! - Target position visualization with angular coordinates
//! - Bounding box drawing for detected targets
//! - Telemetry data overlay (azimuth, elevation, etc.)
//!
//! The module handles coordinate transformations between angular space (azimuth/elevation)
//! and screen space, accounting for camera configuration parameters like FOV and offsets.
use minifb::Window;
use opencv::{
    core::{Mat, Scalar},
    imgproc,
    prelude::*,
    videoio,
};
use shared::{Camera, TurretGunTelemetry};

/// Calculates the target position in screen coordinates from angular coordinates.
fn get_target_position(
    azimuth: f64,
    elevation: f64,
    dimensions: (i32, i32),
    cam_conf: &Camera,
) -> (f64, f64) {
    let (width, height): (f64, f64) = (dimensions.0.into(), dimensions.1.into());

    // First subtract the camera offsets
    let azimuth_adjusted = azimuth - cam_conf.azimuth_offset;
    let elevation_adjusted = elevation - cam_conf.elevation_offset;

    // Convert angles back to normalized coordinates [-1, 1]
    let x_norm = azimuth_adjusted / (cam_conf.horizontal_fov / 2.0);
    let y_norm = elevation_adjusted / (cam_conf.vertical_fov / 2.0);

    // Convert normalized coordinates back to pixel coordinates
    let x = (x_norm * (width / 2.0)) + (width / 2.0);
    let y = (height / 2.0) - (y_norm * (height / 2.0));

    // Return the center point of where the bounding box should be
    (x, y)
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

/// Draws a green bounding box on the input image.
fn draw_bounding_box(
    input_image: &mut opencv::core::Mat,
    bbox: opencv::core::Rect,
) -> Result<(), opencv::Error> {
    imgproc::rectangle(
        input_image,
        bbox,
        Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        8,
        0,
    )?;

    Ok(())
}

/// Converts an OpenCV Mat to a buffer compatible with minifb window display.
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

/// Renders turret telemetry data to a display buffer and updates the window.
pub fn render_telemetry(
    window: &mut Window,
    dev: &mut videoio::VideoCapture,
    telemetry: &TurretGunTelemetry,
    cam_conf: &Camera,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut frame = Mat::default();
    let dimensions = (telemetry.img_width, telemetry.img_height);
    let pos = get_target_position(telemetry.azimuth, telemetry.elevation, dimensions, cam_conf);
    let text_pos = opencv::core::Point::new(10, 20);

    if dev.read(&mut frame)? && !frame.empty() {
        // Detect humans in the frame
        // Calculate and display the azimuth and elevation angles
        draw_text(
            &mut frame,
            &format!(
                "az: {:.2} el: {:.2} firing: {}",
                telemetry.azimuth, telemetry.elevation, telemetry.has_fired
            ),
            text_pos,
        )?;

        // Draw a dot at the target position
        draw_dot(
            &mut frame,
            opencv::core::Point::new(pos.0 as i32, pos.1 as i32),
        )?;

        // Draw a bounding box around the detected human
        draw_bounding_box(
            &mut frame,
            opencv::core::Rect::new(
                telemetry.bounding_box.x,
                telemetry.bounding_box.y,
                telemetry.bounding_box.width,
                telemetry.bounding_box.height,
            ),
        )?;

        // Convert to RGB format (OpenCV uses BGR by default)
        let mut rgb_frame = Mat::default();
        imgproc::cvt_color(&frame, &mut rgb_frame, imgproc::COLOR_BGR2RGB, 0)?;

        // Convert the Mat to a buffer for minifb
        let mut buffer: Vec<u32> = vec![0; (frame.cols() * frame.rows()) as usize];
        mat_to_minifb_buffer(
            &rgb_frame,
            &mut buffer,
            frame.cols() as usize,
            frame.rows() as usize,
        )?;

        // Update the window with the buffer
        window.update_with_buffer(&buffer, frame.cols() as usize, frame.rows() as usize)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::Camera;

    #[test]
    fn get_target_position_maps_center_correctly() {
        let cam_conf = Camera {
            stream_url: url::Url::parse("http://foo.bar").unwrap(),
            frame_rate: 10,
            horizontal_fov: 60.0,
            vertical_fov: 40.0,
            azimuth_offset: 0.0,
            elevation_offset: 0.0,
        };
        let dimensions = (800, 600);

        // Test center point (0, 0 degrees should map to center of screen)
        let (x, y) = get_target_position(0.0, 0.0, dimensions, &cam_conf);
        assert_eq!(x, 400.0);
        assert_eq!(y, 300.0);
    }

    #[test]
    fn get_target_position_maps_corners_correctly() {
        let cam_conf = Camera {
            stream_url: url::Url::parse("http://foo.bar").unwrap(),
            frame_rate: 10,
            horizontal_fov: 60.0,
            vertical_fov: 40.0,
            azimuth_offset: 0.0,
            elevation_offset: 0.0,
        };
        let dimensions = (800, 600);

        // Test top-left corner
        let (x, y) = get_target_position(-30.0, 20.0, dimensions, &cam_conf);
        assert!((x - 0.0).abs() < f64::EPSILON);
        assert!((y - 0.0).abs() < f64::EPSILON);

        // Test top-right corner
        let (x, y) = get_target_position(30.0, 20.0, dimensions, &cam_conf);
        assert!((x - 800.0).abs() < f64::EPSILON);
        assert!((y - 0.0).abs() < f64::EPSILON);

        // Test bottom-left corner
        let (x, y) = get_target_position(-30.0, -20.0, dimensions, &cam_conf);
        assert!((x - 0.0).abs() < f64::EPSILON);
        assert!((y - 600.0).abs() < f64::EPSILON);

        // Test bottom-right corner
        let (x, y) = get_target_position(30.0, -20.0, dimensions, &cam_conf);
        assert!((x - 800.0).abs() < f64::EPSILON);
        assert!((y - 600.0).abs() < f64::EPSILON);
    }

    #[test]
    fn get_target_position_handles_camera_offsets() {
        let cam_conf = Camera {
            stream_url: url::Url::parse("http://foo.bar").unwrap(),
            frame_rate: 10,
            horizontal_fov: 60.0,
            vertical_fov: 40.0,
            azimuth_offset: 10.0,
            elevation_offset: 5.0,
        };
        let dimensions = (800, 600);

        // Test with camera offsets
        let (x, y) = get_target_position(10.0, 5.0, dimensions, &cam_conf);
        assert_eq!(x, 400.0);
        assert_eq!(y, 300.0);
    }

    #[test]
    fn get_target_position_maps_with_nonzero_offsets() {
        let cam_conf = Camera {
            stream_url: url::Url::parse("http://foo.bar").unwrap(),
            frame_rate: 10,
            horizontal_fov: 60.0,
            vertical_fov: 40.0,
            azimuth_offset: 15.0,   // Camera is rotated 15° right
            elevation_offset: 10.0, // Camera is tilted 10° up
        };
        let dimensions = (800, 600);

        // Center point should now be at (15°, 10°) to compensate for offsets
        let (x, y) = get_target_position(15.0, 10.0, dimensions, &cam_conf);
        assert_eq!(x, 400.0);
        assert_eq!(y, 300.0);

        // Test a corner with offsets
        let (x, y) = get_target_position(45.0, 30.0, dimensions, &cam_conf);
        assert!((x - 800.0).abs() < f64::EPSILON);
        assert!((y - 0.0).abs() < f64::EPSILON);

        // Test opposite corner with offsets
        let (x, y) = get_target_position(-15.0, -10.0, dimensions, &cam_conf);
        assert!((x - 0.0).abs() < f64::EPSILON);
        assert!((y - 600.0).abs() < f64::EPSILON);
    }
}
