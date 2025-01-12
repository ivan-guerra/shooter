//! Telemetry visualization module for turret targeting system.
//!
//! This module provides functionality to visualize turret gun telemetry data,
//! including target position and bounding box rendering. It handles coordinate
//! transformations from angular space (azimuth/elevation) to screen space,
//! accounting for camera configuration parameters like field of view and offsets.
//!
//! Key components:
//! - Target position calculation from angular coordinates
//! - Bounding box rendering
//! - Target dot visualization
//! - Buffer management for display output
use minifb::Window;
use shared::{Camera, Rect, TurretGunTelemetry};

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

/// Draws a rectangular outline with specified thickness in the buffer.
fn draw_box_outline(
    buffer: &mut [u32],
    dimensions: (i32, i32),
    rect: &Rect,
    thickness: i32,
    color: u32,
) {
    let x0 = rect.x;
    let y0 = rect.y;
    let x1 = x0 + rect.width;
    let y1 = y0 + rect.height;

    // Draw top edge
    for y in y0..(y0 + thickness) {
        for x in x0..x1 {
            if x >= 0 && x < dimensions.0 && y >= 0 && y < dimensions.1 {
                let idx = (y as usize) * dimensions.0 as usize + (x as usize);
                buffer[idx] = color;
            }
        }
    }

    // Draw bottom edge
    for y in (y1 - thickness)..y1 {
        for x in x0..x1 {
            if x >= 0 && x < dimensions.0 && y >= 0 && y < dimensions.1 {
                let idx = (y as usize) * dimensions.0 as usize + (x as usize);
                buffer[idx] = color;
            }
        }
    }

    // Draw left edge
    for y in y0..y1 {
        for x in x0..(x0 + thickness) {
            if x >= 0 && x < dimensions.0 && y >= 0 && y < dimensions.1 {
                let idx = (y as usize) * dimensions.0 as usize + (x as usize);
                buffer[idx] = color;
            }
        }
    }

    // Draw right edge
    for y in y0..y1 {
        for x in (x1 - thickness)..x1 {
            if x >= 0 && x < dimensions.0 && y >= 0 && y < dimensions.1 {
                let idx = (y as usize) * dimensions.0 as usize + (x as usize);
                buffer[idx] = color;
            }
        }
    }
}

/// Draws a filled circle (dot) in the buffer at the specified position.
fn draw_dot(buffer: &mut [u32], dimensions: (i32, i32), pos: (f64, f64), radius: i32, color: u32) {
    let (width, height) = dimensions;
    let (center_x, center_y) = pos;
    let x0 = center_x as i32;
    let y0 = center_y as i32;

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= radius * radius {
                let x = x0 + dx;
                let y = y0 + dy;

                if x >= 0 && x < width && y >= 0 && y < height {
                    let idx = (y as usize) * width as usize + (x as usize);
                    buffer[idx] = color;
                }
            }
        }
    }
}

/// Renders turret telemetry data to a display buffer and updates the window.
pub fn render_telemetry(
    window: &mut Window,
    buffer: &mut [u32],
    telemetry: &TurretGunTelemetry,
    cam_conf: &Camera,
) -> Result<(), Box<dyn std::error::Error>> {
    let dimensions = (telemetry.img_width, telemetry.img_height);

    // Clear the buffer
    buffer.fill(0);

    // Draw bounding box outline with 5-pixel thickness
    let thickness = 5;
    let white = 0xFFFFFF;
    draw_box_outline(
        buffer,
        dimensions,
        &telemetry.bounding_box,
        thickness,
        white,
    );

    // Draw the target location as a dot with a radius of 5 pixels
    let pos = get_target_position(telemetry.azimuth, telemetry.elevation, dimensions, cam_conf);
    let radius = 5;
    let red = 0xFF0000;
    draw_dot(buffer, dimensions, pos, radius, red);

    window.update_with_buffer(buffer, dimensions.0 as usize, dimensions.1 as usize)?;

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
