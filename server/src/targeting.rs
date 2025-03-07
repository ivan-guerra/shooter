//! Target position calculation and tracking functionality.
//!
//! This module provides utilities for converting detected object coordinates
//! into real-world spherical coordinates (azimuth and elevation angles).
//! It handles:
//! - Transforming pixel coordinates to normalized space
//! - Calculating azimuth and elevation angles based on camera parameters
//!
//! The coordinate system uses:
//! - Azimuth: Horizontal angle in degrees from true north
//! - Elevation: Vertical angle in degrees from the horizontal plane
use opencv::core::Rect;
use shared::Camera;

/// Represents a target's position in spherical coordinates
#[derive(Debug)]
pub struct TargetPosition {
    /// Horizontal angle in degrees from true north (azimuth)
    pub azimuth: f64,
    /// Vertical angle in degrees from horizontal plane (elevation)
    pub elevation: f64,
}

/// Calculates the target position in spherical coordinates (azimuth and elevation)
/// based on the detected object's bounding box and camera parameters
///
/// # Arguments
/// * `bounding_box` - Reference to the detected object's bounding rectangle
/// * `img_dim` - Tuple containing the image dimensions (width, height)
/// * `cam_settings` - Reference to the camera configuration settings
///
/// # Returns
/// * `TargetPosition` - Calculated target position containing azimuth and elevation angles
pub fn get_target_position(
    bounding_box: &Rect,
    img_dim: (i32, i32),
    cam_settings: &Camera,
) -> TargetPosition {
    let (x, y) = (
        bounding_box.x + (bounding_box.width / 2),
        bounding_box.y + (bounding_box.height / 2),
    );
    let (x, y): (f64, f64) = (x.into(), y.into());
    let (width, height): (f64, f64) = (img_dim.0.into(), img_dim.1.into());

    // Normalize the pixel coordinates to the range [-1, 1]
    let x_norm = (x - (width / 2.0)) / (width / 2.0);
    let y_norm = ((height / 2.0) - y) / (height / 2.0);

    // Calculate the azimuth and elevation angles adjusting for the camera offsets
    let azimuth = x_norm * (cam_settings.horizontal_fov / 2.0) + cam_settings.azimuth_offset;
    let elevation = y_norm * (cam_settings.vertical_fov / 2.0) + cam_settings.elevation_offset;

    TargetPosition { azimuth, elevation }
}

#[cfg(test)]
mod tests {
    use super::*;
    use url::Url;

    #[test]
    fn target_position_center() {
        let camera = Camera {
            stream_url: Url::parse("https://example.com/stream").unwrap(),
            frame_rate: 30,
            horizontal_fov: 90.0,
            vertical_fov: 60.0,
            azimuth_offset: 0.0,
            elevation_offset: 0.0,
        };

        // Target at exact center: (320,240) in a (640,480) frame
        let rect = Rect::new(320 - 20, 240 - 20, 40, 40); // Adjust to make center of rect at (320,240)
        let pos = get_target_position(&rect, (640, 480), &camera);

        assert!((pos.azimuth).abs() < f64::EPSILON);
        assert!((pos.elevation).abs() < f64::EPSILON);
    }

    #[test]
    fn target_position_different_fov() {
        let camera = Camera {
            stream_url: Url::parse("https://example.com/stream").unwrap(),
            frame_rate: 30,
            horizontal_fov: 120.0,
            vertical_fov: 90.0,
            azimuth_offset: 0.0,
            elevation_offset: 0.0,
        };

        let rect = Rect::new(480, 360, 40, 40); // 3/4 across and 3/4 down
        let pos = get_target_position(&rect, (640, 480), &camera);

        assert!((pos.azimuth - 33.75).abs() < f64::EPSILON);
        assert!((pos.elevation + 26.25).abs() < f64::EPSILON);
    }
}
