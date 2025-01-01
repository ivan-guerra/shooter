//! Target position calculation and tracking functionality.
//!
//! This module provides utilities for converting detected object coordinates
//! into real-world spherical coordinates (azimuth and elevation angles).
//! It handles:
//! - Converting bounding box coordinates to center points
//! - Transforming pixel coordinates to normalized space
//! - Calculating azimuth and elevation angles based on camera parameters
//!
//! The coordinate system uses:
//! - Azimuth: Horizontal angle in degrees from true north
//! - Elevation: Vertical angle in degrees from the horizontal plane
use crate::config::Camera;
use opencv::core::Rect;

/// Represents a target's position in spherical coordinates
#[derive(Debug)]
pub struct TargetPosition {
    /// Horizontal angle in degrees from true north (azimuth)
    pub azimuth: f64,
    /// Vertical angle in degrees from horizontal plane (elevation)
    pub elevation: f64,
}

/// Calculates the center point of a rectangular region
///
/// # Arguments
/// * `rect` - Reference to a rectangle structure containing x, y coordinates and dimensions
///
/// # Returns
/// * `(i32, i32)` - Tuple containing the (x, y) coordinates of the rectangle's center
pub fn get_center_of_rect(rect: &Rect) -> (i32, i32) {
    let x = rect.x + (rect.width / 2);
    let y = rect.y + (rect.height / 2);
    (x, y)
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
    let (x, y) = get_center_of_rect(bounding_box);
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
