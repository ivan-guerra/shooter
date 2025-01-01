use crate::config::Camera;
use opencv::core::Rect;

#[derive(Debug)]
pub struct TargetPosition {
    pub azimuth: f64,
    pub elevation: f64,
}

pub fn get_center_of_rect(rect: &Rect) -> (i32, i32) {
    let x = rect.x + (rect.width / 2);
    let y = rect.y + (rect.height / 2);
    (x, y)
}

pub fn get_target_position(
    bound_box: &Rect,
    img_dim: (i32, i32),
    cam_settings: &Camera,
) -> TargetPosition {
    let (x, y) = get_center_of_rect(bound_box);
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
