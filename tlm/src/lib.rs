use minifb::Window;
use shared::{Camera, Rect, TurretGunTelemetry};

fn get_pixel_position(
    azimuth: f64,
    elevation: f64,
    img_dim: (i32, i32),
    cam_conf: &Camera,
) -> (f64, f64) {
    let (width, height): (f64, f64) = (img_dim.0.into(), img_dim.1.into());

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

fn draw_box_outline(
    buffer: &mut [u32],
    width: usize,
    height: usize,
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
            if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                let idx = (y as usize) * width + (x as usize);
                buffer[idx] = color;
            }
        }
    }

    // Draw bottom edge
    for y in (y1 - thickness)..y1 {
        for x in x0..x1 {
            if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                let idx = (y as usize) * width + (x as usize);
                buffer[idx] = color;
            }
        }
    }

    // Draw left edge
    for y in y0..y1 {
        for x in x0..(x0 + thickness) {
            if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                let idx = (y as usize) * width + (x as usize);
                buffer[idx] = color;
            }
        }
    }

    // Draw right edge
    for y in y0..y1 {
        for x in (x1 - thickness)..x1 {
            if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                let idx = (y as usize) * width + (x as usize);
                buffer[idx] = color;
            }
        }
    }
}

fn draw_dot(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center_x: f64,
    center_y: f64,
    radius: i32,
    color: u32,
) {
    let x0 = center_x as i32;
    let y0 = center_y as i32;

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= radius * radius {
                let x = x0 + dx;
                let y = y0 + dy;

                if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                    let idx = (y as usize) * width + (x as usize);
                    buffer[idx] = color;
                }
            }
        }
    }
}

pub fn render_telemetry(
    window: &mut Window,
    buffer: &mut [u32],
    telemetry: &TurretGunTelemetry,
    cam_conf: &Camera,
) {
    let width = telemetry.img_width as usize;
    let height = telemetry.img_height as usize;

    // Clear the buffer
    buffer.fill(0);

    // Draw bounding box outline with 10-pixel thickness
    draw_box_outline(
        buffer,
        width,
        height,
        &telemetry.bounding_box,
        10,       // Thickness
        0xFFFFFF, // White
    );

    let (dot_x, dot_y) = get_pixel_position(
        telemetry.azimuth,
        telemetry.elevation,
        (width as i32, height as i32),
        cam_conf,
    );
    // Draw the dot with a radius of 10 pixels
    if dot_x >= 0.0
        && dot_x < telemetry.img_width as f64
        && dot_y >= 0.0
        && dot_y < telemetry.img_height as f64
    {
        draw_dot(buffer, width, height, dot_x, dot_y, 10, 0xFF0000); // Red dot
    }

    window.update_with_buffer(buffer, width, height).unwrap();
}
