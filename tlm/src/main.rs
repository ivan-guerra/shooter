//! A telemetry visualization application for the turret control system.
//!
//! This module provides a graphical interface that displays real-time telemetry data
//! received from a turret gun system over UDP. It uses the minifb library for
//! window management and visualization.
//!
//! The application accepts command-line arguments for:
//! - Configuration file path
//! - Window dimensions (width and height)
//!
//! The program will continue running until either:
//! - The window is closed
//! - The Escape key is pressed
//!
//! # Usage
//! ```shell
//! tlm --width <WIDTH> --height <HEIGHT> <CONFIG_FILE>
//! ```
use clap::Parser;
use minifb::{Key, Window, WindowOptions};
use opencv::{prelude::*, videoio};
use shared::{ShooterConfig, TurretGunTelemetry};
use std::net::UdpSocket;

#[doc(hidden)]
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the configuration file
    #[arg(help = "Path to the configuration file")]
    config: std::path::PathBuf,

    #[arg(
        long,
        short,
        help = "Width of the window. Should match the width of the video stream images."
    )]
    width: usize,

    #[arg(
        long,
        short,
        help = "Height of the window. Should match the height of the video stream images."
    )]
    height: usize,
}

#[doc(hidden)]
fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let config = ShooterConfig::new(&args.config)?;

    let socket = UdpSocket::bind(config.telemetry.recv_addr)?;
    socket.set_nonblocking(true)?;
    let mut buf = [0; 1024];

    let mut dev =
        videoio::VideoCapture::from_file(config.camera.stream_url.as_str(), videoio::CAP_ANY)
            .map_err(|_| "Failed to create VideoCapture")?;
    if !dev.is_opened()? {
        return Err("Video capture device is not opened".into());
    }

    let mut window = Window::new(
        "Turret Gun Telemetry",
        args.width,
        args.height,
        WindowOptions::default(),
    )?;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if let Ok((len, _)) = socket.recv_from(&mut buf) {
            match bincode::deserialize::<TurretGunTelemetry>(&buf[..len]) {
                Ok(telemetry) => {
                    tlm::render_telemetry(&mut window, &mut dev, &telemetry, &config.camera)?;
                }
                Err(e) => {
                    eprintln!("Failed to deserialize telemetry data: {}", e);
                }
            }
        }
    }

    Ok(())
}

#[doc(hidden)]
fn main() {
    let args = Args::parse();
    if let Err(e) = run(args) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
