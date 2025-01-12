//! Turret Gun Control System
//!
//! This module serves as the entry point for the `tgun` application, which provides
//! real-time target detection and tracking capabilities using computer vision.
//!
//! The system:
//! - Processes video input from a configured source
//! - Performs object detection using a YOLO darknet model
//! - Handles telemetry data through UDP communication
//! - Provides graceful shutdown handling via signal interrupts
//!
//! The application is configured via a configuration file specified as a command-line
//! argument and optionally supports custom log file paths.
use crate::detection::DarknetModel;
use async_std::{channel, task};
use clap::Parser;
use log::{error, info};
use opencv::{prelude::*, videoio};
use shared::ShooterConfig;
use simplelog::ConfigBuilder;
use simplelog::*;
use std::net::UdpSocket;

mod detection;
mod shoot;
mod targeting;

#[doc(hidden)]
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(help = "Path to the configuration file")]
    config: std::path::PathBuf,

    #[arg(long, short, help = "Path to the log file")]
    log_path: Option<std::path::PathBuf>,
}

#[doc(hidden)]
async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let configs = ShooterConfig::new(&args.config)?;
    CombinedLogger::init(vec![
        TermLogger::new(
            LevelFilter::Info,
            ConfigBuilder::new().set_time_format_rfc2822().build(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        ),
        WriteLogger::new(
            LevelFilter::Debug,
            ConfigBuilder::new().set_time_format_rfc2822().build(),
            std::fs::File::create(
                args.log_path
                    .unwrap_or(std::path::PathBuf::from("tgun.log")),
            )?,
        ),
    ])?;

    let dev =
        videoio::VideoCapture::from_file(configs.camera.stream_url.as_str(), videoio::CAP_ANY)
            .map_err(|_| "Failed to create VideoCapture")?;
    if !dev.is_opened()? {
        return Err("Video capture device is not opened".into());
    }
    info!("Opened video capture device");

    let tlm_socket = UdpSocket::bind(configs.telemetry.send_addr.as_str())?;
    info!("Opened telemetry socket");

    let model = DarknetModel::new(&configs.yolo)?;
    info!("Loaded YOLO model");

    // Create a channel for signaling shutdown
    let (shutdown_tx, shutdown_rx) = channel::bounded(1);

    // Spawn the control loop in a separate task
    let control_task = task::spawn(shoot::control_loop(
        shutdown_rx,
        configs,
        dev,
        model,
        tlm_socket,
    ));

    // Spawn a signal listener task to handle SIGTERM or SIGINT
    let signal_task = task::spawn(shoot::signal_listener(shutdown_tx));

    // Wait for both tasks to complete
    control_task.await;
    signal_task.await;

    info!("Control loop has exited. tgun shutting down.");
    Ok(())
}

#[doc(hidden)]
#[async_std::main]
async fn main() {
    if let Err(e) = run().await {
        error!("Error: {}", e);
        std::process::exit(1);
    }
}
