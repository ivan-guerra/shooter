//! Main entry point for the Turret Guidance System (TGS) server.
//!
//! This module initializes and orchestrates the core components of the turret guidance system:
//! - Command line argument parsing
//! - Logging configuration
//! - Video capture device initialization
//! - YOLO model loading for object detection
//! - TCP server setup for client communication
//! - Async runtime configuration and task management
//!
//! The server handles incoming connections from turret control clients and manages
//! the main control loop for target detection and tracking.
use crate::detection::DarknetModel;
use async_std::{channel, task};
use clap::Parser;
use log::{error, info};
use opencv::{prelude::*, videoio};
use shared::ShooterParams;
use simplelog::ConfigBuilder;
use simplelog::*;
use std::net::TcpListener;

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
            std::fs::File::create(args.log_path.unwrap_or(std::path::PathBuf::from("tgs.log")))?,
        ),
    ])
    .unwrap_or_else(|e| panic!("Failed to initialize logger: {}", e));

    let conf = ShooterParams::new(&args.config)?;

    let dev =
        videoio::VideoCapture::from_file(conf.server.camera.stream_url.as_str(), videoio::CAP_ANY)
            .map_err(|_| "Failed to create VideoCapture")?;
    if !dev.is_opened()? {
        return Err("Video capture device is not opened".into());
    }
    info!("Opened video capture device");

    let model = DarknetModel::new(&conf.server.yolo)?;
    info!("Loaded YOLO model");

    let listener = TcpListener::bind(format!("0.0.0.0:{}", conf.server.port))?;
    info!("Bound server to port {}", conf.server.port);

    info!("Waiting for incoming connection from client...");
    let stream = match listener.incoming().next() {
        Some(Ok(stream)) => {
            stream.set_nonblocking(true)?;
            stream
        }
        _ => return Err("Failed to accept incoming connection".into()),
    };
    info!("Accepted connection from client");

    // Create a channel for signaling shutdown
    let (shutdown_tx, shutdown_rx) = channel::bounded(1);

    // Spawn the control loop in a separate task
    let control_task = task::spawn(shoot::control_loop(shutdown_rx, conf, dev, model, stream));

    // Spawn a signal listener task to handle SIGTERM or SIGINT
    let signal_task = task::spawn(shoot::signal_listener(shutdown_tx));

    // Wait for the control loop to exit
    control_task.await;

    // If the control loop exited before we received a signal, cancel the signal task
    let signal_handle = signal_task.cancel();
    signal_handle.await;

    info!("Control loop has exited. tgs shutting down.");
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
