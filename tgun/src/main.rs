//! Turret Gun Control System
//!
//! This module provides the main entry point for the turret gun control application.
//! It handles command-line argument parsing, logging setup, and graceful shutdown
//! on SIGTERM. The application reads configuration from a file and controls a
//! turret gun system through the following components:
//!
//! - Detection: Target detection functionality
//! - Shoot: Turret gun control and firing mechanisms
//! - Targeting: Target acquisition and tracking
//!
//! # Usage
//!
//! ```bash
//! tgun <config-file> [--log-path <log-file>]
//! ```
use crate::shoot::TurretGun;
use clap::Parser;
use log::{error, info};
use shared::ShooterConfig;
use simplelog::ConfigBuilder;
use simplelog::*;

mod detection;
mod shoot;
mod targeting;

/// Command line arguments for the application.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the configuration file
    #[arg(help = "Path to the configuration file")]
    config: std::path::PathBuf,

    #[arg(long, short, help = "Path to the log file")]
    log_path: Option<std::path::PathBuf>,
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let shooter_conf = ShooterConfig::new(&args.config)?;
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

    let mut gun = TurretGun::default();

    // Create a channel to communicate between the signal handler and main thread
    let (tx, rx) = std::sync::mpsc::channel();

    // Set up the signal handler
    ctrlc::set_handler(move || {
        tx.send(()).expect("Could not send signal");
    })?;

    info!("Starting turret gun...");
    gun.start(&shooter_conf)?;

    // Wait for SIGTERM
    rx.recv()?;

    // Stop the shooter when signal is received
    info!("Stopping turret gun");
    gun.stop()?;

    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(args) {
        error!("error: {e}");
        std::process::exit(1);
    }
}
