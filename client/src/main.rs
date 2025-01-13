//! Turret Control Client (TGC)
//!
//! This is the main executable for the Turret Control Client (TGC).
//! It handles:
//!
//! - Command-line argument parsing
//! - Configuration file loading
//! - Log setup and initialization
//! - TCP connection establishment
//! - Graceful shutdown handling
//!
//! The client can be configured via command line arguments and a configuration file.
//! It maintains dual logging to both terminal and file outputs, and establishes
//! a TCP connection to the turret control server specified in the configuration.
use async_std::{channel, task};
use clap::Parser;
use log::{error, info};
use shared::ShooterParams;
use simplelog::ConfigBuilder;
use simplelog::*;
use std::net::TcpStream;

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
            std::fs::File::create(args.log_path.unwrap_or(std::path::PathBuf::from("tgc.log")))?,
        ),
    ])
    .unwrap_or_else(|e| panic!("Failed to initialize logger: {}", e));

    let conf = ShooterParams::new(&args.config)?;
    info!("Loaded configuration file");

    let stream = TcpStream::connect(conf.client.server_addr)?;
    info!("Connected to server successfully");

    // Create a channel for signaling shutdown
    let (shutdown_tx, shutdown_rx) = channel::bounded(1);

    // Spawn the control loop in a separate task
    let control_task = task::spawn(client::control_loop(shutdown_rx, stream));

    // Spawn a signal listener task to handle SIGTERM or SIGINT
    let signal_task = task::spawn(client::signal_listener(shutdown_tx));

    // Wait for both tasks to complete
    control_task.await;
    signal_task.await;

    info!("Control loop has exited. tgc shutting down.");
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
