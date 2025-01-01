//! Command-line application for human detection in video streams.
//!
//! This application processes video input and performs real-time human detection
//! using computer vision. It consists of several modules:
//!
//! - `config`: Configuration handling and validation
//! - `detection`: Human detection using neural networks
//! - `playback`: Video capture and processing
//! - `targeting`: Target tracking and analysis
//!
//! # Usage
//!
//! ```bash
//! shooter <config>
//! ```
//!
//! Where `config` is the path to a TOML configuration file containing
//! required settings for video input and detection parameters.
use clap::Parser;

mod config;
mod detection;
mod playback;
mod targeting;

/// Command line arguments for the application.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the configuration file
    #[arg(help = "Path to the configuration file")]
    config: std::path::PathBuf,
}

/// Loads configuration and starts video playback with human detection
fn run(config_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let shooter_conf = config::ShooterConfig::new(config_path)?;
    let mut player = playback::VideoPlayer::new(&shooter_conf)?;
    playback::capture_humans(&mut player)?;
    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(&args.config) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
