use clap::Parser;

mod config;
mod detection;
mod playback;
mod targeting;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(help = "Path to the configuration file")]
    config: std::path::PathBuf,
}

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
