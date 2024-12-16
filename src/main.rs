use clap::Parser;

pub mod detection;
pub mod playback;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(help = "video stream URL")]
    video_stream_url: String,
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut player = playback::VideoPlayer::new(&args.video_stream_url)?;
    playback::capture_humans(&mut player)?;
    Ok(())
}

fn main() {
    if let Err(e) = run(Args::parse()) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
