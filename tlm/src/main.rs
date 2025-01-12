use clap::Parser;
use minifb::{Key, Window, WindowOptions};
use shared::{ShooterConfig, TurretGunTelemetry};
use std::net::UdpSocket;

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

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let conf = ShooterConfig::new(&args.config)?;

    let socket = UdpSocket::bind(conf.telemetry.recv_addr)?;
    socket.set_nonblocking(true)?;
    let mut buf = [0; 1024];

    let mut window = Window::new(
        "Turret Telemetry",
        args.width,
        args.height,
        WindowOptions::default(),
    )?;
    let mut buffer = vec![0; args.width * args.height];

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if let Ok((len, _)) = socket.recv_from(&mut buf) {
            match bincode::deserialize::<TurretGunTelemetry>(&buf[..len]) {
                Ok(telemetry) => {
                    tlm::render_telemetry(&mut window, &mut buffer, &telemetry, &conf.camera);
                }
                Err(e) => {
                    eprintln!("Failed to deserialize telemetry data: {}", e);
                }
            }
        }
    }

    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(args) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
