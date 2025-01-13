//! Turret targeting and control system implementation.
//!
//! This module implements the core targeting and control logic for the turret system, including:
//! - Video frame capture and processing
//! - Human detection using computer vision
//! - Target position calculation
//! - Network communication for turret control commands
//! - Main control loop orchestration
//! - Signal handling for graceful shutdown
//!
//! The system operates by continuously processing video frames, detecting targets,
//! and coordinating with a client over TCP to control turret movement.
use crate::detection::DarknetModel;
use crate::targeting;
use async_signal::Signals;
use async_std::{channel, task};
use futures::stream::StreamExt;
use log::{error, info, warn};
use opencv::{prelude::*, videoio};
use shared::{ShooterParams, TurretCmd, TurretCmdRequest};
use std::io::{ErrorKind, Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

/// Reads a command request from the TCP stream.
async fn read_cmd_request(
    mut stream: &TcpStream,
) -> Result<Option<TurretCmdRequest>, Box<dyn std::error::Error>> {
    let mut buffer = [0; 512];
    match stream.read(&mut buffer) {
        Ok(0) => {
            // Stream closed by the client
            Err("Connection closed by the client.".into())
        }
        Ok(bytes_read) => {
            let received_data = &buffer[..bytes_read];
            match bincode::deserialize::<TurretCmdRequest>(received_data) {
                Ok(request) => Ok(Some(request)),
                Err(e) => Err(Box::new(e)),
            }
        }
        Err(ref e) if e.kind() == ErrorKind::WouldBlock => Ok(None),
        Err(e) => Err(Box::new(e)),
    }
}

/// Sends a turret command over the TCP stream.
async fn send_cmd(
    mut stream: &TcpStream,
    cmd: TurretCmd,
) -> Result<(), Box<dyn std::error::Error>> {
    let serialized = bincode::serialize(&cmd)?;
    stream.write_all(&serialized)?;
    Ok(())
}

/// Main control loop for the turret targeting system.
pub async fn control_loop(
    shutdown_rx: channel::Receiver<()>,
    config: ShooterParams,
    mut dev: videoio::VideoCapture,
    mut model: DarknetModel,
    stream: std::net::TcpStream,
) {
    let interval = Duration::from_millis(1000 / config.server.camera.frame_rate);
    info!(
        "Starting control loop with run rate: {:?}Hz",
        1.0 / interval.as_secs_f64()
    );

    loop {
        let start = Instant::now();

        // Check for shutdown signal
        if shutdown_rx.try_recv().is_ok() {
            info!("Shutdown signal received. Exiting control loop...");
            break;
        }

        // Detect a human, move the gun, and fire
        let mut frame = Mat::default();
        if let Ok(true) = dev.read(&mut frame) {
            if !frame.empty() {
                if let Ok(boxes) = model.find_humans(&frame) {
                    if !boxes.is_empty() {
                        let target_pos = targeting::get_target_position(
                            &boxes[0], // We only care about the first detected target
                            (frame.cols(), frame.rows()),
                            &config.server.camera,
                        );

                        // TODO: Need to decide when to fire

                        // See if a command was requested
                        let request = match read_cmd_request(&stream).await {
                            Ok(Some(req)) => Some(req),
                            Ok(None) => None,
                            Err(e) => {
                                error!("Failed to read command request: {}", e);
                                break;
                            }
                        };

                        // If a command was requested, send the latest command info to the the client
                        if request.is_some() {
                            let cmd =
                                TurretCmd::new(target_pos.azimuth, target_pos.elevation, false);
                            if let Err(e) = send_cmd(&stream, cmd).await {
                                error!("Failed to send command response: {}", e);
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Calculate elapsed time and sleep for the remainder of the interval
        let elapsed = start.elapsed();
        if elapsed < interval {
            task::sleep(interval - elapsed).await;
        } else {
            warn!("Control loop overran by {:?}", elapsed - interval);
        }
    }
}

/// Listens for system termination signals and initiates graceful shutdown
///
/// Monitors for SIGTERM and SIGINT signals. When received, sends shutdown signal
/// through provided channel to trigger application shutdown.
pub async fn signal_listener(shutdown_tx: channel::Sender<()>) {
    let mut signals = Signals::new([async_signal::Signal::Term, async_signal::Signal::Int])
        .expect("Failed to create signal listener");

    // Use StreamExt::next to get the next signal
    if let Some(signal) = signals.next().await {
        info!("Received signal: {:?}", signal);
        info!("Sending shutdown signal...");
        let _ = shutdown_tx.send(()).await; // Ignore errors if receiver is already dropped
    }
}
