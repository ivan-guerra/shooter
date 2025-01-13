//! Control system for target detection and tracking.
//!
//! This module implements the main control loop and signal handling for the TGUN system.
//! It processes video frames, detects human targets using computer vision, and manages
//! turret positioning and telemetry reporting.
//!
//! # Core Components:
//! * `control_loop` - Main processing loop that handles video capture and target detection
//! * `signal_listener` - Handles system shutdown signals (SIGTERM/SIGINT)
//! * `send_telemetry` - Reports target tracking data via UDP
use crate::detection::DarknetModel;
use crate::targeting;
use async_signal::Signals;
use async_std::{channel, task};
use futures::stream::StreamExt;
use log::{error, info, warn};
use opencv::{prelude::*, videoio};
use shared::{Rect, ShooterParams, TurretGunTelemetry};
use std::net::UdpSocket;
use std::time::{Duration, Instant};

/// Sends turret gun telemetry data over UDP to configured receiver
fn send_telemetry(
    tlm: TurretGunTelemetry,
    tlm_socket: &UdpSocket,
    configs: &ShooterParams,
) -> Result<(), Box<dyn std::error::Error>> {
    let buf = bincode::serialize(&tlm)?;
    tlm_socket.send_to(&buf, configs.telemetry.recv_addr.as_str())?;
    Ok(())
}

/// Main control loop for target detection and tracking
///
/// Processes video frames at configured rate to detect human targets and control turret positioning.
/// Sends telemetry data for each detected target. Loop continues until shutdown signal is received.
pub async fn control_loop(
    shutdown_rx: channel::Receiver<()>,
    config: ShooterParams,
    mut dev: videoio::VideoCapture,
    mut model: DarknetModel,
    tlm_socket: UdpSocket,
) {
    let interval = Duration::from_millis(1000 / config.camera.frame_rate);
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
                    for b in &boxes {
                        let target_pos = targeting::get_target_position(
                            b,
                            (frame.cols(), frame.rows()),
                            &config.camera,
                        );
                        // TODO: Move the turret to the target position. Should be async task.
                        // TODO: Fire the gun. Should be async task.

                        let tlm = TurretGunTelemetry::new(
                            target_pos.azimuth,
                            target_pos.elevation,
                            false,
                            Rect::new(b.x, b.y, b.width, b.height),
                            frame.cols(),
                            frame.rows(),
                        );
                        if let Err(e) = send_telemetry(tlm, &tlm_socket, &config) {
                            error!("Failed to send telemetry: {}", e);
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
