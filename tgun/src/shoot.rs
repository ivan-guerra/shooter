//! Turret gun control module that handles video capture, human detection, and targeting
//!
//! This module provides the core functionality for the autonomous turret gun system:
//! - Video stream processing from configured camera
//! - Human detection using YOLO neural network
//! - Target position calculation and tracking
//! - Telemetry data transmission over UDP
//!
//! The main component is the [`TurretGun`] struct which runs the control loop in a
//! separate thread and can be started and stopped safely.
use crate::detection::DarknetModel;
use crate::targeting;
use opencv::{core::Mat, prelude::*, videoio};
use shared::{Rect, ShooterConfig, TurretGunTelemetry};
use std::net::UdpSocket;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Sends telemetry data over UDP socket using the provided configuration
fn send_telemetry(
    tlm: &TurretGunTelemetry,
    tlm_socket: &UdpSocket,
    configs: &ShooterConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let buf = bincode::serialize(tlm)?;
    tlm_socket.send_to(&buf, configs.telemetry.recv_addr.as_str())?;
    Ok(())
}

/// Represents a turret gun controller that can be run in a separate thread
pub struct TurretGun {
    /// Handle to the thread running the turret gun control loop
    thread: Option<JoinHandle<()>>,
    /// Atomic flag indicating whether the control loop should continue running
    is_running: Arc<AtomicBool>,
}

impl Default for TurretGun {
    fn default() -> Self {
        Self {
            thread: None,
            is_running: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl TurretGun {
    /// Starts the turret gun control loop in a separate thread
    pub fn start(&mut self, configs: &ShooterConfig) -> Result<(), Box<dyn std::error::Error>> {
        let configs = configs.clone();
        let mut dev =
            videoio::VideoCapture::from_file(configs.camera.stream_url.as_str(), videoio::CAP_ANY)
                .map_err(|_| "Failed to create VideoCapture")?;
        if !dev.is_opened()? {
            return Err("Video capture device is not opened".into());
        }
        let mut tlm = TurretGunTelemetry::default();
        let tlm_socket = UdpSocket::bind(configs.telemetry.send_addr.as_str())?;

        self.is_running.store(true, Ordering::SeqCst);
        let running = self.is_running.clone();
        self.thread = Some(thread::spawn(move || {
            let mut model = DarknetModel::new(&configs.yolo).expect("Failed to create model");

            while running.load(Ordering::SeqCst) {
                let mut frame = Mat::default();
                if let Ok(true) = dev.read(&mut frame) {
                    if !frame.empty() {
                        if let Ok(boxes) = model.find_humans(&frame) {
                            for b in &boxes {
                                let target_pos = targeting::get_target_position(
                                    b,
                                    (frame.cols(), frame.rows()),
                                    &configs.camera,
                                );
                                // TODO: Move the turret to the target position.
                                // TODO: Fire the gun.

                                tlm.azimuth = target_pos.azimuth;
                                tlm.elevation = target_pos.elevation;
                                tlm.has_fired = false;
                                tlm.bounding_box = Rect::new(b.x, b.y, b.width, b.height);
                                tlm.img_width = frame.cols();
                                tlm.img_height = frame.rows();
                                match send_telemetry(&tlm, &tlm_socket, &configs) {
                                    Ok(_) => {}
                                    Err(e) => eprintln!("Failed to send telemetry: {}", e),
                                }
                            }
                        }
                    }
                }
            }
        }));

        Ok(())
    }

    /// Stops the turret gun control loop and waits for the thread to finish
    pub fn stop(self) -> Result<(), Box<dyn std::error::Error + 'static>> {
        self.is_running.store(false, Ordering::SeqCst);
        if let Some(thread) = self.thread {
            thread.join().map_err(|_| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Failed to join thread",
                ))
            })?;
        }
        Ok(())
    }
}
