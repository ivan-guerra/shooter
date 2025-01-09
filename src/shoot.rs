use crate::config::ShooterConfig;
use crate::detection::DarknetModel;
use crate::targeting;
use opencv::{core::Mat, prelude::*, videoio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

pub struct TurretGun {
    configs: ShooterConfig,
    thread: Option<JoinHandle<()>>,
    is_running: Arc<AtomicBool>,
}

impl TurretGun {
    pub fn new(configs: &ShooterConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let dev =
            videoio::VideoCapture::from_file(configs.camera.stream_url.as_str(), videoio::CAP_ANY)?;
        if !dev.is_opened()? {
            return Err("Unable to open video stream".into());
        }

        Ok(Self {
            configs: configs.clone(),
            thread: None,
            is_running: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let configs = self.configs.clone();
        let mut dev =
            videoio::VideoCapture::from_file(configs.camera.stream_url.as_str(), videoio::CAP_ANY)
                .map_err(|_| "Failed to create VideoCapture")?;

        if !dev.is_opened()? {
            return Err("Video capture device is not opened".into());
        }

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
                                    (configs.yolo.input_size, configs.yolo.input_size),
                                    &configs.camera,
                                );
                                // TODO: Remove this once telemetry is implemented.
                                println!(
                                    "az: {:.2}, el: {:.2}",
                                    target_pos.azimuth, target_pos.elevation
                                );
                                // TODO: Move the turret to the target position.
                                // TODO: Fire the gun.
                                // TODO: Send telemetry over UDP.
                            }
                        }
                    }
                }
            }
        }));

        Ok(())
    }

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
