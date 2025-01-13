//! Turret Control Client Library
//!
//! This module provides the core functionality for controlling a remote turret system
//! over a TCP connection. It includes:
//!
//! - Communication protocols for sending commands and receiving responses
//! - A main control loop for continuous turret operation
//! - Signal handling for graceful shutdown
//!
//! The client maintains a persistent TCP connection with the turret control server,
//! sending command requests and processing responses while monitoring for system
//! shutdown signals.
use async_signal::Signals;
use async_std::channel;
use futures::stream::StreamExt;
use log::{error, info};
use std::io::{Read, Write};

/// Sends a turret command request to the server over a TCP stream.
async fn send_request(
    request: &shared::TurretCmdRequest,
    stream: &mut std::net::TcpStream,
) -> Result<(), Box<dyn std::error::Error>> {
    let buf = bincode::serialize(request)?;
    stream.write_all(&buf)?;
    Ok(())
}

/// Reads a turret command from the server over a TCP stream.
async fn read_cmd(
    stream: &mut std::net::TcpStream,
) -> Result<shared::TurretCmd, Box<dyn std::error::Error>> {
    let mut buf = [0; 1024];
    let n = stream.read(&mut buf)?;
    let cmd: shared::TurretCmd = bincode::deserialize(&buf[..n])?;
    Ok(cmd)
}

/// The main control loop for the turret control client.
///
/// This function maintains a continuous communication loop with the server,
/// sending command requests and receiving turret commands. The loop continues
/// until a shutdown signal is received.
pub async fn control_loop(shutdown_rx: channel::Receiver<()>, mut stream: std::net::TcpStream) {
    let mut request = shared::TurretCmdRequest::default();
    info!("Starting control loop...");

    loop {
        // Check for shutdown signal
        if shutdown_rx.try_recv().is_ok() {
            info!("Shutdown signal received. Exiting control loop...");
            break;
        }

        // Send a request to the server
        request.request_id += 1;
        if let Err(e) = send_request(&request, &mut stream).await {
            error!("Failed to send request: {}", e);
            break;
        }

        // Read a command response from the server
        let cmd = read_cmd(&mut stream).await;
        if cmd.is_err() {
            error!("Failed to read command response: {:?}", cmd.err());
            break;
        }

        // TODO: Move the turret into position.
        // TODO: Fire the turret if 'fire' is true.

        info!(
            "Successfully processed command request #{}",
            request.request_id
        );
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
