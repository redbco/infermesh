//! Process management for runtime adapters

use crate::config::ProcessConfig;
use crate::{Result, RuntimeError};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

/// Process manager for runtime adapters
pub struct ProcessManager {
    config: ProcessConfig,
    child: Option<Child>,
    start_time: Option<Instant>,
    restart_count: u32,
}

/// Process status
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessStatus {
    NotStarted,
    Starting,
    Running,
    Stopping,
    Stopped,
    Failed(String),
}

impl ProcessManager {
    /// Create a new process manager
    pub fn new(config: ProcessConfig) -> Self {
        Self {
            config,
            child: None,
            start_time: None,
            restart_count: 0,
        }
    }

    /// Start the managed process
    pub async fn start(&mut self) -> Result<()> {
        if self.is_running() {
            return Ok(());
        }

        info!("Starting process: {} {:?}", self.config.command, self.config.args);

        let mut command = Command::new(&self.config.command);
        command.args(&self.config.args);

        // Set environment variables
        for (key, value) in &self.config.env {
            command.env(key, value);
        }

        // Set working directory
        if let Some(working_dir) = &self.config.working_dir {
            command.current_dir(working_dir);
        }

        // Configure stdio
        command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null());

        // Spawn the process
        let child = command.spawn()
            .map_err(|e| RuntimeError::Process(format!("Failed to spawn process: {}", e)))?;

        self.child = Some(child);
        self.start_time = Some(Instant::now());

        // Wait for startup with timeout
        let startup_result = timeout(
            self.config.startup_timeout,
            self.wait_for_startup()
        ).await;

        match startup_result {
            Ok(Ok(())) => {
                info!("Process started successfully");
                Ok(())
            }
            Ok(Err(e)) => {
                error!("Process startup failed: {}", e);
                self.stop().await?;
                Err(e)
            }
            Err(_) => {
                error!("Process startup timed out after {:?}", self.config.startup_timeout);
                self.stop().await?;
                Err(RuntimeError::Process("Startup timeout".to_string()))
            }
        }
    }

    /// Stop the managed process
    pub async fn stop(&mut self) -> Result<()> {
        if let Some(mut child) = self.child.take() {
            info!("Stopping process");

            // Try graceful shutdown first
            #[cfg(unix)]
            {
                use nix::sys::signal::{self, Signal};
                use nix::unistd::Pid;
                
                let pid = Pid::from_raw(child.id() as i32);
                if let Err(e) = signal::kill(pid, Signal::SIGTERM) {
                    warn!("Failed to send SIGTERM: {}", e);
                } else {
                    debug!("Sent SIGTERM to process");
                }
            }

            // Wait for graceful shutdown with timeout
            let shutdown_result = timeout(
                self.config.shutdown_timeout,
                async {
                    loop {
                        match child.try_wait() {
                            Ok(Some(_)) => break,
                            Ok(None) => {
                                tokio::time::sleep(Duration::from_millis(100)).await;
                            }
                            Err(e) => {
                                return Err(RuntimeError::Process(format!("Wait failed: {}", e)));
                            }
                        }
                    }
                    Ok(())
                }
            ).await;

            match shutdown_result {
                Ok(Ok(())) => {
                    info!("Process stopped gracefully");
                }
                Ok(Err(e)) => {
                    error!("Error during graceful shutdown: {}", e);
                }
                Err(_) => {
                    warn!("Graceful shutdown timed out, forcing kill");
                    if let Err(e) = child.kill() {
                        error!("Failed to kill process: {}", e);
                    }
                    if let Err(e) = child.wait() {
                        error!("Failed to wait for killed process: {}", e);
                    }
                }
            }

            self.start_time = None;
        }

        Ok(())
    }

    /// Restart the managed process
    pub async fn restart(&mut self) -> Result<()> {
        info!("Restarting process");
        
        if self.restart_count >= self.config.max_restarts {
            return Err(RuntimeError::Process(
                format!("Maximum restart attempts ({}) exceeded", self.config.max_restarts)
            ));
        }

        self.stop().await?;
        self.restart_count += 1;
        self.start().await
    }

    /// Check if the process is running
    pub fn is_running(&mut self) -> bool {
        if let Some(child) = &mut self.child {
            match child.try_wait() {
                Ok(Some(_)) => {
                    // Process has exited
                    self.child = None;
                    self.start_time = None;
                    false
                }
                Ok(None) => {
                    // Process is still running
                    true
                }
                Err(_) => {
                    // Error checking status, assume not running
                    self.child = None;
                    self.start_time = None;
                    false
                }
            }
        } else {
            false
        }
    }

    /// Get process status
    pub fn status(&mut self) -> ProcessStatus {
        if self.is_running() {
            if let Some(start_time) = self.start_time {
                if start_time.elapsed() < Duration::from_secs(5) {
                    ProcessStatus::Starting
                } else {
                    ProcessStatus::Running
                }
            } else {
                ProcessStatus::Running
            }
        } else if self.child.is_some() {
            ProcessStatus::Stopping
        } else if self.restart_count > 0 {
            ProcessStatus::Failed("Process exited".to_string())
        } else {
            ProcessStatus::NotStarted
        }
    }

    /// Get process uptime
    pub fn uptime(&self) -> Option<Duration> {
        self.start_time.map(|start| start.elapsed())
    }

    /// Get restart count
    pub fn restart_count(&self) -> u32 {
        self.restart_count
    }

    /// Get process configuration
    pub fn config(&self) -> &ProcessConfig {
        &self.config
    }

    /// Check if auto-restart is enabled and should restart
    pub async fn check_auto_restart(&mut self) -> Result<bool> {
        if !self.config.auto_restart {
            return Ok(false);
        }

        if !self.is_running() && self.restart_count < self.config.max_restarts {
            warn!("Process died, attempting auto-restart (attempt {}/{})", 
                  self.restart_count + 1, self.config.max_restarts);
            
            match self.restart().await {
                Ok(()) => {
                    info!("Auto-restart successful");
                    Ok(true)
                }
                Err(e) => {
                    error!("Auto-restart failed: {}", e);
                    Err(e)
                }
            }
        } else {
            Ok(false)
        }
    }

    /// Wait for the process to start up (placeholder implementation)
    async fn wait_for_startup(&self) -> Result<()> {
        // In a real implementation, this would check for specific startup indicators
        // such as listening on a port, writing to a log file, etc.
        
        // For now, just wait a short time
        tokio::time::sleep(Duration::from_millis(500)).await;
        
        // Check if process is still running
        if let Some(_child) = &self.child {
            // In a real implementation, we'd check child.try_wait() here
            // For now, assume success
            Ok(())
        } else {
            Err(RuntimeError::Process("Process not found during startup".to_string()))
        }
    }
}

impl Drop for ProcessManager {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            warn!("ProcessManager dropped with running process, attempting cleanup");
            
            // Try to kill the process
            if let Err(e) = child.kill() {
                error!("Failed to kill process during cleanup: {}", e);
            }
            
            // Wait briefly for cleanup
            let _ = child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    //use std::path::PathBuf;

    fn create_test_config() -> ProcessConfig {
        ProcessConfig {
            command: "echo".to_string(),
            args: vec!["hello".to_string()],
            env: HashMap::new(),
            working_dir: None,
            startup_timeout: Duration::from_secs(5),
            shutdown_timeout: Duration::from_secs(5),
            auto_restart: false,
            max_restarts: 3,
        }
    }

    #[test]
    fn test_process_manager_creation() {
        let config = create_test_config();
        let manager = ProcessManager::new(config);
        
        assert_eq!(manager.restart_count(), 0);
        assert!(manager.uptime().is_none());
    }

    #[tokio::test]
    async fn test_process_start_stop() {
        let config = create_test_config();
        let mut manager = ProcessManager::new(config);
        
        // Initially not running
        assert!(!manager.is_running());
        assert_eq!(manager.status(), ProcessStatus::NotStarted);
        
        // Start process (echo command should exit quickly)
        let _result = manager.start().await;
        // Note: echo command exits immediately, so this might fail or succeed
        // depending on timing. In a real test, we'd use a long-running command.
        
        // Stop process
        let _ = manager.stop().await;
        assert!(!manager.is_running());
    }

    #[test]
    fn test_process_status() {
        let config = create_test_config();
        let mut manager = ProcessManager::new(config);
        
        assert_eq!(manager.status(), ProcessStatus::NotStarted);
        
        // After setting start_time, status should change
        manager.start_time = Some(Instant::now());
        // Note: Without actually starting a process, is_running() will return false
        // so status will still be NotStarted in this test
    }

    #[test]
    fn test_process_config() {
        let config = create_test_config();
        let manager = ProcessManager::new(config.clone());
        
        assert_eq!(manager.config().command, "echo");
        assert_eq!(manager.config().args, vec!["hello"]);
        assert_eq!(manager.config().startup_timeout, Duration::from_secs(5));
    }

    #[tokio::test]
    async fn test_auto_restart_disabled() {
        let config = create_test_config();
        let mut manager = ProcessManager::new(config);
        
        let should_restart = manager.check_auto_restart().await.unwrap();
        assert!(!should_restart);
    }

    #[tokio::test]
    async fn test_restart_limit() {
        let mut config = create_test_config();
        config.auto_restart = true;
        config.max_restarts = 1;
        
        let mut manager = ProcessManager::new(config);
        manager.restart_count = 1;
        
        // Should not restart when at limit
        let result = manager.restart().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Maximum restart attempts"));
    }
}
