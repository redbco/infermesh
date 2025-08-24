//! # mesh-adapter-gpu
//!
//! GPU telemetry and monitoring adapters for DCGM, NVML, and other GPU management systems.
//!
//! This crate provides:
//! - GPU utilization and memory monitoring
//! - Temperature and power consumption tracking
//! - MIG (Multi-Instance GPU) partition inventory
//! - ECC error detection and reporting
//! - GPU topology and capability discovery
//! - Integration with NVIDIA DCGM and NVML
//!
//! ## Supported Backends
//!
//! - **NVML**: NVIDIA Management Library for direct GPU access
//! - **DCGM**: NVIDIA Data Center GPU Manager for enterprise monitoring
//! - **ROCm**: AMD GPU monitoring (future support)
//!
//! ## Example
//!
//! ```rust
//! use mesh_adapter_gpu::{GpuMonitor, GpuMonitorConfig, GpuBackend, GpuMonitorTrait};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = GpuMonitorConfig::new(GpuBackend::Nvml)
//!         .with_polling_interval(std::time::Duration::from_secs(1))
//!         .with_detailed_metrics(true);
//!     
//!     let mut monitor = GpuMonitor::new(config).await?;
//!     
//!     // Get GPU information
//!     let gpus = monitor.discover_gpus().await?;
//!     println!("Found {} GPUs", gpus.len());
//!     
//!     // Start monitoring
//!     monitor.start_monitoring().await?;
//!     
//!     // Get current metrics
//!     let metrics = monitor.get_all_metrics().await?;
//!     println!("GPU metrics: {:?}", metrics);
//!     
//!     Ok(())
//! }
//! ```

use thiserror::Error;

pub mod config;
pub mod monitor;
pub mod metrics;
pub mod discovery;
pub mod health;

// Backend-specific modules
#[cfg(feature = "nvml")]
pub mod nvml;

#[cfg(feature = "dcgm")]
pub mod dcgm;

#[cfg(feature = "rocm")]
pub mod rocm;

// Mock implementation for testing
#[cfg(any(feature = "mock", test))]
pub mod mock;

// Re-export main types
pub use config::{GpuMonitorConfig, GpuBackend, MonitoringConfig};
pub use monitor::{GpuMonitor, GpuMonitorTrait};
pub use metrics::{GpuMetrics, GpuInfo, GpuStatus, MemoryInfo, TemperatureInfo, PowerInfo, MigInfo};
pub use discovery::{GpuDiscovery, GpuTopology};
pub use health::{GpuHealth, HealthStatus};

/// Result type for GPU operations
pub type Result<T> = std::result::Result<T, GpuError>;

/// Errors that can occur during GPU operations
#[derive(Error, Debug)]
pub enum GpuError {
    #[error("GPU backend not supported: {0}")]
    UnsupportedBackend(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("GPU not found: {0}")]
    GpuNotFound(String),

    #[error("GPU initialization failed: {0}")]
    InitializationFailed(String),

    #[error("GPU communication error: {0}")]
    CommunicationError(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Driver error: {0}")]
    DriverError(String),

    #[error("NVML error: {0}")]
    NvmlError(String),

    #[error("DCGM error: {0}")]
    DcgmError(String),

    #[error("ROCm error: {0}")]
    RocmError(String),

    #[error("Monitoring error: {0}")]
    MonitoringError(String),

    #[error("Metrics collection failed: {0}")]
    MetricsError(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("GPU unavailable: {0}")]
    Unavailable(String),
}

impl GpuError {
    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            GpuError::CommunicationError(_)
                | GpuError::Timeout(_)
                | GpuError::Unavailable(_)
                | GpuError::MonitoringError(_)
        )
    }

    /// Check if this error indicates a driver issue
    pub fn is_driver_issue(&self) -> bool {
        matches!(
            self,
            GpuError::DriverError(_)
                | GpuError::NvmlError(_)
                | GpuError::DcgmError(_)
                | GpuError::InitializationFailed(_)
        )
    }

    /// Check if this error indicates a permission issue
    pub fn is_permission_issue(&self) -> bool {
        matches!(self, GpuError::PermissionDenied(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_error_properties() {
        let comm_error = GpuError::CommunicationError("test".to_string());
        assert!(comm_error.is_retryable());
        assert!(!comm_error.is_driver_issue());
        assert!(!comm_error.is_permission_issue());

        let driver_error = GpuError::DriverError("test".to_string());
        assert!(!driver_error.is_retryable());
        assert!(driver_error.is_driver_issue());
        assert!(!driver_error.is_permission_issue());

        let perm_error = GpuError::PermissionDenied("test".to_string());
        assert!(!perm_error.is_retryable());
        assert!(!perm_error.is_driver_issue());
        assert!(perm_error.is_permission_issue());
    }

    #[test]
    fn test_error_display() {
        let error = GpuError::UnsupportedBackend("test-backend".to_string());
        assert_eq!(error.to_string(), "GPU backend not supported: test-backend");

        let error = GpuError::GpuNotFound("GPU-0".to_string());
        assert_eq!(error.to_string(), "GPU not found: GPU-0");
    }
}
