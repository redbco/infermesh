//! GPU monitoring implementation

use crate::config::{GpuMonitorConfig, GpuBackend};
use crate::metrics::GpuMetrics;
use crate::health::GpuHealth;
use crate::{Result, GpuError};

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Trait defining the interface for GPU monitors
#[async_trait]
pub trait GpuMonitorTrait: Send + Sync {
    /// Initialize the GPU monitor
    async fn initialize(&mut self) -> Result<()>;

    /// Shutdown the GPU monitor
    async fn shutdown(&mut self) -> Result<()>;

    /// Discover available GPUs
    async fn discover_gpus(&self) -> Result<Vec<u32>>;

    /// Get metrics for a specific GPU
    async fn get_gpu_metrics(&self, gpu_id: u32) -> Result<GpuMetrics>;

    /// Get metrics for all monitored GPUs
    async fn get_all_metrics(&self) -> Result<Vec<GpuMetrics>>;

    /// Start continuous monitoring
    async fn start_monitoring(&mut self) -> Result<()>;

    /// Stop continuous monitoring
    async fn stop_monitoring(&mut self) -> Result<()>;

    /// Check if monitoring is active
    fn is_monitoring(&self) -> bool;

    /// Get GPU health status
    async fn get_gpu_health(&self, gpu_id: u32) -> Result<GpuHealth>;

    /// Get configuration
    fn get_config(&self) -> &GpuMonitorConfig;
}

/// Main GPU monitor that delegates to backend-specific implementations
pub struct GpuMonitor {
    inner: Box<dyn GpuMonitorTrait>,
    config: GpuMonitorConfig,
    monitoring_active: Arc<RwLock<bool>>,
}

impl GpuMonitor {
    /// Create a new GPU monitor
    pub async fn new(config: GpuMonitorConfig) -> Result<Self> {
        info!("Creating GPU monitor for backend: {}", config.backend);
        
        // Validate configuration
        config.validate().map_err(GpuError::Configuration)?;

        // Create backend-specific monitor
        let inner = create_gpu_monitor(&config).await?;

        let monitoring_active = Arc::new(RwLock::new(false));

        Ok(Self {
            inner,
            config,
            monitoring_active,
        })
    }

    /// Start background monitoring tasks
    pub async fn start_background_monitoring(&self) -> Result<()> {
        info!("Starting background GPU monitoring");

        let monitoring_active = Arc::clone(&self.monitoring_active);
        *monitoring_active.write().await = true;

        // In a real implementation, this would spawn background tasks
        // for continuous monitoring, health checking, etc.
        debug!("Background monitoring tasks would be started here");

        Ok(())
    }

    /// Stop background monitoring tasks
    pub async fn stop_background_monitoring(&self) -> Result<()> {
        info!("Stopping background GPU monitoring");

        let monitoring_active = Arc::clone(&self.monitoring_active);
        *monitoring_active.write().await = false;

        // In a real implementation, this would cancel background tasks
        debug!("Background monitoring tasks would be stopped here");

        Ok(())
    }
}

#[async_trait]
impl GpuMonitorTrait for GpuMonitor {
    async fn initialize(&mut self) -> Result<()> {
        info!("Initializing GPU monitor");
        self.inner.initialize().await
    }

    async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down GPU monitor");
        self.stop_background_monitoring().await?;
        self.inner.shutdown().await
    }

    async fn discover_gpus(&self) -> Result<Vec<u32>> {
        self.inner.discover_gpus().await
    }

    async fn get_gpu_metrics(&self, gpu_id: u32) -> Result<GpuMetrics> {
        debug!("Getting metrics for GPU {}", gpu_id);
        self.inner.get_gpu_metrics(gpu_id).await
    }

    async fn get_all_metrics(&self) -> Result<Vec<GpuMetrics>> {
        self.inner.get_all_metrics().await
    }

    async fn start_monitoring(&mut self) -> Result<()> {
        info!("Starting GPU monitoring");
        self.start_background_monitoring().await?;
        self.inner.start_monitoring().await
    }

    async fn stop_monitoring(&mut self) -> Result<()> {
        info!("Stopping GPU monitoring");
        self.stop_background_monitoring().await?;
        self.inner.stop_monitoring().await
    }

    fn is_monitoring(&self) -> bool {
        self.inner.is_monitoring()
    }

    async fn get_gpu_health(&self, gpu_id: u32) -> Result<GpuHealth> {
        self.inner.get_gpu_health(gpu_id).await
    }

    fn get_config(&self) -> &GpuMonitorConfig {
        &self.config
    }
}

/// Create a backend-specific GPU monitor
async fn create_gpu_monitor(config: &GpuMonitorConfig) -> Result<Box<dyn GpuMonitorTrait>> {
    match config.backend {
        #[cfg(feature = "nvml")]
        GpuBackend::Nvml => {
            let monitor = crate::nvml::NvmlMonitor::new(config.clone()).await?;
            Ok(Box::new(monitor))
        }
        
        #[cfg(feature = "dcgm")]
        GpuBackend::Dcgm => {
            let monitor = crate::dcgm::DcgmMonitor::new(config.clone()).await?;
            Ok(Box::new(monitor))
        }
        
        #[cfg(feature = "rocm")]
        GpuBackend::Rocm => {
            let monitor = crate::rocm::RocmMonitor::new(config.clone()).await?;
            Ok(Box::new(monitor))
        }
        
        #[cfg(any(feature = "mock", test))]
        GpuBackend::Mock => {
            let monitor = crate::mock::MockGpuMonitor::new(config.clone()).await?;
            Ok(Box::new(monitor))
        }
        
        _ => {
            warn!("GPU backend {:?} not supported or feature not enabled", config.backend);
            Err(GpuError::UnsupportedBackend(config.backend.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GpuMonitorConfig;
    use std::time::Duration;

    #[tokio::test]
    async fn test_gpu_monitor_creation() {
        let config = GpuMonitorConfig::new(GpuBackend::Mock)
            .with_polling_interval(Duration::from_secs(1));
        
        // This would work if we had the mock implementation
        // let monitor = GpuMonitor::new(config).await;
        // assert!(monitor.is_ok());
        
        // For now, just test config validation
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_unsupported_backend() {
        // Test that unsupported backends return appropriate errors
        let error = GpuError::UnsupportedBackend("test-backend".to_string());
        assert!(error.to_string().contains("test-backend"));
        assert!(!error.is_retryable());
    }
}
