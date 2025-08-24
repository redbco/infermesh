//! Mock GPU monitor for testing

use crate::config::GpuMonitorConfig;
use crate::monitor::GpuMonitorTrait;
use crate::metrics::*;
use crate::health::{GpuHealth, HealthStatus};
use crate::{Result, GpuError};

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Mock GPU monitor for testing
pub struct MockGpuMonitor {
    config: GpuMonitorConfig,
    gpus: Arc<RwLock<HashMap<u32, GpuMetrics>>>,
    monitoring: Arc<RwLock<bool>>,
}

impl MockGpuMonitor {
    /// Create a new mock GPU monitor
    pub async fn new(config: GpuMonitorConfig) -> Result<Self> {
        info!("Creating mock GPU monitor");

        let mut gpus = HashMap::new();
        
        // Create mock GPUs
        let gpu_count = config.gpu_filter.as_ref().map(|f| f.len()).unwrap_or(2) as u32;
        
        for i in 0..gpu_count {
            let gpu_info = create_mock_gpu_info(i);
            let mut metrics = GpuMetrics::new(gpu_info);
            
            // Set some realistic mock values
            metrics.memory.total = 8 * 1024 * 1024 * 1024; // 8GB
            metrics.memory.used = (2 * 1024 * 1024 * 1024) + (i as u64 * 512 * 1024 * 1024); // 2GB + 512MB per GPU
            metrics.memory.free = metrics.memory.total - metrics.memory.used;
            metrics.memory.utilization = (metrics.memory.used as f64 / metrics.memory.total as f64) * 100.0;
            
            metrics.temperature.gpu = 65.0 + (i as f64 * 5.0); // 65°C, 70°C, etc.
            metrics.temperature.thermal_state = ThermalState::Normal;
            
            metrics.power.usage = 150.0 + (i as f64 * 25.0); // 150W, 175W, etc.
            metrics.power.limit = 300.0;
            metrics.power.utilization = (metrics.power.usage / metrics.power.limit) * 100.0;
            
            metrics.clocks.graphics = 1500 + (i * 100); // 1500MHz, 1600MHz, etc.
            metrics.clocks.memory = 6000 + (i * 200); // 6000MHz, 6200MHz, etc.
            
            metrics.utilization.gpu = 45.0 + (i as f64 * 10.0); // 45%, 55%, etc.
            metrics.utilization.memory = 30.0 + (i as f64 * 8.0); // 30%, 38%, etc.
            
            // Add a mock fan
            metrics.fans.push(FanInfo {
                index: 0,
                speed_rpm: 2000 + (i * 200),
                speed_percent: 60.0 + (i as f64 * 5.0),
                target_speed: Some(65.0),
                control_mode: FanControlMode::Auto,
            });
            
            gpus.insert(i, metrics);
        }

        Ok(Self {
            config,
            gpus: Arc::new(RwLock::new(gpus)),
            monitoring: Arc::new(RwLock::new(false)),
        })
    }
}

#[async_trait]
impl GpuMonitorTrait for MockGpuMonitor {
    async fn initialize(&mut self) -> Result<()> {
        info!("Initializing mock GPU monitor");
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down mock GPU monitor");
        *self.monitoring.write().await = false;
        Ok(())
    }

    async fn discover_gpus(&self) -> Result<Vec<u32>> {
        let gpus = self.gpus.read().await;
        let mut gpu_ids: Vec<u32> = gpus.keys().cloned().collect();
        gpu_ids.sort(); // Ensure consistent ordering
        debug!("Discovered {} mock GPUs: {:?}", gpu_ids.len(), gpu_ids);
        Ok(gpu_ids)
    }

    async fn get_gpu_metrics(&self, gpu_id: u32) -> Result<GpuMetrics> {
        let gpus = self.gpus.read().await;
        let metrics = gpus.get(&gpu_id)
            .ok_or_else(|| GpuError::GpuNotFound(gpu_id.to_string()))?
            .clone();
        
        debug!("Retrieved metrics for mock GPU {}", gpu_id);
        Ok(metrics)
    }

    async fn get_all_metrics(&self) -> Result<Vec<GpuMetrics>> {
        let gpus = self.gpus.read().await;
        let all_metrics: Vec<GpuMetrics> = gpus.values().cloned().collect();
        debug!("Retrieved metrics for {} mock GPUs", all_metrics.len());
        Ok(all_metrics)
    }

    async fn start_monitoring(&mut self) -> Result<()> {
        info!("Starting mock GPU monitoring");
        *self.monitoring.write().await = true;
        
        // In a real implementation, this would start background tasks
        // to continuously update metrics
        
        Ok(())
    }

    async fn stop_monitoring(&mut self) -> Result<()> {
        info!("Stopping mock GPU monitoring");
        *self.monitoring.write().await = false;
        Ok(())
    }

    fn is_monitoring(&self) -> bool {
        // This is a simplified check - in a real async context we'd need to handle this differently
        false // Placeholder
    }

    async fn get_gpu_health(&self, gpu_id: u32) -> Result<GpuHealth> {
        let metrics = self.get_gpu_metrics(gpu_id).await?;
        let health = GpuHealth::from_metrics(&metrics);
        debug!("Retrieved health for mock GPU {}: {:?}", gpu_id, health.status);
        Ok(health)
    }

    fn get_config(&self) -> &GpuMonitorConfig {
        &self.config
    }
}

/// Create mock GPU info
fn create_mock_gpu_info(index: u32) -> GpuInfo {
    GpuInfo {
        index,
        uuid: format!("GPU-{:08x}-{:04x}-{:04x}-{:04x}-{:012x}", 
                     0x12345678, 0x1234, 0x5678, 0x9abc, 0xdef012345678u64 + index as u64),
        name: format!("Mock GPU {}", index),
        brand: "MockVendor".to_string(),
        architecture: Some("MockArch".to_string()),
        driver_version: "999.99.99".to_string(),
        vbios_version: Some("99.00.99.00.01".to_string()),
        pci_info: PciInfo {
            bus_id: format!("0000:{:02x}:00.0", index + 1),
            domain: 0,
            bus: index + 1,
            device: 0,
            function: 0,
            device_id: 0x1234 + index,
            subsystem_id: 0x5678 + index,
            link_gen: Some(4),
            link_width: Some(16),
            max_link_gen: Some(4),
            max_link_width: Some(16),
        },
        capabilities: GpuCapabilities {
            cuda_compute_major: Some(8),
            cuda_compute_minor: Some(6),
            multi_gpu_board: false,
            mig_support: true,
            ecc_support: true,
            tcc_support: false,
            virtualization_support: true,
            graphics_apis: vec!["OpenGL".to_string(), "Vulkan".to_string()],
            compute_apis: vec!["CUDA".to_string(), "OpenCL".to_string()],
        },
        total_memory: 8 * 1024 * 1024 * 1024, // 8GB
        memory_bus_width: Some(256),
        memory_type: Some("GDDR6".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{GpuBackend, GpuMonitorConfig};

    #[tokio::test]
    async fn test_mock_gpu_monitor_creation() {
        let config = GpuMonitorConfig::new(GpuBackend::Mock);
        let monitor = MockGpuMonitor::new(config).await;
        assert!(monitor.is_ok());
    }

    #[tokio::test]
    async fn test_mock_gpu_discovery() {
        let config = GpuMonitorConfig::new(GpuBackend::Mock);
        let monitor = MockGpuMonitor::new(config).await.unwrap();
        
        let gpus = monitor.discover_gpus().await.unwrap();
        assert_eq!(gpus.len(), 2); // Default mock GPU count
        assert_eq!(gpus, vec![0, 1]);
    }

    #[tokio::test]
    async fn test_mock_gpu_metrics() {
        let config = GpuMonitorConfig::new(GpuBackend::Mock);
        let monitor = MockGpuMonitor::new(config).await.unwrap();
        
        let metrics = monitor.get_gpu_metrics(0).await.unwrap();
        assert_eq!(metrics.info.index, 0);
        assert_eq!(metrics.info.name, "Mock GPU 0");
        assert_eq!(metrics.info.brand, "MockVendor");
        assert!(metrics.memory.total > 0);
        assert!(metrics.temperature.gpu > 0.0);
    }

    #[tokio::test]
    async fn test_mock_gpu_health() {
        let config = GpuMonitorConfig::new(GpuBackend::Mock);
        let monitor = MockGpuMonitor::new(config).await.unwrap();
        
        let health = monitor.get_gpu_health(0).await.unwrap();
        assert_eq!(health.gpu_id, 0);
        assert!(health.is_healthy()); // Mock GPUs should be healthy by default
        assert!(health.score > 0.0);
    }

    #[tokio::test]
    async fn test_mock_gpu_not_found() {
        let config = GpuMonitorConfig::new(GpuBackend::Mock);
        let monitor = MockGpuMonitor::new(config).await.unwrap();
        
        let result = monitor.get_gpu_metrics(999).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GpuError::GpuNotFound(_)));
    }

    #[tokio::test]
    async fn test_mock_monitoring_lifecycle() {
        let config = GpuMonitorConfig::new(GpuBackend::Mock);
        let mut monitor = MockGpuMonitor::new(config).await.unwrap();
        
        // Test initialization
        assert!(monitor.initialize().await.is_ok());
        
        // Test monitoring start/stop
        assert!(monitor.start_monitoring().await.is_ok());
        assert!(monitor.stop_monitoring().await.is_ok());
        
        // Test shutdown
        assert!(monitor.shutdown().await.is_ok());
    }

    #[test]
    fn test_create_mock_gpu_info() {
        let info = create_mock_gpu_info(0);
        assert_eq!(info.index, 0);
        assert_eq!(info.name, "Mock GPU 0");
        assert_eq!(info.brand, "MockVendor");
        assert!(info.uuid.starts_with("GPU-"));
        assert_eq!(info.pci_info.bus, 1);
        assert_eq!(info.total_memory, 8 * 1024 * 1024 * 1024);
    }
}
