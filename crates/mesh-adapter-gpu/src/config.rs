//! GPU monitoring configuration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// GPU monitoring backends
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuBackend {
    /// NVIDIA Management Library
    Nvml,
    /// NVIDIA Data Center GPU Manager
    Dcgm,
    /// AMD ROCm
    Rocm,
    /// Mock backend for testing
    Mock,
}

/// GPU monitor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMonitorConfig {
    /// GPU backend to use
    pub backend: GpuBackend,
    
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
    
    /// Health check configuration
    pub health_check: HealthCheckConfig,
    
    /// Backend-specific configuration
    pub backend_config: HashMap<String, serde_json::Value>,
    
    /// GPU filter (specific GPU IDs to monitor)
    pub gpu_filter: Option<Vec<u32>>,
    
    /// Enable detailed metrics collection
    pub detailed_metrics: bool,
    
    /// Enable MIG monitoring
    pub mig_monitoring: bool,
    
    /// Enable ECC error monitoring
    pub ecc_monitoring: bool,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Polling interval for metrics collection
    pub polling_interval: Duration,
    
    /// Metrics retention period
    pub retention_period: Duration,
    
    /// Maximum number of metrics samples to keep
    pub max_samples: usize,
    
    /// Enable real-time monitoring
    pub real_time: bool,
    
    /// Metrics to collect
    pub metrics: Vec<String>,
    
    /// Enable metric aggregation
    pub enable_aggregation: bool,
    
    /// Aggregation window size
    pub aggregation_window: Duration,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check interval
    pub interval: Duration,
    
    /// Health check timeout
    pub timeout: Duration,
    
    /// Temperature threshold (Celsius)
    pub temperature_threshold: f64,
    
    /// Memory usage threshold (percentage)
    pub memory_threshold: f64,
    
    /// Power usage threshold (watts)
    pub power_threshold: f64,
    
    /// ECC error threshold
    pub ecc_error_threshold: u64,
    
    /// Enable automatic GPU throttling on overheating
    pub auto_throttle: bool,
}

/// NVML-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NvmlConfig {
    /// NVML library path (optional)
    pub library_path: Option<PathBuf>,
    
    /// Enable persistent mode
    pub persistent_mode: bool,
    
    /// Enable accounting mode
    pub accounting_mode: bool,
    
    /// GPU compute mode
    pub compute_mode: Option<ComputeMode>,
}

/// DCGM-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DcgmConfig {
    /// DCGM host address
    pub host: String,
    
    /// DCGM port
    pub port: u16,
    
    /// Connection timeout
    pub connection_timeout: Duration,
    
    /// Field update frequency
    pub update_frequency: Duration,
    
    /// Maximum keep age for samples
    pub max_keep_age: Duration,
    
    /// Maximum keep samples
    pub max_keep_samples: u32,
}

/// ROCm-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocmConfig {
    /// ROCm installation path
    pub rocm_path: PathBuf,
    
    /// Enable ROCm SMI
    pub enable_smi: bool,
    
    /// SMI update interval
    pub smi_interval: Duration,
}

/// GPU compute modes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeMode {
    /// Default compute mode
    Default,
    /// Exclusive thread mode
    ExclusiveThread,
    /// Prohibited mode
    Prohibited,
    /// Exclusive process mode
    ExclusiveProcess,
}

impl GpuMonitorConfig {
    /// Create a new GPU monitor configuration
    pub fn new(backend: GpuBackend) -> Self {
        Self {
            backend,
            monitoring: MonitoringConfig::default(),
            health_check: HealthCheckConfig::default(),
            backend_config: HashMap::new(),
            gpu_filter: None,
            detailed_metrics: false,
            mig_monitoring: true,
            ecc_monitoring: true,
        }
    }

    /// Set the polling interval
    pub fn with_polling_interval(mut self, interval: Duration) -> Self {
        self.monitoring.polling_interval = interval;
        self
    }

    /// Enable detailed metrics
    pub fn with_detailed_metrics(mut self, enabled: bool) -> Self {
        self.detailed_metrics = enabled;
        self
    }

    /// Set GPU filter
    pub fn with_gpu_filter(mut self, gpu_ids: Vec<u32>) -> Self {
        self.gpu_filter = Some(gpu_ids);
        self
    }

    /// Enable MIG monitoring
    pub fn with_mig_monitoring(mut self, enabled: bool) -> Self {
        self.mig_monitoring = enabled;
        self
    }

    /// Enable ECC monitoring
    pub fn with_ecc_monitoring(mut self, enabled: bool) -> Self {
        self.ecc_monitoring = enabled;
        self
    }

    /// Set health check thresholds
    pub fn with_health_thresholds(
        mut self,
        temperature: f64,
        memory: f64,
        power: f64,
    ) -> Self {
        self.health_check.temperature_threshold = temperature;
        self.health_check.memory_threshold = memory;
        self.health_check.power_threshold = power;
        self
    }

    /// Set backend-specific configuration
    pub fn with_backend_config(mut self, key: String, value: serde_json::Value) -> Self {
        self.backend_config.insert(key, value);
        self
    }

    /// Get backend-specific configuration value
    pub fn get_backend_config(&self, key: &str) -> Option<&serde_json::Value> {
        self.backend_config.get(key)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate polling interval
        if self.monitoring.polling_interval.is_zero() {
            return Err("Polling interval must be greater than zero".to_string());
        }

        // Validate health check thresholds
        if self.health_check.temperature_threshold <= 0.0 {
            return Err("Temperature threshold must be positive".to_string());
        }

        if self.health_check.memory_threshold < 0.0 || self.health_check.memory_threshold > 100.0 {
            return Err("Memory threshold must be between 0 and 100".to_string());
        }

        if self.health_check.power_threshold <= 0.0 {
            return Err("Power threshold must be positive".to_string());
        }

        // Validate retention settings
        if self.monitoring.max_samples == 0 {
            return Err("Max samples must be greater than zero".to_string());
        }

        // Backend-specific validation
        match self.backend {
            GpuBackend::Dcgm => {
                if let Some(host_config) = self.get_backend_config("host") {
                    if !host_config.is_string() {
                        return Err("DCGM host must be a string".to_string());
                    }
                }
            }
            GpuBackend::Rocm => {
                if let Some(path_config) = self.get_backend_config("rocm_path") {
                    if !path_config.is_string() {
                        return Err("ROCm path must be a string".to_string());
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            polling_interval: Duration::from_secs(1),
            retention_period: Duration::from_secs(3600), // 1 hour
            max_samples: 3600, // 1 hour at 1 second intervals
            real_time: true,
            metrics: vec![
                "gpu_utilization".to_string(),
                "memory_utilization".to_string(),
                "temperature".to_string(),
                "power_usage".to_string(),
                "fan_speed".to_string(),
                "clock_speeds".to_string(),
            ],
            enable_aggregation: true,
            aggregation_window: Duration::from_secs(60),
        }
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            temperature_threshold: 85.0, // 85°C
            memory_threshold: 90.0,      // 90%
            power_threshold: 300.0,      // 300W
            ecc_error_threshold: 10,     // 10 errors
            auto_throttle: false,
        }
    }
}

impl Default for NvmlConfig {
    fn default() -> Self {
        Self {
            library_path: None,
            persistent_mode: true,
            accounting_mode: false,
            compute_mode: None,
        }
    }
}

impl Default for DcgmConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 5555,
            connection_timeout: Duration::from_secs(10),
            update_frequency: Duration::from_secs(1),
            max_keep_age: Duration::from_secs(3600),
            max_keep_samples: 3600,
        }
    }
}

impl Default for RocmConfig {
    fn default() -> Self {
        Self {
            rocm_path: PathBuf::from("/opt/rocm"),
            enable_smi: true,
            smi_interval: Duration::from_secs(1),
        }
    }
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackend::Nvml => write!(f, "nvml"),
            GpuBackend::Dcgm => write!(f, "dcgm"),
            GpuBackend::Rocm => write!(f, "rocm"),
            GpuBackend::Mock => write!(f, "mock"),
        }
    }
}

impl std::str::FromStr for GpuBackend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "nvml" => Ok(GpuBackend::Nvml),
            "dcgm" => Ok(GpuBackend::Dcgm),
            "rocm" => Ok(GpuBackend::Rocm),
            "mock" => Ok(GpuBackend::Mock),
            _ => Err(format!("Unknown GPU backend: {}", s)),
        }
    }
}

impl std::fmt::Display for ComputeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputeMode::Default => write!(f, "default"),
            ComputeMode::ExclusiveThread => write!(f, "exclusive_thread"),
            ComputeMode::Prohibited => write!(f, "prohibited"),
            ComputeMode::ExclusiveProcess => write!(f, "exclusive_process"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_monitor_config_creation() {
        let config = GpuMonitorConfig::new(GpuBackend::Nvml);
        assert_eq!(config.backend, GpuBackend::Nvml);
        assert!(!config.detailed_metrics);
        assert!(config.mig_monitoring);
        assert!(config.ecc_monitoring);
    }

    #[test]
    fn test_gpu_monitor_config_builder() {
        let config = GpuMonitorConfig::new(GpuBackend::Dcgm)
            .with_polling_interval(Duration::from_millis(500))
            .with_detailed_metrics(true)
            .with_gpu_filter(vec![0, 1, 2])
            .with_health_thresholds(80.0, 85.0, 250.0);

        assert_eq!(config.monitoring.polling_interval, Duration::from_millis(500));
        assert!(config.detailed_metrics);
        assert_eq!(config.gpu_filter, Some(vec![0, 1, 2]));
        assert_eq!(config.health_check.temperature_threshold, 80.0);
        assert_eq!(config.health_check.memory_threshold, 85.0);
        assert_eq!(config.health_check.power_threshold, 250.0);
    }

    #[test]
    fn test_gpu_backend_parsing() {
        assert_eq!("nvml".parse::<GpuBackend>().unwrap(), GpuBackend::Nvml);
        assert_eq!("dcgm".parse::<GpuBackend>().unwrap(), GpuBackend::Dcgm);
        assert_eq!("rocm".parse::<GpuBackend>().unwrap(), GpuBackend::Rocm);
        assert_eq!("mock".parse::<GpuBackend>().unwrap(), GpuBackend::Mock);
        assert!("unknown".parse::<GpuBackend>().is_err());
    }

    #[test]
    fn test_gpu_backend_display() {
        assert_eq!(GpuBackend::Nvml.to_string(), "nvml");
        assert_eq!(GpuBackend::Dcgm.to_string(), "dcgm");
        assert_eq!(GpuBackend::Rocm.to_string(), "rocm");
        assert_eq!(GpuBackend::Mock.to_string(), "mock");
    }

    #[test]
    fn test_config_validation() {
        let mut config = GpuMonitorConfig::new(GpuBackend::Nvml);
        
        // Should pass with default values
        assert!(config.validate().is_ok());
        
        // Should fail with zero polling interval
        config.monitoring.polling_interval = Duration::from_secs(0);
        assert!(config.validate().is_err());
        
        // Should fail with invalid temperature threshold
        config.monitoring.polling_interval = Duration::from_secs(1);
        config.health_check.temperature_threshold = -10.0;
        assert!(config.validate().is_err());
        
        // Should fail with invalid memory threshold
        config.health_check.temperature_threshold = 85.0;
        config.health_check.memory_threshold = 150.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_monitoring_config_default() {
        let config = MonitoringConfig::default();
        assert_eq!(config.polling_interval, Duration::from_secs(1));
        assert_eq!(config.max_samples, 3600);
        assert!(config.real_time);
        assert!(!config.metrics.is_empty());
        assert!(config.enable_aggregation);
    }

    #[test]
    fn test_health_check_config_default() {
        let config = HealthCheckConfig::default();
        assert_eq!(config.interval, Duration::from_secs(30));
        assert_eq!(config.temperature_threshold, 85.0);
        assert_eq!(config.memory_threshold, 90.0);
        assert_eq!(config.power_threshold, 300.0);
        assert!(!config.auto_throttle);
    }

    #[test]
    fn test_compute_mode_display() {
        assert_eq!(ComputeMode::Default.to_string(), "default");
        assert_eq!(ComputeMode::ExclusiveThread.to_string(), "exclusive_thread");
        assert_eq!(ComputeMode::Prohibited.to_string(), "prohibited");
        assert_eq!(ComputeMode::ExclusiveProcess.to_string(), "exclusive_process");
    }
}
