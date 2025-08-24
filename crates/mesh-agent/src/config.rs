//! Configuration management for the mesh agent

use mesh_core::Config as CoreConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;

/// Complete configuration for the mesh agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Core mesh configuration
    #[serde(flatten)]
    pub core: CoreConfig,
    
    /// Agent-specific configuration
    pub agent: AgentSpecificConfig,
    
    /// Logging configuration
    pub logging: LoggingConfig,
    
    /// Service configuration
    pub services: ServicesConfig,
}

/// Agent-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSpecificConfig {
    /// Agent name/identifier
    pub name: String,
    
    /// Data directory for the agent
    pub data_dir: PathBuf,
    
    /// PID file location
    pub pid_file: Option<PathBuf>,
    
    /// Whether to run in daemon mode
    pub daemon: bool,
    
    /// Graceful shutdown timeout (seconds)
    pub shutdown_timeout_seconds: u64,
    
    /// Health check configuration
    pub health_check: HealthCheckConfig,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Enable health checks
    pub enabled: bool,
    
    /// Health check interval (seconds)
    pub interval_seconds: u64,
    
    /// Health check timeout (seconds)
    pub timeout_seconds: u64,
    
    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: u32,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    
    /// Log format (text, json)
    pub format: String,
    
    /// Show target in logs
    pub show_target: bool,
    
    /// Show thread IDs in logs
    pub show_thread_ids: bool,
    
    /// Show line numbers in logs
    pub show_line_numbers: bool,
    
    /// Log file path (optional)
    pub file: Option<PathBuf>,
    
    /// Log rotation configuration
    pub rotation: LogRotationConfig,
}

/// Log rotation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRotationConfig {
    /// Enable log rotation
    pub enabled: bool,
    
    /// Maximum file size before rotation (MB)
    pub max_size_mb: u64,
    
    /// Maximum number of rotated files to keep
    pub max_files: u32,
}

/// Services configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicesConfig {
    /// Control plane service configuration
    pub control_plane: ServiceConfig,
    
    /// State plane service configuration
    pub state_plane: ServiceConfig,
    
    /// Scoring service configuration
    pub scoring: ServiceConfig,
    
    /// Metrics service configuration
    pub metrics: MetricsServiceConfig,
    
    /// Adapter configuration
    pub adapters: AdapterConfig,
}

/// Individual service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    /// Enable this service
    pub enabled: bool,
    
    /// Service bind address
    pub bind_addr: SocketAddr,
    
    /// Service-specific configuration
    pub config: HashMap<String, serde_json::Value>,
}

/// Metrics service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsServiceConfig {
    /// Enable metrics service
    pub enabled: bool,
    
    /// Metrics bind address
    pub bind_addr: SocketAddr,
    
    /// Prometheus configuration
    pub prometheus: PrometheusConfig,
    
    /// OpenTelemetry configuration
    #[cfg(feature = "opentelemetry")]
    pub opentelemetry: OpenTelemetryConfig,
}

/// Prometheus configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusConfig {
    /// Enable Prometheus metrics
    pub enabled: bool,
    
    /// Metrics collection interval (seconds)
    pub collection_interval_seconds: u64,
    
    /// Additional labels to add to all metrics
    pub global_labels: HashMap<String, String>,
}

/// OpenTelemetry configuration
#[cfg(feature = "opentelemetry")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenTelemetryConfig {
    /// Enable OpenTelemetry
    pub enabled: bool,
    
    /// OTLP endpoint
    pub otlp_endpoint: Option<String>,
    
    /// Service name for tracing
    pub service_name: String,
    
    /// Service version for tracing
    pub service_version: String,
}

/// Adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterConfig {
    /// Runtime adapter configuration
    pub runtime: RuntimeAdapterConfig,
    
    /// GPU adapter configuration
    pub gpu: GpuAdapterConfig,
    
    /// Telemetry streaming configuration
    pub telemetry: TelemetryConfig,
}

/// Runtime adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeAdapterConfig {
    /// Enable runtime adapters
    pub enabled: bool,
    
    /// vLLM adapter configuration
    pub vllm: Option<RuntimeInstanceConfig>,
    
    /// Triton adapter configuration
    pub triton: Option<RuntimeInstanceConfig>,
    
    /// TGI adapter configuration
    pub tgi: Option<RuntimeInstanceConfig>,
    
    /// Health check interval (seconds)
    pub health_check_interval_seconds: u64,
    
    /// Metrics collection interval (seconds)
    pub metrics_collection_interval_seconds: u64,
}

/// GPU adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAdapterConfig {
    /// Enable GPU adapters
    pub enabled: bool,
    
    /// NVML adapter configuration
    pub nvml: Option<GpuInstanceConfig>,
    
    /// DCGM adapter configuration
    pub dcgm: Option<GpuInstanceConfig>,
    
    /// ROCm adapter configuration
    pub rocm: Option<GpuInstanceConfig>,
    
    /// Health check interval (seconds)
    pub health_check_interval_seconds: u64,
    
    /// Metrics collection interval (seconds)
    pub metrics_collection_interval_seconds: u64,
}

/// Runtime instance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeInstanceConfig {
    /// Enable this runtime instance
    pub enabled: bool,
    
    /// Runtime endpoint URL
    pub endpoint: String,
    
    /// Connection timeout (seconds)
    pub timeout_seconds: u64,
    
    /// Maximum retries for requests
    pub max_retries: u32,
    
    /// Instance-specific configuration
    pub config: HashMap<String, serde_json::Value>,
}

/// GPU instance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInstanceConfig {
    /// Enable this GPU adapter instance
    pub enabled: bool,
    
    /// Instance-specific configuration
    pub config: HashMap<String, serde_json::Value>,
}

/// Telemetry streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Enable telemetry streaming
    pub enabled: bool,
    
    /// Buffer size for telemetry data
    pub buffer_size: usize,
    
    /// Flush interval (seconds)
    pub flush_interval_seconds: u64,
    
    /// Maximum batch size for streaming
    pub max_batch_size: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            core: CoreConfig::default(),
            agent: AgentSpecificConfig::default(),
            logging: LoggingConfig::default(),
            services: ServicesConfig::default(),
        }
    }
}

impl Default for AgentSpecificConfig {
    fn default() -> Self {
        Self {
            name: "mesh-agent".to_string(),
            data_dir: PathBuf::from("/var/lib/infermesh"),
            pid_file: Some(PathBuf::from("/var/run/infermesh/meshd.pid")),
            daemon: false,
            shutdown_timeout_seconds: 30,
            health_check: HealthCheckConfig::default(),
        }
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_seconds: 30,
            timeout_seconds: 5,
            failure_threshold: 3,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "text".to_string(),
            show_target: true,
            show_thread_ids: false,
            show_line_numbers: false,
            file: None,
            rotation: LogRotationConfig::default(),
        }
    }
}

impl Default for LogRotationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_size_mb: 100,
            max_files: 10,
        }
    }
}

impl Default for ServicesConfig {
    fn default() -> Self {
        Self {
            control_plane: ServiceConfig {
                enabled: true,
                bind_addr: "127.0.0.1:50051".parse().unwrap(),
                config: HashMap::new(),
            },
            state_plane: ServiceConfig {
                enabled: true,
                bind_addr: "127.0.0.1:50052".parse().unwrap(),
                config: HashMap::new(),
            },
            scoring: ServiceConfig {
                enabled: true,
                bind_addr: "127.0.0.1:50053".parse().unwrap(),
                config: HashMap::new(),
            },
            metrics: MetricsServiceConfig::default(),
            adapters: AdapterConfig::default(),
        }
    }
}

impl Default for MetricsServiceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bind_addr: "127.0.0.1:9090".parse().unwrap(),
            prometheus: PrometheusConfig::default(),
            #[cfg(feature = "opentelemetry")]
            opentelemetry: OpenTelemetryConfig::default(),
        }
    }
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval_seconds: 15,
            global_labels: {
                let mut labels = HashMap::new();
                labels.insert("service".to_string(), "infermesh".to_string());
                labels
            },
        }
    }
}

#[cfg(feature = "opentelemetry")]
impl Default for OpenTelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            otlp_endpoint: None,
            service_name: "infermesh-agent".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            runtime: RuntimeAdapterConfig::default(),
            gpu: GpuAdapterConfig::default(),
            telemetry: TelemetryConfig::default(),
        }
    }
}

impl Default for RuntimeAdapterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            vllm: Some(RuntimeInstanceConfig {
                enabled: false,
                endpoint: "http://localhost:8000".to_string(),
                timeout_seconds: 30,
                max_retries: 3,
                config: HashMap::new(),
            }),
            triton: Some(RuntimeInstanceConfig {
                enabled: false,
                endpoint: "http://localhost:8001".to_string(),
                timeout_seconds: 30,
                max_retries: 3,
                config: HashMap::new(),
            }),
            tgi: Some(RuntimeInstanceConfig {
                enabled: false,
                endpoint: "http://localhost:8080".to_string(),
                timeout_seconds: 30,
                max_retries: 3,
                config: HashMap::new(),
            }),
            health_check_interval_seconds: 30,
            metrics_collection_interval_seconds: 15,
        }
    }
}

impl Default for GpuAdapterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            nvml: Some(GpuInstanceConfig {
                enabled: true,
                config: HashMap::new(),
            }),
            dcgm: Some(GpuInstanceConfig {
                enabled: false,
                config: HashMap::new(),
            }),
            rocm: Some(GpuInstanceConfig {
                enabled: false,
                config: HashMap::new(),
            }),
            health_check_interval_seconds: 30,
            metrics_collection_interval_seconds: 15,
        }
    }
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            buffer_size: 1000,
            flush_interval_seconds: 5,
            max_batch_size: 100,
        }
    }
}

impl AgentConfig {
    /// Load configuration from a file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::AgentError::Config(format!("Failed to read config file: {}", e)))?;
        
        let config: AgentConfig = serde_yaml::from_str(&content)
            .map_err(|e| crate::AgentError::Config(format!("Failed to parse config: {}", e)))?;
        
        Ok(config)
    }
    
    /// Save configuration to a file
    pub fn to_file<P: AsRef<std::path::Path>>(&self, path: P) -> crate::Result<()> {
        let content = serde_yaml::to_string(self)
            .map_err(|e| crate::AgentError::Config(format!("Failed to serialize config: {}", e)))?;
        
        std::fs::write(path, content)
            .map_err(|e| crate::AgentError::Config(format!("Failed to write config file: {}", e)))?;
        
        Ok(())
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> crate::Result<()> {
        // Validate core configuration
        self.core.validate()
            .map_err(|e| crate::AgentError::Config(format!("Core config validation failed: {}", e)))?;
        
        // Validate agent-specific configuration
        if self.agent.name.is_empty() {
            return Err(crate::AgentError::Config("Agent name cannot be empty".to_string()));
        }
        
        if self.agent.shutdown_timeout_seconds == 0 {
            return Err(crate::AgentError::Config("Shutdown timeout must be greater than 0".to_string()));
        }
        
        // Validate logging configuration
        match self.logging.level.as_str() {
            "trace" | "debug" | "info" | "warn" | "error" => {}
            _ => return Err(crate::AgentError::Config(format!("Invalid log level: {}", self.logging.level))),
        }
        
        match self.logging.format.as_str() {
            "text" | "json" => {}
            _ => return Err(crate::AgentError::Config(format!("Invalid log format: {}", self.logging.format))),
        }
        
        Ok(())
    }
    
    /// Get the effective data directory (create if it doesn't exist)
    pub fn ensure_data_dir(&self) -> crate::Result<PathBuf> {
        let data_dir = &self.agent.data_dir;
        
        if !data_dir.exists() {
            std::fs::create_dir_all(data_dir)
                .map_err(|e| crate::AgentError::Config(format!("Failed to create data directory: {}", e)))?;
        }
        
        Ok(data_dir.clone())
    }
    
    /// Get the PID file path
    pub fn pid_file_path(&self) -> Option<&PathBuf> {
        self.agent.pid_file.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();
        assert_eq!(config.agent.name, "mesh-agent");
        assert!(config.services.control_plane.enabled);
        assert!(config.services.metrics.enabled);
    }

    #[test]
    fn test_agent_config_validation() {
        let mut config = AgentConfig::default();
        
        // Valid config should pass
        config.validate().unwrap();
        
        // Invalid agent name should fail
        config.agent.name = String::new();
        assert!(config.validate().is_err());
        
        // Reset and test invalid log level
        config.agent.name = "test".to_string();
        config.logging.level = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_file_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.yaml");
        
        let config = AgentConfig::default();
        
        // Test saving
        config.to_file(&config_path).unwrap();
        assert!(config_path.exists());
        
        // Test loading
        let loaded_config = AgentConfig::from_file(&config_path).unwrap();
        assert_eq!(config.agent.name, loaded_config.agent.name);
        assert_eq!(config.logging.level, loaded_config.logging.level);
    }

    #[test]
    fn test_ensure_data_dir() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = AgentConfig::default();
        config.agent.data_dir = temp_dir.path().join("test_data");
        
        let data_dir = config.ensure_data_dir().unwrap();
        assert!(data_dir.exists());
        assert!(data_dir.is_dir());
    }
}
