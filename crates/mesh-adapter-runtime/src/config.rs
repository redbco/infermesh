//! Runtime configuration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use url::Url;

/// Runtime types supported by the adapter
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeType {
    /// NVIDIA Triton Inference Server
    Triton,
    /// vLLM high-throughput LLM serving
    VLlm,
    /// Hugging Face Text Generation Inference
    Tgi,
    /// PyTorch TorchServe
    TorchServe,
    /// TensorFlow Serving
    TensorFlowServing,
}

/// Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Runtime type
    pub runtime_type: RuntimeType,
    
    /// Runtime endpoint URL
    pub endpoint: Url,
    
    /// Management endpoint (if different from main endpoint)
    pub management_endpoint: Option<Url>,
    
    /// Model repository path
    pub model_repository: Option<PathBuf>,
    
    /// Request timeout
    pub request_timeout: Duration,
    
    /// Health check configuration
    pub health_check: HealthCheckConfig,
    
    /// Metrics collection configuration
    pub metrics: MetricsConfig,
    
    /// Process management configuration
    pub process: Option<ProcessConfig>,
    
    /// Runtime-specific configuration
    pub runtime_specific: HashMap<String, serde_json::Value>,
    
    /// Authentication configuration
    pub auth: Option<AuthConfig>,
    
    /// TLS configuration
    pub tls: Option<TlsConfig>,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check interval
    pub interval: Duration,
    
    /// Health check timeout
    pub timeout: Duration,
    
    /// Number of consecutive failures to mark unhealthy
    pub failure_threshold: u32,
    
    /// Number of consecutive successes to mark healthy
    pub success_threshold: u32,
    
    /// Custom health check endpoint
    pub endpoint: Option<String>,
}

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enabled: bool,
    
    /// Metrics collection interval
    pub interval: Duration,
    
    /// Custom metrics endpoint
    pub endpoint: Option<String>,
    
    /// Metrics to collect
    pub metrics: Vec<String>,
    
    /// Enable detailed model metrics
    pub detailed_model_metrics: bool,
}

/// Process management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessConfig {
    /// Command to start the runtime
    pub command: String,
    
    /// Command arguments
    pub args: Vec<String>,
    
    /// Environment variables
    pub env: HashMap<String, String>,
    
    /// Working directory
    pub working_dir: Option<PathBuf>,
    
    /// Process startup timeout
    pub startup_timeout: Duration,
    
    /// Process shutdown timeout
    pub shutdown_timeout: Duration,
    
    /// Auto-restart on failure
    pub auto_restart: bool,
    
    /// Maximum restart attempts
    pub max_restarts: u32,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Authentication type
    pub auth_type: AuthType,
    
    /// API key (for API key auth)
    pub api_key: Option<String>,
    
    /// Token (for bearer token auth)
    pub token: Option<String>,
    
    /// Username (for basic auth)
    pub username: Option<String>,
    
    /// Password (for basic auth)
    pub password: Option<String>,
}

/// Authentication types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthType {
    None,
    ApiKey,
    BearerToken,
    BasicAuth,
}

/// TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Enable TLS
    pub enabled: bool,
    
    /// CA certificate path
    pub ca_cert: Option<PathBuf>,
    
    /// Client certificate path
    pub client_cert: Option<PathBuf>,
    
    /// Client private key path
    pub client_key: Option<PathBuf>,
    
    /// Skip certificate verification
    pub insecure: bool,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name
    pub name: String,
    
    /// Model version (optional)
    pub version: Option<String>,
    
    /// Model path or URL
    pub path: String,
    
    /// Model format (e.g., "pytorch", "onnx", "tensorrt")
    pub format: Option<String>,
    
    /// Model configuration parameters
    pub config: HashMap<String, serde_json::Value>,
    
    /// Resource requirements
    pub resources: ResourceRequirements,
    
    /// Model-specific settings
    pub settings: HashMap<String, serde_json::Value>,
}

/// Resource requirements for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// CPU cores required
    pub cpu_cores: Option<f64>,
    
    /// Memory required in bytes
    pub memory_bytes: Option<u64>,
    
    /// GPU memory required in bytes
    pub gpu_memory_bytes: Option<u64>,
    
    /// Number of GPUs required
    pub gpu_count: Option<u32>,
    
    /// Specific GPU types required
    pub gpu_types: Vec<String>,
}

impl RuntimeConfig {
    /// Create a new runtime configuration
    pub fn new(runtime_type: RuntimeType) -> Self {
        let default_endpoint = match runtime_type {
            RuntimeType::Triton => "http://localhost:8000",
            RuntimeType::VLlm => "http://localhost:8000",
            RuntimeType::Tgi => "http://localhost:3000",
            RuntimeType::TorchServe => "http://localhost:8080",
            RuntimeType::TensorFlowServing => "http://localhost:8501",
        };

        Self {
            runtime_type,
            endpoint: Url::parse(default_endpoint).unwrap(),
            management_endpoint: None,
            model_repository: None,
            request_timeout: Duration::from_secs(30),
            health_check: HealthCheckConfig::default(),
            metrics: MetricsConfig::default(),
            process: None,
            runtime_specific: HashMap::new(),
            auth: None,
            tls: None,
        }
    }

    /// Set the endpoint URL
    pub fn with_endpoint(mut self, endpoint: &str) -> Result<Self, url::ParseError> {
        self.endpoint = Url::parse(endpoint)?;
        Ok(self)
    }

    /// Set the model repository path
    pub fn with_model_repository(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_repository = Some(path.into());
        self
    }

    /// Set the request timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Enable process management
    pub fn with_process_management(mut self, config: ProcessConfig) -> Self {
        self.process = Some(config);
        self
    }

    /// Set authentication
    pub fn with_auth(mut self, auth: AuthConfig) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Enable TLS
    pub fn with_tls(mut self, tls: TlsConfig) -> Self {
        self.tls = Some(tls);
        self
    }

    /// Set runtime-specific configuration
    pub fn with_runtime_config(mut self, key: String, value: serde_json::Value) -> Self {
        self.runtime_specific.insert(key, value);
        self
    }

    /// Get runtime-specific configuration value
    pub fn get_runtime_config(&self, key: &str) -> Option<&serde_json::Value> {
        self.runtime_specific.get(key)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate endpoint
        if self.endpoint.scheme() != "http" && self.endpoint.scheme() != "https" {
            return Err("Endpoint must use HTTP or HTTPS scheme".to_string());
        }

        // Validate model repository for runtimes that require it
        match self.runtime_type {
            RuntimeType::Triton => {
                if self.model_repository.is_none() {
                    return Err("Triton requires a model repository path".to_string());
                }
            }
            _ => {}
        }

        // Validate timeouts
        if self.request_timeout.is_zero() {
            return Err("Request timeout must be greater than zero".to_string());
        }

        if self.health_check.timeout >= self.health_check.interval {
            return Err("Health check timeout must be less than interval".to_string());
        }

        Ok(())
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            failure_threshold: 3,
            success_threshold: 2,
            endpoint: None,
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(10),
            endpoint: None,
            metrics: vec![
                "requests_total".to_string(),
                "request_duration".to_string(),
                "model_load_time".to_string(),
                "gpu_utilization".to_string(),
                "memory_usage".to_string(),
            ],
            detailed_model_metrics: false,
        }
    }
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: None,
            memory_bytes: None,
            gpu_memory_bytes: None,
            gpu_count: None,
            gpu_types: Vec::new(),
        }
    }
}

impl std::fmt::Display for RuntimeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeType::Triton => write!(f, "triton"),
            RuntimeType::VLlm => write!(f, "vllm"),
            RuntimeType::Tgi => write!(f, "tgi"),
            RuntimeType::TorchServe => write!(f, "torchserve"),
            RuntimeType::TensorFlowServing => write!(f, "tensorflow-serving"),
        }
    }
}

impl std::str::FromStr for RuntimeType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "triton" => Ok(RuntimeType::Triton),
            "vllm" => Ok(RuntimeType::VLlm),
            "tgi" => Ok(RuntimeType::Tgi),
            "torchserve" => Ok(RuntimeType::TorchServe),
            "tensorflow-serving" | "tf-serving" => Ok(RuntimeType::TensorFlowServing),
            _ => Err(format!("Unknown runtime type: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_config_creation() {
        let config = RuntimeConfig::new(RuntimeType::Triton);
        assert_eq!(config.runtime_type, RuntimeType::Triton);
        assert_eq!(config.endpoint.as_str(), "http://localhost:8000/");
        assert_eq!(config.request_timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_runtime_config_builder() {
        let config = RuntimeConfig::new(RuntimeType::VLlm)
            .with_endpoint("http://example.com:8080").unwrap()
            .with_model_repository("/models")
            .with_timeout(Duration::from_secs(60));

        assert_eq!(config.endpoint.as_str(), "http://example.com:8080/");
        assert_eq!(config.model_repository, Some(PathBuf::from("/models")));
        assert_eq!(config.request_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_runtime_type_parsing() {
        assert_eq!("triton".parse::<RuntimeType>().unwrap(), RuntimeType::Triton);
        assert_eq!("vllm".parse::<RuntimeType>().unwrap(), RuntimeType::VLlm);
        assert_eq!("tgi".parse::<RuntimeType>().unwrap(), RuntimeType::Tgi);
        assert!("unknown".parse::<RuntimeType>().is_err());
    }

    #[test]
    fn test_runtime_type_display() {
        assert_eq!(RuntimeType::Triton.to_string(), "triton");
        assert_eq!(RuntimeType::VLlm.to_string(), "vllm");
        assert_eq!(RuntimeType::Tgi.to_string(), "tgi");
    }

    #[test]
    fn test_config_validation() {
        let mut config = RuntimeConfig::new(RuntimeType::Triton);
        
        // Should fail without model repository
        assert!(config.validate().is_err());
        
        // Should pass with model repository
        config.model_repository = Some(PathBuf::from("/models"));
        assert!(config.validate().is_ok());
        
        // Should fail with invalid timeout
        config.request_timeout = Duration::from_secs(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_health_check_config() {
        let config = HealthCheckConfig::default();
        assert_eq!(config.interval, Duration::from_secs(30));
        assert_eq!(config.timeout, Duration::from_secs(5));
        assert_eq!(config.failure_threshold, 3);
        assert_eq!(config.success_threshold, 2);
    }

    #[test]
    fn test_metrics_config() {
        let config = MetricsConfig::default();
        assert!(config.enabled);
        assert_eq!(config.interval, Duration::from_secs(10));
        assert!(!config.metrics.is_empty());
        assert!(!config.detailed_model_metrics);
    }
}
