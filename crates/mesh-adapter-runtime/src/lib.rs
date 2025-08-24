//! # mesh-adapter-runtime
//!
//! Runtime adapters for ML inference engines including Triton, vLLM, TGI, and TorchServe.
//!
//! This crate provides:
//! - Unified adapter interface for different ML inference runtimes
//! - Runtime-specific implementations for popular inference engines
//! - Metric collection and normalization across runtimes
//! - Process management and health monitoring
//! - Feature flags for selective runtime support
//!
//! ## Supported Runtimes
//!
//! - **Triton Inference Server**: NVIDIA's inference serving platform
//! - **vLLM**: High-throughput LLM serving engine
//! - **Text Generation Inference (TGI)**: Hugging Face's inference server
//! - **TorchServe**: PyTorch's model serving framework
//! - **TensorFlow Serving**: TensorFlow's serving system
//!
//! ## Example
//!
//! ```rust,no_run
//! use mesh_adapter_runtime::{RuntimeAdapter, RuntimeAdapterTrait, RuntimeConfig, RuntimeType};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = RuntimeConfig::new(RuntimeType::Triton)
//!         .with_endpoint("http://localhost:8000")?
//!         .with_model_repository("/models");
//!     
//!     let adapter = RuntimeAdapter::new(config).await?;
//!     
//!     // Load a model
//!     adapter.load_model("gpt-7b", None).await?;
//!     
//!     // Check health
//!     let health = adapter.health_check().await?;
//!     println!("Runtime health: {:?}", health);
//!     
//!     Ok(())
//! }
//! ```

use thiserror::Error;

pub mod adapter;
pub mod config;
pub mod health;
pub mod metrics;
pub mod process;

// Runtime-specific modules
#[cfg(feature = "triton")]
pub mod triton;

#[cfg(feature = "vllm")]
pub mod vllm;

#[cfg(feature = "tgi")]
pub mod tgi;

#[cfg(feature = "torchserve")]
pub mod torchserve;

#[cfg(feature = "tensorflow-serving")]
pub mod tensorflow_serving;

// Re-export main types
pub use adapter::{RuntimeAdapter, RuntimeAdapterTrait};
pub use config::{RuntimeConfig, RuntimeType, ModelConfig};
pub use health::{HealthStatus, HealthCheck};
pub use metrics::{RuntimeMetrics, MetricCollector};

/// Result type for runtime operations
pub type Result<T> = std::result::Result<T, RuntimeError>;

/// Errors that can occur during runtime operations
#[derive(Error, Debug)]
pub enum RuntimeError {
    #[error("Runtime not supported: {0}")]
    UnsupportedRuntime(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Process error: {0}")]
    Process(String),

    #[error("Health check failed: {0}")]
    HealthCheck(String),

    #[error("Metrics collection failed: {0}")]
    Metrics(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::Status),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Runtime unavailable: {0}")]
    Unavailable(String),
}

impl RuntimeError {
    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            RuntimeError::Connection(_)
                | RuntimeError::Timeout(_)
                | RuntimeError::Unavailable(_)
                | RuntimeError::Http(_)
        )
    }

    /// Check if this error indicates the runtime is unhealthy
    pub fn is_health_issue(&self) -> bool {
        matches!(
            self,
            RuntimeError::HealthCheck(_)
                | RuntimeError::Connection(_)
                | RuntimeError::Process(_)
                | RuntimeError::Unavailable(_)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_error_properties() {
        let connection_error = RuntimeError::Connection("test".to_string());
        assert!(connection_error.is_retryable());
        assert!(connection_error.is_health_issue());

        let config_error = RuntimeError::Configuration("test".to_string());
        assert!(!config_error.is_retryable());
        assert!(!config_error.is_health_issue());

        let health_error = RuntimeError::HealthCheck("test".to_string());
        assert!(!health_error.is_retryable());
        assert!(health_error.is_health_issue());
    }

    #[test]
    fn test_error_display() {
        let error = RuntimeError::UnsupportedRuntime("test-runtime".to_string());
        assert_eq!(error.to_string(), "Runtime not supported: test-runtime");

        let error = RuntimeError::Model("model not found".to_string());
        assert_eq!(error.to_string(), "Model error: model not found");
    }
}
