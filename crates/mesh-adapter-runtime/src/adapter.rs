//! Runtime adapter interface and implementation

use crate::config::{RuntimeConfig, RuntimeType, ModelConfig};
use crate::health::{HealthStatus, HealthCheck};
use crate::metrics::{RuntimeMetrics, MetricCollector};
use crate::{Result, RuntimeError};

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Trait defining the interface for runtime adapters
#[async_trait]
pub trait RuntimeAdapterTrait: Send + Sync {
    /// Get the runtime type
    fn runtime_type(&self) -> RuntimeType;

    /// Initialize the adapter
    async fn initialize(&mut self) -> Result<()>;

    /// Shutdown the adapter
    async fn shutdown(&mut self) -> Result<()>;

    /// Check if the runtime is healthy
    async fn health_check(&self) -> Result<HealthStatus>;

    /// Load a model
    async fn load_model(&self, name: &str, config: Option<ModelConfig>) -> Result<()>;

    /// Unload a model
    async fn unload_model(&self, name: &str) -> Result<()>;

    /// List loaded models
    async fn list_models(&self) -> Result<Vec<String>>;

    /// Get model information
    async fn get_model_info(&self, name: &str) -> Result<ModelInfo>;

    /// Make an inference request
    async fn inference(&self, request: InferenceRequest) -> Result<InferenceResponse>;

    /// Get runtime metrics
    async fn get_metrics(&self) -> Result<RuntimeMetrics>;

    /// Get runtime configuration
    fn get_config(&self) -> &RuntimeConfig;
}

/// Main runtime adapter that delegates to specific implementations
pub struct RuntimeAdapter {
    inner: Box<dyn RuntimeAdapterTrait>,
    config: RuntimeConfig,
    health_checker: Arc<RwLock<HealthCheck>>,
    metric_collector: Arc<RwLock<MetricCollector>>,
}

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub version: Option<String>,
    pub status: ModelStatus,
    pub config: HashMap<String, serde_json::Value>,
    pub metadata: ModelMetadata,
}

/// Model status
#[derive(Debug, Clone, PartialEq)]
pub enum ModelStatus {
    Loading,
    Ready,
    Unloading,
    Failed(String),
}

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub input_shapes: Vec<TensorShape>,
    pub output_shapes: Vec<TensorShape>,
    pub batch_size: Option<u32>,
    pub max_sequence_length: Option<u32>,
    pub memory_usage: Option<u64>,
}

/// Tensor shape information
#[derive(Debug, Clone)]
pub struct TensorShape {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: String,
}

/// Inference request
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub model_name: String,
    pub model_version: Option<String>,
    pub inputs: HashMap<String, TensorData>,
    pub parameters: HashMap<String, serde_json::Value>,
    pub request_id: Option<String>,
}

/// Inference response
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub outputs: HashMap<String, TensorData>,
    pub model_name: String,
    pub model_version: Option<String>,
    pub request_id: Option<String>,
    pub metadata: ResponseMetadata,
}

/// Tensor data
#[derive(Debug, Clone)]
pub struct TensorData {
    pub shape: Vec<u64>,
    pub dtype: String,
    pub data: Vec<u8>,
}

/// Response metadata
#[derive(Debug, Clone)]
pub struct ResponseMetadata {
    pub inference_time_ms: f64,
    pub queue_time_ms: Option<f64>,
    pub preprocessing_time_ms: Option<f64>,
    pub postprocessing_time_ms: Option<f64>,
}

impl RuntimeAdapter {
    /// Create a new runtime adapter
    pub async fn new(config: RuntimeConfig) -> Result<Self> {
        info!("Creating runtime adapter for {}", config.runtime_type);
        
        // Validate configuration
        config.validate().map_err(RuntimeError::Configuration)?;

        // Create runtime-specific adapter
        let inner = create_runtime_adapter(&config).await?;

        // Create health checker
        let health_checker = Arc::new(RwLock::new(HealthCheck::new(config.health_check.clone())));

        // Create metric collector
        let metric_collector = Arc::new(RwLock::new(MetricCollector::new(config.metrics.clone())));

        Ok(Self {
            inner,
            config,
            health_checker,
            metric_collector,
        })
    }

    /// Start background tasks (health checking, metrics collection)
    pub async fn start_background_tasks(&self) -> Result<()> {
        info!("Starting background tasks for runtime adapter");

        // Start health checking
        if self.config.health_check.interval > std::time::Duration::from_secs(0) {
            let _health_checker = Arc::clone(&self.health_checker);
            let _adapter = self.inner.as_ref();
            
            // Note: In a real implementation, this would spawn a background task
            // For now, we'll just log that it would start
            debug!("Would start health check task with interval {:?}", self.config.health_check.interval);
        }

        // Start metrics collection
        if self.config.metrics.enabled {
            let _metric_collector = Arc::clone(&self.metric_collector);
            
            // Note: In a real implementation, this would spawn a background task
            debug!("Would start metrics collection task with interval {:?}", self.config.metrics.interval);
        }

        Ok(())
    }

    /// Stop background tasks
    pub async fn stop_background_tasks(&self) -> Result<()> {
        info!("Stopping background tasks for runtime adapter");
        
        // Note: In a real implementation, this would cancel background tasks
        debug!("Background tasks stopped");
        
        Ok(())
    }
}

#[async_trait]
impl RuntimeAdapterTrait for RuntimeAdapter {
    fn runtime_type(&self) -> RuntimeType {
        self.inner.runtime_type()
    }

    async fn initialize(&mut self) -> Result<()> {
        info!("Initializing runtime adapter");
        self.inner.initialize().await
    }

    async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down runtime adapter");
        self.stop_background_tasks().await?;
        self.inner.shutdown().await
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        self.inner.health_check().await
    }

    async fn load_model(&self, name: &str, config: Option<ModelConfig>) -> Result<()> {
        info!("Loading model: {}", name);
        self.inner.load_model(name, config).await
    }

    async fn unload_model(&self, name: &str) -> Result<()> {
        info!("Unloading model: {}", name);
        self.inner.unload_model(name).await
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        self.inner.list_models().await
    }

    async fn get_model_info(&self, name: &str) -> Result<ModelInfo> {
        self.inner.get_model_info(name).await
    }

    async fn inference(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        debug!("Processing inference request for model: {}", request.model_name);
        self.inner.inference(request).await
    }

    async fn get_metrics(&self) -> Result<RuntimeMetrics> {
        self.inner.get_metrics().await
    }

    fn get_config(&self) -> &RuntimeConfig {
        &self.config
    }
}

/// Create a runtime-specific adapter
async fn create_runtime_adapter(config: &RuntimeConfig) -> Result<Box<dyn RuntimeAdapterTrait>> {
    match config.runtime_type {
        #[cfg(feature = "triton")]
        RuntimeType::Triton => {
            let adapter = crate::triton::TritonAdapter::new(config.clone()).await?;
            Ok(Box::new(adapter))
        }
        
        #[cfg(feature = "vllm")]
        RuntimeType::VLlm => {
            let adapter = crate::vllm::VLlmAdapter::new(config.clone()).await?;
            Ok(Box::new(adapter))
        }
        
        #[cfg(feature = "tgi")]
        RuntimeType::Tgi => {
            let adapter = crate::tgi::TgiAdapter::new(config.clone()).await?;
            Ok(Box::new(adapter))
        }
        
        #[cfg(feature = "torchserve")]
        RuntimeType::TorchServe => {
            let adapter = crate::torchserve::TorchServeAdapter::new(config.clone()).await?;
            Ok(Box::new(adapter))
        }
        
        #[cfg(feature = "tensorflow-serving")]
        RuntimeType::TensorFlowServing => {
            let adapter = crate::tensorflow_serving::TensorFlowServingAdapter::new(config.clone()).await?;
            Ok(Box::new(adapter))
        }
        
        _ => {
            warn!("Runtime type {:?} not supported or feature not enabled", config.runtime_type);
            Err(RuntimeError::UnsupportedRuntime(config.runtime_type.to_string()))
        }
    }
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
            batch_size: None,
            max_sequence_length: None,
            memory_usage: None,
        }
    }
}

impl Default for ResponseMetadata {
    fn default() -> Self {
        Self {
            inference_time_ms: 0.0,
            queue_time_ms: None,
            preprocessing_time_ms: None,
            postprocessing_time_ms: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    //use crate::config::RuntimeType;

    #[test]
    fn test_model_status() {
        let status = ModelStatus::Ready;
        assert_eq!(status, ModelStatus::Ready);
        
        let failed_status = ModelStatus::Failed("error".to_string());
        assert_ne!(failed_status, ModelStatus::Ready);
    }

    #[test]
    fn test_tensor_shape() {
        let shape = TensorShape {
            name: "input".to_string(),
            shape: vec![1, 3, 224, 224],
            dtype: "float32".to_string(),
        };
        
        assert_eq!(shape.name, "input");
        assert_eq!(shape.shape, vec![1, 3, 224, 224]);
        assert_eq!(shape.dtype, "float32");
    }

    #[test]
    fn test_inference_request() {
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), TensorData {
            shape: vec![1, 224, 224, 3],
            dtype: "float32".to_string(),
            data: vec![0u8; 1 * 224 * 224 * 3 * 4], // 4 bytes per float32
        });

        let request = InferenceRequest {
            model_name: "resnet50".to_string(),
            model_version: Some("1".to_string()),
            inputs,
            parameters: HashMap::new(),
            request_id: Some("req-123".to_string()),
        };

        assert_eq!(request.model_name, "resnet50");
        assert_eq!(request.model_version, Some("1".to_string()));
        assert_eq!(request.request_id, Some("req-123".to_string()));
    }

    #[tokio::test]
    async fn test_unsupported_runtime() {
        // This test would work if we had a runtime type that's not enabled
        // For now, we'll test the error creation
        let error = RuntimeError::UnsupportedRuntime("test-runtime".to_string());
        assert!(error.to_string().contains("test-runtime"));
    }
}
