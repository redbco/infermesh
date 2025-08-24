//! Core traits for infermesh components
//!
//! These traits define the interfaces for runtime control and GPU telemetry
//! that are implemented by various adapters throughout the system.

use crate::{GpuStateDelta, Labels, ModelStateDelta, Result};
use async_trait::async_trait;
use tokio::sync::mpsc;

/// Trait for controlling inference runtimes (load/unload/warm models)
#[async_trait]
pub trait RuntimeControl: Send + Sync {
    /// Load a model with the given configuration
    async fn load_model(&self, labels: &Labels, config: &ModelConfig) -> Result<()>;
    
    /// Unload a model
    async fn unload_model(&self, labels: &Labels) -> Result<()>;
    
    /// Warm up a model (prepare for inference without fully loading)
    async fn warm_model(&self, labels: &Labels) -> Result<()>;
    
    /// Check if a model is currently loaded
    async fn is_model_loaded(&self, labels: &Labels) -> Result<bool>;
    
    /// Get the current status of a model
    async fn get_model_status(&self, labels: &Labels) -> Result<ModelStatus>;
    
    /// List all currently loaded models
    async fn list_loaded_models(&self) -> Result<Vec<Labels>>;
}

/// Trait for collecting GPU telemetry data
#[async_trait]
pub trait GpuTelemetry: Send + Sync {
    /// Start collecting GPU metrics and send deltas to the provided channel
    async fn start_collection(&self, sender: mpsc::UnboundedSender<GpuStateDelta>) -> Result<()>;
    
    /// Stop collecting GPU metrics
    async fn stop_collection(&self) -> Result<()>;
    
    /// Get current GPU state snapshot
    async fn get_gpu_state(&self, gpu_uuid: &str) -> Result<crate::GpuState>;
    
    /// List all available GPUs
    async fn list_gpus(&self) -> Result<Vec<String>>;
    
    /// Check if a specific GPU is healthy
    async fn is_gpu_healthy(&self, gpu_uuid: &str) -> Result<bool>;
}

/// Trait for collecting runtime metrics
#[async_trait]
pub trait RuntimeTelemetry: Send + Sync {
    /// Start collecting runtime metrics and send deltas to the provided channel
    async fn start_collection(&self, sender: mpsc::UnboundedSender<ModelStateDelta>) -> Result<()>;
    
    /// Stop collecting runtime metrics
    async fn stop_collection(&self) -> Result<()>;
    
    /// Get current model state snapshot
    async fn get_model_state(&self, labels: &Labels) -> Result<crate::ModelState>;
    
    /// List all models being monitored
    async fn list_monitored_models(&self) -> Result<Vec<Labels>>;
}

/// Configuration for loading a model
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    /// Model repository or path
    pub model_repository: String,
    
    /// Model name within the repository
    pub model_name: String,
    
    /// Model version/revision
    pub model_version: String,
    
    /// Maximum batch size
    pub max_batch_size: Option<u32>,
    
    /// Maximum sequence length
    pub max_sequence_length: Option<u32>,
    
    /// GPU memory pool size in MB
    pub gpu_memory_pool_mb: Option<u32>,
    
    /// Number of instances to load
    pub instance_count: Option<u32>,
    
    /// Runtime-specific configuration
    pub runtime_config: std::collections::HashMap<String, serde_json::Value>,
}

impl ModelConfig {
    /// Create a new ModelConfig with required fields
    pub fn new(
        model_repository: impl Into<String>,
        model_name: impl Into<String>,
        model_version: impl Into<String>,
    ) -> Self {
        Self {
            model_repository: model_repository.into(),
            model_name: model_name.into(),
            model_version: model_version.into(),
            max_batch_size: None,
            max_sequence_length: None,
            gpu_memory_pool_mb: None,
            instance_count: None,
            runtime_config: std::collections::HashMap::new(),
        }
    }

    /// Builder pattern for optional fields
    pub fn with_max_batch_size(mut self, size: u32) -> Self {
        self.max_batch_size = Some(size);
        self
    }

    pub fn with_max_sequence_length(mut self, length: u32) -> Self {
        self.max_sequence_length = Some(length);
        self
    }

    pub fn with_gpu_memory_pool_mb(mut self, mb: u32) -> Self {
        self.gpu_memory_pool_mb = Some(mb);
        self
    }

    pub fn with_instance_count(mut self, count: u32) -> Self {
        self.instance_count = Some(count);
        self
    }

    pub fn with_runtime_config(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.runtime_config.insert(key.into(), value);
        self
    }
}

/// Status of a model in the runtime
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ModelStatus {
    /// Model is not loaded
    NotLoaded,
    
    /// Model is currently loading
    Loading,
    
    /// Model is loaded and ready for inference
    Ready,
    
    /// Model is warming up
    Warming,
    
    /// Model failed to load
    Failed { error: String },
    
    /// Model is being unloaded
    Unloading,
}

impl ModelStatus {
    /// Check if the model is available for inference
    pub fn is_ready(&self) -> bool {
        matches!(self, ModelStatus::Ready)
    }

    /// Check if the model is in a transitional state
    pub fn is_transitioning(&self) -> bool {
        matches!(
            self,
            ModelStatus::Loading | ModelStatus::Warming | ModelStatus::Unloading
        )
    }

    /// Check if the model is in an error state
    pub fn is_error(&self) -> bool {
        matches!(self, ModelStatus::Failed { .. })
    }
}

/// Mock implementations for testing and development

/// Mock runtime control implementation
pub struct MockRuntimeControl {
    loaded_models: std::sync::Arc<tokio::sync::RwLock<std::collections::HashSet<String>>>,
}

impl MockRuntimeControl {
    pub fn new() -> Self {
        Self {
            loaded_models: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashSet::new(),
            )),
        }
    }
}

impl Default for MockRuntimeControl {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl RuntimeControl for MockRuntimeControl {
    async fn load_model(&self, labels: &Labels, _config: &ModelConfig) -> Result<()> {
        let key = labels.key();
        let mut loaded = self.loaded_models.write().await;
        loaded.insert(key);
        
        // Simulate loading time
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        Ok(())
    }

    async fn unload_model(&self, labels: &Labels) -> Result<()> {
        let key = labels.key();
        let mut loaded = self.loaded_models.write().await;
        loaded.remove(&key);
        Ok(())
    }

    async fn warm_model(&self, _labels: &Labels) -> Result<()> {
        // Simulate warming time
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        Ok(())
    }

    async fn is_model_loaded(&self, labels: &Labels) -> Result<bool> {
        let key = labels.key();
        let loaded = self.loaded_models.read().await;
        Ok(loaded.contains(&key))
    }

    async fn get_model_status(&self, labels: &Labels) -> Result<ModelStatus> {
        let is_loaded = self.is_model_loaded(labels).await?;
        Ok(if is_loaded {
            ModelStatus::Ready
        } else {
            ModelStatus::NotLoaded
        })
    }

    async fn list_loaded_models(&self) -> Result<Vec<Labels>> {
        // For mock implementation, return empty list
        // In real implementation, this would parse the keys back to Labels
        Ok(vec![])
    }
}

/// Mock GPU telemetry implementation
pub struct MockGpuTelemetry {
    gpu_uuids: Vec<String>,
    collection_active: std::sync::Arc<tokio::sync::RwLock<bool>>,
}

impl MockGpuTelemetry {
    pub fn new(gpu_uuids: Vec<String>) -> Self {
        Self {
            gpu_uuids,
            collection_active: std::sync::Arc::new(tokio::sync::RwLock::new(false)),
        }
    }

    pub fn with_mock_gpus(count: usize) -> Self {
        let gpu_uuids = (0..count)
            .map(|i| format!("GPU-MOCK-{:08}", i))
            .collect();
        Self::new(gpu_uuids)
    }
}

#[async_trait]
impl GpuTelemetry for MockGpuTelemetry {
    async fn start_collection(&self, sender: mpsc::UnboundedSender<GpuStateDelta>) -> Result<()> {
        let mut active = self.collection_active.write().await;
        *active = true;
        
        let gpu_uuids = self.gpu_uuids.clone();
        let collection_active = self.collection_active.clone();
        
        // Spawn background task to send mock deltas
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
            
            while *collection_active.read().await {
                interval.tick().await;
                
                for gpu_uuid in &gpu_uuids {
                    let mut delta = GpuStateDelta::new(gpu_uuid, "mock-node");
                    
                    // Generate mock metrics (deterministic for now)
                    delta.sm_utilization = Some(0.5);
                    delta.memory_utilization = Some(0.6);
                    delta.vram_used_gb = Some(8.0);
                    delta.vram_total_gb = Some(16.0);
                    
                    if sender.send(delta).is_err() {
                        break;
                    }
                }
            }
        });
        
        Ok(())
    }

    async fn stop_collection(&self) -> Result<()> {
        let mut active = self.collection_active.write().await;
        *active = false;
        Ok(())
    }

    async fn get_gpu_state(&self, gpu_uuid: &str) -> Result<crate::GpuState> {
        if !self.gpu_uuids.contains(&gpu_uuid.to_string()) {
            return Err(crate::Error::NotFound(format!("GPU not found: {}", gpu_uuid)));
        }

        let mut state = crate::GpuState::new(gpu_uuid, "mock-node");
        state.update_metrics(
            0.5, // SM utilization
            0.6, // Memory utilization
            8.0, // VRAM used
            16.0, // VRAM total
        );
        
        Ok(state)
    }

    async fn list_gpus(&self) -> Result<Vec<String>> {
        Ok(self.gpu_uuids.clone())
    }

    async fn is_gpu_healthy(&self, gpu_uuid: &str) -> Result<bool> {
        if !self.gpu_uuids.contains(&gpu_uuid.to_string()) {
            return Err(crate::Error::NotFound(format!("GPU not found: {}", gpu_uuid)));
        }
        Ok(true) // Mock GPUs are always healthy
    }
}

// Mock implementations use deterministic values for reproducible testing

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Labels;

    #[tokio::test]
    async fn test_mock_runtime_control() {
        let runtime = MockRuntimeControl::new();
        let labels = Labels::new("gpt-4", "v1.0", "triton", "node1");
        let config = ModelConfig::new("/models", "gpt-4", "v1.0");

        // Initially not loaded
        assert!(!runtime.is_model_loaded(&labels).await.unwrap());
        
        // Load model
        runtime.load_model(&labels, &config).await.unwrap();
        assert!(runtime.is_model_loaded(&labels).await.unwrap());
        
        // Check status
        let status = runtime.get_model_status(&labels).await.unwrap();
        assert!(status.is_ready());
        
        // Unload model
        runtime.unload_model(&labels).await.unwrap();
        assert!(!runtime.is_model_loaded(&labels).await.unwrap());
    }

    #[tokio::test]
    async fn test_mock_gpu_telemetry() {
        let telemetry = MockGpuTelemetry::with_mock_gpus(2);
        
        let gpus = telemetry.list_gpus().await.unwrap();
        assert_eq!(gpus.len(), 2);
        
        let gpu_uuid = &gpus[0];
        assert!(telemetry.is_gpu_healthy(gpu_uuid).await.unwrap());
        
        let state = telemetry.get_gpu_state(gpu_uuid).await.unwrap();
        assert_eq!(state.gpu_uuid, *gpu_uuid);
    }

    #[test]
    fn test_model_config_builder() {
        let config = ModelConfig::new("/models", "gpt-4", "v1.0")
            .with_max_batch_size(32)
            .with_max_sequence_length(2048)
            .with_runtime_config("temperature", serde_json::json!(0.7));

        assert_eq!(config.max_batch_size, Some(32));
        assert_eq!(config.max_sequence_length, Some(2048));
        assert_eq!(
            config.runtime_config.get("temperature"),
            Some(&serde_json::json!(0.7))
        );
    }

    #[test]
    fn test_model_status() {
        assert!(ModelStatus::Ready.is_ready());
        assert!(!ModelStatus::Loading.is_ready());
        
        assert!(ModelStatus::Loading.is_transitioning());
        assert!(!ModelStatus::Ready.is_transitioning());
        
        assert!(ModelStatus::Failed { error: "test".to_string() }.is_error());
        assert!(!ModelStatus::Ready.is_error());
    }
}
