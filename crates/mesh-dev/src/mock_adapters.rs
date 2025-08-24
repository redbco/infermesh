//! Mock implementations of runtime and GPU adapters for testing

use crate::Result;
use mesh_core::{
    GpuStateDelta, GpuTelemetry, Labels, ModelConfig, ModelStateDelta, ModelStatus, RuntimeControl,
    RuntimeTelemetry,
};
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};

/// Mock runtime adapter that simulates inference runtime behavior
#[derive(Debug, Clone)]
pub struct MockRuntimeAdapter {
    inner: Arc<MockRuntimeAdapterInner>,
}

#[derive(Debug)]
struct MockRuntimeAdapterInner {
    /// Currently loaded models
    loaded_models: RwLock<HashMap<String, ModelStatus>>,
    
    /// Mock metrics for each model
    model_metrics: RwLock<HashMap<String, MockModelMetrics>>,
    
    /// Configuration
    config: MockRuntimeConfig,
}

#[derive(Debug, Clone)]
pub struct MockRuntimeConfig {
    /// Base latency for requests (milliseconds)
    pub base_latency_ms: u32,
    
    /// Latency variance (milliseconds)
    pub latency_variance_ms: u32,
    
    /// Base service rate (requests per second)
    pub base_service_rate: f64,
    
    /// Service rate variance
    pub service_rate_variance: f64,
    
    /// Maximum queue depth before rejecting requests
    pub max_queue_depth: u32,
    
    /// Probability of request failure (0.0 to 1.0)
    pub failure_rate: f64,
}

impl Default for MockRuntimeConfig {
    fn default() -> Self {
        Self {
            base_latency_ms: 100,
            latency_variance_ms: 50,
            base_service_rate: 10.0,
            service_rate_variance: 2.0,
            max_queue_depth: 50,
            failure_rate: 0.01, // 1% failure rate
        }
    }
}

#[derive(Debug, Clone)]
struct MockModelMetrics {
    queue_depth: u32,
    service_rate: f64,
    p95_latency_ms: u32,
    batch_fullness: f32,
    last_updated: chrono::DateTime<chrono::Utc>,
}

impl Default for MockModelMetrics {
    fn default() -> Self {
        Self {
            queue_depth: 0,
            service_rate: 0.0,
            p95_latency_ms: 0,
            batch_fullness: 0.0,
            last_updated: chrono::Utc::now(),
        }
    }
}

impl MockRuntimeAdapter {
    /// Create a new mock runtime adapter
    pub fn new(config: MockRuntimeConfig) -> Self {
        let inner = MockRuntimeAdapterInner {
            loaded_models: RwLock::new(HashMap::new()),
            model_metrics: RwLock::new(HashMap::new()),
            config,
        };

        Self {
            inner: Arc::new(inner),
        }
    }

    /// Create with default configuration
    pub fn new_default() -> Self {
        Self::new(MockRuntimeConfig::default())
    }

    /// Simulate processing a request and update metrics
    pub async fn simulate_request(&self, labels: &Labels, _tokens: u32) -> Result<()> {
        let key = labels.key();
        let mut metrics = self.inner.model_metrics.write().await;
        
        if let Some(model_metrics) = metrics.get_mut(&key) {
            // Simulate request processing
            let mut rng = rand::thread_rng();
            
            // Update queue depth (simulate queuing)
            model_metrics.queue_depth = rng.gen_range(0..=self.inner.config.max_queue_depth);
            
            // Update service rate with variance
            let variance = rng.gen_range(-self.inner.config.service_rate_variance..=self.inner.config.service_rate_variance);
            model_metrics.service_rate = (self.inner.config.base_service_rate + variance).max(0.1);
            
            // Update latency with variance
            let latency_variance = rng.gen_range(0..=self.inner.config.latency_variance_ms);
            model_metrics.p95_latency_ms = self.inner.config.base_latency_ms + latency_variance;
            
            // Update batch fullness (simulate batching)
            model_metrics.batch_fullness = rng.gen_range(0.0..=1.0);
            
            model_metrics.last_updated = chrono::Utc::now();
            
            debug!("Simulated request for model {}: queue_depth={}, service_rate={:.2}", 
                   labels.model, model_metrics.queue_depth, model_metrics.service_rate);
        }
        
        Ok(())
    }

    /// Get current metrics for a model
    pub async fn get_metrics(&self, labels: &Labels) -> Option<MockModelMetrics> {
        let metrics = self.inner.model_metrics.read().await;
        metrics.get(&labels.key()).cloned()
    }
}

#[async_trait::async_trait]
impl RuntimeControl for MockRuntimeAdapter {
    async fn load_model(&self, labels: &Labels, _config: &ModelConfig) -> mesh_core::Result<()> {
        let key = labels.key();
        info!("Loading mock model: {}", key);
        
        // Simulate loading time
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        let mut loaded_models = self.inner.loaded_models.write().await;
        loaded_models.insert(key.clone(), ModelStatus::Ready);
        
        // Initialize metrics
        let mut model_metrics = self.inner.model_metrics.write().await;
        model_metrics.insert(key, MockModelMetrics::default());
        
        Ok(())
    }

    async fn unload_model(&self, labels: &Labels) -> mesh_core::Result<()> {
        let key = labels.key();
        info!("Unloading mock model: {}", key);
        
        let mut loaded_models = self.inner.loaded_models.write().await;
        loaded_models.remove(&key);
        
        let mut model_metrics = self.inner.model_metrics.write().await;
        model_metrics.remove(&key);
        
        Ok(())
    }

    async fn warm_model(&self, labels: &Labels) -> mesh_core::Result<()> {
        let key = labels.key();
        info!("Warming mock model: {}", key);
        
        // Simulate warming time
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        
        let mut loaded_models = self.inner.loaded_models.write().await;
        loaded_models.insert(key, ModelStatus::Warming);
        
        Ok(())
    }

    async fn is_model_loaded(&self, labels: &Labels) -> mesh_core::Result<bool> {
        let key = labels.key();
        let loaded_models = self.inner.loaded_models.read().await;
        Ok(loaded_models.contains_key(&key))
    }

    async fn get_model_status(&self, labels: &Labels) -> mesh_core::Result<ModelStatus> {
        let key = labels.key();
        let loaded_models = self.inner.loaded_models.read().await;
        Ok(loaded_models.get(&key).cloned().unwrap_or(ModelStatus::NotLoaded))
    }

    async fn list_loaded_models(&self) -> mesh_core::Result<Vec<Labels>> {
        // For mock implementation, return empty list
        // In a real implementation, we'd parse the keys back to Labels
        Ok(vec![])
    }
}

#[async_trait::async_trait]
impl RuntimeTelemetry for MockRuntimeAdapter {
    async fn start_collection(&self, sender: mpsc::UnboundedSender<ModelStateDelta>) -> mesh_core::Result<()> {
        let inner = self.inner.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
            
            loop {
                interval.tick().await;
                
                let model_metrics = inner.model_metrics.read().await;
                for (key, metrics) in model_metrics.iter() {
                    // Parse key back to labels (simplified for mock)
                    let parts: Vec<&str> = key.split(':').collect();
                    if parts.len() >= 4 {
                        let labels = Labels::new(parts[0], parts[1], parts[3], parts[2]);
                        
                        let mut delta = ModelStateDelta::new(labels);
                        delta.queue_depth = Some(metrics.queue_depth);
                        delta.service_rate = Some(metrics.service_rate);
                        delta.p95_latency_ms = Some(metrics.p95_latency_ms);
                        delta.batch_fullness = Some(metrics.batch_fullness);
                        delta.loaded = Some(true);
                        delta.warming = Some(false);
                        
                        if sender.send(delta).is_err() {
                            warn!("Failed to send model state delta");
                            break;
                        }
                    }
                }
            }
        });
        
        Ok(())
    }

    async fn stop_collection(&self) -> mesh_core::Result<()> {
        // In a real implementation, we'd stop the collection task
        Ok(())
    }

    async fn get_model_state(&self, labels: &Labels) -> mesh_core::Result<mesh_core::ModelState> {
        let key = labels.key();
        let model_metrics = self.inner.model_metrics.read().await;
        
        if let Some(metrics) = model_metrics.get(&key) {
            let mut state = mesh_core::ModelState::new(labels.clone());
            state.update(
                metrics.queue_depth,
                metrics.service_rate,
                metrics.p95_latency_ms,
                metrics.batch_fullness,
            );
            state.mark_loaded();
            Ok(state)
        } else {
            Ok(mesh_core::ModelState::new(labels.clone()))
        }
    }

    async fn list_monitored_models(&self) -> mesh_core::Result<Vec<Labels>> {
        // For mock implementation, return empty list
        Ok(vec![])
    }
}

/// Mock GPU adapter that simulates GPU telemetry
#[derive(Debug, Clone)]
pub struct MockGpuAdapter {
    inner: Arc<MockGpuAdapterInner>,
}

#[derive(Debug)]
struct MockGpuAdapterInner {
    /// GPU UUIDs to simulate
    gpu_uuids: Vec<String>,
    
    /// Mock GPU metrics
    gpu_metrics: RwLock<HashMap<String, MockGpuMetrics>>,
    
    /// Configuration
    config: MockGpuConfig,
}

#[derive(Debug, Clone)]
pub struct MockGpuConfig {
    /// Number of GPUs to simulate
    pub gpu_count: usize,
    
    /// Base GPU utilization (0.0 to 1.0)
    pub base_utilization: f32,
    
    /// Utilization variance
    pub utilization_variance: f32,
    
    /// Total VRAM per GPU (GB)
    pub vram_total_gb: f32,
    
    /// Base temperature (Celsius)
    pub base_temperature_c: f32,
    
    /// Base power consumption (Watts)
    pub base_power_watts: f32,
}

impl Default for MockGpuConfig {
    fn default() -> Self {
        Self {
            gpu_count: 2,
            base_utilization: 0.5,
            utilization_variance: 0.3,
            vram_total_gb: 16.0,
            base_temperature_c: 70.0,
            base_power_watts: 250.0,
        }
    }
}

#[derive(Debug, Clone)]
struct MockGpuMetrics {
    sm_utilization: f32,
    memory_utilization: f32,
    vram_used_gb: f32,
    temperature_c: f32,
    power_watts: f32,
    ecc_errors: bool,
    throttled: bool,
    last_updated: chrono::DateTime<chrono::Utc>,
}

impl MockGpuAdapter {
    /// Create a new mock GPU adapter
    pub fn new(config: MockGpuConfig) -> Self {
        let gpu_uuids: Vec<String> = (0..config.gpu_count)
            .map(|i| format!("GPU-MOCK-{:08}", i))
            .collect();

        let mut gpu_metrics = HashMap::new();
        for gpu_uuid in &gpu_uuids {
            gpu_metrics.insert(gpu_uuid.clone(), MockGpuMetrics {
                sm_utilization: config.base_utilization,
                memory_utilization: config.base_utilization * 0.8,
                vram_used_gb: config.vram_total_gb * config.base_utilization,
                temperature_c: config.base_temperature_c,
                power_watts: config.base_power_watts,
                ecc_errors: false,
                throttled: false,
                last_updated: chrono::Utc::now(),
            });
        }

        let inner = MockGpuAdapterInner {
            gpu_uuids,
            gpu_metrics: RwLock::new(gpu_metrics),
            config,
        };

        Self {
            inner: Arc::new(inner),
        }
    }

    /// Create with default configuration
    pub fn new_default() -> Self {
        Self::new(MockGpuConfig::default())
    }

    /// Update GPU metrics with random variations
    pub async fn update_metrics(&self) {
        let mut rng = rand::thread_rng();
        let mut gpu_metrics = self.inner.gpu_metrics.write().await;

        for (gpu_uuid, metrics) in gpu_metrics.iter_mut() {
            // Add random variations
            let util_variance = rng.gen_range(-self.inner.config.utilization_variance..=self.inner.config.utilization_variance);
            metrics.sm_utilization = (self.inner.config.base_utilization + util_variance).clamp(0.0, 1.0);
            metrics.memory_utilization = (metrics.sm_utilization * 0.8).clamp(0.0, 1.0);
            metrics.vram_used_gb = self.inner.config.vram_total_gb * metrics.memory_utilization;
            
            // Temperature varies with utilization
            metrics.temperature_c = self.inner.config.base_temperature_c + (metrics.sm_utilization * 20.0);
            
            // Power varies with utilization
            metrics.power_watts = self.inner.config.base_power_watts * (0.5 + metrics.sm_utilization * 0.5);
            
            // Rarely simulate errors
            metrics.ecc_errors = rng.gen_bool(0.001); // 0.1% chance
            metrics.throttled = metrics.temperature_c > 85.0;
            
            metrics.last_updated = chrono::Utc::now();
            
            debug!("Updated mock GPU {}: util={:.2}, temp={:.1}°C", 
                   gpu_uuid, metrics.sm_utilization, metrics.temperature_c);
        }
    }
}

#[async_trait::async_trait]
impl GpuTelemetry for MockGpuAdapter {
    async fn start_collection(&self, sender: mpsc::UnboundedSender<GpuStateDelta>) -> mesh_core::Result<()> {
        let inner = self.inner.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
            
            loop {
                interval.tick().await;
                
                let gpu_metrics = inner.gpu_metrics.read().await;
                for (gpu_uuid, metrics) in gpu_metrics.iter() {
                    let mut delta = GpuStateDelta::new(gpu_uuid, "mock-node");
                    delta.sm_utilization = Some(metrics.sm_utilization);
                    delta.memory_utilization = Some(metrics.memory_utilization);
                    delta.vram_used_gb = Some(metrics.vram_used_gb);
                    delta.vram_total_gb = Some(inner.config.vram_total_gb);
                    delta.temperature_c = Some(metrics.temperature_c);
                    delta.power_watts = Some(metrics.power_watts);
                    delta.ecc_errors = Some(metrics.ecc_errors);
                    delta.throttled = Some(metrics.throttled);
                    
                    if sender.send(delta).is_err() {
                        warn!("Failed to send GPU state delta");
                        break;
                    }
                }
            }
        });
        
        Ok(())
    }

    async fn stop_collection(&self) -> mesh_core::Result<()> {
        // In a real implementation, we'd stop the collection task
        Ok(())
    }

    async fn get_gpu_state(&self, gpu_uuid: &str) -> mesh_core::Result<mesh_core::GpuState> {
        let gpu_metrics = self.inner.gpu_metrics.read().await;
        
        if let Some(metrics) = gpu_metrics.get(gpu_uuid) {
            let mut state = mesh_core::GpuState::new(gpu_uuid, "mock-node");
            state.update_metrics(
                metrics.sm_utilization,
                metrics.memory_utilization,
                metrics.vram_used_gb,
                self.inner.config.vram_total_gb,
            );
            state.update_thermal(Some(metrics.temperature_c), Some(metrics.power_watts));
            state.update_status(metrics.ecc_errors, metrics.throttled);
            Ok(state)
        } else {
            Err(mesh_core::Error::not_found(format!("GPU not found: {}", gpu_uuid)))
        }
    }

    async fn list_gpus(&self) -> mesh_core::Result<Vec<String>> {
        Ok(self.inner.gpu_uuids.clone())
    }

    async fn is_gpu_healthy(&self, gpu_uuid: &str) -> mesh_core::Result<bool> {
        let gpu_metrics = self.inner.gpu_metrics.read().await;
        
        if let Some(metrics) = gpu_metrics.get(gpu_uuid) {
            Ok(!metrics.ecc_errors && !metrics.throttled)
        } else {
            Err(mesh_core::Error::not_found(format!("GPU not found: {}", gpu_uuid)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_runtime_adapter() {
        let adapter = MockRuntimeAdapter::new_default();
        let labels = Labels::new("test-model", "v1.0", "mock", "test-node");
        let config = ModelConfig::new("/models", "test-model", "v1.0");

        // Test loading
        adapter.load_model(&labels, &config).await.unwrap();
        assert!(adapter.is_model_loaded(&labels).await.unwrap());

        // Test status
        let status = adapter.get_model_status(&labels).await.unwrap();
        assert!(status.is_ready());

        // Test simulation
        adapter.simulate_request(&labels, 100).await.unwrap();
        let metrics = adapter.get_metrics(&labels).await.unwrap();
        assert!(metrics.service_rate > 0.0);

        // Test unloading
        adapter.unload_model(&labels).await.unwrap();
        assert!(!adapter.is_model_loaded(&labels).await.unwrap());
    }

    #[tokio::test]
    async fn test_mock_gpu_adapter() {
        let adapter = MockGpuAdapter::new_default();
        
        let gpus = adapter.list_gpus().await.unwrap();
        assert_eq!(gpus.len(), 2);

        let gpu_uuid = &gpus[0];
        assert!(adapter.is_gpu_healthy(gpu_uuid).await.unwrap());

        let state = adapter.get_gpu_state(gpu_uuid).await.unwrap();
        assert_eq!(state.gpu_uuid, *gpu_uuid);
        assert!(state.vram_total_gb > 0.0);

        // Test metrics update
        adapter.update_metrics().await;
        let updated_state = adapter.get_gpu_state(gpu_uuid).await.unwrap();
        assert!(updated_state.sm_utilization >= 0.0);
    }
}
