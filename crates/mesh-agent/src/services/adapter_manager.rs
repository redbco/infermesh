//! Adapter manager service for integrating runtime and GPU adapters

use crate::config::{AdapterConfig, RuntimeInstanceConfig, GpuInstanceConfig};
use crate::services::StatePlaneService;
use mesh_adapter_runtime::{RuntimeAdapter, RuntimeAdapterTrait, RuntimeConfig, RuntimeType};
use mesh_adapter_runtime::vllm::VLlmAdapter;
use mesh_adapter_runtime::triton::TritonAdapter;
use mesh_adapter_runtime::tgi::TgiAdapter;
use mesh_adapter_gpu::{GpuMonitor, GpuMonitorTrait, GpuMonitorConfig, GpuBackend};
use mesh_adapter_gpu::nvml::NvmlMonitor;
use mesh_adapter_gpu::dcgm::DcgmMonitor;
use mesh_adapter_gpu::metrics::ThermalState;
use mesh_metrics::MetricsRegistry;
use mesh_proto::state::v1::{ModelStateDelta, GpuStateDelta, Labels};
use mesh_proto::timestamp;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::{interval, Interval};
use tracing::{debug, error, info, warn};

/// Adapter manager handles runtime and GPU adapters
#[derive(Clone)]
pub struct AdapterManager {
    config: AdapterConfig,
    metrics_registry: MetricsRegistry,
    runtime_adapters: Arc<RwLock<HashMap<String, Arc<RuntimeAdapter>>>>,
    gpu_adapters: Arc<RwLock<HashMap<String, Arc<GpuMonitor>>>>,
    telemetry_buffer: Arc<RwLock<TelemetryBuffer>>,
    state_plane: Option<Arc<StatePlaneService>>,
    node_id: String,
}

/// Telemetry data buffer for streaming
#[derive(Debug)]
struct TelemetryBuffer {
    runtime_metrics: Vec<mesh_adapter_runtime::RuntimeMetrics>,
    gpu_metrics: Vec<mesh_adapter_gpu::GpuMetrics>,
    last_flush: std::time::Instant,
}

impl AdapterManager {
    /// Create a new adapter manager
    pub fn new(config: AdapterConfig, metrics_registry: MetricsRegistry) -> Self {
        let node_id = std::env::var("NODE_ID").unwrap_or_else(|_| {
            format!("node-{}", uuid::Uuid::new_v4().to_string()[..8].to_string())
        });
        
        Self {
            config,
            metrics_registry,
            runtime_adapters: Arc::new(RwLock::new(HashMap::new())),
            gpu_adapters: Arc::new(RwLock::new(HashMap::new())),
            telemetry_buffer: Arc::new(RwLock::new(TelemetryBuffer {
                runtime_metrics: Vec::new(),
                gpu_metrics: Vec::new(),
                last_flush: std::time::Instant::now(),
            })),
            state_plane: None,
            node_id,
        }
    }

    /// Set the state plane service for telemetry streaming
    pub fn set_state_plane(&mut self, state_plane: Arc<StatePlaneService>) {
        self.state_plane = Some(state_plane);
    }

    /// Initialize all configured adapters
    pub async fn initialize(&self) -> crate::Result<()> {
        info!("Initializing adapter manager");

        // Initialize runtime adapters
        if self.config.runtime.enabled {
            self.initialize_runtime_adapters().await?;
        }

        // Initialize GPU adapters
        if self.config.gpu.enabled {
            self.initialize_gpu_adapters().await?;
        }

        info!("Adapter manager initialized successfully");
        Ok(())
    }

    /// Start the adapter manager and begin telemetry collection
    pub async fn start(&self) -> crate::Result<()> {
        info!("Starting adapter manager");

        // Start runtime adapter monitoring
        if self.config.runtime.enabled {
            self.start_runtime_monitoring().await?;
        }

        // Start GPU adapter monitoring
        if self.config.gpu.enabled {
            self.start_gpu_monitoring().await?;
        }

        // Start telemetry streaming
        if self.config.telemetry.enabled {
            self.start_telemetry_streaming().await?;
        }

        info!("Adapter manager started successfully");
        Ok(())
    }

    /// Shutdown the adapter manager
    pub async fn shutdown(&self) -> crate::Result<()> {
        info!("Shutting down adapter manager");

        // Shutdown runtime adapters
        {
            let runtime_adapters = self.runtime_adapters.read().await;
            for (name, _adapter) in runtime_adapters.iter() {
                info!("Shutting down runtime adapter: {}", name);
                // Note: RuntimeAdapter doesn't have a mutable shutdown method
                // In a real implementation, this would properly shutdown the adapter
            }
        }

        // Shutdown GPU adapters
        {
            let gpu_adapters = self.gpu_adapters.read().await;
            for (name, _adapter) in gpu_adapters.iter() {
                info!("Shutting down GPU adapter: {}", name);
                // Note: GpuMonitor in Arc cannot be mutably accessed
                // In a real implementation, this would properly shutdown the adapter
            }
        }

        info!("Adapter manager shutdown complete");
        Ok(())
    }

    /// Get runtime adapter by name
    pub async fn get_runtime_adapter(&self, name: &str) -> Option<Arc<RuntimeAdapter>> {
        let adapters = self.runtime_adapters.read().await;
        adapters.get(name).cloned()
    }

    /// Get GPU adapter by name
    pub async fn get_gpu_adapter(&self, name: &str) -> Option<Arc<GpuMonitor>> {
        let adapters = self.gpu_adapters.read().await;
        adapters.get(name).cloned()
    }

    /// Get current telemetry data
    pub async fn get_telemetry_data(&self) -> (Vec<mesh_adapter_runtime::RuntimeMetrics>, Vec<mesh_adapter_gpu::GpuMetrics>) {
        let buffer = self.telemetry_buffer.read().await;
        (buffer.runtime_metrics.clone(), buffer.gpu_metrics.clone())
    }

    /// Initialize runtime adapters based on configuration
    async fn initialize_runtime_adapters(&self) -> crate::Result<()> {
        info!("Initializing runtime adapters");
        let mut adapters = self.runtime_adapters.write().await;

        // Initialize vLLM adapter
        if let Some(vllm_config) = &self.config.runtime.vllm {
            if vllm_config.enabled {
                info!("Initializing vLLM adapter at {}", vllm_config.endpoint);
                let adapter_config = RuntimeConfig::new(RuntimeType::VLlm);
                let runtime_adapter = RuntimeAdapter::new(adapter_config).await
                    .map_err(|e| crate::AgentError::Service(format!("Failed to create vLLM adapter: {}", e)))?;
                
                adapters.insert("vllm".to_string(), Arc::new(runtime_adapter));
                info!("vLLM adapter initialized successfully");
            }
        }

        // Initialize Triton adapter
        if let Some(triton_config) = &self.config.runtime.triton {
            if triton_config.enabled {
                info!("Initializing Triton adapter at {}", triton_config.endpoint);
                let adapter_config = RuntimeConfig::new(RuntimeType::Triton);
                let runtime_adapter = RuntimeAdapter::new(adapter_config).await
                    .map_err(|e| crate::AgentError::Service(format!("Failed to create Triton adapter: {}", e)))?;
                
                adapters.insert("triton".to_string(), Arc::new(runtime_adapter));
                info!("Triton adapter initialized successfully");
            }
        }

        // Initialize TGI adapter
        if let Some(tgi_config) = &self.config.runtime.tgi {
            if tgi_config.enabled {
                info!("Initializing TGI adapter at {}", tgi_config.endpoint);
                let adapter_config = RuntimeConfig::new(RuntimeType::Tgi);
                let runtime_adapter = RuntimeAdapter::new(adapter_config).await
                    .map_err(|e| crate::AgentError::Service(format!("Failed to create TGI adapter: {}", e)))?;
                
                adapters.insert("tgi".to_string(), Arc::new(runtime_adapter));
                info!("TGI adapter initialized successfully");
            }
        }

        info!("Initialized {} runtime adapters", adapters.len());
        Ok(())
    }

    /// Initialize GPU adapters based on configuration
    async fn initialize_gpu_adapters(&self) -> crate::Result<()> {
        info!("Initializing GPU adapters");
        let mut adapters = self.gpu_adapters.write().await;

        // Initialize NVML adapter
        if let Some(nvml_config) = &self.config.gpu.nvml {
            if nvml_config.enabled {
                info!("Initializing NVML adapter");
                let adapter_config = GpuMonitorConfig::new(GpuBackend::Nvml);
                let gpu_monitor = GpuMonitor::new(adapter_config).await
                    .map_err(|e| crate::AgentError::Service(format!("Failed to create NVML GPU monitor: {}", e)))?;
                
                adapters.insert("nvml".to_string(), Arc::new(gpu_monitor));
                info!("NVML adapter initialized successfully");
            }
        }

        // Initialize DCGM adapter
        if let Some(dcgm_config) = &self.config.gpu.dcgm {
            if dcgm_config.enabled {
                info!("Initializing DCGM adapter");
                let adapter_config = GpuMonitorConfig::new(GpuBackend::Dcgm);
                let gpu_monitor = GpuMonitor::new(adapter_config).await
                    .map_err(|e| crate::AgentError::Service(format!("Failed to create DCGM GPU monitor: {}", e)))?;
                
                adapters.insert("dcgm".to_string(), Arc::new(gpu_monitor));
                info!("DCGM adapter initialized successfully");
            }
        }

        info!("Initialized {} GPU adapters", adapters.len());
        Ok(())
    }

    /// Start runtime adapter monitoring
    async fn start_runtime_monitoring(&self) -> crate::Result<()> {
        let adapters = self.runtime_adapters.clone();
        let telemetry_buffer = self.telemetry_buffer.clone();
        let interval_duration = Duration::from_secs(self.config.runtime.metrics_collection_interval_seconds);

        tokio::spawn(async move {
            let mut interval = interval(interval_duration);
            loop {
                interval.tick().await;
                
                let adapters_guard = adapters.read().await;
                for (name, adapter) in adapters_guard.iter() {
                    match adapter.get_metrics().await {
                        Ok(metrics) => {
                            debug!("Collected metrics from runtime adapter: {}", name);
                            let mut buffer = telemetry_buffer.write().await;
                            buffer.runtime_metrics.push(metrics);
                            
                            // Limit buffer size
                            if buffer.runtime_metrics.len() > 1000 {
                                buffer.runtime_metrics.drain(0..500);
                            }
                        }
                        Err(e) => {
                            warn!("Failed to collect metrics from runtime adapter {}: {}", name, e);
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Start GPU adapter monitoring
    async fn start_gpu_monitoring(&self) -> crate::Result<()> {
        let adapters = self.gpu_adapters.clone();
        let telemetry_buffer = self.telemetry_buffer.clone();
        let interval_duration = Duration::from_secs(self.config.gpu.metrics_collection_interval_seconds);

        tokio::spawn(async move {
            let mut interval = interval(interval_duration);
            loop {
                interval.tick().await;
                
                let adapters_guard = adapters.read().await;
                for (name, adapter) in adapters_guard.iter() {
                    match adapter.get_all_metrics().await {
                        Ok(metrics_list) => {
                            debug!("Collected {} GPU metrics from adapter: {}", metrics_list.len(), name);
                            let mut buffer = telemetry_buffer.write().await;
                            buffer.gpu_metrics.extend(metrics_list);
                            
                            // Limit buffer size
                            if buffer.gpu_metrics.len() > 1000 {
                                buffer.gpu_metrics.drain(0..500);
                            }
                        }
                        Err(e) => {
                            warn!("Failed to collect GPU metrics from adapter {}: {}", name, e);
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Start telemetry streaming
    async fn start_telemetry_streaming(&self) -> crate::Result<()> {
        let telemetry_buffer = self.telemetry_buffer.clone();
        let state_plane = self.state_plane.clone();
        let node_id = self.node_id.clone();
        let flush_interval = Duration::from_secs(self.config.telemetry.flush_interval_seconds);
        let _max_batch_size = self.config.telemetry.max_batch_size;

        tokio::spawn(async move {
            let mut interval = interval(flush_interval);
            loop {
                interval.tick().await;
                
                let mut buffer = telemetry_buffer.write().await;
                let runtime_metrics_count = buffer.runtime_metrics.len();
                let gpu_metrics_count = buffer.gpu_metrics.len();
                
                if runtime_metrics_count > 0 || gpu_metrics_count > 0 {
                    info!("Streaming telemetry: {} runtime metrics, {} GPU metrics", 
                          runtime_metrics_count, gpu_metrics_count);
                    
                    // Stream runtime metrics to state plane
                    if let Some(ref state_plane) = state_plane {
                        for runtime_metric in &buffer.runtime_metrics {
                            if let Some(model_delta) = Self::convert_runtime_metric_to_model_delta(runtime_metric, &node_id) {
                                Self::stream_model_state_delta(state_plane, model_delta).await;
                            }
                        }
                    }
                    
                    // Stream GPU metrics to state plane
                    if let Some(ref state_plane) = state_plane {
                        for gpu_metric in &buffer.gpu_metrics {
                            if let Some(gpu_delta) = Self::convert_gpu_metric_to_gpu_delta(gpu_metric, &node_id) {
                                Self::stream_gpu_state_delta(state_plane, gpu_delta).await;
                            }
                        }
                    }
                    
                    // Clear the buffer
                    buffer.runtime_metrics.clear();
                    buffer.gpu_metrics.clear();
                    buffer.last_flush = std::time::Instant::now();
                }
            }
        });

        Ok(())
    }

    /// Convert runtime metrics to model state delta
    fn convert_runtime_metric_to_model_delta(
        runtime_metric: &mesh_adapter_runtime::RuntimeMetrics,
        node_id: &str,
    ) -> Option<ModelStateDelta> {
        // Extract model information from runtime metrics
        // Use the first model if available, otherwise create a default entry
        let (model_name, model_status) = if let Some((name, model_metrics)) = runtime_metric.models.iter().next() {
            (name.clone(), model_metrics.status.clone())
        } else {
            ("unknown".to_string(), "unknown".to_string())
        };

        let labels = Labels {
            model: model_name,
            revision: "latest".to_string(),
            quant: String::new(),
            runtime: runtime_metric.runtime_specific.get("runtime_type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            node: node_id.to_string(),
            gpu_uuid: String::new(),
            mig_profile: String::new(),
            tenant: String::new(),
            zone: String::new(),
            custom: HashMap::new(),
        };

        Some(ModelStateDelta {
            labels: Some(labels),
            queue_depth: Some(runtime_metric.requests.queue_size as u32),
            service_rate: Some(runtime_metric.requests.requests_per_second),
            p95_latency_ms: Some(runtime_metric.requests.p95_request_duration_ms as u32),
            batch_fullness: Some(0.0), // Not directly available in current metrics
            loaded: Some(model_status == "loaded" || model_status == "ready"),
            warming: Some(model_status == "loading" || model_status == "warming"),
            timestamp: Some(timestamp::now()),
        })
    }

    /// Convert GPU metrics to GPU state delta
    fn convert_gpu_metric_to_gpu_delta(
        gpu_metric: &mesh_adapter_gpu::GpuMetrics,
        node_id: &str,
    ) -> Option<GpuStateDelta> {
        Some(GpuStateDelta {
            gpu_uuid: gpu_metric.info.uuid.clone(),
            node: node_id.to_string(),
            sm_utilization: Some(gpu_metric.utilization.gpu as f32),
            memory_utilization: Some(gpu_metric.utilization.memory as f32),
            vram_used_gb: Some((gpu_metric.memory.used as f64 / 1024.0 / 1024.0 / 1024.0) as f32),
            vram_total_gb: Some((gpu_metric.memory.total as f64 / 1024.0 / 1024.0 / 1024.0) as f32),
            temperature_c: Some(gpu_metric.temperature.gpu as f32),
            power_watts: Some(gpu_metric.power.usage as f32),
            ecc_errors: Some(false), // Could be derived from GPU health
            throttled: Some(matches!(gpu_metric.temperature.thermal_state, ThermalState::Throttling)),
            timestamp: Some(timestamp::now()),
        })
    }

    /// Stream model state delta to state plane
    async fn stream_model_state_delta(state_plane: &StatePlaneService, delta: ModelStateDelta) {
        // In a real implementation, this would use the gRPC streaming API
        // For now, we'll directly update the state plane's internal storage
        if let Some(ref labels) = delta.labels {
            let key = StatePlaneService::model_state_key(labels);
            let mut states = state_plane.get_model_states().write().await;
            
            let state = states.entry(key.clone()).or_insert_with(|| {
                mesh_proto::state::v1::ModelState {
                    labels: delta.labels.clone(),
                    queue_depth: 0,
                    service_rate: 0.0,
                    p95_latency_ms: 0,
                    batch_fullness: 0.0,
                    loaded: false,
                    warming: false,
                    work_left_seconds: 0.0,
                    last_updated: Some(timestamp::now()),
                }
            });

            // Apply delta updates
            if let Some(queue_depth) = delta.queue_depth {
                state.queue_depth = queue_depth;
            }
            if let Some(service_rate) = delta.service_rate {
                state.service_rate = service_rate;
            }
            if let Some(p95_latency_ms) = delta.p95_latency_ms {
                state.p95_latency_ms = p95_latency_ms;
            }
            if let Some(batch_fullness) = delta.batch_fullness {
                state.batch_fullness = batch_fullness;
            }
            if let Some(loaded) = delta.loaded {
                state.loaded = loaded;
            }
            if let Some(warming) = delta.warming {
                state.warming = warming;
            }
            state.last_updated = delta.timestamp;

            debug!("Updated model state for key: {}", key);
        }
    }

    /// Stream GPU state delta to state plane
    async fn stream_gpu_state_delta(state_plane: &StatePlaneService, delta: GpuStateDelta) {
        // In a real implementation, this would use the gRPC streaming API
        // For now, we'll directly update the state plane's internal storage
        let key = StatePlaneService::gpu_state_key(&delta.gpu_uuid, &delta.node);
        let mut states = state_plane.get_gpu_states().write().await;
        
        let state = states.entry(key.clone()).or_insert_with(|| {
            mesh_proto::state::v1::GpuState {
                gpu_uuid: delta.gpu_uuid.clone(),
                node: delta.node.clone(),
                mig_profile: String::new(),
                sm_utilization: 0.0,
                memory_utilization: 0.0,
                vram_used_gb: 0.0,
                vram_total_gb: 0.0,
                temperature_c: 0.0,
                power_watts: 0.0,
                ecc_errors: false,
                throttled: false,
                last_updated: Some(timestamp::now()),
            }
        });

        // Apply delta updates
        if let Some(sm_utilization) = delta.sm_utilization {
            state.sm_utilization = sm_utilization;
        }
        if let Some(memory_utilization) = delta.memory_utilization {
            state.memory_utilization = memory_utilization;
        }
        if let Some(vram_used_gb) = delta.vram_used_gb {
            state.vram_used_gb = vram_used_gb;
        }
        if let Some(vram_total_gb) = delta.vram_total_gb {
            state.vram_total_gb = vram_total_gb;
        }
        if let Some(temperature_c) = delta.temperature_c {
            state.temperature_c = temperature_c;
        }
        if let Some(power_watts) = delta.power_watts {
            state.power_watts = power_watts;
        }
        if let Some(ecc_errors) = delta.ecc_errors {
            state.ecc_errors = ecc_errors;
        }
        if let Some(throttled) = delta.throttled {
            state.throttled = throttled;
        }
        state.last_updated = delta.timestamp;

        debug!("Updated GPU state for key: {}", key);
    }


}
