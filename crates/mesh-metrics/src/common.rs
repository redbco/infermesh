//! Common metric helpers and standardized metrics for infermesh components

use prometheus::{CounterVec, GaugeVec, HistogramVec};

/// Common metrics for mesh nodes
#[derive(Debug, Clone)]
pub struct NodeMetrics {
    /// Node uptime in seconds
    pub uptime_seconds: GaugeVec,
    
    /// Node health status (1 = healthy, 0 = unhealthy)
    pub health_status: GaugeVec,
    
    /// Number of active connections
    pub active_connections: GaugeVec,
    
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: GaugeVec,
    
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: GaugeVec,
    
    /// Network bytes sent
    pub network_bytes_sent: CounterVec,
    
    /// Network bytes received
    pub network_bytes_received: CounterVec,
}

impl NodeMetrics {
    pub fn new() -> prometheus::Result<Self> {
        Ok(Self {
            uptime_seconds: GaugeVec::new(
                prometheus::Opts::new("infermesh_node_uptime_seconds", "Node uptime in seconds"),
                &["node_id", "zone"],
            )?,
            health_status: GaugeVec::new(
                prometheus::Opts::new("infermesh_node_health_status", "Node health status (1=healthy, 0=unhealthy)"),
                &["node_id", "zone"],
            )?,
            active_connections: GaugeVec::new(
                prometheus::Opts::new("infermesh_node_active_connections", "Number of active connections"),
                &["node_id", "connection_type"],
            )?,
            cpu_utilization: GaugeVec::new(
                prometheus::Opts::new("infermesh_node_cpu_utilization", "CPU utilization ratio"),
                &["node_id", "zone"],
            )?,
            memory_utilization: GaugeVec::new(
                prometheus::Opts::new("infermesh_node_memory_utilization", "Memory utilization ratio"),
                &["node_id", "zone"],
            )?,
            network_bytes_sent: CounterVec::new(
                prometheus::Opts::new("infermesh_node_network_bytes_sent_total", "Total network bytes sent"),
                &["node_id", "interface"],
            )?,
            network_bytes_received: CounterVec::new(
                prometheus::Opts::new("infermesh_node_network_bytes_received_total", "Total network bytes received"),
                &["node_id", "interface"],
            )?,
        })
    }

    /// Register all metrics with the given registry
    pub fn register(&self, registry: &prometheus::Registry) -> prometheus::Result<()> {
        registry.register(Box::new(self.uptime_seconds.clone()))?;
        registry.register(Box::new(self.health_status.clone()))?;
        registry.register(Box::new(self.active_connections.clone()))?;
        registry.register(Box::new(self.cpu_utilization.clone()))?;
        registry.register(Box::new(self.memory_utilization.clone()))?;
        registry.register(Box::new(self.network_bytes_sent.clone()))?;
        registry.register(Box::new(self.network_bytes_received.clone()))?;
        Ok(())
    }
}

impl Default for NodeMetrics {
    fn default() -> Self {
        Self::new().expect("Failed to create NodeMetrics")
    }
}

/// Common metrics for model inference
#[derive(Debug, Clone)]
pub struct ModelMetrics {
    /// Current queue depth for each model
    pub queue_depth: GaugeVec,
    
    /// Service rate (requests/second or tokens/second)
    pub service_rate: GaugeVec,
    
    /// Batch fullness ratio (0.0 to 1.0)
    pub batch_fullness: GaugeVec,
    
    /// Work left in seconds
    pub work_left_seconds: GaugeVec,
    
    /// Model load status (1 = loaded, 0 = not loaded)
    pub load_status: GaugeVec,
    
    /// Model warming status (1 = warming, 0 = not warming)
    pub warming_status: GaugeVec,
}

impl ModelMetrics {
    pub fn new() -> prometheus::Result<Self> {
        Ok(Self {
            queue_depth: GaugeVec::new(
                prometheus::Opts::new("infermesh_model_queue_depth", "Current queue depth for model"),
                &["node", "model", "revision", "runtime"],
            )?,
            service_rate: GaugeVec::new(
                prometheus::Opts::new("infermesh_model_service_rate", "Model service rate (req/s or tokens/s)"),
                &["node", "model", "revision", "runtime"],
            )?,
            batch_fullness: GaugeVec::new(
                prometheus::Opts::new("infermesh_model_batch_fullness", "Model batch fullness ratio"),
                &["node", "model", "revision", "runtime"],
            )?,
            work_left_seconds: GaugeVec::new(
                prometheus::Opts::new("infermesh_model_work_left_seconds", "Estimated work remaining in seconds"),
                &["node", "model", "revision", "runtime"],
            )?,
            load_status: GaugeVec::new(
                prometheus::Opts::new("infermesh_model_load_status", "Model load status (1=loaded, 0=not loaded)"),
                &["node", "model", "revision", "runtime"],
            )?,
            warming_status: GaugeVec::new(
                prometheus::Opts::new("infermesh_model_warming_status", "Model warming status (1=warming, 0=not warming)"),
                &["node", "model", "revision", "runtime"],
            )?,
        })
    }

    /// Update metrics from a ModelState
    pub fn update_from_state(&self, state: &mesh_core::ModelState) {
        let labels = &state.labels;
        let label_values = [
            labels.node.as_str(),
            labels.model.as_str(),
            labels.revision.as_str(),
            labels.runtime.as_str(),
        ];

        self.queue_depth.with_label_values(&label_values).set(state.queue_depth as f64);
        self.service_rate.with_label_values(&label_values).set(state.service_rate);
        self.batch_fullness.with_label_values(&label_values).set(state.batch_fullness as f64);
        self.work_left_seconds.with_label_values(&label_values).set(state.work_left_seconds as f64);
        self.load_status.with_label_values(&label_values).set(if state.loaded { 1.0 } else { 0.0 });
        self.warming_status.with_label_values(&label_values).set(if state.warming { 1.0 } else { 0.0 });
    }

    /// Register all metrics with the given registry
    pub fn register(&self, registry: &prometheus::Registry) -> prometheus::Result<()> {
        registry.register(Box::new(self.queue_depth.clone()))?;
        registry.register(Box::new(self.service_rate.clone()))?;
        registry.register(Box::new(self.batch_fullness.clone()))?;
        registry.register(Box::new(self.work_left_seconds.clone()))?;
        registry.register(Box::new(self.load_status.clone()))?;
        registry.register(Box::new(self.warming_status.clone()))?;
        Ok(())
    }
}

impl Default for ModelMetrics {
    fn default() -> Self {
        Self::new().expect("Failed to create ModelMetrics")
    }
}

/// Common metrics for GPU resources
#[derive(Debug, Clone)]
pub struct GpuMetrics {
    /// SM (Streaming Multiprocessor) utilization
    pub sm_utilization: GaugeVec,
    
    /// Memory utilization
    pub memory_utilization: GaugeVec,
    
    /// VRAM used in GB
    pub vram_used_gb: GaugeVec,
    
    /// VRAM total in GB
    pub vram_total_gb: GaugeVec,
    
    /// VRAM headroom ratio (available/total)
    pub vram_headroom: GaugeVec,
    
    /// Temperature in Celsius
    pub temperature_celsius: GaugeVec,
    
    /// Power consumption in Watts
    pub power_watts: GaugeVec,
    
    /// ECC error status (1 = errors, 0 = no errors)
    pub ecc_errors: GaugeVec,
    
    /// Throttling status (1 = throttled, 0 = not throttled)
    pub throttled: GaugeVec,
    
    /// GPU health status (1 = healthy, 0 = unhealthy)
    pub health_status: GaugeVec,
}

impl GpuMetrics {
    pub fn new() -> prometheus::Result<Self> {
        Ok(Self {
            sm_utilization: GaugeVec::new(
                prometheus::Opts::new("infermesh_gpu_sm_utilization", "GPU SM utilization ratio"),
                &["node", "gpu_uuid", "mig_profile"],
            )?,
            memory_utilization: GaugeVec::new(
                prometheus::Opts::new("infermesh_gpu_memory_utilization", "GPU memory utilization ratio"),
                &["node", "gpu_uuid", "mig_profile"],
            )?,
            vram_used_gb: GaugeVec::new(
                prometheus::Opts::new("infermesh_gpu_vram_used_gb", "GPU VRAM used in GB"),
                &["node", "gpu_uuid", "mig_profile"],
            )?,
            vram_total_gb: GaugeVec::new(
                prometheus::Opts::new("infermesh_gpu_vram_total_gb", "GPU VRAM total in GB"),
                &["node", "gpu_uuid", "mig_profile"],
            )?,
            vram_headroom: GaugeVec::new(
                prometheus::Opts::new("infermesh_gpu_vram_headroom", "GPU VRAM headroom ratio"),
                &["node", "gpu_uuid", "mig_profile"],
            )?,
            temperature_celsius: GaugeVec::new(
                prometheus::Opts::new("infermesh_gpu_temperature_celsius", "GPU temperature in Celsius"),
                &["node", "gpu_uuid"],
            )?,
            power_watts: GaugeVec::new(
                prometheus::Opts::new("infermesh_gpu_power_watts", "GPU power consumption in Watts"),
                &["node", "gpu_uuid"],
            )?,
            ecc_errors: GaugeVec::new(
                prometheus::Opts::new("infermesh_gpu_ecc_errors", "GPU ECC error status"),
                &["node", "gpu_uuid"],
            )?,
            throttled: GaugeVec::new(
                prometheus::Opts::new("infermesh_gpu_throttled", "GPU throttling status"),
                &["node", "gpu_uuid"],
            )?,
            health_status: GaugeVec::new(
                prometheus::Opts::new("infermesh_gpu_health_status", "GPU health status"),
                &["node", "gpu_uuid"],
            )?,
        })
    }

    /// Update metrics from a GpuState
    pub fn update_from_state(&self, state: &mesh_core::GpuState) {
        let mig_profile = state.mig_profile.as_deref().unwrap_or("none");
        let basic_labels = [state.node.as_str(), state.gpu_uuid.as_str()];
        let mig_labels = [state.node.as_str(), state.gpu_uuid.as_str(), mig_profile];

        self.sm_utilization.with_label_values(&mig_labels).set(state.sm_utilization as f64);
        self.memory_utilization.with_label_values(&mig_labels).set(state.memory_utilization as f64);
        self.vram_used_gb.with_label_values(&mig_labels).set(state.vram_used_gb as f64);
        self.vram_total_gb.with_label_values(&mig_labels).set(state.vram_total_gb as f64);
        self.vram_headroom.with_label_values(&mig_labels).set(state.vram_headroom() as f64);

        if let Some(temp) = state.temperature_c {
            self.temperature_celsius.with_label_values(&basic_labels).set(temp as f64);
        }

        if let Some(power) = state.power_watts {
            self.power_watts.with_label_values(&basic_labels).set(power as f64);
        }

        self.ecc_errors.with_label_values(&basic_labels).set(if state.ecc_errors { 1.0 } else { 0.0 });
        self.throttled.with_label_values(&basic_labels).set(if state.throttled { 1.0 } else { 0.0 });
        self.health_status.with_label_values(&basic_labels).set(if state.is_healthy() { 1.0 } else { 0.0 });
    }

    /// Register all metrics with the given registry
    pub fn register(&self, registry: &prometheus::Registry) -> prometheus::Result<()> {
        registry.register(Box::new(self.sm_utilization.clone()))?;
        registry.register(Box::new(self.memory_utilization.clone()))?;
        registry.register(Box::new(self.vram_used_gb.clone()))?;
        registry.register(Box::new(self.vram_total_gb.clone()))?;
        registry.register(Box::new(self.vram_headroom.clone()))?;
        registry.register(Box::new(self.temperature_celsius.clone()))?;
        registry.register(Box::new(self.power_watts.clone()))?;
        registry.register(Box::new(self.ecc_errors.clone()))?;
        registry.register(Box::new(self.throttled.clone()))?;
        registry.register(Box::new(self.health_status.clone()))?;
        Ok(())
    }
}

impl Default for GpuMetrics {
    fn default() -> Self {
        Self::new().expect("Failed to create GpuMetrics")
    }
}

/// Common metrics for inference requests
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    /// Request latency histogram
    pub request_latency: HistogramVec,
    
    /// Queue wait time histogram
    pub queue_wait_time: HistogramVec,
    
    /// Request count by outcome
    pub request_count: CounterVec,
    
    /// Tokens processed
    pub tokens_processed: CounterVec,
    
    /// Current active requests
    pub active_requests: GaugeVec,
}

impl InferenceMetrics {
    pub fn new() -> prometheus::Result<Self> {
        Ok(Self {
            request_latency: HistogramVec::new(
                prometheus::HistogramOpts::new(
                    "infermesh_inference_request_latency_seconds",
                    "Request latency in seconds"
                ).buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
                &["node", "model", "slo_class"],
            )?,
            queue_wait_time: HistogramVec::new(
                prometheus::HistogramOpts::new(
                    "infermesh_inference_queue_wait_seconds",
                    "Queue wait time in seconds"
                ).buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]),
                &["node", "model", "slo_class"],
            )?,
            request_count: CounterVec::new(
                prometheus::Opts::new("infermesh_inference_requests_total", "Total inference requests"),
                &["node", "model", "slo_class", "outcome"],
            )?,
            tokens_processed: CounterVec::new(
                prometheus::Opts::new("infermesh_inference_tokens_total", "Total tokens processed"),
                &["node", "model", "type"],
            )?,
            active_requests: GaugeVec::new(
                prometheus::Opts::new("infermesh_inference_active_requests", "Current active requests"),
                &["node", "model", "slo_class"],
            )?,
        })
    }

    /// Register all metrics with the given registry
    pub fn register(&self, registry: &prometheus::Registry) -> prometheus::Result<()> {
        registry.register(Box::new(self.request_latency.clone()))?;
        registry.register(Box::new(self.queue_wait_time.clone()))?;
        registry.register(Box::new(self.request_count.clone()))?;
        registry.register(Box::new(self.tokens_processed.clone()))?;
        registry.register(Box::new(self.active_requests.clone()))?;
        Ok(())
    }
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self::new().expect("Failed to create InferenceMetrics")
    }
}

/// Common metrics for network operations
#[derive(Debug, Clone)]
pub struct NetworkMetrics {
    /// gRPC request latency
    pub grpc_request_latency: HistogramVec,
    
    /// gRPC request count
    pub grpc_request_count: CounterVec,
    
    /// Connection count by type
    pub connection_count: GaugeVec,
    
    /// Bytes sent/received
    pub bytes_transferred: CounterVec,
}

impl NetworkMetrics {
    pub fn new() -> prometheus::Result<Self> {
        Ok(Self {
            grpc_request_latency: HistogramVec::new(
                prometheus::HistogramOpts::new(
                    "infermesh_grpc_request_latency_seconds",
                    "gRPC request latency in seconds"
                ).buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
                &["service", "method", "status"],
            )?,
            grpc_request_count: CounterVec::new(
                prometheus::Opts::new("infermesh_grpc_requests_total", "Total gRPC requests"),
                &["service", "method", "status"],
            )?,
            connection_count: GaugeVec::new(
                prometheus::Opts::new("infermesh_connections", "Current connections"),
                &["type", "state"],
            )?,
            bytes_transferred: CounterVec::new(
                prometheus::Opts::new("infermesh_bytes_transferred_total", "Total bytes transferred"),
                &["direction", "protocol"],
            )?,
        })
    }

    /// Register all metrics with the given registry
    pub fn register(&self, registry: &prometheus::Registry) -> prometheus::Result<()> {
        registry.register(Box::new(self.grpc_request_latency.clone()))?;
        registry.register(Box::new(self.grpc_request_count.clone()))?;
        registry.register(Box::new(self.connection_count.clone()))?;
        registry.register(Box::new(self.bytes_transferred.clone()))?;
        Ok(())
    }
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        Self::new().expect("Failed to create NetworkMetrics")
    }
}

/// Combined metrics for the entire mesh
#[derive(Debug, Clone)]
pub struct MeshMetrics {
    pub node: NodeMetrics,
    pub model: ModelMetrics,
    pub gpu: GpuMetrics,
    pub inference: InferenceMetrics,
    pub network: NetworkMetrics,
}

impl MeshMetrics {
    pub fn new() -> prometheus::Result<Self> {
        Ok(Self {
            node: NodeMetrics::new()?,
            model: ModelMetrics::new()?,
            gpu: GpuMetrics::new()?,
            inference: InferenceMetrics::new()?,
            network: NetworkMetrics::new()?,
        })
    }

    /// Register all metrics with the given registry
    pub fn register(&self, registry: &prometheus::Registry) -> prometheus::Result<()> {
        self.node.register(registry)?;
        self.model.register(registry)?;
        self.gpu.register(registry)?;
        self.inference.register(registry)?;
        self.network.register(registry)?;
        Ok(())
    }
}

impl Default for MeshMetrics {
    fn default() -> Self {
        Self::new().expect("Failed to create MeshMetrics")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_core::{GpuState, Labels, ModelState};

    #[test]
    fn test_node_metrics_creation() {
        let metrics = NodeMetrics::new().unwrap();
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();
        
        // Test that we can set values
        metrics.uptime_seconds.with_label_values(&["node1", "us-west-2"]).set(3600.0);
        metrics.health_status.with_label_values(&["node1", "us-west-2"]).set(1.0);
    }

    #[test]
    fn test_model_metrics_update() {
        let metrics = ModelMetrics::new().unwrap();
        let labels = Labels::new("gpt-4", "v1.0", "triton", "node1");
        let state = ModelState::new(labels);
        
        metrics.update_from_state(&state);
        
        // Verify metrics were updated
        let queue_depth = metrics.queue_depth
            .with_label_values(&["node1", "gpt-4", "v1.0", "triton"])
            .get();
        assert_eq!(queue_depth, 0.0);
    }

    #[test]
    fn test_gpu_metrics_update() {
        let metrics = GpuMetrics::new().unwrap();
        let mut state = GpuState::new("GPU-12345", "node1");
        state.update_metrics(0.8, 0.6, 8.0, 16.0);
        state.update_thermal(Some(75.0), Some(250.0));
        
        metrics.update_from_state(&state);
        
        // Verify metrics were updated
        let sm_util = metrics.sm_utilization
            .with_label_values(&["node1", "GPU-12345", "none"])
            .get();
        assert!((sm_util - 0.8).abs() < 0.001); // Use floating point comparison
        
        let temp = metrics.temperature_celsius
            .with_label_values(&["node1", "GPU-12345"])
            .get();
        assert_eq!(temp, 75.0);
    }

    #[test]
    fn test_mesh_metrics_registration() {
        let metrics = MeshMetrics::new().unwrap();
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();
        
        // Add some test data to make metrics appear in gather
        metrics.node.uptime_seconds.with_label_values(&["test", "zone"]).set(100.0);
        
        // Verify that metrics are registered by checking the gather output
        let metric_families = registry.gather();
        assert!(!metric_families.is_empty());
    }
}
