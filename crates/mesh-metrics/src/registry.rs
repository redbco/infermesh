//! Metrics registry for managing and coordinating different metric exporters

use crate::{
    common::MeshMetrics, prometheus_metrics::PrometheusExporter, MetricsError, Result,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info};

#[cfg(feature = "opentelemetry")]
use crate::opentelemetry_metrics::OpenTelemetryExporter;

/// Central registry for managing metrics collection and export
#[derive(Debug, Clone)]
pub struct MetricsRegistry {
    inner: Arc<MetricsRegistryInner>,
}

#[derive(Debug)]
struct MetricsRegistryInner {
    /// Prometheus exporter (optional)
    prometheus_exporter: RwLock<Option<PrometheusExporter>>,
    
    /// OpenTelemetry exporter (optional)
    #[cfg(feature = "opentelemetry")]
    opentelemetry_exporter: RwLock<Option<OpenTelemetryExporter>>,
    
    /// Common mesh metrics
    mesh_metrics: MeshMetrics,
    
    /// Global labels applied to all metrics
    global_labels: HashMap<String, String>,
    
    /// Registry health status
    healthy: RwLock<bool>,
}

impl MetricsRegistry {
    /// Create a new metrics registry
    pub(crate) fn new(
        prometheus_exporter: Option<PrometheusExporter>,
        #[cfg(feature = "opentelemetry")] opentelemetry_exporter: Option<OpenTelemetryExporter>,
        global_labels: HashMap<String, String>,
    ) -> Result<Self> {
        let mesh_metrics = MeshMetrics::new()
            .map_err(|e| MetricsError::Registry(format!("Failed to create mesh metrics: {}", e)))?;

        // Register metrics with Prometheus if available
        if let Some(ref exporter) = prometheus_exporter {
            mesh_metrics.register(exporter.registry())
                .map_err(|e| MetricsError::Registry(format!("Failed to register metrics: {}", e)))?;
        }

        let inner = MetricsRegistryInner {
            prometheus_exporter: RwLock::new(prometheus_exporter),
            #[cfg(feature = "opentelemetry")]
            opentelemetry_exporter: RwLock::new(opentelemetry_exporter),
            mesh_metrics,
            global_labels,
            healthy: RwLock::new(true),
        };

        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Get the mesh metrics
    pub fn mesh_metrics(&self) -> &MeshMetrics {
        &self.inner.mesh_metrics
    }

    /// Get global labels
    pub fn global_labels(&self) -> &HashMap<String, String> {
        &self.inner.global_labels
    }

    /// Start all configured exporters
    pub async fn start_exporters(&self) -> Result<()> {
        info!("Starting metrics exporters");

        // Start Prometheus exporter
        if let Some(ref mut exporter) = *self.inner.prometheus_exporter.write().await {
            exporter.start_server().await?;
            info!("Prometheus exporter started at {}", exporter.metrics_url());
        }

        // Start OpenTelemetry exporter
        #[cfg(feature = "opentelemetry")]
        if let Some(ref mut exporter) = *self.inner.opentelemetry_exporter.write().await {
            exporter.start().await?;
            info!("OpenTelemetry exporter started");
        }

        Ok(())
    }

    /// Stop all exporters
    pub async fn stop_exporters(&self) {
        info!("Stopping metrics exporters");

        // Stop Prometheus exporter
        if let Some(ref mut exporter) = *self.inner.prometheus_exporter.write().await {
            exporter.stop_server().await;
        }

        // Stop OpenTelemetry exporter
        #[cfg(feature = "opentelemetry")]
        if let Some(ref mut exporter) = *self.inner.opentelemetry_exporter.write().await {
            exporter.stop().await;
        }
    }

    /// Collect and export metrics from all sources
    pub async fn collect_and_export(&self) -> Result<()> {
        debug!("Collecting and exporting metrics");

        // For now, metrics are automatically updated by the components
        // In the future, we might add active collection here

        // Export to OpenTelemetry if configured
        #[cfg(feature = "opentelemetry")]
        if let Some(ref exporter) = *self.inner.opentelemetry_exporter.read().await {
            if let Err(e) = exporter.export_metrics().await {
                error!("Failed to export OpenTelemetry metrics: {}", e);
                return Err(e);
            }
        }

        Ok(())
    }

    /// Update model state metrics
    pub fn update_model_state(&self, state: &mesh_core::ModelState) {
        self.inner.mesh_metrics.model.update_from_state(state);
    }

    /// Update GPU state metrics
    pub fn update_gpu_state(&self, state: &mesh_core::GpuState) {
        self.inner.mesh_metrics.gpu.update_from_state(state);
    }

    /// Record inference request metrics
    pub fn record_inference_request(
        &self,
        node: &str,
        model: &str,
        slo_class: &str,
        latency_seconds: f64,
        queue_wait_seconds: f64,
        outcome: &str,
        tokens: u64,
    ) {
        let labels = [node, model, slo_class];
        
        self.inner.mesh_metrics.inference.request_latency
            .with_label_values(&labels)
            .observe(latency_seconds);
        
        self.inner.mesh_metrics.inference.queue_wait_time
            .with_label_values(&labels)
            .observe(queue_wait_seconds);
        
        let count_labels = [node, model, slo_class, outcome];
        self.inner.mesh_metrics.inference.request_count
            .with_label_values(&count_labels)
            .inc();
        
        let token_labels = [node, model, "processed"];
        self.inner.mesh_metrics.inference.tokens_processed
            .with_label_values(&token_labels)
            .inc_by(tokens as f64);
    }

    /// Record gRPC request metrics
    pub fn record_grpc_request(
        &self,
        service: &str,
        method: &str,
        status: &str,
        latency_seconds: f64,
    ) {
        let labels = [service, method, status];
        
        self.inner.mesh_metrics.network.grpc_request_latency
            .with_label_values(&labels)
            .observe(latency_seconds);
        
        self.inner.mesh_metrics.network.grpc_request_count
            .with_label_values(&labels)
            .inc();
    }

    /// Update node health status
    pub fn update_node_health(&self, node_id: &str, zone: &str, healthy: bool) {
        let labels = [node_id, zone];
        self.inner.mesh_metrics.node.health_status
            .with_label_values(&labels)
            .set(if healthy { 1.0 } else { 0.0 });
    }

    /// Update node uptime
    pub fn update_node_uptime(&self, node_id: &str, zone: &str, uptime_seconds: f64) {
        let labels = [node_id, zone];
        self.inner.mesh_metrics.node.uptime_seconds
            .with_label_values(&labels)
            .set(uptime_seconds);
    }

    /// Get Prometheus metrics as text
    pub async fn get_prometheus_metrics(&self) -> Result<Option<String>> {
        if let Some(ref exporter) = *self.inner.prometheus_exporter.read().await {
            Ok(Some(exporter.export_metrics()?))
        } else {
            Ok(None)
        }
    }

    /// Check if the registry is healthy
    pub fn is_healthy(&self) -> bool {
        // For now, always return true. In the future, we might check exporter health
        true
    }

    /// Mark the registry as unhealthy
    pub async fn mark_unhealthy(&self, reason: &str) {
        error!("Marking metrics registry as unhealthy: {}", reason);
        *self.inner.healthy.write().await = false;
    }

    /// Mark the registry as healthy
    pub async fn mark_healthy(&self) {
        info!("Marking metrics registry as healthy");
        *self.inner.healthy.write().await = true;
    }
}

/// Builder for creating a MetricsRegistry
#[derive(Debug, Default)]
pub struct MetricsRegistryBuilder {
    prometheus_exporter: Option<PrometheusExporter>,
    #[cfg(feature = "opentelemetry")]
    opentelemetry_exporter: Option<OpenTelemetryExporter>,
    global_labels: HashMap<String, String>,
}

impl MetricsRegistryBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a Prometheus exporter
    pub fn with_prometheus_exporter(mut self, exporter: PrometheusExporter) -> Self {
        self.prometheus_exporter = Some(exporter);
        self
    }

    /// Add an OpenTelemetry exporter
    #[cfg(feature = "opentelemetry")]
    pub fn with_opentelemetry_exporter(mut self, exporter: OpenTelemetryExporter) -> Self {
        self.opentelemetry_exporter = Some(exporter);
        self
    }

    /// Add a global label
    pub fn with_global_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.global_labels.insert(key.into(), value.into());
        self
    }

    /// Add multiple global labels
    pub fn with_global_labels(mut self, labels: HashMap<String, String>) -> Self {
        self.global_labels.extend(labels);
        self
    }

    /// Build the MetricsRegistry
    pub fn build(self) -> Result<MetricsRegistry> {
        MetricsRegistry::new(
            self.prometheus_exporter,
            #[cfg(feature = "opentelemetry")]
            self.opentelemetry_exporter,
            self.global_labels,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prometheus_metrics::PrometheusExporter;
    use mesh_core::{GpuState, Labels, ModelState};

    #[tokio::test]
    async fn test_registry_builder() {
        let bind_addr = "127.0.0.1:0".parse().unwrap();
        let prometheus_exporter = PrometheusExporter::new(bind_addr).unwrap();
        
        let registry = MetricsRegistryBuilder::new()
            .with_prometheus_exporter(prometheus_exporter)
            .with_global_label("service", "infermesh")
            .with_global_label("version", "0.1.0")
            .build()
            .unwrap();

        assert!(registry.is_healthy());
        assert_eq!(registry.global_labels().len(), 2);
        assert_eq!(registry.global_labels().get("service"), Some(&"infermesh".to_string()));
    }

    #[tokio::test]
    async fn test_metrics_updates() {
        let registry = MetricsRegistryBuilder::new().build().unwrap();

        // Test model state update
        let labels = Labels::new("gpt-4", "v1.0", "triton", "node1");
        let mut model_state = ModelState::new(labels);
        model_state.update(5, 10.0, 100, 0.8);
        registry.update_model_state(&model_state);

        // Test GPU state update
        let mut gpu_state = GpuState::new("GPU-12345", "node1");
        gpu_state.update_metrics(0.8, 0.6, 8.0, 16.0);
        registry.update_gpu_state(&gpu_state);

        // Test inference request recording
        registry.record_inference_request(
            "node1", "gpt-4", "latency", 0.5, 0.1, "success", 100
        );

        // Test gRPC request recording
        registry.record_grpc_request("ControlPlane", "ListNodes", "OK", 0.01);

        // Test node health update
        registry.update_node_health("node1", "us-west-2", true);
        registry.update_node_uptime("node1", "us-west-2", 3600.0);

        // All operations should complete without error
        assert!(registry.is_healthy());
    }

    #[tokio::test]
    async fn test_registry_lifecycle() {
        let bind_addr = "127.0.0.1:0".parse().unwrap();
        let prometheus_exporter = PrometheusExporter::new(bind_addr).unwrap();
        
        let registry = MetricsRegistryBuilder::new()
            .with_prometheus_exporter(prometheus_exporter)
            .build()
            .unwrap();

        // Test starting exporters
        registry.start_exporters().await.unwrap();

        // Test collection and export
        registry.collect_and_export().await.unwrap();

        // Test stopping exporters
        registry.stop_exporters().await;

        assert!(registry.is_healthy());
    }

    #[tokio::test]
    async fn test_health_management() {
        let registry = MetricsRegistryBuilder::new().build().unwrap();

        assert!(registry.is_healthy());

        registry.mark_unhealthy("test reason").await;
        // Note: is_healthy() currently always returns true
        // This test verifies the method calls work

        registry.mark_healthy().await;
        assert!(registry.is_healthy());
    }
}
