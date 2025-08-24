//! # mesh-metrics
//!
//! Unified metrics handling for infermesh - Prometheus and OpenTelemetry integration.
//!
//! This crate provides a consistent metrics collection and export system that supports:
//! - Prometheus metrics with /metrics endpoint
//! - OpenTelemetry tracing and metrics (optional)
//! - Common metric helpers for infermesh components
//! - Structured logging with correlation IDs

pub mod common;
pub mod endpoint;
pub mod prometheus_metrics;
pub mod registry;

#[cfg(feature = "opentelemetry")]
pub mod opentelemetry_metrics;

// Re-export commonly used types
pub use common::{
    GpuMetrics, InferenceMetrics, MeshMetrics, ModelMetrics, NetworkMetrics, NodeMetrics,
};
pub use endpoint::MetricsEndpoint;
pub use prometheus_metrics::PrometheusExporter;
pub use registry::{MetricsRegistry, MetricsRegistryBuilder};

// Error handling
#[derive(Debug, thiserror::Error)]
pub enum MetricsError {
    #[error("Registry error: {0}")]
    Registry(String),

    #[error("Export error: {0}")]
    Export(String),

    #[error("HTTP server error: {0}")]
    HttpServer(#[from] hyper::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[cfg(feature = "opentelemetry")]
    #[error("OpenTelemetry error: {0}")]
    OpenTelemetry(#[from] opentelemetry::metrics::MetricsError),

    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, MetricsError>;

/// Configuration for metrics collection and export
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetricsConfig {
    /// Enable Prometheus metrics
    pub prometheus_enabled: bool,

    /// Prometheus metrics endpoint bind address
    pub prometheus_bind_addr: std::net::SocketAddr,

    /// Metrics collection interval in seconds
    pub collection_interval_seconds: u64,

    /// Enable OpenTelemetry metrics
    #[cfg(feature = "opentelemetry")]
    pub opentelemetry_enabled: bool,

    /// OpenTelemetry OTLP endpoint
    #[cfg(feature = "opentelemetry")]
    pub otlp_endpoint: Option<String>,

    /// Additional labels to add to all metrics
    pub global_labels: std::collections::HashMap<String, String>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            prometheus_enabled: true,
            prometheus_bind_addr: "127.0.0.1:9090".parse().unwrap(),
            collection_interval_seconds: 5,
            #[cfg(feature = "opentelemetry")]
            opentelemetry_enabled: false,
            #[cfg(feature = "opentelemetry")]
            otlp_endpoint: None,
            global_labels: std::collections::HashMap::new(),
        }
    }
}

/// Initialize metrics system with the given configuration
pub async fn init_metrics(config: MetricsConfig) -> Result<MetricsRegistry> {
    let mut builder = MetricsRegistryBuilder::new();

    // Add global labels
    for (key, value) in config.global_labels {
        builder = builder.with_global_label(key, value);
    }

    // Configure Prometheus if enabled
    if config.prometheus_enabled {
        let prometheus_exporter = PrometheusExporter::new(config.prometheus_bind_addr)?;
        builder = builder.with_prometheus_exporter(prometheus_exporter);
    }

    // Configure OpenTelemetry if enabled
    #[cfg(feature = "opentelemetry")]
    if config.opentelemetry_enabled {
        if let Some(endpoint) = config.otlp_endpoint {
            let otel_exporter = crate::opentelemetry_metrics::OpenTelemetryExporter::new(&endpoint)?;
            builder = builder.with_opentelemetry_exporter(otel_exporter);
        }
    }

    let registry = builder.build()?;

    // Start collection interval
    if config.collection_interval_seconds > 0 {
        let registry_clone = registry.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                tokio::time::Duration::from_secs(config.collection_interval_seconds),
            );
            loop {
                interval.tick().await;
                if let Err(e) = registry_clone.collect_and_export().await {
                    tracing::error!("Failed to collect and export metrics: {}", e);
                }
            }
        });
    }

    Ok(registry)
}

/// Macro for creating histogram metrics with consistent buckets
#[macro_export]
macro_rules! create_histogram {
    ($name:expr, $help:expr) => {
        prometheus::HistogramVec::new(
            prometheus::HistogramOpts::new($name, $help).buckets(vec![
                0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ]),
            &["node", "model", "runtime"],
        )
    };
}

/// Macro for creating counter metrics with consistent labels
#[macro_export]
macro_rules! create_counter {
    ($name:expr, $help:expr) => {
        prometheus::CounterVec::new(
            prometheus::Opts::new($name, $help),
            &["node", "model", "runtime", "status"],
        )
    };
}

/// Macro for creating gauge metrics with consistent labels
#[macro_export]
macro_rules! create_gauge {
    ($name:expr, $help:expr) => {
        prometheus::GaugeVec::new(
            prometheus::Opts::new($name, $help),
            &["node", "gpu_uuid", "model"],
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_config_default() {
        let config = MetricsConfig::default();
        assert!(config.prometheus_enabled);
        assert_eq!(config.collection_interval_seconds, 5);
        assert!(config.global_labels.is_empty());
    }

    #[tokio::test]
    async fn test_metrics_registry_creation() {
        let config = MetricsConfig {
            prometheus_enabled: false, // Disable to avoid port conflicts in tests
            ..Default::default()
        };

        let registry = init_metrics(config).await.unwrap();
        assert!(registry.is_healthy());
    }

    #[test]
    fn test_metrics_macros() {
        use prometheus::core::Collector;
        
        let histogram = create_histogram!("test_histogram", "Test histogram").unwrap();
        assert_eq!(histogram.desc()[0].fq_name, "test_histogram");

        let counter = create_counter!("test_counter", "Test counter").unwrap();
        assert_eq!(counter.desc()[0].fq_name, "test_counter");

        let gauge = create_gauge!("test_gauge", "Test gauge").unwrap();
        assert_eq!(gauge.desc()[0].fq_name, "test_gauge");
    }
}
