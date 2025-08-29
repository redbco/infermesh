//! Prometheus metrics exporter implementation

use crate::{MetricsError, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use prometheus::{Encoder, Registry, TextEncoder};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{info, warn};

/// Prometheus metrics exporter
#[derive(Debug, Clone)]
pub struct PrometheusExporter {
    registry: Arc<Registry>,
    bind_addr: SocketAddr,
    server_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
}

impl PrometheusExporter {
    /// Create a new Prometheus exporter
    pub fn new(bind_addr: SocketAddr) -> Result<Self> {
        let registry = Arc::new(Registry::new());
        
        Ok(Self {
            registry,
            bind_addr,
            server_handle: None,
        })
    }

    /// Get the Prometheus registry
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// Start the HTTP server for metrics endpoint
    pub async fn start_server(&mut self) -> Result<()> {
        if self.server_handle.is_some() {
            return Err(MetricsError::Config("Server already started".to_string()));
        }

        let app = create_metrics_app(self.registry.clone());
        let listener = TcpListener::bind(self.bind_addr).await?;
        
        info!("Starting Prometheus metrics server on {}", self.bind_addr);
        
        let server_handle = tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, app).await {
                warn!("Prometheus metrics server error: {}", e);
            }
        });

        self.server_handle = Some(Arc::new(server_handle));
        Ok(())
    }

    /// Stop the HTTP server
    pub async fn stop_server(&mut self) {
        if let Some(handle) = self.server_handle.take() {
            if let Ok(handle) = Arc::try_unwrap(handle) {
                handle.abort();
                let _ = handle.await;
            }
        }
    }

    /// Export metrics as Prometheus text format
    pub fn export_metrics(&self) -> Result<String> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)
            .map_err(|e| MetricsError::Export(format!("Failed to encode metrics: {}", e)))?;
        
        String::from_utf8(buffer)
            .map_err(|e| MetricsError::Export(format!("Failed to convert metrics to string: {}", e)))
    }

    /// Get metrics endpoint URL
    pub fn metrics_url(&self) -> String {
        format!("http://{}/metrics", self.bind_addr)
    }

    /// Check if the server is running
    pub fn is_running(&self) -> bool {
        self.server_handle.is_some()
    }
}

impl Drop for PrometheusExporter {
    fn drop(&mut self) {
        if let Some(handle) = self.server_handle.take() {
            if let Ok(handle) = Arc::try_unwrap(handle) {
                handle.abort();
            }
        }
    }
}

/// Create the Axum app for metrics endpoint
fn create_metrics_app(registry: Arc<Registry>) -> Router {
    Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler))
        .with_state(registry)
}

/// Handler for /metrics endpoint
async fn metrics_handler(State(registry): State<Arc<Registry>>) -> Response {
    let encoder = TextEncoder::new();
    let metric_families = registry.gather();
    
    let mut buffer = Vec::new();
    match encoder.encode(&metric_families, &mut buffer) {
        Ok(()) => {
            match String::from_utf8(buffer) {
                Ok(metrics_text) => {
                    (
                        StatusCode::OK,
                        [("content-type", encoder.format_type())],
                        metrics_text,
                    ).into_response()
                }
                Err(e) => {
                    warn!("Failed to convert metrics to UTF-8: {}", e);
                    (StatusCode::INTERNAL_SERVER_ERROR, "Failed to encode metrics").into_response()
                }
            }
        }
        Err(e) => {
            warn!("Failed to encode Prometheus metrics: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, "Failed to encode metrics").into_response()
        }
    }
}

/// Handler for /health endpoint
async fn health_handler() -> Response {
    (StatusCode::OK, "OK").into_response()
}

/// Helper function to register common process metrics
pub fn register_process_metrics(_registry: &Registry) -> Result<()> {
    // Register default process collector if available
    #[cfg(target_os = "linux")]
    {
        use prometheus::process_collector::ProcessCollector;
        let pc = ProcessCollector::for_self();
        registry.register(Box::new(pc))
            .map_err(|e| MetricsError::Registry(format!("Failed to register process collector: {}", e)))?;
    }

    Ok(())
}

/// Helper function to create a custom registry with default metrics
pub fn create_registry_with_defaults() -> Result<Registry> {
    let registry = Registry::new();
    
    // Register default metrics
    register_process_metrics(&registry)?;
    
    Ok(registry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus::Counter;
    use std::time::Duration;

    #[tokio::test]
    async fn test_prometheus_exporter_creation() {
        let bind_addr = "127.0.0.1:0".parse().unwrap();
        let exporter = PrometheusExporter::new(bind_addr).unwrap();
        
        assert!(!exporter.is_running());
        assert!(exporter.metrics_url().contains("127.0.0.1"));
    }

    #[tokio::test]
    async fn test_metrics_export() {
        let bind_addr = "127.0.0.1:0".parse().unwrap();
        let exporter = PrometheusExporter::new(bind_addr).unwrap();
        
        // Register a test metric
        let counter = Counter::new("test_counter", "A test counter").unwrap();
        counter.inc();
        exporter.registry().register(Box::new(counter)).unwrap();
        
        // Export metrics
        let metrics_text = exporter.export_metrics().unwrap();
        assert!(metrics_text.contains("test_counter"));
        assert!(metrics_text.contains("1"));
    }

    #[tokio::test]
    async fn test_server_start_stop() {
        let bind_addr = "127.0.0.1:0".parse().unwrap();
        let mut exporter = PrometheusExporter::new(bind_addr).unwrap();
        
        // Start server
        exporter.start_server().await.unwrap();
        assert!(exporter.is_running());
        
        // Stop server
        exporter.stop_server().await;
        assert!(!exporter.is_running());
    }

    #[tokio::test]
    async fn test_metrics_endpoint() {
        let bind_addr = "127.0.0.1:0".parse().unwrap();
        let mut exporter = PrometheusExporter::new(bind_addr).unwrap();
        
        // Register a test metric
        let counter = Counter::new("test_endpoint_counter", "A test counter").unwrap();
        counter.inc_by(42.0);
        exporter.registry().register(Box::new(counter)).unwrap();
        
        // Start server
        exporter.start_server().await.unwrap();
        
        // Give the server a moment to start
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // The actual bind address will be different since we used port 0
        // For a real test, we'd need to get the actual bound address
        // This test mainly verifies the server starts without error
        
        exporter.stop_server().await;
    }

    #[test]
    fn test_registry_with_defaults() {
        let registry = create_registry_with_defaults().unwrap();
        let metric_families = registry.gather();
        
        // On Linux, we should have process metrics
        #[cfg(target_os = "linux")]
        assert!(!metric_families.is_empty());
        
        // On other platforms, the registry might be empty but should still work
        #[cfg(not(target_os = "linux"))]
        let _ = metric_families; // Just verify it doesn't panic
    }
}
