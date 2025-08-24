//! HTTP endpoint for serving metrics

use crate::{MetricsError, MetricsRegistry, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde_json::json;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{info, warn};

/// HTTP endpoint for serving metrics and health checks
#[derive(Debug, Clone)]
pub struct MetricsEndpoint {
    registry: MetricsRegistry,
    bind_addr: std::net::SocketAddr,
    server_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
}

impl MetricsEndpoint {
    /// Create a new metrics endpoint
    pub fn new(registry: MetricsRegistry, bind_addr: std::net::SocketAddr) -> Self {
        Self {
            registry,
            bind_addr,
            server_handle: None,
        }
    }

    /// Start the HTTP server
    pub async fn start(&mut self) -> Result<()> {
        if self.server_handle.is_some() {
            return Err(MetricsError::Config("Server already started".to_string()));
        }

        let app = create_app(self.registry.clone());
        let listener = TcpListener::bind(self.bind_addr).await?;
        
        info!("Starting metrics endpoint server on {}", self.bind_addr);
        
        let server_handle = tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, app).await {
                warn!("Metrics endpoint server error: {}", e);
            }
        });

        self.server_handle = Some(Arc::new(server_handle));
        Ok(())
    }

    /// Stop the HTTP server
    pub async fn stop(&mut self) {
        if let Some(handle) = self.server_handle.take() {
            if let Ok(handle) = Arc::try_unwrap(handle) {
                handle.abort();
                let _ = handle.await;
            }
        }
    }

    /// Get the metrics URL
    pub fn metrics_url(&self) -> String {
        format!("http://{}/metrics", self.bind_addr)
    }

    /// Get the health URL
    pub fn health_url(&self) -> String {
        format!("http://{}/health", self.bind_addr)
    }

    /// Check if the server is running
    pub fn is_running(&self) -> bool {
        self.server_handle.is_some()
    }
}

impl Drop for MetricsEndpoint {
    fn drop(&mut self) {
        if let Some(handle) = self.server_handle.take() {
            if let Ok(handle) = Arc::try_unwrap(handle) {
                handle.abort();
            }
        }
    }
}

/// Create the Axum application
fn create_app(registry: MetricsRegistry) -> Router {
    Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler))
        .route("/ready", get(ready_handler))
        .route("/info", get(info_handler))
        .with_state(registry)
}

/// Handler for /metrics endpoint
async fn metrics_handler(State(registry): State<MetricsRegistry>) -> Response {
    match registry.get_prometheus_metrics().await {
        Ok(Some(metrics)) => {
            (
                StatusCode::OK,
                [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
                metrics,
            ).into_response()
        }
        Ok(None) => {
            (StatusCode::NOT_FOUND, "Prometheus metrics not available").into_response()
        }
        Err(e) => {
            warn!("Failed to get Prometheus metrics: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, "Failed to get metrics").into_response()
        }
    }
}

/// Handler for /health endpoint
async fn health_handler(State(registry): State<MetricsRegistry>) -> Response {
    if registry.is_healthy() {
        (StatusCode::OK, Json(json!({
            "status": "healthy",
            "timestamp": chrono::Utc::now().to_rfc3339()
        }))).into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(json!({
            "status": "unhealthy",
            "timestamp": chrono::Utc::now().to_rfc3339()
        }))).into_response()
    }
}

/// Handler for /ready endpoint (Kubernetes readiness probe)
async fn ready_handler(State(registry): State<MetricsRegistry>) -> Response {
    if registry.is_healthy() {
        (StatusCode::OK, Json(json!({
            "status": "ready",
            "timestamp": chrono::Utc::now().to_rfc3339()
        }))).into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(json!({
            "status": "not ready",
            "timestamp": chrono::Utc::now().to_rfc3339()
        }))).into_response()
    }
}

/// Handler for /info endpoint
async fn info_handler(State(registry): State<MetricsRegistry>) -> Response {
    let global_labels = registry.global_labels();
    
    (StatusCode::OK, Json(json!({
        "service": "infermesh-metrics",
        "version": env!("CARGO_PKG_VERSION"),
        "global_labels": global_labels,
        "endpoints": {
            "metrics": "/metrics",
            "health": "/health",
            "ready": "/ready",
            "info": "/info"
        },
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MetricsRegistryBuilder, PrometheusExporter};
    use std::time::Duration;

    #[tokio::test]
    async fn test_metrics_endpoint_creation() {
        let registry = MetricsRegistryBuilder::new().build().unwrap();
        let bind_addr = "127.0.0.1:0".parse().unwrap();
        let endpoint = MetricsEndpoint::new(registry, bind_addr);
        
        assert!(!endpoint.is_running());
        assert!(endpoint.metrics_url().contains("127.0.0.1"));
        assert!(endpoint.health_url().contains("127.0.0.1"));
    }

    #[tokio::test]
    async fn test_endpoint_start_stop() {
        let registry = MetricsRegistryBuilder::new().build().unwrap();
        let bind_addr = "127.0.0.1:0".parse().unwrap();
        let mut endpoint = MetricsEndpoint::new(registry, bind_addr);
        
        // Start server
        endpoint.start().await.unwrap();
        assert!(endpoint.is_running());
        
        // Stop server
        endpoint.stop().await;
        assert!(!endpoint.is_running());
    }

    #[tokio::test]
    async fn test_endpoint_with_prometheus() {
        let prometheus_bind_addr = "127.0.0.1:0".parse().unwrap();
        let prometheus_exporter = PrometheusExporter::new(prometheus_bind_addr).unwrap();
        
        let registry = MetricsRegistryBuilder::new()
            .with_prometheus_exporter(prometheus_exporter)
            .with_global_label("test", "value")
            .build()
            .unwrap();
        
        let endpoint_bind_addr = "127.0.0.1:0".parse().unwrap();
        let mut endpoint = MetricsEndpoint::new(registry, endpoint_bind_addr);
        
        // Start endpoint
        endpoint.start().await.unwrap();
        
        // Give the server a moment to start
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // The test mainly verifies the server starts without error
        // In a real test environment, we'd make HTTP requests to verify functionality
        
        endpoint.stop().await;
    }

    #[tokio::test]
    async fn test_app_creation() {
        let registry = MetricsRegistryBuilder::new()
            .with_global_label("service", "test")
            .build()
            .unwrap();
        
        let app = create_app(registry);
        
        // Verify the app was created successfully
        // In a more comprehensive test, we'd use a test client to verify routes
        let _ = app;
    }
}
