//! # mesh-router
//!
//! HTTP/gRPC ingress and request routing for infermesh.
//!
//! This crate provides:
//! - HTTP/1.1 and HTTP/2 ingress endpoints
//! - gRPC service routing and proxying
//! - WebSocket support for streaming responses
//! - Integration with mesh-state for intelligent routing
//! - Request forwarding and response streaming
//! - Load balancing and failover
//!
//! ## Example
//!
//! ```rust,no_run
//! use mesh_router::{Router, RouterConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = RouterConfig::default();
//!     let router = Router::new(config).await?;
//!     
//!     // Start the router
//!     router.serve("0.0.0.0:8080").await?;
//!     Ok(())
//! }
//! ```

use thiserror::Error;

pub mod config;
pub mod handler;
pub mod proxy;
pub mod router;
pub mod server;

// Re-export main types
pub use config::{RouterConfig, RouterConfigBuilder};
pub use router::{Router, RouterStats};
pub use server::{HttpServer, GrpcServer};

/// Result type for router operations
pub type Result<T> = std::result::Result<T, RouterError>;

/// Errors that can occur during router operations
#[derive(Error, Debug)]
pub enum RouterError {
    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Server error: {0}")]
    Server(String),

    #[error("Routing error: {0}")]
    Routing(String),

    #[error("Proxy error: {0}")]
    Proxy(String),

    #[error("State error: {0}")]
    State(#[from] mesh_state::StateError),

    #[error("Network error: {0}")]
    Network(#[from] mesh_net::NetworkError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] hyper::Error),

    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::Status),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Request timeout")]
    Timeout,

    #[error("Service unavailable")]
    ServiceUnavailable,

    #[error("Service discovery error: {0}")]
    ServiceDiscovery(String),

    #[error("Bad gateway")]
    BadGateway,
}

impl RouterError {
    /// Convert to HTTP status code
    pub fn to_status_code(&self) -> u16 {
        match self {
            RouterError::Configuration(_) => 500,
            RouterError::Server(_) => 500,
            RouterError::Routing(_) => 502,
            RouterError::Proxy(_) => 502,
            RouterError::State(_) => 500,
            RouterError::Network(_) => 502,
            RouterError::Io(_) => 500,
            RouterError::Http(_) => 502,
            RouterError::Grpc(_) => 502,
            RouterError::Json(_) => 400,
            RouterError::Timeout => 504,
            RouterError::ServiceUnavailable => 503,
            RouterError::ServiceDiscovery(_) => 503,
            RouterError::BadGateway => 502,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_error_status_codes() {
        assert_eq!(RouterError::Configuration("test".to_string()).to_status_code(), 500);
        assert_eq!(RouterError::Timeout.to_status_code(), 504);
        assert_eq!(RouterError::ServiceUnavailable.to_status_code(), 503);
    }

    #[tokio::test]
    async fn test_router_config_creation() {
        let config = RouterConfig::default();
        assert!(config.http_port > 0);
        assert!(config.grpc_port > 0);
    }
}
