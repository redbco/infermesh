//! # mesh-net
//!
//! Networking helpers and mTLS support for infermesh.
//!
//! This crate provides:
//! - mTLS setup with Rustls for secure communication
//! - Connection pooling for gRPC clients
//! - Context propagation helpers for distributed tracing
//! - Optional QUIC/HTTP3 support for high-performance networking
//! - Network utilities and abstractions
//!
//! ## Example
//!
//! ```rust
//! use mesh_net::{TlsConfig, ConnectionPool};
//!
//! // Configure TLS (insecure for testing)
//! let tls_config = TlsConfig::insecure();
//! assert!(!tls_config.verify_peer);
//!
//! // Connection pool can be created with TLS config
//! // let pool = ConnectionPool::new(tls_config);
//! ```

use thiserror::Error;

pub mod agent_discovery;
pub mod config;
pub mod connection;
pub mod context;
pub mod discovery;
pub mod load_balancer;
pub mod pool;
pub mod tls;
pub mod utils;

#[cfg(feature = "quic")]
pub mod quic;

// Re-export commonly used types
pub use agent_discovery::AgentServiceDiscovery;
pub use config::NetworkConfig;
pub use connection::{Connection, ConnectionManager};
pub use context::{RequestContext, TraceContext};
pub use discovery::{ServiceDiscovery, MemoryServiceDiscovery, ServiceDiscoveryBuilder, ServiceDiscoveryStats};
pub use load_balancer::{LoadBalancer, LoadBalancingStrategy, LoadBalancerStats, LoadBalancerFactory};
pub use pool::{ConnectionPool, PoolConfig};
pub use tls::{TlsConfig, TlsConfigBuilder};
pub use utils::{resolve_address, validate_hostname};

/// Result type for networking operations
pub type Result<T> = std::result::Result<T, NetworkError>;

/// Errors that can occur during networking operations
#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("TLS error: {0}")]
    Tls(#[from] rustls::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Pool error: {0}")]
    Pool(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("DNS resolution error: {0}")]
    DnsResolution(String),

    #[error("Certificate error: {0}")]
    Certificate(String),

    #[error("Transport error: {0}")]
    Transport(String),

    #[cfg(feature = "quic")]
    #[error("QUIC error: {0}")]
    Quic(#[from] quinn::ConnectionError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_config_creation() {
        let config = NetworkConfig::default();
        assert!(config.connect_timeout.as_secs() > 0);
        assert!(config.request_timeout.as_secs() > 0);
    }

    #[test]
    fn test_tls_config_builder() {
        let builder = TlsConfig::builder();
        assert!(builder.cert_file.is_none());
        assert!(builder.key_file.is_none());
        assert!(builder.ca_file.is_none());
    }

    #[tokio::test]
    async fn test_connection_pool_creation() {
        let tls_config = TlsConfig::insecure();
        let pool = ConnectionPool::new(tls_config);
        assert_eq!(pool.active_connections(), 0);
    }
}
