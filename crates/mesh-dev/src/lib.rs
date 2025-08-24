//! # mesh-dev
//!
//! Development and testing utilities for infermesh.
//!
//! This crate provides:
//! - Mock implementations of runtime and GPU adapters
//! - Load generation utilities for testing
//! - Integration test harness for multi-component testing
//! - Example configurations and setups

pub mod load_generator;
pub mod mock_adapters;
pub mod test_harness;
pub mod utils;

// Re-export commonly used types
pub use load_generator::{LoadGenerator, LoadGeneratorConfig, RequestPattern};
pub use mock_adapters::{MockGpuAdapter, MockRuntimeAdapter};
pub use test_harness::{TestCluster, TestClusterBuilder, TestNode};
pub use utils::{generate_test_data, setup_test_logging, TestDataGenerator};

// Error handling
#[derive(Debug, thiserror::Error)]
pub enum DevError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Mock error: {0}")]
    Mock(String),

    #[error("Load generation error: {0}")]
    LoadGeneration(String),

    #[error("Test harness error: {0}")]
    TestHarness(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Core error: {0}")]
    Core(#[from] mesh_core::Error),

    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, DevError>;

/// Initialize development environment with logging and tracing
pub fn init_dev_environment() -> Result<()> {
    setup_test_logging();
    tracing::info!("Development environment initialized");
    Ok(())
}

/// Create a simple test configuration for development
pub fn create_test_config() -> mesh_core::Config {
    let mut config = mesh_core::Config::default();
    
    // Enable mock mode
    config.node.mock = true;
    
    // Use different ports to avoid conflicts
    config.network.grpc_port = 50051;
    config.network.metrics_port = 9090;
    
    // Disable TLS for easier testing
    config.network.tls_enabled = false;
    
    // Set test-friendly timeouts
    config.runtime.request_timeout_seconds = 10;
    config.network.connection_timeout_seconds = 5;
    
    // Add test labels
    config.node.labels.insert("environment".to_string(), "test".to_string());
    config.node.labels.insert("purpose".to_string(), "development".to_string());
    
    config
}

/// Create multiple test configurations for a cluster
pub fn create_test_cluster_configs(node_count: usize) -> Vec<mesh_core::Config> {
    (0..node_count)
        .map(|i| {
            let mut config = create_test_config();
            config.node.id = mesh_core::NodeId::new(format!("test-node-{}", i));
            config.network.grpc_port = 50051 + i as u16;
            config.network.metrics_port = 9090 + i as u16;
            config.gossip.port = 7946 + i as u16;
            
            // Set zone for testing
            config.node.zone = Some(format!("test-zone-{}", i % 3));
            
            config
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_dev_environment() {
        let result = init_dev_environment();
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_test_config() {
        let config = create_test_config();
        assert!(config.node.mock);
        assert!(!config.network.tls_enabled);
        assert_eq!(config.network.grpc_port, 50051);
        assert!(config.node.labels.contains_key("environment"));
    }

    #[test]
    fn test_create_test_cluster_configs() {
        let configs = create_test_cluster_configs(3);
        assert_eq!(configs.len(), 3);
        
        // Check that ports are different
        assert_eq!(configs[0].network.grpc_port, 50051);
        assert_eq!(configs[1].network.grpc_port, 50052);
        assert_eq!(configs[2].network.grpc_port, 50053);
        
        // Check that node IDs are different
        assert_ne!(configs[0].node.id, configs[1].node.id);
        assert_ne!(configs[1].node.id, configs[2].node.id);
    }
}
