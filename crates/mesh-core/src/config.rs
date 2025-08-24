//! Configuration management for infermesh
//!
//! Provides a unified configuration system that supports YAML files,
//! environment variables, and command-line argument overrides.

use crate::{NodeId, NodeRole, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;

/// Main configuration structure for infermesh components
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    /// Node-specific configuration
    pub node: NodeConfig,
    
    /// Runtime configuration
    pub runtime: RuntimeConfig,
    
    /// Network configuration
    pub network: NetworkConfig,
    
    /// Observability configuration
    pub observability: ObservabilityConfig,
    
    /// Gossip protocol configuration
    pub gossip: GossipConfig,
    
    /// Raft consensus configuration
    pub raft: RaftConfig,
}

impl Config {
    /// Load configuration from multiple sources with precedence:
    /// 1. Command line arguments (highest)
    /// 2. Environment variables
    /// 3. Configuration file
    /// 4. Defaults (lowest)
    pub fn load() -> Result<Self> {
        let mut builder = config::Config::builder();

        // Start with defaults
        builder = builder.add_source(config::Config::try_from(&Self::default())?);

        // Add configuration file if it exists
        if let Ok(config_path) = std::env::var("INFERMESH_CONFIG") {
            builder = builder.add_source(config::File::with_name(&config_path).required(false));
        } else {
            // Try common config file locations
            for path in &["./infermesh.yaml", "/etc/infermesh/config.yaml"] {
                builder = builder.add_source(config::File::with_name(path).required(false));
            }
        }

        // Add environment variables with INFERMESH_ prefix
        builder = builder.add_source(
            config::Environment::with_prefix("INFERMESH")
                .separator("_")
                .try_parsing(true),
        );

        let config = builder.build()?;
        let parsed: Self = config.try_deserialize()?;
        
        // Validate the configuration
        parsed.validate()?;
        
        Ok(parsed)
    }

    /// Load configuration from a specific file
    pub fn load_from_file(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let builder = config::Config::builder()
            .add_source(config::Config::try_from(&Self::default())?)
            .add_source(config::File::from(path));

        let config = builder.build()?;
        let parsed: Self = config.try_deserialize()?;
        parsed.validate()?;
        
        Ok(parsed)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Validate node configuration
        self.node.validate()?;
        
        // Validate network configuration
        self.network.validate()?;
        
        // Validate runtime configuration
        self.runtime.validate()?;
        
        Ok(())
    }

    /// Get the bind address for gRPC server
    pub fn grpc_bind_addr(&self) -> SocketAddr {
        SocketAddr::new(self.network.bind_ip, self.network.grpc_port)
    }

    /// Get the bind address for metrics server
    pub fn metrics_bind_addr(&self) -> SocketAddr {
        SocketAddr::new(self.network.bind_ip, self.network.metrics_port)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            node: NodeConfig::default(),
            runtime: RuntimeConfig::default(),
            network: NetworkConfig::default(),
            observability: ObservabilityConfig::default(),
            gossip: GossipConfig::default(),
            raft: RaftConfig::default(),
        }
    }
}

/// Node-specific configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Unique node identifier
    pub id: NodeId,
    
    /// Roles this node fulfills
    pub roles: Vec<NodeRole>,
    
    /// Zone/region identifier
    pub zone: Option<String>,
    
    /// Custom node labels
    pub labels: HashMap<String, String>,
    
    /// Enable mock mode for testing
    pub mock: bool,
}

impl NodeConfig {
    pub fn validate(&self) -> Result<()> {
        if self.roles.is_empty() {
            return Err(crate::Error::config("Node must have at least one role"));
        }
        
        Ok(())
    }

    /// Check if this node has a specific role
    pub fn has_role(&self, role: &NodeRole) -> bool {
        self.roles.contains(role)
    }

    /// Check if this node is a router
    pub fn is_router(&self) -> bool {
        self.has_role(&NodeRole::Router)
    }

    /// Check if this node is a GPU node
    pub fn is_gpu_node(&self) -> bool {
        self.has_role(&NodeRole::Gpu)
    }

    /// Check if this node is an edge node
    pub fn is_edge_node(&self) -> bool {
        self.has_role(&NodeRole::Edge)
    }
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            id: NodeId::generate(),
            roles: vec![NodeRole::Router, NodeRole::Gpu],
            zone: None,
            labels: HashMap::new(),
            mock: false,
        }
    }
}

/// Runtime configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Runtime type (triton, vllm, tgi, etc.)
    pub runtime_type: String,
    
    /// Runtime-specific configuration
    pub config: HashMap<String, serde_json::Value>,
    
    /// Model repository path
    pub model_repository: Option<PathBuf>,
    
    /// Maximum concurrent requests per model
    pub max_concurrent_requests: u32,
    
    /// Request timeout in seconds
    pub request_timeout_seconds: u64,
    
    /// Enable runtime control API
    pub enable_control: bool,
}

impl RuntimeConfig {
    pub fn validate(&self) -> Result<()> {
        if self.runtime_type.is_empty() {
            return Err(crate::Error::config("Runtime type cannot be empty"));
        }
        
        if self.max_concurrent_requests == 0 {
            return Err(crate::Error::config("Max concurrent requests must be > 0"));
        }
        
        Ok(())
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            runtime_type: "mock".to_string(),
            config: HashMap::new(),
            model_repository: None,
            max_concurrent_requests: 100,
            request_timeout_seconds: 30,
            enable_control: true,
        }
    }
}

/// Network configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// IP address to bind to
    pub bind_ip: std::net::IpAddr,
    
    /// gRPC server port
    pub grpc_port: u16,
    
    /// Metrics server port
    pub metrics_port: u16,
    
    /// Enable TLS
    pub tls_enabled: bool,
    
    /// TLS certificate file path
    pub tls_cert_file: Option<PathBuf>,
    
    /// TLS private key file path
    pub tls_key_file: Option<PathBuf>,
    
    /// TLS CA certificate file path
    pub tls_ca_file: Option<PathBuf>,
    
    /// Connection timeout in seconds
    pub connection_timeout_seconds: u64,
    
    /// Keep-alive interval in seconds
    pub keepalive_interval_seconds: u64,
}

impl NetworkConfig {
    pub fn validate(&self) -> Result<()> {
        if self.tls_enabled {
            if self.tls_cert_file.is_none() {
                return Err(crate::Error::config("TLS cert file required when TLS is enabled"));
            }
            if self.tls_key_file.is_none() {
                return Err(crate::Error::config("TLS key file required when TLS is enabled"));
            }
        }
        
        Ok(())
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            bind_ip: "127.0.0.1".parse().unwrap(),
            grpc_port: 50051,
            metrics_port: 9090,
            tls_enabled: false,
            tls_cert_file: None,
            tls_key_file: None,
            tls_ca_file: None,
            connection_timeout_seconds: 10,
            keepalive_interval_seconds: 30,
        }
    }
}

/// Observability configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Enable Prometheus metrics
    pub metrics_enabled: bool,
    
    /// Enable OpenTelemetry tracing
    pub tracing_enabled: bool,
    
    /// OpenTelemetry endpoint
    pub otlp_endpoint: Option<String>,
    
    /// Log level
    pub log_level: String,
    
    /// Log format (json or text)
    pub log_format: String,
    
    /// Metrics collection interval in seconds
    pub metrics_interval_seconds: u64,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            metrics_enabled: true,
            tracing_enabled: false,
            otlp_endpoint: None,
            log_level: "info".to_string(),
            log_format: "text".to_string(),
            metrics_interval_seconds: 5,
        }
    }
}

/// Gossip protocol configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GossipConfig {
    /// Gossip bind port
    pub port: u16,
    
    /// Initial seed nodes
    pub seeds: Vec<String>,
    
    /// Gossip interval in milliseconds
    pub gossip_interval_ms: u64,
    
    /// Probe timeout in milliseconds
    pub probe_timeout_ms: u64,
    
    /// Number of indirect probes
    pub probe_indirect_count: u32,
    
    /// Suspicion timeout in milliseconds
    pub suspicion_timeout_ms: u64,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            port: 7946,
            seeds: vec![],
            gossip_interval_ms: 200,
            probe_timeout_ms: 500,
            probe_indirect_count: 3,
            suspicion_timeout_ms: 1000,
        }
    }
}

/// Raft consensus configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RaftConfig {
    /// Raft data directory
    pub data_dir: PathBuf,
    
    /// Election timeout in milliseconds
    pub election_timeout_ms: u64,
    
    /// Heartbeat interval in milliseconds
    pub heartbeat_interval_ms: u64,
    
    /// Maximum log entries per append
    pub max_append_entries: u64,
    
    /// Snapshot threshold (log entries)
    pub snapshot_threshold: u64,
    
    /// Enable Raft (only for control plane nodes)
    pub enabled: bool,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data/raft"),
            election_timeout_ms: 1000,
            heartbeat_interval_ms: 100,
            max_append_entries: 100,
            snapshot_threshold: 1000,
            enabled: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.validate().is_ok());
        assert!(!config.node.roles.is_empty());
        assert!(config.observability.metrics_enabled);
    }

    #[test]
    fn test_node_config_roles() {
        let mut config = NodeConfig::default();
        
        assert!(config.is_router());
        assert!(config.is_gpu_node());
        assert!(!config.is_edge_node());
        
        config.roles = vec![NodeRole::Edge];
        assert!(!config.is_router());
        assert!(!config.is_gpu_node());
        assert!(config.is_edge_node());
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        
        // Valid config should pass
        assert!(config.validate().is_ok());
        
        // Empty roles should fail
        config.node.roles.clear();
        assert!(config.validate().is_err());
        
        // Reset roles
        config.node.roles = vec![NodeRole::Router];
        
        // Invalid runtime config should fail
        config.runtime.max_concurrent_requests = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_network_config_addresses() {
        let config = Config::default();
        
        let grpc_addr = config.grpc_bind_addr();
        assert_eq!(grpc_addr.port(), 50051);
        
        let metrics_addr = config.metrics_bind_addr();
        assert_eq!(metrics_addr.port(), 9090);
    }

    #[test]
    fn test_tls_validation() {
        let mut network_config = NetworkConfig::default();
        
        // TLS disabled should be valid
        assert!(network_config.validate().is_ok());
        
        // TLS enabled without cert should fail
        network_config.tls_enabled = true;
        assert!(network_config.validate().is_err());
        
        // TLS enabled with cert but no key should fail
        network_config.tls_cert_file = Some(PathBuf::from("cert.pem"));
        assert!(network_config.validate().is_err());
        
        // TLS enabled with both cert and key should pass
        network_config.tls_key_file = Some(PathBuf::from("key.pem"));
        assert!(network_config.validate().is_ok());
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        
        // Test YAML serialization
        let yaml = serde_yaml::to_string(&config).unwrap();
        let deserialized: Config = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(config.runtime.runtime_type, deserialized.runtime.runtime_type);
        
        // Test JSON serialization
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(config.network.grpc_port, deserialized.network.grpc_port);
    }
}
