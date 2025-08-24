//! Router configuration

use mesh_core::NodeId;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::Duration;

/// Router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// HTTP server listen port
    pub http_port: u16,
    
    /// gRPC server listen port
    pub grpc_port: u16,
    
    /// Bind address for servers
    pub bind_address: String,
    
    /// Local mesh-agent connection
    pub agent_address: SocketAddr,
    
    /// Request timeout
    pub request_timeout: Duration,
    
    /// Connection timeout for upstream services
    pub upstream_timeout: Duration,
    
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    
    /// Enable gRPC reflection
    pub enable_grpc_reflection: bool,
    
    /// Enable WebSocket support
    pub enable_websockets: bool,
    
    /// Enable CORS
    pub enable_cors: bool,
    
    /// Enable request compression
    pub enable_compression: bool,
    
    /// Maximum request body size in bytes
    pub max_request_size: usize,
    
    /// Health check configuration
    pub health_check: HealthCheckConfig,
    
    /// Routing configuration
    pub routing: RoutingConfig,
    
    /// Metrics configuration
    pub metrics: MetricsConfig,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Enable health check endpoint
    pub enabled: bool,
    
    /// Health check path
    pub path: String,
    
    /// Health check interval
    pub interval: Duration,
    
    /// Timeout for upstream health checks
    pub timeout: Duration,
}

/// Routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// Enable intelligent routing based on scoring
    pub enable_scoring: bool,
    
    /// Fallback to round-robin if scoring fails
    pub fallback_round_robin: bool,
    
    /// Maximum retries for failed requests
    pub max_retries: u32,
    
    /// Retry delay
    pub retry_delay: Duration,
    
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin load balancing
    RoundRobin,
    
    /// Random selection
    Random,
    
    /// Least connections
    LeastConnections,
    
    /// Weighted round-robin
    WeightedRoundRobin,
    
    /// Consistent hashing
    ConsistentHashing,
    
    /// Score-based routing (default)
    ScoreBased,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Enable circuit breaker
    pub enabled: bool,
    
    /// Failure threshold to open circuit
    pub failure_threshold: u32,
    
    /// Success threshold to close circuit
    pub success_threshold: u32,
    
    /// Timeout before attempting to close circuit
    pub timeout: Duration,
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enabled: bool,
    
    /// Metrics endpoint path
    pub path: String,
    
    /// Metrics port (if different from HTTP port)
    pub port: Option<u16>,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            http_port: 8080,
            grpc_port: 9090,
            bind_address: "0.0.0.0".to_string(),
            agent_address: "127.0.0.1:50051".parse().unwrap(),
            request_timeout: Duration::from_secs(30),
            upstream_timeout: Duration::from_secs(10),
            max_concurrent_requests: 1000,
            enable_grpc_reflection: true,
            enable_websockets: true,
            enable_cors: true,
            enable_compression: true,
            max_request_size: 10 * 1024 * 1024, // 10MB
            health_check: HealthCheckConfig::default(),
            routing: RoutingConfig::default(),
            metrics: MetricsConfig::default(),
        }
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: "/health".to_string(),
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
        }
    }
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            enable_scoring: true,
            fallback_round_robin: true,
            max_retries: 3,
            retry_delay: Duration::from_millis(100),
            load_balancing: LoadBalancingStrategy::ScoreBased,
            circuit_breaker: CircuitBreakerConfig::default(),
        }
    }
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(60),
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: "/metrics".to_string(),
            port: None, // Use same port as HTTP server
        }
    }
}

/// Builder for RouterConfig
pub struct RouterConfigBuilder {
    config: RouterConfig,
}

impl RouterConfigBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: RouterConfig::default(),
        }
    }

    /// Set HTTP port
    pub fn http_port(mut self, port: u16) -> Self {
        self.config.http_port = port;
        self
    }

    /// Set gRPC port
    pub fn grpc_port(mut self, port: u16) -> Self {
        self.config.grpc_port = port;
        self
    }

    /// Set bind address
    pub fn bind_address(mut self, address: impl Into<String>) -> Self {
        self.config.bind_address = address.into();
        self
    }

    /// Set agent address
    pub fn agent_address(mut self, address: SocketAddr) -> Self {
        self.config.agent_address = address;
        self
    }

    /// Set request timeout
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.config.request_timeout = timeout;
        self
    }

    /// Set upstream timeout
    pub fn upstream_timeout(mut self, timeout: Duration) -> Self {
        self.config.upstream_timeout = timeout;
        self
    }

    /// Set maximum concurrent requests
    pub fn max_concurrent_requests(mut self, max: usize) -> Self {
        self.config.max_concurrent_requests = max;
        self
    }

    /// Enable or disable gRPC reflection
    pub fn enable_grpc_reflection(mut self, enabled: bool) -> Self {
        self.config.enable_grpc_reflection = enabled;
        self
    }

    /// Enable or disable WebSocket support
    pub fn enable_websockets(mut self, enabled: bool) -> Self {
        self.config.enable_websockets = enabled;
        self
    }

    /// Enable or disable CORS
    pub fn enable_cors(mut self, enabled: bool) -> Self {
        self.config.enable_cors = enabled;
        self
    }

    /// Set load balancing strategy
    pub fn load_balancing(mut self, strategy: LoadBalancingStrategy) -> Self {
        self.config.routing.load_balancing = strategy;
        self
    }

    /// Build the configuration
    pub fn build(self) -> RouterConfig {
        self.config
    }
}

impl Default for RouterConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate router configuration
pub fn validate_config(config: &RouterConfig) -> Result<(), String> {
    if config.http_port == 0 {
        return Err("HTTP port must be greater than 0".to_string());
    }

    if config.grpc_port == 0 {
        return Err("gRPC port must be greater than 0".to_string());
    }

    if config.http_port == config.grpc_port {
        return Err("HTTP and gRPC ports must be different".to_string());
    }

    if config.request_timeout.is_zero() {
        return Err("Request timeout must be greater than 0".to_string());
    }

    if config.upstream_timeout.is_zero() {
        return Err("Upstream timeout must be greater than 0".to_string());
    }

    if config.max_concurrent_requests == 0 {
        return Err("Max concurrent requests must be greater than 0".to_string());
    }

    if config.max_request_size == 0 {
        return Err("Max request size must be greater than 0".to_string());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RouterConfig::default();
        assert_eq!(config.http_port, 8080);
        assert_eq!(config.grpc_port, 9090);
        assert_eq!(config.bind_address, "0.0.0.0");
        assert!(config.enable_grpc_reflection);
        assert!(config.enable_websockets);
        assert!(config.enable_cors);
    }

    #[test]
    fn test_config_builder() {
        let config = RouterConfigBuilder::new()
            .http_port(3000)
            .grpc_port(3001)
            .bind_address("127.0.0.1")
            .enable_cors(false)
            .load_balancing(LoadBalancingStrategy::RoundRobin)
            .build();

        assert_eq!(config.http_port, 3000);
        assert_eq!(config.grpc_port, 3001);
        assert_eq!(config.bind_address, "127.0.0.1");
        assert!(!config.enable_cors);
        assert!(matches!(config.routing.load_balancing, LoadBalancingStrategy::RoundRobin));
    }

    #[test]
    fn test_config_validation() {
        let mut config = RouterConfig::default();
        assert!(validate_config(&config).is_ok());

        // Test invalid HTTP port
        config.http_port = 0;
        assert!(validate_config(&config).is_err());

        // Test same ports
        config.http_port = 8080;
        config.grpc_port = 8080;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_load_balancing_strategies() {
        let strategies = vec![
            LoadBalancingStrategy::RoundRobin,
            LoadBalancingStrategy::Random,
            LoadBalancingStrategy::LeastConnections,
            LoadBalancingStrategy::WeightedRoundRobin,
            LoadBalancingStrategy::ConsistentHashing,
            LoadBalancingStrategy::ScoreBased,
        ];

        for strategy in strategies {
            let config = RouterConfigBuilder::new()
                .load_balancing(strategy)
                .build();
            assert!(validate_config(&config).is_ok());
        }
    }
}
