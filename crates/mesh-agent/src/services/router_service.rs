//! Router service integration for mesh-agent

use crate::config::AgentConfig;
use crate::Result;
use mesh_metrics::MetricsRegistry;
use mesh_router::{Router, RouterConfig, RouterConfigBuilder};

use tokio::task::JoinHandle;
use tracing::{error, info, warn};

/// Router service that integrates mesh-router into mesh-agent
pub struct RouterService {
    /// Router instance
    router: Option<Router>,
    
    /// Router configuration
    config: RouterConfig,
    
    /// Service handle
    handle: Option<JoinHandle<()>>,
}

impl RouterService {
    /// Create a new router service
    pub async fn new(agent_config: &AgentConfig, _metrics_registry: &MetricsRegistry) -> Result<Self> {
        // Build router configuration from agent config
        let router_config = Self::build_router_config(agent_config)?;
        
        // Create router instance
        let router = Router::new(router_config.clone()).await
            .map_err(|e| crate::AgentError::Other(e.into()))?;
        
        Ok(Self {
            router: Some(router),
            config: router_config,
            handle: None,
        })
    }
    
    /// Start the router service
    pub async fn start(&mut self) -> Result<()> {
        if let Some(router) = self.router.take() {
            info!("Starting integrated router service");
            info!("  HTTP port: {}", self.config.http_port);
            info!("  gRPC port: {}", self.config.grpc_port);
            info!("  Bind address: {}", self.config.bind_address);
            
            // Start router in background task
            let handle = tokio::spawn(async move {
                if let Err(e) = router.serve("").await {
                    error!("Router service error: {}", e);
                }
            });
            
            self.handle = Some(handle);
            info!("Router service started successfully");
        } else {
            warn!("Router service already started or not initialized");
        }
        
        Ok(())
    }
    
    /// Stop the router service
    pub async fn stop(&mut self) -> Result<()> {
        if let Some(handle) = self.handle.take() {
            info!("Stopping router service");
            handle.abort();
            
            // Wait for graceful shutdown
            match tokio::time::timeout(std::time::Duration::from_secs(5), handle).await {
                Ok(_) => info!("Router service stopped gracefully"),
                Err(_) => warn!("Router service shutdown timed out"),
            }
        }
        
        Ok(())
    }
    
    /// Build router configuration from agent configuration
    fn build_router_config(agent_config: &AgentConfig) -> Result<RouterConfig> {
        let mut builder = RouterConfigBuilder::new();
        
        // Use different ports for router to avoid conflicts with agent gRPC
        let bind_address = agent_config.core.network.bind_ip.to_string();
        builder = builder
            .http_port(8080)  // HTTP ingress
            .grpc_port(8081)  // gRPC ingress
            .bind_address(&bind_address);
        
        // Set agent address for service discovery
        let agent_grpc_port = agent_config.core.network.grpc_port;
        let agent_addr = format!("{}:{}", bind_address, agent_grpc_port)
            .parse()
            .map_err(|e| crate::AgentError::Config(format!("Invalid agent address: {}", e)))?;
        builder = builder.agent_address(agent_addr);
        
        // Configure timeouts
        builder = builder
            .request_timeout(std::time::Duration::from_secs(30))
            .upstream_timeout(std::time::Duration::from_secs(10));
        
        // Configure features
        builder = builder
            .enable_cors(true)
            .enable_websockets(true)
            .enable_grpc_reflection(true);
        
        // Configure load balancing
        builder = builder.load_balancing(mesh_router::config::LoadBalancingStrategy::ScoreBased);
        
        let config = builder.build();
        
        // Validate configuration
        mesh_router::config::validate_config(&config)
            .map_err(|e| crate::AgentError::Config(format!("Router configuration validation failed: {}", e)))?;
        
        Ok(config)
    }
    
    /// Get router statistics
    pub async fn get_stats(&self) -> Option<mesh_router::RouterStats> {
        // Router stats would be available if we had a reference to the running router
        // For now, return None since the router is moved into the background task
        None
    }
    
    /// Check if router service is running
    pub fn is_running(&self) -> bool {
        self.handle.as_ref().map_or(false, |h| !h.is_finished())
    }
}

impl Drop for RouterService {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentConfig;
    
    #[tokio::test]
    async fn test_router_service_creation() {
        let agent_config = AgentConfig::default();
        
        let metrics = mesh_metrics::MetricsRegistryBuilder::new().build().unwrap();
        let router_service = RouterService::new(&agent_config, &metrics).await;
        
        assert!(router_service.is_ok());
        let service = router_service.unwrap();
        assert!(!service.is_running());
    }
    
    #[test]
    fn test_router_config_building() {
        let agent_config = AgentConfig::default();
        
        let router_config = RouterService::build_router_config(&agent_config);
        assert!(router_config.is_ok());
        
        let config = router_config.unwrap();
        assert_eq!(config.http_port, 8080);
        assert_eq!(config.grpc_port, 8081);
        // Default bind address should be 127.0.0.1
        assert!(config.bind_address.contains("127.0.0.1"));
    }
}
