//! Service discovery and endpoint management

use crate::config::NetworkConfig;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use mesh_core::{NodeId, ServiceEndpoint};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Service discovery trait
#[async_trait]
pub trait ServiceDiscovery: Send + Sync {
    /// Register a service endpoint
    async fn register(&self, node_id: &NodeId, endpoint: ServiceEndpoint) -> Result<()>;
    
    /// Unregister a service endpoint
    async fn unregister(&self, node_id: &NodeId, service_name: &str) -> Result<()>;
    
    /// Discover service endpoints
    async fn discover(&self, service_name: &str) -> Result<Vec<(NodeId, ServiceEndpoint)>>;
    
    /// Get all registered services for a node
    async fn get_node_services(&self, node_id: &NodeId) -> Result<Vec<ServiceEndpoint>>;
    
    /// Get all nodes providing a service
    async fn get_service_nodes(&self, service_name: &str) -> Result<Vec<NodeId>>;
    
    /// Health check a service endpoint
    async fn health_check(&self, node_id: &NodeId, service_name: &str) -> Result<bool>;
}

/// In-memory service discovery implementation
pub struct MemoryServiceDiscovery {
    /// Services by node ID
    services: Arc<RwLock<HashMap<NodeId, HashMap<String, ServiceEndpoint>>>>,
    
    /// Configuration
    config: NetworkConfig,
}

impl MemoryServiceDiscovery {
    /// Create a new memory service discovery
    pub fn new(config: NetworkConfig) -> Self {
        Self {
            services: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }
    
    /// Get service statistics
    pub async fn get_stats(&self) -> ServiceDiscoveryStats {
        let services = self.services.read().await;
        let total_nodes = services.len();
        let mut total_services = 0;
        let mut services_by_type = HashMap::new();
        
        for node_services in services.values() {
            total_services += node_services.len();
            for (service_name, _) in node_services {
                *services_by_type.entry(service_name.clone()).or_insert(0) += 1;
            }
        }
        
        ServiceDiscoveryStats {
            total_nodes,
            total_services,
            services_by_type,
        }
    }
    
    /// List all registered services
    pub async fn list_all_services(&self) -> HashMap<NodeId, HashMap<String, ServiceEndpoint>> {
        self.services.read().await.clone()
    }
    
    /// Clear all registered services
    pub async fn clear(&self) {
        let mut services = self.services.write().await;
        services.clear();
        info!("Cleared all registered services");
    }
}

#[async_trait]
impl ServiceDiscovery for MemoryServiceDiscovery {
    async fn register(&self, node_id: &NodeId, endpoint: ServiceEndpoint) -> Result<()> {
        let mut services = self.services.write().await;
        let node_services = services.entry(node_id.clone()).or_insert_with(HashMap::new);
        
        let service_name = endpoint.service_name.clone();
        node_services.insert(service_name.clone(), endpoint);
        
        info!(
            node_id = %node_id,
            service_name = %service_name,
            "Registered service endpoint"
        );
        
        Ok(())
    }
    
    async fn unregister(&self, node_id: &NodeId, service_name: &str) -> Result<()> {
        let mut services = self.services.write().await;
        
        if let Some(node_services) = services.get_mut(node_id) {
            if node_services.remove(service_name).is_some() {
                info!(
                    node_id = %node_id,
                    service_name = %service_name,
                    "Unregistered service endpoint"
                );
                
                // Remove node entry if no services left
                if node_services.is_empty() {
                    services.remove(node_id);
                }
                
                return Ok(());
            }
        }
        
        warn!(
            node_id = %node_id,
            service_name = %service_name,
            "Attempted to unregister non-existent service"
        );
        
        Err(anyhow!("Service not found"))
    }
    
    async fn discover(&self, service_name: &str) -> Result<Vec<(NodeId, ServiceEndpoint)>> {
        let services = self.services.read().await;
        let mut results = Vec::new();
        
        for (node_id, node_services) in services.iter() {
            if let Some(endpoint) = node_services.get(service_name) {
                results.push((node_id.clone(), endpoint.clone()));
            }
        }
        
        debug!(
            service_name = %service_name,
            count = results.len(),
            "Discovered service endpoints"
        );
        
        Ok(results)
    }
    
    async fn get_node_services(&self, node_id: &NodeId) -> Result<Vec<ServiceEndpoint>> {
        let services = self.services.read().await;
        
        if let Some(node_services) = services.get(node_id) {
            Ok(node_services.values().cloned().collect())
        } else {
            Ok(Vec::new())
        }
    }
    
    async fn get_service_nodes(&self, service_name: &str) -> Result<Vec<NodeId>> {
        let services = self.services.read().await;
        let mut nodes = Vec::new();
        
        for (node_id, node_services) in services.iter() {
            if node_services.contains_key(service_name) {
                nodes.push(node_id.clone());
            }
        }
        
        Ok(nodes)
    }
    
    async fn health_check(&self, node_id: &NodeId, service_name: &str) -> Result<bool> {
        let services = self.services.read().await;
        
        if let Some(node_services) = services.get(node_id) {
            if let Some(endpoint) = node_services.get(service_name) {
                // Simple health check - try to connect to the endpoint
                match self.check_endpoint_health(&endpoint.address).await {
                    Ok(healthy) => {
                        debug!(
                            node_id = %node_id,
                            service_name = %service_name,
                            healthy = healthy,
                            "Health check completed"
                        );
                        Ok(healthy)
                    }
                    Err(e) => {
                        error!(
                            node_id = %node_id,
                            service_name = %service_name,
                            error = %e,
                            "Health check failed"
                        );
                        Ok(false)
                    }
                }
            } else {
                Err(anyhow!("Service not found"))
            }
        } else {
            Err(anyhow!("Node not found"))
        }
    }
}

impl MemoryServiceDiscovery {
    /// Check endpoint health by attempting to connect
    async fn check_endpoint_health(&self, address: &SocketAddr) -> Result<bool> {
        // Simple TCP connection test
        match tokio::time::timeout(
            self.config.connect_timeout,
            tokio::net::TcpStream::connect(address)
        ).await {
            Ok(Ok(_)) => Ok(true),
            Ok(Err(_)) => Ok(false),
            Err(_) => Ok(false), // Timeout
        }
    }
}

/// Service discovery statistics
#[derive(Debug, Clone)]
pub struct ServiceDiscoveryStats {
    /// Total number of registered nodes
    pub total_nodes: usize,
    
    /// Total number of registered services
    pub total_services: usize,
    
    /// Number of services by type
    pub services_by_type: HashMap<String, usize>,
}

/// Service discovery builder
pub struct ServiceDiscoveryBuilder {
    config: NetworkConfig,
}

impl ServiceDiscoveryBuilder {
    /// Create a new service discovery builder
    pub fn new(config: NetworkConfig) -> Self {
        Self { config }
    }
    
    /// Build a memory-based service discovery
    pub fn build_memory(self) -> Arc<dyn ServiceDiscovery> {
        Arc::new(MemoryServiceDiscovery::new(self.config))
    }
    
    /// Build service discovery based on configuration
    pub fn build(self) -> Arc<dyn ServiceDiscovery> {
        // For now, only memory-based discovery is implemented
        // In the future, this could support etcd, consul, etc.
        self.build_memory()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_core::Labels;

    fn create_test_config() -> NetworkConfig {
        NetworkConfig::default()
            .with_connect_timeout(std::time::Duration::from_millis(5000))
            .with_max_connections_per_host(100)
    }

    fn create_test_endpoint(service_name: &str, port: u16) -> ServiceEndpoint {
        ServiceEndpoint {
            service_name: service_name.to_string(),
            address: format!("127.0.0.1:{}", port).parse().unwrap(),
            labels: Labels::new("test", "v1", "test", "node1"),
            health_check_path: Some("/health".to_string()),
        }
    }

    #[tokio::test]
    async fn test_service_registration() {
        let discovery = MemoryServiceDiscovery::new(create_test_config());
        let node_id = NodeId::new("test-node");
        let endpoint = create_test_endpoint("test-service", 8080);
        
        // Register service
        discovery.register(&node_id, endpoint.clone()).await.unwrap();
        
        // Verify registration
        let services = discovery.get_node_services(&node_id).await.unwrap();
        assert_eq!(services.len(), 1);
        assert_eq!(services[0].service_name, "test-service");
    }

    #[tokio::test]
    async fn test_service_discovery() {
        let discovery = MemoryServiceDiscovery::new(create_test_config());
        let node1 = NodeId::new("node1");
        let node2 = NodeId::new("node2");
        
        let endpoint1 = create_test_endpoint("api", 8080);
        let endpoint2 = create_test_endpoint("api", 8081);
        
        // Register services
        discovery.register(&node1, endpoint1).await.unwrap();
        discovery.register(&node2, endpoint2).await.unwrap();
        
        // Discover services
        let results = discovery.discover("api").await.unwrap();
        assert_eq!(results.len(), 2);
        
        let node_ids: Vec<_> = results.iter().map(|(id, _)| id.clone()).collect();
        assert!(node_ids.contains(&node1));
        assert!(node_ids.contains(&node2));
    }

    #[tokio::test]
    async fn test_service_unregistration() {
        let discovery = MemoryServiceDiscovery::new(create_test_config());
        let node_id = NodeId::new("test-node");
        let endpoint = create_test_endpoint("test-service", 8080);
        
        // Register and then unregister
        discovery.register(&node_id, endpoint).await.unwrap();
        discovery.unregister(&node_id, "test-service").await.unwrap();
        
        // Verify unregistration
        let services = discovery.get_node_services(&node_id).await.unwrap();
        assert!(services.is_empty());
        
        let results = discovery.discover("test-service").await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_service_stats() {
        let discovery = MemoryServiceDiscovery::new(create_test_config());
        let node1 = NodeId::new("node1");
        let node2 = NodeId::new("node2");
        
        let endpoint1 = create_test_endpoint("api", 8080);
        let endpoint2 = create_test_endpoint("web", 8081);
        let endpoint3 = create_test_endpoint("api", 8082);
        
        // Register services
        discovery.register(&node1, endpoint1).await.unwrap();
        discovery.register(&node1, endpoint2).await.unwrap();
        discovery.register(&node2, endpoint3).await.unwrap();
        
        // Check stats
        let stats = discovery.get_stats().await;
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.total_services, 3);
        assert_eq!(stats.services_by_type.get("api"), Some(&2));
        assert_eq!(stats.services_by_type.get("web"), Some(&1));
    }

    #[tokio::test]
    async fn test_builder() {
        let config = create_test_config();
        let discovery = ServiceDiscoveryBuilder::new(config).build();
        
        let node_id = NodeId::new("test-node");
        let endpoint = create_test_endpoint("test-service", 8080);
        
        discovery.register(&node_id, endpoint).await.unwrap();
        let services = discovery.get_node_services(&node_id).await.unwrap();
        assert_eq!(services.len(), 1);
    }
}
