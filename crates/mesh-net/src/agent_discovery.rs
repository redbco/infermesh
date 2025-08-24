//! Agent-based service discovery that connects to mesh-agent control plane

use crate::discovery::{ServiceDiscovery, ServiceDiscoveryStats};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use mesh_core::{NodeId, ServiceEndpoint};
use mesh_proto::control::v1::{
    control_plane_client::ControlPlaneClient, ListNodesRequest
};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::transport::Channel;
use tracing::{debug, info, warn};

/// Service discovery that connects to mesh-agent control plane
pub struct AgentServiceDiscovery {
    /// gRPC client to mesh-agent control plane
    client: Arc<RwLock<Option<ControlPlaneClient<Channel>>>>,
    
    /// Agent address
    agent_address: SocketAddr,
    
    /// Cached services (for performance)
    cache: Arc<RwLock<HashMap<NodeId, HashMap<String, ServiceEndpoint>>>>,
    
    /// Cache TTL
    cache_ttl: std::time::Duration,
    
    /// Last cache update
    last_cache_update: Arc<RwLock<std::time::Instant>>,
}

impl AgentServiceDiscovery {
    /// Create a new agent service discovery
    pub fn new(agent_address: SocketAddr) -> Self {
        Self {
            client: Arc::new(RwLock::new(None)),
            agent_address,
            cache: Arc::new(RwLock::new(HashMap::new())),
            cache_ttl: std::time::Duration::from_secs(30), // 30 second cache
            last_cache_update: Arc::new(RwLock::new(std::time::Instant::now() - std::time::Duration::from_secs(60))),
        }
    }
    
    /// Connect to the mesh-agent control plane
    async fn ensure_connected(&self) -> Result<()> {
        let mut client_guard = self.client.write().await;
        
        if client_guard.is_none() {
            let endpoint = format!("http://{}", self.agent_address);
            debug!("Connecting to mesh-agent at {}", endpoint);
            
            let channel = tonic::transport::Channel::from_shared(endpoint)?
                .connect()
                .await?;
                
            let client = ControlPlaneClient::new(channel);
            *client_guard = Some(client);
            
            info!("Connected to mesh-agent control plane at {}", self.agent_address);
        }
        
        Ok(())
    }
    
    /// Update the service cache from mesh-agent
    async fn update_cache(&self) -> Result<()> {
        let now = std::time::Instant::now();
        let last_update = *self.last_cache_update.read().await;
        
        // Check if cache is still valid
        if now.duration_since(last_update) < self.cache_ttl {
            return Ok(());
        }
        
        self.ensure_connected().await?;
        
        let client_guard = self.client.read().await;
        let mut client = client_guard.as_ref()
            .ok_or_else(|| anyhow!("Not connected to mesh-agent"))?
            .clone();
        drop(client_guard);
        
        // Get list of nodes from control plane
        let request = tonic::Request::new(ListNodesRequest {
            role_filter: vec![],
            zone_filter: String::new(),
            label_filter: HashMap::new(),
        });
        let response = client.list_nodes(request).await
            .map_err(|e| anyhow!("Failed to list nodes: {}", e))?;
            
        let nodes = response.into_inner().nodes;
        
        // Update cache with node information
        let mut cache = self.cache.write().await;
        cache.clear();
        
        for node in nodes {
            let node_id: NodeId = node.id.clone().into();
            let mut node_services = HashMap::new();
            
            // Create service endpoints based on node roles and configuration
            // For now, we'll create standard endpoints based on known services
            
            // Control plane service (always available)
            let control_labels = mesh_core::Labels::new(
                "control-plane", 
                "v1", 
                "mesh-agent", 
                &node.id
            );
            node_services.insert(
                "control-plane".to_string(),
                ServiceEndpoint::with_health_check(
                    "control-plane",
                    self.agent_address,
                    control_labels.clone(),
                    "/health"
                )
            );
            
            // State plane service
            node_services.insert(
                "state-plane".to_string(),
                ServiceEndpoint::with_health_check(
                    "state-plane",
                    self.agent_address,
                    control_labels.clone(),
                    "/health"
                )
            );
            
            // Inference service (if node has GPU role)
            if node.roles.contains(&(mesh_proto::control::v1::NodeRole::Gpu as i32)) {
                // Use a different port for inference service (8080 for HTTP, 8081 for gRPC)
                let inference_addr = SocketAddr::new(self.agent_address.ip(), 8080);
                node_services.insert(
                    "inference".to_string(),
                    ServiceEndpoint::with_health_check(
                        "inference",
                        inference_addr,
                        control_labels.clone(),
                        "/health"
                    )
                );
                
                let inference_grpc_addr = SocketAddr::new(self.agent_address.ip(), 8081);
                node_services.insert(
                    "inference-grpc".to_string(),
                    ServiceEndpoint::with_health_check(
                        "inference-grpc",
                        inference_grpc_addr,
                        control_labels,
                        "/health"
                    )
                );
            }
            
            cache.insert(node_id, node_services);
        }
        
        *self.last_cache_update.write().await = now;
        
        debug!("Updated service discovery cache with {} nodes", cache.len());
        Ok(())
    }
    
    /// Get service statistics
    pub async fn get_stats(&self) -> Result<ServiceDiscoveryStats> {
        self.update_cache().await?;
        
        let cache = self.cache.read().await;
        let total_nodes = cache.len();
        let mut total_services = 0;
        let mut services_by_type = HashMap::new();
        
        for node_services in cache.values() {
            total_services += node_services.len();
            for (service_name, _) in node_services {
                *services_by_type.entry(service_name.clone()).or_insert(0) += 1;
            }
        }
        
        Ok(ServiceDiscoveryStats {
            total_nodes,
            total_services,
            services_by_type,
        })
    }
}

#[async_trait]
impl ServiceDiscovery for AgentServiceDiscovery {
    async fn register(&self, node_id: &NodeId, endpoint: ServiceEndpoint) -> Result<()> {
        // For agent-based discovery, registration is handled by the mesh-agent itself
        // This is a no-op since nodes register themselves with the control plane
        debug!(
            node_id = %node_id,
            service_name = %endpoint.service_name,
            "Service registration handled by mesh-agent control plane"
        );
        Ok(())
    }
    
    async fn unregister(&self, node_id: &NodeId, service_name: &str) -> Result<()> {
        // For agent-based discovery, unregistration is handled by the mesh-agent itself
        debug!(
            node_id = %node_id,
            service_name = %service_name,
            "Service unregistration handled by mesh-agent control plane"
        );
        Ok(())
    }
    
    async fn discover(&self, service_name: &str) -> Result<Vec<(NodeId, ServiceEndpoint)>> {
        self.update_cache().await?;
        
        let cache = self.cache.read().await;
        let mut results = Vec::new();
        
        for (node_id, node_services) in cache.iter() {
            if let Some(endpoint) = node_services.get(service_name) {
                results.push((node_id.clone(), endpoint.clone()));
            }
        }
        
        debug!(
            service_name = %service_name,
            count = results.len(),
            "Discovered service endpoints from mesh-agent"
        );
        
        Ok(results)
    }
    
    async fn get_node_services(&self, node_id: &NodeId) -> Result<Vec<ServiceEndpoint>> {
        self.update_cache().await?;
        
        let cache = self.cache.read().await;
        if let Some(node_services) = cache.get(node_id) {
            Ok(node_services.values().cloned().collect())
        } else {
            Ok(Vec::new())
        }
    }
    
    async fn get_service_nodes(&self, service_name: &str) -> Result<Vec<NodeId>> {
        self.update_cache().await?;
        
        let cache = self.cache.read().await;
        let mut nodes = Vec::new();
        
        for (node_id, node_services) in cache.iter() {
            if node_services.contains_key(service_name) {
                nodes.push(node_id.clone());
            }
        }
        
        Ok(nodes)
    }
    
    async fn health_check(&self, node_id: &NodeId, service_name: &str) -> Result<bool> {
        self.update_cache().await?;
        
        let cache = self.cache.read().await;
        if let Some(node_services) = cache.get(node_id) {
            if let Some(endpoint) = node_services.get(service_name) {
                // Perform actual health check
                if let Some(health_path) = &endpoint.health_check_path {
                    let health_url = format!("http://{}{}", 
                        endpoint.address, 
                        health_path
                    );
                    
                    // Simple HTTP health check
                    match reqwest::get(&health_url).await {
                        Ok(response) => {
                            let is_healthy = response.status().is_success();
                            debug!(
                                node_id = %node_id,
                                service_name = %service_name,
                                health_url = %health_url,
                                healthy = is_healthy,
                                "Health check completed"
                            );
                            Ok(is_healthy)
                        }
                        Err(e) => {
                            warn!(
                                node_id = %node_id,
                                service_name = %service_name,
                                error = %e,
                                "Health check failed"
                            );
                            Ok(false)
                        }
                    }
                } else {
                    // No health check path, assume healthy if service exists
                    Ok(true)
                }
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};
    
    #[tokio::test]
    async fn test_agent_service_discovery_creation() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 50051);
        let discovery = AgentServiceDiscovery::new(addr);
        
        assert_eq!(discovery.agent_address, addr);
        assert_eq!(discovery.cache_ttl, std::time::Duration::from_secs(30));
    }
    
    #[tokio::test]
    async fn test_service_registration_noop() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 50051);
        let discovery = AgentServiceDiscovery::new(addr);
        
        let node_id: NodeId = "test-node".into();
        let endpoint = ServiceEndpoint::with_health_check(
            "test-service",
            addr,
            mesh_core::Labels::new("test", "v1", "test", "test-node"),
            "/health"
        );
        
        // Should not fail (no-op)
        let result = discovery.register(&node_id, endpoint).await;
        assert!(result.is_ok());
    }
}
