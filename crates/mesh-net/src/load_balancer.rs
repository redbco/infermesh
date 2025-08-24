//! Load balancing strategies for service endpoints

use anyhow::Result;
use mesh_core::{NodeId, ServiceEndpoint};
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::debug;

/// Load balancing strategy
#[derive(Debug, Clone, PartialEq)]
pub enum LoadBalancingStrategy {
    /// Round-robin selection
    RoundRobin,
    
    /// Random selection
    Random,
    
    /// Weighted round-robin based on endpoint weights
    WeightedRoundRobin,
    
    /// Least connections (requires connection tracking)
    LeastConnections,
    
    /// Consistent hashing based on request key
    ConsistentHash,
}

/// Load balancer trait
pub trait LoadBalancer: Send + Sync {
    /// Select an endpoint from available options
    fn select(&self, endpoints: &[(NodeId, ServiceEndpoint)], request_key: Option<&str>) -> Option<(NodeId, ServiceEndpoint)>;
    
    /// Update endpoint weights (if supported)
    fn update_weights(&self, weights: HashMap<NodeId, f64>);
    
    /// Record connection start (for connection tracking)
    fn record_connection_start(&self, node_id: &NodeId);
    
    /// Record connection end (for connection tracking)
    fn record_connection_end(&self, node_id: &NodeId);
    
    /// Get load balancer statistics
    fn get_stats(&self) -> LoadBalancerStats;
}

/// Round-robin load balancer
pub struct RoundRobinLoadBalancer {
    counter: AtomicUsize,
    stats: Arc<LoadBalancerStatsInner>,
}

/// Random load balancer
pub struct RandomLoadBalancer {
    stats: Arc<LoadBalancerStatsInner>,
}

/// Weighted round-robin load balancer
pub struct WeightedRoundRobinLoadBalancer {
    counter: AtomicUsize,
    weights: parking_lot::RwLock<HashMap<NodeId, f64>>,
    stats: Arc<LoadBalancerStatsInner>,
}

/// Least connections load balancer
pub struct LeastConnectionsLoadBalancer {
    connections: parking_lot::RwLock<HashMap<NodeId, usize>>,
    stats: Arc<LoadBalancerStatsInner>,
}

/// Consistent hash load balancer
pub struct ConsistentHashLoadBalancer {
    stats: Arc<LoadBalancerStatsInner>,
}

/// Load balancer statistics
#[derive(Debug, Clone)]
pub struct LoadBalancerStats {
    /// Total number of selections made
    pub total_selections: usize,
    
    /// Selections per node
    pub selections_per_node: HashMap<NodeId, usize>,
    
    /// Current connections per node (for connection tracking)
    pub connections_per_node: HashMap<NodeId, usize>,
    
    /// Average response time per node (if tracked)
    pub avg_response_time_ms: HashMap<NodeId, f64>,
}

#[derive(Debug, Default)]
struct LoadBalancerStatsInner {
    total_selections: AtomicUsize,
    selections_per_node: parking_lot::RwLock<HashMap<NodeId, usize>>,
    connections_per_node: parking_lot::RwLock<HashMap<NodeId, usize>>,
}

impl RoundRobinLoadBalancer {
    /// Create a new round-robin load balancer
    pub fn new() -> Self {
        Self {
            counter: AtomicUsize::new(0),
            stats: Arc::new(LoadBalancerStatsInner::default()),
        }
    }
}

impl LoadBalancer for RoundRobinLoadBalancer {
    fn select(&self, endpoints: &[(NodeId, ServiceEndpoint)], _request_key: Option<&str>) -> Option<(NodeId, ServiceEndpoint)> {
        if endpoints.is_empty() {
            return None;
        }
        
        let index = self.counter.fetch_add(1, Ordering::Relaxed) % endpoints.len();
        let selected = endpoints[index].clone();
        
        // Update stats
        self.stats.total_selections.fetch_add(1, Ordering::Relaxed);
        let mut selections = self.stats.selections_per_node.write();
        *selections.entry(selected.0.clone()).or_insert(0) += 1;
        
        debug!(
            node_id = %selected.0,
            service = %selected.1.service_name,
            index = index,
            "Selected endpoint (round-robin)"
        );
        
        Some(selected)
    }
    
    fn update_weights(&self, _weights: HashMap<NodeId, f64>) {
        // Round-robin doesn't use weights
    }
    
    fn record_connection_start(&self, node_id: &NodeId) {
        let mut connections = self.stats.connections_per_node.write();
        *connections.entry(node_id.clone()).or_insert(0) += 1;
    }
    
    fn record_connection_end(&self, node_id: &NodeId) {
        let mut connections = self.stats.connections_per_node.write();
        if let Some(count) = connections.get_mut(node_id) {
            if *count > 0 {
                *count -= 1;
            }
        }
    }
    
    fn get_stats(&self) -> LoadBalancerStats {
        LoadBalancerStats {
            total_selections: self.stats.total_selections.load(Ordering::Relaxed),
            selections_per_node: self.stats.selections_per_node.read().clone(),
            connections_per_node: self.stats.connections_per_node.read().clone(),
            avg_response_time_ms: HashMap::new(),
        }
    }
}

impl RandomLoadBalancer {
    /// Create a new random load balancer
    pub fn new() -> Self {
        Self {
            stats: Arc::new(LoadBalancerStatsInner::default()),
        }
    }
}

impl LoadBalancer for RandomLoadBalancer {
    fn select(&self, endpoints: &[(NodeId, ServiceEndpoint)], _request_key: Option<&str>) -> Option<(NodeId, ServiceEndpoint)> {
        if endpoints.is_empty() {
            return None;
        }
        
        let selected = endpoints.choose(&mut rand::thread_rng())?.clone();
        
        // Update stats
        self.stats.total_selections.fetch_add(1, Ordering::Relaxed);
        let mut selections = self.stats.selections_per_node.write();
        *selections.entry(selected.0.clone()).or_insert(0) += 1;
        
        debug!(
            node_id = %selected.0,
            service = %selected.1.service_name,
            "Selected endpoint (random)"
        );
        
        Some(selected)
    }
    
    fn update_weights(&self, _weights: HashMap<NodeId, f64>) {
        // Random doesn't use weights
    }
    
    fn record_connection_start(&self, node_id: &NodeId) {
        let mut connections = self.stats.connections_per_node.write();
        *connections.entry(node_id.clone()).or_insert(0) += 1;
    }
    
    fn record_connection_end(&self, node_id: &NodeId) {
        let mut connections = self.stats.connections_per_node.write();
        if let Some(count) = connections.get_mut(node_id) {
            if *count > 0 {
                *count -= 1;
            }
        }
    }
    
    fn get_stats(&self) -> LoadBalancerStats {
        LoadBalancerStats {
            total_selections: self.stats.total_selections.load(Ordering::Relaxed),
            selections_per_node: self.stats.selections_per_node.read().clone(),
            connections_per_node: self.stats.connections_per_node.read().clone(),
            avg_response_time_ms: HashMap::new(),
        }
    }
}

impl WeightedRoundRobinLoadBalancer {
    /// Create a new weighted round-robin load balancer
    pub fn new() -> Self {
        Self {
            counter: AtomicUsize::new(0),
            weights: parking_lot::RwLock::new(HashMap::new()),
            stats: Arc::new(LoadBalancerStatsInner::default()),
        }
    }
}

impl LoadBalancer for WeightedRoundRobinLoadBalancer {
    fn select(&self, endpoints: &[(NodeId, ServiceEndpoint)], _request_key: Option<&str>) -> Option<(NodeId, ServiceEndpoint)> {
        if endpoints.is_empty() {
            return None;
        }
        
        let weights = self.weights.read();
        
        // Build weighted list
        let mut weighted_endpoints = Vec::new();
        for (node_id, endpoint) in endpoints {
            let weight = weights.get(node_id).copied().unwrap_or(1.0);
            let weight_count = (weight * 10.0) as usize; // Scale weights
            for _ in 0..weight_count.max(1) {
                weighted_endpoints.push((node_id.clone(), endpoint.clone()));
            }
        }
        
        if weighted_endpoints.is_empty() {
            return None;
        }
        
        let index = self.counter.fetch_add(1, Ordering::Relaxed) % weighted_endpoints.len();
        let selected = weighted_endpoints[index].clone();
        
        // Update stats
        self.stats.total_selections.fetch_add(1, Ordering::Relaxed);
        let mut selections = self.stats.selections_per_node.write();
        *selections.entry(selected.0.clone()).or_insert(0) += 1;
        
        debug!(
            node_id = %selected.0,
            service = %selected.1.service_name,
            weight = weights.get(&selected.0).copied().unwrap_or(1.0),
            "Selected endpoint (weighted round-robin)"
        );
        
        Some(selected)
    }
    
    fn update_weights(&self, weights: HashMap<NodeId, f64>) {
        let mut current_weights = self.weights.write();
        *current_weights = weights;
    }
    
    fn record_connection_start(&self, node_id: &NodeId) {
        let mut connections = self.stats.connections_per_node.write();
        *connections.entry(node_id.clone()).or_insert(0) += 1;
    }
    
    fn record_connection_end(&self, node_id: &NodeId) {
        let mut connections = self.stats.connections_per_node.write();
        if let Some(count) = connections.get_mut(node_id) {
            if *count > 0 {
                *count -= 1;
            }
        }
    }
    
    fn get_stats(&self) -> LoadBalancerStats {
        LoadBalancerStats {
            total_selections: self.stats.total_selections.load(Ordering::Relaxed),
            selections_per_node: self.stats.selections_per_node.read().clone(),
            connections_per_node: self.stats.connections_per_node.read().clone(),
            avg_response_time_ms: HashMap::new(),
        }
    }
}

impl LeastConnectionsLoadBalancer {
    /// Create a new least connections load balancer
    pub fn new() -> Self {
        Self {
            connections: parking_lot::RwLock::new(HashMap::new()),
            stats: Arc::new(LoadBalancerStatsInner::default()),
        }
    }
}

impl LoadBalancer for LeastConnectionsLoadBalancer {
    fn select(&self, endpoints: &[(NodeId, ServiceEndpoint)], _request_key: Option<&str>) -> Option<(NodeId, ServiceEndpoint)> {
        if endpoints.is_empty() {
            return None;
        }
        
        let connections = self.connections.read();
        
        // Find endpoint with least connections
        let mut min_connections = usize::MAX;
        let mut selected = None;
        
        for (node_id, endpoint) in endpoints {
            let conn_count = connections.get(node_id).copied().unwrap_or(0);
            if conn_count < min_connections {
                min_connections = conn_count;
                selected = Some((node_id.clone(), endpoint.clone()));
            }
        }
        
        if let Some(ref selected_endpoint) = selected {
            // Update stats
            self.stats.total_selections.fetch_add(1, Ordering::Relaxed);
            let mut selections = self.stats.selections_per_node.write();
            *selections.entry(selected_endpoint.0.clone()).or_insert(0) += 1;
            
            debug!(
                node_id = %selected_endpoint.0,
                service = %selected_endpoint.1.service_name,
                connections = min_connections,
                "Selected endpoint (least connections)"
            );
        }
        
        selected
    }
    
    fn update_weights(&self, _weights: HashMap<NodeId, f64>) {
        // Least connections doesn't use weights
    }
    
    fn record_connection_start(&self, node_id: &NodeId) {
        let mut connections = self.connections.write();
        *connections.entry(node_id.clone()).or_insert(0) += 1;
        
        let mut stats_connections = self.stats.connections_per_node.write();
        *stats_connections.entry(node_id.clone()).or_insert(0) += 1;
    }
    
    fn record_connection_end(&self, node_id: &NodeId) {
        let mut connections = self.connections.write();
        if let Some(count) = connections.get_mut(node_id) {
            if *count > 0 {
                *count -= 1;
            }
        }
        
        let mut stats_connections = self.stats.connections_per_node.write();
        if let Some(count) = stats_connections.get_mut(node_id) {
            if *count > 0 {
                *count -= 1;
            }
        }
    }
    
    fn get_stats(&self) -> LoadBalancerStats {
        LoadBalancerStats {
            total_selections: self.stats.total_selections.load(Ordering::Relaxed),
            selections_per_node: self.stats.selections_per_node.read().clone(),
            connections_per_node: self.stats.connections_per_node.read().clone(),
            avg_response_time_ms: HashMap::new(),
        }
    }
}

impl ConsistentHashLoadBalancer {
    /// Create a new consistent hash load balancer
    pub fn new() -> Self {
        Self {
            stats: Arc::new(LoadBalancerStatsInner::default()),
        }
    }
    
    /// Hash a string to a u64
    fn hash_key(&self, key: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}

impl LoadBalancer for ConsistentHashLoadBalancer {
    fn select(&self, endpoints: &[(NodeId, ServiceEndpoint)], request_key: Option<&str>) -> Option<(NodeId, ServiceEndpoint)> {
        if endpoints.is_empty() {
            return None;
        }
        
        let key = request_key.unwrap_or("default");
        let hash = self.hash_key(key);
        let index = (hash as usize) % endpoints.len();
        let selected = endpoints[index].clone();
        
        // Update stats
        self.stats.total_selections.fetch_add(1, Ordering::Relaxed);
        let mut selections = self.stats.selections_per_node.write();
        *selections.entry(selected.0.clone()).or_insert(0) += 1;
        
        debug!(
            node_id = %selected.0,
            service = %selected.1.service_name,
            key = key,
            hash = hash,
            index = index,
            "Selected endpoint (consistent hash)"
        );
        
        Some(selected)
    }
    
    fn update_weights(&self, _weights: HashMap<NodeId, f64>) {
        // Consistent hash doesn't use weights
    }
    
    fn record_connection_start(&self, node_id: &NodeId) {
        let mut connections = self.stats.connections_per_node.write();
        *connections.entry(node_id.clone()).or_insert(0) += 1;
    }
    
    fn record_connection_end(&self, node_id: &NodeId) {
        let mut connections = self.stats.connections_per_node.write();
        if let Some(count) = connections.get_mut(node_id) {
            if *count > 0 {
                *count -= 1;
            }
        }
    }
    
    fn get_stats(&self) -> LoadBalancerStats {
        LoadBalancerStats {
            total_selections: self.stats.total_selections.load(Ordering::Relaxed),
            selections_per_node: self.stats.selections_per_node.read().clone(),
            connections_per_node: self.stats.connections_per_node.read().clone(),
            avg_response_time_ms: HashMap::new(),
        }
    }
}

/// Load balancer factory
pub struct LoadBalancerFactory;

impl LoadBalancerFactory {
    /// Create a load balancer based on strategy
    pub fn create(strategy: LoadBalancingStrategy) -> Result<Box<dyn LoadBalancer>> {
        match strategy {
            LoadBalancingStrategy::RoundRobin => Ok(Box::new(RoundRobinLoadBalancer::new())),
            LoadBalancingStrategy::Random => Ok(Box::new(RandomLoadBalancer::new())),
            LoadBalancingStrategy::WeightedRoundRobin => Ok(Box::new(WeightedRoundRobinLoadBalancer::new())),
            LoadBalancingStrategy::LeastConnections => Ok(Box::new(LeastConnectionsLoadBalancer::new())),
            LoadBalancingStrategy::ConsistentHash => Ok(Box::new(ConsistentHashLoadBalancer::new())),
        }
    }
}

impl Default for RoundRobinLoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for RandomLoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for WeightedRoundRobinLoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for LeastConnectionsLoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ConsistentHashLoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_core::Labels;

    fn create_test_endpoints() -> Vec<(NodeId, ServiceEndpoint)> {
        vec![
            (
                NodeId::new("node1"),
                ServiceEndpoint {
                    service_name: "api".to_string(),
                    address: "127.0.0.1:8080".parse().unwrap(),
                    labels: Labels::new("api", "v1", "test", "node1"),
                    health_check_path: None,
                }
            ),
            (
                NodeId::new("node2"),
                ServiceEndpoint {
                    service_name: "api".to_string(),
                    address: "127.0.0.1:8081".parse().unwrap(),
                    labels: Labels::new("api", "v1", "test", "node2"),
                    health_check_path: None,
                }
            ),
            (
                NodeId::new("node3"),
                ServiceEndpoint {
                    service_name: "api".to_string(),
                    address: "127.0.0.1:8082".parse().unwrap(),
                    labels: Labels::new("api", "v1", "test", "node3"),
                    health_check_path: None,
                }
            ),
        ]
    }

    #[test]
    fn test_round_robin_load_balancer() {
        let lb = RoundRobinLoadBalancer::new();
        let endpoints = create_test_endpoints();
        
        // Test selection order
        let selections: Vec<_> = (0..6)
            .map(|_| lb.select(&endpoints, None).unwrap().0)
            .collect();
        
        // Should cycle through endpoints
        assert_eq!(selections[0], NodeId::new("node1"));
        assert_eq!(selections[1], NodeId::new("node2"));
        assert_eq!(selections[2], NodeId::new("node3"));
        assert_eq!(selections[3], NodeId::new("node1"));
        assert_eq!(selections[4], NodeId::new("node2"));
        assert_eq!(selections[5], NodeId::new("node3"));
        
        // Check stats
        let stats = lb.get_stats();
        assert_eq!(stats.total_selections, 6);
        assert_eq!(stats.selections_per_node.len(), 3);
    }

    #[test]
    fn test_random_load_balancer() {
        let lb = RandomLoadBalancer::new();
        let endpoints = create_test_endpoints();
        
        // Test multiple selections
        let mut selections = Vec::new();
        for _ in 0..100 {
            if let Some((node_id, _)) = lb.select(&endpoints, None) {
                selections.push(node_id);
            }
        }
        
        // Should have selections from all nodes (with high probability)
        let unique_selections: std::collections::HashSet<_> = selections.into_iter().collect();
        assert!(unique_selections.len() >= 2); // At least 2 different nodes selected
        
        // Check stats
        let stats = lb.get_stats();
        assert_eq!(stats.total_selections, 100);
    }

    #[test]
    fn test_weighted_round_robin_load_balancer() {
        let lb = WeightedRoundRobinLoadBalancer::new();
        let endpoints = create_test_endpoints();
        
        // Set weights
        let mut weights = HashMap::new();
        weights.insert(NodeId::new("node1"), 2.0);
        weights.insert(NodeId::new("node2"), 1.0);
        weights.insert(NodeId::new("node3"), 1.0);
        lb.update_weights(weights);
        
        // Test selections
        let mut selections = Vec::new();
        for _ in 0..40 {
            if let Some((node_id, _)) = lb.select(&endpoints, None) {
                selections.push(node_id);
            }
        }
        
        // Count selections per node
        let mut counts = HashMap::new();
        for node_id in selections {
            *counts.entry(node_id).or_insert(0) += 1;
        }
        
        // node1 should have roughly twice as many selections
        let node1_count = counts.get(&NodeId::new("node1")).copied().unwrap_or(0);
        let node2_count = counts.get(&NodeId::new("node2")).copied().unwrap_or(0);
        let node3_count = counts.get(&NodeId::new("node3")).copied().unwrap_or(0);
        
        assert!(node1_count > node2_count);
        assert!(node1_count > node3_count);
    }

    #[test]
    fn test_least_connections_load_balancer() {
        let lb = LeastConnectionsLoadBalancer::new();
        let endpoints = create_test_endpoints();
        
        let node1 = NodeId::new("node1");
        let node2 = NodeId::new("node2");
        let node3 = NodeId::new("node3");
        
        // Start connections on node1 and node2
        lb.record_connection_start(&node1);
        lb.record_connection_start(&node1);
        lb.record_connection_start(&node2);
        
        // Next selection should be node3 (0 connections)
        let (selected_node, _) = lb.select(&endpoints, None).unwrap();
        assert_eq!(selected_node, node3);
        
        // End a connection on node1
        lb.record_connection_end(&node1);
        
        // Next selection should be node1 or node3 (both have 1 or 0 connections)
        let (selected_node, _) = lb.select(&endpoints, None).unwrap();
        assert!(selected_node == node1 || selected_node == node3);
    }

    #[test]
    fn test_consistent_hash_load_balancer() {
        let lb = ConsistentHashLoadBalancer::new();
        let endpoints = create_test_endpoints();
        
        // Same key should always select same endpoint
        let key = "test-key";
        let (node1, _) = lb.select(&endpoints, Some(key)).unwrap();
        let (node2, _) = lb.select(&endpoints, Some(key)).unwrap();
        let (node3, _) = lb.select(&endpoints, Some(key)).unwrap();
        
        assert_eq!(node1, node2);
        assert_eq!(node2, node3);
        
        // Different keys might select different endpoints
        let (_other_node, _) = lb.select(&endpoints, Some("other-key")).unwrap();
        // Note: This might be the same node due to hash collision, but that's OK
        
        // Check stats
        let stats = lb.get_stats();
        assert_eq!(stats.total_selections, 4);
    }

    #[test]
    fn test_load_balancer_factory() {
        let strategies = vec![
            LoadBalancingStrategy::RoundRobin,
            LoadBalancingStrategy::Random,
            LoadBalancingStrategy::WeightedRoundRobin,
            LoadBalancingStrategy::LeastConnections,
            LoadBalancingStrategy::ConsistentHash,
        ];
        
        for strategy in strategies {
            let lb = LoadBalancerFactory::create(strategy).unwrap();
            let endpoints = create_test_endpoints();
            
            // Should be able to select an endpoint
            let result = lb.select(&endpoints, None);
            assert!(result.is_some());
        }
    }

    #[test]
    fn test_empty_endpoints() {
        let lb = RoundRobinLoadBalancer::new();
        let endpoints = vec![];
        
        let result = lb.select(&endpoints, None);
        assert!(result.is_none());
    }
}
