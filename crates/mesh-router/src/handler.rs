//! Request handling and routing logic

use crate::config::{RouterConfig, LoadBalancingStrategy};
use crate::{Result, RouterError};

use mesh_core::{Labels, NodeId};
use mesh_net::{ConnectionPool, ServiceDiscovery};
use mesh_state::{StateStore, QueryEngine, ScoringEngine};

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Request context for tracking and tracing
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Unique request ID
    pub request_id: String,
    
    /// Client IP address
    pub client_ip: Option<String>,
    
    /// User agent
    pub user_agent: Option<String>,
    
    /// Request headers
    pub headers: HashMap<String, String>,
    
    /// Request timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl RequestContext {
    /// Create a new request context
    pub fn new() -> Self {
        Self {
            request_id: Uuid::new_v4().to_string(),
            client_ip: None,
            user_agent: None,
            headers: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Set client IP
    pub fn with_client_ip(mut self, ip: impl Into<String>) -> Self {
        self.client_ip = Some(ip.into());
        self
    }

    /// Set user agent
    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = Some(ua.into());
        self
    }

    /// Add header
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }
}

impl Default for RequestContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Target for routing requests
#[derive(Debug, Clone)]
pub struct RoutingTarget {
    /// Node ID
    pub node_id: NodeId,
    
    /// Service endpoint address
    pub address: std::net::SocketAddr,
    
    /// Routing score (higher is better)
    pub score: f64,
    
    /// Labels for this target
    pub labels: Labels,
}

/// Request handler that manages routing and proxying
pub struct RequestHandler {
    /// Configuration
    config: RouterConfig,
    
    /// State store for querying node information
    state_store: StateStore,
    
    /// Query engine for finding suitable targets
    query_engine: QueryEngine,
    
    /// Scoring engine for ranking targets
    scoring_engine: ScoringEngine,
    
    /// Connection pool for upstream connections
    connection_pool: ConnectionPool,
    
    /// Service discovery
    service_discovery: Arc<dyn ServiceDiscovery>,
    
    /// Load balancer
    #[allow(dead_code)]
    load_balancer: Arc<dyn mesh_net::LoadBalancer>,
    
    /// Circuit breaker states
    circuit_breakers: RwLock<HashMap<NodeId, CircuitBreakerState>>,
}

/// Circuit breaker state
#[derive(Debug, Clone)]
struct CircuitBreakerState {
    /// Current state
    state: CircuitState,
    
    /// Failure count
    failure_count: u32,
    
    /// Success count (for half-open state)
    success_count: u32,
    
    /// Last failure time
    last_failure: Option<chrono::DateTime<chrono::Utc>>,
    
    /// Next attempt time (for open state)
    next_attempt: Option<chrono::DateTime<chrono::Utc>>,
}

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl RequestHandler {
    /// Create a new request handler
    pub async fn new(
        config: RouterConfig,
        state_store: StateStore,
        query_engine: QueryEngine,
        scoring_engine: ScoringEngine,
        connection_pool: ConnectionPool,
    ) -> Result<Self> {
        // Create service discovery that connects to mesh-agent
        let agent_address = config.agent_address;
        let service_discovery = Arc::new(mesh_net::AgentServiceDiscovery::new(agent_address));

        // Create load balancer
        let load_balancer = mesh_net::LoadBalancerFactory::create(
            convert_load_balancing_strategy(&config.routing.load_balancing)
        ).map_err(|e| RouterError::Configuration(format!("Failed to create load balancer: {}", e)))?;

        Ok(Self {
            config,
            state_store,
            query_engine,
            scoring_engine,
            connection_pool,
            service_discovery,
            load_balancer: Arc::from(load_balancer),
            circuit_breakers: RwLock::new(HashMap::new()),
        })
    }

    /// Route a request to the best available target
    pub async fn route_request(
        &self,
        context: &RequestContext,
        labels: &Labels,
    ) -> Result<RoutingTarget> {
        debug!("Routing request {} with labels: {:?}", context.request_id, labels);

        // Find potential targets using scoring
        let targets = if self.config.routing.enable_scoring {
            self.find_targets_by_score(labels).await?
        } else {
            self.find_targets_basic(labels).await?
        };

        if targets.is_empty() {
            warn!("No targets found for request {}", context.request_id);
            return Err(RouterError::ServiceUnavailable);
        }

        // Filter out targets with open circuit breakers
        let available_targets = self.filter_available_targets(targets).await;

        if available_targets.is_empty() {
            warn!("All targets unavailable due to circuit breakers for request {}", 
                  context.request_id);
            return Err(RouterError::ServiceUnavailable);
        }

        // Select target using load balancing strategy
        let target = self.select_target(available_targets).await?;

        info!("Routed request {} to target: {:?}", context.request_id, target.node_id);
        Ok(target)
    }

    /// Find targets using scoring engine
    async fn find_targets_by_score(&self, labels: &Labels) -> Result<Vec<RoutingTarget>> {
        let scored_targets = self.scoring_engine
            .score_targets(&self.state_store, labels, 100)
            .await
            .map_err(|e| RouterError::State(e))?;

        let mut targets = Vec::new();
        for scored_target in scored_targets {
            // Convert scored target to routing target
            if let Some(model_key) = &scored_target.model_key {
                if let Some(model_state) = self.state_store
                    .get_model_state(&scored_target.node_id, model_key) {
                
                // Get address from service discovery
                let address = match self.get_node_address(&scored_target.node_id).await {
                    Ok(addr) => addr,
                    Err(e) => {
                        warn!("Failed to get address for node {}: {}", scored_target.node_id, e);
                        continue;
                    }
                };
                
                    targets.push(RoutingTarget {
                        node_id: scored_target.node_id,
                        address,
                        score: scored_target.score,
                        labels: model_state.state.labels.clone(),
                    });
                }
            }
        }

        Ok(targets)
    }

    /// Find targets using basic query
    async fn find_targets_basic(&self, labels: &Labels) -> Result<Vec<RoutingTarget>> {
        let model_results = self.query_engine
            .find_ready_models(labels)
            .await
            .map_err(|e| RouterError::State(e))?;

        let mut targets = Vec::new();
        for result in model_results {
            // Get address from service discovery
            let address = match self.get_node_address(&result.node_id).await {
                Ok(addr) => addr,
                Err(e) => {
                    warn!("Failed to get address for node {}: {}", result.node_id, e);
                    continue;
                }
            };
            
            targets.push(RoutingTarget {
                node_id: result.node_id,
                address,
                score: 1.0, // Default score
                labels: result.labels,
            });
        }

        Ok(targets)
    }

    /// Filter targets based on circuit breaker state
    async fn filter_available_targets(&self, targets: Vec<RoutingTarget>) -> Vec<RoutingTarget> {
        let circuit_breakers = self.circuit_breakers.read().await;
        let mut available = Vec::new();

        for target in targets {
            if let Some(cb_state) = circuit_breakers.get(&target.node_id) {
                if self.is_target_available(cb_state) {
                    available.push(target);
                }
            } else {
                // No circuit breaker state means target is available
                available.push(target);
            }
        }

        available
    }

    /// Get node address from service discovery
    async fn get_node_address(&self, node_id: &NodeId) -> Result<std::net::SocketAddr> {
        // Try to get inference service first, fallback to control plane
        let services = self.service_discovery.get_node_services(node_id).await
            .map_err(|e| RouterError::ServiceDiscovery(format!("Failed to get services for node {}: {}", node_id, e)))?;
        
        // Prefer inference service for routing requests
        for service in &services {
            if service.service_name == "inference" || service.service_name == "inference-grpc" {
                return Ok(service.address);
            }
        }
        
        // Fallback to control plane service
        for service in &services {
            if service.service_name == "control-plane" {
                return Ok(service.address);
            }
        }
        
        Err(RouterError::ServiceDiscovery(format!("No suitable service found for node {}", node_id)))
    }

    /// Check if target is available based on circuit breaker state
    fn is_target_available(&self, cb_state: &CircuitBreakerState) -> bool {
        match cb_state.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if we should attempt to close the circuit
                if let Some(next_attempt) = cb_state.next_attempt {
                    chrono::Utc::now() >= next_attempt
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Select target using load balancing strategy
    async fn select_target(&self, targets: Vec<RoutingTarget>) -> Result<RoutingTarget> {
        match self.config.routing.load_balancing {
            LoadBalancingStrategy::ScoreBased => {
                // Select target with highest score
                targets.into_iter()
                    .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
                    .ok_or(RouterError::ServiceUnavailable)
            }
            LoadBalancingStrategy::Random => {
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                targets.choose(&mut rng)
                    .cloned()
                    .ok_or(RouterError::ServiceUnavailable)
            }
            LoadBalancingStrategy::RoundRobin => {
                // For simplicity, just return the first target
                // In a real implementation, this would maintain round-robin state
                targets.into_iter().next()
                    .ok_or(RouterError::ServiceUnavailable)
            }
            _ => {
                // Fallback to first available target
                targets.into_iter().next()
                    .ok_or(RouterError::ServiceUnavailable)
            }
        }
    }

    /// Record successful request for circuit breaker
    pub async fn record_success(&self, node_id: &NodeId) {
        let mut circuit_breakers = self.circuit_breakers.write().await;
        
        if let Some(cb_state) = circuit_breakers.get_mut(node_id) {
            match cb_state.state {
                CircuitState::HalfOpen => {
                    cb_state.success_count += 1;
                    if cb_state.success_count >= self.config.routing.circuit_breaker.success_threshold {
                        cb_state.state = CircuitState::Closed;
                        cb_state.failure_count = 0;
                        cb_state.success_count = 0;
                        info!("Circuit breaker closed for node: {:?}", node_id);
                    }
                }
                CircuitState::Open => {
                    // Transition to half-open
                    cb_state.state = CircuitState::HalfOpen;
                    cb_state.success_count = 1;
                    info!("Circuit breaker half-open for node: {:?}", node_id);
                }
                CircuitState::Closed => {
                    // Reset failure count on success
                    cb_state.failure_count = 0;
                }
            }
        }
    }

    /// Record failed request for circuit breaker
    pub async fn record_failure(&self, node_id: &NodeId) {
        let mut circuit_breakers = self.circuit_breakers.write().await;
        
        let cb_state = circuit_breakers.entry(node_id.clone()).or_insert_with(|| {
            CircuitBreakerState {
                state: CircuitState::Closed,
                failure_count: 0,
                success_count: 0,
                last_failure: None,
                next_attempt: None,
            }
        });

        cb_state.failure_count += 1;
        cb_state.last_failure = Some(chrono::Utc::now());

        match cb_state.state {
            CircuitState::Closed => {
                if cb_state.failure_count >= self.config.routing.circuit_breaker.failure_threshold {
                    cb_state.state = CircuitState::Open;
                    cb_state.next_attempt = Some(
                        chrono::Utc::now() + chrono::Duration::from_std(
                            self.config.routing.circuit_breaker.timeout
                        ).unwrap_or_else(|_| chrono::Duration::seconds(60))
                    );
                    warn!("Circuit breaker opened for node: {:?}", node_id);
                }
            }
            CircuitState::HalfOpen => {
                // Go back to open state
                cb_state.state = CircuitState::Open;
                cb_state.success_count = 0;
                cb_state.next_attempt = Some(
                    chrono::Utc::now() + chrono::Duration::from_std(
                        self.config.routing.circuit_breaker.timeout
                    ).unwrap_or_else(|_| chrono::Duration::seconds(60))
                );
                warn!("Circuit breaker re-opened for node: {:?}", node_id);
            }
            CircuitState::Open => {
                // Update next attempt time
                cb_state.next_attempt = Some(
                    chrono::Utc::now() + chrono::Duration::from_std(
                        self.config.routing.circuit_breaker.timeout
                    ).unwrap_or_else(|_| chrono::Duration::seconds(60))
                );
            }
        }
    }

    /// Health check for the handler
    pub async fn health_check(&self) -> Result<bool> {
        // Check if we can query the state store
        let model_count = self.state_store.model_count();
        debug!("Health check: {} models in state store", model_count);
        
        // Handler is healthy if it can access its dependencies
        Ok(true)
    }

    /// Get connection pool
    pub fn connection_pool(&self) -> &ConnectionPool {
        &self.connection_pool
    }

    /// Get service discovery
    pub fn service_discovery(&self) -> &Arc<dyn ServiceDiscovery> {
        &self.service_discovery
    }
}

// Helper function to convert config load balancing strategy to mesh-net strategy
fn convert_load_balancing_strategy(strategy: &LoadBalancingStrategy) -> mesh_net::LoadBalancingStrategy {
    match strategy {
        LoadBalancingStrategy::RoundRobin => mesh_net::LoadBalancingStrategy::RoundRobin,
        LoadBalancingStrategy::Random => mesh_net::LoadBalancingStrategy::Random,
        LoadBalancingStrategy::LeastConnections => mesh_net::LoadBalancingStrategy::LeastConnections,
        LoadBalancingStrategy::WeightedRoundRobin => mesh_net::LoadBalancingStrategy::WeightedRoundRobin,
        LoadBalancingStrategy::ConsistentHashing => mesh_net::LoadBalancingStrategy::ConsistentHash,
        LoadBalancingStrategy::ScoreBased => mesh_net::LoadBalancingStrategy::Random, // Fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RouterConfigBuilder;

    #[test]
    fn test_request_context() {
        let context = RequestContext::new()
            .with_client_ip("192.168.1.1")
            .with_user_agent("test-agent")
            .with_header("X-Test", "value");

        assert_eq!(context.client_ip, Some("192.168.1.1".to_string()));
        assert_eq!(context.user_agent, Some("test-agent".to_string()));
        assert_eq!(context.headers.get("X-Test"), Some(&"value".to_string()));
        assert!(!context.request_id.is_empty());
    }

    #[test]
    fn test_circuit_breaker_state() {
        let state = CircuitBreakerState {
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            last_failure: None,
            next_attempt: None,
        };

        assert_eq!(state.state, CircuitState::Closed);
        assert_eq!(state.failure_count, 0);
    }

    #[tokio::test]
    async fn test_request_handler_creation() {
        let config = RouterConfigBuilder::new().build();
        let state_store = StateStore::new();
        let query_engine = QueryEngine::new(state_store.clone());
        let scoring_engine = ScoringEngine::new();
        let connection_pool = ConnectionPool::new(mesh_net::TlsConfig::insecure());

        let handler = RequestHandler::new(
            config,
            state_store,
            query_engine,
            scoring_engine,
            connection_pool,
        ).await;

        assert!(handler.is_ok());
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = RouterConfigBuilder::new().build();
        let state_store = StateStore::new();
        let query_engine = QueryEngine::new(state_store.clone());
        let scoring_engine = ScoringEngine::new();
        let connection_pool = ConnectionPool::new(mesh_net::TlsConfig::insecure());

        let handler = RequestHandler::new(
            config,
            state_store,
            query_engine,
            scoring_engine,
            connection_pool,
        ).await.unwrap();

        let health = handler.health_check().await;
        assert!(health.is_ok());
        assert!(health.unwrap());
    }
}
