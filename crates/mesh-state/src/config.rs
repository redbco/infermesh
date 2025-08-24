//! Configuration for state management

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for the state management system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateConfig {
    /// Maximum number of model states to track
    pub max_model_states: usize,
    
    /// Maximum number of GPU states to track
    pub max_gpu_states: usize,
    
    /// Interval for state cleanup
    pub cleanup_interval: Duration,
    
    /// Maximum age of state entries before cleanup
    pub max_state_age: Duration,
    
    /// Enable state persistence
    pub enable_persistence: bool,
    
    /// Persistence file path
    pub persistence_path: Option<String>,
    
    /// Scoring configuration
    pub scoring: ScoringConfig,
    
    /// Query configuration
    pub query: QueryConfig,
    
    /// Synchronization configuration
    pub sync: SyncConfig,
}

/// Configuration for the scoring engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringConfig {
    /// Enable scoring
    pub enabled: bool,
    
    /// Default scoring algorithm
    pub default_algorithm: String,
    
    /// Scoring weights
    pub weights: ScoringWeights,
    
    /// Cache size for scoring results
    pub cache_size: usize,
    
    /// Cache TTL for scoring results
    pub cache_ttl: Duration,
}

/// Weights for different scoring factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringWeights {
    /// Weight for GPU utilization (lower is better)
    pub gpu_utilization: f64,
    
    /// Weight for queue depth (lower is better)
    pub queue_depth: f64,
    
    /// Weight for memory usage (lower is better)
    pub memory_usage: f64,
    
    /// Weight for network latency (lower is better)
    pub network_latency: f64,
    
    /// Weight for model compatibility (higher is better)
    pub model_compatibility: f64,
    
    /// Weight for node health (higher is better)
    pub node_health: f64,
}

/// Configuration for query operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConfig {
    /// Maximum number of results to return
    pub max_results: usize,
    
    /// Query timeout
    pub timeout: Duration,
    
    /// Enable query caching
    pub enable_caching: bool,
    
    /// Query cache size
    pub cache_size: usize,
    
    /// Query cache TTL
    pub cache_ttl: Duration,
}

/// Configuration for state synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Enable state synchronization
    pub enabled: bool,
    
    /// Synchronization interval
    pub sync_interval: Duration,
    
    /// Batch size for synchronization
    pub batch_size: usize,
    
    /// Maximum number of pending sync operations
    pub max_pending_ops: usize,
    
    /// Sync timeout
    pub sync_timeout: Duration,
}

impl Default for StateConfig {
    fn default() -> Self {
        Self {
            max_model_states: 10000,
            max_gpu_states: 1000,
            cleanup_interval: Duration::from_secs(300), // 5 minutes
            max_state_age: Duration::from_secs(3600),   // 1 hour
            enable_persistence: false,
            persistence_path: None,
            scoring: ScoringConfig::default(),
            query: QueryConfig::default(),
            sync: SyncConfig::default(),
        }
    }
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_algorithm: "weighted".to_string(),
            weights: ScoringWeights::default(),
            cache_size: 1000,
            cache_ttl: Duration::from_secs(60),
        }
    }
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            gpu_utilization: 0.3,
            queue_depth: 0.25,
            memory_usage: 0.2,
            network_latency: 0.1,
            model_compatibility: 0.1,
            node_health: 0.05,
        }
    }
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            max_results: 100,
            timeout: Duration::from_millis(500),
            enable_caching: true,
            cache_size: 500,
            cache_ttl: Duration::from_secs(30),
        }
    }
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sync_interval: Duration::from_millis(100),
            batch_size: 50,
            max_pending_ops: 1000,
            sync_timeout: Duration::from_secs(5),
        }
    }
}

impl StateConfig {
    /// Create a new state configuration
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the maximum number of model states
    pub fn with_max_model_states(mut self, max: usize) -> Self {
        self.max_model_states = max;
        self
    }
    
    /// Set the maximum number of GPU states
    pub fn with_max_gpu_states(mut self, max: usize) -> Self {
        self.max_gpu_states = max;
        self
    }
    
    /// Enable persistence with a file path
    pub fn with_persistence(mut self, path: String) -> Self {
        self.enable_persistence = true;
        self.persistence_path = Some(path);
        self
    }
    
    /// Set scoring weights
    pub fn with_scoring_weights(mut self, weights: ScoringWeights) -> Self {
        self.scoring.weights = weights;
        self
    }
    
    /// Disable scoring
    pub fn without_scoring(mut self) -> Self {
        self.scoring.enabled = false;
        self
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.max_model_states == 0 {
            return Err("max_model_states must be greater than zero".to_string());
        }
        
        if self.max_gpu_states == 0 {
            return Err("max_gpu_states must be greater than zero".to_string());
        }
        
        if self.cleanup_interval.is_zero() {
            return Err("cleanup_interval must be greater than zero".to_string());
        }
        
        if self.max_state_age.is_zero() {
            return Err("max_state_age must be greater than zero".to_string());
        }
        
        if self.enable_persistence && self.persistence_path.is_none() {
            return Err("persistence_path required when persistence is enabled".to_string());
        }
        
        // Validate scoring weights sum to approximately 1.0
        let weights = &self.scoring.weights;
        let total_weight = weights.gpu_utilization
            + weights.queue_depth
            + weights.memory_usage
            + weights.network_latency
            + weights.model_compatibility
            + weights.node_health;
        
        if (total_weight - 1.0).abs() > 0.1 {
            return Err(format!(
                "scoring weights should sum to approximately 1.0, got {}",
                total_weight
            ));
        }
        
        if self.query.max_results == 0 {
            return Err("query max_results must be greater than zero".to_string());
        }
        
        if self.query.timeout.is_zero() {
            return Err("query timeout must be greater than zero".to_string());
        }
        
        if self.sync.batch_size == 0 {
            return Err("sync batch_size must be greater than zero".to_string());
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = StateConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.scoring.enabled);
        assert!(!config.enable_persistence);
    }

    #[test]
    fn test_config_builder() {
        let config = StateConfig::new()
            .with_max_model_states(5000)
            .with_max_gpu_states(500)
            .with_persistence("/tmp/state.json".to_string())
            .without_scoring();
        
        assert_eq!(config.max_model_states, 5000);
        assert_eq!(config.max_gpu_states, 500);
        assert!(config.enable_persistence);
        assert!(!config.scoring.enabled);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let mut config = StateConfig::default();
        
        // Test zero max_model_states
        config.max_model_states = 0;
        assert!(config.validate().is_err());
        
        // Test persistence without path
        config = StateConfig::default();
        config.enable_persistence = true;
        config.persistence_path = None;
        assert!(config.validate().is_err());
        
        // Test invalid scoring weights
        config = StateConfig::default();
        config.scoring.weights.gpu_utilization = 2.0; // This will make total > 1.1
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_scoring_weights_default() {
        let weights = ScoringWeights::default();
        let total = weights.gpu_utilization
            + weights.queue_depth
            + weights.memory_usage
            + weights.network_latency
            + weights.model_compatibility
            + weights.node_health;
        
        // Should sum to approximately 1.0
        assert!((total - 1.0).abs() < 0.01);
    }
}
