//! Scoring engine for routing decisions

use crate::{config::ScoringConfig, store::StateStore, Result};
use mesh_core::{Labels, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::debug;

/// Scoring engine for routing decisions
#[derive(Debug)]
pub struct ScoringEngine {
    config: ScoringConfig,
    cache: Arc<RwLock<HashMap<String, CachedScore>>>,
    metrics: Arc<ScoringMetrics>,
}

/// A scored target for routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredTarget {
    /// Node ID
    pub node_id: NodeId,
    
    /// GPU UUID (if applicable)
    pub gpu_uuid: Option<String>,
    
    /// Model key (if applicable)
    pub model_key: Option<String>,
    
    /// Overall score (higher is better)
    pub score: f64,
    
    /// Individual scoring factors
    pub factors: ScoringFactors,
    
    /// Estimated latency in milliseconds
    pub estimated_latency_ms: f64,
    
    /// Estimated queue time in milliseconds
    pub estimated_queue_time_ms: f64,
    
    /// Reason for the score
    pub reason: String,
}

/// Individual scoring factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringFactors {
    /// GPU utilization factor (0.0 = fully utilized, 1.0 = idle)
    pub gpu_utilization: f64,
    
    /// Queue depth factor (0.0 = full queue, 1.0 = empty queue)
    pub queue_depth: f64,
    
    /// Memory usage factor (0.0 = full memory, 1.0 = empty memory)
    pub memory_usage: f64,
    
    /// Network latency factor (0.0 = high latency, 1.0 = low latency)
    pub network_latency: f64,
    
    /// Model compatibility factor (0.0 = incompatible, 1.0 = perfect match)
    pub model_compatibility: f64,
    
    /// Node health factor (0.0 = unhealthy, 1.0 = healthy)
    pub node_health: f64,
}

/// Cached scoring result
#[derive(Debug, Clone)]
struct CachedScore {
    targets: Vec<ScoredTarget>,
    cached_at: Instant,
}

/// Scoring metrics
#[derive(Debug, Default)]
pub struct ScoringMetrics {
    pub total_requests: std::sync::atomic::AtomicU64,
    pub cache_hits: std::sync::atomic::AtomicU64,
    pub cache_misses: std::sync::atomic::AtomicU64,
    pub scoring_duration_ms: std::sync::atomic::AtomicU64,
}

impl ScoringEngine {
    /// Create a new scoring engine
    pub fn new() -> Self {
        Self::with_config(ScoringConfig::default())
    }
    
    /// Create a new scoring engine with configuration
    pub fn with_config(config: ScoringConfig) -> Self {
        Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(ScoringMetrics::default()),
        }
    }
    
    /// Check if scoring is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
    
    /// Score targets for a given request
    pub async fn score_targets(
        &self,
        store: &StateStore,
        labels: &Labels,
        estimated_tokens: u32,
    ) -> Result<Vec<ScoredTarget>> {
        if !self.config.enabled {
            return Ok(Vec::new());
        }
        
        let start_time = Instant::now();
        self.metrics.total_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        // Check cache first
        let cache_key = self.generate_cache_key(labels, estimated_tokens);
        if let Some(cached) = self.get_cached_result(&cache_key).await {
            self.metrics.cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(cached);
        }
        
        self.metrics.cache_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        // Find matching models
        let model_matches = store.find_models(labels);
        if model_matches.is_empty() {
            debug!("No models found matching labels: {:?}", labels);
            return Ok(Vec::new());
        }
        
        let mut targets = Vec::new();
        
        for (node_id, model_key, model_state) in model_matches {
            // Get GPU states for this node
            let gpu_states = store.get_node_gpu_states(&node_id);
            
            if gpu_states.is_empty() {
                // Score without GPU information
                let target = self.score_target_without_gpu(
                    node_id,
                    Some(model_key),
                    &model_state.state,
                    estimated_tokens,
                ).await?;
                targets.push(target);
            } else {
                // Score each GPU separately
                for (gpu_uuid, gpu_state) in gpu_states {
                    let target = self.score_target_with_gpu(
                        node_id.clone(),
                        Some(model_key.clone()),
                        &model_state.state,
                        &gpu_uuid,
                        &gpu_state.state,
                        estimated_tokens,
                    ).await?;
                    targets.push(target);
                }
            }
        }
        
        // Sort by score (highest first)
        targets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Cache the result
        self.cache_result(cache_key, targets.clone()).await;
        
        let duration = start_time.elapsed();
        self.metrics.scoring_duration_ms.fetch_add(
            duration.as_millis() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        
        debug!("Scored {} targets in {:?}", targets.len(), duration);
        
        Ok(targets)
    }
    
    /// Score a target without GPU information
    async fn score_target_without_gpu(
        &self,
        node_id: NodeId,
        model_key: Option<String>,
        model_state: &mesh_core::ModelState,
        estimated_tokens: u32,
    ) -> Result<ScoredTarget> {
        let factors = ScoringFactors {
            gpu_utilization: 0.5, // Unknown, assume moderate
            queue_depth: self.calculate_queue_depth_factor(model_state.queue_depth),
            memory_usage: 0.5, // Unknown, assume moderate
            network_latency: 1.0, // Assume good network
            model_compatibility: 1.0, // Already matched
            node_health: 1.0, // Assume healthy
        };
        
        let score = self.calculate_weighted_score(&factors);
        let estimated_latency = self.estimate_latency(estimated_tokens, &factors);
        let estimated_queue_time = self.estimate_queue_time(model_state.queue_depth);
        
        Ok(ScoredTarget {
            node_id,
            gpu_uuid: None,
            model_key,
            score,
            factors,
            estimated_latency_ms: estimated_latency,
            estimated_queue_time_ms: estimated_queue_time,
            reason: "Scored without GPU information".to_string(),
        })
    }
    
    /// Score a target with GPU information
    async fn score_target_with_gpu(
        &self,
        node_id: NodeId,
        model_key: Option<String>,
        model_state: &mesh_core::ModelState,
        gpu_uuid: &str,
        gpu_state: &mesh_core::GpuState,
        estimated_tokens: u32,
    ) -> Result<ScoredTarget> {
        let factors = ScoringFactors {
            gpu_utilization: 1.0 - gpu_state.sm_utilization as f64,
            queue_depth: self.calculate_queue_depth_factor(model_state.queue_depth),
            memory_usage: self.calculate_memory_usage_factor(gpu_state),
            network_latency: 1.0, // TODO: Implement network latency measurement
            model_compatibility: 1.0, // Already matched
            node_health: self.calculate_node_health_factor(gpu_state),
        };
        
        let score = self.calculate_weighted_score(&factors);
        let estimated_latency = self.estimate_latency(estimated_tokens, &factors);
        let estimated_queue_time = self.estimate_queue_time(model_state.queue_depth);
        
        Ok(ScoredTarget {
            node_id,
            gpu_uuid: Some(gpu_uuid.to_string()),
            model_key,
            score,
            factors,
            estimated_latency_ms: estimated_latency,
            estimated_queue_time_ms: estimated_queue_time,
            reason: format!("Scored with GPU {} information", gpu_uuid),
        })
    }
    
    /// Calculate weighted score from factors
    fn calculate_weighted_score(&self, factors: &ScoringFactors) -> f64 {
        let weights = &self.config.weights;
        
        factors.gpu_utilization * weights.gpu_utilization
            + factors.queue_depth * weights.queue_depth
            + factors.memory_usage * weights.memory_usage
            + factors.network_latency * weights.network_latency
            + factors.model_compatibility * weights.model_compatibility
            + factors.node_health * weights.node_health
    }
    
    /// Calculate queue depth factor (0.0 = full queue, 1.0 = empty queue)
    fn calculate_queue_depth_factor(&self, queue_depth: u32) -> f64 {
        // Assume max reasonable queue depth of 100
        let max_queue = 100.0;
        let normalized = (queue_depth as f64 / max_queue).min(1.0);
        1.0 - normalized
    }
    
    /// Calculate memory usage factor (0.0 = full memory, 1.0 = empty memory)
    fn calculate_memory_usage_factor(&self, gpu_state: &mesh_core::GpuState) -> f64 {
        // Calculate memory usage from actual memory values
        if gpu_state.vram_total_gb <= 0.0 {
            return 0.0; // No memory available
        }
        
        let used_ratio = gpu_state.vram_used_gb / gpu_state.vram_total_gb;
        1.0 - (used_ratio.min(1.0).max(0.0) as f64)
    }
    
    /// Calculate node health factor
    fn calculate_node_health_factor(&self, gpu_state: &mesh_core::GpuState) -> f64 {
        let mut health = 1.0;
        
        // Check temperature
        if let Some(temp) = gpu_state.temperature_c {
            if temp > 85.0 {
                health *= 0.7; // High temperature penalty
            } else if temp > 75.0 {
                health *= 0.9; // Moderate temperature penalty
            }
        }
        
        // Check power usage
        // TODO: Add power usage to GpuState
        // For now, skip power-based health calculation
        if false { // Placeholder
            let power = 0.0; // Placeholder
            if power > 300.0 {
                health *= 0.9; // High power usage penalty
            }
        }
        
        health
    }
    
    /// Estimate latency based on tokens and factors
    fn estimate_latency(&self, estimated_tokens: u32, factors: &ScoringFactors) -> f64 {
        // Base latency per token (in milliseconds)
        let base_latency_per_token = 2.0;
        
        // Adjust based on GPU utilization
        let utilization_multiplier = 1.0 + (1.0 - factors.gpu_utilization) * 2.0;
        
        // Adjust based on memory pressure
        let memory_multiplier = 1.0 + (1.0 - factors.memory_usage) * 0.5;
        
        let base_latency = estimated_tokens as f64 * base_latency_per_token;
        base_latency * utilization_multiplier * memory_multiplier
    }
    
    /// Estimate queue time based on queue depth
    fn estimate_queue_time(&self, queue_depth: u32) -> f64 {
        // Assume average request takes 1 second
        let avg_request_time_ms = 1000.0;
        queue_depth as f64 * avg_request_time_ms
    }
    
    /// Generate cache key for a request
    fn generate_cache_key(&self, labels: &Labels, estimated_tokens: u32) -> String {
        format!("{}:{}", labels.key(), estimated_tokens)
    }
    
    /// Get cached result if available and not expired
    async fn get_cached_result(&self, cache_key: &str) -> Option<Vec<ScoredTarget>> {
        let cache = self.cache.read().await;
        if let Some(cached) = cache.get(cache_key) {
            if cached.cached_at.elapsed() < self.config.cache_ttl {
                return Some(cached.targets.clone());
            }
        }
        None
    }
    
    /// Cache a scoring result
    async fn cache_result(&self, cache_key: String, targets: Vec<ScoredTarget>) {
        let mut cache = self.cache.write().await;
        
        // Limit cache size
        if cache.len() >= self.config.cache_size {
            // Remove oldest entries (simple LRU approximation)
            let now = Instant::now();
            cache.retain(|_, cached| now.duration_since(cached.cached_at) < self.config.cache_ttl);
            
            // If still too large, clear some entries
            if cache.len() >= self.config.cache_size {
                let keys_to_remove: Vec<String> = cache.keys().take(cache.len() / 4).cloned().collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }
        }
        
        cache.insert(cache_key, CachedScore {
            targets,
            cached_at: Instant::now(),
        });
    }
    
    /// Clear the scoring cache
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }
    
    /// Get scoring metrics
    pub fn metrics(&self) -> &ScoringMetrics {
        &self.metrics
    }
}

impl Default for ScoringEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::StateStore;
    use mesh_core::{GpuState, Labels, ModelState, NodeId};

    #[tokio::test]
    async fn test_scoring_engine_creation() {
        let engine = ScoringEngine::new();
        assert!(engine.is_enabled());
    }

    #[tokio::test]
    async fn test_score_targets_no_models() {
        let engine = ScoringEngine::new();
        let store = StateStore::new();
        let labels = Labels::new("nonexistent", "v1", "runtime", "node1")
            .with_custom("model", "nonexistent");
        
        let targets = engine.score_targets(&store, &labels, 100).await.unwrap();
        assert!(targets.is_empty());
    }

    #[tokio::test]
    async fn test_score_targets_with_models() {
        let engine = ScoringEngine::new();
        let store = StateStore::new();
        let node_id = NodeId::new("test-node");
        
        // Add a model state
        let labels = Labels::new("gpt-7b", "v1", "runtime", "node1")
            .with_custom("model", "gpt-7b");
        let mut model_state = ModelState::new(labels.clone());
        model_state.mark_loaded();
        model_state.queue_depth = 5;
        
        store.update_model_state(node_id.clone(), "gpt-7b".to_string(), model_state).await.unwrap();
        
        // Add a GPU state
        let mut gpu_state = GpuState::new("GPU-12345", &node_id.to_string());
        gpu_state.update_metrics(0.3, 0.2, 8.0, 16.0); // 30% utilization, 8GB used of 16GB
        
        store.update_gpu_state(node_id, "GPU-12345".to_string(), gpu_state).await.unwrap();
        
        // Score targets
        let targets = engine.score_targets(&store, &labels, 100).await.unwrap();
        assert_eq!(targets.len(), 1);
        
        let target = &targets[0];
        assert!(target.score > 0.0);
        assert!(target.score <= 1.0);
        assert_eq!(target.gpu_uuid, Some("GPU-12345".to_string()));
        assert_eq!(target.model_key, Some("gpt-7b".to_string()));
        assert!(target.estimated_latency_ms > 0.0);
        assert!(target.estimated_queue_time_ms > 0.0);
    }

    #[tokio::test]
    async fn test_scoring_factors() {
        let engine = ScoringEngine::new();
        
        // Test queue depth factor
        assert_eq!(engine.calculate_queue_depth_factor(0), 1.0); // Empty queue
        assert!(engine.calculate_queue_depth_factor(50) < 1.0); // Partial queue
        assert_eq!(engine.calculate_queue_depth_factor(100), 0.0); // Full queue
        
        // Test memory usage factor
        let mut gpu_state = GpuState::new("GPU-12345", "node1");
        gpu_state.update_metrics(0.5, 0.3, 0.0, 16.0); // 0GB used of 16GB
        assert_eq!(engine.calculate_memory_usage_factor(&gpu_state), 1.0);
        
        gpu_state.update_metrics(0.5, 0.3, 16.0, 16.0); // 16GB used of 16GB
        assert_eq!(engine.calculate_memory_usage_factor(&gpu_state), 0.0);
        
        gpu_state.update_metrics(0.5, 0.3, 8.0, 16.0); // 8GB used of 16GB
        assert_eq!(engine.calculate_memory_usage_factor(&gpu_state), 0.5);
    }

    #[tokio::test]
    async fn test_caching() {
        let engine = ScoringEngine::new();
        let store = StateStore::new();
        let node_id = NodeId::new("test-node");
        let labels = Labels::new("test", "v1", "runtime", "node1")
            .with_custom("model", "test");
        
        // Add a model state so there's something to score
        let mut model_state = ModelState::new(labels.clone());
        model_state.mark_loaded();
        store.update_model_state(node_id, "test-model".to_string(), model_state).await.unwrap();
        
        // First request (cache miss)
        let targets1 = engine.score_targets(&store, &labels, 100).await.unwrap();
        assert_eq!(engine.metrics().cache_misses.load(std::sync::atomic::Ordering::Relaxed), 1);
        
        // Second request (cache hit)
        let targets2 = engine.score_targets(&store, &labels, 100).await.unwrap();
        assert_eq!(engine.metrics().cache_hits.load(std::sync::atomic::Ordering::Relaxed), 1);
        
        assert_eq!(targets1.len(), targets2.len());
    }

    #[tokio::test]
    async fn test_disabled_scoring() {
        let mut config = ScoringConfig::default();
        config.enabled = false;
        
        let engine = ScoringEngine::with_config(config);
        let store = StateStore::new();
        let labels = Labels::new("test", "v1", "runtime", "node1")
            .with_custom("model", "test");
        
        let targets = engine.score_targets(&store, &labels, 100).await.unwrap();
        assert!(targets.is_empty());
        assert!(!engine.is_enabled());
    }
}
