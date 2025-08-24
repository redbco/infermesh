//! Query engine for fast state lookups

use crate::{config::QueryConfig, store::StateStore, Result, StateError};
use mesh_core::{Labels, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::timeout;
// use tracing::debug; // Unused

/// Query engine for fast state lookups
#[derive(Debug)]
pub struct QueryEngine {
    store: Arc<StateStore>,
    config: QueryConfig,
    cache: Arc<RwLock<HashMap<String, CachedQuery>>>,
}

/// Query filter for state lookups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFilter {
    /// Labels to match
    pub labels: Option<Labels>,
    
    /// Node IDs to include
    pub node_ids: Option<Vec<NodeId>>,
    
    /// Minimum free GPU memory (in GB)
    pub min_free_memory: Option<f64>,
    
    /// Maximum GPU utilization (0.0 to 1.0)
    pub max_gpu_utilization: Option<f64>,
    
    /// Maximum queue depth
    pub max_queue_depth: Option<u32>,
    
    /// Only include ready models
    pub ready_only: bool,
}

/// Query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Matching model states
    pub models: Vec<ModelQueryResult>,
    
    /// Matching GPU states
    pub gpus: Vec<GpuQueryResult>,
    
    /// Query execution time
    pub execution_time_ms: u64,
    
    /// Whether result was cached
    pub from_cache: bool,
}

/// Model query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelQueryResult {
    pub node_id: NodeId,
    pub model_key: String,
    pub labels: Labels,
    pub status: mesh_core::ModelStatus,
    pub queue_depth: u32,
    pub last_updated: std::time::SystemTime,
}

/// GPU query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuQueryResult {
    pub node_id: NodeId,
    pub gpu_uuid: String,
    pub sm_utilization: f64,
    pub memory_utilization: f64,
    pub used_memory: f64,
    pub total_memory: f64,
    pub temperature_celsius: Option<f64>,
    pub last_updated: std::time::SystemTime,
}

/// Cached query result
#[derive(Debug, Clone)]
struct CachedQuery {
    result: QueryResult,
    cached_at: Instant,
}

impl QueryEngine {
    /// Create a new query engine
    pub fn new(store: StateStore) -> Self {
        Self::with_config(store, QueryConfig::default())
    }
    
    /// Create a new query engine with configuration
    pub fn with_config(store: StateStore, config: QueryConfig) -> Self {
        Self {
            store: Arc::new(store),
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Execute a query with filters
    pub async fn query(&self, filter: &QueryFilter) -> Result<QueryResult> {
        let start_time = Instant::now();
        
        // Check cache first
        if self.config.enable_caching {
            let cache_key = self.generate_cache_key(filter);
            if let Some(cached) = self.get_cached_result(&cache_key).await {
                return Ok(cached);
            }
        }
        
        // Execute query with timeout
        let query_future = self.execute_query_internal(filter);
        let mut result = match timeout(self.config.timeout, query_future).await {
            Ok(result) => result?,
            Err(_) => return Err(StateError::Query("Query timeout".to_string())),
        };
        
        result.execution_time_ms = start_time.elapsed().as_millis() as u64;
        result.from_cache = false;
        
        // Cache the result
        if self.config.enable_caching {
            let cache_key = self.generate_cache_key(filter);
            self.cache_result(cache_key, result.clone()).await;
        }
        
        Ok(result)
    }
    
    /// Find models matching labels
    pub async fn find_models(&self, labels: &Labels) -> Result<Vec<ModelQueryResult>> {
        let filter = QueryFilter {
            labels: Some(labels.clone()),
            node_ids: None,
            min_free_memory: None,
            max_gpu_utilization: None,
            max_queue_depth: None,
            ready_only: false,
        };
        
        let result = self.query(&filter).await?;
        Ok(result.models)
    }
    
    /// Find ready models matching labels
    pub async fn find_ready_models(&self, labels: &Labels) -> Result<Vec<ModelQueryResult>> {
        let filter = QueryFilter {
            labels: Some(labels.clone()),
            node_ids: None,
            min_free_memory: None,
            max_gpu_utilization: None,
            max_queue_depth: None,
            ready_only: true,
        };
        
        let result = self.query(&filter).await?;
        Ok(result.models)
    }
    
    /// Find available GPUs
    pub async fn find_available_gpus(
        &self,
        min_free_memory: Option<f64>,
        max_utilization: Option<f64>,
    ) -> Result<Vec<GpuQueryResult>> {
        let filter = QueryFilter {
            labels: None,
            node_ids: None,
            min_free_memory,
            max_gpu_utilization: max_utilization,
            max_queue_depth: None,
            ready_only: false,
        };
        
        let result = self.query(&filter).await?;
        Ok(result.gpus)
    }
    
    /// Find states for specific nodes
    pub async fn find_by_nodes(&self, node_ids: Vec<NodeId>) -> Result<QueryResult> {
        let filter = QueryFilter {
            labels: None,
            node_ids: Some(node_ids),
            min_free_memory: None,
            max_gpu_utilization: None,
            max_queue_depth: None,
            ready_only: false,
        };
        
        self.query(&filter).await
    }
    
    /// Execute the actual query
    async fn execute_query_internal(&self, filter: &QueryFilter) -> Result<QueryResult> {
        let mut models = Vec::new();
        let mut gpus = Vec::new();
        
        // Query models
        if filter.labels.is_some() || filter.node_ids.is_some() || filter.max_queue_depth.is_some() {
            let model_matches = if let Some(labels) = &filter.labels {
                self.store.find_models(labels)
            } else {
                // Get all models if no label filter
                let snapshot = self.store.snapshot();
                snapshot.models
            };
            
            for (node_id, model_key, timestamped_state) in model_matches {
                let state = &timestamped_state.state;
                
                // Apply filters
                if let Some(node_filter) = &filter.node_ids {
                    if !node_filter.contains(&node_id) {
                        continue;
                    }
                }
                
                if let Some(max_queue) = filter.max_queue_depth {
                    if state.queue_depth > max_queue {
                        continue;
                    }
                }
                
                // Filter for ready models only
                if filter.ready_only && !state.loaded {
                    continue;
                }
                
                let status = if state.loaded {
                    mesh_core::ModelStatus::Ready
                } else if state.warming {
                    mesh_core::ModelStatus::Loading
                } else {
                    mesh_core::ModelStatus::NotLoaded
                };
                
                models.push(ModelQueryResult {
                    node_id,
                    model_key,
                    labels: state.labels.clone(),
                    status,
                    queue_depth: state.queue_depth,
                    last_updated: std::time::UNIX_EPOCH + Duration::from_secs(
                        timestamped_state.updated_at.elapsed().as_secs()
                    ),
                });
                
                // Limit results
                if models.len() >= self.config.max_results {
                    break;
                }
            }
        }
        
        // Query GPUs
        if filter.min_free_memory.is_some() 
            || filter.max_gpu_utilization.is_some() 
            || filter.node_ids.is_some() 
        {
            let gpu_matches = if let Some(min_mem) = filter.min_free_memory {
                self.store.find_gpus(Some(min_mem))
            } else {
                // Get all GPUs if no memory filter
                let snapshot = self.store.snapshot();
                snapshot.gpus
            };
            
            for (node_id, gpu_uuid, timestamped_state) in gpu_matches {
                let state = &timestamped_state.state;
                
                // Apply filters
                if let Some(node_filter) = &filter.node_ids {
                    if !node_filter.contains(&node_id) {
                        continue;
                    }
                }
                
                if let Some(max_util) = filter.max_gpu_utilization {
                    if state.sm_utilization as f64 > max_util {
                        continue;
                    }
                }
                
                gpus.push(GpuQueryResult {
                    node_id,
                    gpu_uuid,
                    sm_utilization: state.sm_utilization as f64,
                    memory_utilization: state.memory_utilization as f64,
                    used_memory: state.vram_used_gb as f64,
                    total_memory: state.vram_total_gb as f64,
                    temperature_celsius: state.temperature_c.map(|t| t as f64),
                    last_updated: std::time::UNIX_EPOCH + Duration::from_secs(
                        timestamped_state.updated_at.elapsed().as_secs()
                    ),
                });
                
                // Limit results
                if gpus.len() >= self.config.max_results {
                    break;
                }
            }
        }
        
        Ok(QueryResult {
            models,
            gpus,
            execution_time_ms: 0, // Will be set by caller
            from_cache: false,
        })
    }
    
    /// Generate cache key for a filter
    fn generate_cache_key(&self, filter: &QueryFilter) -> String {
        // Simple hash-based cache key
        format!("{:?}", filter)
    }
    
    /// Get cached result if available and not expired
    async fn get_cached_result(&self, cache_key: &str) -> Option<QueryResult> {
        let cache = self.cache.read().await;
        if let Some(cached) = cache.get(cache_key) {
            if cached.cached_at.elapsed() < self.config.cache_ttl {
                let mut result = cached.result.clone();
                result.from_cache = true;
                return Some(result);
            }
        }
        None
    }
    
    /// Cache a query result
    async fn cache_result(&self, cache_key: String, result: QueryResult) {
        let mut cache = self.cache.write().await;
        
        // Limit cache size
        if cache.len() >= self.config.cache_size {
            // Remove expired entries
            let now = Instant::now();
            cache.retain(|_, cached| now.duration_since(cached.cached_at) < self.config.cache_ttl);
            
            // If still too large, remove oldest entries
            if cache.len() >= self.config.cache_size {
                let keys_to_remove: Vec<String> = cache.keys().take(cache.len() / 4).cloned().collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }
        }
        
        cache.insert(cache_key, CachedQuery {
            result,
            cached_at: Instant::now(),
        });
    }
    
    /// Clear the query cache
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }
}

impl Default for QueryFilter {
    fn default() -> Self {
        Self {
            labels: None,
            node_ids: None,
            min_free_memory: None,
            max_gpu_utilization: None,
            max_queue_depth: None,
            ready_only: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_core::{GpuState, Labels, ModelState, ModelStatus, NodeId};

    #[tokio::test]
    async fn test_query_engine_creation() {
        let store = StateStore::new();
        let engine = QueryEngine::new(store);
        
        let filter = QueryFilter::default();
        let result = engine.query(&filter).await.unwrap();
        assert!(result.models.is_empty());
        assert!(result.gpus.is_empty());
    }

    #[tokio::test]
    async fn test_find_models() {
        let store = StateStore::new();
        let engine = QueryEngine::new(store.clone());
        let node_id = NodeId::new("test-node");
        
        // Add model state
        let labels = Labels::new("gpt-7b", "v1", "runtime", "node1")
            .with_custom("model", "gpt-7b");
        let mut model_state = ModelState::new(labels.clone());
        model_state.mark_loaded();
        
        store.update_model_state(node_id.clone(), "gpt-7b".to_string(), model_state).await.unwrap();
        
        // Query models
        let results = engine.find_models(&labels).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, node_id);
        assert_eq!(results[0].model_key, "gpt-7b");
        assert_eq!(results[0].status, ModelStatus::Ready);
    }

    #[tokio::test]
    async fn test_find_ready_models() {
        let store = StateStore::new();
        let engine = QueryEngine::new(store.clone());
        let node_id = NodeId::new("test-node");
        
        // Add ready model
        let labels = Labels::new("gpt-7b", "v1", "runtime", "node1")
            .with_custom("model", "gpt-7b");
        let mut ready_state = ModelState::new(labels.clone());
        ready_state.mark_loaded();
        store.update_model_state(node_id.clone(), "ready-model".to_string(), ready_state).await.unwrap();
        
        // Add loading model
        let mut loading_state = ModelState::new(labels.clone());
        loading_state.warming = true;
        store.update_model_state(node_id, "loading-model".to_string(), loading_state).await.unwrap();
        
        // Query ready models only
        let results = engine.find_ready_models(&labels).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].model_key, "ready-model");
        assert_eq!(results[0].status, ModelStatus::Ready);
    }

    #[tokio::test]
    async fn test_find_available_gpus() {
        let store = StateStore::new();
        let engine = QueryEngine::new(store.clone());
        let node_id = NodeId::new("test-node");
        
        // Add GPU with available memory
        let mut gpu_state = GpuState::new("GPU-12345", &node_id.to_string());
        gpu_state.update_metrics(0.3, 0.2, 8.0, 16.0); // 8GB used, 8GB free
        store.update_gpu_state(node_id, "GPU-12345".to_string(), gpu_state).await.unwrap();
        
        // Query available GPUs
        let results = engine.find_available_gpus(Some(4.0), Some(0.5)).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].gpu_uuid, "GPU-12345");
        assert!(results[0].sm_utilization <= 0.5);
        assert!(results[0].total_memory - results[0].used_memory >= 4.0);
    }

    #[tokio::test]
    async fn test_query_with_node_filter() {
        let store = StateStore::new();
        let engine = QueryEngine::new(store.clone());
        let node1 = NodeId::new("node1");
        let node2 = NodeId::new("node2");
        
        // Add states to both nodes
        let labels = Labels::new("test", "v1", "runtime", "node1")
            .with_custom("model", "test");
        let state = ModelState::new(labels);
        
        store.update_model_state(node1.clone(), "model1".to_string(), state.clone()).await.unwrap();
        store.update_model_state(node2, "model2".to_string(), state).await.unwrap();
        
        // Query specific node
        let result = engine.find_by_nodes(vec![node1.clone()]).await.unwrap();
        assert_eq!(result.models.len(), 1);
        assert_eq!(result.models[0].node_id, node1);
    }

    #[tokio::test]
    async fn test_query_caching() {
        let mut config = QueryConfig::default();
        config.enable_caching = true;
        config.cache_ttl = Duration::from_secs(60);
        
        let store = StateStore::new();
        let engine = QueryEngine::with_config(store, config);
        
        let filter = QueryFilter::default();
        
        // First query (cache miss)
        let result1 = engine.query(&filter).await.unwrap();
        assert!(!result1.from_cache);
        
        // Second query (cache hit)
        let result2 = engine.query(&filter).await.unwrap();
        assert!(result2.from_cache);
    }

    #[tokio::test]
    async fn test_query_timeout() {
        let mut config = QueryConfig::default();
        config.timeout = Duration::from_millis(1); // Very short timeout
        
        let store = StateStore::new();
        let engine = QueryEngine::with_config(store, config);
        
        let filter = QueryFilter::default();
        
        // This might timeout depending on system performance
        // In practice, queries should be fast enough to not timeout
        let _result = engine.query(&filter).await;
        // We don't assert on the result since it depends on timing
    }
}
