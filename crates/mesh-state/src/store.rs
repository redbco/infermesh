//! In-memory state store for models and GPUs

use crate::{config::StateConfig, Result, StateError};
use dashmap::DashMap;
use mesh_core::{GpuState, Labels, ModelState, NodeId};
// use serde::{Deserialize, Serialize}; // Unused for now
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// In-memory state store for models and GPUs
#[derive(Debug, Clone)]
pub struct StateStore {
    /// Model states indexed by (node_id, model_key)
    model_states: Arc<DashMap<(NodeId, String), TimestampedModelState>>,
    
    /// GPU states indexed by (node_id, gpu_uuid)
    gpu_states: Arc<DashMap<(NodeId, String), TimestampedGpuState>>,
    
    /// Configuration
    config: StateConfig,
    
    /// Statistics
    stats: Arc<StoreStats>,
    
    /// Last cleanup time
    last_cleanup: Arc<RwLock<Instant>>,
}

/// Model state with timestamp
#[derive(Debug, Clone)]
// Note: Instant doesn't implement Serialize/Deserialize
pub struct TimestampedModelState {
    pub state: ModelState,
    pub updated_at: Instant,
    pub version: u64,
}

/// GPU state with timestamp
#[derive(Debug, Clone)]
// Note: Instant doesn't implement Serialize/Deserialize
pub struct TimestampedGpuState {
    pub state: GpuState,
    pub updated_at: Instant,
    pub version: u64,
}

/// Store statistics
#[derive(Debug, Default)]
pub struct StoreStats {
    pub model_updates: AtomicU64,
    pub gpu_updates: AtomicU64,
    pub model_queries: AtomicU64,
    pub gpu_queries: AtomicU64,
    pub cleanup_runs: AtomicU64,
    pub entries_cleaned: AtomicU64,
}

/// Snapshot of the current state
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub models: Vec<(NodeId, String, TimestampedModelState)>,
    pub gpus: Vec<(NodeId, String, TimestampedGpuState)>,
    pub timestamp: Instant,
}

impl StateStore {
    /// Create a new state store
    pub fn new() -> Self {
        Self::with_config(StateConfig::default())
    }
    
    /// Create a new state store with configuration
    pub fn with_config(config: StateConfig) -> Self {
        Self {
            model_states: Arc::new(DashMap::new()),
            gpu_states: Arc::new(DashMap::new()),
            config,
            stats: Arc::new(StoreStats::default()),
            last_cleanup: Arc::new(RwLock::new(Instant::now())),
        }
    }
    
    /// Update a model state
    pub async fn update_model_state(
        &self,
        node_id: NodeId,
        model_key: String,
        state: ModelState,
    ) -> Result<()> {
        let key = (node_id.clone(), model_key.clone());
        let now = Instant::now();
        
        // Check capacity
        if self.model_states.len() >= self.config.max_model_states
            && !self.model_states.contains_key(&key)
        {
            return Err(StateError::Store(format!(
                "Model state capacity exceeded: {}",
                self.config.max_model_states
            )));
        }
        
        let version = match self.model_states.get(&key) {
            Some(existing) => existing.version + 1,
            None => 1,
        };
        
        let timestamped_state = TimestampedModelState {
            state,
            updated_at: now,
            version,
        };
        
        self.model_states.insert(key, timestamped_state);
        self.stats.model_updates.fetch_add(1, Ordering::Relaxed);
        
        debug!("Updated model state for {} on node {}", model_key, node_id);
        
        // Trigger cleanup if needed
        self.maybe_cleanup().await;
        
        Ok(())
    }
    
    /// Update a GPU state
    pub async fn update_gpu_state(
        &self,
        node_id: NodeId,
        gpu_uuid: String,
        state: GpuState,
    ) -> Result<()> {
        let key = (node_id.clone(), gpu_uuid.clone());
        let now = Instant::now();
        
        // Check capacity
        if self.gpu_states.len() >= self.config.max_gpu_states
            && !self.gpu_states.contains_key(&key)
        {
            return Err(StateError::Store(format!(
                "GPU state capacity exceeded: {}",
                self.config.max_gpu_states
            )));
        }
        
        let version = match self.gpu_states.get(&key) {
            Some(existing) => existing.version + 1,
            None => 1,
        };
        
        let timestamped_state = TimestampedGpuState {
            state,
            updated_at: now,
            version,
        };
        
        self.gpu_states.insert(key, timestamped_state);
        self.stats.gpu_updates.fetch_add(1, Ordering::Relaxed);
        
        debug!("Updated GPU state for {} on node {}", gpu_uuid, node_id);
        
        // Trigger cleanup if needed
        self.maybe_cleanup().await;
        
        Ok(())
    }
    
    /// Get a model state
    pub fn get_model_state(&self, node_id: &NodeId, model_key: &str) -> Option<TimestampedModelState> {
        let key = (node_id.clone(), model_key.to_string());
        self.stats.model_queries.fetch_add(1, Ordering::Relaxed);
        self.model_states.get(&key).map(|entry| entry.clone())
    }
    
    /// Get a GPU state
    pub fn get_gpu_state(&self, node_id: &NodeId, gpu_uuid: &str) -> Option<TimestampedGpuState> {
        let key = (node_id.clone(), gpu_uuid.to_string());
        self.stats.gpu_queries.fetch_add(1, Ordering::Relaxed);
        self.gpu_states.get(&key).map(|entry| entry.clone())
    }
    
    /// Get all model states for a node
    pub fn get_node_model_states(&self, node_id: &NodeId) -> Vec<(String, TimestampedModelState)> {
        self.model_states
            .iter()
            .filter_map(|entry| {
                let ((node, model_key), state) = entry.pair();
                if node == node_id {
                    Some((model_key.clone(), state.clone()))
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Get all GPU states for a node
    pub fn get_node_gpu_states(&self, node_id: &NodeId) -> Vec<(String, TimestampedGpuState)> {
        self.gpu_states
            .iter()
            .filter_map(|entry| {
                let ((node, gpu_uuid), state) = entry.pair();
                if node == node_id {
                    Some((gpu_uuid.clone(), state.clone()))
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Find model states matching labels
    pub fn find_models(&self, labels: &Labels) -> Vec<(NodeId, String, TimestampedModelState)> {
        self.model_states
            .iter()
            .filter_map(|entry| {
                let ((node_id, model_key), state) = entry.pair();
                // Check if the state's labels match the search criteria
                let matches = state.state.labels.model == labels.model
                    && state.state.labels.revision == labels.revision
                    && state.state.labels.runtime == labels.runtime
                    && (labels.node.is_empty() || state.state.labels.node == labels.node)
                    && labels.custom.iter().all(|(key, value)| {
                        state.state.labels.custom.get(key) == Some(value)
                    });
                
                if matches {
                    Some((node_id.clone(), model_key.clone(), state.clone()))
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Find GPU states matching criteria
    pub fn find_gpus(&self, min_free_memory: Option<f64>) -> Vec<(NodeId, String, TimestampedGpuState)> {
        self.gpu_states
            .iter()
            .filter_map(|entry| {
                let ((node_id, gpu_uuid), state) = entry.pair();
                
                if let Some(min_mem) = min_free_memory {
                    // Calculate free memory from GPU state
                    let free_memory = state.state.vram_total_gb - state.state.vram_used_gb;
                    if free_memory < min_mem as f32 {
                        return None;
                    }
                }
                
                Some((node_id.clone(), gpu_uuid.clone(), state.clone()))
            })
            .collect()
    }
    
    /// Get all active nodes
    pub fn get_active_nodes(&self) -> Vec<NodeId> {
        let mut nodes = std::collections::HashSet::new();
        
        // Collect nodes from model states
        for entry in self.model_states.iter() {
            let ((node_id, _), _) = entry.pair();
            nodes.insert(node_id.clone());
        }
        
        // Collect nodes from GPU states
        for entry in self.gpu_states.iter() {
            let ((node_id, _), _) = entry.pair();
            nodes.insert(node_id.clone());
        }
        
        nodes.into_iter().collect()
    }
    
    /// Create a snapshot of the current state
    pub fn snapshot(&self) -> StateSnapshot {
        let models = self
            .model_states
            .iter()
            .map(|entry| {
                let ((node_id, model_key), state) = entry.pair();
                (node_id.clone(), model_key.clone(), state.clone())
            })
            .collect();
        
        let gpus = self
            .gpu_states
            .iter()
            .map(|entry| {
                let ((node_id, gpu_uuid), state) = entry.pair();
                (node_id.clone(), gpu_uuid.clone(), state.clone())
            })
            .collect();
        
        StateSnapshot {
            models,
            gpus,
            timestamp: Instant::now(),
        }
    }
    
    /// Remove states for a specific node
    pub async fn remove_node(&self, node_id: &NodeId) -> Result<usize> {
        let mut removed = 0;
        
        // Remove model states
        self.model_states.retain(|(node, _), _| {
            if node == node_id {
                removed += 1;
                false
            } else {
                true
            }
        });
        
        // Remove GPU states
        self.gpu_states.retain(|(node, _), _| {
            if node == node_id {
                removed += 1;
                false
            } else {
                true
            }
        });
        
        if removed > 0 {
            info!("Removed {} states for node {}", removed, node_id);
        }
        
        Ok(removed)
    }
    
    /// Get store statistics
    pub fn stats(&self) -> StoreStats {
        StoreStats {
            model_updates: AtomicU64::new(self.stats.model_updates.load(Ordering::Relaxed)),
            gpu_updates: AtomicU64::new(self.stats.gpu_updates.load(Ordering::Relaxed)),
            model_queries: AtomicU64::new(self.stats.model_queries.load(Ordering::Relaxed)),
            gpu_queries: AtomicU64::new(self.stats.gpu_queries.load(Ordering::Relaxed)),
            cleanup_runs: AtomicU64::new(self.stats.cleanup_runs.load(Ordering::Relaxed)),
            entries_cleaned: AtomicU64::new(self.stats.entries_cleaned.load(Ordering::Relaxed)),
        }
    }
    
    /// Get the number of model states
    pub fn model_count(&self) -> usize {
        self.model_states.len()
    }
    
    /// Get the number of GPU states
    pub fn gpu_count(&self) -> usize {
        self.gpu_states.len()
    }
    
    /// Check if cleanup should be triggered
    async fn maybe_cleanup(&self) {
        let last_cleanup = *self.last_cleanup.read().await;
        if last_cleanup.elapsed() >= self.config.cleanup_interval {
            if let Err(e) = self.cleanup().await {
                warn!("Cleanup failed: {}", e);
            }
        }
    }
    
    /// Clean up old state entries
    pub async fn cleanup(&self) -> Result<usize> {
        let mut last_cleanup = self.last_cleanup.write().await;
        *last_cleanup = Instant::now();
        drop(last_cleanup);
        
        let now = Instant::now();
        let max_age = self.config.max_state_age;
        let mut cleaned = 0;
        
        // Clean up model states
        self.model_states.retain(|_, state| {
            if now.duration_since(state.updated_at) > max_age {
                cleaned += 1;
                false
            } else {
                true
            }
        });
        
        // Clean up GPU states
        self.gpu_states.retain(|_, state| {
            if now.duration_since(state.updated_at) > max_age {
                cleaned += 1;
                false
            } else {
                true
            }
        });
        
        self.stats.cleanup_runs.fetch_add(1, Ordering::Relaxed);
        self.stats.entries_cleaned.fetch_add(cleaned as u64, Ordering::Relaxed);
        
        if cleaned > 0 {
            info!("Cleaned up {} old state entries", cleaned);
        }
        
        Ok(cleaned)
    }
    
    /// Clear all states
    pub fn clear(&self) {
        self.model_states.clear();
        self.gpu_states.clear();
        info!("Cleared all state entries");
    }
}

impl Default for StateStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_core::{GpuState, Labels, ModelState, NodeId};
    use std::time::Duration;

    #[tokio::test]
    async fn test_state_store_creation() {
        let store = StateStore::new();
        assert_eq!(store.model_count(), 0);
        assert_eq!(store.gpu_count(), 0);
    }

    #[tokio::test]
    async fn test_model_state_operations() {
        let store = StateStore::new();
        let node_id = NodeId::new("test-node");
        let model_key = "gpt-7b";
        
        let mut state = ModelState::new(
            Labels::new("gpt-7b", "v1", "runtime", "node1")
                .with_custom("model", "gpt-7b")
        );
        state.mark_loaded();
        
        // Update state
        store.update_model_state(node_id.clone(), model_key.to_string(), state.clone()).await.unwrap();
        assert_eq!(store.model_count(), 1);
        
        // Get state
        let retrieved = store.get_model_state(&node_id, model_key).unwrap();
        assert!(retrieved.state.loaded);
        assert_eq!(retrieved.version, 1);
        
        // Update again
        state.mark_unloaded();
        store.update_model_state(node_id.clone(), model_key.to_string(), state).await.unwrap();
        
        let updated = store.get_model_state(&node_id, model_key).unwrap();
        assert!(!updated.state.loaded);
        assert_eq!(updated.version, 2);
    }

    #[tokio::test]
    async fn test_gpu_state_operations() {
        let store = StateStore::new();
        let node_id = NodeId::new("test-node");
        let gpu_uuid = "GPU-12345";
        
        let mut state = GpuState::new(gpu_uuid, &node_id.to_string());
        state.update_metrics(0.5, 0.3, 8.0, 16.0);
        
        // Update state
        store.update_gpu_state(node_id.clone(), gpu_uuid.to_string(), state.clone()).await.unwrap();
        assert_eq!(store.gpu_count(), 1);
        
        // Get state
        let retrieved = store.get_gpu_state(&node_id, gpu_uuid).unwrap();
        assert_eq!(retrieved.state.sm_utilization, 0.5);
        assert_eq!(retrieved.version, 1);
    }

    #[tokio::test]
    async fn test_find_operations() {
        let store = StateStore::new();
        let node_id = NodeId::new("test-node");
        
        // Add model states
        let labels1 = Labels::new("gpt-7b", "v1", "runtime", "node1")
            .with_custom("model", "gpt-7b")
            .with_custom("size", "7b");
        let state1 = ModelState::new(labels1.clone());
        store.update_model_state(node_id.clone(), "model1".to_string(), state1).await.unwrap();
        
        let labels2 = Labels::new("gpt-13b", "v1", "runtime", "node1")
            .with_custom("model", "gpt-13b")
            .with_custom("size", "13b");
        let state2 = ModelState::new(labels2);
        store.update_model_state(node_id.clone(), "model2".to_string(), state2).await.unwrap();
        
        // Find by labels
        let search_labels = Labels::new("gpt-7b", "v1", "runtime", "node1")
            .with_custom("model", "gpt-7b");
        let results = store.find_models(&search_labels);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, "model1");
        
        // Add GPU state
        let mut gpu_state = GpuState::new("GPU-12345", &node_id.to_string());
        gpu_state.update_metrics(0.5, 0.3, 8.0, 16.0);
        store.update_gpu_state(node_id.clone(), "GPU-12345".to_string(), gpu_state).await.unwrap();
        
        // Find GPUs with free memory
        let gpu_results = store.find_gpus(Some(4.0)); // Need at least 4GB free
        assert_eq!(gpu_results.len(), 1);
    }

    #[tokio::test]
    async fn test_node_operations() {
        let store = StateStore::new();
        let node_id = NodeId::new("test-node");
        
        // Add states
        let state = ModelState::new(Labels::new("test", "v1", "runtime", "node1")
            .with_custom("model", "test"));
        store.update_model_state(node_id.clone(), "model1".to_string(), state).await.unwrap();
        
        let gpu_state = GpuState::new("GPU-12345", &node_id.to_string());
        store.update_gpu_state(node_id.clone(), "GPU-12345".to_string(), gpu_state).await.unwrap();
        
        // Get active nodes
        let nodes = store.get_active_nodes();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0], node_id);
        
        // Remove node
        let removed = store.remove_node(&node_id).await.unwrap();
        assert_eq!(removed, 2);
        assert_eq!(store.model_count(), 0);
        assert_eq!(store.gpu_count(), 0);
    }

    #[tokio::test]
    async fn test_cleanup() {
        let mut config = StateConfig::default();
        config.max_state_age = Duration::from_millis(10); // Very short age for testing
        
        let store = StateStore::with_config(config);
        let node_id = NodeId::new("test-node");
        
        // Add state
        let state = ModelState::new(Labels::new("test", "v1", "runtime", "node1")
            .with_custom("model", "test"));
        store.update_model_state(node_id, "model1".to_string(), state).await.unwrap();
        assert_eq!(store.model_count(), 1);
        
        // Wait for state to age
        tokio::time::sleep(Duration::from_millis(20)).await;
        
        // Cleanup
        let cleaned = store.cleanup().await.unwrap();
        assert_eq!(cleaned, 1);
        assert_eq!(store.model_count(), 0);
    }

    #[tokio::test]
    async fn test_capacity_limits() {
        let mut config = StateConfig::default();
        config.max_model_states = 2;
        
        let store = StateStore::with_config(config);
        let node_id = NodeId::new("test-node");
        
        // Add states up to capacity
        for i in 0..2 {
            let state = ModelState::new(Labels::new("test", "v1", "runtime", "node1")
                .with_custom("model", "test"));
            store.update_model_state(node_id.clone(), format!("model{}", i), state).await.unwrap();
        }
        
        // Try to add one more (should fail)
        let state = ModelState::new(Labels::new("test", "v1", "runtime", "node1")
            .with_custom("model", "test"));
        let result = store.update_model_state(node_id, "model3".to_string(), state).await;
        assert!(result.is_err());
    }
}
