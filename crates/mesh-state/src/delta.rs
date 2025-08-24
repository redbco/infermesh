//! Delta application for state updates

use crate::{store::StateStore, Result, StateError};
use mesh_core::{GpuStateDelta, ModelStateDelta, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Delta applier for state updates
#[derive(Debug)]
pub struct DeltaApplier {
    store: Arc<StateStore>,
    metrics: Arc<DeltaMetrics>,
    pending_deltas: Arc<RwLock<HashMap<String, PendingDelta>>>,
}

/// A state delta that can be applied
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateDelta {
    /// Model state delta
    Model {
        node_id: NodeId,
        model_key: String,
        delta: ModelStateDelta,
    },
    /// GPU state delta
    Gpu {
        node_id: NodeId,
        gpu_uuid: String,
        delta: GpuStateDelta,
    },
    /// Batch of deltas
    Batch(Vec<StateDelta>),
}

/// Pending delta waiting to be applied
#[derive(Debug, Clone)]
pub struct PendingDelta {
    delta: StateDelta,
    received_at: std::time::Instant,
    retry_count: u32,
}

/// Metrics for delta operations
#[derive(Debug, Default)]
pub struct DeltaMetrics {
    pub deltas_received: AtomicU64,
    pub deltas_applied: AtomicU64,
    pub deltas_failed: AtomicU64,
    pub deltas_retried: AtomicU64,
    pub model_deltas: AtomicU64,
    pub gpu_deltas: AtomicU64,
    pub batch_deltas: AtomicU64,
}

impl DeltaApplier {
    /// Create a new delta applier
    pub fn new(store: StateStore) -> Self {
        Self {
            store: Arc::new(store),
            metrics: Arc::new(DeltaMetrics::default()),
            pending_deltas: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Apply a single delta
    pub async fn apply_delta(&self, delta: StateDelta) -> Result<()> {
        self.metrics.deltas_received.fetch_add(1, Ordering::Relaxed);
        
        match self.apply_delta_internal(&delta).await {
            Ok(()) => {
                self.metrics.deltas_applied.fetch_add(1, Ordering::Relaxed);
                debug!("Successfully applied delta: {:?}", delta);
                Ok(())
            }
            Err(e) => {
                self.metrics.deltas_failed.fetch_add(1, Ordering::Relaxed);
                warn!("Failed to apply delta: {:?}, error: {}", delta, e);
                Err(e)
            }
        }
    }
    
    /// Apply multiple deltas in batch
    pub async fn apply_deltas(&self, deltas: Vec<StateDelta>) -> Result<Vec<Result<()>>> {
        let mut results = Vec::with_capacity(deltas.len());
        
        for delta in deltas {
            results.push(self.apply_delta(delta).await);
        }
        
        Ok(results)
    }
    
    /// Apply a delta with retry logic
    pub async fn apply_delta_with_retry(&self, delta: StateDelta, max_retries: u32) -> Result<()> {
        let delta_id = self.generate_delta_id(&delta);
        
        for attempt in 0..=max_retries {
            match self.apply_delta_internal(&delta).await {
                Ok(()) => {
                    self.metrics.deltas_applied.fetch_add(1, Ordering::Relaxed);
                    
                    // Remove from pending if it was there
                    let mut pending = self.pending_deltas.write().await;
                    pending.remove(&delta_id);
                    
                    return Ok(());
                }
                Err(e) => {
                    if attempt < max_retries {
                        self.metrics.deltas_retried.fetch_add(1, Ordering::Relaxed);
                        warn!("Delta application failed (attempt {}), retrying: {}", attempt + 1, e);
                        
                        // Add to pending deltas
                        let mut pending = self.pending_deltas.write().await;
                        pending.insert(delta_id.clone(), PendingDelta {
                            delta: delta.clone(),
                            received_at: std::time::Instant::now(),
                            retry_count: attempt + 1,
                        });
                        
                        // Wait before retry
                        let delay = std::time::Duration::from_millis(100 * (1 << attempt));
                        tokio::time::sleep(delay).await;
                    } else {
                        self.metrics.deltas_failed.fetch_add(1, Ordering::Relaxed);
                        return Err(e);
                    }
                }
            }
        }
        
        unreachable!()
    }
    
    /// Internal delta application logic
    async fn apply_delta_internal(&self, delta: &StateDelta) -> Result<()> {
        match delta {
            StateDelta::Model { node_id, model_key, delta } => {
                self.metrics.model_deltas.fetch_add(1, Ordering::Relaxed);
                self.apply_model_delta(node_id, model_key, delta).await
            }
            StateDelta::Gpu { node_id, gpu_uuid, delta } => {
                self.metrics.gpu_deltas.fetch_add(1, Ordering::Relaxed);
                self.apply_gpu_delta(node_id, gpu_uuid, delta).await
            }
            StateDelta::Batch(deltas) => {
                self.metrics.batch_deltas.fetch_add(1, Ordering::Relaxed);
                self.apply_batch_deltas(deltas).await
            }
        }
    }
    
    /// Apply a model state delta
    async fn apply_model_delta(
        &self,
        node_id: &NodeId,
        model_key: &str,
        delta: &ModelStateDelta,
    ) -> Result<()> {
        // Get current state or create new one
        let mut current_state = self.store
            .get_model_state(node_id, model_key)
            .map(|ts| ts.state)
            .unwrap_or_else(|| {
                // Create a default model state using the delta's labels
                mesh_core::ModelState::new(delta.labels.clone())
            });
        
        // Apply delta using the built-in apply_to method
        delta.apply_to(&mut current_state);
        
        // Update store
        self.store.update_model_state(
            node_id.clone(),
            model_key.to_string(),
            current_state,
        ).await?;
        
        Ok(())
    }
    
    /// Apply a GPU state delta
    async fn apply_gpu_delta(
        &self,
        node_id: &NodeId,
        gpu_uuid: &str,
        delta: &GpuStateDelta,
    ) -> Result<()> {
        // Get current state or create new one
        let mut current_state = self.store
            .get_gpu_state(node_id, gpu_uuid)
            .map(|ts| ts.state)
            .unwrap_or_else(|| {
                mesh_core::GpuState::new(gpu_uuid, &node_id.to_string())
            });
        
        // Apply delta using the built-in apply_to method
        delta.apply_to(&mut current_state);
        
        // Update store
        self.store.update_gpu_state(
            node_id.clone(),
            gpu_uuid.to_string(),
            current_state,
        ).await?;
        
        Ok(())
    }
    
    /// Apply a batch of deltas
    async fn apply_batch_deltas(&self, deltas: &[StateDelta]) -> Result<()> {
        let mut errors = Vec::new();
        
        for delta in deltas {
            // Handle recursion by directly applying the delta types instead of calling apply_delta_internal
            let result = match delta {
                StateDelta::Model { node_id, model_key, delta } => {
                    self.apply_model_delta(node_id, model_key, delta).await
                }
                StateDelta::Gpu { node_id, gpu_uuid, delta } => {
                    self.apply_gpu_delta(node_id, gpu_uuid, delta).await
                }
                StateDelta::Batch(nested_deltas) => {
                    // For nested batches, apply each delta individually to avoid recursion
                    let mut nested_errors = Vec::new();
                    for nested_delta in nested_deltas {
                        match nested_delta {
                            StateDelta::Model { node_id, model_key, delta } => {
                                if let Err(e) = self.apply_model_delta(node_id, model_key, delta).await {
                                    nested_errors.push(e);
                                }
                            }
                            StateDelta::Gpu { node_id, gpu_uuid, delta } => {
                                if let Err(e) = self.apply_gpu_delta(node_id, gpu_uuid, delta).await {
                                    nested_errors.push(e);
                                }
                            }
                            StateDelta::Batch(_) => {
                                // Don't support deeply nested batches to avoid complexity
                                nested_errors.push(StateError::Delta("Nested batches not supported".to_string()));
                            }
                        }
                    }
                    if nested_errors.is_empty() {
                        Ok(())
                    } else {
                        Err(StateError::Delta(format!(
                            "Nested batch failed with {} errors: {:?}",
                            nested_errors.len(),
                            nested_errors
                        )))
                    }
                }
            };
            
            if let Err(e) = result {
                errors.push(e);
            }
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(StateError::Delta(format!(
                "Batch application failed with {} errors: {:?}",
                errors.len(),
                errors
            )))
        }
    }
    
    /// Generate a unique ID for a delta
    fn generate_delta_id(&self, delta: &StateDelta) -> String {
        match delta {
            StateDelta::Model { node_id, model_key, .. } => {
                format!("model:{}:{}", node_id, model_key)
            }
            StateDelta::Gpu { node_id, gpu_uuid, .. } => {
                format!("gpu:{}:{}", node_id, gpu_uuid)
            }
            StateDelta::Batch(deltas) => {
                format!("batch:{}", deltas.len())
            }
        }
    }
    
    /// Get pending deltas
    pub async fn get_pending_deltas(&self) -> HashMap<String, PendingDelta> {
        self.pending_deltas.read().await.clone()
    }
    
    /// Clear pending deltas
    pub async fn clear_pending_deltas(&self) {
        let mut pending = self.pending_deltas.write().await;
        pending.clear();
    }
    
    /// Get delta metrics
    pub fn metrics(&self) -> &DeltaMetrics {
        &self.metrics
    }
    
    /// Create a model delta
    pub fn create_model_delta(
        node_id: NodeId,
        model_key: String,
        delta: ModelStateDelta,
    ) -> StateDelta {
        StateDelta::Model {
            node_id,
            model_key,
            delta,
        }
    }
    
    /// Create a GPU delta
    pub fn create_gpu_delta(
        node_id: NodeId,
        gpu_uuid: String,
        delta: GpuStateDelta,
    ) -> StateDelta {
        StateDelta::Gpu {
            node_id,
            gpu_uuid,
            delta,
        }
    }
    
    /// Create a batch delta
    pub fn create_batch_delta(deltas: Vec<StateDelta>) -> StateDelta {
        StateDelta::Batch(deltas)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_core::{GpuStateDelta, Labels, ModelStateDelta, NodeId};

    #[tokio::test]
    async fn test_delta_applier_creation() {
        let store = StateStore::new();
        let applier = DeltaApplier::new(store);
        
        let metrics = applier.metrics();
        assert_eq!(metrics.deltas_received.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.deltas_applied.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_apply_model_delta() {
        let store = StateStore::new();
        let applier = DeltaApplier::new(store.clone());
        let node_id = NodeId::new("test-node");
        
        // Create labels for the delta
        let labels = Labels::new("gpt-7b", "v1", "runtime", "node1")
            .with_custom("model", "gpt-7b");
        // Don't create initial state - let the delta applier create it
        
        // Create delta
        let mut delta = ModelStateDelta::new(labels.clone());
        delta.loaded = Some(true);
        delta.queue_depth = Some(5);
        
        let state_delta = DeltaApplier::create_model_delta(
            node_id.clone(),
            "gpt-7b".to_string(),
            delta,
        );
        
        // Apply delta
        applier.apply_delta(state_delta).await.unwrap();
        
        // Verify state was updated
        let updated_state = store.get_model_state(&node_id, "gpt-7b").unwrap();
        assert!(updated_state.state.loaded);
        assert_eq!(updated_state.state.queue_depth, 5);
        
        // Check metrics
        assert_eq!(applier.metrics().deltas_applied.load(Ordering::Relaxed), 1);
        assert_eq!(applier.metrics().model_deltas.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_basic_store_operations() {
        let store = StateStore::new();
        let node_id = NodeId::new("test-node");
        
        // Test basic GPU state storage and retrieval
        let mut gpu_state = mesh_core::GpuState::new("GPU-12345", &node_id.to_string());
        gpu_state.sm_utilization = 0.5;
        
        store.update_gpu_state(node_id.clone(), "GPU-12345".to_string(), gpu_state).await.unwrap();
        
        let retrieved = store.get_gpu_state(&node_id, "GPU-12345");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().state.sm_utilization, 0.5);
    }

    #[tokio::test]
    async fn test_store_clone_sharing() {
        let store1 = StateStore::new();
        let store2 = store1.clone();
        let node_id = NodeId::new("test-node");
        
        // Store in store1
        let labels = Labels::new("test", "v1", "runtime", "node1");
        let model_state = mesh_core::ModelState::new(labels);
        store1.update_model_state(node_id.clone(), "test-model".to_string(), model_state).await.unwrap();
        
        // Retrieve from store2
        let retrieved = store2.get_model_state(&node_id, "test-model");
        assert!(retrieved.is_some(), "Store clone should share the same data");
        
        println!("Store1 model count: {}", store1.model_count());
        println!("Store2 model count: {}", store2.model_count());
    }

    #[tokio::test]
    async fn test_apply_gpu_delta() {
        let store = StateStore::new();
        let applier = DeltaApplier::new(store.clone());
        let node_id = NodeId::new("test-node");
        
        // Don't create initial state - let the delta applier create it
        
        // Create delta
        let mut delta = GpuStateDelta::new("GPU-12345".to_string(), node_id.to_string());
        delta.sm_utilization = Some(0.8);
        delta.memory_utilization = Some(0.6);
        delta.temperature_c = Some(75.0);
        
        let state_delta = DeltaApplier::create_gpu_delta(
            node_id.clone(),
            "GPU-12345".to_string(),
            delta,
        );
        
        // Apply delta
        applier.apply_delta(state_delta).await.unwrap();
        
        // Verify state was updated
        let updated_state = store.get_gpu_state(&node_id, "GPU-12345").unwrap();
        assert_eq!(updated_state.state.sm_utilization, 0.8);
        assert_eq!(updated_state.state.memory_utilization, 0.6);
        assert_eq!(updated_state.state.temperature_c, Some(75.0));
        
        // Check metrics
        assert_eq!(applier.metrics().deltas_applied.load(Ordering::Relaxed), 1);
        assert_eq!(applier.metrics().gpu_deltas.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_apply_batch_delta() {
        let store = StateStore::new();
        let applier = DeltaApplier::new(store.clone());
        let node_id = NodeId::new("test-node");
        
        // Create model delta
        let labels = Labels::new("gpt-7b", "v1", "runtime", "node1");
        let mut model_delta = ModelStateDelta::new(labels);
        model_delta.loaded = Some(true);
        let model_state_delta = DeltaApplier::create_model_delta(
            node_id.clone(),
            "gpt-7b".to_string(),
            model_delta,
        );
        
        // Create GPU delta
        let mut gpu_delta = GpuStateDelta::new("GPU-12345".to_string(), node_id.to_string());
        gpu_delta.sm_utilization = Some(0.5);
        let gpu_state_delta = DeltaApplier::create_gpu_delta(
            node_id.clone(),
            "GPU-12345".to_string(),
            gpu_delta,
        );
        
        // Create batch delta
        let batch_delta = DeltaApplier::create_batch_delta(vec![
            model_state_delta,
            gpu_state_delta,
        ]);
        
        // Apply batch delta
        applier.apply_delta(batch_delta).await.unwrap();
        
        // Verify both states were updated
        let model_state = store.get_model_state(&node_id, "gpt-7b").unwrap();
        assert!(model_state.state.loaded);
        
        let gpu_state = store.get_gpu_state(&node_id, "GPU-12345").unwrap();
        assert_eq!(gpu_state.state.sm_utilization, 0.5);
        
        // Check metrics
        assert_eq!(applier.metrics().deltas_applied.load(Ordering::Relaxed), 1);
        assert_eq!(applier.metrics().batch_deltas.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_apply_multiple_deltas() {
        let store = StateStore::new();
        let applier = DeltaApplier::new(store.clone());
        let node_id = NodeId::new("test-node");
        
        // Create multiple deltas
        let labels1 = Labels::new("model1", "v1", "runtime", "node1");
        let mut delta1 = ModelStateDelta::new(labels1);
        delta1.loaded = Some(true);
        
        let labels2 = Labels::new("model2", "v1", "runtime", "node1");
        let mut delta2 = ModelStateDelta::new(labels2);
        delta2.loaded = Some(false);
        
        let deltas = vec![
            DeltaApplier::create_model_delta(
                node_id.clone(),
                "model1".to_string(),
                delta1,
            ),
            DeltaApplier::create_model_delta(
                node_id.clone(),
                "model2".to_string(),
                delta2,
            ),
        ];
        
        // Apply deltas
        let results = applier.apply_deltas(deltas).await.unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
        
        // Check metrics
        assert_eq!(applier.metrics().deltas_applied.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn test_delta_retry_logic() {
        let store = StateStore::new();
        let applier = DeltaApplier::new(store);
        let node_id = NodeId::new("test-node");
        
        // Create a valid delta
        let labels = Labels::new("test-model", "v1", "runtime", "node1");
        let mut model_delta = ModelStateDelta::new(labels);
        model_delta.loaded = Some(true);
        
        let delta = DeltaApplier::create_model_delta(
            node_id,
            "test-model".to_string(),
            model_delta,
        );
        
        // Apply with retry (should succeed on first try)
        applier.apply_delta_with_retry(delta, 3).await.unwrap();
        
        // Check metrics
        assert_eq!(applier.metrics().deltas_applied.load(Ordering::Relaxed), 1);
        assert_eq!(applier.metrics().deltas_retried.load(Ordering::Relaxed), 0);
    }
}
