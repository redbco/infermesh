//! State synchronization for distributed consistency

use crate::{config::SyncConfig, delta::StateDelta, store::StateStore, Result};
use mesh_core::NodeId;
use mesh_gossip::{GossipNode, protocol::ProtocolEvent};
use serde::{Deserialize, Serialize};
// use std::collections::HashMap; // Unused
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

/// State synchronizer for distributed consistency
// Note: GossipNode doesn't implement Debug, so we can't derive Debug
pub struct StateSynchronizer {
    config: SyncConfig,
    store: Arc<StateStore>,
    gossip_node: Option<Arc<GossipNode>>,
    sync_queue: Arc<RwLock<Vec<SyncOperation>>>,
    metrics: Arc<SyncMetrics>,
    event_tx: Option<mpsc::UnboundedSender<SyncEvent>>,
    running: Arc<RwLock<bool>>,
}

/// Synchronization operation
#[derive(Debug, Clone)]
// Note: Instant doesn't implement Serialize/Deserialize
pub struct SyncOperation {
    pub id: String,
    pub operation_type: SyncOperationType,
    pub created_at: Instant,
    pub retry_count: u32,
}

/// Type of synchronization operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncOperationType {
    /// Broadcast state delta to other nodes
    BroadcastDelta(StateDelta),
    /// Request full state from a node
    RequestState(NodeId),
    /// Send full state to a node
    SendState(NodeId),
    /// Reconcile state differences
    Reconcile(Vec<StateDelta>),
}

/// Synchronization event
#[derive(Debug, Clone)]
pub enum SyncEvent {
    /// State synchronized with peer
    StateSynchronized(NodeId),
    /// Synchronization failed
    SyncFailed(NodeId, String),
    /// Delta received from peer
    DeltaReceived(NodeId, StateDelta),
    /// Full state received from peer
    StateReceived(NodeId, usize),
}

/// Synchronization metrics
#[derive(Debug, Default)]
pub struct SyncMetrics {
    pub sync_operations: AtomicU64,
    pub deltas_sent: AtomicU64,
    pub deltas_received: AtomicU64,
    pub state_requests: AtomicU64,
    pub state_responses: AtomicU64,
    pub sync_failures: AtomicU64,
    pub reconciliations: AtomicU64,
}

impl StateSynchronizer {
    /// Create a new state synchronizer
    pub fn new(store: StateStore) -> Self {
        Self::with_config(store, SyncConfig::default())
    }
    
    /// Create a new state synchronizer with configuration
    pub fn with_config(store: StateStore, config: SyncConfig) -> Self {
        Self {
            config,
            store: Arc::new(store),
            gossip_node: None,
            sync_queue: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(SyncMetrics::default()),
            event_tx: None,
            running: Arc::new(RwLock::new(false)),
        }
    }
    
    /// Set the gossip node for communication
    pub fn set_gossip_node(&mut self, gossip_node: Arc<GossipNode>) {
        self.gossip_node = Some(gossip_node);
    }
    
    /// Set event sender for notifications
    pub fn set_event_sender(&mut self, tx: mpsc::UnboundedSender<SyncEvent>) {
        self.event_tx = Some(tx);
    }
    
    /// Start the synchronizer
    pub async fn start(&mut self) -> Result<()> {
        if !self.config.enabled {
            info!("State synchronization is disabled");
            return Ok(());
        }
        
        let mut running = self.running.write().await;
        if *running {
            return Ok(());
        }
        
        *running = true;
        drop(running);
        
        info!("Starting state synchronizer");
        
        // Start sync loop
        self.start_sync_loop().await;
        
        Ok(())
    }
    
    /// Stop the synchronizer
    pub async fn stop(&mut self) -> Result<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Ok(());
        }
        
        *running = false;
        info!("Stopping state synchronizer");
        
        Ok(())
    }
    
    /// Broadcast a state delta to other nodes
    pub async fn broadcast_delta(&self, delta: StateDelta) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        
        let operation = SyncOperation {
            id: uuid::Uuid::new_v4().to_string(),
            operation_type: SyncOperationType::BroadcastDelta(delta),
            created_at: Instant::now(),
            retry_count: 0,
        };
        
        self.queue_sync_operation(operation).await;
        Ok(())
    }
    
    /// Request full state from a specific node
    pub async fn request_state(&self, node_id: NodeId) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        
        let operation = SyncOperation {
            id: uuid::Uuid::new_v4().to_string(),
            operation_type: SyncOperationType::RequestState(node_id),
            created_at: Instant::now(),
            retry_count: 0,
        };
        
        self.queue_sync_operation(operation).await;
        self.metrics.state_requests.fetch_add(1, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Handle incoming gossip protocol events
    pub async fn handle_gossip_event(&self, event: ProtocolEvent) -> Result<()> {
        match event {
            ProtocolEvent::MemberJoined(node_id) => {
                info!("New member joined: {}, requesting state sync", node_id);
                self.request_state(node_id).await?;
            }
            ProtocolEvent::MemberLeft(node_id) => {
                info!("Member left: {}, cleaning up state", node_id);
                self.store.remove_node(&node_id).await?;
            }
            ProtocolEvent::MemberDead(node_id) => {
                warn!("Member dead: {}, cleaning up state", node_id);
                self.store.remove_node(&node_id).await?;
            }
            ProtocolEvent::UserData { key, value, source, .. } => {
                if key.starts_with("state:") {
                    self.handle_state_data(source, value).await?;
                }
            }
            _ => {
                // Ignore other events
            }
        }
        
        Ok(())
    }
    
    /// Handle state data received via gossip
    async fn handle_state_data(&self, source: NodeId, data: Vec<u8>) -> Result<()> {
        match serde_json::from_slice::<StateDelta>(&data) {
            Ok(delta) => {
                debug!("Received state delta from {}", source);
                self.metrics.deltas_received.fetch_add(1, Ordering::Relaxed);
                
                if let Some(tx) = &self.event_tx {
                    let _ = tx.send(SyncEvent::DeltaReceived(source, delta));
                }
            }
            Err(e) => {
                warn!("Failed to deserialize state data from {}: {}", source, e);
            }
        }
        
        Ok(())
    }
    
    /// Start the synchronization loop
    async fn start_sync_loop(&self) {
        let running = self.running.clone();
        let sync_queue = self.sync_queue.clone();
        let config = self.config.clone();
        let metrics = self.metrics.clone();
        let gossip_node = self.gossip_node.clone();
        let event_tx = self.event_tx.clone();
        
        tokio::spawn(async move {
            let mut sync_interval = interval(config.sync_interval);
            
            while *running.read().await {
                tokio::select! {
                    _ = sync_interval.tick() => {
                        Self::process_sync_queue(
                            &sync_queue,
                            &config,
                            &metrics,
                            &gossip_node,
                            &event_tx,
                        ).await;
                    }
                }
            }
        });
    }
    
    /// Process the synchronization queue
    async fn process_sync_queue(
        sync_queue: &Arc<RwLock<Vec<SyncOperation>>>,
        config: &SyncConfig,
        metrics: &Arc<SyncMetrics>,
        gossip_node: &Option<Arc<GossipNode>>,
        event_tx: &Option<mpsc::UnboundedSender<SyncEvent>>,
    ) {
        let mut queue = sync_queue.write().await;
        let mut operations_to_retry = Vec::new();
        
        // Process operations in batches
        let batch_size = config.batch_size.min(queue.len());
        let operations: Vec<SyncOperation> = queue.drain(..batch_size).collect();
        drop(queue);
        
        for operation in operations {
            match Self::execute_sync_operation(
                &operation,
                config,
                metrics,
                gossip_node,
                event_tx,
            ).await {
                Ok(()) => {
                    metrics.sync_operations.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => {
                    error!("Sync operation failed: {}", e);
                    metrics.sync_failures.fetch_add(1, Ordering::Relaxed);
                    
                    // Retry if not exceeded max retries
                    if operation.retry_count < 3 {
                        let mut retry_op = operation;
                        retry_op.retry_count += 1;
                        operations_to_retry.push(retry_op);
                    }
                }
            }
        }
        
        // Re-queue operations for retry
        if !operations_to_retry.is_empty() {
            let mut queue = sync_queue.write().await;
            queue.extend(operations_to_retry);
        }
    }
    
    /// Execute a single synchronization operation
    async fn execute_sync_operation(
        operation: &SyncOperation,
        _config: &SyncConfig,
        metrics: &Arc<SyncMetrics>,
        gossip_node: &Option<Arc<GossipNode>>,
        _event_tx: &Option<mpsc::UnboundedSender<SyncEvent>>,
    ) -> Result<()> {
        match &operation.operation_type {
            SyncOperationType::BroadcastDelta(delta) => {
                if let Some(gossip) = gossip_node {
                    let data = serde_json::to_vec(delta)?;
                    let key = format!("state:delta:{}", uuid::Uuid::new_v4());
                    gossip.add_user_data(key, data, 1).await?;
                    metrics.deltas_sent.fetch_add(1, Ordering::Relaxed);
                }
            }
            SyncOperationType::RequestState(node_id) => {
                if let Some(gossip) = gossip_node {
                    let request_data = serde_json::to_vec(&format!("request_state:{}", node_id))?;
                    let key = format!("state:request:{}", uuid::Uuid::new_v4());
                    gossip.add_user_data(key, request_data, 1).await?;
                }
            }
            SyncOperationType::SendState(node_id) => {
                // This would involve sending full state to a specific node
                // Implementation depends on the gossip protocol capabilities
                debug!("Sending state to node: {}", node_id);
                metrics.state_responses.fetch_add(1, Ordering::Relaxed);
            }
            SyncOperationType::Reconcile(deltas) => {
                debug!("Reconciling {} deltas", deltas.len());
                metrics.reconciliations.fetch_add(1, Ordering::Relaxed);
                
                // Broadcast reconciliation deltas
                for delta in deltas {
                    if let Some(gossip) = gossip_node {
                        let data = serde_json::to_vec(delta)?;
                        let key = format!("state:reconcile:{}", uuid::Uuid::new_v4());
                        gossip.add_user_data(key, data, 1).await?;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Queue a synchronization operation
    async fn queue_sync_operation(&self, operation: SyncOperation) {
        let mut queue = self.sync_queue.write().await;
        
        // Limit queue size
        if queue.len() >= self.config.max_pending_ops {
            // Remove oldest operations
            let remove_count = queue.len() - self.config.max_pending_ops + 1;
            queue.drain(..remove_count);
            warn!("Sync queue full, removed {} old operations", remove_count);
        }
        
        queue.push(operation);
    }
    
    /// Get synchronization metrics
    pub fn metrics(&self) -> &SyncMetrics {
        &self.metrics
    }
    
    /// Check if synchronizer is running
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }
    
    /// Get pending sync operations count
    pub async fn pending_operations_count(&self) -> usize {
        self.sync_queue.read().await.len()
    }
    
    /// Clear all pending sync operations
    pub async fn clear_pending_operations(&self) {
        let mut queue = self.sync_queue.write().await;
        queue.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::DeltaApplier;
    use mesh_core::{Labels, ModelStateDelta, NodeId};

    #[tokio::test]
    async fn test_synchronizer_creation() {
        let store = StateStore::new();
        let sync = StateSynchronizer::new(store);
        
        assert!(!sync.is_running().await);
        assert_eq!(sync.pending_operations_count().await, 0);
    }

    #[tokio::test]
    async fn test_broadcast_delta() {
        let store = StateStore::new();
        let sync = StateSynchronizer::new(store);
        
        // Create a delta
        let labels = Labels::new("test-model", "v1", "runtime", "test-node");
        let mut model_delta = ModelStateDelta::new(labels);
        model_delta.loaded = Some(true);
        
        let delta = DeltaApplier::create_model_delta(
            NodeId::new("test-node"),
            "test-model".to_string(),
            model_delta,
        );
        
        // Broadcast delta
        sync.broadcast_delta(delta).await.unwrap();
        
        // Check that operation was queued
        assert_eq!(sync.pending_operations_count().await, 1);
    }

    #[tokio::test]
    async fn test_request_state() {
        let store = StateStore::new();
        let sync = StateSynchronizer::new(store);
        let node_id = NodeId::new("test-node");
        
        // Request state
        sync.request_state(node_id).await.unwrap();
        
        // Check metrics
        assert_eq!(sync.metrics().state_requests.load(Ordering::Relaxed), 1);
        assert_eq!(sync.pending_operations_count().await, 1);
    }

    #[tokio::test]
    async fn test_disabled_synchronization() {
        let store = StateStore::new();
        let mut config = SyncConfig::default();
        config.enabled = false;
        
        let sync = StateSynchronizer::with_config(store, config);
        
        // Operations should not be queued when disabled
        let delta = DeltaApplier::create_model_delta(
            NodeId::new("test-node"),
            "test-model".to_string(),
            ModelStateDelta::new(Labels::new("test-model", "v1", "runtime", "test-node")),
        );
        
        sync.broadcast_delta(delta).await.unwrap();
        assert_eq!(sync.pending_operations_count().await, 0);
    }

    #[tokio::test]
    async fn test_queue_size_limit() {
        let store = StateStore::new();
        let mut config = SyncConfig::default();
        config.max_pending_ops = 2; // Very small limit
        
        let sync = StateSynchronizer::with_config(store, config);
        
        // Add operations beyond limit
        for i in 0..5 {
            let delta = DeltaApplier::create_model_delta(
                NodeId::new("test-node"),
                format!("model-{}", i),
                ModelStateDelta::new(Labels::new("test-model", "v1", "runtime", "test-node")),
            );
            sync.broadcast_delta(delta).await.unwrap();
        }
        
        // Should be limited to max_pending_ops
        assert_eq!(sync.pending_operations_count().await, 2);
    }

    #[tokio::test]
    async fn test_clear_pending_operations() {
        let store = StateStore::new();
        let sync = StateSynchronizer::new(store);
        
        // Add some operations
        let delta = DeltaApplier::create_model_delta(
            NodeId::new("test-node"),
            "test-model".to_string(),
            ModelStateDelta::new(Labels::new("test-model", "v1", "runtime", "test-node")),
        );
        
        sync.broadcast_delta(delta).await.unwrap();
        assert_eq!(sync.pending_operations_count().await, 1);
        
        // Clear operations
        sync.clear_pending_operations().await;
        assert_eq!(sync.pending_operations_count().await, 0);
    }
}
