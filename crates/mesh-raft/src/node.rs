//! Raft node implementation

use crate::config::{RaftConfig, validate_config};
use crate::policy::Policy;
use crate::state_machine::{PolicyStateMachine, OperationResult};
use crate::storage::{RaftStorage, RaftStorageBackend, MemoryStorage, DiskStorage};
use crate::config::StorageType;
use crate::state_machine::PolicyOperation;
use crate::{Result, RaftError};

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use raft::{
    Config, RawNode, Ready, StateRole,
    eraftpb::{Entry, EntryType, Message},
};
use tokio::sync::{mpsc, RwLock, Mutex};
use tokio::time::{interval, sleep};
use tracing::{debug, error, info, warn};

/// Raft node for distributed policy management
pub struct RaftNode {
    /// Node configuration
    config: RaftConfig,
    
    /// Raw raft node
    raw_node: Arc<Mutex<RawNode<RaftStorageBackend>>>,
    
    /// Policy state machine
    state_machine: Arc<RwLock<PolicyStateMachine>>,
    
    /// Node statistics
    stats: Arc<RaftNodeStats>,
    
    /// Message sender for raft messages
    message_tx: mpsc::UnboundedSender<Message>,
    
    /// Message receiver for raft messages
    message_rx: Arc<Mutex<mpsc::UnboundedReceiver<Message>>>,
    
    /// Proposal sender for client requests
    proposal_tx: mpsc::UnboundedSender<ProposalRequest>,
    
    /// Proposal receiver for client requests
    proposal_rx: Arc<Mutex<mpsc::UnboundedReceiver<ProposalRequest>>>,
    
    /// Running flag
    running: Arc<std::sync::atomic::AtomicBool>,
    
    /// Task handles
    task_handles: Arc<RwLock<Vec<tokio::task::JoinHandle<()>>>>,
}

/// Raft node statistics
#[derive(Debug)]
pub struct RaftNodeStats {
    /// Node ID
    pub node_id: u64,
    
    /// Current term
    pub term: AtomicU64,
    
    /// Current role
    pub role: Arc<RwLock<StateRole>>,
    
    /// Leader ID
    pub leader_id: AtomicU64,
    
    /// Applied index
    pub applied_index: AtomicU64,
    
    /// Committed index
    pub committed_index: AtomicU64,
    
    /// Number of proposals
    pub proposals_total: AtomicU64,
    
    /// Number of successful proposals
    pub proposals_success: AtomicU64,
    
    /// Number of failed proposals
    pub proposals_failed: AtomicU64,
    
    /// Number of messages sent
    pub messages_sent: AtomicU64,
    
    /// Number of messages received
    pub messages_received: AtomicU64,
    
    /// Start time
    pub start_time: Instant,
}

impl Default for RaftNodeStats {
    fn default() -> Self {
        Self {
            node_id: 0,
            term: AtomicU64::new(0),
            role: Arc::new(RwLock::new(StateRole::Follower)),
            leader_id: AtomicU64::new(0),
            committed_index: AtomicU64::new(0),
            applied_index: AtomicU64::new(0),
            proposals_total: AtomicU64::new(0),
            proposals_success: AtomicU64::new(0),
            proposals_failed: AtomicU64::new(0),
            messages_sent: AtomicU64::new(0),
            messages_received: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }
}

/// Proposal request from client
#[derive(Debug)]
struct ProposalRequest {
    /// Policy operation to propose
    operation: PolicyOperation,
    
    /// Response sender
    response_tx: tokio::sync::oneshot::Sender<Result<OperationResult>>,
}

/// Raft message wrapper
#[derive(Debug, Clone)]
pub struct RaftMessage {
    /// Source node ID
    pub from: u64,
    
    /// Target node ID
    pub to: u64,
    
    /// Raft message
    pub message: Message,
}

impl RaftNode {
    /// Create a new raft node
    pub async fn new(node_id: u64, mut config: RaftConfig) -> Result<Self> {
        config.node_id = node_id;
        validate_config(&config).map_err(RaftError::Configuration)?;
        
        info!("Creating raft node {} with {} peers", node_id, config.peers.len());
        
        // Create storage
        let storage = match config.storage.storage_type {
            StorageType::Memory => {
                let mut mem_storage = MemoryStorage::new();
                mem_storage.initialize()?;
                RaftStorageBackend::Memory(mem_storage)
            }
            StorageType::Disk => {
                let data_dir = config.storage.data_dir.as_ref()
                    .ok_or_else(|| RaftError::Configuration("Data directory required for disk storage".to_string()))?;
                let mut disk_storage = DiskStorage::new(
                    data_dir.clone(),
                    config.storage.sync_writes,
                    config.storage.compress_entries,
                    config.storage.max_memory_entries,
                );
                disk_storage.initialize()?;
                RaftStorageBackend::Disk(disk_storage)
            }
        };
        
        // Create raft config
        let raft_config = Config {
            id: node_id,
            election_tick: (config.election_timeout.as_millis() / config.heartbeat_interval.as_millis()) as usize,
            heartbeat_tick: 1,
            max_size_per_msg: 1024 * 1024, // 1MB
            max_inflight_msgs: 256,
            applied: 0,
            check_quorum: config.check_quorum,
            pre_vote: config.pre_vote,
            ..Default::default()
        };
        
        // Create raw node with logger
        let logger = slog::Logger::root(slog::Discard, slog::o!());
        let raw_node = RawNode::new(&raft_config, storage, &logger)
            .map_err(RaftError::Raft)?;
        
        // Create channels
        let (message_tx, message_rx) = mpsc::unbounded_channel();
        let (proposal_tx, proposal_rx) = mpsc::unbounded_channel();
        
        // Create statistics
        let stats = Arc::new(RaftNodeStats {
            node_id,
            start_time: Instant::now(),
            ..Default::default()
        });
        
        Ok(Self {
            config,
            raw_node: Arc::new(Mutex::new(raw_node)),
            state_machine: Arc::new(RwLock::new(PolicyStateMachine::new())),
            stats,
            message_tx,
            message_rx: Arc::new(Mutex::new(message_rx)),
            proposal_tx,
            proposal_rx: Arc::new(Mutex::new(proposal_rx)),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            task_handles: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Start the raft node
    pub async fn start(&self) -> Result<()> {
        if self.running.load(Ordering::Relaxed) {
            return Err(RaftError::Node("Node is already running".to_string()));
        }
        
        info!("Starting raft node {}", self.config.node_id);
        self.running.store(true, Ordering::Relaxed);
        
        let mut handles = self.task_handles.write().await;
        
        // Start main raft loop
        let main_handle = {
            let node = self.clone();
            tokio::spawn(async move {
                if let Err(e) = node.run_main_loop().await {
                    error!("Main raft loop error: {}", e);
                }
            })
        };
        handles.push(main_handle);
        
        // Start tick timer
        let tick_handle = {
            let node = self.clone();
            tokio::spawn(async move {
                node.run_tick_loop().await;
            })
        };
        handles.push(tick_handle);
        
        // Start proposal handler
        let proposal_handle = {
            let node = self.clone();
            tokio::spawn(async move {
                node.run_proposal_loop().await;
            })
        };
        handles.push(proposal_handle);
        
        info!("Raft node {} started successfully", self.config.node_id);
        Ok(())
    }

    /// Stop the raft node
    pub async fn stop(&self) -> Result<()> {
        if !self.running.load(Ordering::Relaxed) {
            return Ok(());
        }
        
        info!("Stopping raft node {}", self.config.node_id);
        self.running.store(false, Ordering::Relaxed);
        
        // Cancel all tasks
        let mut handles = self.task_handles.write().await;
        for handle in handles.drain(..) {
            handle.abort();
        }
        
        info!("Raft node {} stopped", self.config.node_id);
        Ok(())
    }

    /// Propose a policy operation
    pub async fn propose_policy(&self, operation: PolicyOperation) -> Result<OperationResult> {
        if !self.running.load(Ordering::Relaxed) {
            return Err(RaftError::Node("Node is not running".to_string()));
        }
        
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        let request = ProposalRequest {
            operation,
            response_tx,
        };
        
        self.proposal_tx.send(request)
            .map_err(|_| RaftError::Node("Failed to send proposal".to_string()))?;
        
        response_rx.await
            .map_err(|_| RaftError::Node("Failed to receive proposal response".to_string()))?
    }

    /// Send a raft message to this node
    pub async fn receive_message(&self, message: Message) -> Result<()> {
        self.stats.messages_received.fetch_add(1, Ordering::Relaxed);
        
        self.message_tx.send(message)
            .map_err(|_| RaftError::Node("Failed to send message".to_string()))?;
        
        Ok(())
    }

    /// Get node statistics
    pub fn stats(&self) -> &RaftNodeStats {
        &self.stats
    }

    /// Get current policies
    pub async fn get_policies(&self) -> HashMap<String, Policy> {
        let state_machine = self.state_machine.read().await;
        state_machine.get_all_policies().clone()
    }

    /// Get a specific policy
    pub async fn get_policy(&self, id: &str) -> Option<Policy> {
        let state_machine = self.state_machine.read().await;
        state_machine.get_policy(id).cloned()
    }

    /// Check if this node is the leader
    pub async fn is_leader(&self) -> bool {
        let role = self.stats.role.read().await;
        matches!(*role, StateRole::Leader)
    }

    /// Get current leader ID
    pub fn leader_id(&self) -> Option<u64> {
        let leader_id = self.stats.leader_id.load(Ordering::Relaxed);
        if leader_id == 0 { None } else { Some(leader_id) }
    }

    /// Main raft loop
    async fn run_main_loop(&self) -> Result<()> {
        let mut message_rx = self.message_rx.lock().await;
        
        while self.running.load(Ordering::Relaxed) {
            // Process incoming messages
            while let Ok(message) = message_rx.try_recv() {
                let mut raw_node = self.raw_node.lock().await;
                if let Err(e) = raw_node.step(message) {
                    warn!("Failed to step raft: {}", e);
                }
            }
            
            // Process ready state
            let mut raw_node = self.raw_node.lock().await;
            if raw_node.has_ready() {
                let ready = raw_node.ready();
                self.handle_ready(ready).await?;
            }
            
            drop(raw_node);
            
            // Small delay to prevent busy loop
            sleep(Duration::from_millis(1)).await;
        }
        
        Ok(())
    }

    /// Handle ready state from raft
    async fn handle_ready(&self, mut ready: Ready) -> Result<()> {
        // Update statistics
        if let Some(hs) = ready.hs() {
            self.stats.term.store(hs.term, Ordering::Relaxed);
            self.stats.committed_index.store(hs.commit, Ordering::Relaxed);
        }
        
        // Update role and leader
        if let Some(ss) = ready.ss() {
            let mut role = self.stats.role.write().await;
            *role = ss.raft_state;
            
            if ss.leader_id != raft::INVALID_ID {
                self.stats.leader_id.store(ss.leader_id, Ordering::Relaxed);
            }
        }
        
        // Send messages
        for message in ready.take_messages() {
            self.send_message(message).await;
        }
        
        // Apply committed entries
        for entry in ready.take_committed_entries() {
            self.apply_entry(entry).await?;
        }
        
        // Advance raft
        let mut raw_node = self.raw_node.lock().await;
        raw_node.advance(ready);
        
        Ok(())
    }

    /// Apply a committed entry
    async fn apply_entry(&self, entry: Entry) -> Result<()> {
        if entry.entry_type == EntryType::EntryNormal && !entry.data.is_empty() {
            let mut state_machine = self.state_machine.write().await;
            let result = state_machine.apply_entry(&entry)?;
            
            self.stats.applied_index.store(entry.index, Ordering::Relaxed);
            
            debug!("Applied entry {} with result: {:?}", entry.index, result);
        }
        
        Ok(())
    }

    /// Send a raft message
    async fn send_message(&self, message: Message) {
        self.stats.messages_sent.fetch_add(1, Ordering::Relaxed);
        
        // In a real implementation, this would send the message over the network
        // For now, we just log it
        debug!("Sending message from {} to {}: {:?}", 
               message.from, message.to, message.msg_type);
    }

    /// Tick timer loop
    async fn run_tick_loop(&self) {
        let mut ticker = interval(self.config.heartbeat_interval);
        
        while self.running.load(Ordering::Relaxed) {
            ticker.tick().await;
            
            let mut raw_node = self.raw_node.lock().await;
            raw_node.tick();
        }
    }

    /// Proposal handling loop
    async fn run_proposal_loop(&self) {
        let mut proposal_rx = self.proposal_rx.lock().await;
        
        while self.running.load(Ordering::Relaxed) {
            if let Some(request) = proposal_rx.recv().await {
                let result = self.handle_proposal(request.operation).await;
                let _ = request.response_tx.send(result);
            }
        }
    }

    /// Handle a proposal request
    async fn handle_proposal(&self, operation: PolicyOperation) -> Result<OperationResult> {
        self.stats.proposals_total.fetch_add(1, Ordering::Relaxed);
        
        // Check if we're the leader
        if !self.is_leader().await {
            self.stats.proposals_failed.fetch_add(1, Ordering::Relaxed);
            return Err(RaftError::NotLeader);
        }
        
        // Serialize the operation
        let data = PolicyStateMachine::serialize_operation(operation)?;
        
        // Propose to raft
        let mut raw_node = self.raw_node.lock().await;
        match raw_node.propose(vec![], data) {
            Ok(_) => {
                self.stats.proposals_success.fetch_add(1, Ordering::Relaxed);
                // In a real implementation, we would wait for the proposal to be committed
                // and return the actual result. For now, return a placeholder.
                Ok(OperationResult {
                    success: true,
                    error: None,
                    policy_id: None,
                    metadata: crate::state_machine::OperationMetadata {
                        timestamp: chrono::Utc::now(),
                        applied_index: 0, // Would be filled when actually applied
                        operation_type: "proposed".to_string(),
                    },
                })
            }
            Err(e) => {
                self.stats.proposals_failed.fetch_add(1, Ordering::Relaxed);
                Err(RaftError::Raft(e))
            }
        }
    }
}

impl Clone for RaftNode {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            raw_node: Arc::clone(&self.raw_node),
            state_machine: Arc::clone(&self.state_machine),
            stats: Arc::clone(&self.stats),
            message_tx: self.message_tx.clone(),
            message_rx: Arc::clone(&self.message_rx),
            proposal_tx: self.proposal_tx.clone(),
            proposal_rx: Arc::clone(&self.proposal_rx),
            running: Arc::clone(&self.running),
            task_handles: Arc::clone(&self.task_handles),
        }
    }
}

impl RaftNodeStats {
    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Get current term
    pub fn term(&self) -> u64 {
        self.term.load(Ordering::Relaxed)
    }

    /// Get current role
    pub async fn role(&self) -> StateRole {
        *self.role.read().await
    }

    /// Get leader ID
    pub fn leader_id(&self) -> u64 {
        self.leader_id.load(Ordering::Relaxed)
    }

    /// Get applied index
    pub fn applied_index(&self) -> u64 {
        self.applied_index.load(Ordering::Relaxed)
    }

    /// Get committed index
    pub fn committed_index(&self) -> u64 {
        self.committed_index.load(Ordering::Relaxed)
    }

    /// Get total proposals
    pub fn proposals_total(&self) -> u64 {
        self.proposals_total.load(Ordering::Relaxed)
    }

    /// Get successful proposals
    pub fn proposals_success(&self) -> u64 {
        self.proposals_success.load(Ordering::Relaxed)
    }

    /// Get failed proposals
    pub fn proposals_failed(&self) -> u64 {
        self.proposals_failed.load(Ordering::Relaxed)
    }

    /// Get messages sent
    pub fn messages_sent(&self) -> u64 {
        self.messages_sent.load(Ordering::Relaxed)
    }

    /// Get messages received
    pub fn messages_received(&self) -> u64 {
        self.messages_received.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RaftConfigBuilder;
    use crate::policy::Policy;

    #[tokio::test]
    async fn test_node_creation() {
        let config = RaftConfigBuilder::new()
            .node_id(1)
            .build();
        
        let node = RaftNode::new(1, config).await;
        assert!(node.is_ok());
        
        let node = node.unwrap();
        assert_eq!(node.stats().node_id, 1);
        assert!(!node.is_leader().await);
    }

    #[tokio::test]
    async fn test_node_start_stop() {
        let config = RaftConfigBuilder::new()
            .node_id(1)
            .build();
        
        let node = RaftNode::new(1, config).await.unwrap();
        
        // Start node
        assert!(node.start().await.is_ok());
        assert!(node.running.load(Ordering::Relaxed));
        
        // Stop node
        assert!(node.stop().await.is_ok());
        assert!(!node.running.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_policy_operations() {
        let config = RaftConfigBuilder::new()
            .node_id(1)
            .build();
        
        let node = RaftNode::new(1, config).await.unwrap();
        node.start().await.unwrap();
        
        // Wait a bit for node to initialize
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        let policy = Policy::pin_model("gpt-7b".to_string(), vec![]);
        let operation = PolicyOperation::Create(policy.clone());
        
        // This will fail because we're not the leader in a single-node setup
        // In a real test, we'd need to set up a proper cluster
        let result = node.propose_policy(operation).await;
        
        // Should fail with NotLeader error
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), RaftError::NotLeader));
        
        node.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_node_stats() {
        let config = RaftConfigBuilder::new()
            .node_id(42)
            .build();
        
        let node = RaftNode::new(42, config).await.unwrap();
        let stats = node.stats();
        
        assert_eq!(stats.node_id, 42);
        assert_eq!(stats.term(), 0);
        assert_eq!(stats.applied_index(), 0);
        assert_eq!(stats.committed_index(), 0);
        assert_eq!(stats.proposals_total(), 0);
    }

    #[tokio::test]
    async fn test_message_handling() {
        let config = RaftConfigBuilder::new()
            .node_id(1)
            .build();
        
        let node = RaftNode::new(1, config).await.unwrap();
        node.start().await.unwrap();
        
        // Create a test message
        let message = Message::default();
        
        // Send message to node
        let result = node.receive_message(message).await;
        assert!(result.is_ok());
        
        // Check that message count increased
        assert_eq!(node.stats().messages_received(), 1);
        
        node.stop().await.unwrap();
    }
}
