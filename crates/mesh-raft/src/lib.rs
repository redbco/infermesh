//! # mesh-raft
//!
//! Distributed consensus and policy storage for infermesh.
//!
//! This crate provides:
//! - Raft consensus using tikv-raft
//! - Policy storage for model pinning, quotas, and ACLs
//! - Leader election and log replication
//! - Disk persistence for logs and snapshots
//!
//! ## Example
//!
//! ```rust,no_run
//! use mesh_raft::{RaftNode, RaftConfig, Policy, PolicyOperation};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = RaftConfig::default();
//!     let mut node = RaftNode::new(1, config).await?;
//!     
//!     // Start the raft node
//!     node.start().await?;
//!     
//!     // Propose a policy (only works if this node is the leader)
//!     let policy = Policy::pin_model("gpt-7b".to_string(), vec!["node1".to_string().into()]);
//!     node.propose_policy(PolicyOperation::Create(policy)).await?;
//!     
//!     Ok(())
//! }
//! ```

use thiserror::Error;

pub mod config;
pub mod node;
pub mod policy;
pub mod storage;
pub mod state_machine;

// Re-export main types
pub use config::{RaftConfig, RaftConfigBuilder};
pub use node::{RaftNode, RaftNodeStats};
pub use policy::{Policy, PolicyType, ModelPinPolicy, QuotaPolicy, AclPolicy};
pub use storage::{RaftStorage, RaftStorageBackend, MemoryStorage, DiskStorage};
pub use state_machine::{PolicyStateMachine, StateMachineSnapshot, PolicyOperation};

/// Result type for raft operations
pub type Result<T> = std::result::Result<T, RaftError>;

/// Errors that can occur during raft operations
#[derive(Error, Debug)]
pub enum RaftError {
    #[error("Raft error: {0}")]
    Raft(#[from] raft::Error),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Node error: {0}")]
    Node(String),

    #[error("Policy error: {0}")]
    Policy(String),

    #[error("Leadership error: {0}")]
    Leadership(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Not leader")]
    NotLeader,

    #[error("Node not found: {0}")]
    NodeNotFound(u64),

    #[error("Invalid proposal")]
    InvalidProposal,
}

impl RaftError {
    /// Check if this error indicates the node is not the leader
    pub fn is_not_leader(&self) -> bool {
        matches!(self, RaftError::NotLeader)
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            RaftError::NotLeader | RaftError::Timeout(_) | RaftError::Leadership(_)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raft_error_properties() {
        let not_leader = RaftError::NotLeader;
        assert!(not_leader.is_not_leader());
        assert!(not_leader.is_retryable());

        let timeout = RaftError::Timeout("test".to_string());
        assert!(!timeout.is_not_leader());
        assert!(timeout.is_retryable());

        let storage = RaftError::Storage("test".to_string());
        assert!(!storage.is_not_leader());
        assert!(!storage.is_retryable());
    }

    #[test]
    fn test_error_display() {
        let error = RaftError::NotLeader;
        assert_eq!(error.to_string(), "Not leader");

        let error = RaftError::Policy("invalid policy".to_string());
        assert_eq!(error.to_string(), "Policy error: invalid policy");
    }
}
