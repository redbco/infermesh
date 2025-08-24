//! # mesh-gossip
//!
//! SWIM-style gossip protocol implementation for infermesh node membership and failure detection.
//!
//! This crate provides:
//! - Node membership management with failure detection
//! - Lightweight state dissemination across the mesh
//! - UDP/TCP transport layer for gossip messages
//! - Configurable gossip intervals and failure detection timeouts
//!
//! ## Example
//!
//! ```rust
//! use mesh_gossip::{GossipNode, GossipConfig};
//! use mesh_core::NodeId;
//! use std::net::SocketAddr;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = GossipConfig::default();
//! let node_id = NodeId::new("test-node");
//! let bind_addr = "127.0.0.1:7946".parse::<SocketAddr>()?;
//!
//! let mut gossip_node = GossipNode::new(node_id, bind_addr, config).await?;
//! gossip_node.start().await?;
//! # Ok(())
//! # }
//! ```

use thiserror::Error;

pub mod config;
pub mod membership;
pub mod message;
pub mod node;
pub mod protocol;
pub mod transport;

// Re-export commonly used types
pub use config::GossipConfig;
pub use membership::{Member, MemberState, Membership};
pub use message::{GossipMessage, MessageType};
pub use node::GossipNode;
pub use protocol::SwimProtocol;
pub use transport::{Transport, UdpTransport};

/// Result type for gossip operations
pub type Result<T> = std::result::Result<T, GossipError>;

/// Errors that can occur during gossip operations
#[derive(Error, Debug)]
pub enum GossipError {
    #[error("Network error: {0}")]
    Network(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid message: {0}")]
    InvalidMessage(String),

    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Transport error: {0}")]
    Transport(String),

    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Timeout: {0}")]
    Timeout(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_core::NodeId;
    use std::net::SocketAddr;

    #[test]
    fn test_gossip_config_creation() {
        let config = GossipConfig::default();
        assert!(config.gossip_interval.as_millis() > 0);
        assert!(config.failure_timeout.as_millis() > 0);
        assert!(config.max_gossip_packet_size > 0);
    }

    #[tokio::test]
    async fn test_gossip_node_creation() {
        let config = GossipConfig::default();
        let node_id = NodeId::new("test-node");
        let bind_addr = "127.0.0.1:0".parse::<SocketAddr>().unwrap();

        let result = GossipNode::new(node_id, bind_addr, config).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_member_state_transitions() {
        use membership::MemberState;
        
        let alive = MemberState::Alive;
        let suspect = MemberState::Suspect;
        let dead = MemberState::Dead;
        
        assert_ne!(alive, suspect);
        assert_ne!(suspect, dead);
        assert_ne!(alive, dead);
    }
}
