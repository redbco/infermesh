//! Gossip message types and serialization

use crate::membership::{Member, MemberState};
use mesh_core::NodeId;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::{SystemTime, UNIX_EPOCH};

/// Type of gossip message
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    /// Ping message for failure detection
    Ping,
    /// Ack response to ping
    Ack,
    /// Indirect ping request
    IndirectPing,
    /// Membership update
    Membership,
    /// User data dissemination
    UserData,
    /// Join request
    Join,
    /// Leave notification
    Leave,
}

/// Gossip message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipMessage {
    /// Type of message
    pub message_type: MessageType,
    
    /// Source node ID
    pub source: NodeId,
    
    /// Target node ID (for directed messages)
    pub target: Option<NodeId>,
    
    /// Sequence number for ordering
    pub sequence: u64,
    
    /// Timestamp when message was created
    pub timestamp: u64,
    
    /// Message payload
    pub payload: MessagePayload,
    
    /// Message authentication code (if encryption enabled)
    pub mac: Option<Vec<u8>>,
}

/// Payload of a gossip message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePayload {
    /// Empty payload (for ping/ack)
    Empty,
    
    /// Membership information
    Membership(Vec<MembershipUpdate>),
    
    /// User data
    UserData {
        key: String,
        value: Vec<u8>,
        version: u64,
    },
    
    /// Join request
    Join {
        node_id: NodeId,
        gossip_addr: SocketAddr,
        metadata: std::collections::HashMap<String, String>,
    },
    
    /// Leave notification
    Leave {
        node_id: NodeId,
        incarnation: u64,
    },
    
    /// Indirect ping request
    IndirectPing {
        target: NodeId,
        target_addr: SocketAddr,
    },
}

/// Membership update information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MembershipUpdate {
    pub node_id: NodeId,
    pub gossip_addr: SocketAddr,
    pub state: MemberState,
    pub incarnation: u64,
    pub metadata: std::collections::HashMap<String, String>,
}

impl From<&Member> for MembershipUpdate {
    fn from(member: &Member) -> Self {
        Self {
            node_id: member.node_id.clone(),
            gossip_addr: member.gossip_addr,
            state: member.state,
            incarnation: member.incarnation,
            metadata: member.metadata.clone(),
        }
    }
}

impl GossipMessage {
    /// Create a new gossip message
    pub fn new(
        message_type: MessageType,
        source: NodeId,
        target: Option<NodeId>,
        sequence: u64,
        payload: MessagePayload,
    ) -> Self {
        Self {
            message_type,
            source,
            target,
            sequence,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            payload,
            mac: None,
        }
    }
    
    /// Create a ping message
    pub fn ping(source: NodeId, target: NodeId, sequence: u64) -> Self {
        Self::new(
            MessageType::Ping,
            source,
            Some(target),
            sequence,
            MessagePayload::Empty,
        )
    }
    
    /// Create an ack message
    pub fn ack(source: NodeId, target: NodeId, sequence: u64) -> Self {
        Self::new(
            MessageType::Ack,
            source,
            Some(target),
            sequence,
            MessagePayload::Empty,
        )
    }
    
    /// Create an indirect ping message
    pub fn indirect_ping(
        source: NodeId,
        proxy: NodeId,
        target: NodeId,
        target_addr: SocketAddr,
        sequence: u64,
    ) -> Self {
        Self::new(
            MessageType::IndirectPing,
            source,
            Some(proxy),
            sequence,
            MessagePayload::IndirectPing {
                target,
                target_addr,
            },
        )
    }
    
    /// Create a membership message
    pub fn membership(source: NodeId, updates: Vec<MembershipUpdate>, sequence: u64) -> Self {
        Self::new(
            MessageType::Membership,
            source,
            None,
            sequence,
            MessagePayload::Membership(updates),
        )
    }
    
    /// Create a join message
    pub fn join(
        source: NodeId,
        gossip_addr: SocketAddr,
        metadata: std::collections::HashMap<String, String>,
        sequence: u64,
    ) -> Self {
        Self::new(
            MessageType::Join,
            source.clone(),
            None,
            sequence,
            MessagePayload::Join {
                node_id: source,
                gossip_addr,
                metadata,
            },
        )
    }
    
    /// Create a leave message
    pub fn leave(source: NodeId, incarnation: u64, sequence: u64) -> Self {
        Self::new(
            MessageType::Leave,
            source.clone(),
            None,
            sequence,
            MessagePayload::Leave {
                node_id: source,
                incarnation,
            },
        )
    }
    
    /// Create a user data message
    pub fn user_data(
        source: NodeId,
        key: String,
        value: Vec<u8>,
        version: u64,
        sequence: u64,
    ) -> Self {
        Self::new(
            MessageType::UserData,
            source,
            None,
            sequence,
            MessagePayload::UserData { key, value, version },
        )
    }
    
    /// Check if the message is expired
    pub fn is_expired(&self, max_age_secs: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        now.saturating_sub(self.timestamp) > max_age_secs
    }
    
    /// Get the age of the message in seconds
    pub fn age_secs(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        now.saturating_sub(self.timestamp)
    }
    
    /// Serialize the message to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }
    
    /// Deserialize a message from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
    
    /// Calculate message size in bytes
    pub fn size(&self) -> usize {
        self.to_bytes().map(|b| b.len()).unwrap_or(0)
    }
    
    /// Set message authentication code
    pub fn set_mac(&mut self, mac: Vec<u8>) {
        self.mac = Some(mac);
    }
    
    /// Verify message authentication code
    pub fn verify_mac(&self, expected_mac: &[u8]) -> bool {
        match &self.mac {
            Some(mac) => mac == expected_mac,
            None => false,
        }
    }
}

/// Message builder for creating gossip messages
pub struct MessageBuilder {
    source: NodeId,
    sequence: u64,
}

impl MessageBuilder {
    /// Create a new message builder
    pub fn new(source: NodeId, sequence: u64) -> Self {
        Self { source, sequence }
    }
    
    /// Build a ping message
    pub fn ping(&self, target: NodeId) -> GossipMessage {
        GossipMessage::ping(self.source.clone(), target, self.sequence)
    }
    
    /// Build an ack message
    pub fn ack(&self, target: NodeId) -> GossipMessage {
        GossipMessage::ack(self.source.clone(), target, self.sequence)
    }
    
    /// Build a membership message
    pub fn membership(&self, updates: Vec<MembershipUpdate>) -> GossipMessage {
        GossipMessage::membership(self.source.clone(), updates, self.sequence)
    }
    
    /// Build a join message
    pub fn join(
        &self,
        gossip_addr: SocketAddr,
        metadata: std::collections::HashMap<String, String>,
    ) -> GossipMessage {
        GossipMessage::join(self.source.clone(), gossip_addr, metadata, self.sequence)
    }
    
    /// Build a leave message
    pub fn leave(&self, incarnation: u64) -> GossipMessage {
        GossipMessage::leave(self.source.clone(), incarnation, self.sequence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_core::NodeId;
    // use std::collections::HashMap; // Unused in current tests
    use std::net::SocketAddr;

    #[test]
    fn test_message_creation() {
        let source = NodeId::new("test-node");
        let target = NodeId::new("test-node");
        let message = GossipMessage::ping(source.clone(), target.clone(), 1);
        
        assert_eq!(message.message_type, MessageType::Ping);
        assert_eq!(message.source, source);
        assert_eq!(message.target, Some(target));
        assert_eq!(message.sequence, 1);
        assert!(matches!(message.payload, MessagePayload::Empty));
    }

    #[test]
    fn test_message_serialization() {
        let source = NodeId::new("test-node");
        let target = NodeId::new("test-node");
        let message = GossipMessage::ping(source, target, 1);
        
        let bytes = message.to_bytes().unwrap();
        let deserialized = GossipMessage::from_bytes(&bytes).unwrap();
        
        assert_eq!(message.message_type, deserialized.message_type);
        assert_eq!(message.source, deserialized.source);
        assert_eq!(message.target, deserialized.target);
        assert_eq!(message.sequence, deserialized.sequence);
    }

    #[test]
    fn test_membership_update() {
        let node_id = NodeId::new("test-node");
        let addr = "127.0.0.1:7946".parse::<SocketAddr>().unwrap();
        let mut member = Member::new(node_id.clone(), addr);
        member.state = MemberState::Suspect;
        member.incarnation = 5;
        member.metadata.insert("role".to_string(), "gpu".to_string());
        
        let update = MembershipUpdate::from(&member);
        assert_eq!(update.node_id, node_id);
        assert_eq!(update.gossip_addr, addr);
        assert_eq!(update.state, MemberState::Suspect);
        assert_eq!(update.incarnation, 5);
        assert_eq!(update.metadata.get("role"), Some(&"gpu".to_string()));
    }

    #[test]
    fn test_message_builder() {
        let source = NodeId::new("test-node");
        let builder = MessageBuilder::new(source.clone(), 42);
        
        let target = NodeId::new("test-node");
        let ping = builder.ping(target.clone());
        
        assert_eq!(ping.message_type, MessageType::Ping);
        assert_eq!(ping.source, source);
        assert_eq!(ping.target, Some(target));
        assert_eq!(ping.sequence, 42);
    }

    #[test]
    fn test_message_expiration() {
        let source = NodeId::new("test-node");
        let mut message = GossipMessage::ping(source, NodeId::new("test-node"), 1);
        
        // Fresh message should not be expired
        assert!(!message.is_expired(60));
        
        // Artificially age the message
        message.timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .saturating_sub(120);
        
        // Now it should be expired
        assert!(message.is_expired(60));
    }

    #[test]
    fn test_message_mac() {
        let source = NodeId::new("test-node");
        let mut message = GossipMessage::ping(source, NodeId::new("test-node"), 1);
        
        let mac = vec![1, 2, 3, 4];
        message.set_mac(mac.clone());
        
        assert!(message.verify_mac(&mac));
        assert!(!message.verify_mac(&[5, 6, 7, 8]));
    }
}
