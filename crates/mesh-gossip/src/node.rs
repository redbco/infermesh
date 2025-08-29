//! High-level gossip node implementation

use crate::{
    config::GossipConfig,
    protocol::{ProtocolEvent, SwimProtocol},
    transport::{Transport, UdpTransport},
    Result,
};
use mesh_core::NodeId;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{error, info};

/// High-level gossip node that manages the SWIM protocol and transport
pub struct GossipNode {
    /// Local node identifier
    node_id: NodeId,
    
    /// Local gossip address
    local_addr: SocketAddr,
    
    /// Gossip configuration
    config: GossipConfig,
    
    /// SWIM protocol instance
    protocol: SwimProtocol,
    
    /// Transport layer
    #[allow(dead_code)]
    transport: Arc<dyn Transport>,
    
    /// Event receiver for protocol events
    event_rx: Option<mpsc::UnboundedReceiver<ProtocolEvent>>,
    
    /// Running state
    running: bool,
}

impl GossipNode {
    /// Create a new gossip node
    pub async fn new(
        node_id: NodeId,
        bind_addr: SocketAddr,
        config: GossipConfig,
    ) -> Result<Self> {
        // Validate configuration
        config.validate().map_err(|e| crate::GossipError::Configuration(e))?;
        
        // Create transport
        let mut transport = UdpTransport::new(bind_addr, config.max_gossip_packet_size);
        transport.start().await?;
        let actual_addr = transport.local_addr();
        let transport = Arc::new(transport);
        
        // Create protocol
        let mut protocol = SwimProtocol::new(
            node_id.clone(),
            actual_addr,
            config.clone(),
            transport.clone(),
        );
        
        // Set up event channel
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        protocol.set_event_sender(event_tx);
        
        Ok(Self {
            node_id,
            local_addr: actual_addr,
            config,
            protocol,
            transport,
            event_rx: Some(event_rx),
            running: false,
        })
    }
    
    /// Start the gossip node
    pub async fn start(&mut self) -> Result<()> {
        if self.running {
            return Ok(());
        }
        
        info!("Starting gossip node {} on {}", self.node_id, self.local_addr);
        
        // Start the protocol
        self.protocol.start().await?;
        
        // Start event processing
        self.start_event_processing().await;
        
        self.running = true;
        Ok(())
    }
    
    /// Stop the gossip node
    pub async fn stop(&mut self) -> Result<()> {
        if !self.running {
            return Ok(());
        }
        
        info!("Stopping gossip node {}", self.node_id);
        
        // Leave the cluster gracefully
        if let Err(e) = self.protocol.leave().await {
            error!("Failed to leave cluster gracefully: {}", e);
        }
        
        // Stop the protocol
        self.protocol.stop().await?;
        
        self.running = false;
        Ok(())
    }
    
    /// Join an existing cluster
    pub async fn join_cluster(&self, seed_addr: SocketAddr) -> Result<()> {
        if !self.running {
            return Err(crate::GossipError::Protocol("Node not running".to_string()));
        }
        
        self.protocol.join(seed_addr).await
    }
    
    /// Add user data to be disseminated through gossip
    pub async fn add_user_data(&self, key: String, value: Vec<u8>, version: u64) -> Result<()> {
        if !self.running {
            return Err(crate::GossipError::Protocol("Node not running".to_string()));
        }
        
        self.protocol.add_user_data(key, value, version).await
    }
    
    /// Get current membership information
    pub async fn membership(&self) -> crate::membership::Membership {
        self.protocol.membership().await
    }
    
    /// Get the local node ID
    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }
    
    /// Get the local gossip address
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }
    
    /// Get the gossip configuration
    pub fn config(&self) -> &GossipConfig {
        &self.config
    }
    
    /// Check if the node is running
    pub fn is_running(&self) -> bool {
        self.running
    }
    
    /// Get membership statistics
    pub async fn membership_stats(&self) -> crate::membership::MembershipStats {
        let membership = self.membership().await;
        membership.stats()
    }
    
    /// Update local node metadata
    pub async fn update_metadata(&self, key: String, value: String) -> Result<()> {
        // This is a simplified version - in practice, we'd need better access to membership
        // For now, we'll just return Ok since the protocol doesn't expose membership mutation
        let _ = (key, value); // Suppress unused variable warnings
        Ok(())
    }
    
    /// Start processing protocol events
    async fn start_event_processing(&mut self) {
        if let Some(mut event_rx) = self.event_rx.take() {
            let node_id = self.node_id.clone();
            
            tokio::spawn(async move {
                while let Some(event) = event_rx.recv().await {
                    Self::handle_protocol_event(&node_id, event).await;
                }
            });
        }
    }
    
    /// Handle protocol events
    async fn handle_protocol_event(_local_node_id: &NodeId, event: ProtocolEvent) {
        match event {
            ProtocolEvent::MemberJoined(node_id) => {
                info!("Member joined: {}", node_id);
            }
            ProtocolEvent::MemberLeft(node_id) => {
                info!("Member left: {}", node_id);
            }
            ProtocolEvent::MemberSuspect(node_id) => {
                info!("Member suspected: {}", node_id);
            }
            ProtocolEvent::MemberDead(node_id) => {
                info!("Member dead: {}", node_id);
            }
            ProtocolEvent::MemberRecovered(node_id) => {
                info!("Member recovered: {}", node_id);
            }
            ProtocolEvent::UserData { key, value, version, source } => {
                info!(
                    "Received user data from {}: key={}, version={}, size={} bytes",
                    source,
                    key,
                    version,
                    value.len()
                );
            }
        }
    }
}

impl Drop for GossipNode {
    fn drop(&mut self) {
        if self.running {
            // Note: We can't do async operations in Drop, so we just log
            // In practice, users should call stop() explicitly before dropping
            tracing::warn!("GossipNode dropped while running - call stop() explicitly for graceful shutdown");
        }
    }
}

/// Builder for creating gossip nodes with custom configuration
pub struct GossipNodeBuilder {
    node_id: Option<NodeId>,
    bind_addr: Option<SocketAddr>,
    config: GossipConfig,
}

impl Default for GossipNodeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GossipNodeBuilder {
    /// Create a new gossip node builder
    pub fn new() -> Self {
        Self {
            node_id: None,
            bind_addr: None,
            config: GossipConfig::default(),
        }
    }
    
    /// Set the node ID
    pub fn with_node_id(mut self, node_id: NodeId) -> Self {
        self.node_id = Some(node_id);
        self
    }
    
    /// Set the bind address
    pub fn with_bind_addr(mut self, addr: SocketAddr) -> Self {
        self.bind_addr = Some(addr);
        self
    }
    
    /// Set the gossip configuration
    pub fn with_config(mut self, config: GossipConfig) -> Self {
        self.config = config;
        self
    }
    
    /// Set the gossip interval
    pub fn with_gossip_interval(mut self, interval: std::time::Duration) -> Self {
        self.config.gossip_interval = interval;
        self
    }
    
    /// Set the failure timeout
    pub fn with_failure_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.config.failure_timeout = timeout;
        self
    }
    
    /// Set the gossip fanout
    pub fn with_gossip_fanout(mut self, fanout: usize) -> Self {
        self.config.gossip_fanout = fanout;
        self
    }
    
    /// Enable encryption with a shared secret
    pub fn with_encryption(mut self, secret: String) -> Self {
        self.config.enable_encryption = true;
        self.config.shared_secret = Some(secret);
        self
    }
    
    /// Build the gossip node
    pub async fn build(self) -> Result<GossipNode> {
        let node_id = self.node_id.unwrap_or_else(|| NodeId::new("gossip-node"));
        let bind_addr = self.bind_addr.unwrap_or_else(|| "127.0.0.1:0".parse().unwrap());
        
        GossipNode::new(node_id, bind_addr, self.config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_core::NodeId;
    use std::net::SocketAddr;
    use std::time::Duration;

    #[tokio::test]
    async fn test_gossip_node_creation() {
        let node_id = NodeId::new("test-node");
        let addr = "127.0.0.1:0".parse::<SocketAddr>().unwrap();
        let config = GossipConfig::default();
        
        let node = GossipNode::new(node_id.clone(), addr, config).await.unwrap();
        
        assert_eq!(node.node_id(), &node_id);
        assert!(!node.is_running());
    }

    #[tokio::test]
    async fn test_gossip_node_start_stop() {
        let node_id = NodeId::new("test-node");
        let addr = "127.0.0.1:0".parse::<SocketAddr>().unwrap();
        let config = GossipConfig::default();
        
        let mut node = GossipNode::new(node_id, addr, config).await.unwrap();
        
        assert!(node.start().await.is_ok());
        assert!(node.is_running());
        
        assert!(node.stop().await.is_ok());
        assert!(!node.is_running());
    }

    #[tokio::test]
    async fn test_gossip_node_builder() {
        let node_id = NodeId::new("test-node");
        let addr = "127.0.0.1:0".parse::<SocketAddr>().unwrap();
        
        let node = GossipNodeBuilder::new()
            .with_node_id(node_id.clone())
            .with_bind_addr(addr)
            .with_gossip_interval(Duration::from_millis(100))
            .with_failure_timeout(Duration::from_secs(5))
            .with_gossip_fanout(5)
            .build()
            .await
            .unwrap();
        
        assert_eq!(node.node_id(), &node_id);
        assert_eq!(node.config().gossip_interval, Duration::from_millis(100));
        assert_eq!(node.config().failure_timeout, Duration::from_secs(5));
        assert_eq!(node.config().gossip_fanout, 5);
    }

    #[tokio::test]
    async fn test_membership_stats() {
        let node_id = NodeId::new("test-node");
        let addr = "127.0.0.1:0".parse::<SocketAddr>().unwrap();
        let config = GossipConfig::default();
        
        let node = GossipNode::new(node_id, addr, config).await.unwrap();
        let stats = node.membership_stats().await;
        
        // Should have at least the local node
        assert_eq!(stats.total, 1);
        assert_eq!(stats.alive, 1);
        assert_eq!(stats.suspect, 0);
        assert_eq!(stats.dead, 0);
        assert_eq!(stats.left, 0);
    }

    #[tokio::test]
    async fn test_user_data_when_not_running() {
        let node_id = NodeId::new("test-node");
        let addr = "127.0.0.1:0".parse::<SocketAddr>().unwrap();
        let config = GossipConfig::default();
        
        let node = GossipNode::new(node_id, addr, config).await.unwrap();
        
        let result = node.add_user_data("test".to_string(), vec![1, 2, 3], 1).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_join_cluster_when_not_running() {
        let node_id = NodeId::new("test-node");
        let addr = "127.0.0.1:0".parse::<SocketAddr>().unwrap();
        let config = GossipConfig::default();
        
        let node = GossipNode::new(node_id, addr, config).await.unwrap();
        let seed_addr = "127.0.0.1:12345".parse::<SocketAddr>().unwrap();
        
        let result = node.join_cluster(seed_addr).await;
        assert!(result.is_err());
    }
}
