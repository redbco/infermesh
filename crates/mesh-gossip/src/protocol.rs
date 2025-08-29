//! SWIM protocol implementation for failure detection and membership management

use crate::{
    config::GossipConfig,
    membership::{Member, Membership},
    message::{GossipMessage, MessagePayload, MessageType, MembershipUpdate},
    transport::Transport,
    Result,
};
use mesh_core::NodeId;
use rand::seq::SliceRandom;
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{interval, timeout};
use tracing::{debug, error, info, warn};

/// SWIM protocol implementation
pub struct SwimProtocol {
    /// Local node configuration
    local_node_id: NodeId,
    local_addr: SocketAddr,
    
    /// Protocol configuration
    config: GossipConfig,
    
    /// Membership management
    membership: Arc<RwLock<Membership>>,
    
    /// Transport layer
    transport: Arc<dyn Transport>,
    
    /// Message sequence counter
    sequence_counter: Arc<RwLock<u64>>,
    
    /// Pending probe requests
    pending_probes: Arc<RwLock<HashMap<NodeId, ProbeRequest>>>,
    
    /// Message handlers
    #[allow(dead_code)]
    message_handlers: HashMap<MessageType, Box<dyn MessageHandler>>,
    
    /// Running state
    running: Arc<RwLock<bool>>,
    
    /// Event sender for external notifications
    event_tx: Option<mpsc::UnboundedSender<ProtocolEvent>>,
}

/// Probe request tracking
#[derive(Debug)]
struct ProbeRequest {
    #[allow(dead_code)]
    target: NodeId,
    #[allow(dead_code)]
    target_addr: SocketAddr,
    started_at: Instant,
    #[allow(dead_code)]
    indirect_probes: HashSet<NodeId>,
}

/// Protocol events for external consumption
#[derive(Debug, Clone)]
pub enum ProtocolEvent {
    /// A member joined the cluster
    MemberJoined(NodeId),
    /// A member left the cluster
    MemberLeft(NodeId),
    /// A member was marked as suspect
    MemberSuspect(NodeId),
    /// A member was marked as dead
    MemberDead(NodeId),
    /// A member recovered from suspect state
    MemberRecovered(NodeId),
    /// User data was received
    UserData {
        key: String,
        value: Vec<u8>,
        version: u64,
        source: NodeId,
    },
}

/// Trait for handling specific message types
trait MessageHandler: Send + Sync {
    #[allow(dead_code)]
    fn handle(&self, message: &GossipMessage, from: SocketAddr) -> Result<Option<GossipMessage>>;
}

impl SwimProtocol {
    /// Create a new SWIM protocol instance
    pub fn new(
        local_node_id: NodeId,
        local_addr: SocketAddr,
        config: GossipConfig,
        transport: Arc<dyn Transport>,
    ) -> Self {
        let membership = Arc::new(RwLock::new(Membership::new(
            local_node_id.clone(),
            local_addr,
            config.max_members,
        )));
        
        Self {
            local_node_id,
            local_addr,
            config,
            membership,
            transport,
            sequence_counter: Arc::new(RwLock::new(0)),
            pending_probes: Arc::new(RwLock::new(HashMap::new())),
            message_handlers: HashMap::new(),
            running: Arc::new(RwLock::new(false)),
            event_tx: None,
        }
    }
    
    /// Set event sender for protocol notifications
    pub fn set_event_sender(&mut self, tx: mpsc::UnboundedSender<ProtocolEvent>) {
        self.event_tx = Some(tx);
    }
    
    /// Start the SWIM protocol
    pub async fn start(&mut self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Ok(());
        }
        
        *running = true;
        drop(running);
        
        info!("Starting SWIM protocol for node {}", self.local_node_id);
        
        // Start the main protocol loop
        self.start_protocol_loop().await?;
        
        Ok(())
    }
    
    /// Stop the SWIM protocol
    pub async fn stop(&mut self) -> Result<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Ok(());
        }
        
        *running = false;
        info!("Stopping SWIM protocol for node {}", self.local_node_id);
        
        Ok(())
    }
    
    /// Join an existing cluster by contacting a seed node
    pub async fn join(&self, seed_addr: SocketAddr) -> Result<()> {
        let sequence = self.next_sequence().await;
        let membership = self.membership.read().await;
        let local_member = membership.local_member();
        
        let message = GossipMessage::join(
            self.local_node_id.clone(),
            self.local_addr,
            local_member.metadata.clone(),
            sequence,
        );
        
        info!("Joining cluster via seed node {}", seed_addr);
        self.transport.send_to(&message, seed_addr).await?;
        
        Ok(())
    }
    
    /// Leave the cluster gracefully
    pub async fn leave(&self) -> Result<()> {
        let sequence = self.next_sequence().await;
        let membership = self.membership.read().await;
        let incarnation = membership.local_member().incarnation;
        
        let message = GossipMessage::leave(self.local_node_id.clone(), incarnation, sequence);
        
        // Broadcast leave message to all active members
        let active_members = membership.active_members();
        let addrs: Vec<SocketAddr> = active_members
            .iter()
            .filter(|m| m.node_id != self.local_node_id)
            .map(|m| m.gossip_addr)
            .collect();
        
        if !addrs.is_empty() {
            info!("Broadcasting leave message to {} members", addrs.len());
            self.transport.broadcast(&message, &addrs).await?;
        }
        
        Ok(())
    }
    
    /// Get current membership information
    pub async fn membership(&self) -> Membership {
        self.membership.read().await.clone()
    }
    
    /// Add user data to be disseminated
    pub async fn add_user_data(&self, key: String, value: Vec<u8>, version: u64) -> Result<()> {
        let sequence = self.next_sequence().await;
        let message = GossipMessage::user_data(
            self.local_node_id.clone(),
            key,
            value,
            version,
            sequence,
        );
        
        // Gossip to random subset of members
        let membership = self.membership.read().await;
        let targets = membership.random_members(self.config.gossip_fanout);
        let addrs: Vec<SocketAddr> = targets.iter().map(|m| m.gossip_addr).collect();
        
        if !addrs.is_empty() {
            self.transport.broadcast(&message, &addrs).await?;
        }
        
        Ok(())
    }
    
    /// Start the main protocol loop
    async fn start_protocol_loop(&self) -> Result<()> {
        let running = self.running.clone();
        let membership = self.membership.clone();
        let transport = self.transport.clone();
        let config = self.config.clone();
        let sequence_counter = self.sequence_counter.clone();
        let pending_probes = self.pending_probes.clone();
        let local_node_id = self.local_node_id.clone();
        let event_tx = self.event_tx.clone();
        
        tokio::spawn(async move {
            let mut gossip_interval = interval(config.gossip_interval);
            let mut cleanup_interval = interval(config.cleanup_interval);
            
            while *running.read().await {
                tokio::select! {
                    _ = gossip_interval.tick() => {
                        if let Err(e) = Self::gossip_round(
                            &membership,
                            &transport,
                            &config,
                            &sequence_counter,
                            &pending_probes,
                            &local_node_id,
                            &event_tx,
                        ).await {
                            error!("Gossip round failed: {}", e);
                        }
                    }
                    
                    _ = cleanup_interval.tick() => {
                        Self::cleanup_round(&membership, &pending_probes, &config).await;
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Perform a single gossip round
    async fn gossip_round(
        membership: &Arc<RwLock<Membership>>,
        transport: &Arc<dyn Transport>,
        config: &GossipConfig,
        sequence_counter: &Arc<RwLock<u64>>,
        pending_probes: &Arc<RwLock<HashMap<NodeId, ProbeRequest>>>,
        local_node_id: &NodeId,
        event_tx: &Option<mpsc::UnboundedSender<ProtocolEvent>>,
    ) -> Result<()> {
        // 1. Select a random member to probe
        let probe_target = {
            let membership_guard = membership.read().await;
            let active_members = membership_guard.active_members();
            let mut candidates: Vec<&Member> = active_members
                .iter()
                .filter(|m| m.node_id != *local_node_id)
                .cloned()
                .collect();
            
            if candidates.is_empty() {
                return Ok(());
            }
            
            let mut rng = rand::thread_rng();
            candidates.shuffle(&mut rng);
            candidates.first().map(|m| (m.node_id.clone(), m.gossip_addr))
        };
        
        if let Some((target_id, target_addr)) = probe_target {
            Self::probe_member(
                &target_id,
                target_addr,
                membership,
                transport,
                config,
                sequence_counter,
                pending_probes,
                local_node_id,
                event_tx,
            ).await?;
        }
        
        // 2. Gossip membership updates to random members
        Self::gossip_membership(
            membership,
            transport,
            config,
            sequence_counter,
            local_node_id,
        ).await?;
        
        Ok(())
    }
    
    /// Probe a specific member for failure detection
    async fn probe_member(
        target_id: &NodeId,
        target_addr: SocketAddr,
        membership: &Arc<RwLock<Membership>>,
        transport: &Arc<dyn Transport>,
        config: &GossipConfig,
        sequence_counter: &Arc<RwLock<u64>>,
        pending_probes: &Arc<RwLock<HashMap<NodeId, ProbeRequest>>>,
        local_node_id: &NodeId,
        event_tx: &Option<mpsc::UnboundedSender<ProtocolEvent>>,
    ) -> Result<()> {
        // Check if we're already probing this member
        {
            let probes = pending_probes.read().await;
            if probes.contains_key(target_id) {
                return Ok(());
            }
        }
        
        let sequence = {
            let mut seq = sequence_counter.write().await;
            *seq += 1;
            *seq
        };
        
        // Send direct probe
        let ping_message = GossipMessage::ping(local_node_id.clone(), target_id.clone(), sequence);
        
        debug!("Probing member {} at {}", target_id, target_addr);
        
        // Record the probe request
        {
            let mut probes = pending_probes.write().await;
            probes.insert(target_id.clone(), ProbeRequest {
                target: target_id.clone(),
                target_addr,
                started_at: Instant::now(),
                indirect_probes: HashSet::new(),
            });
        }
        
        // Send the ping
        if let Err(e) = transport.send_to(&ping_message, target_addr).await {
            warn!("Failed to send ping to {}: {}", target_id, e);
        }
        
        // Wait for response or timeout
        let probe_result = timeout(config.probe_timeout, async {
            // In a real implementation, we would wait for an ACK message
            // For now, we'll simulate a timeout
            tokio::time::sleep(config.probe_timeout).await;
            false // Simulate no response
        }).await;
        
        match probe_result {
            Ok(true) => {
                // Received ACK, member is alive
                debug!("Received ACK from {}", target_id);
                pending_probes.write().await.remove(target_id);
            }
            Ok(false) | Err(_) => {
                // No response, try indirect probing
                Self::indirect_probe(
                    target_id,
                    target_addr,
                    membership,
                    transport,
                    config,
                    sequence_counter,
                    pending_probes,
                    local_node_id,
                    event_tx,
                ).await?;
            }
        }
        
        Ok(())
    }
    
    /// Perform indirect probing through other members
    async fn indirect_probe(
        target_id: &NodeId,
        target_addr: SocketAddr,
        membership: &Arc<RwLock<Membership>>,
        transport: &Arc<dyn Transport>,
        config: &GossipConfig,
        sequence_counter: &Arc<RwLock<u64>>,
        pending_probes: &Arc<RwLock<HashMap<NodeId, ProbeRequest>>>,
        local_node_id: &NodeId,
        event_tx: &Option<mpsc::UnboundedSender<ProtocolEvent>>,
    ) -> Result<()> {
        debug!("Starting indirect probe for {}", target_id);
        
        // Select random members for indirect probing
        let proxy_members = {
            let membership_guard = membership.read().await;
            let active_members = membership_guard.active_members();
            let mut candidates: Vec<&Member> = active_members
                .iter()
                .filter(|m| m.node_id != *local_node_id && m.node_id != *target_id)
                .cloned()
                .collect();
            
            let mut rng = rand::thread_rng();
            candidates.shuffle(&mut rng);
            candidates.into_iter().take(config.probe_fanout).cloned().collect::<Vec<_>>()
        };
        
        if proxy_members.is_empty() {
            // No proxies available, mark as suspect
            Self::mark_member_suspect(target_id, membership, event_tx).await;
            pending_probes.write().await.remove(target_id);
            return Ok(());
        }
        
        let sequence = {
            let mut seq = sequence_counter.write().await;
            *seq += 1;
            *seq
        };
        
        // Send indirect ping requests
        for proxy in &proxy_members {
            let indirect_ping = GossipMessage::indirect_ping(
                local_node_id.clone(),
                proxy.node_id.clone(),
                target_id.clone(),
                target_addr,
                sequence,
            );
            
            if let Err(e) = transport.send_to(&indirect_ping, proxy.gossip_addr).await {
                warn!("Failed to send indirect ping via {}: {}", proxy.node_id, e);
            }
        }
        
        // Wait for indirect probe responses
        let indirect_result = timeout(config.probe_timeout, async {
            // In a real implementation, we would wait for indirect responses
            tokio::time::sleep(config.probe_timeout).await;
            false // Simulate no response
        }).await;
        
        match indirect_result {
            Ok(true) => {
                // Received indirect ACK, member is alive
                debug!("Received indirect ACK for {}", target_id);
            }
            Ok(false) | Err(_) => {
                // No indirect response, mark as suspect
                Self::mark_member_suspect(target_id, membership, event_tx).await;
            }
        }
        
        pending_probes.write().await.remove(target_id);
        Ok(())
    }
    
    /// Mark a member as suspect
    async fn mark_member_suspect(
        target_id: &NodeId,
        membership: &Arc<RwLock<Membership>>,
        event_tx: &Option<mpsc::UnboundedSender<ProtocolEvent>>,
    ) {
        let mut membership_guard = membership.write().await;
        if membership_guard.mark_suspect(target_id) {
            info!("Marked member {} as suspect", target_id);
            
            if let Some(tx) = event_tx {
                let _ = tx.send(ProtocolEvent::MemberSuspect(target_id.clone()));
            }
        }
    }
    
    /// Gossip membership updates to random members
    async fn gossip_membership(
        membership: &Arc<RwLock<Membership>>,
        transport: &Arc<dyn Transport>,
        config: &GossipConfig,
        sequence_counter: &Arc<RwLock<u64>>,
        local_node_id: &NodeId,
    ) -> Result<()> {
        let (updates, target_addrs) = {
            let membership_guard = membership.read().await;
            
            // Collect membership updates
            let updates: Vec<MembershipUpdate> = membership_guard
                .members()
                .values()
                .map(MembershipUpdate::from)
                .collect();
            
            // Select random targets for gossip
            let targets = membership_guard.random_members(config.gossip_fanout);
            let addrs: Vec<SocketAddr> = targets.iter().map(|m| m.gossip_addr).collect();
            (updates, addrs)
        };
        
        if updates.is_empty() || target_addrs.is_empty() {
            return Ok(());
        }
        
        let sequence = {
            let mut seq = sequence_counter.write().await;
            *seq += 1;
            *seq
        };
        
        let message = GossipMessage::membership(local_node_id.clone(), updates, sequence);
        
        debug!("Gossiping membership to {} members", target_addrs.len());
        transport.broadcast(&message, &target_addrs).await?;
        
        Ok(())
    }
    
    /// Cleanup round for removing stale members and expired probes
    async fn cleanup_round(
        membership: &Arc<RwLock<Membership>>,
        pending_probes: &Arc<RwLock<HashMap<NodeId, ProbeRequest>>>,
        config: &GossipConfig,
    ) {
        // Clean up stale members
        {
            let mut membership_guard = membership.write().await;
            membership_guard.cleanup_stale_members(config.max_message_age);
        }
        
        // Clean up expired probe requests
        {
            let mut probes = pending_probes.write().await;
            let now = Instant::now();
            probes.retain(|_, probe| {
                now.duration_since(probe.started_at) < config.failure_timeout
            });
        }
    }
    
    /// Get next sequence number
    async fn next_sequence(&self) -> u64 {
        let mut seq = self.sequence_counter.write().await;
        *seq += 1;
        *seq
    }
    
    /// Handle incoming message
    pub async fn handle_message(&self, message: GossipMessage, from: SocketAddr) -> Result<()> {
        debug!("Handling {} message from {}", 
               format!("{:?}", message.message_type), from);
        
        // Check message age
        if message.is_expired(self.config.max_message_age.as_secs()) {
            debug!("Ignoring expired message from {}", from);
            return Ok(());
        }
        
        match message.message_type {
            MessageType::Ping => self.handle_ping(message, from).await,
            MessageType::Ack => self.handle_ack(message, from).await,
            MessageType::IndirectPing => self.handle_indirect_ping(message, from).await,
            MessageType::Membership => self.handle_membership(message, from).await,
            MessageType::Join => self.handle_join(message, from).await,
            MessageType::Leave => self.handle_leave(message, from).await,
            MessageType::UserData => self.handle_user_data(message, from).await,
        }
    }
    
    /// Handle ping message
    async fn handle_ping(&self, message: GossipMessage, from: SocketAddr) -> Result<()> {
        // Send ACK response
        let sequence = self.next_sequence().await;
        let ack = GossipMessage::ack(self.local_node_id.clone(), message.source, sequence);
        
        self.transport.send_to(&ack, from).await?;
        Ok(())
    }
    
    /// Handle ACK message
    async fn handle_ack(&self, _message: GossipMessage, _from: SocketAddr) -> Result<()> {
        // Remove from pending probes
        // In a real implementation, we would match the sequence number
        Ok(())
    }
    
    /// Handle indirect ping message
    async fn handle_indirect_ping(&self, message: GossipMessage, _from: SocketAddr) -> Result<()> {
        if let MessagePayload::IndirectPing { target, target_addr } = message.payload {
            // Forward ping to target
            let sequence = self.next_sequence().await;
            let ping = GossipMessage::ping(self.local_node_id.clone(), target.clone(), sequence);
            
            if let Err(e) = self.transport.send_to(&ping, target_addr).await {
                warn!("Failed to forward indirect ping to {}: {}", target, e);
            }
        }
        Ok(())
    }
    
    /// Handle membership message
    async fn handle_membership(&self, message: GossipMessage, _from: SocketAddr) -> Result<()> {
        if let MessagePayload::Membership(updates) = message.payload {
            let mut membership = self.membership.write().await;
            
            for update in updates {
                let member = Member {
                    node_id: update.node_id,
                    gossip_addr: update.gossip_addr,
                    state: update.state,
                    incarnation: update.incarnation,
                    last_updated: Instant::now(),
                    metadata: update.metadata,
                };
                
                membership.update_member(member);
            }
        }
        Ok(())
    }
    
    /// Handle join message
    async fn handle_join(&self, message: GossipMessage, _from: SocketAddr) -> Result<()> {
        if let MessagePayload::Join { node_id, gossip_addr, metadata } = message.payload {
            let mut membership = self.membership.write().await;
            let mut member = Member::new(node_id.clone(), gossip_addr);
            
            for (key, value) in metadata {
                member.add_metadata(key, value);
            }
            
            if membership.update_member(member) {
                if let Some(tx) = &self.event_tx {
                    let _ = tx.send(ProtocolEvent::MemberJoined(node_id));
                }
            }
        }
        Ok(())
    }
    
    /// Handle leave message
    async fn handle_leave(&self, message: GossipMessage, _from: SocketAddr) -> Result<()> {
        if let MessagePayload::Leave { node_id, incarnation: _ } = message.payload {
            let mut membership = self.membership.write().await;
            
            if membership.mark_left(&node_id) {
                if let Some(tx) = &self.event_tx {
                    let _ = tx.send(ProtocolEvent::MemberLeft(node_id));
                }
            }
        }
        Ok(())
    }
    
    /// Handle user data message
    async fn handle_user_data(&self, message: GossipMessage, _from: SocketAddr) -> Result<()> {
        if let MessagePayload::UserData { key, value, version } = message.payload {
            if let Some(tx) = &self.event_tx {
                let _ = tx.send(ProtocolEvent::UserData {
                    key,
                    value,
                    version,
                    source: message.source,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::UdpTransport;
    use mesh_core::NodeId;
    use std::net::SocketAddr;

    #[tokio::test]
    async fn test_swim_protocol_creation() {
        let node_id = NodeId::new("test-node");
        let addr = "127.0.0.1:0".parse::<SocketAddr>().unwrap();
        let config = GossipConfig::default();
        let transport = Arc::new(UdpTransport::new(addr, 1400));
        
        let protocol = SwimProtocol::new(node_id.clone(), addr, config, transport);
        
        assert_eq!(protocol.local_node_id, node_id);
        assert_eq!(protocol.local_addr, addr);
        assert!(!*protocol.running.read().await);
    }

    #[tokio::test]
    async fn test_protocol_start_stop() {
        let node_id = NodeId::new("test-node");
        let addr = "127.0.0.1:0".parse::<SocketAddr>().unwrap();
        let config = GossipConfig::default();
        let mut transport = UdpTransport::new(addr, 1400);
        transport.start().await.unwrap();
        let transport = Arc::new(transport);
        
        let mut protocol = SwimProtocol::new(node_id, addr, config, transport);
        
        assert!(protocol.start().await.is_ok());
        assert!(*protocol.running.read().await);
        
        assert!(protocol.stop().await.is_ok());
        assert!(!*protocol.running.read().await);
    }

    #[tokio::test]
    async fn test_membership_access() {
        let node_id = NodeId::new("test-node");
        let addr = "127.0.0.1:0".parse::<SocketAddr>().unwrap();
        let config = GossipConfig::default();
        let transport = Arc::new(UdpTransport::new(addr, 1400));
        
        let protocol = SwimProtocol::new(node_id.clone(), addr, config, transport);
        let membership = protocol.membership().await;
        
        assert_eq!(membership.local_member().node_id, node_id);
        assert_eq!(membership.members().len(), 1);
    }
}
