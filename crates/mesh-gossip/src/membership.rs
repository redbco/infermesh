//! Membership management for the gossip protocol

use mesh_core::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// State of a member in the gossip protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemberState {
    /// Node is alive and responding
    Alive,
    /// Node is suspected to be dead
    Suspect,
    /// Node is confirmed dead
    Dead,
    /// Node has left the cluster gracefully
    Left,
}

impl MemberState {
    /// Check if the member is considered active (alive or suspect)
    pub fn is_active(&self) -> bool {
        matches!(self, MemberState::Alive | MemberState::Suspect)
    }
    
    /// Check if the member is considered inactive (dead or left)
    pub fn is_inactive(&self) -> bool {
        matches!(self, MemberState::Dead | MemberState::Left)
    }
}

/// Information about a cluster member
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Member {
    /// Unique node identifier
    pub node_id: NodeId,
    
    /// Network address for gossip communication
    pub gossip_addr: SocketAddr,
    
    /// Current state of the member
    pub state: MemberState,
    
    /// Incarnation number for conflict resolution
    pub incarnation: u64,
    
    /// Last time this member was updated (as Unix timestamp)
    #[serde(with = "instant_serde")]
    pub last_updated: Instant,
    
    /// Metadata associated with the member
    pub metadata: HashMap<String, String>,
}

impl Member {
    /// Create a new member
    pub fn new(node_id: NodeId, gossip_addr: SocketAddr) -> Self {
        Self {
            node_id,
            gossip_addr,
            state: MemberState::Alive,
            incarnation: 0,
            last_updated: Instant::now(),
            metadata: HashMap::new(),
        }
    }
    
    /// Update the member's state
    pub fn update_state(&mut self, state: MemberState, incarnation: u64) {
        if incarnation > self.incarnation || (incarnation == self.incarnation && state != self.state) {
            self.state = state;
            self.incarnation = incarnation;
            self.last_updated = Instant::now();
        }
    }
    
    /// Check if the member is stale (hasn't been updated recently)
    pub fn is_stale(&self, max_age: Duration) -> bool {
        self.last_updated.elapsed() > max_age
    }
    
    /// Add metadata to the member
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
        self.last_updated = Instant::now();
    }
    
    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
}

/// Manages the membership of the gossip cluster
#[derive(Debug, Clone)]
pub struct Membership {
    /// Local node information
    local_member: Member,
    
    /// Map of all known members
    members: HashMap<NodeId, Member>,
    
    /// Maximum number of members to track
    max_members: usize,
}

impl Membership {
    /// Create a new membership manager
    pub fn new(local_node_id: NodeId, local_gossip_addr: SocketAddr, max_members: usize) -> Self {
        let local_member = Member::new(local_node_id.clone(), local_gossip_addr);
        let mut members = HashMap::new();
        members.insert(local_node_id.clone(), local_member.clone());
        
        Self {
            local_member,
            members,
            max_members,
        }
    }
    
    /// Get the local member
    pub fn local_member(&self) -> &Member {
        &self.local_member
    }
    
    /// Get all members
    pub fn members(&self) -> &HashMap<NodeId, Member> {
        &self.members
    }
    
    /// Get a specific member
    pub fn get_member(&self, node_id: &NodeId) -> Option<&Member> {
        self.members.get(node_id)
    }
    
    /// Get a mutable reference to a specific member
    pub fn get_member_mut(&mut self, node_id: &NodeId) -> Option<&mut Member> {
        self.members.get_mut(node_id)
    }
    
    /// Add or update a member
    pub fn update_member(&mut self, member: Member) -> bool {
        let node_id = member.node_id.clone();
        
        // Don't allow updates to local member from external sources
        if node_id == self.local_member.node_id {
            return false;
        }
        
        // Check if we have room for new members
        if !self.members.contains_key(&node_id) && self.members.len() >= self.max_members {
            warn!("Cannot add member {}: membership table full", node_id);
            return false;
        }
        
        match self.members.get_mut(&node_id) {
            Some(existing) => {
                // Update existing member if incarnation is newer
                if member.incarnation > existing.incarnation {
                    *existing = member;
                    debug!("Updated member {}", node_id);
                    true
                } else if member.incarnation == existing.incarnation && member.state != existing.state {
                    // Same incarnation but different state - take the "worse" state
                    let should_update = match (existing.state, member.state) {
                        (MemberState::Alive, MemberState::Suspect) => true,
                        (MemberState::Alive, MemberState::Dead) => true,
                        (MemberState::Suspect, MemberState::Dead) => true,
                        _ => false,
                    };
                    
                    if should_update {
                        existing.state = member.state;
                        existing.last_updated = Instant::now();
                        debug!("Updated member {} state to {:?}", node_id, member.state);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            None => {
                // Add new member
                info!("Added new member {}", node_id);
                self.members.insert(node_id, member);
                true
            }
        }
    }
    
    /// Mark a member as suspect
    pub fn mark_suspect(&mut self, node_id: &NodeId) -> bool {
        if let Some(member) = self.members.get_mut(node_id) {
            if member.state == MemberState::Alive {
                member.state = MemberState::Suspect;
                member.last_updated = Instant::now();
                debug!("Marked member {} as suspect", node_id);
                return true;
            }
        }
        false
    }
    
    /// Mark a member as dead
    pub fn mark_dead(&mut self, node_id: &NodeId) -> bool {
        if let Some(member) = self.members.get_mut(node_id) {
            if member.state != MemberState::Dead {
                member.state = MemberState::Dead;
                member.last_updated = Instant::now();
                info!("Marked member {} as dead", node_id);
                return true;
            }
        }
        false
    }
    
    /// Mark a member as left
    pub fn mark_left(&mut self, node_id: &NodeId) -> bool {
        if let Some(member) = self.members.get_mut(node_id) {
            member.state = MemberState::Left;
            member.last_updated = Instant::now();
            info!("Marked member {} as left", node_id);
            return true;
        }
        false
    }
    
    /// Get alive members
    pub fn alive_members(&self) -> Vec<&Member> {
        self.members
            .values()
            .filter(|m| m.state == MemberState::Alive)
            .collect()
    }
    
    /// Get suspect members
    pub fn suspect_members(&self) -> Vec<&Member> {
        self.members
            .values()
            .filter(|m| m.state == MemberState::Suspect)
            .collect()
    }
    
    /// Get dead members
    pub fn dead_members(&self) -> Vec<&Member> {
        self.members
            .values()
            .filter(|m| m.state == MemberState::Dead)
            .collect()
    }
    
    /// Get active members (alive or suspect)
    pub fn active_members(&self) -> Vec<&Member> {
        self.members
            .values()
            .filter(|m| m.state.is_active())
            .collect()
    }
    
    /// Remove stale members
    pub fn cleanup_stale_members(&mut self, max_age: Duration) -> usize {
        let initial_count = self.members.len();
        
        // Don't remove local member
        let local_id = self.local_member.node_id.clone();
        
        self.members.retain(|node_id, member| {
            if *node_id == local_id {
                true // Keep local member
            } else if member.state.is_inactive() && member.is_stale(max_age) {
                debug!("Removing stale member {}", node_id);
                false
            } else {
                true
            }
        });
        
        let removed_count = initial_count - self.members.len();
        if removed_count > 0 {
            info!("Cleaned up {} stale members", removed_count);
        }
        
        removed_count
    }
    
    /// Get a random subset of active members for gossip
    pub fn random_members(&self, count: usize) -> Vec<&Member> {
        use rand::seq::SliceRandom;
        
        let mut active: Vec<&Member> = self.active_members();
        
        // Remove local member from gossip targets
        active.retain(|m| m.node_id != self.local_member.node_id);
        
        let mut rng = rand::thread_rng();
        active.shuffle(&mut rng);
        
        active.into_iter().take(count).collect()
    }
    
    /// Update local member metadata
    pub fn update_local_metadata(&mut self, key: String, value: String) {
        self.local_member.add_metadata(key.clone(), value.clone());
        
        // Also update in the members map
        if let Some(local) = self.members.get_mut(&self.local_member.node_id) {
            local.add_metadata(key, value);
        }
    }
    
    /// Increment local incarnation number
    pub fn increment_incarnation(&mut self) {
        self.local_member.incarnation += 1;
        self.local_member.last_updated = Instant::now();
        
        // Also update in the members map
        if let Some(local) = self.members.get_mut(&self.local_member.node_id) {
            local.incarnation = self.local_member.incarnation;
            local.last_updated = self.local_member.last_updated;
        }
    }
    
    /// Get membership statistics
    pub fn stats(&self) -> MembershipStats {
        let alive = self.alive_members().len();
        let suspect = self.suspect_members().len();
        let dead = self.dead_members().len();
        let left = self.members.values().filter(|m| m.state == MemberState::Left).count();
        
        MembershipStats {
            total: self.members.len(),
            alive,
            suspect,
            dead,
            left,
        }
    }
}

/// Statistics about the membership
#[derive(Debug, Clone)]
pub struct MembershipStats {
    pub total: usize,
    pub alive: usize,
    pub suspect: usize,
    pub dead: usize,
    pub left: usize,
}

/// Serde module for Instant serialization
mod instant_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Instant;
    
    pub fn serialize<S>(instant: &Instant, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Convert Instant to a duration since a reference point
        // For simplicity, we'll use a fixed reference point
        let duration_since_epoch = instant.elapsed().as_secs();
        duration_since_epoch.serialize(serializer)
    }
    
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Instant, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        // Create an Instant that's approximately correct
        // Note: This is a simplified approach for serialization
        Ok(Instant::now() - std::time::Duration::from_secs(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_core::NodeId;
    use std::net::SocketAddr;

    #[test]
    fn test_member_creation() {
        let node_id = NodeId::new("test-node");
        let addr = "127.0.0.1:7946".parse::<SocketAddr>().unwrap();
        let member = Member::new(node_id.clone(), addr);
        
        assert_eq!(member.node_id, node_id);
        assert_eq!(member.gossip_addr, addr);
        assert_eq!(member.state, MemberState::Alive);
        assert_eq!(member.incarnation, 0);
    }

    #[test]
    fn test_member_state_transitions() {
        let node_id = NodeId::new("test-node");
        let addr = "127.0.0.1:7946".parse::<SocketAddr>().unwrap();
        let mut member = Member::new(node_id, addr);
        
        // Test state update with higher incarnation
        member.update_state(MemberState::Suspect, 1);
        assert_eq!(member.state, MemberState::Suspect);
        assert_eq!(member.incarnation, 1);
        
        // Test state update with same incarnation
        member.update_state(MemberState::Dead, 1);
        assert_eq!(member.state, MemberState::Dead);
        assert_eq!(member.incarnation, 1);
        
        // Test state update with lower incarnation (should be ignored)
        member.update_state(MemberState::Alive, 0);
        assert_eq!(member.state, MemberState::Dead);
        assert_eq!(member.incarnation, 1);
    }

    #[test]
    fn test_membership_management() {
        let local_id = NodeId::new("test-node");
        let local_addr = "127.0.0.1:7946".parse::<SocketAddr>().unwrap();
        let mut membership = Membership::new(local_id.clone(), local_addr, 100);
        
        // Test adding new member
        let remote_id = NodeId::new("remote-node");
        let remote_addr = "127.0.0.1:7947".parse::<SocketAddr>().unwrap();
        let remote_member = Member::new(remote_id.clone(), remote_addr);
        
        assert!(membership.update_member(remote_member));
        assert_eq!(membership.members().len(), 2);
        
        // Test marking member as suspect
        assert!(membership.mark_suspect(&remote_id));
        assert_eq!(membership.get_member(&remote_id).unwrap().state, MemberState::Suspect);
        
        // Test marking member as dead
        assert!(membership.mark_dead(&remote_id));
        assert_eq!(membership.get_member(&remote_id).unwrap().state, MemberState::Dead);
    }

    #[test]
    fn test_membership_stats() {
        let local_id = NodeId::new("test-node");
        let local_addr = "127.0.0.1:7946".parse::<SocketAddr>().unwrap();
        let mut membership = Membership::new(local_id, local_addr, 100);
        
        // Add some members in different states
        for i in 0..5 {
            let node_id = NodeId::new(format!("test-node-{}", i));
            let addr = format!("127.0.0.1:{}", 7947 + i).parse::<SocketAddr>().unwrap();
            let mut member = Member::new(node_id.clone(), addr);
            
            match i {
                0 | 1 => member.state = MemberState::Alive,
                2 => member.state = MemberState::Suspect,
                3 => member.state = MemberState::Dead,
                4 => member.state = MemberState::Left,
                _ => {}
            }
            
            membership.update_member(member);
        }
        
        let stats = membership.stats();
        assert_eq!(stats.total, 6); // 5 + local
        assert_eq!(stats.alive, 3); // 2 + local
        assert_eq!(stats.suspect, 1);
        assert_eq!(stats.dead, 1);
        assert_eq!(stats.left, 1);
    }
}
