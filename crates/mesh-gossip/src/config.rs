//! Configuration for the gossip protocol

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for the SWIM gossip protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipConfig {
    /// Interval between gossip rounds
    pub gossip_interval: Duration,
    
    /// Timeout for considering a node failed
    pub failure_timeout: Duration,
    
    /// Timeout for probe requests
    pub probe_timeout: Duration,
    
    /// Number of nodes to gossip with in each round
    pub gossip_fanout: usize,
    
    /// Number of nodes to probe for indirect probing
    pub probe_fanout: usize,
    
    /// Maximum size of gossip packets in bytes
    pub max_gossip_packet_size: usize,
    
    /// Maximum number of members to track
    pub max_members: usize,
    
    /// Enable TCP fallback for large messages
    pub tcp_fallback: bool,
    
    /// TCP port for fallback (0 = auto-assign)
    pub tcp_port: u16,
    
    /// Enable message encryption
    pub enable_encryption: bool,
    
    /// Shared secret for message authentication (base64 encoded)
    pub shared_secret: Option<String>,
    
    /// Maximum age of gossip messages to accept
    pub max_message_age: Duration,
    
    /// Interval for cleanup of old members
    pub cleanup_interval: Duration,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            gossip_interval: Duration::from_millis(200),
            failure_timeout: Duration::from_secs(10),
            probe_timeout: Duration::from_millis(500),
            gossip_fanout: 3,
            probe_fanout: 3,
            max_gossip_packet_size: 1400, // Safe UDP packet size
            max_members: 1000,
            tcp_fallback: true,
            tcp_port: 0, // Auto-assign
            enable_encryption: false,
            shared_secret: None,
            max_message_age: Duration::from_secs(30),
            cleanup_interval: Duration::from_secs(60),
        }
    }
}

impl GossipConfig {
    /// Create a new gossip configuration with custom settings
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the gossip interval
    pub fn with_gossip_interval(mut self, interval: Duration) -> Self {
        self.gossip_interval = interval;
        self
    }
    
    /// Set the failure timeout
    pub fn with_failure_timeout(mut self, timeout: Duration) -> Self {
        self.failure_timeout = timeout;
        self
    }
    
    /// Set the probe timeout
    pub fn with_probe_timeout(mut self, timeout: Duration) -> Self {
        self.probe_timeout = timeout;
        self
    }
    
    /// Set the gossip fanout
    pub fn with_gossip_fanout(mut self, fanout: usize) -> Self {
        self.gossip_fanout = fanout;
        self
    }
    
    /// Set the probe fanout
    pub fn with_probe_fanout(mut self, fanout: usize) -> Self {
        self.probe_fanout = fanout;
        self
    }
    
    /// Enable encryption with a shared secret
    pub fn with_encryption(mut self, secret: String) -> Self {
        self.enable_encryption = true;
        self.shared_secret = Some(secret);
        self
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.gossip_interval.is_zero() {
            return Err("Gossip interval must be greater than zero".to_string());
        }
        
        if self.failure_timeout.is_zero() {
            return Err("Failure timeout must be greater than zero".to_string());
        }
        
        if self.probe_timeout.is_zero() {
            return Err("Probe timeout must be greater than zero".to_string());
        }
        
        if self.gossip_fanout == 0 {
            return Err("Gossip fanout must be greater than zero".to_string());
        }
        
        if self.probe_fanout == 0 {
            return Err("Probe fanout must be greater than zero".to_string());
        }
        
        if self.max_gossip_packet_size < 64 {
            return Err("Max gossip packet size must be at least 64 bytes".to_string());
        }
        
        if self.max_members == 0 {
            return Err("Max members must be greater than zero".to_string());
        }
        
        if self.enable_encryption && self.shared_secret.is_none() {
            return Err("Shared secret required when encryption is enabled".to_string());
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GossipConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.gossip_interval.as_millis() > 0);
        assert!(config.failure_timeout.as_millis() > 0);
        assert!(!config.enable_encryption);
    }

    #[test]
    fn test_config_builder() {
        let config = GossipConfig::new()
            .with_gossip_interval(Duration::from_millis(100))
            .with_failure_timeout(Duration::from_secs(5))
            .with_gossip_fanout(5);
        
        assert_eq!(config.gossip_interval, Duration::from_millis(100));
        assert_eq!(config.failure_timeout, Duration::from_secs(5));
        assert_eq!(config.gossip_fanout, 5);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let mut config = GossipConfig::default();
        
        // Test zero gossip interval
        config.gossip_interval = Duration::from_millis(0);
        assert!(config.validate().is_err());
        
        // Test encryption without secret
        config = GossipConfig::default();
        config.enable_encryption = true;
        config.shared_secret = None;
        assert!(config.validate().is_err());
        
        // Test valid encryption config
        config.shared_secret = Some("test-secret".to_string());
        assert!(config.validate().is_ok());
    }
}
