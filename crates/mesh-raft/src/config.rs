//! Raft configuration

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Raft node configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftConfig {
    /// Node ID (must be unique in the cluster)
    pub node_id: u64,
    
    /// Cluster peers (node_id -> address mapping)
    pub peers: Vec<RaftPeer>,
    
    /// Election timeout range
    pub election_timeout: Duration,
    
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    
    /// Maximum number of entries per append
    pub max_entries_per_append: usize,
    
    /// Maximum size of uncommitted entries in bytes
    pub max_uncommitted_size: u64,
    
    /// Storage configuration
    pub storage: StorageConfig,
    
    /// Snapshot configuration
    pub snapshot: SnapshotConfig,
    
    /// Network configuration
    pub network: NetworkConfig,
    
    /// Enable pre-vote to avoid unnecessary leader elections
    pub pre_vote: bool,
    
    /// Check quorum to ensure leader has majority support
    pub check_quorum: bool,
    
    /// Batch append entries for better performance
    pub batch_append: bool,
}

/// Raft peer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftPeer {
    /// Peer node ID
    pub id: u64,
    
    /// Peer address
    pub address: String,
    
    /// Whether this peer is a voter
    pub voter: bool,
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Storage type
    pub storage_type: StorageType,
    
    /// Data directory for disk storage
    pub data_dir: Option<PathBuf>,
    
    /// Maximum log entries to keep in memory
    pub max_memory_entries: usize,
    
    /// Sync writes to disk
    pub sync_writes: bool,
    
    /// Compress log entries
    pub compress_entries: bool,
}

/// Storage type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageType {
    /// In-memory storage (for testing)
    Memory,
    
    /// Disk-based storage
    Disk,
}

/// Snapshot configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotConfig {
    /// Enable automatic snapshots
    pub enabled: bool,
    
    /// Threshold of log entries to trigger snapshot
    pub threshold: u64,
    
    /// Maximum number of snapshots to keep
    pub max_snapshots: usize,
    
    /// Compress snapshots
    pub compress: bool,
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Request timeout
    pub request_timeout: Duration,
    
    /// Connection timeout
    pub connection_timeout: Duration,
    
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    
    /// Retry configuration
    pub retry: RetryConfig,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,
    
    /// Base delay between retries
    pub base_delay: Duration,
    
    /// Maximum delay between retries
    pub max_delay: Duration,
    
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            node_id: 1,
            peers: Vec::new(),
            election_timeout: Duration::from_millis(1000),
            heartbeat_interval: Duration::from_millis(100),
            max_entries_per_append: 1000,
            max_uncommitted_size: 1024 * 1024, // 1MB
            storage: StorageConfig::default(),
            snapshot: SnapshotConfig::default(),
            network: NetworkConfig::default(),
            pre_vote: true,
            check_quorum: true,
            batch_append: true,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            storage_type: StorageType::Memory,
            data_dir: None,
            max_memory_entries: 10000,
            sync_writes: true,
            compress_entries: false,
        }
    }
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 1000,
            max_snapshots: 3,
            compress: true,
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(5),
            connection_timeout: Duration::from_secs(3),
            max_concurrent_requests: 100,
            retry: RetryConfig::default(),
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
        }
    }
}

/// Builder for RaftConfig
pub struct RaftConfigBuilder {
    config: RaftConfig,
}

impl RaftConfigBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: RaftConfig::default(),
        }
    }

    /// Set node ID
    pub fn node_id(mut self, id: u64) -> Self {
        self.config.node_id = id;
        self
    }

    /// Add a peer
    pub fn add_peer(mut self, id: u64, address: impl Into<String>, voter: bool) -> Self {
        self.config.peers.push(RaftPeer {
            id,
            address: address.into(),
            voter,
        });
        self
    }

    /// Set election timeout
    pub fn election_timeout(mut self, timeout: Duration) -> Self {
        self.config.election_timeout = timeout;
        self
    }

    /// Set heartbeat interval
    pub fn heartbeat_interval(mut self, interval: Duration) -> Self {
        self.config.heartbeat_interval = interval;
        self
    }

    /// Set storage type
    pub fn storage_type(mut self, storage_type: StorageType) -> Self {
        self.config.storage.storage_type = storage_type;
        self
    }

    /// Set data directory for disk storage
    pub fn data_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.storage.data_dir = Some(dir.into());
        self
    }

    /// Enable or disable snapshots
    pub fn enable_snapshots(mut self, enabled: bool) -> Self {
        self.config.snapshot.enabled = enabled;
        self
    }

    /// Set snapshot threshold
    pub fn snapshot_threshold(mut self, threshold: u64) -> Self {
        self.config.snapshot.threshold = threshold;
        self
    }

    /// Enable or disable pre-vote
    pub fn pre_vote(mut self, enabled: bool) -> Self {
        self.config.pre_vote = enabled;
        self
    }

    /// Enable or disable check quorum
    pub fn check_quorum(mut self, enabled: bool) -> Self {
        self.config.check_quorum = enabled;
        self
    }

    /// Build the configuration
    pub fn build(self) -> RaftConfig {
        self.config
    }
}

impl Default for RaftConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate raft configuration
pub fn validate_config(config: &RaftConfig) -> Result<(), String> {
    if config.node_id == 0 {
        return Err("Node ID must be greater than 0".to_string());
    }

    if config.election_timeout <= config.heartbeat_interval {
        return Err("Election timeout must be greater than heartbeat interval".to_string());
    }

    if config.max_entries_per_append == 0 {
        return Err("Max entries per append must be greater than 0".to_string());
    }

    if config.max_uncommitted_size == 0 {
        return Err("Max uncommitted size must be greater than 0".to_string());
    }

    // Check for duplicate peer IDs
    let mut peer_ids = std::collections::HashSet::new();
    for peer in &config.peers {
        if peer.id == config.node_id {
            return Err("Peer ID cannot be the same as node ID".to_string());
        }
        if !peer_ids.insert(peer.id) {
            return Err(format!("Duplicate peer ID: {}", peer.id));
        }
    }

    // Validate storage configuration
    if let StorageType::Disk = config.storage.storage_type {
        if config.storage.data_dir.is_none() {
            return Err("Data directory must be specified for disk storage".to_string());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RaftConfig::default();
        assert_eq!(config.node_id, 1);
        assert!(config.peers.is_empty());
        assert_eq!(config.election_timeout, Duration::from_millis(1000));
        assert_eq!(config.heartbeat_interval, Duration::from_millis(100));
        assert!(config.pre_vote);
        assert!(config.check_quorum);
    }

    #[test]
    fn test_config_builder() {
        let config = RaftConfigBuilder::new()
            .node_id(5)
            .add_peer(1, "127.0.0.1:8001", true)
            .add_peer(2, "127.0.0.1:8002", true)
            .election_timeout(Duration::from_millis(2000))
            .heartbeat_interval(Duration::from_millis(200))
            .storage_type(StorageType::Disk)
            .data_dir("/tmp/raft")
            .enable_snapshots(false)
            .pre_vote(false)
            .build();

        assert_eq!(config.node_id, 5);
        assert_eq!(config.peers.len(), 2);
        assert_eq!(config.peers[0].id, 1);
        assert_eq!(config.peers[0].address, "127.0.0.1:8001");
        assert_eq!(config.election_timeout, Duration::from_millis(2000));
        assert_eq!(config.heartbeat_interval, Duration::from_millis(200));
        assert!(matches!(config.storage.storage_type, StorageType::Disk));
        assert!(!config.snapshot.enabled);
        assert!(!config.pre_vote);
    }

    #[test]
    fn test_config_validation() {
        let mut config = RaftConfig::default();
        assert!(validate_config(&config).is_ok());

        // Test invalid node ID
        config.node_id = 0;
        assert!(validate_config(&config).is_err());

        // Test invalid timeout relationship
        config.node_id = 1;
        config.election_timeout = Duration::from_millis(50);
        config.heartbeat_interval = Duration::from_millis(100);
        assert!(validate_config(&config).is_err());

        // Test duplicate peer ID
        config.election_timeout = Duration::from_millis(1000);
        config.peers = vec![
            RaftPeer { id: 2, address: "127.0.0.1:8001".to_string(), voter: true },
            RaftPeer { id: 2, address: "127.0.0.1:8002".to_string(), voter: true },
        ];
        assert!(validate_config(&config).is_err());

        // Test peer ID same as node ID
        config.peers = vec![
            RaftPeer { id: 1, address: "127.0.0.1:8001".to_string(), voter: true },
        ];
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_storage_config() {
        let config = StorageConfig::default();
        assert!(matches!(config.storage_type, StorageType::Memory));
        assert!(config.data_dir.is_none());
        assert_eq!(config.max_memory_entries, 10000);
        assert!(config.sync_writes);
    }

    #[test]
    fn test_snapshot_config() {
        let config = SnapshotConfig::default();
        assert!(config.enabled);
        assert_eq!(config.threshold, 1000);
        assert_eq!(config.max_snapshots, 3);
        assert!(config.compress);
    }
}
