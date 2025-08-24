//! Raft storage implementations

use crate::{Result, RaftError};
use serde::{Deserialize, Serialize};

use std::path::PathBuf;
use std::sync::Arc;
use raft::{
    eraftpb::{ConfState, Entry, HardState, Snapshot},
    Error as TikvRaftError,
    GetEntriesContext, RaftState, Storage, StorageError,
};
use tokio::sync::RwLock;

/// Raft storage backend enum to replace trait objects
#[derive(Debug)]
pub enum RaftStorageBackend {
    Memory(MemoryStorage),
    Disk(DiskStorage),
}

impl RaftStorageBackend {
    /// Initialize storage
    pub fn initialize(&mut self) -> Result<()> {
        match self {
            Self::Memory(storage) => storage.initialize(),
            Self::Disk(storage) => storage.initialize(),
        }
    }
    
    /// Close storage
    pub fn close(&mut self) -> Result<()> {
        match self {
            Self::Memory(storage) => storage.close(),
            Self::Disk(storage) => storage.close(),
        }
    }
    
    /// Get storage statistics
    pub fn stats(&self) -> StorageStats {
        match self {
            Self::Memory(storage) => storage.stats(),
            Self::Disk(storage) => storage.stats(),
        }
    }
    
    /// Compact log entries up to the given index
    pub fn compact(&mut self, compact_index: u64) -> Result<()> {
        match self {
            Self::Memory(storage) => storage.compact(compact_index),
            Self::Disk(storage) => storage.compact(compact_index),
        }
    }
    
    /// Create a snapshot at the given index
    pub fn create_snapshot(&mut self, index: u64, data: Vec<u8>) -> Result<()> {
        match self {
            Self::Memory(storage) => storage.create_snapshot(index, data),
            Self::Disk(storage) => storage.create_snapshot(index, data),
        }
    }
    
    /// Apply a snapshot
    pub fn apply_snapshot(&mut self, snapshot: Snapshot) -> Result<()> {
        match self {
            Self::Memory(storage) => storage.apply_snapshot(snapshot),
            Self::Disk(storage) => storage.apply_snapshot(snapshot),
        }
    }
}

// Implement the tikv-raft Storage trait for RaftStorageBackend
impl Storage for RaftStorageBackend {
    fn initial_state(&self) -> raft::Result<RaftState> {
        match self {
            Self::Memory(storage) => storage.initial_state(),
            Self::Disk(storage) => storage.initial_state(),
        }
    }

    fn entries(
        &self,
        low: u64,
        high: u64,
        max_size: impl Into<Option<u64>>,
        _context: GetEntriesContext,
    ) -> raft::Result<Vec<Entry>> {
        match self {
            Self::Memory(storage) => storage.entries(low, high, max_size, _context),
            Self::Disk(storage) => storage.entries(low, high, max_size, _context),
        }
    }

    fn term(&self, idx: u64) -> raft::Result<u64> {
        match self {
            Self::Memory(storage) => storage.term(idx),
            Self::Disk(storage) => storage.term(idx),
        }
    }

    fn first_index(&self) -> raft::Result<u64> {
        match self {
            Self::Memory(storage) => storage.first_index(),
            Self::Disk(storage) => storage.first_index(),
        }
    }

    fn last_index(&self) -> raft::Result<u64> {
        match self {
            Self::Memory(storage) => storage.last_index(),
            Self::Disk(storage) => storage.last_index(),
        }
    }

    fn snapshot(&self, request_index: u64, to: u64) -> raft::Result<Snapshot> {
        match self {
            Self::Memory(storage) => storage.snapshot(request_index, to),
            Self::Disk(storage) => storage.snapshot(request_index, to),
        }
    }
}

/// Raft storage trait for individual implementations
pub trait RaftStorage: Send + Sync {
    /// Initialize storage
    fn initialize(&mut self) -> Result<()>;
    
    /// Close storage
    fn close(&mut self) -> Result<()>;
    
    /// Get storage statistics
    fn stats(&self) -> StorageStats;
    
    /// Compact log entries up to the given index
    fn compact(&mut self, compact_index: u64) -> Result<()>;
    
    /// Create a snapshot at the given index
    fn create_snapshot(&mut self, index: u64, data: Vec<u8>) -> Result<()>;
    
    /// Apply a snapshot
    fn apply_snapshot(&mut self, snapshot: Snapshot) -> Result<()>;
}

/// Storage statistics
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Number of log entries
    pub entry_count: u64,
    
    /// Size of log entries in bytes
    pub entry_size_bytes: u64,
    
    /// Number of snapshots
    pub snapshot_count: u64,
    
    /// Size of snapshots in bytes
    pub snapshot_size_bytes: u64,
    
    /// First log index
    pub first_index: u64,
    
    /// Last log index
    pub last_index: u64,
    
    /// Applied index
    pub applied_index: u64,
    
    /// Committed index
    pub committed_index: u64,
}

/// In-memory storage implementation
#[derive(Debug)]
pub struct MemoryStorage {
    /// Raft hard state
    hard_state: Arc<RwLock<HardState>>,
    
    /// Configuration state
    conf_state: Arc<RwLock<ConfState>>,
    
    /// Log entries
    entries: Arc<RwLock<Vec<Entry>>>,
    
    /// Current snapshot
    snapshot: Arc<RwLock<Snapshot>>,
    
    /// Applied index
    applied_index: Arc<RwLock<u64>>,
    
    /// Storage statistics
    stats: Arc<RwLock<StorageStats>>,
}

/// Disk-based storage implementation
#[derive(Debug)]
pub struct DiskStorage {
    /// Data directory
    data_dir: PathBuf,
    
    /// In-memory cache
    memory_storage: MemoryStorage,
    
    /// Sync writes to disk
    sync_writes: bool,
    
    /// Compress entries
    compress_entries: bool,
    
    /// Maximum entries in memory
    max_memory_entries: usize,
}

/// Persisted raft state (without serde derives due to tikv-raft types)
#[derive(Debug, Clone)]
struct PersistedState {
    hard_state: HardState,
    conf_state: ConfState,
    entries: Vec<Entry>,
    snapshot: Option<Snapshot>,
    applied_index: u64,
}

/// Serializable version of raft state
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializablePersistedState {
    hard_state: SerializableHardState,
    conf_state: Vec<u64>, // Simplified representation
    entries: Vec<SerializableEntry>,
    snapshot_data: Option<Vec<u8>>,
    applied_index: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableHardState {
    term: u64,
    vote: u64,
    commit: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableEntry {
    entry_type: i32,
    term: u64,
    index: u64,
    data: Vec<u8>,
    context: Vec<u8>,
}

impl From<&HardState> for SerializableHardState {
    fn from(hs: &HardState) -> Self {
        Self {
            term: hs.term,
            vote: hs.vote,
            commit: hs.commit,
        }
    }
}

impl From<&SerializableHardState> for HardState {
    fn from(shs: &SerializableHardState) -> Self {
        let mut hs = HardState::default();
        hs.term = shs.term;
        hs.vote = shs.vote;
        hs.commit = shs.commit;
        hs
    }
}

impl From<&Entry> for SerializableEntry {
    fn from(entry: &Entry) -> Self {
        Self {
            entry_type: entry.entry_type as i32,
            term: entry.term,
            index: entry.index,
            data: entry.data.to_vec(),
            context: entry.context.to_vec(),
        }
    }
}

impl From<&SerializableEntry> for Entry {
    fn from(se: &SerializableEntry) -> Self {
        let mut entry = Entry::default();
        entry.set_entry_type(match se.entry_type {
            0 => raft::eraftpb::EntryType::EntryNormal,
            1 => raft::eraftpb::EntryType::EntryConfChange,
            2 => raft::eraftpb::EntryType::EntryConfChangeV2,
            _ => raft::eraftpb::EntryType::EntryNormal, // Default fallback
        });
        entry.term = se.term;
        entry.index = se.index;
        entry.data = se.data.clone().into();
        entry.context = se.context.clone().into();
        entry
    }
}

impl From<&PersistedState> for SerializablePersistedState {
    fn from(state: &PersistedState) -> Self {
        Self {
            hard_state: (&state.hard_state).into(),
            conf_state: state.conf_state.voters.clone(),
            entries: state.entries.iter().map(|e| e.into()).collect(),
            snapshot_data: state.snapshot.as_ref().map(|s| s.data.to_vec()),
            applied_index: state.applied_index,
        }
    }
}

impl From<&SerializablePersistedState> for PersistedState {
    fn from(serializable: &SerializablePersistedState) -> Self {
        let mut snapshot = None;
        if let Some(data) = &serializable.snapshot_data {
            let mut snap = Snapshot::default();
            snap.data = bytes::Bytes::from(data.clone());
            snapshot = Some(snap);
        }
        
        Self {
            hard_state: (&serializable.hard_state).into(),
            conf_state: {
                let mut cs = ConfState::default();
                cs.voters = serializable.conf_state.clone();
                cs
            },
            entries: serializable.entries.iter().map(|e| e.into()).collect(),
            snapshot,
            applied_index: serializable.applied_index,
        }
    }
}

impl MemoryStorage {
    /// Create a new memory storage
    pub fn new() -> Self {
        Self {
            hard_state: Arc::new(RwLock::new(HardState::default())),
            conf_state: Arc::new(RwLock::new(ConfState::default())),
            entries: Arc::new(RwLock::new(Vec::new())),
            snapshot: Arc::new(RwLock::new(Snapshot::default())),
            applied_index: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(StorageStats::default())),
        }
    }

    /// Append entries to the log
    pub async fn append_entries(&self, entries: Vec<Entry>) -> Result<()> {
        let mut log_entries = self.entries.write().await;
        let mut stats = self.stats.write().await;
        
        for entry in entries {
            log_entries.push(entry.clone());
            stats.entry_count += 1;
            stats.entry_size_bytes += entry.data.len() as u64;
            stats.last_index = entry.index;
        }
        
        if stats.first_index == 0 && !log_entries.is_empty() {
            stats.first_index = log_entries[0].index;
        }
        
        Ok(())
    }

    /// Set hard state
    pub async fn set_hard_state(&self, hard_state: HardState) -> Result<()> {
        let mut hs = self.hard_state.write().await;
        let mut stats = self.stats.write().await;
        
        *hs = hard_state.clone();
        stats.committed_index = hard_state.commit;
        
        Ok(())
    }

    /// Set configuration state
    pub async fn set_conf_state(&self, conf_state: ConfState) -> Result<()> {
        let mut cs = self.conf_state.write().await;
        *cs = conf_state;
        Ok(())
    }

    /// Set applied index
    pub async fn set_applied_index(&self, index: u64) -> Result<()> {
        let mut applied = self.applied_index.write().await;
        let mut stats = self.stats.write().await;
        
        *applied = index;
        stats.applied_index = index;
        
        Ok(())
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl RaftStorage for MemoryStorage {
    fn initialize(&mut self) -> Result<()> {
        // Memory storage is always initialized
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        // Nothing to close for memory storage
        Ok(())
    }

    fn stats(&self) -> StorageStats {
        // This is a blocking call, but in a real implementation you might want to make it async
        futures::executor::block_on(async {
            self.stats.read().await.clone()
        })
    }

    fn compact(&mut self, compact_index: u64) -> Result<()> {
        futures::executor::block_on(async {
            let mut entries = self.entries.write().await;
            let mut stats = self.stats.write().await;
            
            // Remove entries up to compact_index
            let original_len = entries.len();
            entries.retain(|entry| entry.index > compact_index);
            let removed_count = original_len - entries.len();
            
            stats.entry_count = stats.entry_count.saturating_sub(removed_count as u64);
            if !entries.is_empty() {
                stats.first_index = entries[0].index;
            }
            
            Ok(())
        })
    }

    fn create_snapshot(&mut self, index: u64, data: Vec<u8>) -> Result<()> {
        futures::executor::block_on(async {
            let mut snapshot = self.snapshot.write().await;
            let mut stats = self.stats.write().await;
            
            let mut new_snapshot = Snapshot::default();
            new_snapshot.mut_metadata().index = index;
            new_snapshot.data = bytes::Bytes::from(data.clone());
            
            *snapshot = new_snapshot;
            stats.snapshot_count += 1;
            stats.snapshot_size_bytes += data.len() as u64;
            
            Ok(())
        })
    }

    fn apply_snapshot(&mut self, snapshot: Snapshot) -> Result<()> {
        futures::executor::block_on(async {
            let mut current_snapshot = self.snapshot.write().await;
            let mut stats = self.stats.write().await;
            
            *current_snapshot = snapshot.clone();
            stats.applied_index = snapshot.get_metadata().index;
            
            Ok(())
        })
    }
}

impl Storage for MemoryStorage {
    fn initial_state(&self) -> raft::Result<RaftState> {
        futures::executor::block_on(async {
            let hard_state = self.hard_state.read().await.clone();
            let conf_state = self.conf_state.read().await.clone();
            
            Ok(RaftState {
                hard_state,
                conf_state,
            })
        })
    }

    fn entries(
        &self,
        low: u64,
        high: u64,
        max_size: impl Into<Option<u64>>,
        _context: GetEntriesContext,
    ) -> raft::Result<Vec<Entry>> {
        futures::executor::block_on(async {
            let entries = self.entries.read().await;
            let max_size = max_size.into().unwrap_or(u64::MAX);
            
            let mut result = Vec::new();
            let mut total_size = 0u64;
            
            for entry in entries.iter() {
                if entry.index >= low && entry.index < high {
                    let entry_size = entry.data.len() as u64;
                    if total_size + entry_size > max_size && !result.is_empty() {
                        break;
                    }
                    result.push(entry.clone());
                    total_size += entry_size;
                }
            }
            
            if result.is_empty() && low < high {
                return Err(TikvRaftError::Store(StorageError::Unavailable));
            }
            
            Ok(result)
        })
    }

    fn term(&self, idx: u64) -> raft::Result<u64> {
        futures::executor::block_on(async {
            let entries = self.entries.read().await;
            let snapshot = self.snapshot.read().await;
            
            // Check snapshot first
            if idx == snapshot.get_metadata().index {
                return Ok(snapshot.get_metadata().term);
            }
            
            // Check entries
            for entry in entries.iter() {
                if entry.index == idx {
                    return Ok(entry.term);
                }
            }
            
            Err(TikvRaftError::Store(StorageError::Unavailable))
        })
    }

    fn first_index(&self) -> raft::Result<u64> {
        futures::executor::block_on(async {
            let entries = self.entries.read().await;
            let snapshot = self.snapshot.read().await;
            
            if !entries.is_empty() {
                Ok(entries[0].index)
            } else if snapshot.get_metadata().index > 0 {
                Ok(snapshot.get_metadata().index + 1)
            } else {
                Ok(1)
            }
        })
    }

    fn last_index(&self) -> raft::Result<u64> {
        futures::executor::block_on(async {
            let entries = self.entries.read().await;
            let snapshot = self.snapshot.read().await;
            
            if !entries.is_empty() {
                Ok(entries.last().unwrap().index)
            } else {
                Ok(snapshot.get_metadata().index)
            }
        })
    }

    fn snapshot(&self, _request_index: u64, _to: u64) -> raft::Result<Snapshot> {
        futures::executor::block_on(async {
            let snapshot = self.snapshot.read().await;
            Ok(snapshot.clone())
        })
    }
}

impl DiskStorage {
    /// Create a new disk storage
    pub fn new(data_dir: PathBuf, sync_writes: bool, compress_entries: bool, max_memory_entries: usize) -> Self {
        Self {
            data_dir,
            memory_storage: MemoryStorage::new(),
            sync_writes,
            compress_entries,
            max_memory_entries,
        }
    }

    /// Load state from disk
    async fn load_from_disk(&mut self) -> Result<()> {
        let state_file = self.data_dir.join("raft_state.json");
        
        if !state_file.exists() {
            return Ok(());
        }
        
        let data = tokio::fs::read(&state_file).await?;
        let serializable_state: SerializablePersistedState = serde_json::from_slice(&data)?;
        let state: PersistedState = (&serializable_state).into();
        
        // Load into memory storage
        self.memory_storage.set_hard_state(state.hard_state).await?;
        self.memory_storage.set_conf_state(state.conf_state).await?;
        self.memory_storage.append_entries(state.entries).await?;
        self.memory_storage.set_applied_index(state.applied_index).await?;
        
        if let Some(snapshot) = state.snapshot {
            self.memory_storage.apply_snapshot(snapshot)?;
        }
        
        Ok(())
    }

    /// Save state to disk
    async fn save_to_disk(&self) -> Result<()> {
        let hard_state = self.memory_storage.hard_state.read().await.clone();
        let conf_state = self.memory_storage.conf_state.read().await.clone();
        let entries = self.memory_storage.entries.read().await.clone();
        let snapshot = self.memory_storage.snapshot.read().await.clone();
        let applied_index = *self.memory_storage.applied_index.read().await;
        
        let state = PersistedState {
            hard_state,
            conf_state,
            entries,
            snapshot: if snapshot.get_metadata().index > 0 { Some(snapshot) } else { None },
            applied_index,
        };
        
        let serializable_state: SerializablePersistedState = (&state).into();
        let data = serde_json::to_vec_pretty(&serializable_state)?;
        
        // Ensure directory exists
        tokio::fs::create_dir_all(&self.data_dir).await?;
        
        let state_file = self.data_dir.join("raft_state.json");
        let temp_file = self.data_dir.join("raft_state.json.tmp");
        
        // Write to temporary file first
        tokio::fs::write(&temp_file, &data).await?;
        
        if self.sync_writes {
            // Sync to disk
            let file = tokio::fs::OpenOptions::new()
                .write(true)
                .open(&temp_file)
                .await?;
            file.sync_all().await?;
        }
        
        // Atomic rename
        tokio::fs::rename(&temp_file, &state_file).await?;
        
        Ok(())
    }
}

impl RaftStorage for DiskStorage {
    fn initialize(&mut self) -> Result<()> {
        // Create data directory if it doesn't exist
        std::fs::create_dir_all(&self.data_dir)
            .map_err(|e| RaftError::Storage(format!("Failed to create data directory: {}", e)))?;
        
        // Load existing state
        futures::executor::block_on(self.load_from_disk())?;
        
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        // Save current state to disk
        futures::executor::block_on(self.save_to_disk())?;
        Ok(())
    }

    fn stats(&self) -> StorageStats {
        self.memory_storage.stats()
    }

    fn compact(&mut self, compact_index: u64) -> Result<()> {
        self.memory_storage.compact(compact_index)?;
        
        // Save to disk after compaction
        futures::executor::block_on(self.save_to_disk())?;
        
        Ok(())
    }

    fn create_snapshot(&mut self, index: u64, data: Vec<u8>) -> Result<()> {
        self.memory_storage.create_snapshot(index, data)?;
        
        // Save to disk after snapshot creation
        futures::executor::block_on(self.save_to_disk())?;
        
        Ok(())
    }

    fn apply_snapshot(&mut self, snapshot: Snapshot) -> Result<()> {
        self.memory_storage.apply_snapshot(snapshot)?;
        
        // Save to disk after snapshot application
        futures::executor::block_on(self.save_to_disk())?;
        
        Ok(())
    }
}

impl Storage for DiskStorage {
    fn initial_state(&self) -> raft::Result<RaftState> {
        self.memory_storage.initial_state()
    }

    fn entries(
        &self,
        low: u64,
        high: u64,
        max_size: impl Into<Option<u64>>,
        context: GetEntriesContext,
    ) -> raft::Result<Vec<Entry>> {
        self.memory_storage.entries(low, high, max_size, context)
    }

    fn term(&self, idx: u64) -> raft::Result<u64> {
        self.memory_storage.term(idx)
    }

    fn first_index(&self) -> raft::Result<u64> {
        self.memory_storage.first_index()
    }

    fn last_index(&self) -> raft::Result<u64> {
        self.memory_storage.last_index()
    }

    fn snapshot(&self, request_index: u64, to: u64) -> raft::Result<Snapshot> {
        self.memory_storage.snapshot(request_index, to)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use raft::eraftpb::Entry;

    #[tokio::test]
    async fn test_memory_storage() {
        let mut storage = MemoryStorage::new();
        storage.initialize().unwrap();
        
        // Test initial state
        let state = storage.initial_state().unwrap();
        assert_eq!(state.hard_state.term, 0);
        assert_eq!(state.hard_state.commit, 0);
        
        // Test append entries
        let mut entry1 = Entry::default();
        entry1.index = 1;
        entry1.term = 1;
        entry1.data = bytes::Bytes::from(b"entry1".to_vec());
        
        let mut entry2 = Entry::default();
        entry2.index = 2;
        entry2.term = 1;
        entry2.data = bytes::Bytes::from(b"entry2".to_vec());
        
        let entries = vec![entry1, entry2];
        storage.append_entries(entries.clone()).await.unwrap();
        
        // Test first/last index
        assert_eq!(storage.first_index().unwrap(), 1);
        assert_eq!(storage.last_index().unwrap(), 2);
        
        // Test entries retrieval
        let retrieved = storage.entries(1, 3, None, GetEntriesContext::empty(false)).unwrap();
        assert_eq!(retrieved.len(), 2);
        assert_eq!(retrieved[0].index, 1);
        assert_eq!(retrieved[1].index, 2);
        
        // Test term lookup
        assert_eq!(storage.term(1).unwrap(), 1);
        assert_eq!(storage.term(2).unwrap(), 1);
        
        // Test stats
        let stats = storage.stats();
        assert_eq!(stats.entry_count, 2);
        assert_eq!(stats.first_index, 1);
        assert_eq!(stats.last_index, 2);
    }

    #[tokio::test]
    async fn test_disk_storage() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = DiskStorage::new(temp_dir.path().to_path_buf(), false, false, 1000);
        storage.initialize().unwrap();
        
        // Test append entries
        let mut entry1 = Entry::default();
        entry1.index = 1;
        entry1.term = 1;
        entry1.data = bytes::Bytes::from(b"entry1".to_vec());
        
        let entries = vec![entry1];
        storage.memory_storage.append_entries(entries).await.unwrap();
        
        // Save to disk
        storage.save_to_disk().await.unwrap();
        
        // Create new storage and load from disk
        let mut storage2 = DiskStorage::new(temp_dir.path().to_path_buf(), false, false, 1000);
        storage2.initialize().unwrap();
        
        // Verify data was loaded
        assert_eq!(storage2.first_index().unwrap(), 1);
        assert_eq!(storage2.last_index().unwrap(), 1);
        
        let retrieved = storage2.entries(1, 2, None, GetEntriesContext::empty(false)).unwrap();
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].data.as_ref(), b"entry1");
    }

    #[tokio::test]
    async fn test_storage_compaction() {
        let mut storage = MemoryStorage::new();
        storage.initialize().unwrap();
        
        // Add entries
        let mut entry1 = Entry::default();
        entry1.index = 1;
        entry1.term = 1;
        entry1.data = bytes::Bytes::from(b"entry1".to_vec());
        
        let mut entry2 = Entry::default();
        entry2.index = 2;
        entry2.term = 1;
        entry2.data = bytes::Bytes::from(b"entry2".to_vec());
        
        let mut entry3 = Entry::default();
        entry3.index = 3;
        entry3.term = 1;
        entry3.data = bytes::Bytes::from(b"entry3".to_vec());
        
        let entries = vec![entry1, entry2, entry3];
        storage.append_entries(entries).await.unwrap();
        
        // Compact up to index 2
        storage.compact(2).unwrap();
        
        // Verify only entry 3 remains
        assert_eq!(storage.first_index().unwrap(), 3);
        assert_eq!(storage.last_index().unwrap(), 3);
        
        let stats = storage.stats();
        assert_eq!(stats.entry_count, 1);
    }

    #[tokio::test]
    async fn test_snapshot_creation() {
        let mut storage = MemoryStorage::new();
        storage.initialize().unwrap();
        
        let snapshot_data = b"snapshot_data".to_vec();
        storage.create_snapshot(5, snapshot_data.clone()).unwrap();
        
        let snapshot = storage.snapshot(0, 0).unwrap();
        assert_eq!(snapshot.get_metadata().index, 5);
        assert_eq!(snapshot.data.as_ref(), snapshot_data.as_slice());
        
        let stats = storage.stats();
        assert_eq!(stats.snapshot_count, 1);
        assert_eq!(stats.snapshot_size_bytes, snapshot_data.len() as u64);
    }
}
