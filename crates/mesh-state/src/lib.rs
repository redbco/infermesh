//! # mesh-state
//!
//! State management and routing intelligence for infermesh.
//!
//! This crate provides:
//! - In-memory state store for `ModelState` and `GpuState`
//! - Delta application from adapters and gossip
//! - Scoring algorithm implementation for routing decisions
//! - Fast query API for routers (O(1) hot-path)
//! - State synchronization and consistency management
//!
//! ## Example
//!
//! ```rust
//! use mesh_state::{StateStore, ScoringEngine};
//! use mesh_core::{Labels, NodeId};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let store = StateStore::new();
//! let scoring_engine = ScoringEngine::new();
//!
//! // Query for best targets
//! let labels = Labels::new("gpt-7b", "v1", "runtime", "node1")
//!     .with_custom("model", "gpt-7b");
//! let targets = scoring_engine.score_targets(&store, &labels, 128).await?;
//!
//! println!("Found {} targets", targets.len());
//! # Ok(())
//! # }
//! ```

use thiserror::Error;

pub mod config;
pub mod delta;
pub mod query;
pub mod scoring;
pub mod store;
pub mod sync;

// Re-export commonly used types
pub use config::StateConfig;
pub use delta::{DeltaApplier, StateDelta};
pub use query::{QueryEngine, QueryFilter, QueryResult};
pub use scoring::{ScoredTarget, ScoringEngine, ScoringMetrics};
pub use store::{StateStore, StateSnapshot};
pub use sync::{StateSynchronizer, SyncEvent};

/// Result type for state operations
pub type Result<T> = std::result::Result<T, StateError>;

/// Errors that can occur during state operations
#[derive(Error, Debug)]
pub enum StateError {
    #[error("Store error: {0}")]
    Store(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Scoring error: {0}")]
    Scoring(String),

    #[error("Synchronization error: {0}")]
    Sync(String),

    #[error("Delta error: {0}")]
    Delta(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Core error: {0}")]
    Core(#[from] mesh_core::Error),

    #[error("Gossip error: {0}")]
    Gossip(#[from] mesh_gossip::GossipError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_core::Labels;

    #[test]
    fn test_state_store_creation() {
        let store = StateStore::new();
        assert_eq!(store.model_count(), 0);
        assert_eq!(store.gpu_count(), 0);
    }

    #[test]
    fn test_scoring_engine_creation() {
        let engine = ScoringEngine::new();
        assert!(engine.is_enabled());
    }

    #[tokio::test]
    async fn test_query_engine_creation() {
        let store = StateStore::new();
        let query_engine = QueryEngine::new(store);
        
        let labels = Labels::new("test", "v1", "runtime", "node1")
            .with_custom("model", "test");
        let results = query_engine.find_models(&labels).await.unwrap();
        assert!(results.is_empty());
    }
}
