//! gRPC service implementations for the mesh agent

pub mod adapter_manager;
pub mod control_plane;
pub mod router_service;
pub mod scoring;
pub mod state_plane;
pub mod state_sync;

// Re-export service implementations
pub use adapter_manager::AdapterManager;
pub use control_plane::ControlPlaneService;
pub use router_service::RouterService;
pub use scoring::ScoringService;
pub use state_plane::StatePlaneService;
pub use state_sync::StateSyncService;
