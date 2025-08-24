//! # mesh-core
//!
//! Core types, traits, and utilities for infermesh - a GPU-aware inference mesh.
//!
//! This crate provides the foundational data structures and interfaces that are
//! shared across all other infermesh components. It includes:
//!
//! - Core data structures for labels, model state, and GPU state
//! - Traits for runtime control and GPU telemetry
//! - Configuration schema and parsing utilities
//! - Error handling types and utilities
//! - Common constants and feature flags

pub mod config;
pub mod error;
pub mod labels;
pub mod state;
pub mod traits;
pub mod types;

// Re-export commonly used types at the crate root
pub use config::{Config, NodeConfig, RuntimeConfig};
pub use error::{Error, Result};
pub use labels::Labels;
pub use state::{GpuState, GpuStateDelta, ModelState, ModelStateDelta};
pub use traits::{GpuTelemetry, RuntimeControl, RuntimeTelemetry, ModelConfig, ModelStatus};
pub use types::{NodeId, NodeRole, ServiceEndpoint, SloClass};
