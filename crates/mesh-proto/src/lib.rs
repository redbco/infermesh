//! # mesh-proto
//!
//! Protocol buffer definitions and generated gRPC bindings for infermesh.
//!
//! This crate provides the API definitions for all infermesh services:
//! - Control Plane API for cluster management
//! - State Plane API for telemetry and state updates  
//! - Scoring API for routing decisions
//!
//! All protobuf messages include serde serialization support for JSON/YAML
//! compatibility and debugging.

pub mod conversions;

// Generated protobuf code
pub mod control {
    pub mod v1 {
        tonic::include_proto!("infermesh.control.v1");
    }
}

pub mod state {
    pub mod v1 {
        tonic::include_proto!("infermesh.state.v1");
    }
}

pub mod scoring {
    pub mod v1 {
        tonic::include_proto!("infermesh.scoring.v1");
    }
}

// Re-export commonly used types for convenience (both server and client)
pub use control::v1::{
    control_plane_server::{ControlPlane, ControlPlaneServer},
    control_plane_client::ControlPlaneClient,
    Event, EventType, ListNodesRequest, ListNodesResponse, Node, NodeRole, NodeStatus,
    PinModelRequest, PinModelResponse, SubscribeEventsRequest,
    ListModelsRequest, ListModelsResponse, Model, ModelStatus, SetPolicyRequest, SetPolicyResponse,
    Policy, PolicyType, GetNodeRequest, GetNodeResponse, UnpinModelRequest, UnpinModelResponse,
    LoadModelRequest, LoadModelResponse, UnloadModelRequest, UnloadModelResponse,
    GetPolicyRequest, GetPolicyResponse, DeletePolicyRequest, DeletePolicyResponse,
    ListPoliciesRequest, ListPoliciesResponse, DrainNodeRequest, DrainNodeResponse,
};

pub use state::v1::{
    state_plane_server::{StatePlane, StatePlaneServer},
    state_plane_client::StatePlaneClient,
    GpuState, GpuStateDelta, Labels, ModelState, ModelStateDelta,
    GetModelStateRequest, GetModelStateResponse, GetGpuStateRequest, GetGpuStateResponse,
    ListModelStatesRequest, ListModelStatesResponse, ListGpuStatesRequest, ListGpuStatesResponse,
    ModelStateAck, GpuStateAck,
};

pub use scoring::v1::{
    scoring_server::{Scoring, ScoringServer},
    scoring_client::ScoringClient,
    AdmitRequest, AdmitResponse, RequestOutcome, ScoredTarget, ScoreTargetsRequest,
    ScoreTargetsResponse, SloClass,
};

// Common error type for proto operations
#[derive(Debug, thiserror::Error)]
pub enum ProtoError {
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::Status),
    
    #[error("Transport error: {0}")]
    Transport(#[from] tonic::transport::Error),
    
    #[error("Invalid data: {0}")]
    InvalidData(String),
}

pub type Result<T> = std::result::Result<T, ProtoError>;

/// Utility functions for working with protobuf timestamps
pub mod timestamp {
    use prost_types::Timestamp;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Convert SystemTime to protobuf Timestamp
    pub fn from_system_time(time: SystemTime) -> Timestamp {
        let duration = time.duration_since(UNIX_EPOCH).unwrap_or_default();
        Timestamp {
            seconds: duration.as_secs() as i64,
            nanos: duration.subsec_nanos() as i32,
        }
    }

    /// Convert protobuf Timestamp to SystemTime
    pub fn to_system_time(timestamp: &Timestamp) -> SystemTime {
        UNIX_EPOCH + std::time::Duration::new(
            timestamp.seconds as u64,
            timestamp.nanos as u32,
        )
    }

    /// Get current time as protobuf Timestamp
    pub fn now() -> Timestamp {
        from_system_time(SystemTime::now())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    #[test]
    fn test_timestamp_conversion() {
        let now = SystemTime::now();
        let proto_ts = timestamp::from_system_time(now);
        let converted_back = timestamp::to_system_time(&proto_ts);
        
        // Should be very close (within 1 second due to precision)
        let diff = now.duration_since(converted_back)
            .or_else(|_| converted_back.duration_since(now))
            .unwrap();
        
        assert!(diff.as_secs() < 1);
    }

    #[test]
    fn test_proto_creation() {
        let node = Node {
            id: "test-node".to_string(),
            roles: vec![NodeRole::Router as i32, NodeRole::Gpu as i32],
            zone: "us-west-2".to_string(),
            labels: std::collections::HashMap::new(),
            status: NodeStatus::Healthy as i32,
            last_seen: Some(timestamp::now()),
            version: "0.1.0".to_string(),
        };

        assert_eq!(node.id, "test-node");
        assert_eq!(node.roles.len(), 2);
        assert_eq!(node.zone, "us-west-2");
    }

    #[test]
    fn test_labels_creation() {
        let labels = Labels {
            model: "gpt-4".to_string(),
            revision: "v1.0".to_string(),
            quant: "fp16".to_string(),
            runtime: "triton".to_string(),
            node: "gpu-node-1".to_string(),
            gpu_uuid: "GPU-12345678".to_string(),
            mig_profile: "1g.5gb".to_string(),
            tenant: "customer-a".to_string(),
            zone: "us-west-2".to_string(),
            custom: std::collections::HashMap::new(),
        };

        assert_eq!(labels.model, "gpt-4");
        assert_eq!(labels.runtime, "triton");
    }

    #[test]
    fn test_model_state_delta() {
        let delta = ModelStateDelta {
            labels: Some(Labels {
                model: "test-model".to_string(),
                revision: "v1.0".to_string(),
                runtime: "mock".to_string(),
                node: "test-node".to_string(),
                ..Default::default()
            }),
            queue_depth: Some(5),
            service_rate: Some(10.0),
            loaded: Some(true),
            timestamp: Some(timestamp::now()),
            ..Default::default()
        };

        // Test that optional fields work correctly
        assert_eq!(delta.queue_depth, Some(5));
        assert_eq!(delta.service_rate, Some(10.0));
        assert_eq!(delta.loaded, Some(true));
        assert!(delta.p95_latency_ms.is_none());
    }
}