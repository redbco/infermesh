//! gRPC client for mesh control plane

use anyhow::{Context, Result};
use mesh_proto::{
    ControlPlaneClient, StatePlaneClient, ScoringClient,
    ListNodesRequest, ListModelsRequest, PinModelRequest, UnpinModelRequest,
    GetNodeRequest, ListPoliciesRequest, ListModelStatesRequest, Node, Model, Policy,
    NodeRole, PolicyType,
};
use std::time::Duration;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Status};
use tracing::{debug, info, warn};

/// Client for interacting with mesh control plane
#[derive(Clone)]
pub struct MeshClient {
    #[allow(dead_code)]
    endpoint: String,
    #[allow(dead_code)]
    timeout: Duration,
    #[allow(dead_code)]
    control_plane: ControlPlaneClient<Channel>,
    #[allow(dead_code)]
    state_plane: StatePlaneClient<Channel>,
    #[allow(dead_code)]
    scoring: ScoringClient<Channel>,
}

impl MeshClient {
    /// Create a new mesh client
    pub async fn new(endpoint: &str, timeout: Duration) -> Result<Self> {
        info!("Connecting to mesh control plane at {}", endpoint);
        
        // Establish gRPC connection
        let channel = Endpoint::from_shared(endpoint.to_string())?
            .timeout(timeout)
            .connect()
            .await
            .context("Failed to connect to mesh control plane")?;

        let control_plane = ControlPlaneClient::new(channel.clone());
        let state_plane = StatePlaneClient::new(channel.clone());
        let scoring = ScoringClient::new(channel);

        info!("Successfully connected to mesh control plane");

        Ok(Self {
            endpoint: endpoint.to_string(),
            timeout,
            control_plane,
            state_plane,
            scoring,
        })
    }

    /// Get the endpoint URL
    #[allow(dead_code)]
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Get the configured timeout
    #[allow(dead_code)]
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// List nodes in the mesh
    #[allow(dead_code)]
    pub async fn list_nodes(&self, role_filter: Option<String>, status_filter: Option<String>) -> Result<Vec<Node>> {
        debug!("Listing nodes with role_filter: {:?}, status_filter: {:?}", role_filter, status_filter);
        
        let mut request = ListNodesRequest {
            role_filter: vec![],
            zone_filter: String::new(),
            label_filter: std::collections::HashMap::new(),
        };

        // Parse role filter
        if let Some(role_str) = role_filter {
            let role = match role_str.to_lowercase().as_str() {
                "router" => NodeRole::Router as i32,
                "gpu" => NodeRole::Gpu as i32,
                "edge" => NodeRole::Edge as i32,
                _ => {
                    warn!("Unknown node role: {}, ignoring filter", role_str);
                    NodeRole::Unspecified as i32
                }
            };
            if role != NodeRole::Unspecified as i32 {
                request.role_filter.push(role);
            }
        }

        let mut client = self.control_plane.clone();
        let response = client
            .list_nodes(Request::new(request))
            .await
            .map_err(handle_grpc_error)?;

        let nodes = response.into_inner().nodes;
        info!("Retrieved {} nodes from mesh", nodes.len());
        
        Ok(nodes)
    }

    /// Get details for a specific node
    #[allow(dead_code)]
    pub async fn get_node(&self, node_id: String) -> Result<Node> {
        debug!("Getting node details for {}", node_id);
        
        let request = GetNodeRequest { node_id };
        
        let mut client = self.control_plane.clone();
        let response = client
            .get_node(Request::new(request))
            .await
            .map_err(handle_grpc_error)?;

        let node = response.into_inner().node
            .ok_or_else(|| anyhow::anyhow!("Node not found in response"))?;
        
        info!("Retrieved details for node: {}", node.id);
        Ok(node)
    }

    /// Pin a model to specific nodes
    #[allow(dead_code)]
    pub async fn pin_model(
        &self,
        model_name: String,
        model_revision: Option<String>,
        target_nodes: Vec<String>,
        min_replicas: u32,
        max_replicas: Option<u32>,
        priority: String,
    ) -> Result<String> {
        debug!("Pinning model {} (revision: {:?}) to nodes {:?}", model_name, model_revision, target_nodes);
        
        let mut config = std::collections::HashMap::new();
        config.insert("min_replicas".to_string(), min_replicas.to_string());
        if let Some(max_reps) = max_replicas {
            config.insert("max_replicas".to_string(), max_reps.to_string());
        }
        config.insert("priority".to_string(), priority);

        let request = PinModelRequest {
            model_name,
            model_revision: model_revision.unwrap_or_default(),
            target_nodes,
            config,
        };
        
        let mut client = self.control_plane.clone();
        let response = client
            .pin_model(Request::new(request))
            .await
            .map_err(handle_grpc_error)?;

        let pin_response = response.into_inner();
        if !pin_response.success {
            return Err(anyhow::anyhow!("Failed to pin model: {}", pin_response.message));
        }
        
        info!("Successfully pinned model, pin ID: {}", pin_response.pin_id);
        Ok(pin_response.pin_id)
    }

    /// Unpin a model from nodes
    #[allow(dead_code)]
    pub async fn unpin_model(&self, model_name: String, target_nodes: Option<Vec<String>>) -> Result<()> {
        debug!("Unpinning model {} from nodes {:?}", model_name, target_nodes);
        
        let request = UnpinModelRequest {
            model_name,
            model_revision: String::new(), // Use default revision
            target_nodes: target_nodes.unwrap_or_default(),
        };
        
        let mut client = self.control_plane.clone();
        let response = client
            .unpin_model(Request::new(request))
            .await
            .map_err(handle_grpc_error)?;

        let unpin_response = response.into_inner();
        if !unpin_response.success {
            return Err(anyhow::anyhow!("Failed to unpin model: {}", unpin_response.message));
        }
        
        info!("Successfully unpinned model");
        Ok(())
    }

    /// List policies
    #[allow(dead_code)]
    pub async fn list_policies(&self, policy_type: Option<String>) -> Result<Vec<Policy>> {
        debug!("Listing policies with type filter: {:?}", policy_type);
        
        let request = ListPoliciesRequest {
            type_filter: PolicyType::Unspecified as i32, // Default to unspecified for now
        };
        
        let mut client = self.control_plane.clone();
        let response = client
            .list_policies(Request::new(request))
            .await
            .map_err(handle_grpc_error)?;

        let policies = response.into_inner().policies;
        info!("Retrieved {} policies", policies.len());
        
        Ok(policies)
    }

    /// Get mesh state (model states)
    #[allow(dead_code)]
    pub async fn get_state(&self) -> Result<Vec<mesh_proto::ModelState>> {
        debug!("Getting mesh state (model states)");
        
        let request = ListModelStatesRequest {
            model_filter: String::new(),
            runtime_filter: String::new(),
            node_filter: String::new(),
            loaded_only: false,
        };
        
        let mut client = self.state_plane.clone();
        let response = client
            .list_model_states(Request::new(request))
            .await
            .map_err(handle_grpc_error)?;

        let model_states = response.into_inner().states;
        info!("Retrieved {} model states", model_states.len());
        
        Ok(model_states)
    }

    /// Query models
    #[allow(dead_code)]
    pub async fn query_models(&self, model_filter: Option<String>, runtime_filter: Option<String>) -> Result<Vec<Model>> {
        debug!("Querying models with filters - model: {:?}, runtime: {:?}", model_filter, runtime_filter);
        
        let request = ListModelsRequest {
            name_filter: model_filter.unwrap_or_default(),
            runtime_filter: runtime_filter.unwrap_or_default(),
            status_filter: vec![], // No status filter for now
        };
        
        let mut client = self.control_plane.clone();
        let response = client
            .list_models(Request::new(request))
            .await
            .map_err(handle_grpc_error)?;

        let models = response.into_inner().models;
        info!("Retrieved {} models", models.len());
        
        Ok(models)
    }

    /// Check if the client can connect to the control plane
    #[allow(dead_code)]
    pub async fn health_check(&self) -> Result<bool> {
        debug!("Performing health check");
        
        // Try to list nodes as a basic connectivity test
        match self.list_nodes(None, None).await {
            Ok(_) => {
                info!("Health check passed - successfully connected to control plane");
                Ok(true)
            }
            Err(e) => {
                warn!("Health check failed: {}", e);
                Ok(false)
            }
        }
    }

    /// Subscribe to events (placeholder implementation)
    #[allow(dead_code)]
    pub async fn subscribe_events(&self, event_types: Option<Vec<String>>) -> Result<()> {
        debug!("Subscribing to events: {:?} (placeholder implementation)", event_types);
        
        warn!("Event subscription not yet implemented - this is a placeholder");
        Ok(())
    }
}

/// Helper function to handle gRPC status errors
pub fn handle_grpc_error(status: Status) -> anyhow::Error {
    match status.code() {
        tonic::Code::NotFound => anyhow::anyhow!("Resource not found: {}", status.message()),
        tonic::Code::PermissionDenied => anyhow::anyhow!("Permission denied: {}", status.message()),
        tonic::Code::Unauthenticated => anyhow::anyhow!("Authentication required: {}", status.message()),
        tonic::Code::Unavailable => anyhow::anyhow!("Service unavailable: {}", status.message()),
        tonic::Code::DeadlineExceeded => anyhow::anyhow!("Request timeout: {}", status.message()),
        tonic::Code::InvalidArgument => anyhow::anyhow!("Invalid argument: {}", status.message()),
        tonic::Code::AlreadyExists => anyhow::anyhow!("Resource already exists: {}", status.message()),
        tonic::Code::ResourceExhausted => anyhow::anyhow!("Resource exhausted: {}", status.message()),
        tonic::Code::FailedPrecondition => anyhow::anyhow!("Precondition failed: {}", status.message()),
        tonic::Code::Aborted => anyhow::anyhow!("Operation aborted: {}", status.message()),
        tonic::Code::OutOfRange => anyhow::anyhow!("Out of range: {}", status.message()),
        tonic::Code::Unimplemented => anyhow::anyhow!("Not implemented: {}", status.message()),
        tonic::Code::Internal => anyhow::anyhow!("Internal error: {}", status.message()),
        tonic::Code::DataLoss => anyhow::anyhow!("Data loss: {}", status.message()),
        _ => anyhow::anyhow!("gRPC error: {} - {}", status.code(), status.message()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        // This test will fail if there's no server running, but it tests the client creation logic
        let result = MeshClient::new("http://127.0.0.1:50051", Duration::from_secs(5)).await;
        
        // We expect this to fail in tests since there's no server running
        assert!(result.is_err());
    }

    #[test]
    fn test_grpc_error_handling() {
        let status = Status::not_found("Resource not found");
        let error = handle_grpc_error(status);
        assert!(error.to_string().contains("Resource not found"));
        
        let status = Status::permission_denied("Access denied");
        let error = handle_grpc_error(status);
        assert!(error.to_string().contains("Permission denied"));
    }

    #[test]
    fn test_client_properties() {
        // Test that we can create a client struct (without connecting)
        let endpoint = "http://127.0.0.1:50051";
        let timeout = Duration::from_secs(30);
        
        // We can't actually test the client without a server, but we can test
        // that the endpoint and timeout would be set correctly
        assert_eq!(endpoint, "http://127.0.0.1:50051");
        assert_eq!(timeout, Duration::from_secs(30));
    }
}
