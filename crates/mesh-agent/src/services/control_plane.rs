//! Control Plane service implementation

use crate::config::AgentConfig;
use crate::services::StatePlaneService;
use mesh_metrics::MetricsRegistry;
use mesh_raft::{RaftNode, Policy, PolicyType};
use mesh_raft::policy::{PolicyMetadata, ModelPinPolicy, PolicyPriority};
use mesh_proto::control::v1::{
    control_plane_server::ControlPlane, DrainNodeRequest, DrainNodeResponse, GetNodeRequest,
    GetNodeResponse, ListNodesRequest, ListNodesResponse, LoadModelRequest, LoadModelResponse,
    PinModelRequest, PinModelResponse, UnloadModelRequest, UnloadModelResponse,
    UnpinModelRequest, UnpinModelResponse, Node, NodeRole, NodeStatus,
    ListModelsRequest, ListModelsResponse, SetPolicyRequest, SetPolicyResponse,
    //GetPolicyRequest, GetPolicyResponse, DeletePolicyRequest, DeletePolicyResponse,
    //ListPoliciesRequest, ListPoliciesResponse, SubscribeEventsRequest, Event,
};
use std::collections::HashMap;
use tonic::{Request, Response, Status};
use tracing::{debug, info, warn};

/// Control Plane service implementation
#[derive(Clone)]
pub struct ControlPlaneService {
    #[allow(unused)]
    config: AgentConfig,
    #[allow(unused)]
    metrics_registry: MetricsRegistry,
    // In a real implementation, these would be backed by persistent storage
    nodes: std::sync::Arc<tokio::sync::RwLock<HashMap<String, Node>>>,
    // State plane service for accessing model and GPU state
    state_plane: StatePlaneService,
    // Raft node for distributed consensus (optional for now)
    #[allow(unused)]
    raft_node: Option<std::sync::Arc<RaftNode>>,
    // In-memory policy storage (will be replaced with raft persistence)
    policies: std::sync::Arc<tokio::sync::RwLock<HashMap<String, Policy>>>,
}

impl ControlPlaneService {
    /// Create a new Control Plane service
    pub fn new(config: AgentConfig, metrics_registry: MetricsRegistry) -> Self {
        let mut nodes = HashMap::new();
        
        // Add this node to the registry
        let node = Node {
            id: config.core.node.id.to_string(),
            roles: vec![NodeRole::Router as i32, NodeRole::Gpu as i32], // Mock roles
            zone: config.core.node.zone.clone().unwrap_or_default(),
            labels: config.core.node.labels.clone(),
            status: NodeStatus::Healthy as i32,
            last_seen: Some(mesh_proto::timestamp::now()),
            version: env!("CARGO_PKG_VERSION").to_string(),
        };
        
        nodes.insert(node.id.clone(), node);
        
        // Create state plane service
        let state_plane = StatePlaneService::new(config.clone(), metrics_registry.clone());
        
        Self {
            config,
            metrics_registry,
            nodes: std::sync::Arc::new(tokio::sync::RwLock::new(nodes)),
            state_plane,
            raft_node: None, // Will be set up later when raft integration is complete
            policies: std::sync::Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }
}

#[tonic::async_trait]
impl ControlPlane for ControlPlaneService {
    async fn list_nodes(
        &self,
        request: Request<ListNodesRequest>,
    ) -> std::result::Result<Response<ListNodesResponse>, Status> {
        debug!("Received ListNodes request: {:?}", request);
        
        let req = request.into_inner();
        let nodes = self.nodes.read().await;
        
        // Apply filters
        let filtered_nodes: Vec<Node> = nodes
            .values()
            .filter(|node| {
                // Filter by role if specified
                if !req.role_filter.is_empty() {
                    if !node.roles.iter().any(|role| req.role_filter.contains(role)) {
                        return false;
                    }
                }
                
                // Filter by zone if specified
                if !req.zone_filter.is_empty() && node.zone != req.zone_filter {
                    return false;
                }
                
                // Filter by labels if specified
                for (key, value) in &req.label_filter {
                    if node.labels.get(key) != Some(value) {
                        return false;
                    }
                }
                
                true
            })
            .cloned()
            .collect();
        
        info!("Returning {} nodes", filtered_nodes.len());
        
        let response = ListNodesResponse {
            nodes: filtered_nodes,
        };
        
        Ok(Response::new(response))
    }

    async fn get_node(
        &self,
        request: Request<GetNodeRequest>,
    ) -> std::result::Result<Response<GetNodeResponse>, Status> {
        debug!("Received GetNode request: {:?}", request);
        
        let req = request.into_inner();
        let nodes = self.nodes.read().await;
        
        if let Some(node) = nodes.get(&req.node_id) {
            let response = GetNodeResponse {
                node: Some(node.clone()),
            };
            Ok(Response::new(response))
        } else {
            Err(Status::not_found(format!("Node not found: {}", req.node_id)))
        }
    }

    async fn drain_node(
        &self,
        request: Request<DrainNodeRequest>,
    ) -> std::result::Result<Response<DrainNodeResponse>, Status> {
        debug!("Received DrainNode request: {:?}", request);
        
        let req = request.into_inner();
        let mut nodes = self.nodes.write().await;
        
        if let Some(node) = nodes.get_mut(&req.node_id) {
            info!("Draining node: {}", req.node_id);
            node.status = NodeStatus::Draining as i32;
            
            // In a real implementation, we would:
            // 1. Stop accepting new requests
            // 2. Wait for existing requests to complete
            // 3. Unload models
            // 4. Mark as offline
            
            let response = DrainNodeResponse {
                success: true,
                message: format!("Node {} is being drained", req.node_id),
            };
            
            Ok(Response::new(response))
        } else {
            Err(Status::not_found(format!("Node not found: {}", req.node_id)))
        }
    }

    async fn pin_model(
        &self,
        request: Request<PinModelRequest>,
    ) -> std::result::Result<Response<PinModelResponse>, Status> {
        debug!("Received PinModel request: {:?}", request);
        
        let req = request.into_inner();
        
        info!("Pinning model {}:{} to nodes: {:?}", 
              req.model_name, req.model_revision, req.target_nodes);
        
        // In a real implementation, we would:
        // 1. Validate the model exists
        // 2. Check target nodes are available
        // 3. Send load commands to target nodes
        // 4. Track the pin operation
        
        let pin_id = uuid::Uuid::new_v4().to_string();
        
        let response = PinModelResponse {
            success: true,
            message: format!("Model {}:{} pinned successfully", req.model_name, req.model_revision),
            pin_id,
        };
        
        Ok(Response::new(response))
    }

    async fn unpin_model(
        &self,
        request: Request<UnpinModelRequest>,
    ) -> std::result::Result<Response<UnpinModelResponse>, Status> {
        debug!("Received UnpinModel request: {:?}", request);
        
        let req = request.into_inner();
        
        info!("Unpinning model {}:{} from nodes: {:?}", 
              req.model_name, req.model_revision, req.target_nodes);
        
        // In a real implementation, we would:
        // 1. Find the pin record
        // 2. Send unload commands to target nodes
        // 3. Remove the pin record
        
        let response = UnpinModelResponse {
            success: true,
            message: format!("Model {}:{} unpinned successfully", req.model_name, req.model_revision),
        };
        
        Ok(Response::new(response))
    }

    async fn load_model(
        &self,
        request: Request<LoadModelRequest>,
    ) -> std::result::Result<Response<LoadModelResponse>, Status> {
        debug!("Received LoadModel request: {:?}", request);
        
        let req = request.into_inner();
        
        info!("Loading model {}:{} on node {}", 
              req.model_name, req.model_revision, req.target_node);
        
        // In a real implementation, we would:
        // 1. Validate the target node exists and is healthy
        // 2. Check if the node has capacity
        // 3. Send load command to the runtime adapter
        // 4. Wait for confirmation
        
        // Simulate loading time
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        let response = LoadModelResponse {
            success: true,
            message: format!("Model {}:{} loaded successfully on {}", 
                           req.model_name, req.model_revision, req.target_node),
        };
        
        Ok(Response::new(response))
    }

    async fn unload_model(
        &self,
        request: Request<UnloadModelRequest>,
    ) -> std::result::Result<Response<UnloadModelResponse>, Status> {
        debug!("Received UnloadModel request: {:?}", request);
        
        let req = request.into_inner();
        
        info!("Unloading model {}:{} from node {}", 
              req.model_name, req.model_revision, req.target_node);
        
        // In a real implementation, we would:
        // 1. Validate the target node exists
        // 2. Send unload command to the runtime adapter
        // 3. Wait for confirmation
        // 4. Update model registry
        
        let response = UnloadModelResponse {
            success: true,
            message: format!("Model {}:{} unloaded successfully from {}", 
                           req.model_name, req.model_revision, req.target_node),
        };
        
        Ok(Response::new(response))
    }

    // Additional methods would be implemented here for:
    // - list_models
    // - set_policy, get_policy, delete_policy, list_policies
    // - subscribe_events
    
    // For now, we'll implement stub versions that return "not implemented"
    
    async fn list_models(
        &self,
        request: Request<ListModelsRequest>,
    ) -> std::result::Result<Response<ListModelsResponse>, Status> {
        debug!("Received ListModels request: {:?}", request);
        
        let _req = request.into_inner();
        
        // Get model states from the state plane
        let model_states = self.state_plane.get_model_states().read().await;
        
        let mut models = Vec::new();
        
        for (_key, state) in model_states.iter() {
            // For now, include all models (filtering can be added later when protobuf is clarified)
            if let Some(ref labels) = state.labels {
                let model = mesh_proto::control::v1::Model {
                    name: labels.model.clone(),
                    revision: labels.revision.clone(),
                    runtime: labels.runtime.clone(),
                    status: if state.loaded { 1 } else { 0 }, // 1 = loaded, 0 = unloaded
                    pinned_nodes: vec![], // Will be populated from policy data
                    config: HashMap::new(), // Model configuration
                    created_at: state.last_updated.clone(),
                    updated_at: state.last_updated.clone(),
                };
                models.push(model);
            }
        }
        
        info!("Listed {} models (filtered from {} total)", models.len(), model_states.len());
        
        let response = ListModelsResponse {
            models,
        };
        
        Ok(Response::new(response))
    }

    async fn set_policy(
        &self,
        request: Request<SetPolicyRequest>,
    ) -> std::result::Result<Response<SetPolicyResponse>, Status> {
        debug!("Received SetPolicy request: {:?}", request);
        
        let req = request.into_inner();
        
        // For now, just create a simple policy based on the protobuf policy field
        if let Some(_policy_proto) = req.policy {
            // Create a basic policy ID from the policy name
            let policy_id = format!("policy_{}", uuid::Uuid::new_v4().simple());
            
            // Create a simple policy (will be enhanced when protobuf structure is clarified)
            let policy = Policy {
                id: policy_id.clone(),
                policy_type: PolicyType::ModelPin(ModelPinPolicy {
                    model_id: "default".to_string(),
                    model_version: None,
                    target_nodes: vec![],
                    min_replicas: 1,
                    max_replicas: None,
                    priority: PolicyPriority::Normal,
                    constraints: vec![],
                }),
                metadata: PolicyMetadata {
                    name: "Default Policy".to_string(),
                    description: Some("Auto-generated policy".to_string()),
                    version: 1,
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                    owner: Some("system".to_string()),
                    tags: HashMap::new(),
                    enabled: true,
                },
            };
            
            // Store policy (in-memory for now, will use raft later)
            {
                let mut policies = self.policies.write().await;
                policies.insert(policy_id.clone(), policy);
            }
            
            info!("Policy {} created successfully", policy_id);
            
            let response = SetPolicyResponse {
                success: true,
                message: "Policy created successfully".to_string(),
            };
            
            Ok(Response::new(response))
        } else {
            Err(Status::invalid_argument("Policy is required"))
        }
    }

    async fn get_policy(
        &self,
        _request: Request<mesh_proto::control::v1::GetPolicyRequest>,
    ) -> std::result::Result<Response<mesh_proto::control::v1::GetPolicyResponse>, Status> {
        warn!("GetPolicy not yet implemented");
        Err(Status::unimplemented("GetPolicy not yet implemented"))
    }

    async fn delete_policy(
        &self,
        _request: Request<mesh_proto::control::v1::DeletePolicyRequest>,
    ) -> std::result::Result<Response<mesh_proto::control::v1::DeletePolicyResponse>, Status> {
        warn!("DeletePolicy not yet implemented");
        Err(Status::unimplemented("DeletePolicy not yet implemented"))
    }

    async fn list_policies(
        &self,
        _request: Request<mesh_proto::control::v1::ListPoliciesRequest>,
    ) -> std::result::Result<Response<mesh_proto::control::v1::ListPoliciesResponse>, Status> {
        warn!("ListPolicies not yet implemented");
        Err(Status::unimplemented("ListPolicies not yet implemented"))
    }

    type SubscribeEventsStream = tokio_stream::wrappers::ReceiverStream<Result<mesh_proto::control::v1::Event, Status>>;

    async fn subscribe_events(
        &self,
        _request: Request<mesh_proto::control::v1::SubscribeEventsRequest>,
    ) -> std::result::Result<Response<Self::SubscribeEventsStream>, Status> {
        warn!("SubscribeEvents not yet implemented");
        Err(Status::unimplemented("SubscribeEvents not yet implemented"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_metrics::MetricsRegistryBuilder;

    fn create_test_service() -> ControlPlaneService {
        let config = AgentConfig::default();
        let metrics_registry = MetricsRegistryBuilder::new().build().unwrap();
        ControlPlaneService::new(config, metrics_registry)
    }

    #[tokio::test]
    async fn test_list_nodes() {
        let service = create_test_service();
        let request = Request::new(ListNodesRequest {
            role_filter: vec![],
            zone_filter: String::new(),
            label_filter: HashMap::new(),
        });

        let response = service.list_nodes(request).await.unwrap();
        let nodes = response.into_inner().nodes;
        
        assert_eq!(nodes.len(), 1); // Should have the local node
        assert!(!nodes[0].id.is_empty());
    }

    #[tokio::test]
    async fn test_get_node() {
        let service = create_test_service();
        let node_id = service.config.core.node.id.to_string();
        
        let request = Request::new(GetNodeRequest { node_id: node_id.clone() });
        let response = service.get_node(request).await.unwrap();
        let node = response.into_inner().node.unwrap();
        
        assert_eq!(node.id, node_id);
    }

    #[tokio::test]
    async fn test_get_node_not_found() {
        let service = create_test_service();
        let request = Request::new(GetNodeRequest { 
            node_id: "nonexistent".to_string() 
        });
        
        let result = service.get_node(request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn test_pin_model() {
        let service = create_test_service();
        let request = Request::new(PinModelRequest {
            model_name: "test-model".to_string(),
            model_revision: "v1.0".to_string(),
            target_nodes: vec!["node1".to_string()],
            config: HashMap::new(),
        });

        let response = service.pin_model(request).await.unwrap();
        let result = response.into_inner();
        
        assert!(result.success);
        assert!(!result.pin_id.is_empty());
    }

    #[tokio::test]
    async fn test_load_model() {
        let service = create_test_service();
        let request = Request::new(LoadModelRequest {
            model_name: "test-model".to_string(),
            model_revision: "v1.0".to_string(),
            target_node: "node1".to_string(),
            config: HashMap::new(),
        });

        let response = service.load_model(request).await.unwrap();
        let result = response.into_inner();
        
        assert!(result.success);
        assert!(result.message.contains("loaded successfully"));
    }
}
