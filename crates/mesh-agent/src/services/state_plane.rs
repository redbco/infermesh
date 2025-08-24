//! State Plane service implementation

use crate::config::AgentConfig;
use mesh_metrics::MetricsRegistry;
use mesh_proto::state::v1::{
    state_plane_server::StatePlane, GetGpuStateRequest, GetGpuStateResponse, GetModelStateRequest,
    GetModelStateResponse, GpuStateDelta, ListGpuStatesRequest, ListGpuStatesResponse,
    ListModelStatesRequest, ListModelStatesResponse, ModelStateDelta, GpuStateAck, ModelStateAck,
};
use std::collections::HashMap;
use tonic::{Request, Response, Status, Streaming};
use tracing::{debug, info, warn};

/// State Plane service implementation
#[derive(Debug, Clone)]
pub struct StatePlaneService {
    config: AgentConfig,
    metrics_registry: MetricsRegistry,
    // In a real implementation, these would be backed by persistent storage
    model_states: std::sync::Arc<tokio::sync::RwLock<HashMap<String, mesh_proto::state::v1::ModelState>>>,
    gpu_states: std::sync::Arc<tokio::sync::RwLock<HashMap<String, mesh_proto::state::v1::GpuState>>>,
}

impl StatePlaneService {
    /// Create a new State Plane service
    pub fn new(config: AgentConfig, metrics_registry: MetricsRegistry) -> Self {
        Self {
            config,
            metrics_registry,
            model_states: std::sync::Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            gpu_states: std::sync::Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    /// Generate a key for model state storage
    pub fn model_state_key(labels: &mesh_proto::state::v1::Labels) -> String {
        format!("{}:{}:{}:{}", labels.model, labels.revision, labels.runtime, labels.node)
    }

    /// Generate a key for GPU state storage
    pub fn gpu_state_key(gpu_uuid: &str, node: &str) -> String {
        format!("{}:{}", gpu_uuid, node)
    }
    
    /// Get access to model states for control plane
    pub fn get_model_states(&self) -> &std::sync::Arc<tokio::sync::RwLock<HashMap<String, mesh_proto::state::v1::ModelState>>> {
        &self.model_states
    }

    /// Get access to GPU states for adapter integration
    pub fn get_gpu_states(&self) -> &std::sync::Arc<tokio::sync::RwLock<HashMap<String, mesh_proto::state::v1::GpuState>>> {
        &self.gpu_states
    }
}

#[tonic::async_trait]
impl StatePlane for StatePlaneService {
    type ReportModelStateStream = tokio_stream::wrappers::ReceiverStream<std::result::Result<ModelStateAck, Status>>;

    async fn report_model_state(
        &self,
        request: Request<Streaming<ModelStateDelta>>,
    ) -> std::result::Result<Response<Self::ReportModelStateStream>, Status> {
        debug!("Received ReportModelState stream");

        let mut stream = request.into_inner();
        let model_states = self.model_states.clone();
        let _metrics_registry = self.metrics_registry.clone();

        let (tx, rx) = tokio::sync::mpsc::channel(100);

        tokio::spawn(async move {
            while let Some(delta_result) = stream.message().await.transpose() {
                match delta_result {
                    Ok(delta) => {
                        if let Some(ref labels) = delta.labels {
                            let key = Self::model_state_key(labels);
                            
                            // Update or create model state
                            let mut states = model_states.write().await;
                            let state = states.entry(key.clone()).or_insert_with(|| {
                                mesh_proto::state::v1::ModelState {
                                    labels: Some(labels.clone()),
                                    queue_depth: 0,
                                    service_rate: 0.0,
                                    p95_latency_ms: 0,
                                    batch_fullness: 0.0,
                                    loaded: false,
                                    warming: false,
                                    work_left_seconds: 0.0,
                                    last_updated: Some(mesh_proto::timestamp::now()),
                                }
                            });

                            // Apply delta updates
                            if let Some(queue_depth) = delta.queue_depth {
                                state.queue_depth = queue_depth;
                            }
                            if let Some(service_rate) = delta.service_rate {
                                state.service_rate = service_rate;
                            }
                            if let Some(p95_latency_ms) = delta.p95_latency_ms {
                                state.p95_latency_ms = p95_latency_ms;
                            }
                            if let Some(batch_fullness) = delta.batch_fullness {
                                state.batch_fullness = batch_fullness;
                            }
                            if let Some(loaded) = delta.loaded {
                                state.loaded = loaded;
                            }
                            if let Some(warming) = delta.warming {
                                state.warming = warming;
                            }
                            state.last_updated = delta.timestamp;

                            // Update metrics
                            // TODO: Implement conversion from proto to core types
                            // metrics_registry.update_model_state(&core_state);

                            debug!("Updated model state for key: {}", key);

                            // Send acknowledgment
                            let ack = ModelStateAck {
                                request_id: uuid::Uuid::new_v4().to_string(),
                                success: true,
                                error_message: String::new(),
                            };

                            if tx.send(Ok(ack)).await.is_err() {
                                warn!("Failed to send model state ack");
                                break;
                            }
                        } else {
                            warn!("Received model state delta without labels");
                            let ack = ModelStateAck {
                                request_id: uuid::Uuid::new_v4().to_string(),
                                success: false,
                                error_message: "Missing labels".to_string(),
                            };
                            let _ = tx.send(Ok(ack)).await;
                        }
                    }
                    Err(e) => {
                        warn!("Error receiving model state delta: {}", e);
                        let ack = ModelStateAck {
                            request_id: uuid::Uuid::new_v4().to_string(),
                            success: false,
                            error_message: format!("Stream error: {}", e),
                        };
                        let _ = tx.send(Ok(ack)).await;
                        break;
                    }
                }
            }
            info!("Model state reporting stream ended");
        });

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    type ReportGpuStateStream = tokio_stream::wrappers::ReceiverStream<std::result::Result<GpuStateAck, Status>>;

    async fn report_gpu_state(
        &self,
        request: Request<Streaming<GpuStateDelta>>,
    ) -> std::result::Result<Response<Self::ReportGpuStateStream>, Status> {
        debug!("Received ReportGpuState stream");

        let mut stream = request.into_inner();
        let gpu_states = self.gpu_states.clone();
        let _metrics_registry = self.metrics_registry.clone();

        let (tx, rx) = tokio::sync::mpsc::channel(100);

        tokio::spawn(async move {
            while let Some(delta_result) = stream.message().await.transpose() {
                match delta_result {
                    Ok(delta) => {
                        let key = Self::gpu_state_key(&delta.gpu_uuid, &delta.node);
                        
                        // Update or create GPU state
                        let mut states = gpu_states.write().await;
                        let state = states.entry(key.clone()).or_insert_with(|| {
                            mesh_proto::state::v1::GpuState {
                                gpu_uuid: delta.gpu_uuid.clone(),
                                node: delta.node.clone(),
                                mig_profile: String::new(),
                                sm_utilization: 0.0,
                                memory_utilization: 0.0,
                                vram_used_gb: 0.0,
                                vram_total_gb: 0.0,
                                temperature_c: 0.0,
                                power_watts: 0.0,
                                ecc_errors: false,
                                throttled: false,
                                last_updated: Some(mesh_proto::timestamp::now()),
                            }
                        });

                        // Apply delta updates
                        if let Some(sm_utilization) = delta.sm_utilization {
                            state.sm_utilization = sm_utilization;
                        }
                        if let Some(memory_utilization) = delta.memory_utilization {
                            state.memory_utilization = memory_utilization;
                        }
                        if let Some(vram_used_gb) = delta.vram_used_gb {
                            state.vram_used_gb = vram_used_gb;
                        }
                        if let Some(vram_total_gb) = delta.vram_total_gb {
                            state.vram_total_gb = vram_total_gb;
                        }
                        if let Some(temperature_c) = delta.temperature_c {
                            state.temperature_c = temperature_c;
                        }
                        if let Some(power_watts) = delta.power_watts {
                            state.power_watts = power_watts;
                        }
                        if let Some(ecc_errors) = delta.ecc_errors {
                            state.ecc_errors = ecc_errors;
                        }
                        if let Some(throttled) = delta.throttled {
                            state.throttled = throttled;
                        }
                        state.last_updated = delta.timestamp;

                        // Update metrics
                        // TODO: Implement conversion from proto to core types
                        // let core_state: mesh_core::GpuState = state.clone().into();
                        // metrics_registry.update_gpu_state(&core_state);

                        debug!("Updated GPU state for key: {}", key);

                        // Send acknowledgment
                        let ack = GpuStateAck {
                            request_id: uuid::Uuid::new_v4().to_string(),
                            success: true,
                            error_message: String::new(),
                        };

                        if tx.send(Ok(ack)).await.is_err() {
                            warn!("Failed to send GPU state ack");
                            break;
                        }
                    }
                    Err(e) => {
                        warn!("Error receiving GPU state delta: {}", e);
                        let ack = GpuStateAck {
                            request_id: uuid::Uuid::new_v4().to_string(),
                            success: false,
                            error_message: format!("Stream error: {}", e),
                        };
                        let _ = tx.send(Ok(ack)).await;
                        break;
                    }
                }
            }
            info!("GPU state reporting stream ended");
        });

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn get_model_state(
        &self,
        request: Request<GetModelStateRequest>,
    ) -> std::result::Result<Response<GetModelStateResponse>, Status> {
        debug!("Received GetModelState request: {:?}", request);

        let req = request.into_inner();
        if let Some(labels) = req.labels {
            let key = Self::model_state_key(&labels);
            let states = self.model_states.read().await;
            
            if let Some(state) = states.get(&key) {
                let response = GetModelStateResponse {
                    state: Some(state.clone()),
                    found: true,
                };
                Ok(Response::new(response))
            } else {
                let response = GetModelStateResponse {
                    state: None,
                    found: false,
                };
                Ok(Response::new(response))
            }
        } else {
            Err(Status::invalid_argument("Missing labels"))
        }
    }

    async fn get_gpu_state(
        &self,
        request: Request<GetGpuStateRequest>,
    ) -> std::result::Result<Response<GetGpuStateResponse>, Status> {
        debug!("Received GetGpuState request: {:?}", request);

        let req = request.into_inner();
        let key = Self::gpu_state_key(&req.gpu_uuid, &req.node);
        let states = self.gpu_states.read().await;
        
        if let Some(state) = states.get(&key) {
            let response = GetGpuStateResponse {
                state: Some(state.clone()),
                found: true,
            };
            Ok(Response::new(response))
        } else {
            let response = GetGpuStateResponse {
                state: None,
                found: false,
            };
            Ok(Response::new(response))
        }
    }

    async fn list_model_states(
        &self,
        request: Request<ListModelStatesRequest>,
    ) -> std::result::Result<Response<ListModelStatesResponse>, Status> {
        debug!("Received ListModelStates request: {:?}", request);

        let req = request.into_inner();
        let states = self.model_states.read().await;
        
        // Apply filters
        let filtered_states: Vec<mesh_proto::state::v1::ModelState> = states
            .values()
            .filter(|state| {
                if let Some(ref labels) = state.labels {
                    // Filter by model if specified
                    if !req.model_filter.is_empty() && labels.model != req.model_filter {
                        return false;
                    }
                    
                    // Filter by runtime if specified
                    if !req.runtime_filter.is_empty() && labels.runtime != req.runtime_filter {
                        return false;
                    }
                    
                    // Filter by node if specified
                    if !req.node_filter.is_empty() && labels.node != req.node_filter {
                        return false;
                    }
                    
                    // Filter by loaded status if specified
                    if req.loaded_only && !state.loaded {
                        return false;
                    }
                    
                    true
                } else {
                    false
                }
            })
            .cloned()
            .collect();

        info!("Returning {} model states", filtered_states.len());

        let response = ListModelStatesResponse {
            states: filtered_states,
        };

        Ok(Response::new(response))
    }

    async fn list_gpu_states(
        &self,
        request: Request<ListGpuStatesRequest>,
    ) -> std::result::Result<Response<ListGpuStatesResponse>, Status> {
        debug!("Received ListGpuStates request: {:?}", request);

        let req = request.into_inner();
        let states = self.gpu_states.read().await;
        
        // Apply filters
        let filtered_states: Vec<mesh_proto::state::v1::GpuState> = states
            .values()
            .filter(|state| {
                // Filter by node if specified
                if !req.node_filter.is_empty() && state.node != req.node_filter {
                    return false;
                }
                
                // Filter by healthy status if specified
                if req.healthy_only && (state.ecc_errors || state.throttled) {
                    return false;
                }
                
                true
            })
            .cloned()
            .collect();

        info!("Returning {} GPU states", filtered_states.len());

        let response = ListGpuStatesResponse {
            states: filtered_states,
        };

        Ok(Response::new(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_metrics::MetricsRegistryBuilder;

    fn create_test_service() -> StatePlaneService {
        let config = AgentConfig::default();
        let metrics_registry = MetricsRegistryBuilder::new().build().unwrap();
        StatePlaneService::new(config, metrics_registry)
    }

    #[tokio::test]
    async fn test_get_model_state_not_found() {
        let service = create_test_service();
        let labels = mesh_proto::state::v1::Labels {
            model: "test-model".to_string(),
            revision: "v1.0".to_string(),
            runtime: "test".to_string(),
            node: "test-node".to_string(),
            ..Default::default()
        };
        
        let request = Request::new(GetModelStateRequest {
            labels: Some(labels),
        });

        let response = service.get_model_state(request).await.unwrap();
        let result = response.into_inner();
        
        assert!(!result.found);
        assert!(result.state.is_none());
    }

    #[tokio::test]
    async fn test_get_gpu_state_not_found() {
        let service = create_test_service();
        let request = Request::new(GetGpuStateRequest {
            gpu_uuid: "GPU-12345".to_string(),
            node: "test-node".to_string(),
        });

        let response = service.get_gpu_state(request).await.unwrap();
        let result = response.into_inner();
        
        assert!(!result.found);
        assert!(result.state.is_none());
    }

    #[tokio::test]
    async fn test_list_model_states_empty() {
        let service = create_test_service();
        let request = Request::new(ListModelStatesRequest {
            model_filter: String::new(),
            runtime_filter: String::new(),
            node_filter: String::new(),
            loaded_only: false,
        });

        let response = service.list_model_states(request).await.unwrap();
        let result = response.into_inner();
        
        assert_eq!(result.states.len(), 0);
    }

    #[tokio::test]
    async fn test_list_gpu_states_empty() {
        let service = create_test_service();
        let request = Request::new(ListGpuStatesRequest {
            node_filter: String::new(),
            healthy_only: false,
        });

        let response = service.list_gpu_states(request).await.unwrap();
        let result = response.into_inner();
        
        assert_eq!(result.states.len(), 0);
    }
}
