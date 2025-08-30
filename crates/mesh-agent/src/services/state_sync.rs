//! State synchronization service for connecting adapters with state plane

use crate::services::{AdapterManager, StatePlaneService};
use mesh_state::{StateStore, StateSynchronizer};
use mesh_core::{NodeId, ModelState, GpuState, Labels};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::interval;
use tracing::{info, warn};

/// State synchronization service that bridges adapters and state plane
pub struct StateSyncService {
    adapter_manager: Option<Arc<AdapterManager>>,
    state_plane: Option<Arc<StatePlaneService>>,
    state_store: Arc<StateStore>,
    #[allow(unused)]
    state_synchronizer: Arc<StateSynchronizer>,
    node_id: NodeId,
    sync_interval: Duration,
    running: Arc<RwLock<bool>>,
}

impl StateSyncService {
    /// Create a new state synchronization service
    pub fn new(
        state_store: Arc<StateStore>,
        state_synchronizer: Arc<StateSynchronizer>,
        node_id: NodeId,
        sync_interval: Duration,
    ) -> Self {
        Self {
            adapter_manager: None,
            state_plane: None,
            state_store,
            state_synchronizer,
            node_id,
            sync_interval,
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Set the adapter manager
    pub fn set_adapter_manager(&mut self, adapter_manager: Arc<AdapterManager>) {
        self.adapter_manager = Some(adapter_manager);
    }

    /// Set the state plane service
    pub fn set_state_plane(&mut self, state_plane: Arc<StatePlaneService>) {
        self.state_plane = Some(state_plane);
    }



    /// Start the state synchronization service
    pub async fn start(&self) -> crate::Result<()> {
        info!("Starting state synchronization service");
        
        {
            let mut running = self.running.write().await;
            *running = true;
        }

        // Start state synchronization loop
        self.start_sync_loop().await?;

        // Start adapter telemetry integration
        self.start_telemetry_integration().await?;

        info!("State synchronization service started successfully");
        Ok(())
    }

    /// Stop the state synchronization service
    pub async fn stop(&self) -> crate::Result<()> {
        info!("Stopping state synchronization service");
        
        {
            let mut running = self.running.write().await;
            *running = false;
        }

        info!("State synchronization service stopped");
        Ok(())
    }



    /// Start the main synchronization loop
    async fn start_sync_loop(&self) -> crate::Result<()> {
        let state_store = self.state_store.clone();
        let state_plane = self.state_plane.clone();
        let node_id = self.node_id.clone();
        let sync_interval = self.sync_interval;
        let running = self.running.clone();

        tokio::spawn(async move {
            let mut interval = interval(sync_interval);
            
            while *running.read().await {
                interval.tick().await;
                
                // Sync state plane data to local store
                if let Some(ref state_plane) = state_plane {
                    Self::sync_state_plane_to_store(&state_store, state_plane, &node_id).await;
                }
            }
            
            info!("State synchronization loop stopped");
        });

        Ok(())
    }

    /// Start telemetry integration with adapters
    async fn start_telemetry_integration(&self) -> crate::Result<()> {
        let adapter_manager = self.adapter_manager.clone();
        let state_store = self.state_store.clone();
        let node_id = self.node_id.clone();
        let running = self.running.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10)); // Collect telemetry every 10 seconds
            
            while *running.read().await {
                interval.tick().await;
                
                if let Some(ref adapter_manager) = adapter_manager {
                    let (runtime_metrics, gpu_metrics) = adapter_manager.get_telemetry_data().await;
                    
                    // Convert and store runtime metrics
                    for runtime_metric in runtime_metrics {
                        if let Some(model_state) = Self::convert_runtime_metric_to_model_state(&runtime_metric, &node_id) {
                            if let Err(e) = state_store.update_model_state(node_id.clone(), model_state.labels.model.clone(), model_state).await {
                                warn!("Failed to update model state in store: {}", e);
                            }
                        }
                    }
                    
                    // Convert and store GPU metrics
                    for gpu_metric in gpu_metrics {
                        if let Some(gpu_state) = Self::convert_gpu_metric_to_gpu_state(&gpu_metric, &node_id) {
                            if let Err(e) = state_store.update_gpu_state(node_id.clone(), gpu_metric.info.uuid.clone(), gpu_state).await {
                                warn!("Failed to update GPU state in store: {}", e);
                            }
                        }
                    }
                }
            }
            
            info!("Telemetry integration loop stopped");
        });

        Ok(())
    }

    /// Sync state plane data to local store
    async fn sync_state_plane_to_store(
        state_store: &StateStore,
        state_plane: &StatePlaneService,
        node_id: &NodeId,
    ) {
        // Sync model states
        {
            let model_states = state_plane.get_model_states().read().await;
            for (_key, proto_state) in model_states.iter() {
                if let Some(ref labels) = proto_state.labels {
                    if labels.node == node_id.to_string() {
                        if let Some(model_state) = Self::convert_proto_model_state_to_core(proto_state) {
                            if let Err(e) = state_store.update_model_state(node_id.clone(), labels.model.clone(), model_state).await {
                                warn!("Failed to sync model state to store: {}", e);
                            }
                        }
                    }
                }
            }
        }

        // Sync GPU states
        {
            let gpu_states = state_plane.get_gpu_states().read().await;
            for (_key, proto_state) in gpu_states.iter() {
                if proto_state.node == node_id.to_string() {
                    if let Some(gpu_state) = Self::convert_proto_gpu_state_to_core(proto_state) {
                        if let Err(e) = state_store.update_gpu_state(node_id.clone(), proto_state.gpu_uuid.clone(), gpu_state).await {
                            warn!("Failed to sync GPU state to store: {}", e);
                        }
                    }
                }
            }
        }
    }



    /// Convert runtime metrics to core ModelState
    fn convert_runtime_metric_to_model_state(
        runtime_metric: &mesh_adapter_runtime::RuntimeMetrics,
        node_id: &NodeId,
    ) -> Option<ModelState> {
        // Use the first model if available
        let (model_name, model_metrics) = runtime_metric.models.iter().next()?;
        
        let labels = Labels {
            model: model_name.clone(),
            revision: model_metrics.version.clone().unwrap_or_else(|| "latest".to_string()),
            quant: None,
            runtime: runtime_metric.runtime_specific.get("runtime_type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            node: node_id.to_string(),
            gpu_uuid: None,
            mig_profile: None,
            tenant: None,
            zone: None,
            custom: HashMap::new(),
        };

        Some(ModelState {
            labels,
            queue_depth: runtime_metric.requests.queue_size as u32,
            service_rate: runtime_metric.requests.requests_per_second,
            p95_latency_ms: runtime_metric.requests.p95_request_duration_ms as u32,
            batch_fullness: 0.0, // Not directly available
            loaded: model_metrics.status == "loaded" || model_metrics.status == "ready",
            warming: model_metrics.status == "loading" || model_metrics.status == "warming",
            work_left_seconds: 0.0, // Not directly available
            last_updated: chrono::Utc::now(),
        })
    }

    /// Convert GPU metrics to core GpuState
    fn convert_gpu_metric_to_gpu_state(
        gpu_metric: &mesh_adapter_gpu::GpuMetrics,
        node_id: &NodeId,
    ) -> Option<GpuState> {
        Some(GpuState {
            gpu_uuid: gpu_metric.info.uuid.clone(),
            node: node_id.to_string(),
            mig_profile: Some(String::new()),
            sm_utilization: gpu_metric.utilization.gpu as f32,
            memory_utilization: gpu_metric.utilization.memory as f32,
            vram_used_gb: (gpu_metric.memory.used as f64 / 1024.0 / 1024.0 / 1024.0) as f32,
            vram_total_gb: (gpu_metric.memory.total as f64 / 1024.0 / 1024.0 / 1024.0) as f32,
            temperature_c: Some(gpu_metric.temperature.gpu as f32),
            power_watts: Some(gpu_metric.power.usage as f32),
            ecc_errors: false, // Could be derived from GPU health
            throttled: matches!(gpu_metric.temperature.thermal_state, mesh_adapter_gpu::metrics::ThermalState::Throttling),
            last_updated: chrono::Utc::now(),
        })
    }

    /// Convert protobuf ModelState to core ModelState
    fn convert_proto_model_state_to_core(proto_state: &mesh_proto::state::v1::ModelState) -> Option<ModelState> {
        let labels = proto_state.labels.as_ref()?;
        
        Some(ModelState {
            labels: Labels {
                model: labels.model.clone(),
                revision: labels.revision.clone(),
                quant: if labels.quant.is_empty() { None } else { Some(labels.quant.clone()) },
                runtime: labels.runtime.clone(),
                node: labels.node.clone(),
                gpu_uuid: if labels.gpu_uuid.is_empty() { None } else { Some(labels.gpu_uuid.clone()) },
                mig_profile: if labels.mig_profile.is_empty() { None } else { Some(labels.mig_profile.clone()) },
                tenant: if labels.tenant.is_empty() { None } else { Some(labels.tenant.clone()) },
                zone: if labels.zone.is_empty() { None } else { Some(labels.zone.clone()) },
                custom: labels.custom.clone(),
            },
            queue_depth: proto_state.queue_depth,
            service_rate: proto_state.service_rate,
            p95_latency_ms: proto_state.p95_latency_ms,
            batch_fullness: proto_state.batch_fullness,
            loaded: proto_state.loaded,
            warming: proto_state.warming,
            work_left_seconds: proto_state.work_left_seconds,
            last_updated: chrono::Utc::now(),
        })
    }

    /// Convert protobuf GpuState to core GpuState
    fn convert_proto_gpu_state_to_core(proto_state: &mesh_proto::state::v1::GpuState) -> Option<GpuState> {
        Some(GpuState {
            gpu_uuid: proto_state.gpu_uuid.clone(),
            node: proto_state.node.clone(),
            mig_profile: Some(proto_state.mig_profile.clone()),
            sm_utilization: proto_state.sm_utilization,
            memory_utilization: proto_state.memory_utilization,
            vram_used_gb: proto_state.vram_used_gb,
            vram_total_gb: proto_state.vram_total_gb,
            temperature_c: Some(proto_state.temperature_c),
            power_watts: Some(proto_state.power_watts),
            ecc_errors: proto_state.ecc_errors,
            throttled: proto_state.throttled,
            last_updated: chrono::Utc::now(),
        })
    }
}
