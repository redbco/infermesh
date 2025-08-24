# Technical Implementation Specification - infermesh

This document provides detailed technical specifications for implementing the missing components in infermesh.

---

## 🔧 **Phase A: Compilation Fixes - Technical Details**

### A.1 mesh-raft Storage Trait Redesign

#### Problem Analysis
```rust
// Current problematic code in crates/mesh-raft/src/node.rs:27
raw_node: Arc<Mutex<RawNode<Box<dyn RaftStorage>>>>,
```

**Issues**:
1. `RaftStorage` trait is not dyn-compatible due to generic methods
2. `Box<dyn RaftStorage>` doesn't implement `Storage` trait required by tikv-raft
3. Trait object approach conflicts with tikv-raft's concrete type requirements

#### Solution Architecture
```rust
// New approach: Concrete enum instead of trait objects
#[derive(Debug)]
pub enum RaftStorageBackend {
    Memory(MemoryStorage),
    Disk(DiskStorage),
}

impl Storage for RaftStorageBackend {
    fn initial_state(&self) -> raft::Result<RaftState> {
        match self {
            Self::Memory(storage) => storage.initial_state(),
            Self::Disk(storage) => storage.initial_state(),
        }
    }
    
    fn entries(&self, low: u64, high: u64, max_size: impl Into<Option<u64>>) -> raft::Result<Vec<Entry>> {
        match self {
            Self::Memory(storage) => storage.entries(low, high, max_size),
            Self::Disk(storage) => storage.entries(low, high, max_size),
        }
    }
    
    // ... implement all Storage trait methods with delegation
}
```

#### Implementation Steps
1. **Create new enum** in `crates/mesh-raft/src/storage.rs`
2. **Implement Storage trait** with match-based delegation
3. **Update RaftNode** to use concrete enum type
4. **Update all RawNode usage** throughout the crate
5. **Remove RaftStorage trait** or make it internal-only

### A.2 Serialization Fix

#### Problem Analysis
```rust
// Current problematic code in crates/mesh-raft/src/storage.rs:104
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedState {
    hard_state: HardState,      // tikv-raft type, no serde support
    conf_state: ConfState,      // tikv-raft type, no serde support
    entries: Vec<Entry>,        // tikv-raft type, no serde support
    snapshot: Option<Snapshot>, // tikv-raft type, no serde support
}
```

#### Solution Architecture
```rust
// Create serializable wrapper types
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializablePersistedState {
    hard_state: SerializableHardState,
    conf_state: Vec<u64>, // Simplified representation
    entries: Vec<SerializableEntry>,
    snapshot_data: Option<Vec<u8>>,
    snapshot_metadata: Option<SerializableSnapshotMetadata>,
}

// Conversion functions
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
```

### A.3 API Mismatch Fixes

#### Field vs Method Access
```rust
// Fix in crates/mesh-raft/src/node.rs:375
// WRONG:
if entry.entry_type() == EntryType::EntryNormal && !entry.data.is_empty() {

// CORRECT:
if entry.entry_type == EntryType::EntryNormal && !entry.data.is_empty() {
```

#### Bytes Conversion
```rust
// Fix in crates/mesh-raft/src/storage.rs:225
// WRONG:
new_snapshot.data = data.clone();

// CORRECT:
new_snapshot.data = bytes::Bytes::from(data.clone());
```

#### Default Implementation
```rust
// Fix in crates/mesh-raft/src/node.rs:55
#[derive(Debug)]
pub struct RaftNodeStats {
    // Remove Default derive, implement manually
    pub start_time: Instant,
    // ... other fields
}

impl Default for RaftNodeStats {
    fn default() -> Self {
        Self {
            start_time: Instant::now(), // Use now() instead of default
            term: AtomicU64::new(0),
            // ... initialize other fields
        }
    }
}
```

---

## 🔧 **Phase B: Core Services - Technical Details**

### B.1 Control Plane Service Implementation

#### list_models() Implementation
```rust
// File: crates/mesh-agent/src/services/control_plane.rs
async fn list_models(
    &self,
    request: Request<ListModelsRequest>,
) -> Result<Response<ListModelsResponse>, Status> {
    let req = request.into_inner();
    
    // Get models from state store
    let state_store = self.get_state_store().await?;
    let models = state_store.list_models(&req.node_filter, &req.model_filter).await
        .map_err(|e| Status::internal(format!("Failed to list models: {}", e)))?;
    
    // Convert to protobuf response
    let response = ListModelsResponse {
        models: models.into_iter().map(|m| m.into()).collect(),
        total_count: models.len() as u32,
    };
    
    Ok(Response::new(response))
}
```

#### Policy Management Integration
```rust
async fn set_policy(
    &self,
    request: Request<SetPolicyRequest>,
) -> Result<Response<SetPolicyResponse>, Status> {
    let req = request.into_inner();
    
    // Validate policy
    let policy = Policy::try_from(req.policy.ok_or_else(|| {
        Status::invalid_argument("Policy is required")
    })?)?;
    
    // Check if we're the raft leader
    let raft_node = self.get_raft_node().await?;
    if !raft_node.is_leader().await {
        return Err(Status::failed_precondition("Not the raft leader"));
    }
    
    // Propose policy to raft
    let operation = PolicyOperation::Set {
        policy_id: policy.id.clone(),
        policy,
    };
    
    let result = raft_node.propose_policy(operation).await
        .map_err(|e| Status::internal(format!("Raft proposal failed: {}", e)))?;
    
    let response = SetPolicyResponse {
        policy_id: result.policy_id.unwrap_or_default(),
        success: result.success,
        error_message: result.error,
    };
    
    Ok(Response::new(response))
}
```

#### Event Streaming Implementation
```rust
type SubscribeEventsStream = Pin<Box<dyn Stream<Item = Result<Event, Status>> + Send>>;

async fn subscribe_events(
    &self,
    request: Request<SubscribeEventsRequest>,
) -> Result<Response<Self::SubscribeEventsStream>, Status> {
    let req = request.into_inner();
    
    // Create event stream
    let (tx, rx) = mpsc::channel(100);
    
    // Subscribe to event bus
    let event_bus = self.get_event_bus().await?;
    let subscription = event_bus.subscribe(req.event_types, req.filters).await?;
    
    // Spawn task to forward events
    tokio::spawn(async move {
        while let Some(event) = subscription.recv().await {
            if tx.send(Ok(event)).await.is_err() {
                break; // Client disconnected
            }
        }
    });
    
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Ok(Response::new(Box::pin(stream)))
}
```

### B.2 Router Proxy Implementation

#### HTTP Request Forwarding
```rust
// File: crates/mesh-router/src/proxy.rs
pub async fn forward_http_request(
    &self,
    context: &RequestContext,
    target: &RoutingTarget,
    method: hyper::Method,
    uri: hyper::Uri,
    headers: hyper::HeaderMap,
    body: hyper::body::Bytes,
) -> Result<ProxyResponse> {
    let client = hyper::Client::builder()
        .pool_timeout(self.timeout)
        .build_http();
    
    // Build target URL
    let target_url = format!("http://{}:{}{}", 
        target.address, target.port, uri.path_and_query().map(|pq| pq.as_str()).unwrap_or(""));
    
    // Create request
    let mut req_builder = hyper::Request::builder()
        .method(method)
        .uri(target_url);
    
    // Copy headers
    for (name, value) in headers.iter() {
        req_builder = req_builder.header(name, value);
    }
    
    // Add tracing headers
    req_builder = req_builder.header("x-request-id", &context.request_id);
    
    let request = req_builder.body(hyper::Body::from(body))
        .map_err(|e| ProxyError::RequestBuild(e.to_string()))?;
    
    // Send request with timeout
    let start_time = Instant::now();
    let response = tokio::time::timeout(self.timeout, client.request(request))
        .await
        .map_err(|_| ProxyError::Timeout)?
        .map_err(|e| ProxyError::Network(e.to_string()))?;
    
    // Convert response
    let status = response.status().as_u16();
    let headers = response.headers().clone();
    let body = hyper::body::to_bytes(response.into_body())
        .await
        .map_err(|e| ProxyError::ResponseRead(e.to_string()))?;
    
    Ok(ProxyResponse {
        status,
        headers: headers.into_iter().collect(),
        body,
        response_time_ms: start_time.elapsed().as_millis() as u64,
        target: target.clone(),
    })
}
```

#### gRPC Request Proxying
```rust
pub async fn forward_grpc_request(
    &self,
    context: &RequestContext,
    target: &RoutingTarget,
    service_name: &str,
    method_name: &str,
    request_data: Bytes,
) -> Result<ProxyResponse> {
    // Create gRPC channel to target
    let endpoint = tonic::transport::Endpoint::from_shared(
        format!("http://{}:{}", target.address, target.port)
    )?;
    
    let channel = endpoint
        .timeout(self.timeout)
        .connect()
        .await
        .map_err(|e| ProxyError::Connection(e.to_string()))?;
    
    // Create generic gRPC client
    let mut client = tonic::client::Grpc::new(channel);
    
    // Build request with metadata
    let mut request = tonic::Request::new(request_data);
    request.metadata_mut().insert("x-request-id", 
        context.request_id.parse().unwrap());
    
    // Forward request
    let start_time = Instant::now();
    let response = client
        .unary(request, tonic::codegen::Path::new(service_name, method_name), tonic::codegen::Codec::default())
        .await
        .map_err(|e| ProxyError::Grpc(e.to_string()))?;
    
    // Convert response
    let (metadata, response_data, _) = response.into_parts();
    
    Ok(ProxyResponse {
        status: 200, // gRPC success
        headers: metadata.into_headers().into_iter().collect(),
        body: response_data,
        response_time_ms: start_time.elapsed().as_millis() as u64,
        target: target.clone(),
    })
}
```

---

## 🔌 **Phase C: Adapter Implementations - Technical Details**

### C.1 Triton Adapter Implementation

#### Protobuf Integration
```toml
# Add to crates/mesh-adapter-runtime/Cargo.toml
[build-dependencies]
tonic-build = "0.12"

[dependencies]
# Triton inference server protobuf
tritonclient = { git = "https://github.com/triton-inference-server/client", features = ["grpc"] }
```

#### Triton gRPC Client
```rust
// File: crates/mesh-adapter-runtime/src/triton.rs
use tritonclient::grpc::{
    inference_client::InferenceClient,
    ModelInferRequest, ModelInferResponse,
    ServerLiveRequest, ServerReadyRequest,
    RepositoryModelLoadRequest, RepositoryModelUnloadRequest,
};

pub struct TritonAdapter {
    config: RuntimeConfig,
    client: Option<InferenceClient<tonic::transport::Channel>>,
    metrics_collector: Arc<RwLock<MetricCollector>>,
}

impl TritonAdapter {
    pub async fn new(config: RuntimeConfig) -> Result<Self> {
        let endpoint = tonic::transport::Endpoint::from_shared(config.endpoint.clone())?;
        let channel = endpoint.connect().await?;
        let client = InferenceClient::new(channel);
        
        Ok(Self {
            config,
            client: Some(client),
            metrics_collector: Arc::new(RwLock::new(MetricCollector::new())),
        })
    }
}

#[async_trait]
impl RuntimeAdapterTrait for TritonAdapter {
    async fn health_check(&self) -> Result<HealthStatus> {
        let mut client = self.client.as_ref().unwrap().clone();
        
        // Check server live
        let live_request = ServerLiveRequest {};
        let live_response = client.server_live(live_request).await?;
        
        // Check server ready
        let ready_request = ServerReadyRequest {};
        let ready_response = client.server_ready(ready_request).await?;
        
        Ok(HealthStatus {
            healthy: live_response.into_inner().live && ready_response.into_inner().ready,
            message: "Triton server status".to_string(),
            details: HashMap::new(),
        })
    }
    
    async fn load_model(&self, name: &str, config: Option<ModelConfig>) -> Result<()> {
        let mut client = self.client.as_ref().unwrap().clone();
        
        let request = RepositoryModelLoadRequest {
            model_name: name.to_string(),
            parameters: config.map(|c| c.into_triton_params()).unwrap_or_default(),
        };
        
        client.repository_model_load(request).await?;
        Ok(())
    }
    
    async fn inference(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let mut client = self.client.as_ref().unwrap().clone();
        
        // Convert request to Triton format
        let triton_request = ModelInferRequest {
            model_name: request.model_name,
            model_version: request.model_version.unwrap_or_default(),
            inputs: request.inputs.into_iter().map(|i| i.into_triton_input()).collect(),
            outputs: request.outputs.into_iter().map(|o| o.into_triton_output()).collect(),
            parameters: request.parameters,
        };
        
        let start_time = Instant::now();
        let response = client.model_infer(triton_request).await?;
        let inference_time = start_time.elapsed();
        
        // Convert response
        let triton_response = response.into_inner();
        Ok(InferenceResponse {
            model_name: triton_response.model_name,
            outputs: triton_response.outputs.into_iter().map(|o| o.into_output()).collect(),
            metadata: ResponseMetadata {
                inference_time_ms: inference_time.as_millis() as f64,
                ..Default::default()
            },
        })
    }
}
```

### C.2 NVML Adapter Implementation

#### NVML Library Integration
```toml
# Add to crates/mesh-adapter-gpu/Cargo.toml
[dependencies]
nvml-wrapper = "0.9"
# OR
nvidia-ml-py = "12.0"
```

#### NVML GPU Monitoring
```rust
// File: crates/mesh-adapter-gpu/src/nvml.rs
use nvml_wrapper::{Nvml, Device, error::NvmlError};

pub struct NvmlMonitor {
    config: GpuMonitorConfig,
    nvml: Nvml,
    devices: Vec<Device>,
}

impl NvmlMonitor {
    pub async fn new(config: GpuMonitorConfig) -> Result<Self> {
        let nvml = Nvml::init()
            .map_err(|e| GpuError::Initialization(format!("NVML init failed: {}", e)))?;
        
        let device_count = nvml.device_count()
            .map_err(|e| GpuError::Discovery(format!("Failed to get device count: {}", e)))?;
        
        let mut devices = Vec::new();
        for i in 0..device_count {
            let device = nvml.device_by_index(i)
                .map_err(|e| GpuError::Discovery(format!("Failed to get device {}: {}", i, e)))?;
            devices.push(device);
        }
        
        Ok(Self {
            config,
            nvml,
            devices,
        })
    }
}

#[async_trait]
impl GpuMonitorTrait for NvmlMonitor {
    async fn get_gpu_metrics(&self, gpu_id: u32) -> Result<GpuMetrics> {
        let device = self.devices.get(gpu_id as usize)
            .ok_or_else(|| GpuError::InvalidGpu(gpu_id))?;
        
        // Collect basic info
        let name = device.name()
            .map_err(|e| GpuError::MetricCollection(format!("Failed to get name: {}", e)))?;
        let uuid = device.uuid()
            .map_err(|e| GpuError::MetricCollection(format!("Failed to get UUID: {}", e)))?;
        
        // Memory information
        let memory_info = device.memory_info()
            .map_err(|e| GpuError::MetricCollection(format!("Failed to get memory info: {}", e)))?;
        
        // Utilization
        let utilization = device.utilization_rates()
            .map_err(|e| GpuError::MetricCollection(format!("Failed to get utilization: {}", e)))?;
        
        // Temperature
        let temperature = device.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
            .map_err(|e| GpuError::MetricCollection(format!("Failed to get temperature: {}", e)))?;
        
        // Power
        let power_usage = device.power_usage()
            .map_err(|e| GpuError::MetricCollection(format!("Failed to get power usage: {}", e)))?;
        let power_limit = device.enforced_power_limit()
            .map_err(|e| GpuError::MetricCollection(format!("Failed to get power limit: {}", e)))?;
        
        // Clock speeds
        let graphics_clock = device.clock_info(nvml_wrapper::enum_wrappers::device::Clock::Graphics)
            .map_err(|e| GpuError::MetricCollection(format!("Failed to get graphics clock: {}", e)))?;
        let memory_clock = device.clock_info(nvml_wrapper::enum_wrappers::device::Clock::Memory)
            .map_err(|e| GpuError::MetricCollection(format!("Failed to get memory clock: {}", e)))?;
        
        // Build metrics
        Ok(GpuMetrics {
            info: GpuInfo {
                index: gpu_id,
                uuid: uuid.to_string(),
                name: name.to_string(),
                architecture: device.architecture().ok().map(|a| format!("{:?}", a)),
                driver_version: self.nvml.sys_driver_version().ok(),
                cuda_version: self.nvml.sys_cuda_driver_version().ok().map(|v| format!("{}", v)),
                pci_info: device.pci_info().ok().map(|pci| PciInfo {
                    bus: pci.bus as u32,
                    device: pci.device as u32,
                    domain: pci.domain as u32,
                    device_id: pci.pci_device_id as u32,
                    subsystem_id: pci.pci_subsystem_id as u32,
                }),
            },
            status: GpuStatus::Active,
            memory: MemoryInfo {
                total_bytes: memory_info.total,
                used_bytes: memory_info.used,
                free_bytes: memory_info.free,
                utilization_percent: utilization.memory as f32,
            },
            temperature: TemperatureInfo {
                gpu_temp_celsius: temperature as f32,
                memory_temp_celsius: None, // May not be available on all GPUs
                hotspot_temp_celsius: None,
            },
            power: PowerInfo {
                usage_watts: power_usage as f32 / 1000.0, // Convert mW to W
                limit_watts: power_limit as f32 / 1000.0,
                utilization_percent: (power_usage as f32 / power_limit as f32) * 100.0,
            },
            clocks: ClockInfo {
                graphics_mhz: graphics_clock as u32,
                memory_mhz: memory_clock as u32,
                sm_mhz: None,
            },
            utilization: UtilizationInfo {
                gpu_percent: utilization.gpu as f32,
                memory_percent: utilization.memory as f32,
                encoder_percent: None,
                decoder_percent: None,
            },
            fans: Vec::new(), // Fan info may not be available via NVML
            mig: None, // MIG info requires additional NVML calls
            ecc: None, // ECC info requires additional NVML calls
            performance_state: device.performance_state().ok().map(|ps| PerformanceState {
                current_state: format!("P{}", ps as u32),
                available_states: Vec::new(),
            }),
            processes: Vec::new(), // Process info requires additional NVML calls
            timestamp: SystemTime::now(),
            collection_duration: Duration::from_millis(0), // Set by caller
        })
    }
}
```

---

## 🔗 **Phase D: Integration - Technical Details**

### D.1 State Management Integration

#### Real Data Streaming
```rust
// File: crates/mesh-state/src/store.rs
impl StateStore {
    pub async fn handle_model_state_delta(&mut self, delta: ModelStateDelta) -> Result<()> {
        let key = self.model_state_key(&delta.labels);
        
        match self.model_states.get_mut(&key) {
            Some(existing_state) => {
                // Apply delta to existing state
                self.apply_model_delta(existing_state, &delta)?;
            }
            None => {
                // Create new state from delta
                let new_state = ModelState::from_delta(delta)?;
                self.model_states.insert(key, new_state);
            }
        }
        
        // Update derived metrics
        self.update_derived_metrics(&key).await?;
        
        // Notify subscribers
        self.notify_state_change(StateChangeEvent::ModelState { key }).await?;
        
        Ok(())
    }
    
    pub async fn handle_gpu_state_delta(&mut self, delta: GpuStateDelta) -> Result<()> {
        let key = format!("{}:{}", delta.gpu_uuid, delta.node_id);
        
        match self.gpu_states.get_mut(&key) {
            Some(existing_state) => {
                self.apply_gpu_delta(existing_state, &delta)?;
            }
            None => {
                let new_state = GpuState::from_delta(delta)?;
                self.gpu_states.insert(key.clone(), new_state);
            }
        }
        
        // Update scoring cache
        self.invalidate_scoring_cache(&key).await?;
        
        Ok(())
    }
}
```

#### Enhanced Scoring Algorithm
```rust
// File: crates/mesh-state/src/scoring.rs
impl ScoringEngine {
    pub async fn score_targets(
        &self,
        request: &ScoreTargetsRequest,
    ) -> Result<Vec<ScoredTarget>> {
        let model_filter = ModelFilter {
            name: request.model_name.clone(),
            version: request.model_version.clone(),
            ..Default::default()
        };
        
        // Get candidate nodes with the model
        let candidates = self.state_store.find_model_instances(&model_filter).await?;
        
        let mut scored_targets = Vec::new();
        
        for candidate in candidates {
            let score = self.calculate_composite_score(&candidate, request).await?;
            
            scored_targets.push(ScoredTarget {
                node_id: candidate.node_id.clone(),
                address: candidate.address.clone(),
                port: candidate.port,
                score,
                metadata: self.build_target_metadata(&candidate).await?,
            });
        }
        
        // Sort by score (higher is better)
        scored_targets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(scored_targets)
    }
    
    async fn calculate_composite_score(
        &self,
        candidate: &ModelInstance,
        request: &ScoreTargetsRequest,
    ) -> Result<f64> {
        let mut score = 0.0;
        let mut weight_sum = 0.0;
        
        // GPU utilization score (lower utilization = higher score)
        if let Some(gpu_state) = self.state_store.get_gpu_state(&candidate.gpu_uuid).await? {
            let gpu_score = 100.0 - gpu_state.utilization.gpu_percent as f64;
            score += gpu_score * self.config.gpu_utilization_weight;
            weight_sum += self.config.gpu_utilization_weight;
        }
        
        // Memory availability score
        if let Some(gpu_state) = self.state_store.get_gpu_state(&candidate.gpu_uuid).await? {
            let memory_available = gpu_state.memory.free_bytes as f64 / gpu_state.memory.total_bytes as f64;
            let memory_score = memory_available * 100.0;
            score += memory_score * self.config.memory_weight;
            weight_sum += self.config.memory_weight;
        }
        
        // Queue depth score (lower queue = higher score)
        if let Some(model_state) = self.state_store.get_model_state(&candidate.model_key).await? {
            let queue_score = match model_state.queue_depth {
                0 => 100.0,
                depth => (100.0 / (1.0 + depth as f64)).max(0.0),
            };
            score += queue_score * self.config.queue_weight;
            weight_sum += self.config.queue_weight;
        }
        
        // Network latency score (lower latency = higher score)
        if let Some(latency_ms) = self.get_network_latency(&candidate.node_id).await? {
            let latency_score = (1000.0 / (latency_ms + 1.0)).min(100.0);
            score += latency_score * self.config.network_weight;
            weight_sum += self.config.network_weight;
        }
        
        // Historical performance score
        if let Some(perf_score) = self.get_historical_performance(&candidate.node_id, &request.model_name).await? {
            score += perf_score * self.config.performance_weight;
            weight_sum += self.config.performance_weight;
        }
        
        // Normalize score
        if weight_sum > 0.0 {
            Ok(score / weight_sum)
        } else {
            Ok(50.0) // Default neutral score
        }
    }
}
```

---

This technical specification provides the detailed implementation guidance needed to transform infermesh from its current architectural state into a fully functional production system. Each section includes specific code examples, error handling patterns, and integration points between components.

The key insight is that while the architecture is excellent, the gap between interfaces and implementations requires systematic, methodical work to bridge. The compilation fixes are the critical first step, followed by implementing the core business logic in the service layers, and finally connecting everything with real data adapters.
