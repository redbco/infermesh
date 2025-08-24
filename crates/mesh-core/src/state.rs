//! State representations for models and GPUs in the mesh
//!
//! These structures represent the current state of inference runtimes and
//! GPU resources, used for routing decisions and observability.

use crate::Labels;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Current state of a model instance on a specific node/GPU
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelState {
    /// Labels identifying this model instance
    pub labels: Labels,
    
    /// Current queue depth (number of pending requests)
    pub queue_depth: u32,
    
    /// Service rate in requests per second or tokens per second
    pub service_rate: f64,
    
    /// 95th percentile latency in milliseconds
    pub p95_latency_ms: u32,
    
    /// Batch fullness ratio (0.0 to 1.0)
    pub batch_fullness: f32,
    
    /// Whether the model is currently loaded and ready
    pub loaded: bool,
    
    /// Whether the model is currently warming up
    pub warming: bool,
    
    /// Estimated work remaining in seconds based on current queue
    pub work_left_seconds: f32,
    
    /// Last time this state was updated
    pub last_updated: DateTime<Utc>,
}

impl ModelState {
    /// Create a new ModelState with default values
    pub fn new(labels: Labels) -> Self {
        Self {
            labels,
            queue_depth: 0,
            service_rate: 0.0,
            p95_latency_ms: 0,
            batch_fullness: 0.0,
            loaded: false,
            warming: false,
            work_left_seconds: 0.0,
            last_updated: Utc::now(),
        }
    }

    /// Update the state with new metrics
    pub fn update(
        &mut self,
        queue_depth: u32,
        service_rate: f64,
        p95_latency_ms: u32,
        batch_fullness: f32,
    ) {
        self.queue_depth = queue_depth;
        self.service_rate = service_rate;
        self.p95_latency_ms = p95_latency_ms;
        self.batch_fullness = batch_fullness.clamp(0.0, 1.0);
        self.work_left_seconds = if service_rate > 0.0 {
            queue_depth as f32 / service_rate as f32
        } else {
            f32::INFINITY
        };
        self.last_updated = Utc::now();
    }

    /// Mark the model as loaded and ready
    pub fn mark_loaded(&mut self) {
        self.loaded = true;
        self.warming = false;
        self.last_updated = Utc::now();
    }

    /// Mark the model as unloaded
    pub fn mark_unloaded(&mut self) {
        self.loaded = false;
        self.warming = false;
        self.queue_depth = 0;
        self.service_rate = 0.0;
        self.batch_fullness = 0.0;
        self.work_left_seconds = 0.0;
        self.last_updated = Utc::now();
    }

    /// Mark the model as warming up
    pub fn mark_warming(&mut self) {
        self.warming = true;
        self.loaded = false;
        self.last_updated = Utc::now();
    }

    /// Check if this state is stale (older than threshold)
    pub fn is_stale(&self, threshold_seconds: u64) -> bool {
        let threshold = chrono::Duration::seconds(threshold_seconds as i64);
        Utc::now() - self.last_updated > threshold
    }

    /// Calculate a simple load score (0.0 = no load, 1.0+ = overloaded)
    pub fn load_score(&self) -> f32 {
        if !self.loaded || self.warming {
            return f32::INFINITY;
        }

        // Combine queue depth and batch fullness
        let queue_factor = self.queue_depth as f32 / 10.0; // Normalize assuming 10 is "full"
        let batch_factor = self.batch_fullness;
        
        (queue_factor + batch_factor) / 2.0
    }
}

/// Current state of a GPU resource
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuState {
    /// GPU UUID (NVIDIA format)
    pub gpu_uuid: String,
    
    /// Node where this GPU is located
    pub node: String,
    
    /// MIG profile if applicable (e.g., "1g.5gb", "3g.20gb")
    pub mig_profile: Option<String>,
    
    /// SM (Streaming Multiprocessor) utilization (0.0 to 1.0)
    pub sm_utilization: f32,
    
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: f32,
    
    /// Used VRAM in GB
    pub vram_used_gb: f32,
    
    /// Total VRAM in GB
    pub vram_total_gb: f32,
    
    /// Temperature in Celsius
    pub temperature_c: Option<f32>,
    
    /// Power consumption in Watts
    pub power_watts: Option<f32>,
    
    /// Whether there are ECC errors
    pub ecc_errors: bool,
    
    /// Whether the GPU is being throttled
    pub throttled: bool,
    
    /// Last time this state was updated
    pub last_updated: DateTime<Utc>,
}

impl GpuState {
    /// Create a new GpuState
    pub fn new(gpu_uuid: impl Into<String>, node: impl Into<String>) -> Self {
        Self {
            gpu_uuid: gpu_uuid.into(),
            node: node.into(),
            mig_profile: None,
            sm_utilization: 0.0,
            memory_utilization: 0.0,
            vram_used_gb: 0.0,
            vram_total_gb: 0.0,
            temperature_c: None,
            power_watts: None,
            ecc_errors: false,
            throttled: false,
            last_updated: Utc::now(),
        }
    }

    /// Update GPU metrics
    pub fn update_metrics(
        &mut self,
        sm_utilization: f32,
        memory_utilization: f32,
        vram_used_gb: f32,
        vram_total_gb: f32,
    ) {
        self.sm_utilization = sm_utilization.clamp(0.0, 1.0);
        self.memory_utilization = memory_utilization.clamp(0.0, 1.0);
        self.vram_used_gb = vram_used_gb.max(0.0);
        self.vram_total_gb = vram_total_gb.max(0.0);
        self.last_updated = Utc::now();
    }

    /// Update thermal and power metrics
    pub fn update_thermal(&mut self, temperature_c: Option<f32>, power_watts: Option<f32>) {
        self.temperature_c = temperature_c;
        self.power_watts = power_watts;
        self.last_updated = Utc::now();
    }

    /// Update error and throttling status
    pub fn update_status(&mut self, ecc_errors: bool, throttled: bool) {
        self.ecc_errors = ecc_errors;
        self.throttled = throttled;
        self.last_updated = Utc::now();
    }

    /// Calculate available VRAM in GB
    pub fn vram_available_gb(&self) -> f32 {
        (self.vram_total_gb - self.vram_used_gb).max(0.0)
    }

    /// Calculate VRAM headroom ratio (0.0 = full, 1.0 = empty)
    pub fn vram_headroom(&self) -> f32 {
        if self.vram_total_gb > 0.0 {
            self.vram_available_gb() / self.vram_total_gb
        } else {
            0.0
        }
    }

    /// Check if this GPU is healthy (no errors, not throttled)
    pub fn is_healthy(&self) -> bool {
        !self.ecc_errors && !self.throttled
    }

    /// Check if this state is stale (older than threshold)
    pub fn is_stale(&self, threshold_seconds: u64) -> bool {
        let threshold = chrono::Duration::seconds(threshold_seconds as i64);
        Utc::now() - self.last_updated > threshold
    }

    /// Calculate overall GPU load score (0.0 = idle, 1.0 = fully loaded)
    pub fn load_score(&self) -> f32 {
        if !self.is_healthy() {
            return f32::INFINITY;
        }

        // Weight SM utilization more heavily than memory
        (self.sm_utilization * 0.7) + (self.memory_utilization * 0.3)
    }
}

/// Delta update for model state (used for efficient state propagation)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelStateDelta {
    /// Labels identifying the model instance
    pub labels: Labels,
    
    /// Updated fields (None means no change)
    pub queue_depth: Option<u32>,
    pub service_rate: Option<f64>,
    pub p95_latency_ms: Option<u32>,
    pub batch_fullness: Option<f32>,
    pub loaded: Option<bool>,
    pub warming: Option<bool>,
    
    /// Timestamp of this delta
    pub timestamp: DateTime<Utc>,
}

impl ModelStateDelta {
    /// Create a new delta with timestamp
    pub fn new(labels: Labels) -> Self {
        Self {
            labels,
            queue_depth: None,
            service_rate: None,
            p95_latency_ms: None,
            batch_fullness: None,
            loaded: None,
            warming: None,
            timestamp: Utc::now(),
        }
    }

    /// Apply this delta to a ModelState
    pub fn apply_to(&self, state: &mut ModelState) {
        if let Some(queue_depth) = self.queue_depth {
            state.queue_depth = queue_depth;
        }
        if let Some(service_rate) = self.service_rate {
            state.service_rate = service_rate;
        }
        if let Some(p95_latency_ms) = self.p95_latency_ms {
            state.p95_latency_ms = p95_latency_ms;
        }
        if let Some(batch_fullness) = self.batch_fullness {
            state.batch_fullness = batch_fullness.clamp(0.0, 1.0);
        }
        if let Some(loaded) = self.loaded {
            state.loaded = loaded;
        }
        if let Some(warming) = self.warming {
            state.warming = warming;
        }

        // Recalculate derived fields
        state.work_left_seconds = if state.service_rate > 0.0 {
            state.queue_depth as f32 / state.service_rate as f32
        } else {
            f32::INFINITY
        };
        state.last_updated = self.timestamp;
    }
}

/// Delta update for GPU state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuStateDelta {
    /// GPU UUID
    pub gpu_uuid: String,
    
    /// Node identifier
    pub node: String,
    
    /// Updated fields (None means no change)
    pub sm_utilization: Option<f32>,
    pub memory_utilization: Option<f32>,
    pub vram_used_gb: Option<f32>,
    pub vram_total_gb: Option<f32>,
    pub temperature_c: Option<f32>,
    pub power_watts: Option<f32>,
    pub ecc_errors: Option<bool>,
    pub throttled: Option<bool>,
    
    /// Timestamp of this delta
    pub timestamp: DateTime<Utc>,
}

impl GpuStateDelta {
    /// Create a new delta with timestamp
    pub fn new(gpu_uuid: impl Into<String>, node: impl Into<String>) -> Self {
        Self {
            gpu_uuid: gpu_uuid.into(),
            node: node.into(),
            sm_utilization: None,
            memory_utilization: None,
            vram_used_gb: None,
            vram_total_gb: None,
            temperature_c: None,
            power_watts: None,
            ecc_errors: None,
            throttled: None,
            timestamp: Utc::now(),
        }
    }

    /// Apply this delta to a GpuState
    pub fn apply_to(&self, state: &mut GpuState) {
        if let Some(sm_utilization) = self.sm_utilization {
            state.sm_utilization = sm_utilization.clamp(0.0, 1.0);
        }
        if let Some(memory_utilization) = self.memory_utilization {
            state.memory_utilization = memory_utilization.clamp(0.0, 1.0);
        }
        if let Some(vram_used_gb) = self.vram_used_gb {
            state.vram_used_gb = vram_used_gb.max(0.0);
        }
        if let Some(vram_total_gb) = self.vram_total_gb {
            state.vram_total_gb = vram_total_gb.max(0.0);
        }
        if let Some(temperature_c) = self.temperature_c {
            state.temperature_c = Some(temperature_c);
        }
        if let Some(power_watts) = self.power_watts {
            state.power_watts = Some(power_watts);
        }
        if let Some(ecc_errors) = self.ecc_errors {
            state.ecc_errors = ecc_errors;
        }
        if let Some(throttled) = self.throttled {
            state.throttled = throttled;
        }

        state.last_updated = self.timestamp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Labels;

    #[test]
    fn test_model_state_creation() {
        let labels = Labels::new("gpt-4", "v1.0", "triton", "node1");
        let state = ModelState::new(labels.clone());
        
        assert_eq!(state.labels, labels);
        assert_eq!(state.queue_depth, 0);
        assert!(!state.loaded);
        assert!(!state.warming);
    }

    #[test]
    fn test_model_state_update() {
        let labels = Labels::new("gpt-4", "v1.0", "triton", "node1");
        let mut state = ModelState::new(labels);
        
        state.update(5, 10.0, 100, 0.8);
        
        assert_eq!(state.queue_depth, 5);
        assert_eq!(state.service_rate, 10.0);
        assert_eq!(state.p95_latency_ms, 100);
        assert_eq!(state.batch_fullness, 0.8);
        assert_eq!(state.work_left_seconds, 0.5); // 5 / 10.0
    }

    #[test]
    fn test_model_state_load_score() {
        let labels = Labels::new("gpt-4", "v1.0", "triton", "node1");
        let mut state = ModelState::new(labels);
        
        // Unloaded model should have infinite load
        assert_eq!(state.load_score(), f32::INFINITY);
        
        // Loaded model with no queue
        state.mark_loaded();
        assert_eq!(state.load_score(), 0.0);
        
        // Loaded model with some load
        state.update(5, 10.0, 100, 0.6);
        let score = state.load_score();
        assert!(score > 0.0 && score < 1.0);
    }

    #[test]
    fn test_gpu_state_creation() {
        let state = GpuState::new("GPU-12345", "node1");
        
        assert_eq!(state.gpu_uuid, "GPU-12345");
        assert_eq!(state.node, "node1");
        assert_eq!(state.sm_utilization, 0.0);
        assert!(state.is_healthy());
    }

    #[test]
    fn test_gpu_state_vram_calculations() {
        let mut state = GpuState::new("GPU-12345", "node1");
        state.update_metrics(0.5, 0.6, 8.0, 16.0);
        
        assert_eq!(state.vram_available_gb(), 8.0);
        assert_eq!(state.vram_headroom(), 0.5);
    }

    #[test]
    fn test_gpu_state_health() {
        let mut state = GpuState::new("GPU-12345", "node1");
        
        assert!(state.is_healthy());
        
        state.update_status(true, false); // ECC errors
        assert!(!state.is_healthy());
        
        state.update_status(false, true); // Throttled
        assert!(!state.is_healthy());
    }

    #[test]
    fn test_model_state_delta() {
        let labels = Labels::new("gpt-4", "v1.0", "triton", "node1");
        let mut state = ModelState::new(labels.clone());
        
        let mut delta = ModelStateDelta::new(labels);
        delta.queue_depth = Some(10);
        delta.loaded = Some(true);
        
        delta.apply_to(&mut state);
        
        assert_eq!(state.queue_depth, 10);
        assert!(state.loaded);
    }

    #[test]
    fn test_gpu_state_delta() {
        let mut state = GpuState::new("GPU-12345", "node1");
        
        let mut delta = GpuStateDelta::new("GPU-12345", "node1");
        delta.sm_utilization = Some(0.8);
        delta.ecc_errors = Some(true);
        
        delta.apply_to(&mut state);
        
        assert_eq!(state.sm_utilization, 0.8);
        assert!(state.ecc_errors);
    }
}
