use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use rand::prelude::*;
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};

/// Unique identifier for requests
pub type RequestId = u64;

/// Unique identifier for nodes
pub type NodeId = u32;

/// Unique identifier for models
pub type ModelId = String;

/// Target for routing decisions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Target {
    pub node_id: NodeId,
    pub model_id: ModelId,
}

/// Events that drive the discrete-event simulation
#[derive(Debug, Clone)]
pub enum EventKind {
    /// A new request arrives in the system
    Arrival(Request),
    /// A request is dispatched to a target node
    Dispatch(RequestId, Target),
    /// A batch window closes and processing begins
    BatchClose(NodeId, ModelId),
    /// Service for a request completes
    ServiceDone(RequestId, NodeId),
    /// A hedge request fires
    HedgeFire(RequestId),
    /// Cancel a request at a target
    Cancel(RequestId, Target),
    /// Update signals for a node
    SignalUpdate(NodeId),
}

/// A timestamped event in the simulation
#[derive(Debug, Clone)]
pub struct SimEvent {
    pub at: f64,  // timestamp in milliseconds
    pub kind: EventKind,
}

impl PartialEq for SimEvent {
    fn eq(&self, other: &Self) -> bool {
        self.at == other.at
    }
}

impl Eq for SimEvent {}

impl PartialOrd for SimEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SimEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other.at.partial_cmp(&self.at).unwrap_or(Ordering::Equal)
    }
}

/// Request types supported by the simulator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RequestType {
    LLM,
    Vision,
    ASR,
}

/// A request in the system
#[derive(Debug, Clone)]
pub struct Request {
    pub id: RequestId,
    pub request_type: RequestType,
    pub tenant_id: String,
    pub model_id: ModelId,
    pub arrival_time: f64,
    pub input_tokens: u32,
    pub expected_output_tokens: u32,
    pub sla_ms: Option<f64>,
}

/// Context provided to routing strategies
pub struct RequestCtx {
    pub request: Request,
    pub current_time: f64,
}

/// Node information for routing decisions
#[derive(Debug, Clone)]
pub struct RoutingNodeInfo {
    pub node_id: NodeId,
    pub cell_id: u32,
    pub gpu_profile: crate::gpu::GpuProfile,
    pub queue_depths: HashMap<ModelId, u32>,
    pub utilization: f64,
    pub vram_usage_gb: f64,
    pub last_updated: f64,
}

/// View of the system state for routing decisions
pub struct StateView {
    pub nodes: HashMap<NodeId, RoutingNodeInfo>,
    pub current_time: f64,
}

/// Trait for routing strategies (thread-safe for parallel processing)
pub trait RouterStrategy: Send + Sync {
    fn choose(&mut self, ctx: &RequestCtx, view: &StateView) -> Target;
    fn decision_cost_us(&self) -> u64 {
        50 // Default decision cost in microseconds
    }
}

/// The main discrete-event simulator
pub struct Sim {
    /// Current simulation time in milliseconds
    pub now_ms: f64,
    /// Event queue ordered by time
    pub events: BinaryHeap<SimEvent>,
    /// Random number generator with fixed seed for reproducibility
    rng: SmallRng,
    /// Next request ID to assign
    next_request_id: RequestId,
    // TODO: Add world field when implemented - World state (topology, nodes, etc.)
    // TODO: Add metrics field when implemented - Metrics collection
    // TODO: Add router field when implemented - Router strategy
}

impl Sim {
    /// Create a new simulator with the given seed
    pub fn new(seed: u64) -> Self {
        Self {
            now_ms: 0.0,
            events: BinaryHeap::new(),
            rng: SmallRng::seed_from_u64(seed),
            next_request_id: 1,
        }
    }

    /// Schedule an event at the given time
    pub fn schedule(&mut self, kind: EventKind, at: f64) {
        self.events.push(SimEvent { at, kind });
    }

    /// Schedule an event after a delay from current time
    pub fn schedule_after(&mut self, kind: EventKind, delay_ms: f64) {
        self.schedule(kind, self.now_ms + delay_ms);
    }

    /// Generate a new unique request ID
    pub fn next_request_id(&mut self) -> RequestId {
        let id = self.next_request_id;
        self.next_request_id += 1;
        id
    }

    /// Get a mutable reference to the RNG
    pub fn rng(&mut self) -> &mut SmallRng {
        &mut self.rng
    }

    /// Run the simulation until the given time
    pub fn run(&mut self, until_ms: f64) -> anyhow::Result<()> {
        while let Some(event) = self.events.pop() {
            if event.at > until_ms {
                // Put the event back and stop
                self.events.push(event);
                break;
            }
            
            self.now_ms = event.at;
            self.handle_event(event.kind)?;
        }
        Ok(())
    }

    /// Handle a single event
    fn handle_event(&mut self, event: EventKind) -> anyhow::Result<()> {
        match event {
            EventKind::Arrival(request) => {
                self.handle_arrival(request)?;
            }
            EventKind::Dispatch(request_id, target) => {
                self.handle_dispatch(request_id, target)?;
            }
            EventKind::BatchClose(node_id, model_id) => {
                self.handle_batch_close(node_id, model_id)?;
            }
            EventKind::ServiceDone(request_id, node_id) => {
                self.handle_service_done(request_id, node_id)?;
            }
            EventKind::HedgeFire(request_id) => {
                self.handle_hedge_fire(request_id)?;
            }
            EventKind::Cancel(request_id, target) => {
                self.handle_cancel(request_id, target)?;
            }
            EventKind::SignalUpdate(node_id) => {
                self.handle_signal_update(node_id)?;
            }
        }
        Ok(())
    }

    fn handle_arrival(&mut self, request: Request) -> anyhow::Result<()> {
        // TODO: Implement arrival handling
        // 1. Use router strategy to choose target
        // 2. Account for decision cost
        // 3. Schedule dispatch event
        tracing::debug!("Handling arrival for request {}", request.id);
        Ok(())
    }

    fn handle_dispatch(&mut self, request_id: RequestId, target: Target) -> anyhow::Result<()> {
        // TODO: Implement dispatch handling
        // 1. Add request to target node's queue
        // 2. If batch window not open, open it and schedule batch close
        tracing::debug!("Handling dispatch for request {} to {:?}", request_id, target);
        Ok(())
    }

    fn handle_batch_close(&mut self, node_id: NodeId, model_id: ModelId) -> anyhow::Result<()> {
        // TODO: Implement batch close handling
        // 1. Collect requests from queue up to max batch size
        // 2. Calculate service time based on batch and model
        // 3. Schedule service done events
        tracing::debug!("Handling batch close for node {} model {}", node_id, model_id);
        Ok(())
    }

    fn handle_service_done(&mut self, request_id: RequestId, node_id: NodeId) -> anyhow::Result<()> {
        // TODO: Implement service done handling
        // 1. Remove request from active processing
        // 2. Record metrics (latency, etc.)
        // 3. Update node state (VRAM, utilization)
        tracing::debug!("Handling service done for request {} at node {}", request_id, node_id);
        Ok(())
    }

    fn handle_hedge_fire(&mut self, request_id: RequestId) -> anyhow::Result<()> {
        // TODO: Implement hedge fire handling
        // 1. Check if original request is still pending
        // 2. If so, choose alternative target and dispatch
        tracing::debug!("Handling hedge fire for request {}", request_id);
        Ok(())
    }

    fn handle_cancel(&mut self, request_id: RequestId, target: Target) -> anyhow::Result<()> {
        // TODO: Implement cancel handling
        // 1. Remove request from target's queue or processing
        // 2. Update metrics (wasted work)
        tracing::debug!("Handling cancel for request {} at {:?}", request_id, target);
        Ok(())
    }

    fn handle_signal_update(&mut self, node_id: NodeId) -> anyhow::Result<()> {
        // TODO: Implement signal update handling
        // 1. Update node metrics (queue depth, VRAM, p95, etc.)
        // 2. Schedule next signal update with jitter
        tracing::debug!("Handling signal update for node {}", node_id);
        Ok(())
    }

    /// Get the current simulation time
    pub fn current_time(&self) -> f64 {
        self.now_ms
    }

    /// Check if there are more events to process
    pub fn has_events(&self) -> bool {
        !self.events.is_empty()
    }

    /// Get the time of the next event (if any)
    pub fn next_event_time(&self) -> Option<f64> {
        self.events.peek().map(|e| e.at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sim_creation() {
        let sim = Sim::new(42);
        assert_eq!(sim.current_time(), 0.0);
        assert!(!sim.has_events());
    }

    #[test]
    fn test_event_scheduling() {
        let mut sim = Sim::new(42);
        
        // Schedule some events
        sim.schedule(EventKind::SignalUpdate(1), 100.0);
        sim.schedule(EventKind::SignalUpdate(2), 50.0);
        sim.schedule(EventKind::SignalUpdate(3), 150.0);
        
        assert!(sim.has_events());
        assert_eq!(sim.next_event_time(), Some(50.0));
    }

    #[test]
    fn test_request_id_generation() {
        let mut sim = Sim::new(42);
        assert_eq!(sim.next_request_id(), 1);
        assert_eq!(sim.next_request_id(), 2);
        assert_eq!(sim.next_request_id(), 3);
    }
}
