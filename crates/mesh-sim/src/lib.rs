//! # Discrete-Event Simulator for InferMesh
//!
//! A discrete-event simulator for InferMesh that lets you vary scale (nodes/GPUs), 
//! request mixes, and mesh algorithms, then measure the impact on p50/p95/p99 latency, 
//! GPU utilization, and cost.
//!
//! ## Features
//!
//! - Compare routing strategies: baseline, heuristic, mesh, mesh+hedging
//! - Scale from hundreds to 1M nodes via cells/shards
//! - Vary workloads: token length distributions, image/ASR mix, burstiness, tenant skew
//! - Account for decision cost: per-request routing compute + signal fusion delay
//! - Comprehensive metrics collection and analysis

pub mod engine;
pub mod world;
pub mod gpu;
pub mod runtime;
pub mod workload;
pub mod router;
pub mod signals;
pub mod net;
pub mod metrics;

pub use engine::{Sim, SimEvent, EventKind, Request, RequestType, RequestId, NodeId, ModelId, Target, RouterStrategy, RequestCtx, StateView, RoutingNodeInfo};
pub use world::{World, TopologyConfig, Cell, Node, NodeState};
pub use gpu::{GpuProfile, MigProfile, ServiceModel};
pub use runtime::{RuntimeState, Batch, RequestQueue};
pub use workload::{WorkloadGenerator, WorkloadConfig};
pub use router::{RouterFactory, RouterConfig, StrategyType};
pub use signals::{SignalView, SignalConfig, SignalGenerator};
pub use net::{NetworkConfig, NetworkTopology};
pub use metrics::{Metrics, MetricsSummary};

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use anyhow::Result;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;

/// Complete simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Random seed for reproducibility
    pub seed: u64,
    /// Simulation duration in seconds
    pub duration_s: f64,
    /// Workload configuration
    pub workload: WorkloadConfig,
    /// Topology configuration
    pub topology: TopologyConfig,
    /// Network configuration
    pub network: NetworkConfig,
    /// Signal configuration
    pub signals: SignalConfig,
    /// Routing strategies to test
    pub strategies: Vec<String>,
}

impl SimulationConfig {
    /// Load configuration from YAML file
    pub fn from_yaml_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to YAML file
    pub fn to_yaml_file(&self, path: &str) -> Result<()> {
        let content = serde_yaml::to_string(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Validate workload
        if let Err(e) = self.workload.mix.validate() {
            return Err(anyhow::anyhow!("Invalid workload mix: {}", e));
        }

        // Validate topology
        if self.topology.cells == 0 {
            return Err(anyhow::anyhow!("Must have at least 1 cell"));
        }
        if self.topology.nodes_per_cell == 0 {
            return Err(anyhow::anyhow!("Must have at least 1 node per cell"));
        }
        if self.topology.gpu_profiles.is_empty() {
            return Err(anyhow::anyhow!("Must have at least 1 GPU profile"));
        }

        // Validate duration
        if self.duration_s <= 0.0 {
            return Err(anyhow::anyhow!("Duration must be positive"));
        }

        // Validate strategies
        if self.strategies.is_empty() {
            return Err(anyhow::anyhow!("Must specify at least one routing strategy"));
        }

        Ok(())
    }
}

/// Integrated simulator that combines all components
struct IntegratedSim {
    #[allow(dead_code)]
    config: SimulationConfig,
    sim: Sim,
    world: World,
    #[allow(dead_code)]
    network_topology: NetworkTopology,
    workload_generator: WorkloadGenerator,
    #[allow(dead_code)]
    signal_generator: SignalGenerator,
    router_strategy: Box<dyn RouterStrategy>,
    metrics: Metrics,
    node_runtimes: HashMap<NodeId, RuntimeState>,
    request_tracker: HashMap<RequestId, Request>,
    rng: SmallRng,
}

impl IntegratedSim {
    fn new(
        config: SimulationConfig,
        world: World,
        network_topology: NetworkTopology,
        workload_generator: WorkloadGenerator,
        signal_generator: SignalGenerator,
        router_strategy: Box<dyn RouterStrategy>,
    ) -> Self {
        let sim = Sim::new(config.seed);
        let metrics = Metrics::new();
        let mut node_runtimes = HashMap::new();
        
        // Initialize runtime state for each node
        for (node_id, node) in &world.nodes {
            let runtime = RuntimeState::new_for_gpu(*node_id, node.gpu_profile.clone());
            node_runtimes.insert(*node_id, runtime);
        }
        
        Self {
            config: config.clone(),
            sim,
            world,
            network_topology,
            workload_generator,
            signal_generator,
            router_strategy,
            metrics,
            node_runtimes,
            request_tracker: HashMap::new(),
            rng: SmallRng::seed_from_u64(config.seed),
        }
    }
    
    fn run(&mut self, duration_ms: f64, strategy_name: &str) -> Result<MetricsSummary> {
        let node_count = self.world.nodes.len();
        tracing::info!(
            "Starting simulation [{}]: {} nodes, {:.1}s duration",
            strategy_name,
            node_count,
            duration_ms / 1000.0
        );
        
        // Schedule initial request
        let first_arrival = self.workload_generator.next_arrival_time(0.0);
        let first_request = self.workload_generator.generate_request(first_arrival);
        self.metrics.record_arrival(first_request.request_type);
        self.sim.schedule(EventKind::Arrival(first_request), first_arrival);
        
        // Schedule initial signal updates for all nodes (less frequent)
        for (i, node_id) in self.world.nodes.keys().enumerate() {
            // Stagger signal updates to avoid thundering herd
            let signal_time = (i as f64 * 100.0) + 1000.0; // Start after 1s, stagger by 100ms
            self.sim.schedule(EventKind::SignalUpdate(*node_id), signal_time);
        }
        
        // Main simulation loop with progress tracking
        let mut events_processed = 0;
        let mut last_progress_time = 0.0;
        
        // Adaptive progress interval based on topology size
        let progress_interval_ms = if node_count > 50000 {
            2000.0  // 2 seconds for very large topologies
        } else if node_count > 10000 {
            5000.0  // 5 seconds for large topologies  
        } else {
            10000.0 // 10 seconds for normal topologies
        };
        
        while let Some(event) = self.sim.events.pop() {
            if event.at > duration_ms {
                break;
            }
            
            self.sim.now_ms = event.at;
            self.handle_event(event.kind)?;
            
            events_processed += 1;
            
            // Adaptive progress logging based on topology size
            if self.sim.now_ms - last_progress_time > progress_interval_ms {
                tracing::info!(
                    "Simulation progress [{}]: {:.1}s/{:.1}s ({:.1}%), {} events processed, {} requests tracked",
                    strategy_name,
                    self.sim.now_ms / 1000.0,
                    duration_ms / 1000.0,
                    (self.sim.now_ms / duration_ms) * 100.0,
                    events_processed,
                    self.request_tracker.len()
                );
                last_progress_time = self.sim.now_ms;
            }
            
            // Additional progress updates for very large simulations based on event count
            if node_count > 50000 && events_processed % 100000 == 0 {
                tracing::info!(
                    "Processing events [{}]: {} events completed, simulation time: {:.1}s",
                    strategy_name,
                    events_processed,
                    self.sim.now_ms / 1000.0
                );
            }
            
            // Safety limit to prevent infinite loops
            if events_processed > 10_000_000 {
                tracing::warn!("Event limit reached (10M events), terminating simulation");
                break;
            }
        }
        
        Ok(self.metrics.generate_summary())
    }
    
    fn handle_event(&mut self, event: EventKind) -> Result<()> {
        match event {
            EventKind::Arrival(request) => self.handle_arrival(request),
            EventKind::Dispatch(request_id, target) => self.handle_dispatch(request_id, target),
            EventKind::BatchClose(node_id, model_id) => self.handle_batch_close(node_id, model_id),
            EventKind::ServiceDone(request_id, node_id) => self.handle_service_done(request_id, node_id),
            EventKind::HedgeFire(request_id) => self.handle_hedge_fire(request_id),
            EventKind::Cancel(request_id, _target) => self.handle_cancel(request_id),
            EventKind::SignalUpdate(node_id) => self.handle_signal_update(node_id),
        }
    }
    
    fn handle_arrival(&mut self, request: Request) -> Result<()> {
        // Generate next request to maintain arrival process
        if self.workload_generator.should_continue(self.sim.now_ms) {
            let next_arrival = self.workload_generator.next_arrival_time(self.sim.now_ms);
            let next_request = self.workload_generator.generate_request(next_arrival);
            self.metrics.record_arrival(next_request.request_type);
            self.sim.schedule(EventKind::Arrival(next_request), next_arrival);
        }
        
        // Track the request
        self.request_tracker.insert(request.id, request.clone());
        
        // Route the current request
        let ctx = RequestCtx {
            request: request.clone(),
            current_time: self.sim.now_ms,
        };
        let view = self.build_state_view();
        let target = self.router_strategy.choose(&ctx, &view);
        
        // Account for decision cost
        let decision_delay = self.router_strategy.decision_cost_us() as f64 / 1000.0; // Convert to ms
        let dispatch_time = self.sim.now_ms + decision_delay;
        
        self.sim.schedule(EventKind::Dispatch(request.id, target), dispatch_time);
        
        // Schedule hedge fire only for mesh_hedge strategy and requests with SLA
        if request.sla_ms.is_some() && self.rng.gen::<f64>() < 0.1 { // Only 10% of requests get hedging
            let hedge_fire_time = self.sim.now_ms + (request.sla_ms.unwrap() * 0.7); // Fire at 70% of SLA
            self.sim.schedule(EventKind::HedgeFire(request.id), hedge_fire_time);
        }
        Ok(())
    }
    
    fn handle_dispatch(&mut self, request_id: RequestId, target: Target) -> Result<()> {
        // Get the original request
        if let Some(request) = self.request_tracker.get(&request_id).cloned() {
            // Add to node's queue
            if let Some(runtime) = self.node_runtimes.get_mut(&target.node_id) {
                if runtime.enqueue_request(request, self.sim.now_ms) {
                    // Check if we should close a batch
                    if runtime.should_close_batch(&target.model_id, self.sim.now_ms) {
                        self.sim.schedule(
                            EventKind::BatchClose(target.node_id, target.model_id),
                            self.sim.now_ms
                        );
                    }
                }
            }
        }
        Ok(())
    }
    
    fn handle_batch_close(&mut self, node_id: NodeId, model_id: ModelId) -> Result<()> {
        if let Some(runtime) = self.node_runtimes.get_mut(&node_id) {
            if let Some(batch) = runtime.close_batch(&model_id, self.sim.now_ms) {
                // Schedule service completion
                for request_id in batch.requests.iter().map(|r| r.id) {
                    self.sim.schedule(
                        EventKind::ServiceDone(request_id, node_id),
                        batch.estimated_completion_time
                    );
                }
            }
        }
        Ok(())
    }
    
    fn handle_service_done(&mut self, request_id: RequestId, node_id: NodeId) -> Result<()> {
        // Get the original request to calculate proper latency
        if let Some(request) = self.request_tracker.remove(&request_id) {
            let latency = self.sim.now_ms - request.arrival_time;
            self.metrics.record_latency(request.request_type, latency);
            
            // Check if this was a hedge win (completed before SLA deadline)
            if let Some(sla_ms) = request.sla_ms {
                if latency < sla_ms {
                    self.metrics.hedge_metrics.hedge_wins += 1;
                }
            }
            
            // Record throughput
            let tokens_processed = request.input_tokens + request.expected_output_tokens;
            self.metrics.record_throughput(tokens_processed as u64, 1, 1.0);
            
            // Calculate cost based on GPU usage
            if let Some(node) = self.world.nodes.get(&node_id) {
                // Estimate GPU time used for this request (simplified)
                let service_time_s = latency / 1000.0; // Convert ms to seconds
                let gpu_hours = service_time_s / 3600.0; // Convert to hours
                let cost_per_gpu_hour = match node.gpu_profile.name.as_str() {
                    "H100-80G" => 4.0,  // $4/hour for H100
                    "A100-80G" => 3.0,  // $3/hour for A100
                    "V100-32G" => 2.0,  // $2/hour for V100
                    _ => 2.5,           // Default cost
                };
                
                self.metrics.update_cost_metrics(gpu_hours, cost_per_gpu_hour, tokens_processed as u64);
            }
            
            // Cancel any pending hedge events for this request
            let dummy_target = Target { node_id: 0, model_id: "dummy".to_string() };
            self.sim.schedule(EventKind::Cancel(request_id, dummy_target), self.sim.now_ms);
        }
        
        // Update node utilization
        if let Some(runtime) = self.node_runtimes.get(&node_id) {
            self.metrics.update_node_utilization(
                node_id,
                runtime.get_utilization(),
                runtime.get_vram_utilization(),
                self.sim.now_ms
            );
        }
        
        Ok(())
    }
    
    fn handle_signal_update(&mut self, node_id: NodeId) -> Result<()> {
        // Schedule next signal update (much less frequent - every 5 seconds)
        let next_update = self.sim.now_ms + 5000.0; // 5 second intervals
        self.sim.schedule(EventKind::SignalUpdate(node_id), next_update);
        Ok(())
    }
    
    fn handle_hedge_fire(&mut self, request_id: RequestId) -> Result<()> {
        // Fire a hedge request - dispatch to alternative nodes
        if let Some(request) = self.request_tracker.get(&request_id).cloned() {
            // Find alternative nodes for hedging
            let view = self.build_state_view();
            let _ctx = RequestCtx {
                request: request.clone(),
                current_time: self.sim.now_ms,
            };
            
            // Get second-best target for hedge
            let mut targets: Vec<_> = view.nodes.keys().copied().collect();
            targets.sort_by_key(|&node_id| {
                if let Some(node_state) = view.nodes.get(&node_id) {
                    (node_state.queue_depths.values().sum::<u32>() as f64 * 1000.0) as i32
                } else {
                    i32::MAX
                }
            });
            
            if targets.len() > 1 {
                let hedge_target = Target {
                    node_id: targets[1], // Second best node
                    model_id: request.model_id.clone(),
                };
                
                self.sim.schedule(EventKind::Dispatch(request_id, hedge_target), self.sim.now_ms);
                self.metrics.hedge_metrics.total_hedges += 1;
            }
        }
        Ok(())
    }
    
    fn handle_cancel(&mut self, request_id: RequestId) -> Result<()> {
        // Cancel a request (remove from tracking)
        if self.request_tracker.remove(&request_id).is_some() {
            self.metrics.hedge_metrics.cancellations += 1;
        }
        Ok(())
    }
    
    fn take_metrics(self) -> Metrics {
        self.metrics
    }
    
    fn build_state_view(&self) -> StateView {
        // Build a real state view with current node information using parallel processing
        let node_infos: HashMap<NodeId, RoutingNodeInfo> = self.node_runtimes
            .par_iter()  // Parallel iterator over node runtimes
            .filter_map(|(node_id, runtime)| {
                self.world.nodes.get(node_id).map(|node| {
                    let utilization = runtime.get_utilization();
                    let vram_usage = runtime.get_vram_utilization();
                    
                    (*node_id, RoutingNodeInfo {
                        node_id: *node_id,
                        cell_id: node.cell_id,
                        gpu_profile: node.gpu_profile.clone(),
                        queue_depths: runtime.queues.iter().map(|(model, queue)| (model.clone(), queue.requests.len() as u32)).collect(),
                        utilization,
                        vram_usage_gb: vram_usage * node.gpu_profile.vram_total_gb,
                        last_updated: self.sim.now_ms,
                    })
                })
            })
            .collect();  // Collect results into HashMap
        
        StateView {
            nodes: node_infos,
            current_time: self.sim.now_ms,
        }
    }
}

/// Simulation runner that orchestrates the entire simulation
pub struct SimulationRunner {
    config: SimulationConfig,
    world: World,
    network_topology: NetworkTopology,
    workload_generator: WorkloadGenerator,
    signal_generator: SignalGenerator,
    metrics: Metrics,
}

impl SimulationRunner {
    /// Create a new simulation runner
    pub fn new(config: SimulationConfig) -> Result<Self> {
        config.validate()?;

        // Create world
        let world = World::new(config.topology.clone(), config.network.clone());

        // Create network topology
        let mut network_topology = NetworkTopology::new(config.network.clone());
        for cell_id in 0..config.topology.cells {
            network_topology.add_cell(cell_id);
        }

        // Create workload generator
        let workload_generator = match WorkloadGenerator::new(config.workload.clone(), config.seed) {
            Ok(gen) => gen,
            Err(e) => return Err(anyhow::anyhow!("Failed to create workload generator: {}", e)),
        };

        // Create signal generator
        let signal_generator = SignalGenerator::new(config.signals.clone());

        // Create metrics collector
        let metrics = Metrics::new();

        Ok(Self {
            config,
            world,
            network_topology,
            workload_generator,
            signal_generator,
            metrics,
        })
    }

    /// Run the simulation with all configured strategies
    pub fn run_all_strategies(&mut self) -> Result<HashMap<String, MetricsSummary>> {
        let strategies = self.config.strategies.clone();
        
        // For small numbers of strategies, use parallel execution
        if strategies.len() <= 8 {
            tracing::info!("Running {} strategies in parallel", strategies.len());
            
            // Clone the necessary data for parallel execution
            let config = self.config.clone();
            let world = self.world.clone();
            let workload_generator = self.workload_generator.clone();
            let signal_generator = self.signal_generator.clone();
            let network_topology = self.network_topology.clone();
            
            let results: Result<HashMap<String, MetricsSummary>> = strategies
                .par_iter()  // Parallel iterator over strategies
                .map(|strategy_name| {
                    tracing::info!("Running simulation with strategy: {}", strategy_name);
                    
                    // Create a new simulation runner for this strategy
                    let mut runner = SimulationRunner {
                        config: config.clone(),
                        world: world.clone(),
                        workload_generator: workload_generator.clone(),
                        signal_generator: signal_generator.clone(),
                        network_topology: network_topology.clone(),
                        metrics: Metrics::new(),
                    };
                    
                    let summary = runner.run_single_strategy(strategy_name)?;
                    Ok((strategy_name.clone(), summary))
                })
                .collect::<Result<Vec<_>>>()  // Collect results as Vec
                .map(|vec| vec.into_iter().collect()); // Convert Vec to HashMap
            
            results
        } else {
            // For large numbers of strategies, use sequential execution to avoid resource exhaustion
            tracing::info!("Running {} strategies sequentially (too many for parallel execution)", strategies.len());
            
            let mut results = HashMap::new();
            for strategy_name in &strategies {
                tracing::info!("Running simulation with strategy: {}", strategy_name);
                
                let summary = self.run_single_strategy(strategy_name)?;
                results.insert(strategy_name.clone(), summary);
            }
            Ok(results)
        }
    }

    /// Run the simulation with a single strategy
    pub fn run_single_strategy(&mut self, strategy_name: &str) -> Result<MetricsSummary> {
        // Reset metrics for this run
        self.metrics = Metrics::new();

        // Create router strategy
        let router_config = self.create_router_config(strategy_name)?;
        let router_strategy = RouterFactory::create_strategy(
            router_config,
            self.config.seed,
            Some(self.network_topology.clone()),
        );

        // Create simulator with integrated components
        let mut sim = IntegratedSim::new(
            self.config.clone(),
            self.world.clone(),
            self.network_topology.clone(),
            self.workload_generator.clone(),
            self.signal_generator.clone(),
            router_strategy,
        );

        // Run the integrated simulation
        let duration_ms = self.config.duration_s * 1000.0;
        let summary = sim.run(duration_ms, strategy_name)?;

        // Update our metrics with the results
        self.metrics = sim.take_metrics();

        Ok(summary)
    }

    /// Create router configuration from strategy name
    fn create_router_config(&self, strategy_name: &str) -> Result<RouterConfig> {
        let strategy_type = match strategy_name {
            "baseline_rr" => StrategyType::BaselineRoundRobin,
            "least_queue" => StrategyType::LeastQueue,
            "heuristic" => StrategyType::Heuristic {
                alpha: 0.4,
                beta: 0.3,
                gamma: 0.3,
            },
            "mesh" => StrategyType::Mesh,
            "mesh_hedge" => StrategyType::MeshHedge,
            "mesh_stale" => StrategyType::MeshStale {
                staleness_threshold_ms: 200.0,
            },
            "adaptive_mesh" => StrategyType::AdaptiveMesh {
                load_threshold: 0.7,
            },
            "predictive_mesh" => StrategyType::PredictiveMesh {
                prediction_window_ms: 5000.0,
            },
            "hybrid_mesh" => StrategyType::HybridMesh,
            "ml_enhanced_mesh" => StrategyType::MlEnhancedMesh,
            _ => return Err(anyhow::anyhow!("Unknown strategy: {}", strategy_name)),
        };

        Ok(RouterConfig {
            strategy: strategy_type,
            decision_cost_us: Some(50),
            hedge_config: None,
        })
    }

    /// Get the world reference
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Get the network topology reference
    pub fn network_topology(&self) -> &NetworkTopology {
        &self.network_topology
    }

    /// Get the current metrics
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }
}

/// Utility functions for simulation analysis
pub mod analysis {
    use super::*;
    use std::collections::HashMap;

    /// Compare multiple simulation results
    pub fn compare_strategies(results: &HashMap<String, MetricsSummary>) -> StrategyComparison {
        let mut comparison = StrategyComparison::default();

        for (strategy, summary) in results {
            // Extract key metrics for comparison
            if let Some(llm_latency) = summary.latency_by_type.get(&RequestType::LLM) {
                comparison.p95_latency.insert(strategy.clone(), llm_latency.p95);
                comparison.p99_latency.insert(strategy.clone(), llm_latency.p99);
            }

            comparison.utilization.insert(strategy.clone(), summary.average_utilization);
            comparison.cost_per_1k_tokens.insert(strategy.clone(), summary.cost_metrics.cost_per_1k_tokens());
            comparison.hedge_win_rate.insert(strategy.clone(), summary.hedge_metrics.hedge_win_rate());
        }

        comparison
    }

    /// Strategy comparison results
    #[derive(Debug, Default, Serialize, Deserialize)]
    pub struct StrategyComparison {
        pub p95_latency: HashMap<String, f64>,
        pub p99_latency: HashMap<String, f64>,
        pub utilization: HashMap<String, f64>,
        pub cost_per_1k_tokens: HashMap<String, f64>,
        pub hedge_win_rate: HashMap<String, f64>,
    }

    impl StrategyComparison {
        /// Find the best strategy for a given metric
        pub fn best_strategy_for_metric(&self, metric: &str) -> Option<String> {
            let values = match metric {
                "p95_latency" => &self.p95_latency,
                "p99_latency" => &self.p99_latency,
                "utilization" => &self.utilization,
                "cost_per_1k_tokens" => &self.cost_per_1k_tokens,
                "hedge_win_rate" => &self.hedge_win_rate,
                _ => return None,
            };

            // For latency and cost, lower is better; for utilization and hedge win rate, higher is better
            let is_lower_better = matches!(metric, "p95_latency" | "p99_latency" | "cost_per_1k_tokens");

            if is_lower_better {
                values.iter()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(strategy, _)| strategy.clone())
            } else {
                values.iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(strategy, _)| strategy.clone())
            }
        }

        /// Export comparison to CSV
        pub fn to_csv(&self) -> String {
            let mut csv = String::new();
            csv.push_str("strategy,p95_latency,p99_latency,utilization,cost_per_1k_tokens,hedge_win_rate\n");

            let strategies: std::collections::HashSet<_> = self.p95_latency.keys()
                .chain(self.p99_latency.keys())
                .chain(self.utilization.keys())
                .chain(self.cost_per_1k_tokens.keys())
                .chain(self.hedge_win_rate.keys())
                .collect();

            for strategy in strategies {
                csv.push_str(&format!(
                    "{},{},{},{},{},{}\n",
                    strategy,
                    self.p95_latency.get(strategy).unwrap_or(&0.0),
                    self.p99_latency.get(strategy).unwrap_or(&0.0),
                    self.utilization.get(strategy).unwrap_or(&0.0),
                    self.cost_per_1k_tokens.get(strategy).unwrap_or(&0.0),
                    self.hedge_win_rate.get(strategy).unwrap_or(&0.0),
                ));
            }

            csv
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> SimulationConfig {
        SimulationConfig {
            seed: 42,
            duration_s: 60.0,
            workload: WorkloadConfig {
                duration_s: 60.0,
                arrival: workload::ArrivalConfig::Poisson { rps: 100.0 },
                mix: workload::RequestMix {
                    llm: 1.0,
                    vision: 0.0,
                    asr: 0.0,
                },
                llm: workload::LlmConfig {
                    in_tokens: workload::TokenDistribution::Constant { value: 100 },
                    out_tokens: workload::TokenDistribution::Constant { value: 200 },
                },
                vision: None,
                asr: None,
                tenants: None,
            },
            topology: TopologyConfig {
                cells: 2,
                nodes_per_cell: 4,
                gpu_profiles: vec![GpuProfile {
                    name: "H100-80G".to_string(),
                    tokens_per_s: 240000,
                    concurrency: 16,
                    vram_total_gb: 80.0,
                    batch_window_ms: 8.0,
                    kv_cache_gb_per_req: 1.2,
                }],
                mig: None,
            },
            network: NetworkConfig::default(),
            signals: SignalConfig {
                queue_depth_ms: signals::UpdateFrequency { min: 50.0, max: 100.0 },
                vram_ms: signals::UpdateFrequency { min: 200.0, max: 400.0 },
                p95_ms: signals::UpdateFrequency { min: 1000.0, max: 1500.0 },
                transport_ms: signals::TransportDelayConfig {
                    intra_cell: [5.0, 25.0],
                    inter_cell: [50.0, 200.0],
                },
            },
            strategies: vec!["baseline_rr".to_string(), "mesh".to_string()],
        }
    }

    #[test]
    fn test_config_validation() {
        let config = create_test_config();
        assert!(config.validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.topology.cells = 0;
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_simulation_runner_creation() {
        let config = create_test_config();
        let runner = SimulationRunner::new(config);
        assert!(runner.is_ok());

        let runner = runner.unwrap();
        assert_eq!(runner.world().total_cells(), 2);
        assert_eq!(runner.world().total_nodes(), 8);
    }

    #[test]
    fn test_router_config_creation() {
        let config = create_test_config();
        let runner = SimulationRunner::new(config).unwrap();

        let router_config = runner.create_router_config("baseline_rr");
        assert!(router_config.is_ok());

        let router_config = runner.create_router_config("invalid_strategy");
        assert!(router_config.is_err());
    }
}
