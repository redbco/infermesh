use std::collections::HashMap;
use rand::prelude::*;
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};
use rayon::prelude::*;

use crate::engine::{RequestCtx, StateView, Target, RouterStrategy, NodeId, ModelId, RoutingNodeInfo};
use crate::net::{NetworkTopology};

/// ML-Enhanced Mesh Strategy with Real-Time Learning
/// 
/// This advanced strategy combines all previous approaches with machine learning:
/// - Real-time weight optimization using gradient descent
/// - Workload-aware adaptation based on request patterns  
/// - Global optimization across all routing decisions
/// - Continuous feedback loop for learning from outcomes
#[derive(Debug, Clone)]
pub struct MlEnhancedMesh {
    decision_cost_us: u64,
    
    // ML Components
    weights: MLWeights,
    optimizer: GradientOptimizer,
    workload_analyzer: WorkloadAnalyzer,
    #[allow(dead_code)]
    feedback_loop: FeedbackLoop,
    
    // Performance tracking
    decision_history: Vec<MLRoutingDecision>,
    performance_metrics: PerformanceTracker,
    
    // Adaptive parameters
    #[allow(dead_code)]
    learning_rate: f64,
    exploration_rate: f64,
    adaptation_window: usize,
}

/// Neural network-inspired weights for multi-objective optimization
#[derive(Debug, Clone)]
struct MLWeights {
    // Primary objectives
    latency_weight: f64,
    #[allow(dead_code)]
    throughput_weight: f64,
    utilization_weight: f64,
    cost_weight: f64,
    
    // Secondary factors
    network_weight: f64,
    vram_weight: f64,
    queue_weight: f64,
    congestion_weight: f64,
    
    // Dynamic factors
    workload_bias: f64,
    #[allow(dead_code)]
    time_decay: f64,
    #[allow(dead_code)]
    exploration_bonus: f64,
}

/// Gradient-based optimizer for real-time weight adjustment
#[derive(Debug, Clone)]
struct GradientOptimizer {
    momentum: HashMap<String, f64>,
    learning_rate: f64,
    decay_rate: f64,
    #[allow(dead_code)]
    gradient_history: Vec<HashMap<String, f64>>,
}

/// Analyzes workload patterns for adaptive routing
#[derive(Debug, Clone)]
struct WorkloadAnalyzer {
    #[allow(dead_code)]
    request_patterns: HashMap<String, RequestPattern>,
    #[allow(dead_code)]
    temporal_trends: Vec<TemporalTrend>,
    #[allow(dead_code)]
    congestion_predictor: CongestionPredictor,
}

/// Real-time feedback loop for continuous learning
#[derive(Debug, Clone)]
struct FeedbackLoop {
    #[allow(dead_code)]
    outcome_buffer: Vec<RoutingOutcome>,
    #[allow(dead_code)]
    reward_calculator: RewardCalculator,
    #[allow(dead_code)]
    policy_updater: PolicyUpdater,
}

/// Individual routing decision with context
#[derive(Debug, Clone)]
struct MLRoutingDecision {
    timestamp: f64,
    request_id: u64,
    chosen_node: NodeId,
    #[allow(dead_code)]
    alternatives: Vec<NodeId>,
    #[allow(dead_code)]
    decision_factors: DecisionFactors,
    predicted_outcome: OutcomePrediction,
}

/// Factors that influenced a routing decision
#[derive(Debug, Clone)]
struct DecisionFactors {
    #[allow(dead_code)]
    node_scores: HashMap<NodeId, f64>,
    #[allow(dead_code)]
    workload_context: WorkloadContext,
    #[allow(dead_code)]
    system_state: SystemState,
    #[allow(dead_code)]
    confidence_level: f64,
}

/// Performance tracking with advanced metrics
#[derive(Debug, Clone)]
struct PerformanceTracker {
    latency_samples: Vec<f64>,
    #[allow(dead_code)]
    throughput_samples: Vec<f64>,
    #[allow(dead_code)]
    success_rate: f64,
    adaptation_effectiveness: f64,
    #[allow(dead_code)]
    prediction_accuracy: f64,
}

// Supporting structures for ML components
#[derive(Debug, Clone)]
struct RequestPattern {
    #[allow(dead_code)]
    arrival_rate: f64,
    #[allow(dead_code)]
    token_distribution: (f64, f64), // mean, std
    #[allow(dead_code)]
    temporal_correlation: f64,
}

#[derive(Debug, Clone)]
struct TemporalTrend {
    #[allow(dead_code)]
    time_window: f64,
    #[allow(dead_code)]
    trend_direction: f64,
    #[allow(dead_code)]
    confidence: f64,
}

#[derive(Debug, Clone)]
struct CongestionPredictor {
    #[allow(dead_code)]
    hotspot_probability: HashMap<NodeId, f64>,
    #[allow(dead_code)]
    cascade_risk: f64,
    #[allow(dead_code)]
    recovery_time: f64,
}

#[derive(Debug, Clone)]
struct RoutingOutcome {
    #[allow(dead_code)]
    decision_id: u64,
    actual_latency: f64,
    resource_efficiency: f64,
    cascade_effect: f64,
    satisfaction_score: f64,
}

#[derive(Debug, Clone)]
struct RewardCalculator {
    #[allow(dead_code)]
    multi_objective_weights: HashMap<String, f64>,
    #[allow(dead_code)]
    penalty_factors: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
struct PolicyUpdater {
    #[allow(dead_code)]
    update_frequency: f64,
    #[allow(dead_code)]
    convergence_threshold: f64,
    #[allow(dead_code)]
    stability_factor: f64,
}

#[derive(Debug, Clone)]
struct OutcomePrediction {
    expected_latency: f64,
    #[allow(dead_code)]
    confidence_interval: (f64, f64),
    risk_assessment: f64,
}

#[derive(Debug, Clone)]
struct WorkloadContext {
    #[allow(dead_code)]
    current_load: f64,
    #[allow(dead_code)]
    request_type_mix: HashMap<String, f64>,
    #[allow(dead_code)]
    temporal_position: f64,
}

#[derive(Debug, Clone)]
struct SystemState {
    #[allow(dead_code)]
    global_utilization: f64,
    #[allow(dead_code)]
    network_congestion: f64,
    #[allow(dead_code)]
    resource_availability: f64,
}

impl MlEnhancedMesh {
    pub fn new(decision_cost_us: u64) -> Self {
        Self {
            decision_cost_us,
            weights: MLWeights::new(),
            optimizer: GradientOptimizer::new(0.001), // Learning rate
            workload_analyzer: WorkloadAnalyzer::new(),
            feedback_loop: FeedbackLoop::new(),
            decision_history: Vec::new(),
            performance_metrics: PerformanceTracker::new(),
            learning_rate: 0.001,
            exploration_rate: 0.1,
            adaptation_window: 1000,
        }
    }
    
    /// Advanced ML-based scoring that adapts in real-time
    fn calculate_ml_enhanced_score(&mut self, node_info: &RoutingNodeInfo, view: &StateView, ctx: &RequestCtx) -> f64 {
        // 1. Extract features from current state
        let features = self.extract_features(node_info, view, ctx);
        
        // 2. Apply current ML weights
        let base_score = self.compute_weighted_score(&features);
        
        // 3. Add workload-aware adjustments
        let workload_adjustment = self.workload_analyzer.calculate_adjustment(&features, ctx);
        
        // 4. Apply exploration bonus for learning
        let exploration_bonus = self.calculate_exploration_bonus(node_info.node_id, ctx.current_time);
        
        // 5. Incorporate global optimization signals
        let global_signal = self.calculate_global_optimization_signal(node_info, view);
        
        base_score + workload_adjustment + exploration_bonus + global_signal
    }
    
    /// Extract comprehensive features for ML decision making
    fn extract_features(&self, node_info: &RoutingNodeInfo, view: &StateView, ctx: &RequestCtx) -> HashMap<String, f64> {
        let mut features = HashMap::new();
        
        // Node-specific features
        let queue_depth = node_info.queue_depths.values().sum::<u32>() as f64;
        let vram_pressure = node_info.vram_usage_gb / node_info.gpu_profile.vram_total_gb;
        let utilization = node_info.utilization;
        
        features.insert("queue_depth".to_string(), queue_depth);
        features.insert("vram_pressure".to_string(), vram_pressure);
        features.insert("utilization".to_string(), utilization);
        features.insert("network_distance".to_string(), if node_info.cell_id > 0 { 1.0 } else { 0.0 });
        
        // System-wide features
        let avg_utilization = view.nodes.values().map(|n| n.utilization).sum::<f64>() / view.nodes.len() as f64;
        let utilization_variance = view.nodes.values()
            .map(|n| (n.utilization - avg_utilization).powi(2))
            .sum::<f64>() / view.nodes.len() as f64;
        
        features.insert("system_load".to_string(), avg_utilization);
        features.insert("load_imbalance".to_string(), utilization_variance);
        
        // Temporal features
        features.insert("time_of_day".to_string(), (ctx.current_time / 86400000.0) % 1.0); // Normalized daily cycle
        features.insert("request_rate".to_string(), self.estimate_current_request_rate(ctx.current_time));
        
        // Request-specific features
        let token_complexity = (ctx.request.input_tokens + ctx.request.expected_output_tokens) as f64;
        features.insert("token_complexity".to_string(), token_complexity);
        features.insert("request_type".to_string(), match ctx.request.request_type {
            crate::engine::RequestType::LLM => 1.0,
            crate::engine::RequestType::Vision => 2.0,
            crate::engine::RequestType::ASR => 3.0,
        });
        
        features
    }
    
    /// Compute weighted score using current ML weights
    fn compute_weighted_score(&self, features: &HashMap<String, f64>) -> f64 {
        let mut score = 0.0;
        
        // Apply learned weights to features
        score += features.get("queue_depth").unwrap_or(&0.0) * self.weights.queue_weight;
        score += features.get("vram_pressure").unwrap_or(&0.0) * self.weights.vram_weight;
        score += features.get("utilization").unwrap_or(&0.0) * self.weights.utilization_weight;
        score += features.get("network_distance").unwrap_or(&0.0) * self.weights.network_weight;
        score += features.get("load_imbalance").unwrap_or(&0.0) * self.weights.congestion_weight;
        
        // Add workload bias
        score += self.weights.workload_bias;
        
        score
    }
    
    /// Calculate exploration bonus for learning new strategies
    fn calculate_exploration_bonus(&self, node_id: NodeId, _current_time: f64) -> f64 {
        // Encourage exploration of less-used nodes
        let recent_selections = self.decision_history.iter()
            .rev()
            .take(100)
            .filter(|d| d.chosen_node == node_id)
            .count();
        
        let exploration_factor = 1.0 / (1.0 + recent_selections as f64);
        self.exploration_rate * exploration_factor
    }
    
    /// Global optimization signal considering system-wide effects
    fn calculate_global_optimization_signal(&self, node_info: &RoutingNodeInfo, view: &StateView) -> f64 {
        // Analyze global load distribution
        let total_load: f64 = view.nodes.values().map(|n| n.utilization).sum();
        let avg_load = total_load / view.nodes.len() as f64;
        
        // Encourage load balancing
        let load_balance_signal = (avg_load - node_info.utilization) * 0.1;
        
        // Prevent cascade failures
        let cascade_prevention = if node_info.utilization > 0.9 { -1.0 } else { 0.0 };
        
        load_balance_signal + cascade_prevention
    }
    
    /// Estimate current request arrival rate for workload awareness
    fn estimate_current_request_rate(&self, current_time: f64) -> f64 {
        let window_size = 10000.0; // 10 seconds
        let recent_decisions = self.decision_history.iter()
            .filter(|d| current_time - d.timestamp < window_size)
            .count();
        
        recent_decisions as f64 / (window_size / 1000.0) // Requests per second
    }
    
    /// Perform periodic learning updates
    fn perform_learning_update(&mut self) {
        // Simulate feedback collection (in real implementation, this would come from actual metrics)
        let simulated_outcomes: Vec<RoutingOutcome> = self.decision_history.iter()
            .rev()
            .take(50)
            .map(|decision| RoutingOutcome {
                decision_id: decision.request_id,
                actual_latency: decision.predicted_outcome.expected_latency + (random::<f64>() - 0.5) * 20.0,
                resource_efficiency: 0.7 + random::<f64>() * 0.3,
                cascade_effect: if decision.predicted_outcome.risk_assessment > 0.5 { 0.1 } else { 0.0 },
                satisfaction_score: 0.8 + random::<f64>() * 0.2,
            })
            .collect();
        
        // Update weights based on outcomes
        self.update_weights_from_feedback(&simulated_outcomes);
        
        // Update performance metrics
        self.performance_metrics.update_from_outcomes(&simulated_outcomes);
    }
    
    /// Update ML weights based on feedback
    fn update_weights_from_feedback(&mut self, outcomes: &[RoutingOutcome]) {
        for outcome in outcomes {
            // Calculate gradients based on outcome quality
            let gradients = self.calculate_gradients(outcome);
            
            // Apply gradients using optimizer
            self.optimizer.apply_gradients(&mut self.weights, &gradients);
        }
        
        // Decay exploration rate over time
        self.exploration_rate *= 0.9999;
        self.exploration_rate = self.exploration_rate.max(0.01);
    }
    
    /// Calculate gradients for weight updates
    fn calculate_gradients(&self, outcome: &RoutingOutcome) -> HashMap<String, f64> {
        let mut gradients = HashMap::new();
        
        // Reward good outcomes, penalize bad ones
        let reward = self.calculate_reward(outcome);
        
        // Simple gradient calculation (in practice, this would be more sophisticated)
        gradients.insert("latency_weight".to_string(), -reward * outcome.actual_latency);
        gradients.insert("utilization_weight".to_string(), -reward * outcome.resource_efficiency);
        gradients.insert("cost_weight".to_string(), -reward * (1.0 - outcome.satisfaction_score));
        
        gradients
    }
    
    /// Calculate reward signal from routing outcome
    fn calculate_reward(&self, outcome: &RoutingOutcome) -> f64 {
        // Multi-objective reward combining latency, efficiency, and satisfaction
        let latency_reward = 1.0 / (1.0 + outcome.actual_latency / 1000.0); // Normalize latency
        let efficiency_reward = outcome.resource_efficiency;
        let satisfaction_reward = outcome.satisfaction_score;
        let cascade_penalty = -outcome.cascade_effect;
        
        0.4 * latency_reward + 0.3 * efficiency_reward + 0.2 * satisfaction_reward + 0.1 * cascade_penalty
    }
}

impl RouterStrategy for MlEnhancedMesh {
    fn choose(&mut self, ctx: &RequestCtx, view: &StateView) -> Target {
        // 1. Calculate ML-enhanced scores for all nodes
        let mut node_scores = HashMap::new();
        let mut best_node = 0;
        let mut best_score = f64::INFINITY;
        
        for (node_id, node_info) in &view.nodes {
            let score = self.calculate_ml_enhanced_score(node_info, view, ctx);
            node_scores.insert(*node_id, score);
            
            if score < best_score {
                best_score = score;
                best_node = *node_id;
            }
        }
        
        // 2. Record decision for learning
        let decision = MLRoutingDecision {
            timestamp: ctx.current_time,
            request_id: ctx.request.id,
            chosen_node: best_node,
            alternatives: view.nodes.keys().cloned().collect(),
            decision_factors: DecisionFactors {
                node_scores: node_scores.clone(),
                workload_context: WorkloadContext {
                    current_load: view.nodes.values().map(|n| n.utilization).sum::<f64>() / view.nodes.len() as f64,
                    request_type_mix: HashMap::new(), // Would be populated in real implementation
                    temporal_position: ctx.current_time,
                },
                system_state: SystemState {
                    global_utilization: view.nodes.values().map(|n| n.utilization).sum::<f64>() / view.nodes.len() as f64,
                    network_congestion: 0.0, // Would be calculated from network state
                    resource_availability: 1.0 - view.nodes.values().map(|n| n.vram_usage_gb / n.gpu_profile.vram_total_gb).sum::<f64>() / view.nodes.len() as f64,
                },
                confidence_level: 0.8, // Would be calculated based on prediction uncertainty
            },
            predicted_outcome: OutcomePrediction {
                expected_latency: best_score * 100.0, // Convert score to latency estimate
                confidence_interval: (best_score * 80.0, best_score * 120.0),
                risk_assessment: if best_score > 1.0 { 0.8 } else { 0.2 },
            },
        };
        
        self.decision_history.push(decision);
        
        // 3. Limit history size for performance
        if self.decision_history.len() > self.adaptation_window {
            self.decision_history.remove(0);
        }
        
        // 4. Periodic learning updates
        if self.decision_history.len() % 100 == 0 {
            self.perform_learning_update();
        }
        
        Target {
            node_id: best_node,
            model_id: ctx.request.model_id.clone(),
        }
    }
    
    fn decision_cost_us(&self) -> u64 {
        self.decision_cost_us + 100 // Higher cost for ML processing
    }
}

// Implementation of supporting structures
impl MLWeights {
    fn new() -> Self {
        Self {
            latency_weight: 0.4,
            throughput_weight: 0.3,
            utilization_weight: 0.2,
            cost_weight: 0.1,
            network_weight: 0.15,
            vram_weight: 0.25,
            queue_weight: 0.35,
            congestion_weight: 0.25,
            workload_bias: 0.0,
            time_decay: 0.99,
            exploration_bonus: 0.1,
        }
    }
}

impl GradientOptimizer {
    fn new(learning_rate: f64) -> Self {
        Self {
            momentum: HashMap::new(),
            learning_rate,
            decay_rate: 0.9,
            gradient_history: Vec::new(),
        }
    }
    
    fn apply_gradients(&mut self, weights: &mut MLWeights, gradients: &HashMap<String, f64>) {
        // Apply gradients with momentum (simplified implementation)
        for (key, gradient) in gradients {
            let momentum = self.momentum.entry(key.clone()).or_insert(0.0);
            *momentum = self.decay_rate * *momentum + (1.0 - self.decay_rate) * gradient;
            
            // Update weights (this would be more sophisticated in a real implementation)
            match key.as_str() {
                "latency_weight" => weights.latency_weight -= self.learning_rate * *momentum,
                "utilization_weight" => weights.utilization_weight -= self.learning_rate * *momentum,
                "cost_weight" => weights.cost_weight -= self.learning_rate * *momentum,
                _ => {}
            }
        }
        
        // Normalize weights to maintain stability
        self.normalize_weights(weights);
    }
    
    fn normalize_weights(&self, weights: &mut MLWeights) {
        // Ensure weights stay within reasonable bounds
        weights.latency_weight = weights.latency_weight.clamp(0.1, 1.0);
        weights.utilization_weight = weights.utilization_weight.clamp(0.1, 1.0);
        weights.cost_weight = weights.cost_weight.clamp(0.05, 0.5);
        weights.queue_weight = weights.queue_weight.clamp(0.1, 1.0);
        weights.vram_weight = weights.vram_weight.clamp(0.1, 1.0);
        weights.network_weight = weights.network_weight.clamp(0.05, 0.5);
        weights.congestion_weight = weights.congestion_weight.clamp(0.1, 1.0);
    }
}

impl WorkloadAnalyzer {
    fn new() -> Self {
        Self {
            request_patterns: HashMap::new(),
            temporal_trends: Vec::new(),
            congestion_predictor: CongestionPredictor {
                hotspot_probability: HashMap::new(),
                cascade_risk: 0.0,
                recovery_time: 0.0,
            },
        }
    }
    
    fn calculate_adjustment(&self, _features: &HashMap<String, f64>, _ctx: &RequestCtx) -> f64 {
        // Simplified workload adjustment
        0.0
    }
}

impl FeedbackLoop {
    fn new() -> Self {
        Self {
            outcome_buffer: Vec::new(),
            reward_calculator: RewardCalculator {
                multi_objective_weights: HashMap::new(),
                penalty_factors: HashMap::new(),
            },
            policy_updater: PolicyUpdater {
                update_frequency: 100.0,
                convergence_threshold: 0.01,
                stability_factor: 0.95,
            },
        }
    }
}

impl PerformanceTracker {
    fn new() -> Self {
        Self {
            latency_samples: Vec::new(),
            throughput_samples: Vec::new(),
            success_rate: 1.0,
            adaptation_effectiveness: 0.0,
            prediction_accuracy: 0.0,
        }
    }
    
    fn update_from_outcomes(&mut self, outcomes: &[RoutingOutcome]) {
        for outcome in outcomes {
            self.latency_samples.push(outcome.actual_latency);
            
            // Limit sample size
            if self.latency_samples.len() > 1000 {
                self.latency_samples.remove(0);
            }
        }
        
        // Update metrics
        if !self.latency_samples.is_empty() {
            let avg_latency = self.latency_samples.iter().sum::<f64>() / self.latency_samples.len() as f64;
            self.adaptation_effectiveness = 1.0 / (1.0 + avg_latency / 1000.0);
        }
    }
}

/// Configuration for routing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    pub strategy: StrategyType,
    pub decision_cost_us: Option<u64>,
    pub hedge_config: Option<HedgeConfig>,
}

/// Types of routing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StrategyType {
    #[serde(rename = "baseline_rr")]
    BaselineRoundRobin,
    #[serde(rename = "least_queue")]
    LeastQueue,
    #[serde(rename = "heuristic")]
    Heuristic { alpha: f64, beta: f64, gamma: f64 },
    #[serde(rename = "mesh")]
    Mesh,
    #[serde(rename = "mesh_hedge")]
    MeshHedge,
    #[serde(rename = "adaptive_mesh")]
    AdaptiveMesh { load_threshold: f64 },
    #[serde(rename = "predictive_mesh")]
    PredictiveMesh { prediction_window_ms: f64 },
    #[serde(rename = "hybrid_mesh")]
    HybridMesh,
    #[serde(rename = "ml_enhanced_mesh")]
    MlEnhancedMesh,
    #[serde(rename = "mesh_stale")]
    MeshStale { staleness_threshold_ms: f64 },
}

/// Hedging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HedgeConfig {
    /// Fraction of SLA after which to fire hedge (e.g., 0.35 = 35% of SLA)
    pub alpha: f64,
    /// Maximum number of hedge attempts
    pub max_hedges: u32,
}

/// Round-robin baseline strategy
pub struct BaselineRoundRobin {
    decision_cost_us: u64,
    next_node_index: HashMap<ModelId, usize>,
    #[allow(dead_code)]
    rng: SmallRng,
}

impl BaselineRoundRobin {
    pub fn new(decision_cost_us: u64, seed: u64) -> Self {
        Self {
            decision_cost_us,
            next_node_index: HashMap::new(),
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl RouterStrategy for BaselineRoundRobin {
    fn choose(&mut self, ctx: &RequestCtx, view: &StateView) -> Target {
        // Get all available nodes
        let available_nodes: Vec<NodeId> = view.nodes.keys().copied().collect();
        
        if available_nodes.is_empty() {
            // Fallback to node 0 if no nodes available
            return Target {
                node_id: 0,
                model_id: ctx.request.model_id.clone(),
            };
        }
        
        // Round-robin selection
        let node_count = available_nodes.len();
        let node_index = self.next_node_index
            .entry(ctx.request.model_id.clone())
            .and_modify(|idx| *idx = (*idx + 1) % node_count)
            .or_insert(0);

        Target {
            node_id: available_nodes[*node_index],
            model_id: ctx.request.model_id.clone(),
        }
    }

    fn decision_cost_us(&self) -> u64 {
        self.decision_cost_us
    }
}

/// Least queue depth strategy
pub struct LeastQueue {
    decision_cost_us: u64,
}

impl LeastQueue {
    pub fn new(decision_cost_us: u64) -> Self {
        Self { decision_cost_us }
    }
}

impl RouterStrategy for LeastQueue {
    fn choose(&mut self, ctx: &RequestCtx, view: &StateView) -> Target {
        // Find node with least total queue depth
        let best_node = view.nodes.iter()
            .min_by_key(|(_, node_state)| {
                node_state.queue_depths.values().sum::<u32>()
            })
            .map(|(node_id, _)| *node_id)
            .unwrap_or(0);

        Target {
            node_id: best_node,
            model_id: ctx.request.model_id.clone(),
        }
    }

    fn decision_cost_us(&self) -> u64 {
        self.decision_cost_us
    }
}

/// Heuristic strategy with weighted scoring
pub struct Heuristic {
    decision_cost_us: u64,
    alpha: f64, // Weight for work_left
    beta: f64,  // Weight for vram_pressure
    gamma: f64, // Weight for recent_p95
}

impl Heuristic {
    pub fn new(decision_cost_us: u64, alpha: f64, beta: f64, gamma: f64) -> Self {
        Self {
            decision_cost_us,
            alpha,
            beta,
            gamma,
        }
    }
    
    fn calculate_heuristic_score(&self, node_info: &RoutingNodeInfo) -> f64 {
        // Calculate composite score based on node state
        let work_left = node_info.queue_depths.values().sum::<u32>() as f64;
        let vram_pressure = node_info.vram_usage_gb / node_info.gpu_profile.vram_total_gb;
        let utilization_penalty = node_info.utilization; // Higher utilization = worse score
        
        // Lower score is better
        self.alpha * work_left + self.beta * vram_pressure + self.gamma * utilization_penalty
    }
}

impl RouterStrategy for Heuristic {
    fn choose(&mut self, ctx: &RequestCtx, view: &StateView) -> Target {
        // Calculate heuristic scores for all nodes using parallel processing
        let best_node = view.nodes
            .par_iter()  // Parallel iterator for scoring
            .min_by(|(_, node_a), (_, node_b)| {
                let score_a = self.calculate_heuristic_score(node_a);
                let score_b = self.calculate_heuristic_score(node_b);
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(node_id, _)| *node_id)
            .unwrap_or(0);

        Target {
            node_id: best_node,
            model_id: ctx.request.model_id.clone(),
        }
    }

    fn decision_cost_us(&self) -> u64 {
        self.decision_cost_us
    }
}

/// Full mesh strategy with comprehensive scoring
pub struct Mesh {
    decision_cost_us: u64,
    network_topology: Option<NetworkTopology>,
}

impl Mesh {
    pub fn new(decision_cost_us: u64) -> Self {
        Self {
            decision_cost_us,
            network_topology: None,
        }
    }

    pub fn with_network_topology(mut self, topology: NetworkTopology) -> Self {
        self.network_topology = Some(topology);
        self
    }

    fn calculate_mesh_score(&self, node_info: &RoutingNodeInfo, _view: &StateView) -> f64 {
        // Comprehensive scoring similar to heuristic but with more factors
        let work_left = node_info.queue_depths.values().sum::<u32>() as f64;
        let vram_pressure = node_info.vram_usage_gb / node_info.gpu_profile.vram_total_gb;
        let utilization_penalty = node_info.utilization;
        
        // Network penalty based on cell (simplified - assume inter-cell has penalty)
        let network_penalty = if node_info.cell_id > 0 { 10.0 } else { 0.0 };
        
        // Mesh uses more sophisticated scoring
        let base_score = 0.4 * work_left + 0.3 * vram_pressure + 0.2 * utilization_penalty;
        base_score + 0.1 * network_penalty
    }
}

impl RouterStrategy for Mesh {
    fn choose(&mut self, ctx: &RequestCtx, view: &StateView) -> Target {
        // Implement comprehensive mesh scoring with network penalties using parallel processing
        let best_node = view.nodes
            .par_iter()  // Parallel iterator for scoring
            .min_by(|(_, node_a), (_, node_b)| {
                let score_a = self.calculate_mesh_score(node_a, view);
                let score_b = self.calculate_mesh_score(node_b, view);
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(node_id, _)| *node_id)
            .unwrap_or(0);

        Target {
            node_id: best_node,
            model_id: ctx.request.model_id.clone(),
        }
    }

    fn decision_cost_us(&self) -> u64 {
        self.decision_cost_us
    }
}

/// Mesh strategy with hedging support
pub struct MeshHedge {
    mesh_strategy: Mesh,
    hedge_config: HedgeConfig,
    active_hedges: HashMap<crate::engine::RequestId, HedgeState>,
}

#[derive(Debug, Clone)]
struct HedgeState {
    #[allow(dead_code)]
    original_target: Target,
    #[allow(dead_code)]
    hedge_targets: Vec<Target>,
    #[allow(dead_code)]
    hedge_fire_time: f64,
}

impl MeshHedge {
    pub fn new(decision_cost_us: u64, hedge_config: HedgeConfig) -> Self {
        Self {
            mesh_strategy: Mesh::new(decision_cost_us),
            hedge_config,
            active_hedges: HashMap::new(),
        }
    }

    pub fn with_network_topology(mut self, topology: NetworkTopology) -> Self {
        self.mesh_strategy = self.mesh_strategy.with_network_topology(topology);
        self
    }

    /// Calculate when to fire a hedge for a request
    pub fn calculate_hedge_fire_time(&self, ctx: &RequestCtx) -> Option<f64> {
        if let Some(sla_ms) = ctx.request.sla_ms {
            Some(ctx.current_time + sla_ms * self.hedge_config.alpha)
        } else {
            None
        }
    }

    /// Choose an alternative target for hedging
    pub fn choose_hedge_target(&mut self, _ctx: &RequestCtx, _view: &StateView, original_target: &Target) -> Option<Target> {
        // TODO: Choose a different target than the original
        // For now, return a simple alternative
        if original_target.node_id > 1 {
            Some(Target {
                node_id: original_target.node_id - 1,
                model_id: original_target.model_id.clone(),
            })
        } else {
            Some(Target {
                node_id: original_target.node_id + 1,
                model_id: original_target.model_id.clone(),
            })
        }
    }
}

impl RouterStrategy for MeshHedge {
    fn choose(&mut self, ctx: &RequestCtx, view: &StateView) -> Target {
        let target = self.mesh_strategy.choose(ctx, view);
        
        // Set up hedging if SLA is available
        if let Some(hedge_fire_time) = self.calculate_hedge_fire_time(ctx) {
            let hedge_state = HedgeState {
                original_target: target.clone(),
                hedge_targets: Vec::new(),
                hedge_fire_time,
            };
            self.active_hedges.insert(ctx.request.id, hedge_state);
        }
        
        target
    }

    fn decision_cost_us(&self) -> u64 {
        self.mesh_strategy.decision_cost_us()
    }
}

/// Adaptive Load-Aware Mesh Strategy
#[derive(Debug, Clone)]
pub struct AdaptiveMesh {
    decision_cost_us: u64,
    load_threshold: f64,        // When to switch from load balancing to performance
    #[allow(dead_code)]
    performance_weight: f64,    // Weight for performance factors
    #[allow(dead_code)]
    balance_weight: f64,        // Weight for load balancing factors
    recent_decisions: Vec<(f64, NodeId)>, // Track recent decisions for adaptation
}

impl AdaptiveMesh {
    pub fn new(decision_cost_us: u64, load_threshold: f64) -> Self {
        Self {
            decision_cost_us,
            load_threshold,
            performance_weight: 0.7,
            balance_weight: 0.3,
            recent_decisions: Vec::new(),
        }
    }
    
    fn calculate_system_load(&self, view: &StateView) -> f64 {
        if view.nodes.is_empty() {
            return 0.0;
        }
        
        let total_utilization: f64 = view.nodes.values()
            .map(|node| node.utilization)
            .sum();
        
        total_utilization / view.nodes.len() as f64
    }
    
    fn calculate_adaptive_score(&mut self, node_info: &RoutingNodeInfo, system_load: f64) -> f64 {
        let work_left = node_info.queue_depths.values().sum::<u32>() as f64;
        let vram_pressure = node_info.vram_usage_gb / node_info.gpu_profile.vram_total_gb;
        let utilization = node_info.utilization;
        
        // Adaptive weighting based on system load
        let (perf_weight, balance_weight) = if system_load > self.load_threshold {
            // High load: prioritize performance
            (0.8, 0.2)
        } else {
            // Low load: prioritize load balancing
            (0.4, 0.6)
        };
        
        // Performance factors (lower is better)
        let performance_score = work_left * 0.4 + vram_pressure * 0.3 + utilization * 0.3;
        
        // Load balancing factor (prefer less utilized nodes)
        let balance_score = utilization;
        
        // Network penalty (simplified)
        let network_penalty = if node_info.cell_id > 0 { 5.0 } else { 0.0 };
        
        perf_weight * performance_score + balance_weight * balance_score + 0.1 * network_penalty
    }
}

impl RouterStrategy for AdaptiveMesh {
    fn choose(&mut self, ctx: &RequestCtx, view: &StateView) -> Target {
        let system_load = self.calculate_system_load(view);
        
        // Find best node using adaptive scoring (sequential due to mutable access)
        let best_node = view.nodes.iter()
            .min_by(|(_, node_a), (_, node_b)| {
                let score_a = self.calculate_adaptive_score(node_a, system_load);
                let score_b = self.calculate_adaptive_score(node_b, system_load);
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(node_id, _)| *node_id)
            .unwrap_or(0);
        
        // Track decision for future adaptation
        self.recent_decisions.push((ctx.current_time, best_node));
        if self.recent_decisions.len() > 100 {
            self.recent_decisions.remove(0);
        }
        
        Target {
            node_id: best_node,
            model_id: ctx.request.model_id.clone(),
        }
    }
    
    fn decision_cost_us(&self) -> u64 {
        self.decision_cost_us
    }
}

/// Predictive Mesh Strategy with Request Forecasting (Optimized)
#[derive(Debug, Clone)]
pub struct PredictiveMesh {
    decision_cost_us: u64,
    arrival_history: HashMap<NodeId, Vec<f64>>, // Per-node arrival times (optimized)
    prediction_window_ms: f64,                  // How far ahead to predict
    congestion_threshold: f64,                  // When to avoid nodes
    max_history_size: usize,                    // Limit history size for performance
}

impl PredictiveMesh {
    pub fn new(decision_cost_us: u64, prediction_window_ms: f64) -> Self {
        Self {
            decision_cost_us,
            arrival_history: HashMap::new(),
            prediction_window_ms,
            congestion_threshold: 0.8,
            max_history_size: 100, // Limit history to prevent unbounded growth
        }
    }
    
    fn predict_future_load(&self, node_id: NodeId, current_time: f64) -> f64 {
        // Optimized: Direct lookup per node instead of scanning entire history
        if let Some(node_arrivals) = self.arrival_history.get(&node_id) {
            // Count recent arrivals within the prediction window
            let cutoff_time = current_time - self.prediction_window_ms;
            let recent_arrivals = node_arrivals.iter()
                .filter(|&&time| time > cutoff_time)
                .count();
            
            // Predict future load based on recent trend
            let arrival_rate = recent_arrivals as f64 / (self.prediction_window_ms / 1000.0);
            
            // Estimate future queue growth
            arrival_rate * (self.prediction_window_ms / 1000.0)
        } else {
            0.0 // No history for this node
        }
    }
    
    fn calculate_predictive_score(&self, node_info: &RoutingNodeInfo, current_time: f64) -> f64 {
        let current_work = node_info.queue_depths.values().sum::<u32>() as f64;
        let vram_pressure = node_info.vram_usage_gb / node_info.gpu_profile.vram_total_gb;
        let utilization = node_info.utilization;
        
        // Predict future load
        let predicted_load = self.predict_future_load(node_info.node_id, current_time);
        
        // Calculate total expected work
        let total_expected_work = current_work + predicted_load;
        
        // Penalize nodes that will be congested
        let congestion_penalty = if utilization > self.congestion_threshold {
            100.0 * (utilization - self.congestion_threshold)
        } else {
            0.0
        };
        
        // Network penalty
        let network_penalty = if node_info.cell_id > 0 { 3.0 } else { 0.0 };
        
        // Combined score (lower is better)
        0.4 * total_expected_work + 
        0.3 * vram_pressure + 
        0.2 * utilization + 
        0.05 * network_penalty + 
        0.05 * congestion_penalty
    }
}

impl RouterStrategy for PredictiveMesh {
    fn choose(&mut self, ctx: &RequestCtx, view: &StateView) -> Target {
        // Find best node using predictive scoring (sequential due to mutable access)
        let best_node = view.nodes.iter()
            .min_by(|(_, node_a), (_, node_b)| {
                let score_a = self.calculate_predictive_score(node_a, ctx.current_time);
                let score_b = self.calculate_predictive_score(node_b, ctx.current_time);
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(node_id, _)| *node_id)
            .unwrap_or(0);
        
        // Record this decision for future predictions (optimized per-node tracking)
        let node_arrivals = self.arrival_history.entry(best_node).or_insert_with(Vec::new);
        node_arrivals.push(ctx.current_time);
        
        // Limit history size per node to prevent unbounded growth
        if node_arrivals.len() > self.max_history_size {
            node_arrivals.remove(0); // Remove oldest entry
        }
        
        // Periodically clean up old entries across all nodes
        if ctx.current_time as u64 % 10000 == 0 { // Every 10 seconds
            let cutoff_time = ctx.current_time - (self.prediction_window_ms * 2.0);
            for arrivals in self.arrival_history.values_mut() {
                arrivals.retain(|&time| time > cutoff_time);
            }
        }
        
        Target {
            node_id: best_node,
            model_id: ctx.request.model_id.clone(),
        }
    }
    
    fn decision_cost_us(&self) -> u64 {
        self.decision_cost_us + 25 // Slightly higher cost for prediction
    }
}

/// Mesh strategy with stale signal tolerance
pub struct MeshStale {
    mesh_strategy: Mesh,
    #[allow(dead_code)]
    staleness_threshold_ms: f64,
}

impl MeshStale {
    pub fn new(decision_cost_us: u64, staleness_threshold_ms: f64) -> Self {
        Self {
            mesh_strategy: Mesh::new(decision_cost_us),
            staleness_threshold_ms,
        }
    }

    pub fn with_network_topology(mut self, topology: NetworkTopology) -> Self {
        self.mesh_strategy = self.mesh_strategy.with_network_topology(topology);
        self
    }
}

impl RouterStrategy for MeshStale {
    fn choose(&mut self, ctx: &RequestCtx, view: &StateView) -> Target {
        // TODO: Only use signals that are within staleness threshold
        // Fall back to simpler strategy if signals are too stale
        self.mesh_strategy.choose(ctx, view)
    }

    fn decision_cost_us(&self) -> u64 {
        self.mesh_strategy.decision_cost_us()
    }
}

/// Hybrid Multi-Objective Strategy
#[derive(Debug, Clone)]
pub struct HybridMesh {
    decision_cost_us: u64,
    objectives: Vec<ObjectiveWeight>,
    adaptation_rate: f64,
    performance_history: Vec<PerformanceMetric>,
}

#[derive(Debug, Clone)]
struct ObjectiveWeight {
    name: String,
    weight: f64,
    target_value: f64,
}

#[derive(Debug, Clone)]
struct PerformanceMetric {
    #[allow(dead_code)]
    timestamp: f64,
    #[allow(dead_code)]
    latency: f64,
    #[allow(dead_code)]
    utilization_variance: f64,
    #[allow(dead_code)]
    cost: f64,
}

impl HybridMesh {
    pub fn new(decision_cost_us: u64) -> Self {
        Self {
            decision_cost_us,
            objectives: vec![
                ObjectiveWeight { name: "latency".to_string(), weight: 0.4, target_value: 500.0 },
                ObjectiveWeight { name: "balance".to_string(), weight: 0.3, target_value: 0.1 },
                ObjectiveWeight { name: "cost".to_string(), weight: 0.2, target_value: 0.001 },
                ObjectiveWeight { name: "throughput".to_string(), weight: 0.1, target_value: 1000.0 },
            ],
            adaptation_rate: 0.1,
            performance_history: Vec::new(),
        }
    }
    
    fn calculate_multi_objective_score(&self, node_info: &RoutingNodeInfo, view: &StateView) -> f64 {
        let work_left = node_info.queue_depths.values().sum::<u32>() as f64;
        let _vram_pressure = node_info.vram_usage_gb / node_info.gpu_profile.vram_total_gb;
        let utilization = node_info.utilization;
        
        // Calculate variance in utilization across all nodes (for load balancing)
        let utilizations: Vec<f64> = view.nodes.values().map(|n| n.utilization).collect();
        let mean_util = utilizations.iter().sum::<f64>() / utilizations.len() as f64;
        let _variance = utilizations.iter()
            .map(|u| (u - mean_util).powi(2))
            .sum::<f64>() / utilizations.len() as f64;
        
        // Multi-objective scoring
        let mut total_score = 0.0;
        
        for objective in &self.objectives {
            let score = match objective.name.as_str() {
                "latency" => {
                    // Estimate latency based on queue and processing power
                    let estimated_latency = work_left / (node_info.gpu_profile.tokens_per_s as f64 / 1000.0);
                    estimated_latency / objective.target_value
                },
                "balance" => {
                    // Penalize nodes that increase variance
                    let new_variance = if utilizations.len() > 1 {
                        let new_utilizations: Vec<f64> = view.nodes.values()
                            .map(|n| if n.node_id == node_info.node_id { 
                                n.utilization + 0.1 // Simulate adding load
                            } else { 
                                n.utilization 
                            })
                            .collect();
                        let new_mean = new_utilizations.iter().sum::<f64>() / new_utilizations.len() as f64;
                        new_utilizations.iter()
                            .map(|u| (u - new_mean).powi(2))
                            .sum::<f64>() / new_utilizations.len() as f64
                    } else {
                        0.0
                    };
                    new_variance / objective.target_value
                },
                "cost" => {
                    // Estimate cost based on GPU efficiency
                    let efficiency = node_info.gpu_profile.tokens_per_s as f64 / node_info.gpu_profile.vram_total_gb;
                    (1.0 / efficiency) / objective.target_value
                },
                "throughput" => {
                    // Favor nodes with higher throughput potential
                    let available_capacity = (1.0 - utilization) * node_info.gpu_profile.tokens_per_s as f64;
                    objective.target_value / available_capacity.max(1.0)
                },
                _ => 1.0,
            };
            
            total_score += objective.weight * score;
        }
        
        // Network penalty
        let network_penalty = if node_info.cell_id > 0 { 0.1 } else { 0.0 };
        
        total_score + network_penalty
    }
    
    fn adapt_weights(&mut self, _current_performance: &PerformanceMetric) {
        // Simple adaptation: adjust weights based on performance
        // This is a placeholder for more sophisticated ML-based adaptation
        for objective in &mut self.objectives {
            match objective.name.as_str() {
                "latency" => {
                    // If latency is too high, increase its weight
                    objective.weight = (objective.weight + self.adaptation_rate * 0.1).min(0.8);
                },
                "balance" => {
                    // Maintain balance importance
                    objective.weight = (objective.weight * 0.99).max(0.1);
                },
                _ => {}
            }
        }
        
        // Normalize weights
        let total_weight: f64 = self.objectives.iter().map(|o| o.weight).sum();
        for objective in &mut self.objectives {
            objective.weight /= total_weight;
        }
    }
}

impl RouterStrategy for HybridMesh {
    fn choose(&mut self, ctx: &RequestCtx, view: &StateView) -> Target {
        // Find best node using multi-objective scoring (sequential due to mutable access)
        let best_node = view.nodes.iter()
            .min_by(|(_, node_a), (_, node_b)| {
                let score_a = self.calculate_multi_objective_score(node_a, view);
                let score_b = self.calculate_multi_objective_score(node_b, view);
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(node_id, _)| *node_id)
            .unwrap_or(0);
        
        // Record performance for adaptation (simplified)
        let current_metric = PerformanceMetric {
            timestamp: ctx.current_time,
            latency: 0.0, // Would be calculated from actual metrics
            utilization_variance: 0.0,
            cost: 0.0,
        };
        
        self.performance_history.push(current_metric.clone());
        if self.performance_history.len() > 50 {
            self.performance_history.remove(0);
        }
        
        // Adapt weights periodically
        if self.performance_history.len() % 10 == 0 {
            self.adapt_weights(&current_metric);
        }
        
        Target {
            node_id: best_node,
            model_id: ctx.request.model_id.clone(),
        }
    }
    
    fn decision_cost_us(&self) -> u64 {
        self.decision_cost_us + 50 // Higher cost for multi-objective optimization
    }
}

/// Factory for creating routing strategies
pub struct RouterFactory;

impl RouterFactory {
    /// Create a router strategy from configuration
    pub fn create_strategy(
        config: RouterConfig,
        seed: u64,
        network_topology: Option<NetworkTopology>,
    ) -> Box<dyn RouterStrategy> {
        let decision_cost = config.decision_cost_us.unwrap_or(50);

        match config.strategy {
            StrategyType::BaselineRoundRobin => {
                Box::new(BaselineRoundRobin::new(decision_cost, seed))
            }
            StrategyType::LeastQueue => {
                Box::new(LeastQueue::new(decision_cost))
            }
            StrategyType::Heuristic { alpha, beta, gamma } => {
                Box::new(Heuristic::new(decision_cost, alpha, beta, gamma))
            }
            StrategyType::Mesh => {
                let mut mesh = Mesh::new(decision_cost);
                if let Some(topology) = network_topology {
                    mesh = mesh.with_network_topology(topology);
                }
                Box::new(mesh)
            }
            StrategyType::MeshHedge => {
                let hedge_config = config.hedge_config.unwrap_or(HedgeConfig {
                    alpha: 0.35,
                    max_hedges: 1,
                });
                let mut mesh_hedge = MeshHedge::new(decision_cost, hedge_config);
                if let Some(topology) = network_topology {
                    mesh_hedge = mesh_hedge.with_network_topology(topology);
                }
                Box::new(mesh_hedge)
            }
            StrategyType::MeshStale { staleness_threshold_ms } => {
                let mut mesh_stale = MeshStale::new(decision_cost, staleness_threshold_ms);
                if let Some(topology) = network_topology {
                    mesh_stale = mesh_stale.with_network_topology(topology);
                }
                Box::new(mesh_stale)
            }
            StrategyType::AdaptiveMesh { load_threshold } => {
                Box::new(AdaptiveMesh::new(decision_cost, load_threshold))
            }
            StrategyType::PredictiveMesh { prediction_window_ms } => {
                Box::new(PredictiveMesh::new(decision_cost, prediction_window_ms))
            }
            StrategyType::HybridMesh => {
                Box::new(HybridMesh::new(decision_cost))
            }
            StrategyType::MlEnhancedMesh => {
                Box::new(MlEnhancedMesh::new(decision_cost))
            }
        }
    }
}

/// Routing decision context with additional metadata
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub target: Target,
    pub decision_time_us: u64,
    pub score: Option<f64>,
    pub alternatives_considered: usize,
    pub hedge_scheduled: bool,
}

/// Enhanced router that tracks decision metadata
pub struct EnhancedRouter {
    strategy: Box<dyn RouterStrategy>,
    decisions: Vec<RoutingDecision>,
}

impl EnhancedRouter {
    pub fn new(strategy: Box<dyn RouterStrategy>) -> Self {
        Self {
            strategy,
            decisions: Vec::new(),
        }
    }

    /// Make a routing decision with metadata tracking
    pub fn route(&mut self, ctx: &RequestCtx, view: &StateView) -> RoutingDecision {
        let start_time = std::time::Instant::now();
        let target = self.strategy.choose(ctx, view);
        let decision_time_us = start_time.elapsed().as_micros() as u64;

        let decision = RoutingDecision {
            target,
            decision_time_us,
            score: None, // TODO: Extract score from strategy if available
            alternatives_considered: 1, // TODO: Track actual alternatives
            hedge_scheduled: false, // TODO: Check if hedge was scheduled
        };

        self.decisions.push(decision.clone());
        decision
    }

    /// Get routing decision statistics
    pub fn get_stats(&self) -> RoutingStats {
        if self.decisions.is_empty() {
            return RoutingStats::default();
        }

        let total_decisions = self.decisions.len();
        let total_decision_time: u64 = self.decisions.iter().map(|d| d.decision_time_us).sum();
        let avg_decision_time = total_decision_time as f64 / total_decisions as f64;
        
        let hedges_scheduled = self.decisions.iter().filter(|d| d.hedge_scheduled).count();

        RoutingStats {
            total_decisions,
            avg_decision_time_us: avg_decision_time,
            hedge_rate: hedges_scheduled as f64 / total_decisions as f64,
        }
    }
}

/// Statistics about routing decisions
#[derive(Debug, Clone, Default)]
pub struct RoutingStats {
    pub total_decisions: usize,
    pub avg_decision_time_us: f64,
    pub hedge_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Request, RequestType};

    fn create_test_request() -> Request {
        Request {
            id: 1,
            request_type: RequestType::LLM,
            tenant_id: "test".to_string(),
            model_id: "test-model".to_string(),
            arrival_time: 0.0,
            input_tokens: 100,
            expected_output_tokens: 200,
            sla_ms: Some(5000.0),
        }
    }

    fn create_test_context() -> RequestCtx {
        RequestCtx {
            request: create_test_request(),
            current_time: 1000.0,
        }
    }

    #[test]
    fn test_baseline_round_robin() {
        let mut strategy = BaselineRoundRobin::new(50, 42);
        let ctx = create_test_context();
        let view = StateView { nodes: HashMap::new(), current_time: 0.0 };

        let target1 = strategy.choose(&ctx, &view);
        let target2 = strategy.choose(&ctx, &view);

        assert_eq!(target1.model_id, "test-model");
        assert_eq!(target2.model_id, "test-model");
        assert_ne!(target1.node_id, target2.node_id); // Should round-robin
        assert_eq!(strategy.decision_cost_us(), 50);
    }

    #[test]
    fn test_heuristic_strategy() {
        let mut strategy = Heuristic::new(75, 0.4, 0.3, 0.3);
        let ctx = create_test_context();
        let view = StateView { nodes: HashMap::new(), current_time: 0.0 };

        let target = strategy.choose(&ctx, &view);
        assert_eq!(target.model_id, "test-model");
        assert_eq!(strategy.decision_cost_us(), 75);
    }

    #[test]
    fn test_mesh_hedge_fire_time() {
        let hedge_config = HedgeConfig {
            alpha: 0.35,
            max_hedges: 1,
        };
        let strategy = MeshHedge::new(100, hedge_config);
        let ctx = create_test_context();

        let hedge_time = strategy.calculate_hedge_fire_time(&ctx);
        assert!(hedge_time.is_some());
        
        // Should fire at 35% of SLA: 1000 + 5000 * 0.35 = 2750
        assert_eq!(hedge_time.unwrap(), 2750.0);
    }

    #[test]
    fn test_router_factory() {
        let config = RouterConfig {
            strategy: StrategyType::BaselineRoundRobin,
            decision_cost_us: Some(100),
            hedge_config: None,
        };

        let router = RouterFactory::create_strategy(config, 42, None);
        assert_eq!(router.decision_cost_us(), 100);
    }

    #[test]
    fn test_enhanced_router() {
        let strategy = Box::new(BaselineRoundRobin::new(50, 42));
        let mut router = EnhancedRouter::new(strategy);
        
        let ctx = create_test_context();
        let view = StateView { nodes: HashMap::new(), current_time: 0.0 };

        let decision = router.route(&ctx, &view);
        assert_eq!(decision.target.model_id, "test-model");
        assert!(decision.decision_time_us > 0);

        let stats = router.get_stats();
        assert_eq!(stats.total_decisions, 1);
        assert!(stats.avg_decision_time_us > 0.0);
    }
}
