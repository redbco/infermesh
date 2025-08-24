//! Load generation utilities for testing infermesh components

use crate::{DevError, Result};
use mesh_core::SloClass;
use mesh_proto::scoring::v1::{
    scoring_client::ScoringClient, AdmitRequest, RequestOutcome, ReportOutcomeRequest,
    ScoreTargetsRequest,
};
use rand::Rng;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::interval;
use tonic::transport::Channel;
use tracing::{debug, info, warn};

/// Configuration for load generation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoadGeneratorConfig {
    /// Target requests per second
    pub target_rps: f64,
    
    /// Duration to run the load test (seconds)
    pub duration_seconds: u64,
    
    /// Model to target
    pub model: String,
    
    /// Model revision
    pub revision: String,
    
    /// SLO class for requests
    pub slo_class: SloClass,
    
    /// Request patterns to use
    pub patterns: Vec<RequestPattern>,
    
    /// Timeout for individual requests (seconds)
    pub request_timeout_seconds: u64,
    
    /// Number of concurrent workers
    pub worker_count: usize,
    
    /// Target endpoint
    pub endpoint: String,
}

impl Default for LoadGeneratorConfig {
    fn default() -> Self {
        Self {
            target_rps: 10.0,
            duration_seconds: 60,
            model: "test-model".to_string(),
            revision: "v1.0".to_string(),
            slo_class: SloClass::Latency,
            patterns: vec![RequestPattern::default()],
            request_timeout_seconds: 30,
            worker_count: 4,
            endpoint: "http://127.0.0.1:50051".to_string(),
        }
    }
}

/// Request pattern for load generation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RequestPattern {
    /// Pattern name
    pub name: String,
    
    /// Weight of this pattern (relative to others)
    pub weight: f64,
    
    /// Token count distribution
    pub token_distribution: TokenDistribution,
    
    /// Think time between requests (milliseconds)
    pub think_time_ms: u64,
}

impl Default for RequestPattern {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            weight: 1.0,
            token_distribution: TokenDistribution::Fixed(128),
            think_time_ms: 0,
        }
    }
}

/// Token count distribution for requests
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum TokenDistribution {
    /// Fixed token count
    Fixed(u32),
    
    /// Uniform distribution between min and max
    Uniform { min: u32, max: u32 },
    
    /// Normal distribution with mean and std dev
    Normal { mean: f64, std_dev: f64 },
    
    /// Exponential distribution with lambda
    Exponential { lambda: f64 },
}

impl TokenDistribution {
    /// Sample a token count from this distribution
    pub fn sample(&self) -> u32 {
        let mut rng = rand::thread_rng();
        
        match self {
            TokenDistribution::Fixed(count) => *count,
            TokenDistribution::Uniform { min, max } => rng.gen_range(*min..=*max),
            TokenDistribution::Normal { mean, std_dev } => {
                use rand_distr::{Distribution, Normal};
                let normal = Normal::new(*mean, *std_dev).unwrap_or_else(|_| Normal::new(128.0, 32.0).unwrap());
                normal.sample(&mut rng).max(1.0) as u32
            }
            TokenDistribution::Exponential { lambda } => {
                use rand_distr::{Distribution, Exp};
                let exp = Exp::new(*lambda).unwrap_or_else(|_| Exp::new(0.01).unwrap());
                exp.sample(&mut rng).max(1.0) as u32
            }
        }
    }
}

/// Load generator statistics
#[derive(Debug, Clone)]
pub struct LoadGeneratorStats {
    /// Total requests sent
    pub requests_sent: u64,
    
    /// Total requests completed
    pub requests_completed: u64,
    
    /// Total requests failed
    pub requests_failed: u64,
    
    /// Total latency (for average calculation)
    pub total_latency_ms: u64,
    
    /// Minimum latency observed
    pub min_latency_ms: u64,
    
    /// Maximum latency observed
    pub max_latency_ms: u64,
    
    /// Start time
    pub start_time: Instant,
    
    /// End time (if completed)
    pub end_time: Option<Instant>,
}

impl LoadGeneratorStats {
    pub fn new() -> Self {
        Self {
            requests_sent: 0,
            requests_completed: 0,
            requests_failed: 0,
            total_latency_ms: 0,
            min_latency_ms: u64::MAX,
            max_latency_ms: 0,
            start_time: Instant::now(),
            end_time: None,
        }
    }

    /// Calculate average latency
    pub fn average_latency_ms(&self) -> f64 {
        if self.requests_completed > 0 {
            self.total_latency_ms as f64 / self.requests_completed as f64
        } else {
            0.0
        }
    }

    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.requests_sent > 0 {
            (self.requests_completed as f64) / (self.requests_sent as f64)
        } else {
            0.0
        }
    }

    /// Calculate actual RPS
    pub fn actual_rps(&self) -> f64 {
        let duration = if let Some(end_time) = self.end_time {
            end_time.duration_since(self.start_time)
        } else {
            self.start_time.elapsed()
        };
        
        if duration.as_secs_f64() > 0.0 {
            self.requests_sent as f64 / duration.as_secs_f64()
        } else {
            0.0
        }
    }
}

impl Default for LoadGeneratorStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Load generator for testing infermesh components
pub struct LoadGenerator {
    config: LoadGeneratorConfig,
    stats: Arc<LoadGeneratorStatsInner>,
}

struct LoadGeneratorStatsInner {
    requests_sent: AtomicU64,
    requests_completed: AtomicU64,
    requests_failed: AtomicU64,
    total_latency_ms: AtomicU64,
    min_latency_ms: AtomicU64,
    max_latency_ms: AtomicU64,
    start_time: Instant,
}

impl LoadGenerator {
    /// Create a new load generator
    pub fn new(config: LoadGeneratorConfig) -> Self {
        let stats = LoadGeneratorStatsInner {
            requests_sent: AtomicU64::new(0),
            requests_completed: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
            min_latency_ms: AtomicU64::new(u64::MAX),
            max_latency_ms: AtomicU64::new(0),
            start_time: Instant::now(),
        };

        Self {
            config,
            stats: Arc::new(stats),
        }
    }

    /// Run the load test
    pub async fn run(&self) -> Result<LoadGeneratorStats> {
        info!("Starting load test: {} RPS for {} seconds", 
              self.config.target_rps, self.config.duration_seconds);

        // Create gRPC client
        let channel = Channel::from_shared(self.config.endpoint.clone())
            .map_err(|e| DevError::LoadGeneration(format!("Invalid endpoint: {}", e)))?
            .connect()
            .await
            .map_err(|e| DevError::LoadGeneration(format!("Failed to connect: {}", e)))?;

        let client = ScoringClient::new(channel);

        // Calculate requests per worker
        let requests_per_second_per_worker = self.config.target_rps / self.config.worker_count as f64;
        let interval_ms = if requests_per_second_per_worker > 0.0 {
            (1000.0 / requests_per_second_per_worker) as u64
        } else {
            1000
        };

        // Start workers
        let mut handles = Vec::new();
        for worker_id in 0..self.config.worker_count {
            let client = client.clone();
            let config = self.config.clone();
            let stats = self.stats.clone();
            
            let handle = tokio::spawn(async move {
                Self::worker_loop(worker_id, client, config, stats, interval_ms).await
            });
            
            handles.push(handle);
        }

        // Wait for duration
        tokio::time::sleep(Duration::from_secs(self.config.duration_seconds)).await;

        // Wait for workers to complete
        for handle in handles {
            if let Err(e) = handle.await {
                warn!("Worker failed: {}", e);
            }
        }

        // Collect final stats
        let final_stats = LoadGeneratorStats {
            requests_sent: self.stats.requests_sent.load(Ordering::Relaxed),
            requests_completed: self.stats.requests_completed.load(Ordering::Relaxed),
            requests_failed: self.stats.requests_failed.load(Ordering::Relaxed),
            total_latency_ms: self.stats.total_latency_ms.load(Ordering::Relaxed),
            min_latency_ms: self.stats.min_latency_ms.load(Ordering::Relaxed),
            max_latency_ms: self.stats.max_latency_ms.load(Ordering::Relaxed),
            start_time: self.stats.start_time,
            end_time: Some(Instant::now()),
        };

        info!("Load test completed: {} requests sent, {:.2}% success rate, {:.2} avg latency ms",
              final_stats.requests_sent, final_stats.success_rate() * 100.0, final_stats.average_latency_ms());

        Ok(final_stats)
    }

    /// Worker loop for generating load
    async fn worker_loop(
        worker_id: usize,
        mut client: ScoringClient<Channel>,
        config: LoadGeneratorConfig,
        stats: Arc<LoadGeneratorStatsInner>,
        interval_ms: u64,
    ) {
        debug!("Worker {} started with interval {}ms", worker_id, interval_ms);
        
        let mut interval = interval(Duration::from_millis(interval_ms));
        let start_time = Instant::now();
        
        while start_time.elapsed().as_secs() < config.duration_seconds {
            interval.tick().await;
            
            // Select a request pattern
            let pattern = Self::select_pattern(&config.patterns);
            let tokens = pattern.token_distribution.sample();
            
            // Generate request
            let request_start = Instant::now();
            stats.requests_sent.fetch_add(1, Ordering::Relaxed);
            
            let result = Self::send_request(&mut client, &config, tokens).await;
            let latency_ms = request_start.elapsed().as_millis() as u64;
            
            match result {
                Ok(_) => {
                    stats.requests_completed.fetch_add(1, Ordering::Relaxed);
                    stats.total_latency_ms.fetch_add(latency_ms, Ordering::Relaxed);
                    
                    // Update min/max latency
                    let mut current_min = stats.min_latency_ms.load(Ordering::Relaxed);
                    while latency_ms < current_min {
                        match stats.min_latency_ms.compare_exchange_weak(
                            current_min, latency_ms, Ordering::Relaxed, Ordering::Relaxed
                        ) {
                            Ok(_) => break,
                            Err(x) => current_min = x,
                        }
                    }
                    
                    let mut current_max = stats.max_latency_ms.load(Ordering::Relaxed);
                    while latency_ms > current_max {
                        match stats.max_latency_ms.compare_exchange_weak(
                            current_max, latency_ms, Ordering::Relaxed, Ordering::Relaxed
                        ) {
                            Ok(_) => break,
                            Err(x) => current_max = x,
                        }
                    }
                }
                Err(e) => {
                    stats.requests_failed.fetch_add(1, Ordering::Relaxed);
                    debug!("Request failed: {}", e);
                }
            }
            
            // Think time
            if pattern.think_time_ms > 0 {
                tokio::time::sleep(Duration::from_millis(pattern.think_time_ms)).await;
            }
        }
        
        debug!("Worker {} completed", worker_id);
    }

    /// Select a request pattern based on weights
    fn select_pattern(patterns: &[RequestPattern]) -> &RequestPattern {
        if patterns.len() == 1 {
            return &patterns[0];
        }
        
        let total_weight: f64 = patterns.iter().map(|p| p.weight).sum();
        let mut rng = rand::thread_rng();
        let mut target = rng.gen_range(0.0..total_weight);
        
        for pattern in patterns {
            target -= pattern.weight;
            if target <= 0.0 {
                return pattern;
            }
        }
        
        &patterns[0] // Fallback
    }

    /// Send a single request
    async fn send_request(
        client: &mut ScoringClient<Channel>,
        config: &LoadGeneratorConfig,
        tokens: u32,
    ) -> Result<()> {
        let request_id = uuid::Uuid::new_v4().to_string();
        
        // Score targets
        let score_request = ScoreTargetsRequest {
            model: config.model.clone(),
            revision: config.revision.clone(),
            slo_class: config.slo_class as i32,
            estimated_tokens: tokens,
            timeout_seconds: config.request_timeout_seconds as f32,
            filters: std::collections::HashMap::new(),
            request_id: request_id.clone(),
        };
        
        let score_response = client
            .score_targets(score_request)
            .await
            .map_err(|e| DevError::LoadGeneration(format!("Score targets failed: {}", e)))?;
        
        let targets = score_response.into_inner().targets;
        if targets.is_empty() {
            return Err(DevError::LoadGeneration("No targets available".to_string()));
        }
        
        // Use the best target
        let best_target = &targets[0];
        
        // Admit request
        let admit_request = AdmitRequest {
            model: config.model.clone(),
            revision: config.revision.clone(),
            target_node: best_target.node_id.clone(),
            target_gpu: best_target.gpu_uuid.clone(),
            slo_class: config.slo_class as i32,
            estimated_tokens: tokens,
            timeout_seconds: config.request_timeout_seconds as f32,
            request_id: request_id.clone(),
        };
        
        let admit_response = client
            .admit(admit_request)
            .await
            .map_err(|e| DevError::LoadGeneration(format!("Admit failed: {}", e)))?;
        
        let admit_result = admit_response.into_inner();
        if !admit_result.admitted {
            return Err(DevError::LoadGeneration(format!("Request not admitted: {}", admit_result.reason)));
        }
        
        // Simulate processing time
        let processing_time = Duration::from_millis(50 + (tokens as u64 * 2));
        tokio::time::sleep(processing_time).await;
        
        // Report outcome
        let outcome_request = ReportOutcomeRequest {
            request_id: request_id.clone(),
            admission_token: admit_result.admission_token,
            target_node: best_target.node_id.clone(),
            target_gpu: best_target.gpu_uuid.clone(),
            outcome: RequestOutcome::Success as i32,
            actual_latency_ms: processing_time.as_millis() as f32,
            actual_queue_time_ms: 10.0, // Mock queue time
            actual_tokens: tokens,
            error_message: String::new(),
            completed_at: Some(mesh_proto::timestamp::now()),
        };
        
        client
            .report_outcome(outcome_request)
            .await
            .map_err(|e| DevError::LoadGeneration(format!("Report outcome failed: {}", e)))?;
        
        Ok(())
    }

    /// Get current statistics
    pub fn get_stats(&self) -> LoadGeneratorStats {
        LoadGeneratorStats {
            requests_sent: self.stats.requests_sent.load(Ordering::Relaxed),
            requests_completed: self.stats.requests_completed.load(Ordering::Relaxed),
            requests_failed: self.stats.requests_failed.load(Ordering::Relaxed),
            total_latency_ms: self.stats.total_latency_ms.load(Ordering::Relaxed),
            min_latency_ms: self.stats.min_latency_ms.load(Ordering::Relaxed),
            max_latency_ms: self.stats.max_latency_ms.load(Ordering::Relaxed),
            start_time: self.stats.start_time,
            end_time: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_distribution_fixed() {
        let dist = TokenDistribution::Fixed(100);
        assert_eq!(dist.sample(), 100);
    }

    #[test]
    fn test_token_distribution_uniform() {
        let dist = TokenDistribution::Uniform { min: 50, max: 150 };
        let sample = dist.sample();
        assert!(sample >= 50 && sample <= 150);
    }

    #[test]
    fn test_load_generator_stats() {
        let mut stats = LoadGeneratorStats::new();
        assert_eq!(stats.success_rate(), 0.0);
        assert_eq!(stats.average_latency_ms(), 0.0);
        
        stats.requests_sent = 100;
        stats.requests_completed = 95;
        stats.total_latency_ms = 9500;
        
        assert_eq!(stats.success_rate(), 0.95);
        assert_eq!(stats.average_latency_ms(), 100.0);
    }

    #[test]
    fn test_pattern_selection() {
        let patterns = vec![
            RequestPattern { name: "pattern1".to_string(), weight: 1.0, ..Default::default() },
            RequestPattern { name: "pattern2".to_string(), weight: 2.0, ..Default::default() },
            RequestPattern { name: "pattern3".to_string(), weight: 1.0, ..Default::default() },
        ];
        
        let selected = LoadGenerator::select_pattern(&patterns);
        assert!(["pattern1", "pattern2", "pattern3"].contains(&selected.name.as_str()));
    }
}
