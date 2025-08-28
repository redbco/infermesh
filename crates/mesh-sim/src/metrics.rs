use std::collections::HashMap;
use hdrhistogram::Histogram;
use serde::{Deserialize, Serialize};
use crate::engine::{RequestId, RequestType, NodeId};

/// Metrics collector for the simulation
#[derive(Debug)]
pub struct Metrics {
    /// Latency histograms by request type
    pub latency_histograms: HashMap<RequestType, Histogram<u64>>,
    /// Queue wait time histograms by request type
    pub queue_wait_histograms: HashMap<RequestType, Histogram<u64>>,
    /// Service time histograms by request type
    pub service_time_histograms: HashMap<RequestType, Histogram<u64>>,
    /// Time-to-first-token histograms for streaming requests
    pub ttft_histograms: HashMap<RequestType, Histogram<u64>>,
    /// Request counters
    pub request_counters: RequestCounters,
    /// Node utilization tracking
    pub node_utilization: HashMap<NodeId, UtilizationTracker>,
    /// Throughput tracking
    pub throughput_tracker: ThroughputTracker,
    /// Hedge and abort metrics
    pub hedge_metrics: HedgeMetrics,
    /// Cost metrics
    pub cost_metrics: CostMetrics,
}

impl Metrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            latency_histograms: HashMap::new(),
            queue_wait_histograms: HashMap::new(),
            service_time_histograms: HashMap::new(),
            ttft_histograms: HashMap::new(),
            request_counters: RequestCounters::default(),
            node_utilization: HashMap::new(),
            throughput_tracker: ThroughputTracker::new(),
            hedge_metrics: HedgeMetrics::default(),
            cost_metrics: CostMetrics::default(),
        }
    }

    /// Initialize histograms for a request type if not already present
    fn ensure_histograms(&mut self, request_type: RequestType) {
        // Create histograms with 3 significant digits and max value of 1 hour (3.6M ms)
        let max_value = 3_600_000; // 1 hour in milliseconds
        
        self.latency_histograms.entry(request_type).or_insert_with(|| {
            Histogram::new_with_bounds(1, max_value, 3).unwrap()
        });
        
        self.queue_wait_histograms.entry(request_type).or_insert_with(|| {
            Histogram::new_with_bounds(1, max_value, 3).unwrap()
        });
        
        self.service_time_histograms.entry(request_type).or_insert_with(|| {
            Histogram::new_with_bounds(1, max_value, 3).unwrap()
        });
        
        self.ttft_histograms.entry(request_type).or_insert_with(|| {
            Histogram::new_with_bounds(1, max_value, 3).unwrap()
        });
    }

    /// Record end-to-end latency for a request
    pub fn record_latency(&mut self, request_type: RequestType, latency_ms: f64) {
        self.ensure_histograms(request_type);
        
        if let Some(histogram) = self.latency_histograms.get_mut(&request_type) {
            let _ = histogram.record(latency_ms as u64);
        }
        
        self.request_counters.increment_completed(request_type);
    }

    /// Record queue wait time for a request
    pub fn record_queue_wait(&mut self, request_type: RequestType, wait_time_ms: f64) {
        self.ensure_histograms(request_type);
        
        if let Some(histogram) = self.queue_wait_histograms.get_mut(&request_type) {
            let _ = histogram.record(wait_time_ms as u64);
        }
    }

    /// Record service time for a request
    pub fn record_service_time(&mut self, request_type: RequestType, service_time_ms: f64) {
        self.ensure_histograms(request_type);
        
        if let Some(histogram) = self.service_time_histograms.get_mut(&request_type) {
            let _ = histogram.record(service_time_ms as u64);
        }
    }

    /// Record time-to-first-token for streaming requests
    pub fn record_ttft(&mut self, request_type: RequestType, ttft_ms: f64) {
        self.ensure_histograms(request_type);
        
        if let Some(histogram) = self.ttft_histograms.get_mut(&request_type) {
            let _ = histogram.record(ttft_ms as u64);
        }
    }

    /// Record request arrival
    pub fn record_arrival(&mut self, request_type: RequestType) {
        self.request_counters.increment_arrived(request_type);
    }

    /// Record request abort
    pub fn record_abort(&mut self, request_type: RequestType, reason: AbortReason) {
        self.request_counters.increment_aborted(request_type);
        match reason {
            AbortReason::Timeout => self.hedge_metrics.timeouts += 1,
            AbortReason::Cancelled => self.hedge_metrics.cancellations += 1,
            AbortReason::ResourceExhaustion => self.hedge_metrics.resource_exhaustion += 1,
        }
    }

    /// Record hedge request
    pub fn record_hedge(&mut self, original_request_id: RequestId, hedge_request_id: RequestId) {
        self.hedge_metrics.total_hedges += 1;
        self.hedge_metrics.active_hedges.insert(original_request_id, hedge_request_id);
    }

    /// Record hedge completion (either original or hedge completed first)
    pub fn record_hedge_completion(&mut self, request_id: RequestId, was_hedge_winner: bool) {
        if was_hedge_winner {
            self.hedge_metrics.hedge_wins += 1;
        }
        
        // Remove from active hedges
        self.hedge_metrics.active_hedges.retain(|_, &mut hedge_id| hedge_id != request_id);
    }

    /// Update node utilization
    pub fn update_node_utilization(&mut self, node_id: NodeId, utilization: f64, vram_usage: f64, current_time: f64) {
        let tracker = self.node_utilization.entry(node_id).or_insert_with(|| {
            UtilizationTracker::new(node_id)
        });
        
        tracker.update(utilization, vram_usage, current_time);
    }

    /// Record throughput data
    pub fn record_throughput(&mut self, tokens_processed: u64, requests_completed: u64, time_window_s: f64) {
        self.throughput_tracker.record(tokens_processed, requests_completed, time_window_s);
    }

    /// Update cost metrics
    pub fn update_cost_metrics(&mut self, gpu_hours: f64, cost_per_gpu_hour: f64, tokens_processed: u64) {
        self.cost_metrics.total_gpu_hours += gpu_hours;
        self.cost_metrics.total_cost += gpu_hours * cost_per_gpu_hour;
        self.cost_metrics.total_tokens += tokens_processed;
    }

    /// Get latency percentiles for a request type
    pub fn get_latency_percentiles(&self, request_type: RequestType) -> Option<LatencyPercentiles> {
        let histogram = self.latency_histograms.get(&request_type)?;
        
        Some(LatencyPercentiles {
            p50: histogram.value_at_percentile(50.0) as f64,
            p95: histogram.value_at_percentile(95.0) as f64,
            p99: histogram.value_at_percentile(99.0) as f64,
            p999: histogram.value_at_percentile(99.9) as f64,
            mean: histogram.mean(),
            max: histogram.max() as f64,
            count: histogram.len(),
        })
    }

    /// Get queue wait percentiles for a request type
    pub fn get_queue_wait_percentiles(&self, request_type: RequestType) -> Option<LatencyPercentiles> {
        let histogram = self.queue_wait_histograms.get(&request_type)?;
        
        Some(LatencyPercentiles {
            p50: histogram.value_at_percentile(50.0) as f64,
            p95: histogram.value_at_percentile(95.0) as f64,
            p99: histogram.value_at_percentile(99.0) as f64,
            p999: histogram.value_at_percentile(99.9) as f64,
            mean: histogram.mean(),
            max: histogram.max() as f64,
            count: histogram.len(),
        })
    }

    /// Get TTFT percentiles for a request type
    pub fn get_ttft_percentiles(&self, request_type: RequestType) -> Option<LatencyPercentiles> {
        let histogram = self.ttft_histograms.get(&request_type)?;
        
        Some(LatencyPercentiles {
            p50: histogram.value_at_percentile(50.0) as f64,
            p95: histogram.value_at_percentile(95.0) as f64,
            p99: histogram.value_at_percentile(99.0) as f64,
            p999: histogram.value_at_percentile(99.9) as f64,
            mean: histogram.mean(),
            max: histogram.max() as f64,
            count: histogram.len(),
        })
    }

    /// Generate a summary report
    pub fn generate_summary(&self) -> MetricsSummary {
        let mut summary = MetricsSummary::default();
        
        // Aggregate latency metrics across request types
        for request_type in [RequestType::LLM, RequestType::Vision, RequestType::ASR] {
            if let Some(percentiles) = self.get_latency_percentiles(request_type) {
                summary.latency_by_type.insert(request_type, percentiles);
            }
            
            if let Some(percentiles) = self.get_queue_wait_percentiles(request_type) {
                summary.queue_wait_by_type.insert(request_type, percentiles);
            }
            
            if let Some(percentiles) = self.get_ttft_percentiles(request_type) {
                summary.ttft_by_type.insert(request_type, percentiles);
            }
        }
        
        // Request counters
        summary.request_counters = self.request_counters.clone();
        
        // Throughput
        summary.throughput = self.throughput_tracker.get_average_throughput();
        
        // Utilization
        summary.average_utilization = self.get_average_utilization();
        summary.average_vram_utilization = self.get_average_vram_utilization();
        
        // Hedge metrics
        summary.hedge_metrics = self.hedge_metrics.clone();
        
        // Cost metrics
        summary.cost_metrics = self.cost_metrics.clone();
        
        summary
    }

    /// Get utilization load balancing score (lower = better balanced)
    /// Returns standard deviation of utilization across nodes
    fn get_average_utilization(&self) -> f64 {
        if self.node_utilization.is_empty() {
            return 0.0;
        }
        
        let utilizations: Vec<f64> = self.node_utilization.values()
            .map(|tracker| tracker.get_average_utilization())
            .collect();
        
        if utilizations.len() < 2 {
            return utilizations.first().copied().unwrap_or(0.0);
        }
        
        // Calculate standard deviation as load balancing metric
        let mean: f64 = utilizations.iter().sum::<f64>() / utilizations.len() as f64;
        let variance: f64 = utilizations.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / utilizations.len() as f64;
        
        variance.sqrt()
    }

    /// Get average VRAM utilization across all nodes
    fn get_average_vram_utilization(&self) -> f64 {
        if self.node_utilization.is_empty() {
            return 0.0;
        }
        
        let total: f64 = self.node_utilization.values()
            .map(|tracker| tracker.get_average_vram_utilization())
            .sum();
        
        total / self.node_utilization.len() as f64
    }
}

/// Latency percentiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub p999: f64,
    pub mean: f64,
    pub max: f64,
    pub count: u64,
}

/// Request counters by type
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestCounters {
    pub arrived: HashMap<RequestType, u64>,
    pub completed: HashMap<RequestType, u64>,
    pub aborted: HashMap<RequestType, u64>,
}

impl RequestCounters {
    fn increment_arrived(&mut self, request_type: RequestType) {
        *self.arrived.entry(request_type).or_insert(0) += 1;
    }
    
    fn increment_completed(&mut self, request_type: RequestType) {
        *self.completed.entry(request_type).or_insert(0) += 1;
    }
    
    fn increment_aborted(&mut self, request_type: RequestType) {
        *self.aborted.entry(request_type).or_insert(0) += 1;
    }
    
    /// Get completion rate for a request type
    pub fn completion_rate(&self, request_type: RequestType) -> f64 {
        let arrived = self.arrived.get(&request_type).copied().unwrap_or(0);
        let completed = self.completed.get(&request_type).copied().unwrap_or(0);
        
        if arrived == 0 {
            0.0
        } else {
            completed as f64 / arrived as f64
        }
    }
    
    /// Get total requests across all types
    pub fn total_arrived(&self) -> u64 {
        self.arrived.values().sum()
    }
    
    pub fn total_completed(&self) -> u64 {
        self.completed.values().sum()
    }
    
    pub fn total_aborted(&self) -> u64 {
        self.aborted.values().sum()
    }
}

/// Node utilization tracker
#[derive(Debug, Clone)]
pub struct UtilizationTracker {
    pub node_id: NodeId,
    utilization_samples: Vec<f64>,
    vram_samples: Vec<f64>,
    sample_times: Vec<f64>,
}

impl UtilizationTracker {
    fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            utilization_samples: Vec::new(),
            vram_samples: Vec::new(),
            sample_times: Vec::new(),
        }
    }
    
    fn update(&mut self, utilization: f64, vram_usage: f64, time: f64) {
        self.utilization_samples.push(utilization);
        self.vram_samples.push(vram_usage);
        self.sample_times.push(time);
    }
    
    /// Get average utilization
    pub fn get_average_utilization(&self) -> f64 {
        if self.utilization_samples.is_empty() {
            0.0
        } else {
            self.utilization_samples.iter().sum::<f64>() / self.utilization_samples.len() as f64
        }
    }
    
    /// Get average VRAM utilization
    pub fn get_average_vram_utilization(&self) -> f64 {
        if self.vram_samples.is_empty() {
            0.0
        } else {
            self.vram_samples.iter().sum::<f64>() / self.vram_samples.len() as f64
        }
    }
    
    /// Get utilization percentiles
    pub fn get_utilization_percentiles(&self) -> Option<UtilizationPercentiles> {
        if self.utilization_samples.is_empty() {
            return None;
        }
        
        let mut sorted = self.utilization_samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let len = sorted.len();
        Some(UtilizationPercentiles {
            p50: sorted[len * 50 / 100],
            p95: sorted[len * 95 / 100],
            p99: sorted[len * 99 / 100],
            mean: self.get_average_utilization(),
            max: sorted[len - 1],
        })
    }
}

/// Utilization percentiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilizationPercentiles {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub mean: f64,
    pub max: f64,
}

/// Throughput tracker
#[derive(Debug, Clone)]
pub struct ThroughputTracker {
    tokens_per_second_samples: Vec<f64>,
    requests_per_second_samples: Vec<f64>,
}

impl ThroughputTracker {
    fn new() -> Self {
        Self {
            tokens_per_second_samples: Vec::new(),
            requests_per_second_samples: Vec::new(),
        }
    }
    
    fn record(&mut self, tokens: u64, requests: u64, time_window_s: f64) {
        if time_window_s > 0.0 {
            self.tokens_per_second_samples.push(tokens as f64 / time_window_s);
            self.requests_per_second_samples.push(requests as f64 / time_window_s);
        }
    }
    
    /// Get average throughput
    pub fn get_average_throughput(&self) -> ThroughputMetrics {
        ThroughputMetrics {
            tokens_per_second: if self.tokens_per_second_samples.is_empty() {
                0.0
            } else {
                self.tokens_per_second_samples.iter().sum::<f64>() / self.tokens_per_second_samples.len() as f64
            },
            requests_per_second: if self.requests_per_second_samples.is_empty() {
                0.0
            } else {
                self.requests_per_second_samples.iter().sum::<f64>() / self.requests_per_second_samples.len() as f64
            },
        }
    }
}

/// Throughput metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    pub tokens_per_second: f64,
    pub requests_per_second: f64,
}

/// Hedge and abort metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HedgeMetrics {
    pub total_hedges: u64,
    pub hedge_wins: u64,
    pub timeouts: u64,
    pub cancellations: u64,
    pub resource_exhaustion: u64,
    #[serde(skip)]
    pub active_hedges: HashMap<RequestId, RequestId>,
}

impl HedgeMetrics {
    /// Get hedge win rate
    pub fn hedge_win_rate(&self) -> f64 {
        if self.total_hedges == 0 {
            0.0
        } else {
            self.hedge_wins as f64 / self.total_hedges as f64
        }
    }
    
    /// Get wasted work percentage (hedges that didn't win)
    pub fn wasted_work_rate(&self) -> f64 {
        if self.total_hedges == 0 {
            0.0
        } else {
            (self.total_hedges - self.hedge_wins) as f64 / self.total_hedges as f64
        }
    }
}

/// Cost metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostMetrics {
    pub total_gpu_hours: f64,
    pub total_cost: f64,
    pub total_tokens: u64,
}

impl CostMetrics {
    /// Get cost per 1k tokens
    pub fn cost_per_1k_tokens(&self) -> f64 {
        if self.total_tokens == 0 {
            0.0
        } else {
            self.total_cost / (self.total_tokens as f64 / 1000.0)
        }
    }
    
    /// Get effective cost per GPU hour (accounting for utilization)
    pub fn effective_cost_per_gpu_hour(&self, average_utilization: f64) -> f64 {
        if average_utilization == 0.0 {
            self.total_cost
        } else {
            self.total_cost / average_utilization
        }
    }
}

/// Abort reasons
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbortReason {
    Timeout,
    Cancelled,
    ResourceExhaustion,
}

/// Complete metrics summary
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub latency_by_type: HashMap<RequestType, LatencyPercentiles>,
    pub queue_wait_by_type: HashMap<RequestType, LatencyPercentiles>,
    pub ttft_by_type: HashMap<RequestType, LatencyPercentiles>,
    pub request_counters: RequestCounters,
    pub throughput: ThroughputMetrics,
    pub average_utilization: f64,
    pub average_vram_utilization: f64,
    pub hedge_metrics: HedgeMetrics,
    pub cost_metrics: CostMetrics,
}

impl MetricsSummary {
    /// Export to CSV format
    pub fn to_csv(&self) -> anyhow::Result<String> {
        let mut csv = String::new();
        
        // Header
        csv.push_str("metric_type,request_type,p50,p95,p99,p999,mean,max,count\n");
        
        // Latency metrics
        for (request_type, percentiles) in &self.latency_by_type {
            csv.push_str(&format!(
                "latency,{:?},{},{},{},{},{},{},{}\n",
                request_type, percentiles.p50, percentiles.p95, percentiles.p99,
                percentiles.p999, percentiles.mean, percentiles.max, percentiles.count
            ));
        }
        
        // Queue wait metrics
        for (request_type, percentiles) in &self.queue_wait_by_type {
            csv.push_str(&format!(
                "queue_wait,{:?},{},{},{},{},{},{},{}\n",
                request_type, percentiles.p50, percentiles.p95, percentiles.p99,
                percentiles.p999, percentiles.mean, percentiles.max, percentiles.count
            ));
        }
        
        // TTFT metrics
        for (request_type, percentiles) in &self.ttft_by_type {
            csv.push_str(&format!(
                "ttft,{:?},{},{},{},{},{},{},{}\n",
                request_type, percentiles.p50, percentiles.p95, percentiles.p99,
                percentiles.p999, percentiles.mean, percentiles.max, percentiles.count
            ));
        }
        
        Ok(csv)
    }
    
    /// Export to JSON format
    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let mut metrics = Metrics::new();
        
        // Record some latencies
        metrics.record_latency(RequestType::LLM, 100.0);
        metrics.record_latency(RequestType::LLM, 200.0);
        metrics.record_latency(RequestType::LLM, 150.0);
        
        let percentiles = metrics.get_latency_percentiles(RequestType::LLM).unwrap();
        assert_eq!(percentiles.count, 3);
        assert!(percentiles.mean > 0.0);
        assert!(percentiles.p50 > 0.0);
    }

    #[test]
    fn test_request_counters() {
        let mut counters = RequestCounters::default();
        
        counters.increment_arrived(RequestType::LLM);
        counters.increment_arrived(RequestType::LLM);
        counters.increment_completed(RequestType::LLM);
        
        assert_eq!(counters.total_arrived(), 2);
        assert_eq!(counters.total_completed(), 1);
        assert_eq!(counters.completion_rate(RequestType::LLM), 0.5);
    }

    #[test]
    fn test_utilization_tracker() {
        let mut tracker = UtilizationTracker::new(1);
        
        tracker.update(0.5, 0.6, 1000.0);
        tracker.update(0.7, 0.8, 2000.0);
        tracker.update(0.6, 0.7, 3000.0);
        
        assert_eq!(tracker.get_average_utilization(), 0.6);
        assert!((tracker.get_average_vram_utilization() - 0.7).abs() < 0.001);
        
        let percentiles = tracker.get_utilization_percentiles().unwrap();
        assert!(percentiles.mean > 0.0);
        assert!(percentiles.p50 > 0.0);
    }

    #[test]
    fn test_hedge_metrics() {
        let mut hedge_metrics = HedgeMetrics::default();
        
        hedge_metrics.total_hedges = 10;
        hedge_metrics.hedge_wins = 3;
        
        assert_eq!(hedge_metrics.hedge_win_rate(), 0.3);
        assert_eq!(hedge_metrics.wasted_work_rate(), 0.7);
    }

    #[test]
    fn test_cost_metrics() {
        let mut cost_metrics = CostMetrics::default();
        
        cost_metrics.total_gpu_hours = 10.0;
        cost_metrics.total_cost = 50.0; // $5/hour
        cost_metrics.total_tokens = 1_000_000;
        
        assert_eq!(cost_metrics.cost_per_1k_tokens(), 0.05); // $0.05 per 1k tokens
    }

    #[test]
    fn test_metrics_summary() {
        let mut metrics = Metrics::new();
        
        // Add some data
        metrics.record_latency(RequestType::LLM, 100.0);
        metrics.record_arrival(RequestType::LLM);
        
        let summary = metrics.generate_summary();
        assert!(summary.latency_by_type.contains_key(&RequestType::LLM));
        assert_eq!(summary.request_counters.total_arrived(), 1);
    }
}
