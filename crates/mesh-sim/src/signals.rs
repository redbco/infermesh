use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use rand::Rng;
use crate::engine::NodeId;

/// Signal configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalConfig {
    /// Queue depth signal update frequency
    pub queue_depth_ms: UpdateFrequency,
    /// VRAM usage signal update frequency
    pub vram_ms: UpdateFrequency,
    /// P95 latency signal update frequency
    pub p95_ms: UpdateFrequency,
    /// Transport delay configuration
    pub transport_ms: TransportDelayConfig,
}

/// Update frequency configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateFrequency {
    pub min: f64,
    pub max: f64,
}

impl UpdateFrequency {
    /// Sample an update interval with jitter
    pub fn sample(&self) -> f64 {
        let mut rng = rand::thread_rng();
        rng.gen_range(self.min..self.max)
    }
}

/// Transport delay configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportDelayConfig {
    /// Intra-cell transport delay range [min, max] in ms
    pub intra_cell: [f64; 2],
    /// Inter-cell transport delay range [min, max] in ms
    pub inter_cell: [f64; 2],
}

impl TransportDelayConfig {
    /// Sample transport delay
    pub fn sample_delay(&self, same_cell: bool) -> f64 {
        let mut rng = rand::thread_rng();
        let range = if same_cell { self.intra_cell } else { self.inter_cell };
        rng.gen_range(range[0]..range[1])
    }
}

/// Types of signals that can be tracked
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignalType {
    QueueDepth,
    VramUsage,
    P95Latency,
    Utilization,
    Temperature,
    PowerDraw,
}

/// A single signal measurement
#[derive(Debug, Clone)]
pub struct Signal {
    pub signal_type: SignalType,
    pub node_id: NodeId,
    pub value: f64,
    pub timestamp: f64,
    pub transport_delay: f64,
}

impl Signal {
    /// Check if this signal is stale relative to current time
    pub fn is_stale(&self, current_time: f64, staleness_threshold_ms: f64) -> bool {
        let age = current_time - (self.timestamp + self.transport_delay);
        age > staleness_threshold_ms
    }

    /// Get the effective timestamp when this signal becomes available
    pub fn effective_timestamp(&self) -> f64 {
        self.timestamp + self.transport_delay
    }
}

/// Collection of signals for a node
#[derive(Debug, Clone)]
pub struct NodeSignals {
    pub node_id: NodeId,
    pub signals: HashMap<SignalType, Signal>,
    pub last_update: HashMap<SignalType, f64>,
}

impl NodeSignals {
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            signals: HashMap::new(),
            last_update: HashMap::new(),
        }
    }

    /// Update a signal for this node
    pub fn update_signal(&mut self, signal: Signal) {
        self.last_update.insert(signal.signal_type, signal.timestamp);
        self.signals.insert(signal.signal_type, signal);
    }

    /// Get a signal value, checking for staleness
    pub fn get_signal(&self, signal_type: SignalType, current_time: f64, staleness_threshold_ms: f64) -> Option<f64> {
        self.signals.get(&signal_type)
            .filter(|signal| !signal.is_stale(current_time, staleness_threshold_ms))
            .map(|signal| signal.value)
    }

    /// Get all fresh signals
    pub fn get_fresh_signals(&self, current_time: f64, staleness_threshold_ms: f64) -> HashMap<SignalType, f64> {
        self.signals.iter()
            .filter(|(_, signal)| !signal.is_stale(current_time, staleness_threshold_ms))
            .map(|(signal_type, signal)| (*signal_type, signal.value))
            .collect()
    }

    /// Check if we need to update a signal type
    pub fn needs_update(&self, signal_type: SignalType, current_time: f64, update_frequency: &UpdateFrequency) -> bool {
        if let Some(last_update) = self.last_update.get(&signal_type) {
            let time_since_update = current_time - last_update;
            time_since_update >= update_frequency.min
        } else {
            true // Never updated before
        }
    }
}

/// Global signal view for routing decisions
#[derive(Debug, Clone)]
pub struct SignalView {
    pub node_signals: HashMap<NodeId, NodeSignals>,
    pub current_time: f64,
    pub staleness_threshold_ms: f64,
}

impl SignalView {
    pub fn new(current_time: f64, staleness_threshold_ms: f64) -> Self {
        Self {
            node_signals: HashMap::new(),
            current_time,
            staleness_threshold_ms,
        }
    }

    /// Get signals for a specific node
    pub fn get_node_signals(&self, node_id: NodeId) -> Option<&NodeSignals> {
        self.node_signals.get(&node_id)
    }

    /// Get a specific signal value for a node
    pub fn get_signal(&self, node_id: NodeId, signal_type: SignalType) -> Option<f64> {
        self.node_signals.get(&node_id)?
            .get_signal(signal_type, self.current_time, self.staleness_threshold_ms)
    }

    /// Get all nodes with fresh signals of a specific type
    pub fn get_nodes_with_fresh_signal(&self, signal_type: SignalType) -> Vec<(NodeId, f64)> {
        self.node_signals.iter()
            .filter_map(|(node_id, signals)| {
                signals.get_signal(signal_type, self.current_time, self.staleness_threshold_ms)
                    .map(|value| (*node_id, value))
            })
            .collect()
    }

    /// Update the current time and staleness threshold
    pub fn update_time(&mut self, current_time: f64) {
        self.current_time = current_time;
    }

    /// Add or update node signals
    pub fn update_node_signals(&mut self, node_signals: NodeSignals) {
        self.node_signals.insert(node_signals.node_id, node_signals);
    }

    /// Get statistics about signal freshness
    pub fn get_freshness_stats(&self) -> SignalFreshnessStats {
        let mut stats = SignalFreshnessStats::default();
        
        for signals in self.node_signals.values() {
            for (signal_type, signal) in &signals.signals {
                stats.total_signals += 1;
                
                if signal.is_stale(self.current_time, self.staleness_threshold_ms) {
                    stats.stale_signals += 1;
                    *stats.stale_by_type.entry(*signal_type).or_insert(0) += 1;
                } else {
                    stats.fresh_signals += 1;
                    *stats.fresh_by_type.entry(*signal_type).or_insert(0) += 1;
                }

                let age = self.current_time - signal.effective_timestamp();
                stats.total_age += age;
                if age > stats.max_age {
                    stats.max_age = age;
                }
            }
        }

        if stats.total_signals > 0 {
            stats.avg_age = stats.total_age / stats.total_signals as f64;
            stats.freshness_ratio = stats.fresh_signals as f64 / stats.total_signals as f64;
        }

        stats
    }
}

/// Statistics about signal freshness
#[derive(Debug, Clone, Default)]
pub struct SignalFreshnessStats {
    pub total_signals: usize,
    pub fresh_signals: usize,
    pub stale_signals: usize,
    pub freshness_ratio: f64,
    pub avg_age: f64,
    pub max_age: f64,
    pub total_age: f64,
    pub fresh_by_type: HashMap<SignalType, usize>,
    pub stale_by_type: HashMap<SignalType, usize>,
}

/// Signal generator for producing realistic signal updates
#[derive(Clone)]
pub struct SignalGenerator {
    config: SignalConfig,
    next_update_times: HashMap<(NodeId, SignalType), f64>,
}

impl SignalGenerator {
    pub fn new(config: SignalConfig) -> Self {
        Self {
            config,
            next_update_times: HashMap::new(),
        }
    }

    /// Check if a signal needs to be updated
    pub fn needs_update(&mut self, node_id: NodeId, signal_type: SignalType, current_time: f64) -> bool {
        let key = (node_id, signal_type);
        
        if let Some(next_update) = self.next_update_times.get(&key) {
            current_time >= *next_update
        } else {
            // First update
            true
        }
    }

    /// Schedule the next update for a signal
    pub fn schedule_next_update(&mut self, node_id: NodeId, signal_type: SignalType, current_time: f64) -> f64 {
        let update_frequency = match signal_type {
            SignalType::QueueDepth => &self.config.queue_depth_ms,
            SignalType::VramUsage => &self.config.vram_ms,
            SignalType::P95Latency => &self.config.p95_ms,
            _ => &self.config.queue_depth_ms, // Default frequency
        };

        let interval = update_frequency.sample();
        let next_update = current_time + interval;
        
        self.next_update_times.insert((node_id, signal_type), next_update);
        next_update
    }

    /// Generate a signal with transport delay
    pub fn generate_signal(
        &self,
        signal_type: SignalType,
        node_id: NodeId,
        value: f64,
        timestamp: f64,
        source_cell: u32,
        target_cell: u32,
    ) -> Signal {
        let transport_delay = self.config.transport_ms.sample_delay(source_cell == target_cell);
        
        Signal {
            signal_type,
            node_id,
            value,
            timestamp,
            transport_delay,
        }
    }
}

/// Signal aggregator for combining signals across nodes
pub struct SignalAggregator;

impl SignalAggregator {
    /// Calculate average signal value across nodes
    pub fn average(signals: &[(NodeId, f64)]) -> Option<f64> {
        if signals.is_empty() {
            None
        } else {
            let sum: f64 = signals.iter().map(|(_, value)| value).sum();
            Some(sum / signals.len() as f64)
        }
    }

    /// Calculate percentile of signal values
    pub fn percentile(signals: &[(NodeId, f64)], percentile: f64) -> Option<f64> {
        if signals.is_empty() {
            return None;
        }

        let mut values: Vec<f64> = signals.iter().map(|(_, value)| *value).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (percentile / 100.0 * (values.len() - 1) as f64).round() as usize;
        Some(values[index.min(values.len() - 1)])
    }

    /// Find minimum signal value
    pub fn min(signals: &[(NodeId, f64)]) -> Option<(NodeId, f64)> {
        signals.iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .copied()
    }

    /// Find maximum signal value
    pub fn max(signals: &[(NodeId, f64)]) -> Option<(NodeId, f64)> {
        signals.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .copied()
    }

    /// Calculate weighted average based on node capacity
    pub fn weighted_average(signals: &[(NodeId, f64)], weights: &HashMap<NodeId, f64>) -> Option<f64> {
        if signals.is_empty() {
            return None;
        }

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (node_id, value) in signals {
            let weight = weights.get(node_id).copied().unwrap_or(1.0);
            weighted_sum += value * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            Some(weighted_sum / total_weight)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_staleness() {
        let signal = Signal {
            signal_type: SignalType::QueueDepth,
            node_id: 1,
            value: 5.0,
            timestamp: 1000.0,
            transport_delay: 50.0,
        };

        // Signal is fresh within staleness threshold
        assert!(!signal.is_stale(1200.0, 200.0)); // Age: 150ms < 200ms threshold
        
        // Signal is stale beyond threshold
        assert!(signal.is_stale(1400.0, 200.0)); // Age: 350ms > 200ms threshold
    }

    #[test]
    fn test_node_signals() {
        let mut node_signals = NodeSignals::new(1);
        
        let signal = Signal {
            signal_type: SignalType::QueueDepth,
            node_id: 1,
            value: 10.0,
            timestamp: 1000.0,
            transport_delay: 25.0,
        };

        node_signals.update_signal(signal);

        // Fresh signal should be available
        assert_eq!(node_signals.get_signal(SignalType::QueueDepth, 1100.0, 200.0), Some(10.0));
        
        // Stale signal should not be available
        assert_eq!(node_signals.get_signal(SignalType::QueueDepth, 1500.0, 200.0), None);
    }

    #[test]
    fn test_signal_view() {
        let mut view = SignalView::new(1000.0, 200.0);
        
        let mut node_signals = NodeSignals::new(1);
        let signal = Signal {
            signal_type: SignalType::VramUsage,
            node_id: 1,
            value: 0.75,
            timestamp: 950.0,
            transport_delay: 30.0,
        };
        node_signals.update_signal(signal);
        
        view.update_node_signals(node_signals);

        // Should be able to get fresh signal
        assert_eq!(view.get_signal(1, SignalType::VramUsage), Some(0.75));
        
        // Update time to make signal stale
        view.update_time(1300.0);
        assert_eq!(view.get_signal(1, SignalType::VramUsage), None);
    }

    #[test]
    fn test_signal_generator() {
        let config = SignalConfig {
            queue_depth_ms: UpdateFrequency { min: 50.0, max: 100.0 },
            vram_ms: UpdateFrequency { min: 200.0, max: 500.0 },
            p95_ms: UpdateFrequency { min: 1000.0, max: 2000.0 },
            transport_ms: TransportDelayConfig {
                intra_cell: [5.0, 50.0],
                inter_cell: [50.0, 300.0],
            },
        };

        let mut generator = SignalGenerator::new(config);

        // First update should be needed
        assert!(generator.needs_update(1, SignalType::QueueDepth, 1000.0));

        // Schedule next update
        let next_update = generator.schedule_next_update(1, SignalType::QueueDepth, 1000.0);
        assert!(next_update > 1000.0);
        assert!(next_update <= 1100.0); // Within max frequency

        // Should not need update before scheduled time
        assert!(!generator.needs_update(1, SignalType::QueueDepth, next_update - 1.0));
        assert!(generator.needs_update(1, SignalType::QueueDepth, next_update));
    }

    #[test]
    fn test_signal_aggregator() {
        let signals = vec![
            (1, 10.0),
            (2, 20.0),
            (3, 30.0),
            (4, 40.0),
        ];

        assert_eq!(SignalAggregator::average(&signals), Some(25.0));
        assert_eq!(SignalAggregator::percentile(&signals, 50.0), Some(20.0));
        assert_eq!(SignalAggregator::min(&signals), Some((1, 10.0)));
        assert_eq!(SignalAggregator::max(&signals), Some((4, 40.0)));

        // Test weighted average
        let mut weights = HashMap::new();
        weights.insert(1, 1.0);
        weights.insert(2, 2.0);
        weights.insert(3, 3.0);
        weights.insert(4, 4.0);

        let weighted_avg = SignalAggregator::weighted_average(&signals, &weights);
        // (10*1 + 20*2 + 30*3 + 40*4) / (1+2+3+4) = 300/10 = 30.0
        assert_eq!(weighted_avg, Some(30.0));
    }

    #[test]
    fn test_freshness_stats() {
        let mut view = SignalView::new(1000.0, 100.0);
        
        // Add some fresh and stale signals
        let mut node1_signals = NodeSignals::new(1);
        node1_signals.update_signal(Signal {
            signal_type: SignalType::QueueDepth,
            node_id: 1,
            value: 5.0,
            timestamp: 950.0, // Fresh
            transport_delay: 10.0,
        });
        node1_signals.update_signal(Signal {
            signal_type: SignalType::VramUsage,
            node_id: 1,
            value: 0.8,
            timestamp: 800.0, // Stale
            transport_delay: 10.0,
        });
        
        view.update_node_signals(node1_signals);
        
        let stats = view.get_freshness_stats();
        assert_eq!(stats.total_signals, 2);
        assert_eq!(stats.fresh_signals, 1);
        assert_eq!(stats.stale_signals, 1);
        assert_eq!(stats.freshness_ratio, 0.5);
    }
}
