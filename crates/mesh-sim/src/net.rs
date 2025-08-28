use serde::{Deserialize, Serialize};
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub intra_cell_rtt_ms: LatencyDistribution,
    pub inter_cell_coords: VivalidoConfig,
    pub bw_mbps: BandwidthConfig,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            intra_cell_rtt_ms: LatencyDistribution::Normal { mean: 0.5, std: 0.1 },
            inter_cell_coords: VivalidoConfig {
                dim: 3,
                base_rtt_ms: 25.0,
                noise: 0.1,
            },
            bw_mbps: BandwidthConfig {
                intra_cell: 50000,
                inter_region: 5000,
            },
        }
    }
}

/// Latency distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "dist")]
pub enum LatencyDistribution {
    #[serde(rename = "normal")]
    Normal { mean: f64, std: f64 },
    #[serde(rename = "uniform")]
    Uniform { min: f64, max: f64 },
    #[serde(rename = "constant")]
    Constant { value: f64 },
}

impl LatencyDistribution {
    /// Sample a latency value from this distribution
    pub fn sample(&self) -> f64 {
        let mut rng = rand::thread_rng();
        match self {
            LatencyDistribution::Normal { mean, std } => {
                let normal = Normal::new(*mean, *std).unwrap();
                normal.sample(&mut rng).max(0.0) // Ensure non-negative
            }
            LatencyDistribution::Uniform { min, max } => {
                rng.gen_range(*min..*max)
            }
            LatencyDistribution::Constant { value } => *value,
        }
    }
}

/// Vivaldi coordinate system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VivalidoConfig {
    /// Dimensionality of the coordinate space
    pub dim: usize,
    /// Base RTT in milliseconds
    pub base_rtt_ms: f64,
    /// Noise factor (0.0 to 1.0)
    pub noise: f64,
}

/// Bandwidth configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthConfig {
    /// Intra-cell bandwidth in Mbps
    pub intra_cell: u32,
    /// Inter-region bandwidth in Mbps
    pub inter_region: u32,
}

impl NetworkConfig {
    /// Sample intra-cell RTT
    pub fn sample_intra_cell_rtt(&self) -> f64 {
        self.intra_cell_rtt_ms.sample()
    }

    /// Calculate inter-cell RTT using Vivaldi coordinates
    pub fn calculate_inter_cell_rtt(&self, coords1: &[f64], coords2: &[f64]) -> f64 {
        if coords1.len() != coords2.len() || coords1.len() != self.inter_cell_coords.dim {
            return self.inter_cell_coords.base_rtt_ms;
        }

        // Calculate Euclidean distance
        let distance: f64 = coords1.iter()
            .zip(coords2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        // Convert distance to RTT with base latency and noise
        let base_rtt = self.inter_cell_coords.base_rtt_ms;
        let distance_rtt = distance * 0.1; // Scale factor for distance to RTT
        
        // Add noise
        let mut rng = rand::thread_rng();
        let noise_factor = 1.0 + (rng.gen::<f64>() - 0.5) * 2.0 * self.inter_cell_coords.noise;
        
        (base_rtt + distance_rtt) * noise_factor
    }

    /// Get bandwidth between two locations
    pub fn get_bandwidth(&self, same_cell: bool, same_region: bool) -> u32 {
        if same_cell {
            self.bw_mbps.intra_cell
        } else if same_region {
            // Assume inter-cell but same region has intermediate bandwidth
            (self.bw_mbps.intra_cell + self.bw_mbps.inter_region) / 2
        } else {
            self.bw_mbps.inter_region
        }
    }

    /// Calculate transfer time for a given payload size
    pub fn calculate_transfer_time(&self, payload_bytes: u64, bandwidth_mbps: u32) -> f64 {
        if bandwidth_mbps == 0 {
            return 0.0;
        }
        
        // Convert to bits and calculate time in seconds, then to milliseconds
        let payload_bits = payload_bytes * 8;
        let bandwidth_bps = bandwidth_mbps as f64 * 1_000_000.0;
        (payload_bits as f64 / bandwidth_bps) * 1000.0
    }
}

/// Network topology for routing decisions
#[derive(Debug, Clone)]
pub struct NetworkTopology {
    /// Vivaldi coordinates for each cell
    pub cell_coordinates: std::collections::HashMap<u32, Vec<f64>>,
    /// Cached RTT matrix between cells
    pub rtt_matrix: std::collections::HashMap<(u32, u32), f64>,
    /// Network configuration
    pub config: NetworkConfig,
}

impl NetworkTopology {
    /// Create a new network topology
    pub fn new(config: NetworkConfig) -> Self {
        Self {
            cell_coordinates: std::collections::HashMap::new(),
            rtt_matrix: std::collections::HashMap::new(),
            config,
        }
    }

    /// Add a cell with generated Vivaldi coordinates
    pub fn add_cell(&mut self, cell_id: u32) -> Vec<f64> {
        let coordinates = self.generate_vivaldi_coordinates();
        self.cell_coordinates.insert(cell_id, coordinates.clone());
        
        // Update RTT matrix for this cell
        self.update_rtt_matrix_for_cell(cell_id);
        
        coordinates
    }

    /// Generate Vivaldi coordinates for a new cell
    fn generate_vivaldi_coordinates(&self) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        (0..self.config.inter_cell_coords.dim)
            .map(|_| rng.gen_range(-100.0..100.0))
            .collect()
    }

    /// Update RTT matrix when a new cell is added
    fn update_rtt_matrix_for_cell(&mut self, new_cell_id: u32) {
        let new_coords = self.cell_coordinates.get(&new_cell_id).unwrap().clone();
        
        for (&other_cell_id, other_coords) in &self.cell_coordinates {
            if other_cell_id != new_cell_id {
                let rtt = self.config.calculate_inter_cell_rtt(&new_coords, other_coords);
                self.rtt_matrix.insert((new_cell_id, other_cell_id), rtt);
                self.rtt_matrix.insert((other_cell_id, new_cell_id), rtt);
            }
        }
        
        // Self RTT is 0
        self.rtt_matrix.insert((new_cell_id, new_cell_id), 0.0);
    }

    /// Get RTT between two cells
    pub fn get_cell_rtt(&self, cell1: u32, cell2: u32) -> f64 {
        if cell1 == cell2 {
            return 0.0;
        }
        
        self.rtt_matrix.get(&(cell1, cell2))
            .copied()
            .unwrap_or_else(|| {
                // Calculate on-demand if not cached
                if let (Some(coords1), Some(coords2)) = (
                    self.cell_coordinates.get(&cell1),
                    self.cell_coordinates.get(&cell2)
                ) {
                    self.config.calculate_inter_cell_rtt(coords1, coords2)
                } else {
                    self.config.inter_cell_coords.base_rtt_ms
                }
            })
    }

    /// Get all cells sorted by RTT from a source cell
    pub fn get_cells_by_distance(&self, source_cell: u32) -> Vec<(u32, f64)> {
        let mut cells_with_rtt: Vec<_> = self.cell_coordinates.keys()
            .map(|&cell_id| (cell_id, self.get_cell_rtt(source_cell, cell_id)))
            .collect();
        
        cells_with_rtt.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        cells_with_rtt
    }
}

/// Network penalty calculation for routing decisions
pub struct NetworkPenalty;

impl NetworkPenalty {
    /// Calculate network penalty for routing to a target
    pub fn calculate(
        source_cell: u32,
        target_cell: u32,
        topology: &NetworkTopology,
        request_size_bytes: u64,
    ) -> f64 {
        if source_cell == target_cell {
            // Intra-cell routing - minimal penalty
            return topology.config.sample_intra_cell_rtt();
        }

        // Inter-cell routing
        let rtt = topology.get_cell_rtt(source_cell, target_cell);
        
        // Add transfer time if request is large
        let bandwidth = topology.config.get_bandwidth(false, true); // Assume same region
        let transfer_time = topology.config.calculate_transfer_time(request_size_bytes, bandwidth);
        
        rtt + transfer_time
    }

    /// Calculate penalty for cross-region routing
    pub fn calculate_cross_region(
        topology: &NetworkTopology,
        request_size_bytes: u64,
    ) -> f64 {
        let base_rtt = topology.config.inter_cell_coords.base_rtt_ms * 2.0; // Cross-region penalty
        let bandwidth = topology.config.bw_mbps.inter_region;
        let transfer_time = topology.config.calculate_transfer_time(request_size_bytes, bandwidth);
        
        base_rtt + transfer_time
    }
}

/// Signal transport delay model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalTransportConfig {
    /// Intra-cell transport delay range [min, max] in ms
    pub intra_cell: [f64; 2],
    /// Inter-cell transport delay range [min, max] in ms
    pub inter_cell: [f64; 2],
}

impl SignalTransportConfig {
    /// Sample transport delay for signals
    pub fn sample_delay(&self, same_cell: bool) -> f64 {
        let mut rng = rand::thread_rng();
        let range = if same_cell { self.intra_cell } else { self.inter_cell };
        rng.gen_range(range[0]..range[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_distribution() {
        let normal_dist = LatencyDistribution::Normal { mean: 5.0, std: 1.0 };
        let uniform_dist = LatencyDistribution::Uniform { min: 1.0, max: 10.0 };
        let constant_dist = LatencyDistribution::Constant { value: 3.0 };

        // Test sampling (just ensure no panics and reasonable values)
        for _ in 0..100 {
            let normal_sample = normal_dist.sample();
            let uniform_sample = uniform_dist.sample();
            let constant_sample = constant_dist.sample();

            assert!(normal_sample >= 0.0);
            assert!(uniform_sample >= 1.0 && uniform_sample < 10.0);
            assert_eq!(constant_sample, 3.0);
        }
    }

    #[test]
    fn test_vivaldi_rtt_calculation() {
        let config = NetworkConfig::default();
        
        let coords1 = vec![0.0, 0.0, 0.0];
        let coords2 = vec![10.0, 10.0, 10.0];
        
        let rtt = config.calculate_inter_cell_rtt(&coords1, &coords2);
        
        // Should be base RTT plus some distance-based component
        assert!(rtt > config.inter_cell_coords.base_rtt_ms);
    }

    #[test]
    fn test_network_topology() {
        let config = NetworkConfig::default();
        let mut topology = NetworkTopology::new(config);
        
        // Add some cells
        topology.add_cell(0);
        topology.add_cell(1);
        topology.add_cell(2);
        
        // Test RTT calculations
        let rtt_01 = topology.get_cell_rtt(0, 1);
        let rtt_02 = topology.get_cell_rtt(0, 2);
        let rtt_self = topology.get_cell_rtt(0, 0);
        
        assert!(rtt_01 > 0.0);
        assert!(rtt_02 > 0.0);
        assert_eq!(rtt_self, 0.0);
        
        // Test distance sorting
        let cells_by_distance = topology.get_cells_by_distance(0);
        assert_eq!(cells_by_distance.len(), 3);
        assert_eq!(cells_by_distance[0].0, 0); // Self should be first
        assert_eq!(cells_by_distance[0].1, 0.0); // With 0 RTT
    }

    #[test]
    fn test_bandwidth_calculation() {
        let config = NetworkConfig::default();
        
        // Test different bandwidth scenarios
        assert_eq!(config.get_bandwidth(true, true), 50000); // Same cell
        assert_eq!(config.get_bandwidth(false, false), 5000); // Cross-region
        
        // Test transfer time calculation
        let transfer_time = config.calculate_transfer_time(1_000_000, 1000); // 1MB at 1Gbps
        assert!((transfer_time - 8.0).abs() < 0.1); // Should be ~8ms
    }

    #[test]
    fn test_network_penalty() {
        let config = NetworkConfig::default();
        let mut topology = NetworkTopology::new(config);
        
        topology.add_cell(0);
        topology.add_cell(1);
        
        let intra_penalty = NetworkPenalty::calculate(0, 0, &topology, 1000);
        let inter_penalty = NetworkPenalty::calculate(0, 1, &topology, 1000);
        
        assert!(inter_penalty > intra_penalty);
    }
}
