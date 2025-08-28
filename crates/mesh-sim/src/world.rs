use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::engine::{NodeId, ModelId};
use crate::gpu::{GpuProfile, MigProfile};
use crate::net::NetworkConfig;

/// Unique identifier for cells
pub type CellId = u32;

/// Configuration for the world topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConfig {
    pub cells: u32,
    pub nodes_per_cell: u32,
    pub gpu_profiles: Vec<GpuProfile>,
    pub mig: Option<MigConfig>,
}

/// MIG (Multi-Instance GPU) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigConfig {
    pub enable: bool,
    pub profiles: Vec<MigProfile>,
}

/// A cell in the mesh topology
#[derive(Debug, Clone)]
pub struct Cell {
    pub id: CellId,
    pub nodes: Vec<NodeId>,
    /// Vivaldi coordinates for network distance calculation
    pub coordinates: Vec<f64>,
}

/// A compute node in the mesh
#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub cell_id: CellId,
    pub gpu_profile: GpuProfile,
    pub mig_instances: Vec<MigInstance>,
    /// Models currently loaded on this node
    pub loaded_models: HashMap<ModelId, ModelInfo>,
    /// Current state of the node
    pub state: NodeState,
}

/// MIG instance on a node
#[derive(Debug, Clone)]
pub struct MigInstance {
    pub profile: MigProfile,
    pub instance_id: u32,
    pub loaded_models: HashMap<ModelId, ModelInfo>,
    pub state: NodeState,
}

/// Information about a model loaded on a node
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_id: ModelId,
    pub vram_usage_gb: f64,
    pub load_time: f64,
    pub last_used: f64,
}

/// Current state of a node or MIG instance
#[derive(Debug, Clone)]
pub struct NodeState {
    /// Current queue depth per model
    pub queue_depths: HashMap<ModelId, u32>,
    /// Current VRAM usage in GB
    pub vram_used_gb: f64,
    /// Available VRAM in GB
    pub vram_available_gb: f64,
    /// Current utilization (0.0 to 1.0)
    pub utilization: f64,
    /// Recent p95 latency in ms
    pub p95_latency_ms: f64,
    /// Number of active batches
    pub active_batches: u32,
    /// Last time batch window was opened
    pub batch_open_since: Option<f64>,
    /// Requests currently being processed
    pub processing_requests: Vec<crate::engine::RequestId>,
}

impl NodeState {
    pub fn new(vram_total_gb: f64) -> Self {
        Self {
            queue_depths: HashMap::new(),
            vram_used_gb: 0.0,
            vram_available_gb: vram_total_gb,
            utilization: 0.0,
            p95_latency_ms: 0.0,
            active_batches: 0,
            batch_open_since: None,
            processing_requests: Vec::new(),
        }
    }

    /// Get queue depth for a specific model
    pub fn queue_depth(&self, model_id: &ModelId) -> u32 {
        self.queue_depths.get(model_id).copied().unwrap_or(0)
    }

    /// Add a request to the queue
    pub fn add_to_queue(&mut self, model_id: &ModelId) {
        *self.queue_depths.entry(model_id.clone()).or_insert(0) += 1;
    }

    /// Remove a request from the queue
    pub fn remove_from_queue(&mut self, model_id: &ModelId) {
        if let Some(depth) = self.queue_depths.get_mut(model_id) {
            if *depth > 0 {
                *depth -= 1;
            }
        }
    }

    /// Check if the node can handle more concurrent requests
    pub fn can_accept_batch(&self, max_concurrency: u32) -> bool {
        self.active_batches < max_concurrency
    }

    /// Calculate VRAM pressure (0.0 to 1.0)
    pub fn vram_pressure(&self) -> f64 {
        if self.vram_available_gb + self.vram_used_gb == 0.0 {
            0.0
        } else {
            self.vram_used_gb / (self.vram_available_gb + self.vram_used_gb)
        }
    }

    /// Calculate work left (sum of queue depths)
    pub fn work_left(&self) -> u32 {
        self.queue_depths.values().sum()
    }
}

/// The world contains the complete topology and state
#[derive(Debug, Clone)]
pub struct World {
    pub cells: HashMap<CellId, Cell>,
    pub nodes: HashMap<NodeId, Node>,
    pub network: NetworkConfig,
    next_node_id: NodeId,
}

impl World {
    /// Create a new world from configuration
    pub fn new(config: TopologyConfig, network: NetworkConfig) -> Self {
        let mut world = Self {
            cells: HashMap::new(),
            nodes: HashMap::new(),
            network,
            next_node_id: 1,
        };

        world.build_topology(config);
        world
    }

    /// Build the topology from configuration
    fn build_topology(&mut self, config: TopologyConfig) {
        // Create cells with Vivaldi coordinates
        for cell_id in 0..config.cells {
            let coordinates = self.generate_vivaldi_coordinates();
            let cell = Cell {
                id: cell_id,
                nodes: Vec::new(),
                coordinates,
            };
            self.cells.insert(cell_id, cell);
        }

        // Create nodes and distribute them across cells
        for cell_id in 0..config.cells {
            let mut node_ids = Vec::new();
            
            for _ in 0..config.nodes_per_cell {
                let node_id = self.next_node_id;
                self.next_node_id += 1;

                // Select GPU profile (for now, use first one)
                // TODO: Implement more sophisticated GPU profile selection
                let gpu_profile = config.gpu_profiles[0].clone();

                // Create MIG instances if enabled
                let mig_instances = if let Some(ref mig_config) = config.mig {
                    if mig_config.enable {
                        mig_config.profiles.iter().enumerate().map(|(i, profile)| {
                            MigInstance {
                                profile: profile.clone(),
                                instance_id: i as u32,
                                loaded_models: HashMap::new(),
                                state: NodeState::new(profile.vram_total_gb()),
                            }
                        }).collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };

                let node = Node {
                    id: node_id,
                    cell_id,
                    gpu_profile: gpu_profile.clone(),
                    mig_instances,
                    loaded_models: HashMap::new(),
                    state: NodeState::new(gpu_profile.vram_total_gb),
                };

                node_ids.push(node_id);
                self.nodes.insert(node_id, node);
            }

            // Update cell with node IDs
            if let Some(cell) = self.cells.get_mut(&cell_id) {
                cell.nodes = node_ids;
            }
        }
    }

    /// Generate Vivaldi coordinates for a cell
    fn generate_vivaldi_coordinates(&self) -> Vec<f64> {
        // TODO: Implement proper Vivaldi coordinate generation
        // For now, generate random 3D coordinates
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..3).map(|_| rng.gen_range(-100.0..100.0)).collect()
    }

    /// Get a node by ID
    pub fn get_node(&self, node_id: NodeId) -> Option<&Node> {
        self.nodes.get(&node_id)
    }

    /// Get a mutable reference to a node by ID
    pub fn get_node_mut(&mut self, node_id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&node_id)
    }

    /// Get a cell by ID
    pub fn get_cell(&self, cell_id: CellId) -> Option<&Cell> {
        self.cells.get(&cell_id)
    }

    /// Get all nodes in a cell
    pub fn get_nodes_in_cell(&self, cell_id: CellId) -> Vec<&Node> {
        if let Some(cell) = self.cells.get(&cell_id) {
            cell.nodes.iter()
                .filter_map(|&node_id| self.nodes.get(&node_id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get all nodes that have a specific model loaded
    pub fn get_nodes_with_model(&self, model_id: &ModelId) -> Vec<&Node> {
        self.nodes.values()
            .filter(|node| node.loaded_models.contains_key(model_id))
            .collect()
    }

    /// Get all nodes in the same cell that have a specific model loaded
    pub fn get_local_nodes_with_model(&self, cell_id: CellId, model_id: &ModelId) -> Vec<&Node> {
        self.get_nodes_in_cell(cell_id)
            .into_iter()
            .filter(|node| node.loaded_models.contains_key(model_id))
            .collect()
    }

    /// Calculate network latency between two nodes
    pub fn network_latency(&self, from_node: NodeId, to_node: NodeId) -> f64 {
        let from_node = match self.nodes.get(&from_node) {
            Some(node) => node,
            None => return 0.0,
        };
        let to_node = match self.nodes.get(&to_node) {
            Some(node) => node,
            None => return 0.0,
        };

        if from_node.cell_id == to_node.cell_id {
            // Intra-cell latency
            self.network.sample_intra_cell_rtt()
        } else {
            // Inter-cell latency using Vivaldi coordinates
            let from_cell = self.cells.get(&from_node.cell_id).unwrap();
            let to_cell = self.cells.get(&to_node.cell_id).unwrap();
            self.network.calculate_inter_cell_rtt(&from_cell.coordinates, &to_cell.coordinates)
        }
    }

    /// Get total number of nodes
    pub fn total_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get total number of cells
    pub fn total_cells(&self) -> usize {
        self.cells.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuProfile;
    use crate::net::NetworkConfig;

    fn create_test_config() -> TopologyConfig {
        TopologyConfig {
            cells: 2,
            nodes_per_cell: 4,
            gpu_profiles: vec![
                GpuProfile {
                    name: "H100-80G".to_string(),
                    tokens_per_s: 240000,
                    concurrency: 16,
                    vram_total_gb: 80.0,
                    batch_window_ms: 8.0,
                    kv_cache_gb_per_req: 1.2,
                }
            ],
            mig: None,
        }
    }

    #[test]
    fn test_world_creation() {
        let config = create_test_config();
        let network = NetworkConfig::default();
        let world = World::new(config, network);

        assert_eq!(world.total_cells(), 2);
        assert_eq!(world.total_nodes(), 8);
    }

    #[test]
    fn test_node_state() {
        let mut state = NodeState::new(80.0);
        assert_eq!(state.vram_available_gb, 80.0);
        assert_eq!(state.queue_depth(&"test-model".to_string()), 0);

        state.add_to_queue(&"test-model".to_string());
        assert_eq!(state.queue_depth(&"test-model".to_string()), 1);

        state.remove_from_queue(&"test-model".to_string());
        assert_eq!(state.queue_depth(&"test-model".to_string()), 0);
    }
}
