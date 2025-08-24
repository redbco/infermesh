//! GPU discovery and topology

use crate::metrics::{GpuInfo, PciInfo, GpuCapabilities};
use crate::{Result, GpuError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// GPU discovery interface
pub trait GpuDiscovery {
    /// Discover all available GPUs
    fn discover_gpus(&self) -> Result<Vec<GpuInfo>>;
    
    /// Get GPU topology information
    fn get_topology(&self) -> Result<GpuTopology>;
    
    /// Get GPU by index
    fn get_gpu_info(&self, index: u32) -> Result<GpuInfo>;
}

/// GPU topology information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTopology {
    /// Total number of GPUs
    pub gpu_count: u32,
    
    /// GPU interconnect information
    pub interconnects: Vec<GpuInterconnect>,
    
    /// NUMA topology
    pub numa_nodes: Vec<NumaNode>,
    
    /// CPU affinity information
    pub cpu_affinity: HashMap<u32, Vec<u32>>,
}

/// GPU interconnect information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInterconnect {
    /// Source GPU index
    pub gpu1: u32,
    
    /// Target GPU index
    pub gpu2: u32,
    
    /// Interconnect type
    pub interconnect_type: InterconnectType,
    
    /// Link count
    pub link_count: u32,
    
    /// Bandwidth in GB/s
    pub bandwidth_gbps: f64,
}

/// Interconnect types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InterconnectType {
    /// NVLink
    NvLink,
    /// PCIe
    Pcie,
    /// SXM
    Sxm,
    /// Network (InfiniBand, Ethernet)
    Network,
    /// Unknown
    Unknown,
}

/// NUMA node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaNode {
    /// NUMA node ID
    pub node_id: u32,
    
    /// GPUs in this NUMA node
    pub gpus: Vec<u32>,
    
    /// CPU cores in this NUMA node
    pub cpu_cores: Vec<u32>,
    
    /// Memory size in bytes
    pub memory_size: u64,
}

impl Default for GpuTopology {
    fn default() -> Self {
        Self {
            gpu_count: 0,
            interconnects: Vec::new(),
            numa_nodes: Vec::new(),
            cpu_affinity: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_topology_default() {
        let topology = GpuTopology::default();
        assert_eq!(topology.gpu_count, 0);
        assert!(topology.interconnects.is_empty());
        assert!(topology.numa_nodes.is_empty());
        assert!(topology.cpu_affinity.is_empty());
    }

    #[test]
    fn test_interconnect_type() {
        assert_eq!(InterconnectType::NvLink, InterconnectType::NvLink);
        assert_ne!(InterconnectType::NvLink, InterconnectType::Pcie);
    }
}
