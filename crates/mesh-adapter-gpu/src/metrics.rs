//! GPU metrics and telemetry data structures

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Complete GPU metrics for a single GPU
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// GPU information
    pub info: GpuInfo,
    
    /// Current GPU status
    pub status: GpuStatus,
    
    /// Memory information
    pub memory: MemoryInfo,
    
    /// Temperature information
    pub temperature: TemperatureInfo,
    
    /// Power information
    pub power: PowerInfo,
    
    /// Clock speeds
    pub clocks: ClockInfo,
    
    /// Utilization metrics
    pub utilization: UtilizationInfo,
    
    /// Fan information
    pub fans: Vec<FanInfo>,
    
    /// MIG information (if available)
    pub mig: Option<MigInfo>,
    
    /// ECC error information
    pub ecc: Option<EccInfo>,
    
    /// Performance state
    pub performance_state: Option<PerformanceState>,
    
    /// Processes running on GPU
    pub processes: Vec<GpuProcess>,
    
    /// Timestamp when metrics were collected
    pub timestamp: SystemTime,
    
    /// Collection duration
    pub collection_duration: Duration,
}

/// Basic GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU index
    pub index: u32,
    
    /// GPU UUID
    pub uuid: String,
    
    /// GPU name/model
    pub name: String,
    
    /// GPU brand (NVIDIA, AMD, etc.)
    pub brand: String,
    
    /// GPU architecture
    pub architecture: Option<String>,
    
    /// Driver version
    pub driver_version: String,
    
    /// VBIOS version
    pub vbios_version: Option<String>,
    
    /// PCI bus information
    pub pci_info: PciInfo,
    
    /// GPU capabilities
    pub capabilities: GpuCapabilities,
    
    /// Total memory in bytes
    pub total_memory: u64,
    
    /// Memory bus width in bits
    pub memory_bus_width: Option<u32>,
    
    /// Memory type (GDDR6, HBM2, etc.)
    pub memory_type: Option<String>,
}

/// GPU status information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GpuStatus {
    /// GPU is active and available
    Active,
    /// GPU is idle
    Idle,
    /// GPU is in use
    InUse,
    /// GPU is throttling due to temperature
    Throttling,
    /// GPU has encountered an error
    Error(String),
    /// GPU is not available
    Unavailable,
    /// GPU is in maintenance mode
    Maintenance,
}

/// Memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Total memory in bytes
    pub total: u64,
    
    /// Used memory in bytes
    pub used: u64,
    
    /// Free memory in bytes
    pub free: u64,
    
    /// Memory utilization percentage (0-100)
    pub utilization: f64,
    
    /// Memory bandwidth utilization percentage (0-100)
    pub bandwidth_utilization: Option<f64>,
    
    /// Memory clock speed in MHz
    pub clock_speed: Option<u32>,
    
    /// Memory temperature in Celsius
    pub temperature: Option<f64>,
}

/// Temperature information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureInfo {
    /// GPU core temperature in Celsius
    pub gpu: f64,
    
    /// Memory temperature in Celsius (if available)
    pub memory: Option<f64>,
    
    /// Hotspot temperature in Celsius (if available)
    pub hotspot: Option<f64>,
    
    /// Temperature threshold for throttling
    pub throttle_threshold: Option<f64>,
    
    /// Temperature threshold for shutdown
    pub shutdown_threshold: Option<f64>,
    
    /// Current thermal state
    pub thermal_state: ThermalState,
}

/// Thermal state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ThermalState {
    Normal,
    Warning,
    Critical,
    Throttling,
    Shutdown,
}

/// Power information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerInfo {
    /// Current power usage in watts
    pub usage: f64,
    
    /// Power limit in watts
    pub limit: f64,
    
    /// Default power limit in watts
    pub default_limit: Option<f64>,
    
    /// Maximum power limit in watts
    pub max_limit: Option<f64>,
    
    /// Minimum power limit in watts
    pub min_limit: Option<f64>,
    
    /// Power utilization percentage (0-100)
    pub utilization: f64,
    
    /// Power state
    pub state: PowerState,
}

/// Power state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PowerState {
    P0, // Maximum performance
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    P7,
    P8, // Minimum performance
    Unknown,
}

/// Clock information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClockInfo {
    /// Graphics clock speed in MHz
    pub graphics: u32,
    
    /// Memory clock speed in MHz
    pub memory: u32,
    
    /// SM (Streaming Multiprocessor) clock speed in MHz
    pub sm: Option<u32>,
    
    /// Video clock speed in MHz
    pub video: Option<u32>,
    
    /// Maximum graphics clock speed in MHz
    pub max_graphics: Option<u32>,
    
    /// Maximum memory clock speed in MHz
    pub max_memory: Option<u32>,
    
    /// Base graphics clock speed in MHz
    pub base_graphics: Option<u32>,
    
    /// Base memory clock speed in MHz
    pub base_memory: Option<u32>,
}

/// Utilization information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilizationInfo {
    /// GPU utilization percentage (0-100)
    pub gpu: f64,
    
    /// Memory utilization percentage (0-100)
    pub memory: f64,
    
    /// Encoder utilization percentage (0-100)
    pub encoder: Option<f64>,
    
    /// Decoder utilization percentage (0-100)
    pub decoder: Option<f64>,
    
    /// JPEG utilization percentage (0-100)
    pub jpeg: Option<f64>,
    
    /// OFA (Optical Flow Accelerator) utilization percentage (0-100)
    pub ofa: Option<f64>,
}

/// Fan information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FanInfo {
    /// Fan index
    pub index: u32,
    
    /// Fan speed in RPM
    pub speed_rpm: u32,
    
    /// Fan speed percentage (0-100)
    pub speed_percent: f64,
    
    /// Target fan speed percentage
    pub target_speed: Option<f64>,
    
    /// Fan control mode
    pub control_mode: FanControlMode,
}

/// Fan control mode
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FanControlMode {
    Auto,
    Manual,
    Unknown,
}

/// MIG (Multi-Instance GPU) information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigInfo {
    /// MIG mode enabled
    pub enabled: bool,
    
    /// MIG instances
    pub instances: Vec<MigInstance>,
    
    /// Available MIG profiles
    pub available_profiles: Vec<MigProfile>,
}

/// MIG instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigInstance {
    /// Instance ID
    pub id: u32,
    
    /// Instance UUID
    pub uuid: String,
    
    /// Profile name
    pub profile: String,
    
    /// Memory size in MB
    pub memory_size: u64,
    
    /// GPU slice count
    pub gpu_slice_count: u32,
    
    /// Compute instance count
    pub compute_instance_count: u32,
    
    /// Current utilization
    pub utilization: f64,
}

/// MIG profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigProfile {
    /// Profile ID
    pub id: u32,
    
    /// Profile name
    pub name: String,
    
    /// Memory size in MB
    pub memory_size: u64,
    
    /// GPU slice count
    pub gpu_slice_count: u32,
    
    /// Compute instance count
    pub compute_instance_count: u32,
}

/// ECC (Error Correcting Code) information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EccInfo {
    /// ECC support available
    pub supported: bool,
    
    /// ECC currently enabled
    pub enabled: bool,
    
    /// Single-bit ECC errors
    pub single_bit_errors: EccErrors,
    
    /// Double-bit ECC errors
    pub double_bit_errors: EccErrors,
}

/// ECC error counts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EccErrors {
    /// Volatile (current session) errors
    pub volatile: u64,
    
    /// Aggregate (lifetime) errors
    pub aggregate: u64,
}

/// Performance state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PerformanceState {
    P0,  // Maximum performance
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    P7,
    P8,
    P9,
    P10,
    P11,
    P12, // Minimum performance
    Unknown,
}

/// Process running on GPU
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProcess {
    /// Process ID
    pub pid: u32,
    
    /// Process name
    pub name: String,
    
    /// Memory usage in bytes
    pub memory_usage: u64,
    
    /// GPU utilization percentage
    pub gpu_utilization: Option<f64>,
    
    /// Encoder utilization percentage
    pub encoder_utilization: Option<f64>,
    
    /// Decoder utilization percentage
    pub decoder_utilization: Option<f64>,
    
    /// Process type
    pub process_type: ProcessType,
}

/// GPU process type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProcessType {
    Compute,
    Graphics,
    ComputeAndGraphics,
    Unknown,
}

/// PCI information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PciInfo {
    /// PCI bus ID
    pub bus_id: String,
    
    /// PCI domain
    pub domain: u32,
    
    /// PCI bus
    pub bus: u32,
    
    /// PCI device
    pub device: u32,
    
    /// PCI function
    pub function: u32,
    
    /// Device ID
    pub device_id: u32,
    
    /// Subsystem ID
    pub subsystem_id: u32,
    
    /// Link generation
    pub link_gen: Option<u32>,
    
    /// Link width
    pub link_width: Option<u32>,
    
    /// Maximum link generation
    pub max_link_gen: Option<u32>,
    
    /// Maximum link width
    pub max_link_width: Option<u32>,
}

/// GPU capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCapabilities {
    /// CUDA compute capability major version
    pub cuda_compute_major: Option<u32>,
    
    /// CUDA compute capability minor version
    pub cuda_compute_minor: Option<u32>,
    
    /// Multi-GPU board
    pub multi_gpu_board: bool,
    
    /// MIG support
    pub mig_support: bool,
    
    /// ECC support
    pub ecc_support: bool,
    
    /// TCC mode support (Windows)
    pub tcc_support: bool,
    
    /// Virtualization support
    pub virtualization_support: bool,
    
    /// Supported graphics APIs
    pub graphics_apis: Vec<String>,
    
    /// Supported compute APIs
    pub compute_apis: Vec<String>,
}

impl GpuMetrics {
    /// Create new GPU metrics
    pub fn new(info: GpuInfo) -> Self {
        Self {
            info,
            status: GpuStatus::Active,
            memory: MemoryInfo::default(),
            temperature: TemperatureInfo::default(),
            power: PowerInfo::default(),
            clocks: ClockInfo::default(),
            utilization: UtilizationInfo::default(),
            fans: Vec::new(),
            mig: None,
            ecc: None,
            performance_state: None,
            processes: Vec::new(),
            timestamp: SystemTime::now(),
            collection_duration: Duration::from_millis(0),
        }
    }

    /// Check if GPU is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(self.status, GpuStatus::Active | GpuStatus::Idle | GpuStatus::InUse)
            && self.temperature.thermal_state != ThermalState::Critical
            && self.temperature.thermal_state != ThermalState::Shutdown
    }

    /// Get memory utilization percentage
    pub fn memory_utilization(&self) -> f64 {
        self.memory.utilization
    }

    /// Get GPU utilization percentage
    pub fn gpu_utilization(&self) -> f64 {
        self.utilization.gpu
    }

    /// Get power utilization percentage
    pub fn power_utilization(&self) -> f64 {
        self.power.utilization
    }

    /// Get GPU temperature
    pub fn temperature(&self) -> f64 {
        self.temperature.gpu
    }
}

impl Default for MemoryInfo {
    fn default() -> Self {
        Self {
            total: 0,
            used: 0,
            free: 0,
            utilization: 0.0,
            bandwidth_utilization: None,
            clock_speed: None,
            temperature: None,
        }
    }
}

impl Default for TemperatureInfo {
    fn default() -> Self {
        Self {
            gpu: 0.0,
            memory: None,
            hotspot: None,
            throttle_threshold: None,
            shutdown_threshold: None,
            thermal_state: ThermalState::Normal,
        }
    }
}

impl Default for PowerInfo {
    fn default() -> Self {
        Self {
            usage: 0.0,
            limit: 0.0,
            default_limit: None,
            max_limit: None,
            min_limit: None,
            utilization: 0.0,
            state: PowerState::Unknown,
        }
    }
}

impl Default for ClockInfo {
    fn default() -> Self {
        Self {
            graphics: 0,
            memory: 0,
            sm: None,
            video: None,
            max_graphics: None,
            max_memory: None,
            base_graphics: None,
            base_memory: None,
        }
    }
}

impl Default for UtilizationInfo {
    fn default() -> Self {
        Self {
            gpu: 0.0,
            memory: 0.0,
            encoder: None,
            decoder: None,
            jpeg: None,
            ofa: None,
        }
    }
}

impl Default for GpuCapabilities {
    fn default() -> Self {
        Self {
            cuda_compute_major: None,
            cuda_compute_minor: None,
            multi_gpu_board: false,
            mig_support: false,
            ecc_support: false,
            tcc_support: false,
            virtualization_support: false,
            graphics_apis: Vec::new(),
            compute_apis: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_metrics_creation() {
        let info = GpuInfo {
            index: 0,
            uuid: "GPU-12345".to_string(),
            name: "Test GPU".to_string(),
            brand: "NVIDIA".to_string(),
            architecture: Some("Ampere".to_string()),
            driver_version: "470.0".to_string(),
            vbios_version: Some("90.02.42.00.01".to_string()),
            pci_info: PciInfo {
                bus_id: "0000:01:00.0".to_string(),
                domain: 0,
                bus: 1,
                device: 0,
                function: 0,
                device_id: 0x1234,
                subsystem_id: 0x5678,
                link_gen: Some(3),
                link_width: Some(16),
                max_link_gen: Some(4),
                max_link_width: Some(16),
            },
            capabilities: GpuCapabilities::default(),
            total_memory: 8 * 1024 * 1024 * 1024, // 8GB
            memory_bus_width: Some(256),
            memory_type: Some("GDDR6".to_string()),
        };

        let metrics = GpuMetrics::new(info);
        assert_eq!(metrics.info.index, 0);
        assert_eq!(metrics.info.name, "Test GPU");
        assert!(metrics.is_healthy());
    }

    #[test]
    fn test_gpu_status() {
        assert_eq!(GpuStatus::Active, GpuStatus::Active);
        assert_ne!(GpuStatus::Active, GpuStatus::Idle);
        
        let error_status = GpuStatus::Error("test error".to_string());
        assert!(matches!(error_status, GpuStatus::Error(_)));
    }

    #[test]
    fn test_thermal_state() {
        assert_eq!(ThermalState::Normal, ThermalState::Normal);
        assert_ne!(ThermalState::Normal, ThermalState::Critical);
    }

    #[test]
    fn test_power_state() {
        assert_eq!(PowerState::P0, PowerState::P0);
        assert_ne!(PowerState::P0, PowerState::P8);
    }

    #[test]
    fn test_memory_info_default() {
        let memory = MemoryInfo::default();
        assert_eq!(memory.total, 0);
        assert_eq!(memory.used, 0);
        assert_eq!(memory.utilization, 0.0);
    }

    #[test]
    fn test_utilization_info_default() {
        let utilization = UtilizationInfo::default();
        assert_eq!(utilization.gpu, 0.0);
        assert_eq!(utilization.memory, 0.0);
        assert!(utilization.encoder.is_none());
    }

    #[test]
    fn test_fan_control_mode() {
        assert_eq!(FanControlMode::Auto, FanControlMode::Auto);
        assert_ne!(FanControlMode::Auto, FanControlMode::Manual);
    }

    #[test]
    fn test_process_type() {
        assert_eq!(ProcessType::Compute, ProcessType::Compute);
        assert_ne!(ProcessType::Compute, ProcessType::Graphics);
    }
}
