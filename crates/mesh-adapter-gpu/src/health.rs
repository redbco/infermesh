//! GPU health monitoring

use crate::metrics::{GpuMetrics, ThermalState, GpuStatus};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// GPU health status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// GPU is healthy
    Healthy,
    /// GPU has warnings
    Warning(Vec<String>),
    /// GPU is in critical state
    Critical(Vec<String>),
    /// GPU is unavailable
    Unavailable(String),
}

/// Comprehensive GPU health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuHealth {
    /// GPU index
    pub gpu_id: u32,
    
    /// Overall health status
    pub status: HealthStatus,
    
    /// Individual health checks
    pub checks: HashMap<String, HealthCheck>,
    
    /// Health score (0-100)
    pub score: f64,
    
    /// Timestamp of health check
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Individual health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    /// Check name
    pub name: String,
    
    /// Check status
    pub status: CheckStatus,
    
    /// Check value
    pub value: Option<f64>,
    
    /// Threshold value
    pub threshold: Option<f64>,
    
    /// Check message
    pub message: Option<String>,
}

/// Health check status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CheckStatus {
    Pass,
    Warning,
    Critical,
    Unknown,
}

impl GpuHealth {
    /// Create GPU health from metrics
    pub fn from_metrics(metrics: &GpuMetrics) -> Self {
        let mut checks = HashMap::new();
        let mut warnings = Vec::new();
        let mut criticals = Vec::new();

        // Temperature check
        let temp_check = if metrics.temperature.gpu > 90.0 {
            criticals.push(format!("GPU temperature too high: {:.1}°C", metrics.temperature.gpu));
            HealthCheck {
                name: "temperature".to_string(),
                status: CheckStatus::Critical,
                value: Some(metrics.temperature.gpu),
                threshold: Some(90.0),
                message: Some("Temperature critical".to_string()),
            }
        } else if metrics.temperature.gpu > 80.0 {
            warnings.push(format!("GPU temperature high: {:.1}°C", metrics.temperature.gpu));
            HealthCheck {
                name: "temperature".to_string(),
                status: CheckStatus::Warning,
                value: Some(metrics.temperature.gpu),
                threshold: Some(80.0),
                message: Some("Temperature warning".to_string()),
            }
        } else {
            HealthCheck {
                name: "temperature".to_string(),
                status: CheckStatus::Pass,
                value: Some(metrics.temperature.gpu),
                threshold: Some(80.0),
                message: None,
            }
        };
        checks.insert("temperature".to_string(), temp_check);

        // Memory check
        let mem_check = if metrics.memory.utilization > 95.0 {
            criticals.push(format!("GPU memory usage critical: {:.1}%", metrics.memory.utilization));
            HealthCheck {
                name: "memory".to_string(),
                status: CheckStatus::Critical,
                value: Some(metrics.memory.utilization),
                threshold: Some(95.0),
                message: Some("Memory usage critical".to_string()),
            }
        } else if metrics.memory.utilization > 85.0 {
            warnings.push(format!("GPU memory usage high: {:.1}%", metrics.memory.utilization));
            HealthCheck {
                name: "memory".to_string(),
                status: CheckStatus::Warning,
                value: Some(metrics.memory.utilization),
                threshold: Some(85.0),
                message: Some("Memory usage warning".to_string()),
            }
        } else {
            HealthCheck {
                name: "memory".to_string(),
                status: CheckStatus::Pass,
                value: Some(metrics.memory.utilization),
                threshold: Some(85.0),
                message: None,
            }
        };
        checks.insert("memory".to_string(), mem_check);

        // Power check
        let power_check = if metrics.power.utilization > 95.0 {
            warnings.push(format!("GPU power usage high: {:.1}%", metrics.power.utilization));
            HealthCheck {
                name: "power".to_string(),
                status: CheckStatus::Warning,
                value: Some(metrics.power.utilization),
                threshold: Some(95.0),
                message: Some("Power usage high".to_string()),
            }
        } else {
            HealthCheck {
                name: "power".to_string(),
                status: CheckStatus::Pass,
                value: Some(metrics.power.utilization),
                threshold: Some(95.0),
                message: None,
            }
        };
        checks.insert("power".to_string(), power_check);

        // GPU status check
        let status_check = match &metrics.status {
            GpuStatus::Active | GpuStatus::Idle | GpuStatus::InUse => {
                HealthCheck {
                    name: "status".to_string(),
                    status: CheckStatus::Pass,
                    value: None,
                    threshold: None,
                    message: Some(format!("GPU status: {:?}", metrics.status)),
                }
            }
            GpuStatus::Throttling => {
                warnings.push("GPU is throttling".to_string());
                HealthCheck {
                    name: "status".to_string(),
                    status: CheckStatus::Warning,
                    value: None,
                    threshold: None,
                    message: Some("GPU throttling".to_string()),
                }
            }
            GpuStatus::Error(msg) => {
                criticals.push(format!("GPU error: {}", msg));
                HealthCheck {
                    name: "status".to_string(),
                    status: CheckStatus::Critical,
                    value: None,
                    threshold: None,
                    message: Some(format!("GPU error: {}", msg)),
                }
            }
            GpuStatus::Unavailable | GpuStatus::Maintenance => {
                criticals.push(format!("GPU unavailable: {:?}", metrics.status));
                HealthCheck {
                    name: "status".to_string(),
                    status: CheckStatus::Critical,
                    value: None,
                    threshold: None,
                    message: Some(format!("GPU unavailable: {:?}", metrics.status)),
                }
            }
        };
        checks.insert("status".to_string(), status_check);

        // Thermal state check
        let thermal_check = match metrics.temperature.thermal_state {
            ThermalState::Normal => {
                HealthCheck {
                    name: "thermal_state".to_string(),
                    status: CheckStatus::Pass,
                    value: None,
                    threshold: None,
                    message: Some("Thermal state normal".to_string()),
                }
            }
            ThermalState::Warning => {
                warnings.push("GPU thermal warning".to_string());
                HealthCheck {
                    name: "thermal_state".to_string(),
                    status: CheckStatus::Warning,
                    value: None,
                    threshold: None,
                    message: Some("Thermal warning".to_string()),
                }
            }
            ThermalState::Critical | ThermalState::Throttling | ThermalState::Shutdown => {
                criticals.push(format!("GPU thermal critical: {:?}", metrics.temperature.thermal_state));
                HealthCheck {
                    name: "thermal_state".to_string(),
                    status: CheckStatus::Critical,
                    value: None,
                    threshold: None,
                    message: Some(format!("Thermal critical: {:?}", metrics.temperature.thermal_state)),
                }
            }
        };
        checks.insert("thermal_state".to_string(), thermal_check);

        // Determine overall status
        let status = if !criticals.is_empty() {
            HealthStatus::Critical(criticals)
        } else if !warnings.is_empty() {
            HealthStatus::Warning(warnings)
        } else {
            HealthStatus::Healthy
        };

        // Calculate health score
        let score = Self::calculate_health_score(&checks);

        Self {
            gpu_id: metrics.info.index,
            status,
            checks,
            score,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Calculate health score from checks
    fn calculate_health_score(checks: &HashMap<String, HealthCheck>) -> f64 {
        if checks.is_empty() {
            return 0.0;
        }

        let total_score: f64 = checks.values().map(|check| {
            match check.status {
                CheckStatus::Pass => 100.0,
                CheckStatus::Warning => 70.0,
                CheckStatus::Critical => 0.0,
                CheckStatus::Unknown => 50.0,
            }
        }).sum();

        total_score / checks.len() as f64
    }

    /// Check if GPU is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(self.status, HealthStatus::Healthy)
    }

    /// Check if GPU has warnings
    pub fn has_warnings(&self) -> bool {
        matches!(self.status, HealthStatus::Warning(_))
    }

    /// Check if GPU is critical
    pub fn is_critical(&self) -> bool {
        matches!(self.status, HealthStatus::Critical(_))
    }

    /// Get all warning messages
    pub fn get_warnings(&self) -> Vec<String> {
        match &self.status {
            HealthStatus::Warning(warnings) => warnings.clone(),
            _ => Vec::new(),
        }
    }

    /// Get all critical messages
    pub fn get_criticals(&self) -> Vec<String> {
        match &self.status {
            HealthStatus::Critical(criticals) => criticals.clone(),
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::*;

    #[test]
    fn test_health_status() {
        assert_eq!(HealthStatus::Healthy, HealthStatus::Healthy);
        assert_ne!(HealthStatus::Healthy, HealthStatus::Warning(vec![]));
    }

    #[test]
    fn test_check_status() {
        assert_eq!(CheckStatus::Pass, CheckStatus::Pass);
        assert_ne!(CheckStatus::Pass, CheckStatus::Critical);
    }

    #[test]
    fn test_health_score_calculation() {
        let mut checks = HashMap::new();
        
        checks.insert("test1".to_string(), HealthCheck {
            name: "test1".to_string(),
            status: CheckStatus::Pass,
            value: None,
            threshold: None,
            message: None,
        });
        
        checks.insert("test2".to_string(), HealthCheck {
            name: "test2".to_string(),
            status: CheckStatus::Warning,
            value: None,
            threshold: None,
            message: None,
        });

        let score = GpuHealth::calculate_health_score(&checks);
        assert_eq!(score, 85.0); // (100 + 70) / 2
    }

    #[test]
    fn test_gpu_health_from_metrics() {
        let info = GpuInfo {
            index: 0,
            uuid: "GPU-12345".to_string(),
            name: "Test GPU".to_string(),
            brand: "NVIDIA".to_string(),
            architecture: Some("Ampere".to_string()),
            driver_version: "470.0".to_string(),
            vbios_version: None,
            pci_info: PciInfo {
                bus_id: "0000:01:00.0".to_string(),
                domain: 0,
                bus: 1,
                device: 0,
                function: 0,
                device_id: 0x1234,
                subsystem_id: 0x5678,
                link_gen: None,
                link_width: None,
                max_link_gen: None,
                max_link_width: None,
            },
            capabilities: GpuCapabilities::default(),
            total_memory: 8 * 1024 * 1024 * 1024,
            memory_bus_width: None,
            memory_type: None,
        };

        let mut metrics = GpuMetrics::new(info);
        
        // Set healthy values
        metrics.temperature.gpu = 70.0;
        metrics.memory.utilization = 50.0;
        metrics.power.utilization = 60.0;
        
        let health = GpuHealth::from_metrics(&metrics);
        assert!(health.is_healthy());
        assert_eq!(health.gpu_id, 0);
        assert!(health.score > 90.0);
    }

    #[test]
    fn test_gpu_health_with_warnings() {
        let info = GpuInfo {
            index: 1,
            uuid: "GPU-67890".to_string(),
            name: "Test GPU 2".to_string(),
            brand: "NVIDIA".to_string(),
            architecture: Some("Ampere".to_string()),
            driver_version: "470.0".to_string(),
            vbios_version: None,
            pci_info: PciInfo {
                bus_id: "0000:02:00.0".to_string(),
                domain: 0,
                bus: 2,
                device: 0,
                function: 0,
                device_id: 0x1234,
                subsystem_id: 0x5678,
                link_gen: None,
                link_width: None,
                max_link_gen: None,
                max_link_width: None,
            },
            capabilities: GpuCapabilities::default(),
            total_memory: 8 * 1024 * 1024 * 1024,
            memory_bus_width: None,
            memory_type: None,
        };

        let mut metrics = GpuMetrics::new(info);
        
        // Set warning values
        metrics.temperature.gpu = 85.0; // Warning threshold
        metrics.memory.utilization = 90.0; // Warning threshold
        metrics.power.utilization = 60.0;
        
        let health = GpuHealth::from_metrics(&metrics);
        assert!(health.has_warnings());
        assert!(!health.is_healthy());
        assert!(!health.is_critical());
        
        let warnings = health.get_warnings();
        assert!(!warnings.is_empty());
    }
}
