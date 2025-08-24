//! Health checking for runtime adapters

use crate::config::HealthCheckConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Health status of a runtime
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Runtime is healthy and ready to serve requests
    Healthy,
    /// Runtime is starting up
    Starting,
    /// Runtime is shutting down
    Stopping,
    /// Runtime is unhealthy
    Unhealthy(String),
    /// Runtime status is unknown
    Unknown,
}

/// Detailed health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthInfo {
    /// Overall health status
    pub status: HealthStatus,
    
    /// Health check timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Response time in milliseconds
    pub response_time_ms: f64,
    
    /// Detailed checks
    pub checks: HashMap<String, CheckResult>,
    
    /// Runtime-specific health data
    pub runtime_data: HashMap<String, serde_json::Value>,
}

/// Individual check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    /// Check name
    pub name: String,
    
    /// Check status
    pub status: CheckStatus,
    
    /// Check message
    pub message: Option<String>,
    
    /// Check duration in milliseconds
    pub duration_ms: f64,
}

/// Status of an individual check
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CheckStatus {
    Pass,
    Fail,
    Warn,
}

/// Health checker for runtime adapters
pub struct HealthCheck {
    config: HealthCheckConfig,
    consecutive_failures: u32,
    consecutive_successes: u32,
    last_check: Option<Instant>,
    current_status: HealthStatus,
}

impl HealthCheck {
    /// Create a new health checker
    pub fn new(config: HealthCheckConfig) -> Self {
        Self {
            config,
            consecutive_failures: 0,
            consecutive_successes: 0,
            last_check: None,
            current_status: HealthStatus::Unknown,
        }
    }

    /// Update health status based on a check result
    pub fn update_status(&mut self, health_info: HealthInfo) -> HealthStatus {
        let now = Instant::now();
        self.last_check = Some(now);

        match health_info.status {
            HealthStatus::Healthy => {
                self.consecutive_failures = 0;
                self.consecutive_successes += 1;
                
                if self.consecutive_successes >= self.config.success_threshold {
                    self.current_status = HealthStatus::Healthy;
                }
            }
            HealthStatus::Unhealthy(_) => {
                self.consecutive_successes = 0;
                self.consecutive_failures += 1;
                
                if self.consecutive_failures >= self.config.failure_threshold {
                    self.current_status = health_info.status;
                }
            }
            other => {
                // For Starting, Stopping, Unknown, don't change counters
                self.current_status = other;
            }
        }

        self.current_status.clone()
    }

    /// Check if it's time for a health check
    pub fn should_check(&self) -> bool {
        match self.last_check {
            Some(last) => last.elapsed() >= self.config.interval,
            None => true,
        }
    }

    /// Get current health status
    pub fn current_status(&self) -> &HealthStatus {
        &self.current_status
    }

    /// Get health check configuration
    pub fn config(&self) -> &HealthCheckConfig {
        &self.config
    }

    /// Reset health check state
    pub fn reset(&mut self) {
        self.consecutive_failures = 0;
        self.consecutive_successes = 0;
        self.last_check = None;
        self.current_status = HealthStatus::Unknown;
    }
}

impl HealthInfo {
    /// Create a new health info with healthy status
    pub fn healthy() -> Self {
        Self {
            status: HealthStatus::Healthy,
            timestamp: chrono::Utc::now(),
            response_time_ms: 0.0,
            checks: HashMap::new(),
            runtime_data: HashMap::new(),
        }
    }

    /// Create a new health info with unhealthy status
    pub fn unhealthy(reason: String) -> Self {
        Self {
            status: HealthStatus::Unhealthy(reason),
            timestamp: chrono::Utc::now(),
            response_time_ms: 0.0,
            checks: HashMap::new(),
            runtime_data: HashMap::new(),
        }
    }

    /// Create a new health info with starting status
    pub fn starting() -> Self {
        Self {
            status: HealthStatus::Starting,
            timestamp: chrono::Utc::now(),
            response_time_ms: 0.0,
            checks: HashMap::new(),
            runtime_data: HashMap::new(),
        }
    }

    /// Add a check result
    pub fn add_check(mut self, check: CheckResult) -> Self {
        self.checks.insert(check.name.clone(), check);
        self
    }

    /// Add runtime-specific data
    pub fn add_runtime_data(mut self, key: String, value: serde_json::Value) -> Self {
        self.runtime_data.insert(key, value);
        self
    }

    /// Set response time
    pub fn with_response_time(mut self, response_time_ms: f64) -> Self {
        self.response_time_ms = response_time_ms;
        self
    }

    /// Check if all individual checks passed
    pub fn all_checks_passed(&self) -> bool {
        self.checks.values().all(|check| check.status == CheckStatus::Pass)
    }

    /// Get failed checks
    pub fn failed_checks(&self) -> Vec<&CheckResult> {
        self.checks.values()
            .filter(|check| check.status == CheckStatus::Fail)
            .collect()
    }
}

impl CheckResult {
    /// Create a passing check result
    pub fn pass(name: String) -> Self {
        Self {
            name,
            status: CheckStatus::Pass,
            message: None,
            duration_ms: 0.0,
        }
    }

    /// Create a failing check result
    pub fn fail(name: String, message: String) -> Self {
        Self {
            name,
            status: CheckStatus::Fail,
            message: Some(message),
            duration_ms: 0.0,
        }
    }

    /// Create a warning check result
    pub fn warn(name: String, message: String) -> Self {
        Self {
            name,
            status: CheckStatus::Warn,
            message: Some(message),
            duration_ms: 0.0,
        }
    }

    /// Set the duration
    pub fn with_duration(mut self, duration_ms: f64) -> Self {
        self.duration_ms = duration_ms;
        self
    }
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Starting => write!(f, "starting"),
            HealthStatus::Stopping => write!(f, "stopping"),
            HealthStatus::Unhealthy(reason) => write!(f, "unhealthy: {}", reason),
            HealthStatus::Unknown => write!(f, "unknown"),
        }
    }
}

impl std::fmt::Display for CheckStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckStatus::Pass => write!(f, "pass"),
            CheckStatus::Fail => write!(f, "fail"),
            CheckStatus::Warn => write!(f, "warn"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_health_status_display() {
        assert_eq!(HealthStatus::Healthy.to_string(), "healthy");
        assert_eq!(HealthStatus::Starting.to_string(), "starting");
        assert_eq!(HealthStatus::Unhealthy("error".to_string()).to_string(), "unhealthy: error");
    }

    #[test]
    fn test_health_info_creation() {
        let info = HealthInfo::healthy();
        assert_eq!(info.status, HealthStatus::Healthy);
        assert!(info.checks.is_empty());

        let info = HealthInfo::unhealthy("connection failed".to_string());
        assert!(matches!(info.status, HealthStatus::Unhealthy(_)));
    }

    #[test]
    fn test_health_info_checks() {
        let info = HealthInfo::healthy()
            .add_check(CheckResult::pass("connectivity".to_string()))
            .add_check(CheckResult::fail("memory".to_string(), "low memory".to_string()));

        assert_eq!(info.checks.len(), 2);
        assert!(!info.all_checks_passed());
        assert_eq!(info.failed_checks().len(), 1);
    }

    #[test]
    fn test_check_result_creation() {
        let pass = CheckResult::pass("test".to_string()).with_duration(10.5);
        assert_eq!(pass.status, CheckStatus::Pass);
        assert_eq!(pass.duration_ms, 10.5);

        let fail = CheckResult::fail("test".to_string(), "error".to_string());
        assert_eq!(fail.status, CheckStatus::Fail);
        assert_eq!(fail.message, Some("error".to_string()));
    }

    #[test]
    fn test_health_check_thresholds() {
        let config = HealthCheckConfig {
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            failure_threshold: 3,
            success_threshold: 2,
            endpoint: None,
        };

        let mut checker = HealthCheck::new(config);
        assert_eq!(checker.current_status(), &HealthStatus::Unknown);

        // First healthy check - not enough successes yet
        let status = checker.update_status(HealthInfo::healthy());
        assert_eq!(status, HealthStatus::Unknown);

        // Second healthy check - should be healthy now
        let status = checker.update_status(HealthInfo::healthy());
        assert_eq!(status, HealthStatus::Healthy);

        // First failure - not enough failures yet
        let status = checker.update_status(HealthInfo::unhealthy("error".to_string()));
        assert_eq!(status, HealthStatus::Healthy);

        // More failures
        checker.update_status(HealthInfo::unhealthy("error".to_string()));
        let status = checker.update_status(HealthInfo::unhealthy("error".to_string()));
        assert!(matches!(status, HealthStatus::Unhealthy(_)));
    }

    #[test]
    fn test_health_check_timing() {
        let config = HealthCheckConfig {
            interval: Duration::from_millis(100),
            timeout: Duration::from_secs(5),
            failure_threshold: 1,
            success_threshold: 1,
            endpoint: None,
        };

        let mut checker = HealthCheck::new(config);
        
        // Should check initially
        assert!(checker.should_check());
        
        // After updating, should not check immediately
        checker.update_status(HealthInfo::healthy());
        assert!(!checker.should_check());
        
        // After waiting, should check again
        std::thread::sleep(Duration::from_millis(150));
        assert!(checker.should_check());
    }

    #[test]
    fn test_health_check_reset() {
        let mut config = HealthCheckConfig::default();
        config.success_threshold = 1; // Set threshold to 1 for immediate transition
        let mut checker = HealthCheck::new(config);
        
        checker.update_status(HealthInfo::healthy());
        assert_ne!(checker.current_status(), &HealthStatus::Unknown);
        
        checker.reset();
        assert_eq!(checker.current_status(), &HealthStatus::Unknown);
        assert_eq!(checker.consecutive_failures, 0);
        assert_eq!(checker.consecutive_successes, 0);
    }
}
