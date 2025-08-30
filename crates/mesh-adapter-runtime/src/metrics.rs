//! Metrics collection for runtime adapters

use crate::config::MetricsConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant};

/// Runtime metrics collected from adapters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeMetrics {
    /// Timestamp when metrics were collected
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Request metrics
    pub requests: RequestMetrics,
    
    /// Model metrics
    pub models: HashMap<String, ModelMetrics>,
    
    /// Resource metrics
    pub resources: ResourceMetrics,
    
    /// Runtime-specific metrics
    pub runtime_specific: HashMap<String, serde_json::Value>,
}

/// Request-related metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetrics {
    /// Total number of requests
    pub total_requests: u64,
    
    /// Number of successful requests
    pub successful_requests: u64,
    
    /// Number of failed requests
    pub failed_requests: u64,
    
    /// Current requests per second
    pub requests_per_second: f64,
    
    /// Average request duration in milliseconds
    pub avg_request_duration_ms: f64,
    
    /// P50 request duration in milliseconds
    pub p50_request_duration_ms: f64,
    
    /// P95 request duration in milliseconds
    pub p95_request_duration_ms: f64,
    
    /// P99 request duration in milliseconds
    pub p99_request_duration_ms: f64,
    
    /// Current queue size
    pub queue_size: u64,
    
    /// Average queue time in milliseconds
    pub avg_queue_time_ms: f64,
}

/// Model-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Model name
    pub name: String,
    
    /// Model version
    pub version: Option<String>,
    
    /// Model status
    pub status: String,
    
    /// Number of requests for this model
    pub requests: u64,
    
    /// Average inference time in milliseconds
    pub avg_inference_time_ms: f64,
    
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    
    /// GPU memory usage in bytes
    pub gpu_memory_usage_bytes: Option<u64>,
    
    /// Model load time in milliseconds
    pub load_time_ms: Option<f64>,
    
    /// Last request timestamp
    pub last_request: Option<chrono::DateTime<chrono::Utc>>,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// CPU utilization percentage (0-100)
    pub cpu_utilization: f64,
    
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    
    /// Total memory in bytes
    pub memory_total_bytes: u64,
    
    /// GPU metrics (if available)
    pub gpu: Option<GpuMetrics>,
    
    /// Network I/O metrics
    pub network: NetworkMetrics,
    
    /// Disk I/O metrics
    pub disk: DiskMetrics,
}

/// GPU-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// GPU utilization percentage (0-100)
    pub utilization: f64,
    
    /// GPU memory usage in bytes
    pub memory_usage_bytes: u64,
    
    /// Total GPU memory in bytes
    pub memory_total_bytes: u64,
    
    /// GPU temperature in Celsius
    pub temperature_celsius: f64,
    
    /// Power usage in watts
    pub power_usage_watts: f64,
    
    /// Per-GPU metrics (for multi-GPU systems)
    pub per_gpu: Vec<SingleGpuMetrics>,
}

/// Metrics for a single GPU
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleGpuMetrics {
    /// GPU index
    pub index: u32,
    
    /// GPU name/model
    pub name: String,
    
    /// Utilization percentage (0-100)
    pub utilization: f64,
    
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    
    /// Total memory in bytes
    pub memory_total_bytes: u64,
    
    /// Temperature in Celsius
    pub temperature_celsius: f64,
    
    /// Power usage in watts
    pub power_usage_watts: f64,
}

/// Network I/O metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Bytes received
    pub bytes_received: u64,
    
    /// Bytes sent
    pub bytes_sent: u64,
    
    /// Packets received
    pub packets_received: u64,
    
    /// Packets sent
    pub packets_sent: u64,
    
    /// Receive rate in bytes per second
    pub receive_rate_bps: f64,
    
    /// Send rate in bytes per second
    pub send_rate_bps: f64,
}

/// Disk I/O metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskMetrics {
    /// Bytes read
    pub bytes_read: u64,
    
    /// Bytes written
    pub bytes_written: u64,
    
    /// Read operations
    pub read_ops: u64,
    
    /// Write operations
    pub write_ops: u64,
    
    /// Read rate in bytes per second
    pub read_rate_bps: f64,
    
    /// Write rate in bytes per second
    pub write_rate_bps: f64,
}

/// Metric collector for runtime adapters
pub struct MetricCollector {
    config: MetricsConfig,
    start_time: Instant,
    
    // Request counters
    total_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    
    // Request timing
    request_durations: Vec<f64>, // In a real implementation, this would be a proper histogram
    
    // Model metrics
    model_metrics: HashMap<String, ModelMetrics>,
    
    // Last collection time
    last_collection: Option<Instant>,
}

impl MetricCollector {
    /// Create a new metric collector
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            config,
            start_time: Instant::now(),
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            request_durations: Vec::new(),
            model_metrics: HashMap::new(),
            last_collection: None,
        }
    }

    /// Record a successful request
    pub fn record_request_success(&self, _duration_ms: f64) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.successful_requests.fetch_add(1, Ordering::Relaxed);
        
        // In a real implementation, we'd use a thread-safe histogram
        // For now, we'll just note that we would record the duration
    }

    /// Record a failed request
    pub fn record_request_failure(&self, _duration_ms: f64) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Update model metrics
    pub fn update_model_metrics(&mut self, model_name: String, metrics: ModelMetrics) {
        self.model_metrics.insert(model_name, metrics);
    }

    /// Collect current metrics
    pub async fn collect_metrics(&mut self) -> RuntimeMetrics {
        let now = chrono::Utc::now();
        let collection_time = Instant::now();
        
        // Calculate rates
        let elapsed_seconds = match self.last_collection {
            Some(last) => last.elapsed().as_secs_f64(),
            None => self.start_time.elapsed().as_secs_f64(),
        };
        
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let successful_requests = self.successful_requests.load(Ordering::Relaxed);
        let failed_requests = self.failed_requests.load(Ordering::Relaxed);
        
        let requests_per_second = if elapsed_seconds > 0.0 {
            total_requests as f64 / elapsed_seconds
        } else {
            0.0
        };

        // Create request metrics
        let request_metrics = RequestMetrics {
            total_requests,
            successful_requests,
            failed_requests,
            requests_per_second,
            avg_request_duration_ms: self.calculate_avg_duration(),
            p50_request_duration_ms: self.calculate_percentile(50.0),
            p95_request_duration_ms: self.calculate_percentile(95.0),
            p99_request_duration_ms: self.calculate_percentile(99.0),
            queue_size: 0, // Would be implemented based on runtime
            avg_queue_time_ms: 0.0,
        };

        // Create resource metrics (mock data for now)
        let resource_metrics = ResourceMetrics {
            cpu_utilization: 45.2,
            memory_usage_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            memory_total_bytes: 16 * 1024 * 1024 * 1024, // 16GB
            gpu: Some(GpuMetrics {
                utilization: 78.5,
                memory_usage_bytes: 6 * 1024 * 1024 * 1024, // 6GB
                memory_total_bytes: 8 * 1024 * 1024 * 1024, // 8GB
                temperature_celsius: 72.0,
                power_usage_watts: 250.0,
                per_gpu: vec![SingleGpuMetrics {
                    index: 0,
                    name: "NVIDIA A100".to_string(),
                    utilization: 78.5,
                    memory_usage_bytes: 6 * 1024 * 1024 * 1024,
                    memory_total_bytes: 8 * 1024 * 1024 * 1024,
                    temperature_celsius: 72.0,
                    power_usage_watts: 250.0,
                }],
            }),
            network: NetworkMetrics {
                bytes_received: 1024 * 1024 * 100, // 100MB
                bytes_sent: 1024 * 1024 * 50,      // 50MB
                packets_received: 10000,
                packets_sent: 8000,
                receive_rate_bps: 1024.0 * 1024.0, // 1MB/s
                send_rate_bps: 512.0 * 1024.0,     // 512KB/s
            },
            disk: DiskMetrics {
                bytes_read: 1024 * 1024 * 1024, // 1GB
                bytes_written: 512 * 1024 * 1024, // 512MB
                read_ops: 1000,
                write_ops: 500,
                read_rate_bps: 10.0 * 1024.0 * 1024.0, // 10MB/s
                write_rate_bps: 5.0 * 1024.0 * 1024.0,  // 5MB/s
            },
        };

        self.last_collection = Some(collection_time);

        RuntimeMetrics {
            timestamp: now,
            requests: request_metrics,
            models: self.model_metrics.clone(),
            resources: resource_metrics,
            runtime_specific: HashMap::new(),
        }
    }

    /// Check if it's time to collect metrics
    pub fn should_collect(&self) -> bool {
        if !self.config.enabled {
            return false;
        }

        match self.last_collection {
            Some(last) => last.elapsed() >= self.config.interval,
            None => true,
        }
    }

    /// Get metrics configuration
    pub fn config(&self) -> &MetricsConfig {
        &self.config
    }

    /// Reset metrics
    pub fn reset(&mut self) {
        self.total_requests.store(0, Ordering::Relaxed);
        self.successful_requests.store(0, Ordering::Relaxed);
        self.failed_requests.store(0, Ordering::Relaxed);
        self.request_durations.clear();
        self.model_metrics.clear();
        self.last_collection = None;
        self.start_time = Instant::now();
    }

    // Helper methods for calculating statistics
    fn calculate_avg_duration(&self) -> f64 {
        if self.request_durations.is_empty() {
            0.0
        } else {
            self.request_durations.iter().sum::<f64>() / self.request_durations.len() as f64
        }
    }

    fn calculate_percentile(&self, percentile: f64) -> f64 {
        if self.request_durations.is_empty() {
            return 0.0;
        }

        let mut sorted = self.request_durations.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (percentile / 100.0 * (sorted.len() - 1) as f64) as usize;
        sorted.get(index).copied().unwrap_or(0.0)
    }
}

impl Default for RequestMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            requests_per_second: 0.0,
            avg_request_duration_ms: 0.0,
            p50_request_duration_ms: 0.0,
            p95_request_duration_ms: 0.0,
            p99_request_duration_ms: 0.0,
            queue_size: 0,
            avg_queue_time_ms: 0.0,
        }
    }
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_usage_bytes: 0,
            memory_total_bytes: 0,
            gpu: None,
            network: NetworkMetrics::default(),
            disk: DiskMetrics::default(),
        }
    }
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        Self {
            bytes_received: 0,
            bytes_sent: 0,
            packets_received: 0,
            packets_sent: 0,
            receive_rate_bps: 0.0,
            send_rate_bps: 0.0,
        }
    }
}

impl Default for DiskMetrics {
    fn default() -> Self {
        Self {
            bytes_read: 0,
            bytes_written: 0,
            read_ops: 0,
            write_ops: 0,
            read_rate_bps: 0.0,
            write_rate_bps: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_metric_collector_creation() {
        let config = MetricsConfig::default();
        let collector = MetricCollector::new(config);
        
        assert_eq!(collector.total_requests.load(Ordering::Relaxed), 0);
        assert_eq!(collector.successful_requests.load(Ordering::Relaxed), 0);
        assert_eq!(collector.failed_requests.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_request_recording() {
        let config = MetricsConfig::default();
        let collector = MetricCollector::new(config);
        
        collector.record_request_success(100.0);
        collector.record_request_failure(200.0);
        
        assert_eq!(collector.total_requests.load(Ordering::Relaxed), 2);
        assert_eq!(collector.successful_requests.load(Ordering::Relaxed), 1);
        assert_eq!(collector.failed_requests.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let config = MetricsConfig::default();
        let mut collector = MetricCollector::new(config);
        
        collector.record_request_success(100.0);
        collector.record_request_success(150.0);
        
        let metrics = collector.collect_metrics().await;
        
        assert_eq!(metrics.requests.total_requests, 2);
        assert_eq!(metrics.requests.successful_requests, 2);
        assert_eq!(metrics.requests.failed_requests, 0);
        assert!(metrics.requests.requests_per_second >= 0.0);
    }

    #[test]
    fn test_should_collect_timing() {
        let config = MetricsConfig {
            enabled: true,
            interval: Duration::from_millis(100),
            endpoint: None,
            metrics: vec![],
            detailed_model_metrics: false,
        };
        
        let mut collector = MetricCollector::new(config);
        
        // Should collect initially
        assert!(collector.should_collect());
        
        // After collecting, should not collect immediately
        collector.last_collection = Some(Instant::now());
        assert!(!collector.should_collect());
        
        // After waiting, should collect again
        std::thread::sleep(Duration::from_millis(150));
        assert!(collector.should_collect());
    }

    #[test]
    fn test_metrics_reset() {
        let config = MetricsConfig::default();
        let mut collector = MetricCollector::new(config);
        
        collector.record_request_success(100.0);
        assert_eq!(collector.total_requests.load(Ordering::Relaxed), 1);
        
        collector.reset();
        assert_eq!(collector.total_requests.load(Ordering::Relaxed), 0);
        assert_eq!(collector.successful_requests.load(Ordering::Relaxed), 0);
        assert_eq!(collector.failed_requests.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_percentile_calculation() {
        let config = MetricsConfig::default();
        let mut collector = MetricCollector::new(config);
        
        // Add some test durations
        collector.request_durations = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        
        assert_eq!(collector.calculate_avg_duration(), 55.0);
        assert_eq!(collector.calculate_percentile(50.0), 50.0);
        assert_eq!(collector.calculate_percentile(95.0), 90.0);
    }

    #[test]
    fn test_gpu_metrics() {
        let gpu_metrics = GpuMetrics {
            utilization: 85.0,
            memory_usage_bytes: 6 * 1024 * 1024 * 1024,
            memory_total_bytes: 8 * 1024 * 1024 * 1024,
            temperature_celsius: 75.0,
            power_usage_watts: 300.0,
            per_gpu: vec![],
        };
        
        assert_eq!(gpu_metrics.utilization, 85.0);
        assert_eq!(gpu_metrics.memory_usage_bytes, 6 * 1024 * 1024 * 1024);
        assert_eq!(gpu_metrics.temperature_celsius, 75.0);
    }
}
