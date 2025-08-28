use serde::{Deserialize, Serialize};
use crate::engine::{RequestType, Request};

/// GPU profile configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProfile {
    pub name: String,
    /// Tokens per second throughput
    pub tokens_per_s: u32,
    /// Maximum concurrent batches
    pub concurrency: u32,
    /// Total VRAM in GB
    pub vram_total_gb: f64,
    /// Batch window in milliseconds
    pub batch_window_ms: f64,
    /// KV cache memory per request in GB
    pub kv_cache_gb_per_req: f64,
}

/// MIG (Multi-Instance GPU) profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigProfile {
    pub name: String,
    /// Fraction of the full GPU (e.g., 0.125 for 1/8)
    pub fraction: f64,
    /// Tokens per second for this MIG slice
    pub tokens_per_s: u32,
    /// Maximum concurrent batches for this slice
    pub concurrency: u32,
}

impl MigProfile {
    /// Calculate VRAM total for this MIG instance
    pub fn vram_total_gb(&self) -> f64 {
        // TODO: This should be based on the parent GPU's VRAM
        // For now, assume a base of 80GB H100
        80.0 * self.fraction
    }

    /// Calculate batch window (typically same as parent GPU)
    pub fn batch_window_ms(&self) -> f64 {
        8.0 // Default batch window
    }

    /// Calculate KV cache per request (scaled by fraction)
    pub fn kv_cache_gb_per_req(&self) -> f64 {
        1.2 * self.fraction
    }
}

/// Service model for different request types
#[derive(Debug, Clone)]
pub struct ServiceModel {
    pub request_type: RequestType,
}

impl ServiceModel {
    /// Calculate service time for a batch of requests
    pub fn calculate_service_time(
        &self,
        requests: &[Request],
        gpu_profile: &GpuProfile,
    ) -> f64 {
        if requests.is_empty() {
            return 0.0;
        }

        match self.request_type {
            RequestType::LLM => self.calculate_llm_service_time(requests, gpu_profile),
            RequestType::Vision => self.calculate_vision_service_time(requests, gpu_profile),
            RequestType::ASR => self.calculate_asr_service_time(requests, gpu_profile),
        }
    }

    /// Calculate service time for LLM requests
    fn calculate_llm_service_time(
        &self,
        requests: &[Request],
        gpu_profile: &GpuProfile,
    ) -> f64 {
        // For LLM, service time is primarily determined by output tokens
        // We use a simplified model: total_tokens / tokens_per_s
        
        let total_input_tokens: u32 = requests.iter().map(|r| r.input_tokens).sum();
        let total_output_tokens: u32 = requests.iter().map(|r| r.expected_output_tokens).sum();
        let total_tokens = total_input_tokens + total_output_tokens;

        // Convert to milliseconds
        (total_tokens as f64 / gpu_profile.tokens_per_s as f64) * 1000.0
    }

    /// Calculate service time for Vision requests
    fn calculate_vision_service_time(
        &self,
        requests: &[Request],
        gpu_profile: &GpuProfile,
    ) -> f64 {
        // Vision requests have more uniform processing time
        // Simplified model: fixed time per request + batch efficiency
        let base_time_per_request = 50.0; // ms
        let batch_efficiency = 0.8; // Batching reduces per-request time
        
        let total_time = requests.len() as f64 * base_time_per_request * batch_efficiency;
        
        // Scale by GPU performance (tokens_per_s as a proxy)
        let performance_factor = 240000.0 / gpu_profile.tokens_per_s as f64;
        total_time * performance_factor
    }

    /// Calculate service time for ASR requests
    fn calculate_asr_service_time(
        &self,
        requests: &[Request],
        gpu_profile: &GpuProfile,
    ) -> f64 {
        // ASR requests are similar to vision but typically faster
        let base_time_per_request = 30.0; // ms
        let batch_efficiency = 0.9; // Better batching efficiency than vision
        
        let total_time = requests.len() as f64 * base_time_per_request * batch_efficiency;
        
        // Scale by GPU performance
        let performance_factor = 240000.0 / gpu_profile.tokens_per_s as f64;
        total_time * performance_factor
    }

    /// Calculate VRAM usage for a batch of requests
    pub fn calculate_vram_usage(
        &self,
        requests: &[Request],
        gpu_profile: &GpuProfile,
    ) -> f64 {
        match self.request_type {
            RequestType::LLM => {
                // LLM VRAM usage is dominated by KV cache
                requests.len() as f64 * gpu_profile.kv_cache_gb_per_req
            }
            RequestType::Vision => {
                // Vision models have more uniform VRAM usage
                let base_vram_per_request = 0.5; // GB
                requests.len() as f64 * base_vram_per_request
            }
            RequestType::ASR => {
                // ASR models typically use less VRAM
                let base_vram_per_request = 0.2; // GB
                requests.len() as f64 * base_vram_per_request
            }
        }
    }

    /// Check if a batch can fit in available VRAM
    pub fn can_fit_in_vram(
        &self,
        requests: &[Request],
        gpu_profile: &GpuProfile,
        available_vram_gb: f64,
    ) -> bool {
        let required_vram = self.calculate_vram_usage(requests, gpu_profile);
        required_vram <= available_vram_gb
    }

    /// Calculate the maximum batch size that fits in available VRAM
    pub fn max_batch_size_for_vram(
        &self,
        _sample_request: &Request,
        gpu_profile: &GpuProfile,
        available_vram_gb: f64,
    ) -> usize {
        let vram_per_request = match self.request_type {
            RequestType::LLM => gpu_profile.kv_cache_gb_per_req,
            RequestType::Vision => 0.5,
            RequestType::ASR => 0.2,
        };

        if vram_per_request <= 0.0 {
            return usize::MAX;
        }

        (available_vram_gb / vram_per_request).floor() as usize
    }
}

/// Batching strategy for different request types
#[derive(Debug, Clone)]
pub struct BatchingStrategy {
    pub max_batch_size: usize,
    pub batch_window_ms: f64,
}

impl BatchingStrategy {
    /// Create a batching strategy for a GPU profile
    pub fn for_gpu_profile(gpu_profile: &GpuProfile) -> Self {
        Self {
            max_batch_size: gpu_profile.concurrency as usize * 8, // Heuristic
            batch_window_ms: gpu_profile.batch_window_ms,
        }
    }

    /// Create a batching strategy for a MIG profile
    pub fn for_mig_profile(mig_profile: &MigProfile) -> Self {
        Self {
            max_batch_size: mig_profile.concurrency as usize * 4, // Smaller batches for MIG
            batch_window_ms: mig_profile.batch_window_ms(),
        }
    }

    /// Determine if a batch should be closed based on size and time
    pub fn should_close_batch(
        &self,
        current_batch_size: usize,
        batch_open_duration_ms: f64,
    ) -> bool {
        current_batch_size >= self.max_batch_size || batch_open_duration_ms >= self.batch_window_ms
    }
}

/// GPU utilization and performance metrics
#[derive(Debug, Clone, Default)]
pub struct GpuMetrics {
    /// SM (Streaming Multiprocessor) utilization percentage
    pub sm_utilization: f64,
    /// Memory utilization percentage
    pub memory_utilization: f64,
    /// Current power draw in watts
    pub power_draw_w: f64,
    /// Temperature in Celsius
    pub temperature_c: f64,
    /// Total processed requests
    pub total_requests: u64,
    /// Total processed tokens
    pub total_tokens: u64,
    /// Average batch size
    pub avg_batch_size: f64,
}

impl GpuMetrics {
    /// Update metrics after processing a batch
    pub fn update_after_batch(
        &mut self,
        batch_size: usize,
        tokens_processed: u32,
        service_time_ms: f64,
        gpu_profile: &GpuProfile,
    ) {
        self.total_requests += batch_size as u64;
        self.total_tokens += tokens_processed as u64;
        
        // Update average batch size (exponential moving average)
        let alpha = 0.1;
        self.avg_batch_size = alpha * batch_size as f64 + (1.0 - alpha) * self.avg_batch_size;
        
        // Estimate SM utilization based on throughput
        let theoretical_max_tokens_per_ms = gpu_profile.tokens_per_s as f64 / 1000.0;
        let actual_tokens_per_ms = tokens_processed as f64 / service_time_ms;
        self.sm_utilization = (actual_tokens_per_ms / theoretical_max_tokens_per_ms).min(1.0) * 100.0;
        
        // TODO: Implement more sophisticated utilization modeling
    }

    /// Calculate effective cost per 1k tokens
    pub fn cost_per_1k_tokens(&self, gpu_cost_per_hour: f64) -> f64 {
        if self.total_tokens == 0 {
            return 0.0;
        }
        
        // Simplified cost calculation
        let tokens_per_1k = self.total_tokens as f64 / 1000.0;
        let utilization_factor = self.sm_utilization / 100.0;
        
        // Cost is amortized over utilization
        if utilization_factor > 0.0 {
            gpu_cost_per_hour / (utilization_factor * tokens_per_1k)
        } else {
            gpu_cost_per_hour
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Request, RequestType};

    fn create_test_gpu_profile() -> GpuProfile {
        GpuProfile {
            name: "H100-80G".to_string(),
            tokens_per_s: 240000,
            concurrency: 16,
            vram_total_gb: 80.0,
            batch_window_ms: 8.0,
            kv_cache_gb_per_req: 1.2,
        }
    }

    fn create_test_request(request_type: RequestType, input_tokens: u32, output_tokens: u32) -> Request {
        Request {
            id: 1,
            request_type,
            tenant_id: "test".to_string(),
            model_id: "test-model".to_string(),
            arrival_time: 0.0,
            input_tokens,
            expected_output_tokens: output_tokens,
            sla_ms: None,
        }
    }

    #[test]
    fn test_llm_service_time() {
        let gpu_profile = create_test_gpu_profile();
        let service_model = ServiceModel { request_type: RequestType::LLM };
        
        let requests = vec![
            create_test_request(RequestType::LLM, 100, 200),
            create_test_request(RequestType::LLM, 150, 300),
        ];
        
        let service_time = service_model.calculate_service_time(&requests, &gpu_profile);
        
        // Total tokens: (100+200) + (150+300) = 750
        // Expected time: 750 / 240000 * 1000 = 3.125 ms
        assert!((service_time - 3.125).abs() < 0.001);
    }

    #[test]
    fn test_vram_usage() {
        let gpu_profile = create_test_gpu_profile();
        let service_model = ServiceModel { request_type: RequestType::LLM };
        
        let requests = vec![
            create_test_request(RequestType::LLM, 100, 200),
            create_test_request(RequestType::LLM, 150, 300),
        ];
        
        let vram_usage = service_model.calculate_vram_usage(&requests, &gpu_profile);
        
        // Expected: 2 requests * 1.2 GB/req = 2.4 GB
        assert!((vram_usage - 2.4).abs() < 0.001);
    }

    #[test]
    fn test_mig_profile() {
        let mig_profile = MigProfile {
            name: "1g.10gb".to_string(),
            fraction: 0.125,
            tokens_per_s: 30000,
            concurrency: 2,
        };
        
        assert_eq!(mig_profile.vram_total_gb(), 10.0);
        assert_eq!(mig_profile.batch_window_ms(), 8.0);
    }

    #[test]
    fn test_batching_strategy() {
        let gpu_profile = create_test_gpu_profile();
        let strategy = BatchingStrategy::for_gpu_profile(&gpu_profile);
        
        assert_eq!(strategy.max_batch_size, 128); // 16 * 8
        assert_eq!(strategy.batch_window_ms, 8.0);
        
        assert!(strategy.should_close_batch(128, 5.0)); // Max size reached
        assert!(strategy.should_close_batch(50, 10.0)); // Time limit reached
        assert!(!strategy.should_close_batch(50, 5.0)); // Neither limit reached
    }
}
