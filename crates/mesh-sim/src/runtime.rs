use std::collections::{HashMap, VecDeque};
use crate::engine::{Request, ModelId, NodeId};
use crate::gpu::{GpuProfile, MigProfile, ServiceModel, BatchingStrategy};

/// A batch of requests being processed together
#[derive(Debug, Clone)]
pub struct Batch {
    pub id: BatchId,
    pub requests: Vec<Request>,
    pub model_id: ModelId,
    pub start_time: f64,
    pub estimated_completion_time: f64,
    pub vram_usage_gb: f64,
}

pub type BatchId = u64;

impl Batch {
    /// Create a new batch
    pub fn new(
        id: BatchId,
        requests: Vec<Request>,
        model_id: ModelId,
        start_time: f64,
        gpu_profile: &GpuProfile,
    ) -> Self {
        let service_model = ServiceModel {
            request_type: requests.first().map(|r| r.request_type).unwrap_or(crate::engine::RequestType::LLM),
        };

        let service_time = service_model.calculate_service_time(&requests, gpu_profile);
        let vram_usage = service_model.calculate_vram_usage(&requests, gpu_profile);

        Self {
            id,
            requests,
            model_id,
            start_time,
            estimated_completion_time: start_time + service_time,
            vram_usage_gb: vram_usage,
        }
    }

    /// Get the number of requests in this batch
    pub fn size(&self) -> usize {
        self.requests.len()
    }

    /// Get total input tokens in this batch
    pub fn total_input_tokens(&self) -> u32 {
        self.requests.iter().map(|r| r.input_tokens).sum()
    }

    /// Get total expected output tokens in this batch
    pub fn total_output_tokens(&self) -> u32 {
        self.requests.iter().map(|r| r.expected_output_tokens).sum()
    }

    /// Get total tokens (input + output) in this batch
    pub fn total_tokens(&self) -> u32 {
        self.total_input_tokens() + self.total_output_tokens()
    }

    /// Check if batch is complete
    pub fn is_complete(&self, current_time: f64) -> bool {
        current_time >= self.estimated_completion_time
    }

    /// Get remaining processing time
    pub fn remaining_time(&self, current_time: f64) -> f64 {
        (self.estimated_completion_time - current_time).max(0.0)
    }
}

/// Queue for requests waiting to be batched
#[derive(Debug, Clone)]
pub struct RequestQueue {
    pub model_id: ModelId,
    pub requests: VecDeque<Request>,
    pub batch_open_since: Option<f64>,
    pub total_queued: u64,
}

impl RequestQueue {
    pub fn new(model_id: ModelId) -> Self {
        Self {
            model_id,
            requests: VecDeque::new(),
            batch_open_since: None,
            total_queued: 0,
        }
    }

    /// Add a request to the queue
    pub fn enqueue(&mut self, request: Request) {
        self.requests.push_back(request);
        self.total_queued += 1;
    }

    /// Remove and return up to max_size requests from the front of the queue
    pub fn dequeue_batch(&mut self, max_size: usize) -> Vec<Request> {
        let batch_size = max_size.min(self.requests.len());
        let mut batch = Vec::with_capacity(batch_size);
        
        for _ in 0..batch_size {
            if let Some(request) = self.requests.pop_front() {
                batch.push(request);
            }
        }
        
        batch
    }

    /// Peek at the next request without removing it
    pub fn peek(&self) -> Option<&Request> {
        self.requests.front()
    }

    /// Get current queue depth
    pub fn depth(&self) -> usize {
        self.requests.len()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Get the oldest request's arrival time
    pub fn oldest_request_time(&self) -> Option<f64> {
        self.requests.front().map(|r| r.arrival_time)
    }

    /// Calculate average queue wait time for requests currently in queue
    pub fn average_queue_wait(&self, current_time: f64) -> f64 {
        if self.requests.is_empty() {
            return 0.0;
        }

        let total_wait: f64 = self.requests.iter()
            .map(|r| current_time - r.arrival_time)
            .sum();
        
        total_wait / self.requests.len() as f64
    }
}

/// Runtime state for a node or MIG instance
#[derive(Debug, Clone)]
pub struct RuntimeState {
    pub node_id: NodeId,
    pub gpu_profile: GpuProfile,
    pub mig_profile: Option<MigProfile>,
    pub queues: HashMap<ModelId, RequestQueue>,
    pub active_batches: HashMap<BatchId, Batch>,
    pub batching_strategy: BatchingStrategy,
    pub next_batch_id: BatchId,
    pub stats: RuntimeStats,
}

impl RuntimeState {
    /// Create a new runtime state for a GPU
    pub fn new_for_gpu(node_id: NodeId, gpu_profile: GpuProfile) -> Self {
        let batching_strategy = BatchingStrategy::for_gpu_profile(&gpu_profile);
        
        Self {
            node_id,
            gpu_profile,
            mig_profile: None,
            queues: HashMap::new(),
            active_batches: HashMap::new(),
            batching_strategy,
            next_batch_id: 1,
            stats: RuntimeStats::default(),
        }
    }

    /// Create a new runtime state for a MIG instance
    pub fn new_for_mig(node_id: NodeId, gpu_profile: GpuProfile, mig_profile: MigProfile) -> Self {
        let batching_strategy = BatchingStrategy::for_mig_profile(&mig_profile);
        
        Self {
            node_id,
            gpu_profile,
            mig_profile: Some(mig_profile),
            queues: HashMap::new(),
            active_batches: HashMap::new(),
            batching_strategy,
            next_batch_id: 1,
            stats: RuntimeStats::default(),
        }
    }

    /// Add a request to the appropriate queue
    pub fn enqueue_request(&mut self, request: Request, current_time: f64) -> bool {
        let model_id = request.model_id.clone();
        
        // Check if we can accept more requests (VRAM limit)
        if !self.can_accept_request(&request) {
            return false;
        }

        // Get or create queue for this model
        let queue = self.queues.entry(model_id.clone()).or_insert_with(|| {
            RequestQueue::new(model_id.clone())
        });

        queue.enqueue(request);
        self.stats.total_requests_queued += 1;

        // Open batch window if not already open
        if queue.batch_open_since.is_none() {
            queue.batch_open_since = Some(current_time);
        }

        true
    }

    /// Check if we can accept a new request based on VRAM constraints
    fn can_accept_request(&self, request: &Request) -> bool {
        let service_model = ServiceModel { request_type: request.request_type };
        let required_vram = service_model.calculate_vram_usage(&[request.clone()], &self.gpu_profile);
        
        let current_vram_usage = self.calculate_current_vram_usage();
        let total_vram = if let Some(ref mig_profile) = self.mig_profile {
            mig_profile.vram_total_gb()
        } else {
            self.gpu_profile.vram_total_gb
        };
        let available_vram = total_vram - current_vram_usage;
        
        required_vram <= available_vram
    }

    /// Calculate current VRAM usage from active batches
    fn calculate_current_vram_usage(&self) -> f64 {
        self.active_batches.values().map(|batch| batch.vram_usage_gb).sum()
    }

    /// Get total available VRAM
    fn get_total_vram(&self) -> f64 {
        if let Some(ref mig_profile) = self.mig_profile {
            mig_profile.vram_total_gb()
        } else {
            self.gpu_profile.vram_total_gb
        }
    }

    /// Get maximum concurrency
    fn get_max_concurrency(&self) -> u32 {
        if let Some(ref mig_profile) = self.mig_profile {
            mig_profile.concurrency
        } else {
            self.gpu_profile.concurrency
        }
    }

    /// Check if we should close a batch for a model
    pub fn should_close_batch(&self, model_id: &ModelId, current_time: f64) -> bool {
        if let Some(queue) = self.queues.get(model_id) {
            if let Some(batch_open_since) = queue.batch_open_since {
                let batch_duration = current_time - batch_open_since;
                return self.batching_strategy.should_close_batch(queue.depth(), batch_duration);
            }
        }
        false
    }

    /// Close a batch and start processing
    pub fn close_batch(&mut self, model_id: &ModelId, current_time: f64) -> Option<Batch> {
        // Check concurrency limit first
        let max_concurrency = if let Some(ref mig_profile) = self.mig_profile {
            mig_profile.concurrency
        } else {
            self.gpu_profile.concurrency
        };
        
        if self.active_batches.len() >= max_concurrency as usize {
            return None;
        }

        // Calculate max batch size
        let max_batch_size = {
            let queue = self.queues.get(model_id)?;
            if queue.is_empty() {
                return None;
            }

            if let Some(sample_request) = queue.peek() {
                let service_model = ServiceModel { request_type: sample_request.request_type };
                let current_vram_usage = self.calculate_current_vram_usage();
                let total_vram = if let Some(ref mig_profile) = self.mig_profile {
                    mig_profile.vram_total_gb()
                } else {
                    self.gpu_profile.vram_total_gb
                };
                let available_vram = total_vram - current_vram_usage;
                
                let vram_limited_size = service_model.max_batch_size_for_vram(
                    sample_request,
                    &self.gpu_profile,
                    available_vram,
                );

                // Take minimum of strategy limit and VRAM limit
                vram_limited_size.min(self.batching_strategy.max_batch_size)
            } else {
                self.batching_strategy.max_batch_size
            }
        };

        // Get queue and dequeue requests
        let queue = self.queues.get_mut(model_id)?;
        let requests = queue.dequeue_batch(max_batch_size);
        
        if requests.is_empty() {
            return None;
        }

        // Create batch
        let batch_id = self.next_batch_id;
        self.next_batch_id += 1;

        let batch = Batch::new(
            batch_id,
            requests,
            model_id.clone(),
            current_time,
            &self.gpu_profile,
        );

        // Update stats
        self.stats.total_batches_processed += 1;
        self.stats.total_batch_size += batch.size() as u64;

        // Add to active batches
        self.active_batches.insert(batch_id, batch.clone());

        // Reset batch window
        queue.batch_open_since = if queue.is_empty() { None } else { Some(current_time) };

        Some(batch)
    }

    /// Complete a batch and remove it from active processing
    pub fn complete_batch(&mut self, batch_id: BatchId, current_time: f64) -> Option<Batch> {
        if let Some(batch) = self.active_batches.remove(&batch_id) {
            // Update stats
            self.stats.total_requests_completed += batch.size() as u64;
            self.stats.total_tokens_processed += batch.total_tokens() as u64;
            
            let processing_time = current_time - batch.start_time;
            self.stats.total_processing_time += processing_time;

            Some(batch)
        } else {
            None
        }
    }

    /// Get all batches that should complete by the given time
    pub fn get_completing_batches(&self, current_time: f64) -> Vec<BatchId> {
        self.active_batches.iter()
            .filter(|(_, batch)| batch.is_complete(current_time))
            .map(|(batch_id, _)| *batch_id)
            .collect()
    }

    /// Get current utilization (0.0 to 1.0)
    pub fn get_utilization(&self) -> f64 {
        let max_concurrency = self.get_max_concurrency() as f64;
        if max_concurrency == 0.0 {
            0.0
        } else {
            self.active_batches.len() as f64 / max_concurrency
        }
    }

    /// Get current VRAM utilization (0.0 to 1.0)
    pub fn get_vram_utilization(&self) -> f64 {
        let total_vram = self.get_total_vram();
        if total_vram == 0.0 {
            0.0
        } else {
            self.calculate_current_vram_usage() / total_vram
        }
    }

    /// Get queue depth for a specific model
    pub fn get_queue_depth(&self, model_id: &ModelId) -> usize {
        self.queues.get(model_id).map(|q| q.depth()).unwrap_or(0)
    }

    /// Get total queue depth across all models
    pub fn get_total_queue_depth(&self) -> usize {
        self.queues.values().map(|q| q.depth()).sum()
    }

    /// Get average batch processing time
    pub fn get_average_batch_time(&self) -> f64 {
        if self.stats.total_batches_processed == 0 {
            0.0
        } else {
            self.stats.total_processing_time / self.stats.total_batches_processed as f64
        }
    }

    /// Get average batch size
    pub fn get_average_batch_size(&self) -> f64 {
        if self.stats.total_batches_processed == 0 {
            0.0
        } else {
            self.stats.total_batch_size as f64 / self.stats.total_batches_processed as f64
        }
    }

    /// Get tokens per second throughput
    pub fn get_tokens_per_second(&self, time_window_s: f64) -> f64 {
        if time_window_s <= 0.0 {
            0.0
        } else {
            self.stats.total_tokens_processed as f64 / time_window_s
        }
    }
}

/// Runtime statistics
#[derive(Debug, Clone, Default)]
pub struct RuntimeStats {
    pub total_requests_queued: u64,
    pub total_requests_completed: u64,
    pub total_batches_processed: u64,
    pub total_batch_size: u64,
    pub total_tokens_processed: u64,
    pub total_processing_time: f64,
    pub total_queue_wait_time: f64,
}

impl RuntimeStats {
    /// Calculate completion rate
    pub fn completion_rate(&self) -> f64 {
        if self.total_requests_queued == 0 {
            0.0
        } else {
            self.total_requests_completed as f64 / self.total_requests_queued as f64
        }
    }

    /// Calculate average queue wait time
    pub fn average_queue_wait(&self) -> f64 {
        if self.total_requests_completed == 0 {
            0.0
        } else {
            self.total_queue_wait_time / self.total_requests_completed as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Request, RequestType};
    use crate::gpu::GpuProfile;

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

    fn create_test_request(id: u64, input_tokens: u32, output_tokens: u32) -> Request {
        Request {
            id,
            request_type: RequestType::LLM,
            tenant_id: "test".to_string(),
            model_id: "test-model".to_string(),
            arrival_time: 0.0,
            input_tokens,
            expected_output_tokens: output_tokens,
            sla_ms: None,
        }
    }

    #[test]
    fn test_batch_creation() {
        let gpu_profile = create_test_gpu_profile();
        let requests = vec![
            create_test_request(1, 100, 200),
            create_test_request(2, 150, 300),
        ];

        let batch = Batch::new(1, requests, "test-model".to_string(), 1000.0, &gpu_profile);

        assert_eq!(batch.id, 1);
        assert_eq!(batch.size(), 2);
        assert_eq!(batch.total_input_tokens(), 250);
        assert_eq!(batch.total_output_tokens(), 500);
        assert_eq!(batch.total_tokens(), 750);
        assert!(batch.estimated_completion_time > 1000.0);
    }

    #[test]
    fn test_request_queue() {
        let mut queue = RequestQueue::new("test-model".to_string());
        
        assert!(queue.is_empty());
        assert_eq!(queue.depth(), 0);

        // Add requests
        queue.enqueue(create_test_request(1, 100, 200));
        queue.enqueue(create_test_request(2, 150, 300));

        assert!(!queue.is_empty());
        assert_eq!(queue.depth(), 2);
        assert_eq!(queue.total_queued, 2);

        // Dequeue batch
        let batch_requests = queue.dequeue_batch(1);
        assert_eq!(batch_requests.len(), 1);
        assert_eq!(batch_requests[0].id, 1);
        assert_eq!(queue.depth(), 1);

        // Dequeue remaining
        let remaining = queue.dequeue_batch(10);
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, 2);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_runtime_state() {
        let gpu_profile = create_test_gpu_profile();
        let mut runtime = RuntimeState::new_for_gpu(1, gpu_profile);

        // Enqueue requests
        let request1 = create_test_request(1, 100, 200);
        let request2 = create_test_request(2, 150, 300);

        assert!(runtime.enqueue_request(request1, 1000.0));
        assert!(runtime.enqueue_request(request2, 1001.0));

        assert_eq!(runtime.get_queue_depth(&"test-model".to_string()), 2);
        assert_eq!(runtime.get_total_queue_depth(), 2);

        // Should close batch after window
        assert!(runtime.should_close_batch(&"test-model".to_string(), 1010.0));

        // Close batch
        let batch = runtime.close_batch(&"test-model".to_string(), 1010.0);
        assert!(batch.is_some());

        let batch = batch.unwrap();
        assert_eq!(batch.size(), 2);
        assert_eq!(runtime.active_batches.len(), 1);
        assert_eq!(runtime.get_queue_depth(&"test-model".to_string()), 0);

        // Complete batch
        let completed = runtime.complete_batch(batch.id, batch.estimated_completion_time);
        assert!(completed.is_some());
        assert_eq!(runtime.active_batches.len(), 0);
        assert_eq!(runtime.stats.total_requests_completed, 2);
    }

    #[test]
    fn test_vram_constraints() {
        let gpu_profile = create_test_gpu_profile();
        let mut runtime = RuntimeState::new_for_gpu(1, gpu_profile);

        // Create requests that would exceed VRAM
        // Each request uses ~1.2GB, so 70 requests would use ~84GB > 80GB limit
        for i in 1..=70 {
            let request = create_test_request(i, 100, 200);
            let accepted = runtime.enqueue_request(request, 1000.0);
            
            if i <= 66 { // ~79.2GB should be accepted
                assert!(accepted, "Request {} should be accepted", i);
            } else { // Beyond VRAM limit should be rejected
                assert!(!accepted, "Request {} should be rejected due to VRAM limit", i);
            }
        }
    }

    #[test]
    fn test_utilization_metrics() {
        let gpu_profile = create_test_gpu_profile();
        let mut runtime = RuntimeState::new_for_gpu(1, gpu_profile);

        // Initially no utilization
        assert_eq!(runtime.get_utilization(), 0.0);
        assert_eq!(runtime.get_vram_utilization(), 0.0);

        // Add and process some requests
        runtime.enqueue_request(create_test_request(1, 100, 200), 1000.0);
        runtime.enqueue_request(create_test_request(2, 150, 300), 1001.0);

        let batch = runtime.close_batch(&"test-model".to_string(), 1010.0).unwrap();
        
        // Should have some utilization now
        assert!(runtime.get_utilization() > 0.0);
        assert!(runtime.get_vram_utilization() > 0.0);

        // Complete batch
        runtime.complete_batch(batch.id, batch.estimated_completion_time);
        
        // Utilization should drop back to 0
        assert_eq!(runtime.get_utilization(), 0.0);
        assert_eq!(runtime.get_vram_utilization(), 0.0);
    }
}
