//! Scoring service implementation

use crate::config::AgentConfig;
use mesh_metrics::MetricsRegistry;
use mesh_proto::scoring::v1::{
    scoring_server::Scoring, AdmitRequest, AdmitResponse, HealthCheckRequest, HealthCheckResponse,
    HealthStatus, ReportOutcomeRequest, ReportOutcomeResponse, RequestOutcome, ScoredTarget,
    ScoreTargetsRequest, ScoreTargetsResponse,
};

use std::collections::HashMap;
use tonic::{Request, Response, Status};
use tracing::{debug, info, warn};

/// Scoring service implementation
#[derive(Debug, Clone)]
pub struct ScoringService {
    #[allow(unused)]
    config: AgentConfig,
    metrics_registry: MetricsRegistry,
    // In a real implementation, these would track active requests and admission tokens
    active_requests: std::sync::Arc<tokio::sync::RwLock<HashMap<String, AdmissionRecord>>>,
}

#[derive(Debug, Clone)]
struct AdmissionRecord {
    #[allow(unused)]
    request_id: String,
    admission_token: String,
    target_node: String,
    #[allow(unused)]
    target_gpu: String,
    #[allow(unused)]
    admitted_at: chrono::DateTime<chrono::Utc>,
    #[allow(unused)]
    estimated_tokens: u32,
}

impl ScoringService {
    /// Create a new Scoring service
    pub fn new(config: AgentConfig, metrics_registry: MetricsRegistry) -> Self {
        Self {
            config,
            metrics_registry,
            active_requests: std::sync::Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    /// Generate mock scored targets for a request
    fn generate_mock_targets(&self, request: &ScoreTargetsRequest) -> Vec<ScoredTarget> {
        let mut targets = Vec::new();

        // Generate 1-3 mock targets
        let target_count = (rand::random::<f64>() * 3.0).floor() as usize + 1;
        
        for i in 0..target_count {
            let node_id = format!("node-{}", i + 1);
            let gpu_uuid = format!("GPU-{:08X}", rand::random::<u32>());
            
            // Generate realistic scores (higher is better)
            let base_score = rand::random::<f64>() * 0.7 + 0.3; // 0.3 to 1.0
            let load_factor = rand::random::<f64>() * 0.8 + 0.1; // 0.1 to 0.9
            let score = base_score * (1.0 - load_factor * 0.5);
            
            // Estimate latency based on queue and model complexity
            let base_latency = match request.estimated_tokens {
                0..=100 => rand::random::<f64>() * 150.0 + 50.0, // 50.0 to 200.0
                101..=500 => rand::random::<f64>() * 600.0 + 200.0, // 200.0 to 800.0
                501..=1000 => rand::random::<f64>() * 1200.0 + 800.0, // 800.0 to 2000.0
                _ => rand::random::<f64>() * 3000.0 + 2000.0, // 2000.0 to 5000.0
            };
            
            let queue_time = load_factor * 500.0; // Higher load = more queue time
            
            let target = ScoredTarget {
                node_id: node_id.clone(),
                gpu_uuid: gpu_uuid.clone(),
                score: score as f32,
                estimated_latency_ms: base_latency as f32,
                estimated_queue_time_ms: queue_time as f32,
                load_factor: load_factor as f32,
                reason: format!("Mock target {} with score {:.3}", node_id, score),
            };
            
            targets.push(target);
        }

        // Sort by score (descending)
        targets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        targets
    }
}

#[tonic::async_trait]
impl Scoring for ScoringService {
    async fn score_targets(
        &self,
        request: Request<ScoreTargetsRequest>,
    ) -> std::result::Result<Response<ScoreTargetsResponse>, Status> {
        debug!("Received ScoreTargets request: {:?}", request);

        let req = request.into_inner();
        
        info!("Scoring targets for model {}:{} with {} tokens", 
              req.model, req.revision, req.estimated_tokens);

        // In a real implementation, we would:
        // 1. Query available nodes running the requested model
        // 2. Get current load and capacity information
        // 3. Apply scoring algorithms based on SLO class
        // 4. Consider network topology and affinity rules
        // 5. Return ranked list of targets

        let targets = self.generate_mock_targets(&req);
        
        info!("Generated {} targets for request {}", targets.len(), req.request_id);

        let response = ScoreTargetsResponse {
            targets,
            request_id: req.request_id,
            timestamp: Some(mesh_proto::timestamp::now()),
        };

        Ok(Response::new(response))
    }

    async fn admit(
        &self,
        request: Request<AdmitRequest>,
    ) -> std::result::Result<Response<AdmitResponse>, Status> {
        debug!("Received Admit request: {:?}", request);

        let req = request.into_inner();
        
        info!("Admission request for model {}:{} on {}:{}", 
              req.model, req.revision, req.target_node, req.target_gpu);

        // In a real implementation, we would:
        // 1. Check if the target node/GPU is still available
        // 2. Verify capacity and queue limits
        // 3. Apply admission control policies
        // 4. Reserve resources if admitted
        // 5. Generate admission token for tracking

        // Mock admission logic - admit 90% of requests
        let admitted = rand::random::<f64>() < 0.9;
        
        if admitted {
            let admission_token = uuid::Uuid::new_v4().to_string();
            
            // Record the admission
            let record = AdmissionRecord {
                request_id: req.request_id.clone(),
                admission_token: admission_token.clone(),
                target_node: req.target_node.clone(),
                target_gpu: req.target_gpu.clone(),
                admitted_at: chrono::Utc::now(),
                estimated_tokens: req.estimated_tokens,
            };
            
            let mut active_requests = self.active_requests.write().await;
            active_requests.insert(req.request_id.clone(), record);
            
            // Estimate wait time based on current load
            let estimated_wait_time = rand::random::<f64>() * 490.0 + 10.0; // 10.0 to 500.0
            
            let response = AdmitResponse {
                admitted: true,
                reason: "Request admitted".to_string(),
                estimated_wait_time_ms: estimated_wait_time as f32,
                request_id: req.request_id,
                admission_token,
            };
            
            info!("Request admitted with token");
            Ok(Response::new(response))
        } else {
            let response = AdmitResponse {
                admitted: false,
                reason: "Target overloaded".to_string(),
                estimated_wait_time_ms: 0.0,
                request_id: req.request_id,
                admission_token: String::new(),
            };
            
            warn!("Request rejected due to overload");
            Ok(Response::new(response))
        }
    }

    async fn report_outcome(
        &self,
        request: Request<ReportOutcomeRequest>,
    ) -> std::result::Result<Response<ReportOutcomeResponse>, Status> {
        debug!("Received ReportOutcome request: {:?}", request);

        let req = request.into_inner();
        
        // Verify admission token
        let mut active_requests = self.active_requests.write().await;
        if let Some(record) = active_requests.remove(&req.request_id) {
            if record.admission_token != req.admission_token {
                return Err(Status::permission_denied("Invalid admission token"));
            }
            
            // Record metrics
            let outcome_str = match RequestOutcome::try_from(req.outcome) {
                Ok(RequestOutcome::Success) => "success",
                Ok(RequestOutcome::Timeout) => "timeout",
                Ok(RequestOutcome::Error) => "error",
                Ok(RequestOutcome::Rejected) => "rejected",
                Ok(RequestOutcome::Cancelled) => "cancelled",
                _ => "unknown",
            };
            
            self.metrics_registry.record_inference_request(
                &record.target_node,
                &req.target_node, // Using target_node as model for now
                "latency", // Mock SLO class
                req.actual_latency_ms as f64 / 1000.0,
                req.actual_queue_time_ms as f64 / 1000.0,
                outcome_str,
                req.actual_tokens as u64,
            );
            
            info!("Recorded outcome for request {}: {} ({}ms latency, {} tokens)", 
                  req.request_id, outcome_str, req.actual_latency_ms, req.actual_tokens);
            
            let response = ReportOutcomeResponse {
                success: true,
                message: "Outcome recorded successfully".to_string(),
            };
            
            Ok(Response::new(response))
        } else {
            warn!("Unknown request ID: {}", req.request_id);
            Err(Status::not_found("Request not found"))
        }
    }

    async fn health_check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> std::result::Result<Response<HealthCheckResponse>, Status> {
        debug!("Received HealthCheck request: {:?}", request);

        let req = request.into_inner();
        
        // Check service health
        let status = if req.service.is_empty() || req.service == "scoring" {
            HealthStatus::Serving
        } else {
            HealthStatus::Unknown
        };
        
        let mut details = HashMap::new();
        details.insert("service".to_string(), "scoring".to_string());
        details.insert("version".to_string(), env!("CARGO_PKG_VERSION").to_string());
        
        let active_count = self.active_requests.read().await.len();
        details.insert("active_requests".to_string(), active_count.to_string());
        
        let response = HealthCheckResponse {
            status: status as i32,
            message: match status {
                HealthStatus::Serving => "Scoring service is healthy".to_string(),
                _ => "Unknown service".to_string(),
            },
            details,
        };

        Ok(Response::new(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_metrics::MetricsRegistryBuilder;

    fn create_test_service() -> ScoringService {
        let config = AgentConfig::default();
        let metrics_registry = MetricsRegistryBuilder::new().build().unwrap();
        ScoringService::new(config, metrics_registry)
    }

    #[tokio::test]
    async fn test_score_targets() {
        let service = create_test_service();
        let request = Request::new(ScoreTargetsRequest {
            model: "test-model".to_string(),
            revision: "v1.0".to_string(),
            slo_class: 1, // Latency
            estimated_tokens: 100,
            timeout_seconds: 30.0,
            filters: HashMap::new(),
            request_id: "test-request".to_string(),
        });

        let response = service.score_targets(request).await.unwrap();
        let result = response.into_inner();
        
        assert!(!result.targets.is_empty());
        assert_eq!(result.request_id, "test-request");
        
        // Verify targets are sorted by score (descending)
        for i in 1..result.targets.len() {
            assert!(result.targets[i-1].score >= result.targets[i].score);
        }
    }

    #[tokio::test]
    async fn test_admit_and_report_outcome() {
        let service = create_test_service();
        
        // Test admission
        let admit_request = Request::new(AdmitRequest {
            model: "test-model".to_string(),
            revision: "v1.0".to_string(),
            target_node: "node1".to_string(),
            target_gpu: "GPU-12345".to_string(),
            slo_class: 1,
            estimated_tokens: 100,
            timeout_seconds: 30.0,
            request_id: "test-request".to_string(),
        });

        let admit_response = service.admit(admit_request).await.unwrap();
        let admit_result = admit_response.into_inner();
        
        if admit_result.admitted {
            // Test outcome reporting
            let outcome_request = Request::new(ReportOutcomeRequest {
                request_id: admit_result.request_id,
                admission_token: admit_result.admission_token,
                target_node: "node1".to_string(),
                target_gpu: "GPU-12345".to_string(),
                outcome: RequestOutcome::Success as i32,
                actual_latency_ms: 150.0,
                actual_queue_time_ms: 50.0,
                actual_tokens: 100,
                error_message: String::new(),
                completed_at: Some(mesh_proto::timestamp::now()),
            });

            let outcome_response = service.report_outcome(outcome_request).await.unwrap();
            let outcome_result = outcome_response.into_inner();
            
            assert!(outcome_result.success);
        }
    }

    #[tokio::test]
    async fn test_health_check() {
        let service = create_test_service();
        let request = Request::new(HealthCheckRequest {
            service: "scoring".to_string(),
        });

        let response = service.health_check(request).await.unwrap();
        let result = response.into_inner();
        
        assert_eq!(result.status, HealthStatus::Serving as i32);
        assert!(result.details.contains_key("service"));
        assert!(result.details.contains_key("version"));
    }

    #[tokio::test]
    async fn test_report_outcome_invalid_token() {
        let service = create_test_service();
        let request = Request::new(ReportOutcomeRequest {
            request_id: "nonexistent".to_string(),
            admission_token: "invalid".to_string(),
            target_node: "node1".to_string(),
            target_gpu: "GPU-12345".to_string(),
            outcome: RequestOutcome::Success as i32,
            actual_latency_ms: 150.0,
            actual_queue_time_ms: 50.0,
            actual_tokens: 100,
            error_message: String::new(),
            completed_at: Some(mesh_proto::timestamp::now()),
        });

        let result = service.report_outcome(request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::NotFound);
    }
}
