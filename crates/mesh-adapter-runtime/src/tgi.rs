//! Text Generation Inference (TGI) adapter

use crate::adapter::{RuntimeAdapterTrait, ModelInfo, ModelStatus, ModelMetadata, InferenceRequest, InferenceResponse, TensorData, ResponseMetadata};
use crate::config::{RuntimeConfig, RuntimeType, ModelConfig};
use crate::health::{HealthStatus, HealthInfo, CheckResult};
use crate::metrics::{RuntimeMetrics, MetricCollector};
use crate::{Result, RuntimeError};

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// TGI adapter
pub struct TgiAdapter {
    config: RuntimeConfig,
    client: Client,
    metric_collector: RwLock<MetricCollector>,
}

/// TGI health response
#[derive(Debug, Deserialize)]
struct TgiHealthResponse {
    status: String,
}

/// TGI info response
#[derive(Debug, Deserialize)]
struct TgiInfoResponse {
    model_id: String,
    model_sha: Option<String>,
    model_dtype: String,
    model_device_type: String,
    model_pipeline_tag: Option<String>,
    max_concurrent_requests: u32,
    max_best_of: u32,
    max_stop_sequences: u32,
    max_input_length: u32,
    max_total_tokens: u32,
    waiting_served_ratio: f32,
    max_batch_total_tokens: u32,
    max_waiting_tokens: u32,
    validation_workers: u32,
    version: String,
    sha: Option<String>,
    docker_label: Option<String>,
}

/// TGI generate request
#[derive(Debug, Serialize)]
struct TgiGenerateRequest {
    inputs: String,
    parameters: Option<TgiParameters>,
}

/// TGI generation parameters
#[derive(Debug, Serialize)]
struct TgiParameters {
    best_of: Option<u32>,
    decoder_input_details: Option<bool>,
    details: Option<bool>,
    do_sample: Option<bool>,
    max_new_tokens: Option<u32>,
    repetition_penalty: Option<f32>,
    return_full_text: Option<bool>,
    seed: Option<u64>,
    stop: Option<Vec<String>>,
    stream: Option<bool>,
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    truncate: Option<u32>,
    typical_p: Option<f32>,
    watermark: Option<bool>,
}

/// TGI generate response
#[derive(Debug, Deserialize)]
struct TgiGenerateResponse {
    generated_text: String,
    details: Option<TgiGenerationDetails>,
}

/// TGI generation details
#[derive(Debug, Deserialize)]
struct TgiGenerationDetails {
    finish_reason: String,
    generated_tokens: u32,
    seed: Option<u64>,
    prefill: Vec<TgiPrefillToken>,
    tokens: Vec<TgiToken>,
    best_of_sequences: Option<Vec<TgiSequence>>,
}

/// TGI prefill token
#[derive(Debug, Serialize, Deserialize)]
struct TgiPrefillToken {
    id: u32,
    text: String,
    logprob: f32,
}

/// TGI token
#[derive(Debug, Serialize, Deserialize)]
struct TgiToken {
    id: u32,
    text: String,
    logprob: f32,
    special: bool,
}

/// TGI sequence (for best_of)
#[derive(Debug, Serialize, Deserialize)]
struct TgiSequence {
    generated_text: String,
    finish_reason: String,
    generated_tokens: u32,
    seed: Option<u64>,
    prefill: Vec<TgiPrefillToken>,
    tokens: Vec<TgiToken>,
}

/// TGI metrics response
#[derive(Debug, Deserialize)]
struct TgiMetricsResponse {
    // TGI exposes Prometheus metrics, we'd parse them here
    // For now, we'll use a simple string representation
    metrics: String,
}

impl TgiAdapter {
    /// Create a new TGI adapter
    pub async fn new(config: RuntimeConfig) -> Result<Self> {
        info!("Creating TGI adapter for endpoint: {}", config.endpoint);

        let client = Client::builder()
            .timeout(config.request_timeout)
            .build()
            .map_err(|e| RuntimeError::Configuration(format!("Failed to create HTTP client: {}", e)))?;

        let metric_collector = RwLock::new(MetricCollector::new(config.metrics.clone()));

        Ok(Self {
            config,
            client,
            metric_collector,
        })
    }

    /// Check TGI health
    async fn check_health(&self) -> Result<TgiHealthResponse> {
        let url = format!("{}/health", self.config.endpoint);
        
        debug!("Checking TGI health at: {}", url);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| RuntimeError::Connection(format!("Failed to connect to TGI server: {}", e)))?;

        if !response.status().is_success() {
            return Err(RuntimeError::HealthCheck(
                format!("Health check failed: {}", response.status())
            ));
        }

        // TGI health endpoint returns 200 OK with no body when healthy
        Ok(TgiHealthResponse {
            status: "healthy".to_string(),
        })
    }

    /// Get TGI server info
    async fn get_info(&self) -> Result<TgiInfoResponse> {
        let url = format!("{}/info", self.config.endpoint);
        
        debug!("Fetching TGI server info from: {}", url);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| RuntimeError::Connection(format!("Failed to get server info: {}", e)))?;

        if !response.status().is_success() {
            return Err(RuntimeError::InvalidResponse(
                format!("Info request failed: {}", response.status())
            ));
        }

        let info: TgiInfoResponse = response
            .json()
            .await
            .map_err(|e| RuntimeError::InvalidResponse(format!("Invalid info response: {}", e)))?;

        Ok(info)
    }

    /// Generate text
    async fn generate(&self, request: TgiGenerateRequest) -> Result<TgiGenerateResponse> {
        let url = format!("{}/generate", self.config.endpoint);
        
        debug!("Sending generation request to TGI: {:?}", request);
        
        let response = self.client
            .post(&url)
            .json(&request)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| RuntimeError::Connection(format!("Generation request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            warn!("TGI generation failed: {} - {}", status, error_text);
            return Err(RuntimeError::Model(
                format!("Generation failed: {} - {}", status, error_text)
            ));
        }

        let generation: TgiGenerateResponse = response
            .json()
            .await
            .map_err(|e| RuntimeError::InvalidResponse(format!("Invalid generation response: {}", e)))?;

        debug!("TGI generation successful: {} tokens", 
               generation.details.as_ref().map(|d| d.generated_tokens).unwrap_or(0));

        Ok(generation)
    }

    /// Generate text with streaming (for future implementation)
    async fn generate_stream(&self, request: TgiGenerateRequest) -> Result<TgiGenerateResponse> {
        // For now, fall back to non-streaming generation
        // In a full implementation, this would handle Server-Sent Events (SSE)
        warn!("Streaming generation requested but not fully implemented, falling back to non-streaming");
        self.generate(request).await
    }

    /// Get TGI metrics (Prometheus format)
    async fn get_tgi_metrics(&self) -> Result<String> {
        let url = format!("{}/metrics", self.config.endpoint);
        
        debug!("Fetching TGI metrics from: {}", url);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| RuntimeError::Connection(format!("Failed to get metrics: {}", e)))?;

        if !response.status().is_success() {
            return Err(RuntimeError::Metrics(
                format!("Metrics request failed: {}", response.status())
            ));
        }

        let metrics_text = response
            .text()
            .await
            .map_err(|e| RuntimeError::InvalidResponse(format!("Invalid metrics response: {}", e)))?;

        Ok(metrics_text)
    }
}

#[async_trait]
impl RuntimeAdapterTrait for TgiAdapter {
    fn runtime_type(&self) -> RuntimeType {
        RuntimeType::Tgi
    }

    async fn initialize(&mut self) -> Result<()> {
        info!("Initializing TGI adapter");
        
        // Test connection by checking health and getting info
        let _health = self.check_health().await?;
        let _info = self.get_info().await?;
        
        info!("TGI adapter initialized successfully");
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down TGI adapter");
        Ok(())
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        let start_time = Instant::now();
        
        match self.check_health().await {
            Ok(_) => {
                let response_time = start_time.elapsed().as_millis() as f64;
                
                // Also try to get server info for more detailed health check
                match self.get_info().await {
                    Ok(info) => {
                        let health_info = HealthInfo::healthy()
                            .with_response_time(response_time)
                            .add_check(CheckResult::pass("server_connectivity".to_string()))
                            .add_check(CheckResult::pass("health_endpoint".to_string()))
                            .add_check(CheckResult::pass("info_endpoint".to_string()))
                            .add_runtime_data("model_id".to_string(), serde_json::Value::String(info.model_id))
                            .add_runtime_data("version".to_string(), serde_json::Value::String(info.version))
                            .add_runtime_data("max_input_length".to_string(), serde_json::Value::Number(info.max_input_length.into()))
                            .add_runtime_data("max_total_tokens".to_string(), serde_json::Value::Number(info.max_total_tokens.into()));
                        
                        Ok(health_info.status)
                    }
                    Err(e) => {
                        warn!("TGI info check failed: {}", e);
                        let health_info = HealthInfo::healthy()
                            .with_response_time(response_time)
                            .add_check(CheckResult::pass("server_connectivity".to_string()))
                            .add_check(CheckResult::pass("health_endpoint".to_string()))
                            .add_check(CheckResult::warn("info_endpoint".to_string(), e.to_string()));
                        
                        Ok(health_info.status)
                    }
                }
            }
            Err(e) => {
                warn!("TGI health check failed: {}", e);
                Ok(HealthStatus::Unhealthy(e.to_string()))
            }
        }
    }

    async fn load_model(&self, name: &str, _config: Option<ModelConfig>) -> Result<()> {
        // TGI loads a single model at startup, doesn't support dynamic loading
        warn!("TGI does not support dynamic model loading. Model {} should be loaded at startup.", name);
        Ok(())
    }

    async fn unload_model(&self, name: &str) -> Result<()> {
        // TGI doesn't support dynamic model unloading
        warn!("TGI does not support dynamic model unloading. Model {} remains loaded.", name);
        Ok(())
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        // TGI serves a single model, get it from server info
        match self.get_info().await {
            Ok(info) => Ok(vec![info.model_id]),
            Err(e) => {
                warn!("Failed to get model info: {}", e);
                Ok(vec![])
            }
        }
    }

    async fn get_model_info(&self, name: &str) -> Result<ModelInfo> {
        let info = self.get_info().await?;
        
        if info.model_id != name {
            return Err(RuntimeError::Model(format!("Model {} not found, only {} is available", name, info.model_id)));
        }

        // TGI models are text-based, so no fixed tensor shapes
        let metadata = ModelMetadata {
            input_shapes: vec![],
            output_shapes: vec![],
            batch_size: None,
            max_sequence_length: Some(info.max_input_length),
            memory_usage: None,
        };

        let mut config = HashMap::new();
        config.insert("max_input_length".to_string(), serde_json::Value::Number(info.max_input_length.into()));
        config.insert("max_total_tokens".to_string(), serde_json::Value::Number(info.max_total_tokens.into()));
        config.insert("max_concurrent_requests".to_string(), serde_json::Value::Number(info.max_concurrent_requests.into()));
        config.insert("model_dtype".to_string(), serde_json::Value::String(info.model_dtype));
        config.insert("model_device_type".to_string(), serde_json::Value::String(info.model_device_type));

        Ok(ModelInfo {
            name: info.model_id,
            version: info.model_sha,
            status: ModelStatus::Ready, // TGI model is ready if server is responding
            config,
            metadata,
        })
    }

    async fn inference(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let start_time = Instant::now();
        
        debug!("Processing inference request for model: {}", request.model_name);

        // Extract prompt from inputs (assuming text input)
        let prompt = request.inputs.get("prompt")
            .or_else(|| request.inputs.get("text"))
            .or_else(|| request.inputs.get("input"))
            .and_then(|tensor| {
                // In a real implementation, we'd properly decode the tensor data
                // For now, assume it's a simple string
                String::from_utf8(tensor.data.clone()).ok()
            })
            .ok_or_else(|| RuntimeError::Model("No text prompt found in inputs".to_string()))?;

        // Extract parameters with defaults and validation
        let max_new_tokens = request.parameters.get("max_new_tokens")
            .or_else(|| request.parameters.get("max_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .or_else(|| Some(100)); // Default max tokens
        
        let temperature = request.parameters.get("temperature")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .map(|t| t.max(0.0).min(2.0)); // Clamp temperature to valid range

        let top_p = request.parameters.get("top_p")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .map(|p| p.max(0.0).min(1.0)); // Clamp top_p to valid range

        let top_k = request.parameters.get("top_k")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .map(|k| k.max(1)); // Ensure top_k is at least 1

        let do_sample = request.parameters.get("do_sample")
            .and_then(|v| v.as_bool())
            .or_else(|| Some(temperature.is_some() || top_p.is_some() || top_k.is_some())); // Auto-enable sampling if sampling params are set

        let stop = request.parameters.get("stop")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            });

        // Additional TGI-specific parameters
        let repetition_penalty = request.parameters.get("repetition_penalty")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .map(|p| p.max(0.0).min(2.0)); // Clamp to reasonable range

        let typical_p = request.parameters.get("typical_p")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .map(|p| p.max(0.0).min(1.0));

        let truncate = request.parameters.get("truncate")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        let seed = request.parameters.get("seed")
            .and_then(|v| v.as_u64());

        let best_of = request.parameters.get("best_of")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .map(|b| b.max(1)); // Ensure best_of is at least 1

        let return_full_text = request.parameters.get("return_full_text")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let details = request.parameters.get("details")
            .and_then(|v| v.as_bool())
            .unwrap_or(true); // Enable details by default for better metrics

        let stream = request.parameters.get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false); // Disable streaming by default for simplicity

        let parameters = TgiParameters {
            best_of,
            decoder_input_details: Some(false),
            details: Some(details),
            do_sample,
            max_new_tokens,
            repetition_penalty,
            return_full_text: Some(return_full_text),
            seed,
            stop,
            stream: Some(stream),
            temperature,
            top_k,
            top_p,
            truncate,
            typical_p,
            watermark: request.parameters.get("watermark").and_then(|v| v.as_bool()),
        };

        let tgi_request = TgiGenerateRequest {
            inputs: prompt,
            parameters: Some(parameters),
        };

        let tgi_response = match if stream {
            self.generate_stream(tgi_request).await
        } else {
            self.generate(tgi_request).await
        } {
            Ok(response) => response,
            Err(e) => {
                let inference_time = start_time.elapsed().as_millis() as f64;
                // Record failed request metrics
                {
                    let collector = self.metric_collector.read().await;
                    collector.record_request_failure(inference_time);
                }
                return Err(e);
            }
        };
        let inference_time = start_time.elapsed().as_millis() as f64;

        // Convert to our response format
        let mut outputs = HashMap::new();
        outputs.insert("generated_text".to_string(), TensorData {
            shape: vec![1],
            dtype: "string".to_string(),
            data: tgi_response.generated_text.into_bytes(),
        });

        // Add detailed generation information if available
        if let Some(details) = &tgi_response.details {
            outputs.insert("generated_tokens".to_string(), TensorData {
                shape: vec![1],
                dtype: "uint32".to_string(),
                data: details.generated_tokens.to_le_bytes().to_vec(),
            });

            outputs.insert("finish_reason".to_string(), TensorData {
                shape: vec![1],
                dtype: "string".to_string(),
                data: details.finish_reason.clone().into_bytes(),
            });

            // Add seed if available
            if let Some(seed) = details.seed {
                outputs.insert("seed".to_string(), TensorData {
                    shape: vec![1],
                    dtype: "uint64".to_string(),
                    data: seed.to_le_bytes().to_vec(),
                });
            }

            // Add token information if available
            if !details.tokens.is_empty() {
                let token_count = details.tokens.len() as u32;
                outputs.insert("token_count".to_string(), TensorData {
                    shape: vec![1],
                    dtype: "uint32".to_string(),
                    data: token_count.to_le_bytes().to_vec(),
                });

                // Serialize token details as JSON for advanced use cases
                if let Ok(tokens_json) = serde_json::to_string(&details.tokens) {
                    outputs.insert("tokens_detail".to_string(), TensorData {
                        shape: vec![1],
                        dtype: "string".to_string(),
                        data: tokens_json.into_bytes(),
                    });
                }
            }

            // Add prefill information if available
            if !details.prefill.is_empty() {
                let prefill_count = details.prefill.len() as u32;
                outputs.insert("prefill_count".to_string(), TensorData {
                    shape: vec![1],
                    dtype: "uint32".to_string(),
                    data: prefill_count.to_le_bytes().to_vec(),
                });
            }
        }

        let response_metadata = ResponseMetadata {
            inference_time_ms: inference_time,
            queue_time_ms: None,
            preprocessing_time_ms: None,
            postprocessing_time_ms: None,
        };

        // Record metrics
        {
            let collector = self.metric_collector.read().await;
            collector.record_request_success(inference_time);
        }

        Ok(InferenceResponse {
            outputs,
            model_name: request.model_name,
            model_version: None,
            request_id: request.request_id,
            metadata: response_metadata,
        })
    }

    async fn get_metrics(&self) -> Result<RuntimeMetrics> {
        let mut collector = self.metric_collector.write().await;
        let mut metrics = collector.collect_metrics().await;

        // Try to get TGI-specific metrics
        if let Ok(tgi_metrics) = self.get_tgi_metrics().await {
            metrics.runtime_specific.insert("tgi_prometheus_metrics".to_string(), serde_json::Value::String(tgi_metrics));
        }

        Ok(metrics)
    }

    fn get_config(&self) -> &RuntimeConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuntimeConfig;

    #[test]
    fn test_tgi_adapter_creation() {
        let config = RuntimeConfig::new(RuntimeType::Tgi);
        assert_eq!(config.runtime_type, RuntimeType::Tgi);
        assert_eq!(config.endpoint.as_str(), "http://localhost:3000/");
    }

    #[test]
    fn test_tgi_generate_request() {
        let parameters = TgiParameters {
            best_of: None,
            decoder_input_details: Some(false),
            details: Some(true),
            do_sample: Some(true),
            max_new_tokens: Some(100),
            repetition_penalty: None,
            return_full_text: Some(false),
            seed: None,
            stream: Some(false),
            stop: Some(vec!["</s>".to_string()]),
            temperature: Some(0.7),
            top_k: Some(50),
            top_p: Some(0.9),
            truncate: None,
            typical_p: None,
            watermark: None,
        };

        let request = TgiGenerateRequest {
            inputs: "Hello, world!".to_string(),
            parameters: Some(parameters),
        };

        assert_eq!(request.inputs, "Hello, world!");
        assert!(request.parameters.is_some());
        
        let params = request.parameters.unwrap();
        assert_eq!(params.max_new_tokens, Some(100));
        assert_eq!(params.temperature, Some(0.7));
        assert_eq!(params.do_sample, Some(true));
    }

    #[test]
    fn test_tgi_health_response() {
        let health = TgiHealthResponse {
            status: "healthy".to_string(),
        };

        assert_eq!(health.status, "healthy");
    }
}
