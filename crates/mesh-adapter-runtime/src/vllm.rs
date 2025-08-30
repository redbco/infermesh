//! vLLM adapter for high-throughput LLM serving

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

/// vLLM adapter
pub struct VLlmAdapter {
    config: RuntimeConfig,
    client: Client,
    metric_collector: RwLock<MetricCollector>,
}

/// vLLM health response
#[derive(Debug, Deserialize)]
struct VLlmHealthResponse {
    status: String,
}

/// vLLM model info response
#[derive(Debug, Deserialize)]
struct VLlmModelInfo {
    id: String,
    #[allow(unused)]
    object: String,
    #[allow(unused)]
    created: u64,
    #[allow(unused)]
    owned_by: String,
}

/// vLLM models list response
#[derive(Debug, Deserialize)]
struct VLlmModelsResponse {
    #[allow(unused)]
    object: String,
    data: Vec<VLlmModelInfo>,
}

/// vLLM completion request (OpenAI-compatible)
#[derive(Debug, Serialize)]
struct VLlmCompletionRequest {
    model: String,
    prompt: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    n: Option<u32>,
    stream: Option<bool>,
    logprobs: Option<u32>,
    echo: Option<bool>,
    stop: Option<Vec<String>>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    best_of: Option<u32>,
    logit_bias: Option<HashMap<String, f32>>,
    user: Option<String>,
}

/// vLLM completion response
#[derive(Debug, Deserialize)]
struct VLlmCompletionResponse {
    #[allow(unused)]
    id: String,
    #[allow(unused)]
    object: String,
    #[allow(unused)]
    created: u64,
    #[allow(unused)]
    model: String,
    choices: Vec<VLlmChoice>,
    #[allow(unused)]
    usage: VLlmUsage,
}

/// vLLM choice in completion response
#[derive(Debug, Deserialize)]
struct VLlmChoice {
    text: String,
    #[allow(unused)]
    index: u32,
    #[allow(unused)]
    logprobs: Option<serde_json::Value>,
    #[allow(unused)]
    finish_reason: Option<String>,
}

/// vLLM usage statistics
#[derive(Debug, Deserialize)]
struct VLlmUsage {
    #[allow(unused)]
    prompt_tokens: u32,
    #[allow(unused)]
    completion_tokens: u32,
    #[allow(unused)]
    total_tokens: u32,
}

/// vLLM chat completion request
#[derive(Debug, Serialize)]
struct VLlmChatRequest {
    model: String,
    messages: Vec<VLlmMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    n: Option<u32>,
    stream: Option<bool>,
    stop: Option<Vec<String>>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    logit_bias: Option<HashMap<String, f32>>,
    user: Option<String>,
}

/// vLLM chat message
#[derive(Debug, Serialize, Deserialize)]
struct VLlmMessage {
    role: String,
    content: String,
}

/// vLLM chat completion response
#[derive(Debug, Deserialize)]
struct VLlmChatResponse {
    #[allow(unused)]
    id: String,
    #[allow(unused)]
    object: String,
    #[allow(unused)]
    created: u64,
    model: String,
    choices: Vec<VLlmChatChoice>,
    #[allow(unused)]
    usage: VLlmUsage,
}

/// vLLM chat choice
#[derive(Debug, Deserialize)]
struct VLlmChatChoice {
    #[allow(unused)]
    index: u32,
    message: VLlmMessage,
    #[allow(unused)]
    finish_reason: Option<String>,
}

impl VLlmAdapter {
    /// Create a new vLLM adapter
    pub async fn new(config: RuntimeConfig) -> Result<Self> {
        info!("Creating vLLM adapter for endpoint: {}", config.endpoint);

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

    /// Check vLLM health
    async fn check_health(&self) -> Result<VLlmHealthResponse> {
        let url = format!("{}/health", self.config.endpoint);
        
        debug!("Checking vLLM health at: {}", url);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| RuntimeError::Connection(format!("Failed to connect to vLLM server: {}", e)))?;

        if !response.status().is_success() {
            return Err(RuntimeError::HealthCheck(
                format!("Health check failed: {}", response.status())
            ));
        }

        let health: VLlmHealthResponse = response
            .json()
            .await
            .map_err(|e| RuntimeError::InvalidResponse(format!("Invalid health response: {}", e)))?;

        Ok(health)
    }

    /// Get available models
    async fn get_models(&self) -> Result<VLlmModelsResponse> {
        let url = format!("{}/v1/models", self.config.endpoint);
        
        debug!("Fetching vLLM models from: {}", url);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| RuntimeError::Connection(format!("Failed to get models: {}", e)))?;

        if !response.status().is_success() {
            return Err(RuntimeError::InvalidResponse(
                format!("Models request failed: {}", response.status())
            ));
        }

        let models: VLlmModelsResponse = response
            .json()
            .await
            .map_err(|e| RuntimeError::InvalidResponse(format!("Invalid models response: {}", e)))?;

        Ok(models)
    }

    /// Check if streaming is supported and requested
    #[allow(unused)]
    fn should_stream(&self, request_params: &serde_json::Map<String, serde_json::Value>) -> bool {
        request_params.get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    /// Try to get model max sequence length from vLLM
    async fn get_model_max_length(&self, _model_name: &str) -> Option<u32> {
        // vLLM doesn't expose model configuration through API by default
        // This would require custom endpoint or configuration
        // For now, return a reasonable default for most models
        Some(4096)
    }

    /// Perform text completion
    async fn complete(&self, request: VLlmCompletionRequest) -> Result<VLlmCompletionResponse> {
        let url = format!("{}/v1/completions", self.config.endpoint);
        
        debug!("Sending completion request to vLLM");
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| RuntimeError::Connection(format!("Completion request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(RuntimeError::Model(
                format!("Completion failed: {} - {}", status, error_text)
            ));
        }

        let completion: VLlmCompletionResponse = response
            .json()
            .await
            .map_err(|e| RuntimeError::InvalidResponse(format!("Invalid completion response: {}", e)))?;

        Ok(completion)
    }

    /// Perform chat completion
    async fn chat(&self, request: VLlmChatRequest) -> Result<VLlmChatResponse> {
        let url = format!("{}/v1/chat/completions", self.config.endpoint);
        
        debug!("Sending chat request to vLLM");
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| RuntimeError::Connection(format!("Chat request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(RuntimeError::Model(
                format!("Chat completion failed: {} - {}", status, error_text)
            ));
        }

        let chat_response: VLlmChatResponse = response
            .json()
            .await
            .map_err(|e| RuntimeError::InvalidResponse(format!("Invalid chat response: {}", e)))?;

        Ok(chat_response)
    }
}

#[async_trait]
impl RuntimeAdapterTrait for VLlmAdapter {
    fn runtime_type(&self) -> RuntimeType {
        RuntimeType::VLlm
    }

    async fn initialize(&mut self) -> Result<()> {
        info!("Initializing vLLM adapter");
        
        // Test connection by checking health
        let _health = self.check_health().await?;
        
        info!("vLLM adapter initialized successfully");
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down vLLM adapter");
        Ok(())
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        let start_time = Instant::now();
        
        match self.check_health().await {
            Ok(health) => {
                let response_time = start_time.elapsed().as_millis() as f64;
                
                let health_info = HealthInfo::healthy()
                    .with_response_time(response_time)
                    .add_check(CheckResult::pass("server_connectivity".to_string()))
                    .add_check(CheckResult::pass("health_endpoint".to_string()))
                    .add_runtime_data("status".to_string(), serde_json::Value::String(health.status));
                
                Ok(health_info.status)
            }
            Err(e) => {
                warn!("vLLM health check failed: {}", e);
                Ok(HealthStatus::Unhealthy(e.to_string()))
            }
        }
    }

    async fn load_model(&self, name: &str, _config: Option<ModelConfig>) -> Result<()> {
        // vLLM typically loads models at startup, not dynamically
        // This would require server restart or hot-swapping if supported
        warn!("vLLM does not support dynamic model loading. Model {} should be loaded at startup.", name);
        Ok(())
    }

    async fn unload_model(&self, name: &str) -> Result<()> {
        // vLLM typically doesn't support dynamic model unloading
        warn!("vLLM does not support dynamic model unloading. Model {} remains loaded.", name);
        Ok(())
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let models_response = self.get_models().await?;
        let model_names = models_response.data.iter()
            .map(|model| model.id.clone())
            .collect();
        Ok(model_names)
    }

    async fn get_model_info(&self, name: &str) -> Result<ModelInfo> {
        let models_response = self.get_models().await?;
        
        let model_info = models_response.data.iter()
            .find(|model| model.id == name)
            .ok_or_else(|| RuntimeError::Model(format!("Model {} not found", name)))?;

        // vLLM models are typically always ready if they're listed
        // Try to get additional model information from vLLM if available
        let metadata = ModelMetadata {
            input_shapes: vec![], // Text models don't have fixed tensor shapes
            output_shapes: vec![],
            batch_size: None,
            max_sequence_length: self.get_model_max_length(name).await,
            memory_usage: None, // vLLM doesn't expose memory usage per model
        };

        Ok(ModelInfo {
            name: model_info.id.clone(),
            version: None, // vLLM doesn't typically version models
            status: ModelStatus::Ready,
            config: HashMap::new(),
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
                // Decode tensor data as UTF-8 string
                String::from_utf8(tensor.data.clone()).ok()
            })
            .ok_or_else(|| RuntimeError::Model("No valid text prompt found in inputs".to_string()))?;

        // Extract parameters with defaults
        let max_tokens = request.parameters.get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .or_else(|| Some(512)); // Default max tokens
        
        let temperature = request.parameters.get("temperature")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);
            
        let top_p = request.parameters.get("top_p")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);
            
        let stop_sequences = request.parameters.get("stop")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter()
                .filter_map(|s| s.as_str().map(|s| s.to_string()))
                .collect::<Vec<String>>());

        // Check if this is a chat request (has messages parameter)
        if request.parameters.contains_key("messages") {
            // Handle as chat completion - extract messages from parameters
            let messages = if let Some(msgs) = request.parameters.get("messages").and_then(|v| v.as_array()) {
                msgs.iter()
                    .filter_map(|msg| {
                        if let (Some(role), Some(content)) = (
                            msg.get("role").and_then(|r| r.as_str()),
                            msg.get("content").and_then(|c| c.as_str())
                        ) {
                            Some(VLlmMessage {
                                role: role.to_string(),
                                content: content.to_string(),
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                // Fallback to using the prompt as a user message
                vec![VLlmMessage {
                    role: "user".to_string(),
                    content: prompt,
                }]
            };

            let chat_request = VLlmChatRequest {
                model: request.model_name.clone(),
                messages,
                max_tokens,
                temperature,
                top_p,
                n: None,
                stream: Some(false),
                stop: stop_sequences.clone(),
                presence_penalty: request.parameters.get("presence_penalty")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32),
                frequency_penalty: request.parameters.get("frequency_penalty")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32),
                logit_bias: None, // TODO: Support logit bias if needed
                user: request.parameters.get("user")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
            };

            let chat_response = match self.chat(chat_request).await {
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
            let output_text = chat_response.choices.first()
                .map(|choice| choice.message.content.clone())
                .unwrap_or_default();

            let mut outputs = HashMap::new();
            outputs.insert("text".to_string(), TensorData {
                shape: vec![1],
                dtype: "string".to_string(),
                data: output_text.into_bytes(),
            });

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
                model_name: chat_response.model,
                model_version: None,
                request_id: request.request_id,
                metadata: response_metadata,
            })
        } else {
            // Handle as text completion
            let completion_request = VLlmCompletionRequest {
                model: request.model_name.clone(),
                prompt,
                max_tokens,
                temperature,
                top_p,
                n: None,
                stream: Some(false),
                logprobs: request.parameters.get("logprobs")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32),
                echo: request.parameters.get("echo")
                    .and_then(|v| v.as_bool()),
                stop: stop_sequences,
                presence_penalty: request.parameters.get("presence_penalty")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32),
                frequency_penalty: request.parameters.get("frequency_penalty")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32),
                best_of: request.parameters.get("best_of")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32),
                logit_bias: None, // TODO: Support logit bias if needed
                user: request.parameters.get("user")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
            };

            let completion_response = match self.complete(completion_request).await {
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
            let output_text = completion_response.choices.first()
                .map(|choice| choice.text.clone())
                .unwrap_or_default();

            let mut outputs = HashMap::new();
            outputs.insert("text".to_string(), TensorData {
                shape: vec![1],
                dtype: "string".to_string(),
                data: output_text.into_bytes(),
            });

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
                model_name: completion_response.model,
                model_version: None,
                request_id: request.request_id,
                metadata: response_metadata,
            })
        }
    }

    async fn get_metrics(&self) -> Result<RuntimeMetrics> {
        let mut collector = self.metric_collector.write().await;
        Ok(collector.collect_metrics().await)
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
    fn test_vllm_adapter_creation() {
        let config = RuntimeConfig::new(RuntimeType::VLlm);
        assert_eq!(config.runtime_type, RuntimeType::VLlm);
        assert_eq!(config.endpoint.as_str(), "http://localhost:8000/");
    }

    #[test]
    fn test_vllm_completion_request() {
        let request = VLlmCompletionRequest {
            model: "gpt-7b".to_string(),
            prompt: "Hello, world!".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: None,
            n: None,
            stream: Some(false),
            logprobs: None,
            echo: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            best_of: None,
            logit_bias: None,
            user: None,
        };

        assert_eq!(request.model, "gpt-7b");
        assert_eq!(request.prompt, "Hello, world!");
        assert_eq!(request.max_tokens, Some(100));
        assert_eq!(request.temperature, Some(0.7));
    }

    #[test]
    fn test_vllm_chat_request() {
        let messages = vec![VLlmMessage {
            role: "user".to_string(),
            content: "Hello!".to_string(),
        }];

        let request = VLlmChatRequest {
            model: "gpt-7b".to_string(),
            messages,
            max_tokens: Some(50),
            temperature: Some(0.8),
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
        };

        assert_eq!(request.model, "gpt-7b");
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.messages[0].role, "user");
        assert_eq!(request.messages[0].content, "Hello!");
    }
}
