//! Triton Inference Server adapter

use crate::adapter::{RuntimeAdapterTrait, ModelInfo, ModelStatus, ModelMetadata, TensorShape, InferenceRequest, InferenceResponse, TensorData, ResponseMetadata};
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

/// Triton Inference Server adapter
pub struct TritonAdapter {
    config: RuntimeConfig,
    client: Client,
    metric_collector: RwLock<MetricCollector>,
}

/// Triton server metadata response
#[derive(Debug, Deserialize)]
struct TritonServerMetadata {
    name: String,
    version: String,
    #[allow(unused)]
    extensions: Vec<String>,
}

/// Triton model metadata response
#[derive(Debug, Deserialize)]
struct TritonModelMetadata {
    name: String,
    #[allow(unused)]
    versions: Option<Vec<String>>,
    #[allow(unused)]
    platform: String,
    inputs: Vec<TritonTensorMetadata>,
    outputs: Vec<TritonTensorMetadata>,
}

/// Triton tensor metadata
#[derive(Debug, Deserialize)]
struct TritonTensorMetadata {
    name: String,
    datatype: String,
    shape: Vec<i64>,
}

/// Triton model status response
#[derive(Debug, Deserialize)]
struct TritonModelStatus {
    #[allow(unused)]
    name: String,
    version: Option<String>,
    state: String,
    reason: Option<String>,
}

/// Triton inference request
#[derive(Debug, Serialize)]
struct TritonInferenceRequest {
    id: Option<String>,
    inputs: Vec<TritonInput>,
    outputs: Option<Vec<TritonOutput>>,
    parameters: Option<HashMap<String, serde_json::Value>>,
}

/// Triton input tensor
#[derive(Debug, Serialize)]
struct TritonInput {
    name: String,
    shape: Vec<u64>,
    datatype: String,
    data: Vec<serde_json::Value>,
}

/// Triton output specification
#[derive(Debug, Serialize)]
struct TritonOutput {
    name: String,
    parameters: Option<HashMap<String, serde_json::Value>>,
}

/// Triton inference response
#[derive(Debug, Deserialize)]
struct TritonInferenceResponse {
    id: Option<String>,
    model_name: String,
    model_version: Option<String>,
    outputs: Vec<TritonOutputTensor>,
    #[allow(unused)]
    parameters: Option<HashMap<String, serde_json::Value>>,
}

/// Triton output tensor
#[derive(Debug, Deserialize)]
struct TritonOutputTensor {
    name: String,
    shape: Vec<u64>,
    datatype: String,
    data: Vec<serde_json::Value>,
}

impl TritonAdapter {
    /// Create a new Triton adapter
    pub async fn new(config: RuntimeConfig) -> Result<Self> {
        info!("Creating Triton adapter for endpoint: {}", config.endpoint);

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

    /// Get server metadata
    async fn get_server_metadata(&self) -> Result<TritonServerMetadata> {
        let url = format!("{}/v2", self.config.endpoint);
        
        debug!("Fetching Triton server metadata from: {}", url);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| RuntimeError::Connection(format!("Failed to connect to Triton server: {}", e)))?;

        if !response.status().is_success() {
            return Err(RuntimeError::InvalidResponse(
                format!("Server metadata request failed: {}", response.status())
            ));
        }

        let metadata: TritonServerMetadata = response
            .json()
            .await
            .map_err(|e| RuntimeError::InvalidResponse(format!("Invalid metadata response: {}", e)))?;

        debug!("Triton server metadata: {:?}", metadata);
        Ok(metadata)
    }

    /// Get model metadata
    async fn get_model_metadata(&self, model_name: &str) -> Result<TritonModelMetadata> {
        let url = format!("{}/v2/models/{}", self.config.endpoint, model_name);
        
        debug!("Fetching model metadata for: {}", model_name);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| RuntimeError::Connection(format!("Failed to get model metadata: {}", e)))?;

        if !response.status().is_success() {
            return Err(RuntimeError::Model(
                format!("Model metadata request failed: {}", response.status())
            ));
        }

        let metadata: TritonModelMetadata = response
            .json()
            .await
            .map_err(|e| RuntimeError::InvalidResponse(format!("Invalid model metadata: {}", e)))?;

        Ok(metadata)
    }

    /// Get model status
    async fn get_model_status(&self, model_name: &str) -> Result<TritonModelStatus> {
        let url = format!("{}/v2/models/{}/ready", self.config.endpoint, model_name);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| RuntimeError::Connection(format!("Failed to get model status: {}", e)))?;

        // For ready endpoint, 200 means ready, 400 means not ready
        let status = match response.status().as_u16() {
            200 => TritonModelStatus {
                name: model_name.to_string(),
                version: None,
                state: "READY".to_string(),
                reason: None,
            },
            400 => TritonModelStatus {
                name: model_name.to_string(),
                version: None,
                state: "UNAVAILABLE".to_string(),
                reason: Some("Model not ready".to_string()),
            },
            _ => return Err(RuntimeError::Model(
                format!("Unexpected status response: {}", response.status())
            )),
        };

        Ok(status)
    }

    /// Convert Triton model status to our ModelStatus
    fn convert_model_status(triton_status: &TritonModelStatus) -> ModelStatus {
        match triton_status.state.as_str() {
            "READY" => ModelStatus::Ready,
            "LOADING" => ModelStatus::Loading,
            "UNLOADING" => ModelStatus::Unloading,
            _ => ModelStatus::Failed(
                triton_status.reason.clone().unwrap_or_else(|| "Unknown error".to_string())
            ),
        }
    }

    /// Convert Triton tensor metadata to our TensorShape
    fn convert_tensor_metadata(triton_tensor: &TritonTensorMetadata) -> TensorShape {
        TensorShape {
            name: triton_tensor.name.clone(),
            shape: triton_tensor.shape.clone(),
            dtype: triton_tensor.datatype.clone(),
        }
    }

    /// Convert binary tensor data to JSON values based on datatype
    fn convert_tensor_data_to_json(&self, data: &[u8], dtype: &str) -> Result<Vec<serde_json::Value>> {
        match dtype {
            "BOOL" => {
                // Each bool is 1 byte
                Ok(data.iter().map(|&b| serde_json::Value::Bool(b != 0)).collect())
            }
            "UINT8" => {
                Ok(data.iter().map(|&b| serde_json::Value::Number(b.into())).collect())
            }
            "INT8" => {
                Ok(data.iter().map(|&b| serde_json::Value::Number((b as i8).into())).collect())
            }
            "UINT16" => {
                let values: Vec<u16> = data.chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                Ok(values.into_iter().map(|v| serde_json::Value::Number(v.into())).collect())
            }
            "INT16" => {
                let values: Vec<i16> = data.chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                Ok(values.into_iter().map(|v| serde_json::Value::Number(v.into())).collect())
            }
            "UINT32" => {
                let values: Vec<u32> = data.chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(values.into_iter().map(|v| serde_json::Value::Number(v.into())).collect())
            }
            "INT32" => {
                let values: Vec<i32> = data.chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(values.into_iter().map(|v| serde_json::Value::Number(v.into())).collect())
            }
            "FP32" => {
                let values: Vec<f32> = data.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(values.into_iter()
                    .map(|v| serde_json::Value::Number(
                        serde_json::Number::from_f64(v as f64).unwrap_or_else(|| serde_json::Number::from(0))
                    ))
                    .collect())
            }
            "FP64" => {
                let values: Vec<f64> = data.chunks_exact(8)
                    .map(|chunk| {
                        let mut bytes = [0u8; 8];
                        bytes.copy_from_slice(chunk);
                        f64::from_le_bytes(bytes)
                    })
                    .collect();
                Ok(values.into_iter()
                    .map(|v| serde_json::Value::Number(
                        serde_json::Number::from_f64(v).unwrap_or_else(|| serde_json::Number::from(0))
                    ))
                    .collect())
            }
            "BYTES" => {
                // For string/bytes data, assume UTF-8 encoding
                match String::from_utf8(data.to_vec()) {
                    Ok(s) => Ok(vec![serde_json::Value::String(s)]),
                    Err(_) => {
                        // If not valid UTF-8, encode as base64
                        use base64::{Engine as _, engine::general_purpose};
                        let encoded = general_purpose::STANDARD.encode(data);
                        Ok(vec![serde_json::Value::String(encoded)])
                    }
                }
            }
            _ => {
                warn!("Unknown datatype: {}, treating as raw bytes", dtype);
                use base64::{Engine as _, engine::general_purpose};
                let encoded = general_purpose::STANDARD.encode(data);
                Ok(vec![serde_json::Value::String(encoded)])
            }
        }
    }

    /// Convert JSON values back to binary tensor data
    fn convert_json_to_tensor_data(&self, json_data: &[serde_json::Value], dtype: &str) -> Result<Vec<u8>> {
        let mut data = Vec::new();
        
        match dtype {
            "BOOL" => {
                for value in json_data {
                    if let Some(b) = value.as_bool() {
                        data.push(if b { 1u8 } else { 0u8 });
                    } else {
                        return Err(RuntimeError::InvalidResponse("Invalid bool value".to_string()));
                    }
                }
            }
            "UINT8" => {
                for value in json_data {
                    if let Some(n) = value.as_u64() {
                        if n <= u8::MAX as u64 {
                            data.push(n as u8);
                        } else {
                            return Err(RuntimeError::InvalidResponse("Value out of range for UINT8".to_string()));
                        }
                    } else {
                        return Err(RuntimeError::InvalidResponse("Invalid UINT8 value".to_string()));
                    }
                }
            }
            "FP32" => {
                for value in json_data {
                    if let Some(f) = value.as_f64() {
                        data.extend_from_slice(&(f as f32).to_le_bytes());
                    } else {
                        return Err(RuntimeError::InvalidResponse("Invalid FP32 value".to_string()));
                    }
                }
            }
            "BYTES" => {
                for value in json_data {
                    if let Some(s) = value.as_str() {
                        // Try to decode as base64 first, then fall back to UTF-8
                        use base64::{Engine as _, engine::general_purpose};
                        match general_purpose::STANDARD.decode(s) {
                            Ok(bytes) => data.extend_from_slice(&bytes),
                            Err(_) => data.extend_from_slice(s.as_bytes()),
                        }
                    } else {
                        return Err(RuntimeError::InvalidResponse("Invalid BYTES value".to_string()));
                    }
                }
            }
            _ => {
                return Err(RuntimeError::InvalidResponse(format!("Unsupported datatype: {}", dtype)));
            }
        }
        
        Ok(data)
    }
}

#[async_trait]
impl RuntimeAdapterTrait for TritonAdapter {
    fn runtime_type(&self) -> RuntimeType {
        RuntimeType::Triton
    }

    async fn initialize(&mut self) -> Result<()> {
        info!("Initializing Triton adapter");
        
        // Test connection by getting server metadata
        let _metadata = self.get_server_metadata().await?;
        
        info!("Triton adapter initialized successfully");
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down Triton adapter");
        Ok(())
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        let start_time = Instant::now();
        
        match self.get_server_metadata().await {
            Ok(metadata) => {
                let response_time = start_time.elapsed().as_millis() as f64;
                
                let health_info = HealthInfo::healthy()
                    .with_response_time(response_time)
                    .add_check(CheckResult::pass("server_connectivity".to_string()))
                    .add_check(CheckResult::pass("server_metadata".to_string()))
                    .add_runtime_data("server_name".to_string(), serde_json::Value::String(metadata.name))
                    .add_runtime_data("server_version".to_string(), serde_json::Value::String(metadata.version));
                
                Ok(health_info.status)
            }
            Err(e) => {
                warn!("Triton health check failed: {}", e);
                Ok(HealthStatus::Unhealthy(e.to_string()))
            }
        }
    }

    async fn load_model(&self, name: &str, config: Option<ModelConfig>) -> Result<()> {
        info!("Loading model: {}", name);
        
        // Triton loads models from the model repository automatically
        // We can trigger explicit loading via the model control API
        let url = format!("{}/v2/repository/models/{}/load", self.config.endpoint, name);
        
        let mut request = self.client.post(&url);
        
        // Add model configuration if provided
        if let Some(model_config) = config {
            let load_request = serde_json::json!({
                "parameters": model_config.config
            });
            request = request.json(&load_request);
        }
        
        let response = request
            .send()
            .await
            .map_err(|e| RuntimeError::Connection(format!("Failed to load model: {}", e)))?;

        if !response.status().is_success() {
            return Err(RuntimeError::Model(
                format!("Model load failed: {}", response.status())
            ));
        }

        info!("Model {} loaded successfully", name);
        Ok(())
    }

    async fn unload_model(&self, name: &str) -> Result<()> {
        info!("Unloading model: {}", name);
        
        let url = format!("{}/v2/repository/models/{}/unload", self.config.endpoint, name);
        
        let response = self.client
            .post(&url)
            .send()
            .await
            .map_err(|e| RuntimeError::Connection(format!("Failed to unload model: {}", e)))?;

        if !response.status().is_success() {
            return Err(RuntimeError::Model(
                format!("Model unload failed: {}", response.status())
            ));
        }

        info!("Model {} unloaded successfully", name);
        Ok(())
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/v2/models", self.config.endpoint);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| RuntimeError::Connection(format!("Failed to list models: {}", e)))?;

        if !response.status().is_success() {
            return Err(RuntimeError::InvalidResponse(
                format!("Model list request failed: {}", response.status())
            ));
        }

        // Parse the response to extract model names
        let models_response: serde_json::Value = response
            .json()
            .await
            .map_err(|e| RuntimeError::InvalidResponse(format!("Invalid models response: {}", e)))?;

        let model_names = models_response
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|model| {
                model.get("name")?.as_str().map(|s| s.to_string())
            })
            .collect();

        Ok(model_names)
    }

    async fn get_model_info(&self, name: &str) -> Result<ModelInfo> {
        let metadata = self.get_model_metadata(name).await?;
        let status = self.get_model_status(name).await?;

        let input_shapes = metadata.inputs.iter()
            .map(|t| Self::convert_tensor_metadata(t))
            .collect();

        let output_shapes = metadata.outputs.iter()
            .map(|t| Self::convert_tensor_metadata(t))
            .collect();

        let model_metadata = ModelMetadata {
            input_shapes,
            output_shapes,
            batch_size: None, // Would need to parse from model config
            max_sequence_length: None, // Would need to parse from model config
            memory_usage: None, // Would need to get from model statistics
        };

        Ok(ModelInfo {
            name: metadata.name,
            version: status.version.clone(),
            status: Self::convert_model_status(&status),
            config: HashMap::new(), // Would populate from model config
            metadata: model_metadata,
        })
    }

    async fn inference(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let start_time = Instant::now();
        
        debug!("Processing inference request for model: {}", request.model_name);

        // Convert our request format to Triton format
        let triton_inputs: Vec<TritonInput> = request.inputs.iter()
            .map(|(name, tensor_data)| {
                // Convert binary data to appropriate JSON values based on datatype
                let data = self.convert_tensor_data_to_json(&tensor_data.data, &tensor_data.dtype)?;
                Ok(TritonInput {
                    name: name.clone(),
                    shape: tensor_data.shape.clone(),
                    datatype: tensor_data.dtype.clone(),
                    data,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let triton_request = TritonInferenceRequest {
            id: request.request_id.clone(),
            inputs: triton_inputs,
            outputs: None, // Let Triton return all outputs
            parameters: Some(request.parameters),
        };

        // Build URL
        let mut url = format!("{}/v2/models/{}/infer", self.config.endpoint, request.model_name);
        if let Some(version) = &request.model_version {
            url = format!("{}/v2/models/{}/versions/{}/infer", self.config.endpoint, request.model_name, version);
        }

        // Send request
        let response = match self.client
            .post(&url)
            .json(&triton_request)
            .send()
            .await
        {
            Ok(response) => response,
            Err(e) => {
                let inference_time = start_time.elapsed().as_millis() as f64;
                // Record failed request metrics
                {
                    let collector = self.metric_collector.read().await;
                    collector.record_request_failure(inference_time);
                }
                return Err(RuntimeError::Connection(format!("Inference request failed: {}", e)));
            }
        };

        let inference_time = start_time.elapsed().as_millis() as f64;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            // Record failed request metrics
            {
                let collector = self.metric_collector.read().await;
                collector.record_request_failure(inference_time);
            }
            return Err(RuntimeError::Model(
                format!("Inference failed: {} - {}", status, error_text)
            ));
        }

        let triton_response: TritonInferenceResponse = response
            .json()
            .await
            .map_err(|e| RuntimeError::InvalidResponse(format!("Invalid inference response: {}", e)))?;

        // Convert Triton response to our format
        let outputs: HashMap<String, TensorData> = triton_response.outputs.iter()
            .map(|output| {
                let data = self.convert_json_to_tensor_data(&output.data, &output.datatype)?;
                let tensor_data = TensorData {
                    shape: output.shape.clone(),
                    dtype: output.datatype.clone(),
                    data,
                };
                Ok((output.name.clone(), tensor_data))
            })
            .collect::<Result<HashMap<_, _>>>()?;

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
            model_name: triton_response.model_name,
            model_version: triton_response.model_version,
            request_id: triton_response.id,
            metadata: response_metadata,
        })
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
    fn test_triton_adapter_creation() {
        let config = RuntimeConfig::new(RuntimeType::Triton);
        // Note: We can't actually test the async new() method without a real server
        // In a real test environment, we'd use a mock server
        assert_eq!(config.runtime_type, RuntimeType::Triton);
    }

    #[test]
    fn test_convert_model_status() {
        let ready_status = TritonModelStatus {
            name: "test".to_string(),
            version: None,
            state: "READY".to_string(),
            reason: None,
        };
        assert_eq!(TritonAdapter::convert_model_status(&ready_status), ModelStatus::Ready);

        let loading_status = TritonModelStatus {
            name: "test".to_string(),
            version: None,
            state: "LOADING".to_string(),
            reason: None,
        };
        assert_eq!(TritonAdapter::convert_model_status(&loading_status), ModelStatus::Loading);

        let failed_status = TritonModelStatus {
            name: "test".to_string(),
            version: None,
            state: "FAILED".to_string(),
            reason: Some("Error loading".to_string()),
        };
        assert!(matches!(TritonAdapter::convert_model_status(&failed_status), ModelStatus::Failed(_)));
    }

    #[test]
    fn test_convert_tensor_metadata() {
        let triton_tensor = TritonTensorMetadata {
            name: "input".to_string(),
            datatype: "FP32".to_string(),
            shape: vec![1, 3, 224, 224],
        };

        let tensor_shape = TritonAdapter::convert_tensor_metadata(&triton_tensor);
        assert_eq!(tensor_shape.name, "input");
        assert_eq!(tensor_shape.dtype, "FP32");
        assert_eq!(tensor_shape.shape, vec![1, 3, 224, 224]);
    }
}
