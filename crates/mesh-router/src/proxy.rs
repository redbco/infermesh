//! Request proxying and forwarding

use crate::handler::{RequestContext, RoutingTarget};
use crate::{Result, RouterError};

use bytes::Bytes;
use hyper::{Method, Request, Response, Uri, body::Incoming};
use http_body_util::BodyExt;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::timeout;
// Removed tonic client imports for now - using hyper-based approach
use tracing::{debug, info, warn};

/// HTTP proxy for forwarding requests to upstream services
pub struct HttpProxy {
    client: Client<hyper_util::client::legacy::connect::HttpConnector, http_body_util::Full<Bytes>>,
    timeout: Duration,
}

/// gRPC proxy for forwarding gRPC requests
pub struct GrpcProxy {
    timeout: Duration,
    // HTTP/2 client for gRPC requests (gRPC is HTTP/2 based)
    client: Client<hyper_util::client::legacy::connect::HttpConnector, http_body_util::Full<Bytes>>,
}

/// Proxy response with metadata
#[derive(Debug)]
pub struct ProxyResponse {
    /// HTTP status code
    pub status: u16,
    
    /// Response headers
    pub headers: HashMap<String, String>,
    
    /// Response body
    pub body: Bytes,
    
    /// Response time in milliseconds
    pub response_time_ms: u64,
    
    /// Target that handled the request
    pub target: RoutingTarget,
}

impl HttpProxy {
    /// Create a new HTTP proxy
    pub fn new(request_timeout: Duration) -> Self {
        let client = Client::builder(TokioExecutor::new()).build_http();
        
        Self {
            client,
            timeout: request_timeout,
        }
    }

    /// Forward an HTTP request to the target
    pub async fn forward_request(
        &self,
        context: &RequestContext,
        target: &RoutingTarget,
        method: Method,
        path: &str,
        headers: HashMap<String, String>,
        body: Option<Bytes>,
    ) -> Result<ProxyResponse> {
        let start_time = std::time::Instant::now();
        
        debug!(
            "Forwarding request {} to target {:?} at {}",
            context.request_id, target.node_id, target.address
        );

        // Build target URI
        let uri = format!("http://{}{}", target.address, path)
            .parse::<Uri>()
            .map_err(|e| RouterError::Proxy(format!("Invalid target URI: {}", e)))?;

        // Build request
        let mut request_builder = Request::builder()
            .method(method)
            .uri(uri);

        // Add headers
        for (key, value) in headers {
            request_builder = request_builder.header(&key, &value);
        }

        // Add request ID header for tracing
        request_builder = request_builder.header("X-Request-ID", &context.request_id);

        // Build request with body
        let request = if let Some(body_bytes) = body {
            request_builder
                .body(http_body_util::Full::new(body_bytes))
                .map_err(|e| RouterError::Proxy(format!("Failed to build request: {}", e)))?
        } else {
            request_builder
                .body(http_body_util::Full::new(Bytes::new()))
                .map_err(|e| RouterError::Proxy(format!("Failed to build request: {}", e)))?
        };

        // Send request with timeout
        let response = timeout(self.timeout, self.client.request(request))
            .await
            .map_err(|_| RouterError::Timeout)?
            .map_err(|e| RouterError::Proxy(format!("Request failed: {}", e)))?;

        let response_time_ms = start_time.elapsed().as_millis() as u64;
        let status = response.status().as_u16();

        // Extract headers
        let mut response_headers = HashMap::new();
        for (key, value) in response.headers() {
            if let Ok(value_str) = value.to_str() {
                response_headers.insert(key.to_string(), value_str.to_string());
            }
        }

        // Read response body
        let body_bytes = response.into_body().collect().await
            .map_err(|e| RouterError::Proxy(format!("Failed to read response body: {}", e)))?
            .to_bytes();

        info!(
            "Request {} forwarded successfully to {:?} in {}ms (status: {})",
            context.request_id, target.node_id, response_time_ms, status
        );

        Ok(ProxyResponse {
            status,
            headers: response_headers,
            body: body_bytes,
            response_time_ms,
            target: target.clone(),
        })
    }

    /// Forward a streaming request (for WebSocket upgrades, etc.)
    pub async fn forward_streaming_request(
        &self,
        context: &RequestContext,
        target: &RoutingTarget,
        _request: Request<Incoming>,
    ) -> Result<Response<Incoming>> {
        debug!(
            "Forwarding streaming request {} to target {:?}",
            context.request_id, target.node_id
        );

        // TODO: Implement proper streaming request forwarding
        // For now, return an error as this is not implemented
        Err(RouterError::Proxy("Streaming requests not yet implemented".to_string()))
    }

    /// Health check a target
    pub async fn health_check_target(&self, target: &RoutingTarget) -> Result<bool> {
        let uri = format!("http://{}/health", target.address)
            .parse::<Uri>()
            .map_err(|e| RouterError::Proxy(format!("Invalid health check URI: {}", e)))?;

        let request = Request::builder()
            .method(Method::GET)
            .uri(uri)
            .body(http_body_util::Full::new(Bytes::new()))
            .map_err(|e| RouterError::Proxy(format!("Failed to build health check request: {}", e)))?;

        // Use a shorter timeout for health checks
        let health_timeout = Duration::from_secs(5);
        
        match timeout(health_timeout, self.client.request(request)).await {
            Ok(Ok(response)) => {
                let is_healthy = response.status().is_success();
                debug!("Health check for {:?}: {}", target.node_id, is_healthy);
                Ok(is_healthy)
            }
            Ok(Err(e)) => {
                warn!("Health check failed for {:?}: {}", target.node_id, e);
                Ok(false)
            }
            Err(_) => {
                warn!("Health check timeout for {:?}", target.node_id);
                Ok(false)
            }
        }
    }
}

impl GrpcProxy {
    /// Create a new gRPC proxy
    pub fn new(request_timeout: Duration) -> Self {
        // Create HTTP/2 client for gRPC (gRPC uses HTTP/2)
        let client = Client::builder(TokioExecutor::new()).build_http();
        
        Self {
            timeout: request_timeout,
            client,
        }
    }

    /// Forward a gRPC request to the target using HTTP/2
    pub async fn forward_grpc_request(
        &self,
        context: &RequestContext,
        target: &RoutingTarget,
        service_name: &str,
        method_name: &str,
        request_data: Bytes,
    ) -> Result<ProxyResponse> {
        let start_time = std::time::Instant::now();
        
        debug!(
            "Forwarding gRPC request {} to target {:?} ({}::{}) with {} bytes",
            context.request_id, target.node_id, service_name, method_name, request_data.len()
        );

        // Build gRPC target URI (gRPC uses HTTP/2)
        let uri = format!("http://{}/{}/{}", target.address, service_name, method_name)
            .parse::<Uri>()
            .map_err(|e| RouterError::Proxy(format!("Invalid gRPC URI: {}", e)))?;

        // Build gRPC request (gRPC uses POST with specific headers)
        let mut request_builder = Request::builder()
            .method(Method::POST)
            .uri(uri)
            .header("content-type", "application/grpc")
            .header("te", "trailers")
            .header("grpc-accept-encoding", "identity,deflate,gzip");

        // Add request ID header for tracing
        request_builder = request_builder.header("x-request-id", &context.request_id);

        // Build request with gRPC body
        let request = request_builder
            .body(http_body_util::Full::new(request_data))
            .map_err(|e| RouterError::Proxy(format!("Failed to build gRPC request: {}", e)))?;

        // Send gRPC request with timeout
        let response = timeout(self.timeout, self.client.request(request))
            .await
            .map_err(|_| RouterError::Timeout)?
            .map_err(|e| RouterError::Proxy(format!("gRPC request failed: {}", e)))?;

        let response_time_ms = start_time.elapsed().as_millis() as u64;
        let status = response.status().as_u16();

        // Extract headers (including gRPC metadata)
        let mut response_headers = HashMap::new();
        for (key, value) in response.headers() {
            if let Ok(value_str) = value.to_str() {
                response_headers.insert(key.to_string(), value_str.to_string());
            }
        }

        // Read response body
        let body_bytes = response.into_body().collect().await
            .map_err(|e| RouterError::Proxy(format!("Failed to read gRPC response body: {}", e)))?
            .to_bytes();

        info!(
            "gRPC request {} forwarded successfully to {:?} in {}ms (status: {})",
            context.request_id, target.node_id, response_time_ms, status
        );

        Ok(ProxyResponse {
            status,
            headers: response_headers,
            body: body_bytes,
            response_time_ms,
            target: target.clone(),
        })
    }

    /// Health check a gRPC target using the standard gRPC health checking protocol
    pub async fn health_check_grpc_target(&self, target: &RoutingTarget) -> Result<bool> {
        debug!("gRPC health check for {:?}", target.node_id);
        
        // Use a shorter timeout for health checks
        let health_timeout = Duration::from_secs(5);
        
        // Try to call the standard gRPC health checking service
        let uri = format!("http://{}/grpc.health.v1.Health/Check", target.address)
            .parse::<Uri>()
            .map_err(|e| RouterError::Proxy(format!("Invalid health check URI: {}", e)))?;

        // Build gRPC health check request
        let request = Request::builder()
            .method(Method::POST)
            .uri(uri)
            .header("content-type", "application/grpc")
            .header("te", "trailers")
            .body(http_body_util::Full::new(Bytes::new())) // Empty body for health check
            .map_err(|e| RouterError::Proxy(format!("Failed to build health check request: {}", e)))?;

        // Send health check request
        match timeout(health_timeout, self.client.request(request)).await {
            Ok(Ok(response)) => {
                let is_healthy = response.status().is_success();
                debug!("gRPC health check for {:?}: {}", target.node_id, is_healthy);
                Ok(is_healthy)
            }
            Ok(Err(e)) => {
                warn!("gRPC health check failed for {:?}: {}", target.node_id, e);
                Ok(false)
            }
            Err(_) => {
                warn!("gRPC health check timeout for {:?}", target.node_id);
                Ok(false)
            }
        }
    }
}

/// Proxy factory for creating HTTP and gRPC proxies
pub struct ProxyFactory {
    http_timeout: Duration,
    grpc_timeout: Duration,
}

impl ProxyFactory {
    /// Create a new proxy factory
    pub fn new(http_timeout: Duration, grpc_timeout: Duration) -> Self {
        Self {
            http_timeout,
            grpc_timeout,
        }
    }

    /// Create an HTTP proxy
    pub fn create_http_proxy(&self) -> HttpProxy {
        HttpProxy::new(self.http_timeout)
    }

    /// Create a gRPC proxy
    pub fn create_grpc_proxy(&self) -> GrpcProxy {
        GrpcProxy::new(self.grpc_timeout)
    }
}

/// Retry logic for failed proxy requests
pub struct RetryPolicy {
    max_retries: u32,
    base_delay: Duration,
    max_delay: Duration,
}

impl RetryPolicy {
    /// Create a new retry policy
    pub fn new(max_retries: u32, base_delay: Duration, max_delay: Duration) -> Self {
        Self {
            max_retries,
            base_delay,
            max_delay,
        }
    }

    /// Execute a request with retry logic
    pub async fn execute_with_retry<F, Fut, T>(&self, mut operation: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut last_error = None;
        
        for attempt in 0..=self.max_retries {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                    
                    if attempt < self.max_retries {
                        let delay = self.calculate_delay(attempt);
                        debug!("Request failed, retrying in {:?} (attempt {}/{})", 
                               delay, attempt + 1, self.max_retries);
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| RouterError::Proxy("All retries exhausted".to_string())))
    }

    /// Calculate delay for exponential backoff
    fn calculate_delay(&self, attempt: u32) -> Duration {
        let delay = self.base_delay * 2_u32.pow(attempt);
        std::cmp::min(delay, self.max_delay)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_core::{Labels, NodeId};

    #[allow(dead_code)]
    fn create_test_target() -> RoutingTarget {
        RoutingTarget {
            node_id: NodeId::new("test-node"),
            address: "127.0.0.1:8080".parse().unwrap(),
            score: 1.0,
            labels: Labels::new("test", "v1", "runtime", "node1"),
        }
    }

    #[test]
    fn test_http_proxy_creation() {
        let proxy = HttpProxy::new(Duration::from_secs(30));
        assert_eq!(proxy.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_grpc_proxy_creation() {
        let proxy = GrpcProxy::new(Duration::from_secs(30));
        assert_eq!(proxy.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_proxy_factory() {
        let factory = ProxyFactory::new(
            Duration::from_secs(30),
            Duration::from_secs(60),
        );
        
        let http_proxy = factory.create_http_proxy();
        let grpc_proxy = factory.create_grpc_proxy();
        
        assert_eq!(http_proxy.timeout, Duration::from_secs(30));
        assert_eq!(grpc_proxy.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_retry_policy() {
        let policy = RetryPolicy::new(
            3,
            Duration::from_millis(100),
            Duration::from_secs(1),
        );
        
        assert_eq!(policy.max_retries, 3);
        assert_eq!(policy.base_delay, Duration::from_millis(100));
        assert_eq!(policy.max_delay, Duration::from_secs(1));
    }

    #[test]
    fn test_retry_delay_calculation() {
        let policy = RetryPolicy::new(
            3,
            Duration::from_millis(100),
            Duration::from_secs(1),
        );
        
        assert_eq!(policy.calculate_delay(0), Duration::from_millis(100));
        assert_eq!(policy.calculate_delay(1), Duration::from_millis(200));
        assert_eq!(policy.calculate_delay(2), Duration::from_millis(400));
        
        // Should cap at max_delay
        assert_eq!(policy.calculate_delay(10), Duration::from_secs(1));
    }

    #[tokio::test]
    async fn test_retry_policy_success() {
        let policy = RetryPolicy::new(
            3,
            Duration::from_millis(10),
            Duration::from_millis(100),
        );
        
        let mut call_count = 0;
        let result = policy.execute_with_retry(|| {
            call_count += 1;
            async move {
                if call_count < 2 {
                    Err(RouterError::Proxy("Test error".to_string()))
                } else {
                    Ok("Success")
                }
            }
        }).await;
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Success");
        assert_eq!(call_count, 2);
    }

    #[tokio::test]
    async fn test_retry_policy_exhausted() {
        let policy = RetryPolicy::new(
            2,
            Duration::from_millis(10),
            Duration::from_millis(100),
        );
        
        let mut call_count = 0;
        let result = policy.execute_with_retry(|| {
            call_count += 1;
            async move {
                Err::<&str, _>(RouterError::Proxy("Test error".to_string()))
            }
        }).await;
        
        assert!(result.is_err());
        assert_eq!(call_count, 3); // Initial attempt + 2 retries
    }
}
