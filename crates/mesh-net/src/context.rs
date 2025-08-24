//! Request context and tracing support

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Request context for distributed tracing and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestContext {
    /// Unique request ID
    pub request_id: String,
    
    /// Trace context
    pub trace: TraceContext,
    
    /// Request metadata
    pub metadata: HashMap<String, String>,
    
    /// Request timeout in milliseconds
    pub timeout_ms: Option<u64>,
}

/// Distributed tracing context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    /// Trace ID
    pub trace_id: String,
    
    /// Span ID
    pub span_id: String,
    
    /// Parent span ID
    pub parent_span_id: Option<String>,
    
    /// Trace flags
    pub flags: u8,
    
    /// Baggage items
    pub baggage: HashMap<String, String>,
}

impl RequestContext {
    /// Create a new request context
    pub fn new() -> Self {
        Self {
            request_id: Uuid::new_v4().to_string(),
            trace: TraceContext::new(),
            metadata: HashMap::new(),
            timeout_ms: None,
        }
    }
    
    /// Create a request context with a specific request ID
    pub fn with_request_id(request_id: String) -> Self {
        Self {
            request_id,
            trace: TraceContext::new(),
            metadata: HashMap::new(),
            timeout_ms: None,
        }
    }
    
    /// Add metadata to the request context
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    /// Set request timeout
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
    
    /// Create a child context for a downstream request
    pub fn create_child(&self) -> Self {
        Self {
            request_id: Uuid::new_v4().to_string(),
            trace: self.trace.create_child(),
            metadata: self.metadata.clone(),
            timeout_ms: self.timeout_ms,
        }
    }
    
    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
    
    /// Set metadata value
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
    
    /// Convert to HTTP headers
    pub fn to_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        
        headers.insert("x-request-id".to_string(), self.request_id.clone());
        headers.insert("x-trace-id".to_string(), self.trace.trace_id.clone());
        headers.insert("x-span-id".to_string(), self.trace.span_id.clone());
        
        if let Some(parent_span_id) = &self.trace.parent_span_id {
            headers.insert("x-parent-span-id".to_string(), parent_span_id.clone());
        }
        
        headers.insert("x-trace-flags".to_string(), self.trace.flags.to_string());
        
        // Add baggage
        for (key, value) in &self.trace.baggage {
            headers.insert(format!("x-baggage-{}", key), value.clone());
        }
        
        // Add metadata
        for (key, value) in &self.metadata {
            headers.insert(format!("x-meta-{}", key), value.clone());
        }
        
        if let Some(timeout) = self.timeout_ms {
            headers.insert("x-timeout-ms".to_string(), timeout.to_string());
        }
        
        headers
    }
    
    /// Create from HTTP headers
    pub fn from_headers(headers: &HashMap<String, String>) -> Self {
        let request_id = headers.get("x-request-id")
            .cloned()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        
        let trace_id = headers.get("x-trace-id")
            .cloned()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        
        let span_id = headers.get("x-span-id")
            .cloned()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        
        let parent_span_id = headers.get("x-parent-span-id").cloned();
        
        let flags = headers.get("x-trace-flags")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        
        let timeout_ms = headers.get("x-timeout-ms")
            .and_then(|s| s.parse().ok());
        
        // Extract baggage
        let mut baggage = HashMap::new();
        for (key, value) in headers {
            if let Some(baggage_key) = key.strip_prefix("x-baggage-") {
                baggage.insert(baggage_key.to_string(), value.clone());
            }
        }
        
        // Extract metadata
        let mut metadata = HashMap::new();
        for (key, value) in headers {
            if let Some(meta_key) = key.strip_prefix("x-meta-") {
                metadata.insert(meta_key.to_string(), value.clone());
            }
        }
        
        Self {
            request_id,
            trace: TraceContext {
                trace_id,
                span_id,
                parent_span_id,
                flags,
                baggage,
            },
            metadata,
            timeout_ms,
        }
    }
}

impl TraceContext {
    /// Create a new trace context
    pub fn new() -> Self {
        Self {
            trace_id: Uuid::new_v4().to_string(),
            span_id: Uuid::new_v4().to_string(),
            parent_span_id: None,
            flags: 0,
            baggage: HashMap::new(),
        }
    }
    
    /// Create a child trace context
    pub fn create_child(&self) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            span_id: Uuid::new_v4().to_string(),
            parent_span_id: Some(self.span_id.clone()),
            flags: self.flags,
            baggage: self.baggage.clone(),
        }
    }
    
    /// Add baggage item
    pub fn with_baggage(mut self, key: String, value: String) -> Self {
        self.baggage.insert(key, value);
        self
    }
    
    /// Get baggage item
    pub fn get_baggage(&self, key: &str) -> Option<&String> {
        self.baggage.get(key)
    }
    
    /// Set baggage item
    pub fn set_baggage(&mut self, key: String, value: String) {
        self.baggage.insert(key, value);
    }
}

impl Default for RequestContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_context_creation() {
        let ctx = RequestContext::new();
        assert!(!ctx.request_id.is_empty());
        assert!(!ctx.trace.trace_id.is_empty());
        assert!(!ctx.trace.span_id.is_empty());
        assert!(ctx.trace.parent_span_id.is_none());
        assert!(ctx.metadata.is_empty());
        assert!(ctx.timeout_ms.is_none());
    }

    #[test]
    fn test_request_context_with_metadata() {
        let ctx = RequestContext::new()
            .with_metadata("service".to_string(), "test".to_string())
            .with_timeout_ms(5000);
        
        assert_eq!(ctx.get_metadata("service"), Some(&"test".to_string()));
        assert_eq!(ctx.timeout_ms, Some(5000));
    }

    #[test]
    fn test_child_context_creation() {
        let parent = RequestContext::new();
        let child = parent.create_child();
        
        assert_ne!(parent.request_id, child.request_id);
        assert_eq!(parent.trace.trace_id, child.trace.trace_id);
        assert_ne!(parent.trace.span_id, child.trace.span_id);
        assert_eq!(child.trace.parent_span_id, Some(parent.trace.span_id));
    }

    #[test]
    fn test_headers_conversion() {
        let ctx = RequestContext::new()
            .with_metadata("service".to_string(), "test".to_string())
            .with_timeout_ms(5000);
        
        let headers = ctx.to_headers();
        assert!(headers.contains_key("x-request-id"));
        assert!(headers.contains_key("x-trace-id"));
        assert!(headers.contains_key("x-span-id"));
        assert!(headers.contains_key("x-meta-service"));
        assert!(headers.contains_key("x-timeout-ms"));
        
        let reconstructed = RequestContext::from_headers(&headers);
        assert_eq!(ctx.request_id, reconstructed.request_id);
        assert_eq!(ctx.trace.trace_id, reconstructed.trace.trace_id);
        assert_eq!(ctx.trace.span_id, reconstructed.trace.span_id);
        assert_eq!(ctx.timeout_ms, reconstructed.timeout_ms);
        assert_eq!(ctx.get_metadata("service"), reconstructed.get_metadata("service"));
    }

    #[test]
    fn test_trace_context_baggage() {
        let mut trace = TraceContext::new()
            .with_baggage("user_id".to_string(), "123".to_string());
        
        assert_eq!(trace.get_baggage("user_id"), Some(&"123".to_string()));
        
        trace.set_baggage("session_id".to_string(), "abc".to_string());
        assert_eq!(trace.get_baggage("session_id"), Some(&"abc".to_string()));
    }
}
