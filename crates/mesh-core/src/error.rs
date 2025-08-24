//! Error handling for infermesh
//!
//! Provides a unified error type and result type for use across all infermesh components.

/// Result type alias for infermesh operations
pub type Result<T> = std::result::Result<T, Error>;

/// Unified error type for infermesh
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Configuration-related errors
    #[error("Configuration error: {0}")]
    InvalidConfiguration(String),

    /// Network-related errors
    #[error("Network error: {0}")]
    Network(String),

    /// gRPC/transport errors
    #[error("Transport error: {0}")]
    Transport(String),

    /// Resource not found
    #[error("Resource not found: {0}")]
    NotFound(String),

    /// Resource already exists
    #[error("Resource already exists: {0}")]
    AlreadyExists(String),

    /// Permission denied
    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    /// Resource temporarily unavailable
    #[error("Resource unavailable: {0}")]
    Unavailable(String),

    /// Operation timeout
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// Invalid request or parameters
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Internal server error
    #[error("Internal error: {0}")]
    Internal(String),

    /// Runtime control errors
    #[error("Runtime control error: {0}")]
    RuntimeControl(String),

    /// GPU telemetry errors
    #[error("GPU telemetry error: {0}")]
    GpuTelemetry(String),

    /// State management errors
    #[error("State error: {0}")]
    State(String),

    /// Gossip protocol errors
    #[error("Gossip error: {0}")]
    Gossip(String),

    /// Raft consensus errors
    #[error("Raft error: {0}")]
    Raft(String),

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// YAML parsing errors
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// Configuration parsing errors
    #[error("Config error: {0}")]
    Config(#[from] config::ConfigError),

    /// Generic error with context
    #[error("Error: {0}")]
    Other(#[from] anyhow::Error),
}

impl Error {
    /// Create a configuration error
    pub fn config(msg: impl Into<String>) -> Self {
        Self::InvalidConfiguration(msg.into())
    }

    /// Create a network error
    pub fn network(msg: impl Into<String>) -> Self {
        Self::Network(msg.into())
    }

    /// Create a transport error
    pub fn transport(msg: impl Into<String>) -> Self {
        Self::Transport(msg.into())
    }

    /// Create a not found error
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }

    /// Create an already exists error
    pub fn already_exists(msg: impl Into<String>) -> Self {
        Self::AlreadyExists(msg.into())
    }

    /// Create a permission denied error
    pub fn permission_denied(msg: impl Into<String>) -> Self {
        Self::PermissionDenied(msg.into())
    }

    /// Create an unavailable error
    pub fn unavailable(msg: impl Into<String>) -> Self {
        Self::Unavailable(msg.into())
    }

    /// Create a timeout error
    pub fn timeout(msg: impl Into<String>) -> Self {
        Self::Timeout(msg.into())
    }

    /// Create an invalid request error
    pub fn invalid_request(msg: impl Into<String>) -> Self {
        Self::InvalidRequest(msg.into())
    }

    /// Create an internal error
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }

    /// Create a runtime control error
    pub fn runtime_control(msg: impl Into<String>) -> Self {
        Self::RuntimeControl(msg.into())
    }

    /// Create a GPU telemetry error
    pub fn gpu_telemetry(msg: impl Into<String>) -> Self {
        Self::GpuTelemetry(msg.into())
    }

    /// Create a state error
    pub fn state(msg: impl Into<String>) -> Self {
        Self::State(msg.into())
    }

    /// Create a gossip error
    pub fn gossip(msg: impl Into<String>) -> Self {
        Self::Gossip(msg.into())
    }

    /// Create a raft error
    pub fn raft(msg: impl Into<String>) -> Self {
        Self::Raft(msg.into())
    }

    /// Create a serialization error
    pub fn serialization(msg: impl Into<String>) -> Self {
        Self::Serialization(msg.into())
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Error::Network(_)
                | Error::Transport(_)
                | Error::Unavailable(_)
                | Error::Timeout(_)
                | Error::Internal(_)
        )
    }

    /// Check if this error indicates a client-side problem
    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            Error::InvalidConfiguration(_)
                | Error::InvalidRequest(_)
                | Error::NotFound(_)
                | Error::AlreadyExists(_)
                | Error::PermissionDenied(_)
        )
    }

    /// Check if this error indicates a server-side problem
    pub fn is_server_error(&self) -> bool {
        !self.is_client_error()
    }

    /// Get the error category for metrics/logging
    pub fn category(&self) -> &'static str {
        match self {
            Error::InvalidConfiguration(_) => "configuration",
            Error::Network(_) => "network",
            Error::Transport(_) => "transport",
            Error::NotFound(_) => "not_found",
            Error::AlreadyExists(_) => "already_exists",
            Error::PermissionDenied(_) => "permission_denied",
            Error::Unavailable(_) => "unavailable",
            Error::Timeout(_) => "timeout",
            Error::InvalidRequest(_) => "invalid_request",
            Error::Internal(_) => "internal",
            Error::RuntimeControl(_) => "runtime_control",
            Error::GpuTelemetry(_) => "gpu_telemetry",
            Error::State(_) => "state",
            Error::Gossip(_) => "gossip",
            Error::Raft(_) => "raft",
            Error::Serialization(_) => "serialization",
            Error::Io(_) => "io",
            Error::Json(_) => "json",
            Error::Yaml(_) => "yaml",
            Error::Config(_) => "config",
            Error::Other(_) => "other",
        }
    }

    /// Convert to HTTP status code (useful for REST APIs)
    pub fn to_http_status(&self) -> u16 {
        match self {
            Error::InvalidConfiguration(_) | Error::InvalidRequest(_) => 400, // Bad Request
            Error::PermissionDenied(_) => 403,                                // Forbidden
            Error::NotFound(_) => 404,                                        // Not Found
            Error::AlreadyExists(_) => 409,                                   // Conflict
            Error::Timeout(_) => 408,                                         // Request Timeout
            Error::Unavailable(_) => 503,                                     // Service Unavailable
            Error::Network(_) | Error::Transport(_) => 502,                   // Bad Gateway
            _ => 500,                                                         // Internal Server Error
        }
    }
}

/// Extension trait for adding context to Results
pub trait ErrorContext<T> {
    /// Add context to an error
    fn with_context(self, context: impl Into<String>) -> Result<T>;

    /// Add context to an error using a closure
    fn with_context_fn<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
}

impl<T, E> ErrorContext<T> for std::result::Result<T, E>
where
    E: Into<Error>,
{
    fn with_context(self, context: impl Into<String>) -> Result<T> {
        self.map_err(|e| {
            let original_error = e.into();
            Error::Other(anyhow::anyhow!("{}: {}", context.into(), original_error))
        })
    }

    fn with_context_fn<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| {
            let original_error = e.into();
            Error::Other(anyhow::anyhow!("{}: {}", f(), original_error))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = Error::config("invalid setting");
        assert!(matches!(err, Error::InvalidConfiguration(_)));
        assert_eq!(err.to_string(), "Configuration error: invalid setting");
    }

    #[test]
    fn test_error_categories() {
        assert_eq!(Error::config("test").category(), "configuration");
        assert_eq!(Error::network("test").category(), "network");
        assert_eq!(Error::not_found("test").category(), "not_found");
    }

    #[test]
    fn test_error_classification() {
        let client_err = Error::invalid_request("bad params");
        assert!(client_err.is_client_error());
        assert!(!client_err.is_server_error());
        assert!(!client_err.is_retryable());

        let server_err = Error::internal("database down");
        assert!(!server_err.is_client_error());
        assert!(server_err.is_server_error());
        assert!(server_err.is_retryable());
    }

    #[test]
    fn test_http_status_codes() {
        assert_eq!(Error::invalid_request("test").to_http_status(), 400);
        assert_eq!(Error::not_found("test").to_http_status(), 404);
        assert_eq!(Error::internal("test").to_http_status(), 500);
    }

    #[test]
    fn test_error_context() {
        let result: std::result::Result<(), std::io::Error> =
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"));

        let err = result.with_context("failed to read config file").unwrap_err();
        
        assert!(matches!(err, Error::Other(_)));
        assert!(err.to_string().contains("failed to read config file"));
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn test_error_context_fn() {
        let result: std::result::Result<(), std::io::Error> = 
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "original error"));

        let err = result
            .with_context_fn(|| format!("operation failed at {}", "location"))
            .unwrap_err();

        assert!(err.to_string().contains("operation failed at location"));
        assert!(err.to_string().contains("original error"));
    }
}
