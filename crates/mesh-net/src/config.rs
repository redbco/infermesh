//! Network configuration

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Connection timeout
    pub connect_timeout: Duration,
    
    /// Request timeout
    pub request_timeout: Duration,
    
    /// Keep-alive timeout
    pub keep_alive_timeout: Duration,
    
    /// Maximum number of connections per host
    pub max_connections_per_host: usize,
    
    /// Maximum idle connections
    pub max_idle_connections: usize,
    
    /// Connection idle timeout
    pub idle_timeout: Duration,
    
    /// Enable HTTP/2
    pub enable_http2: bool,
    
    /// Enable connection pooling
    pub enable_pooling: bool,
    
    /// TCP keep-alive settings
    pub tcp_keepalive: Option<TcpKeepAliveConfig>,
    
    /// Buffer sizes
    pub buffer_sizes: BufferConfig,
}

/// TCP keep-alive configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcpKeepAliveConfig {
    /// Keep-alive time
    pub time: Duration,
    
    /// Keep-alive interval
    pub interval: Duration,
    
    /// Keep-alive retries
    pub retries: u32,
}

/// Buffer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferConfig {
    /// Send buffer size
    pub send_buffer_size: usize,
    
    /// Receive buffer size
    pub recv_buffer_size: usize,
    
    /// Initial window size for HTTP/2
    pub initial_window_size: u32,
    
    /// Maximum frame size for HTTP/2
    pub max_frame_size: u32,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(30),
            keep_alive_timeout: Duration::from_secs(90),
            max_connections_per_host: 10,
            max_idle_connections: 100,
            idle_timeout: Duration::from_secs(300),
            enable_http2: true,
            enable_pooling: true,
            tcp_keepalive: Some(TcpKeepAliveConfig::default()),
            buffer_sizes: BufferConfig::default(),
        }
    }
}

impl Default for TcpKeepAliveConfig {
    fn default() -> Self {
        Self {
            time: Duration::from_secs(7200), // 2 hours
            interval: Duration::from_secs(75),
            retries: 9,
        }
    }
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            send_buffer_size: 64 * 1024,    // 64KB
            recv_buffer_size: 64 * 1024,    // 64KB
            initial_window_size: 65535,     // 64KB - 1
            max_frame_size: 16384,          // 16KB
        }
    }
}

impl NetworkConfig {
    /// Create a new network configuration
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set connection timeout
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }
    
    /// Set request timeout
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }
    
    /// Set maximum connections per host
    pub fn with_max_connections_per_host(mut self, max: usize) -> Self {
        self.max_connections_per_host = max;
        self
    }
    
    /// Disable connection pooling
    pub fn without_pooling(mut self) -> Self {
        self.enable_pooling = false;
        self
    }
    
    /// Disable HTTP/2
    pub fn without_http2(mut self) -> Self {
        self.enable_http2 = false;
        self
    }
    
    /// Set TCP keep-alive configuration
    pub fn with_tcp_keepalive(mut self, config: TcpKeepAliveConfig) -> Self {
        self.tcp_keepalive = Some(config);
        self
    }
    
    /// Disable TCP keep-alive
    pub fn without_tcp_keepalive(mut self) -> Self {
        self.tcp_keepalive = None;
        self
    }
    
    /// Set buffer configuration
    pub fn with_buffer_config(mut self, config: BufferConfig) -> Self {
        self.buffer_sizes = config;
        self
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.connect_timeout.is_zero() {
            return Err("connect_timeout must be greater than zero".to_string());
        }
        
        if self.request_timeout.is_zero() {
            return Err("request_timeout must be greater than zero".to_string());
        }
        
        if self.max_connections_per_host == 0 {
            return Err("max_connections_per_host must be greater than zero".to_string());
        }
        
        if self.max_idle_connections == 0 {
            return Err("max_idle_connections must be greater than zero".to_string());
        }
        
        if self.buffer_sizes.send_buffer_size == 0 {
            return Err("send_buffer_size must be greater than zero".to_string());
        }
        
        if self.buffer_sizes.recv_buffer_size == 0 {
            return Err("recv_buffer_size must be greater than zero".to_string());
        }
        
        if self.buffer_sizes.initial_window_size == 0 {
            return Err("initial_window_size must be greater than zero".to_string());
        }
        
        if self.buffer_sizes.max_frame_size == 0 {
            return Err("max_frame_size must be greater than zero".to_string());
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = NetworkConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.enable_http2);
        assert!(config.enable_pooling);
        assert!(config.tcp_keepalive.is_some());
    }

    #[test]
    fn test_config_builder() {
        let config = NetworkConfig::new()
            .with_connect_timeout(Duration::from_secs(5))
            .with_request_timeout(Duration::from_secs(15))
            .with_max_connections_per_host(20)
            .without_pooling()
            .without_http2();
        
        assert_eq!(config.connect_timeout, Duration::from_secs(5));
        assert_eq!(config.request_timeout, Duration::from_secs(15));
        assert_eq!(config.max_connections_per_host, 20);
        assert!(!config.enable_pooling);
        assert!(!config.enable_http2);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let mut config = NetworkConfig::default();
        
        // Test zero connect timeout
        config.connect_timeout = Duration::from_secs(0);
        assert!(config.validate().is_err());
        
        // Test zero max connections
        config = NetworkConfig::default();
        config.max_connections_per_host = 0;
        assert!(config.validate().is_err());
        
        // Test zero buffer size
        config = NetworkConfig::default();
        config.buffer_sizes.send_buffer_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_tcp_keepalive_config() {
        let keepalive = TcpKeepAliveConfig::default();
        assert!(keepalive.time.as_secs() > 0);
        assert!(keepalive.interval.as_secs() > 0);
        assert!(keepalive.retries > 0);
    }

    #[test]
    fn test_buffer_config() {
        let buffer = BufferConfig::default();
        assert!(buffer.send_buffer_size > 0);
        assert!(buffer.recv_buffer_size > 0);
        assert!(buffer.initial_window_size > 0);
        assert!(buffer.max_frame_size > 0);
    }
}
