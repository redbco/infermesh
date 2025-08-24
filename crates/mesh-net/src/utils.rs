//! Network utilities and helper functions

use anyhow::{anyhow, Result};
use std::net::{SocketAddr, ToSocketAddrs};
use tracing::{debug, warn};

/// Resolve a hostname and port to a socket address
pub async fn resolve_address(host: &str, port: u16) -> Result<SocketAddr> {
    let address = format!("{}:{}", host, port);
    
    // Try to parse as a direct socket address first
    if let Ok(addr) = address.parse::<SocketAddr>() {
        debug!(address = %addr, "Resolved address directly");
        return Ok(addr);
    }
    
    // Perform DNS resolution
    match tokio::task::spawn_blocking(move || {
        address.to_socket_addrs()?.next()
            .ok_or_else(|| anyhow!("No addresses found for {}", address))
    }).await {
        Ok(Ok(addr)) => {
            debug!(address = %addr, host = %host, port = port, "Resolved address via DNS");
            Ok(addr)
        }
        Ok(Err(e)) => {
            warn!(error = %e, host = %host, port = port, "Failed to resolve address");
            Err(e)
        }
        Err(e) => {
            warn!(error = %e, host = %host, port = port, "DNS resolution task failed");
            Err(anyhow!("DNS resolution task failed: {}", e))
        }
    }
}

/// Validate a hostname according to RFC standards
pub fn validate_hostname(hostname: &str) -> bool {
    if hostname.is_empty() || hostname.len() > 253 {
        return false;
    }
    
    // Check for valid characters and structure
    for label in hostname.split('.') {
        if label.is_empty() || label.len() > 63 {
            return false;
        }
        
        // Labels must start and end with alphanumeric characters
        if !label.chars().next().unwrap_or(' ').is_ascii_alphanumeric() ||
           !label.chars().last().unwrap_or(' ').is_ascii_alphanumeric() {
            return false;
        }
        
        // Labels can only contain alphanumeric characters and hyphens
        if !label.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
            return false;
        }
    }
    
    true
}

/// Extract hostname from a URL or address string
pub fn extract_hostname(address: &str) -> Option<&str> {
    // Remove protocol if present
    let address = if let Some(stripped) = address.strip_prefix("http://") {
        stripped
    } else if let Some(stripped) = address.strip_prefix("https://") {
        stripped
    } else {
        address
    };
    
    // Find the hostname part (before port or path)
    let hostname = address.split(':').next()?.split('/').next()?;
    
    if hostname.is_empty() {
        None
    } else {
        Some(hostname)
    }
}

/// Check if an address is a loopback address
pub fn is_loopback_address(addr: &SocketAddr) -> bool {
    match addr {
        SocketAddr::V4(v4) => v4.ip().is_loopback(),
        SocketAddr::V6(v6) => v6.ip().is_loopback(),
    }
}

/// Check if an address is a private/internal address
pub fn is_private_address(addr: &SocketAddr) -> bool {
    match addr {
        SocketAddr::V4(v4) => {
            let ip = v4.ip();
            ip.is_private() || ip.is_loopback() || ip.is_link_local()
        }
        SocketAddr::V6(v6) => {
            let ip = v6.ip();
            ip.is_loopback() || ip.is_unspecified() || 
            // Check for private IPv6 ranges
            (ip.segments()[0] & 0xfe00) == 0xfc00 || // fc00::/7 (Unique Local)
            (ip.segments()[0] & 0xffc0) == 0xfe80    // fe80::/10 (Link Local)
        }
    }
}

/// Format a duration in a human-readable way
pub fn format_duration(duration: std::time::Duration) -> String {
    let total_secs = duration.as_secs();
    let millis = duration.subsec_millis();
    
    if total_secs == 0 {
        format!("{}ms", millis)
    } else if total_secs < 60 {
        format!("{}.{:03}s", total_secs, millis)
    } else if total_secs < 3600 {
        let mins = total_secs / 60;
        let secs = total_secs % 60;
        format!("{}m{}s", mins, secs)
    } else {
        let hours = total_secs / 3600;
        let mins = (total_secs % 3600) / 60;
        let secs = total_secs % 60;
        format!("{}h{}m{}s", hours, mins, secs)
    }
}

/// Generate a random port number in the ephemeral range
pub fn random_port() -> u16 {
    use rand::Rng;
    rand::thread_rng().gen_range(49152..65535)
}

/// Check if a port is available for binding
pub async fn is_port_available(port: u16) -> bool {
    match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
        Ok(_) => true,
        Err(_) => false,
    }
}

/// Find an available port starting from a given port
pub async fn find_available_port(start_port: u16) -> Option<u16> {
    for port in start_port..65535 {
        if is_port_available(port).await {
            return Some(port);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_resolve_address() {
        // Test direct IP address
        let addr = resolve_address("127.0.0.1", 8080).await.unwrap();
        assert_eq!(addr.port(), 8080);
        assert!(addr.ip().is_loopback());
        
        // Test localhost resolution
        let addr = resolve_address("localhost", 8080).await.unwrap();
        assert_eq!(addr.port(), 8080);
    }

    #[test]
    fn test_validate_hostname() {
        assert!(validate_hostname("example.com"));
        assert!(validate_hostname("sub.example.com"));
        assert!(validate_hostname("test-host"));
        assert!(validate_hostname("a.b.c.d"));
        
        assert!(!validate_hostname(""));
        assert!(!validate_hostname(".example.com"));
        assert!(!validate_hostname("example.com."));
        assert!(!validate_hostname("ex ample.com"));
        assert!(!validate_hostname("-example.com"));
        assert!(!validate_hostname("example-.com"));
    }

    #[test]
    fn test_extract_hostname() {
        assert_eq!(extract_hostname("example.com"), Some("example.com"));
        assert_eq!(extract_hostname("example.com:8080"), Some("example.com"));
        assert_eq!(extract_hostname("http://example.com"), Some("example.com"));
        assert_eq!(extract_hostname("https://example.com:8080"), Some("example.com"));
        assert_eq!(extract_hostname("https://example.com:8080/path"), Some("example.com"));
        assert_eq!(extract_hostname(""), None);
    }

    #[test]
    fn test_is_loopback_address() {
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        assert!(is_loopback_address(&addr));
        
        let addr: SocketAddr = "[::1]:8080".parse().unwrap();
        assert!(is_loopback_address(&addr));
        
        let addr: SocketAddr = "192.168.1.1:8080".parse().unwrap();
        assert!(!is_loopback_address(&addr));
    }

    #[test]
    fn test_is_private_address() {
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        assert!(is_private_address(&addr));
        
        let addr: SocketAddr = "192.168.1.1:8080".parse().unwrap();
        assert!(is_private_address(&addr));
        
        let addr: SocketAddr = "10.0.0.1:8080".parse().unwrap();
        assert!(is_private_address(&addr));
        
        let addr: SocketAddr = "172.16.0.1:8080".parse().unwrap();
        assert!(is_private_address(&addr));
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(std::time::Duration::from_millis(500)), "500ms");
        assert_eq!(format_duration(std::time::Duration::from_secs(1)), "1.000s");
        assert_eq!(format_duration(std::time::Duration::from_secs(65)), "1m5s");
        assert_eq!(format_duration(std::time::Duration::from_secs(3661)), "1h1m1s");
    }

    #[test]
    fn test_random_port() {
        let port = random_port();
        assert!(port >= 49152);
        assert!(port < 65535);
    }

    #[tokio::test]
    async fn test_is_port_available() {
        // Port 0 should always be available (system assigns)
        assert!(is_port_available(0).await);
        
        // Well-known ports might not be available
        // This test is environment-dependent, so we just check it doesn't panic
        let _ = is_port_available(80).await;
    }

    #[tokio::test]
    async fn test_find_available_port() {
        let port = find_available_port(50000).await;
        assert!(port.is_some());
        if let Some(port) = port {
            assert!(port >= 50000);
        }
    }
}
