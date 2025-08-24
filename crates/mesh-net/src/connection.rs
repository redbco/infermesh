//! Connection management and abstractions

use crate::{NetworkError, Result, TlsConfig};
use std::net::SocketAddr;

use std::time::Duration;
use tokio::net::TcpStream;
use tokio::time::timeout;
use tokio_rustls::{client::TlsStream as ClientTlsStream, server::TlsStream as ServerTlsStream};
use tracing::{debug, error};

/// Connection abstraction
#[derive(Debug)]
pub enum Connection {
    /// Plain TCP connection
    Tcp(TcpStream),
    /// TLS-encrypted connection (client)
    TlsClient(ClientTlsStream<TcpStream>),
    /// TLS-encrypted connection (server)
    TlsServer(ServerTlsStream<TcpStream>),
}

/// Connection manager for creating and managing connections
#[derive(Debug, Clone)]
pub struct ConnectionManager {
    tls_config: TlsConfig,
    connect_timeout: Duration,
}

impl Connection {
    /// Get the peer address of the connection
    pub fn peer_addr(&self) -> Result<SocketAddr> {
        match self {
            Connection::Tcp(stream) => stream.peer_addr()
                .map_err(|e| NetworkError::Connection(format!("Failed to get peer address: {}", e))),
            Connection::TlsClient(stream) => stream.get_ref().0.peer_addr()
                .map_err(|e| NetworkError::Connection(format!("Failed to get peer address: {}", e))),
            Connection::TlsServer(stream) => stream.get_ref().0.peer_addr()
                .map_err(|e| NetworkError::Connection(format!("Failed to get peer address: {}", e))),
        }
    }
    
    /// Get the local address of the connection
    pub fn local_addr(&self) -> Result<SocketAddr> {
        match self {
            Connection::Tcp(stream) => stream.local_addr()
                .map_err(|e| NetworkError::Connection(format!("Failed to get local address: {}", e))),
            Connection::TlsClient(stream) => stream.get_ref().0.local_addr()
                .map_err(|e| NetworkError::Connection(format!("Failed to get local address: {}", e))),
            Connection::TlsServer(stream) => stream.get_ref().0.local_addr()
                .map_err(|e| NetworkError::Connection(format!("Failed to get local address: {}", e))),
        }
    }
    
    /// Check if the connection is encrypted
    pub fn is_encrypted(&self) -> bool {
        matches!(self, Connection::TlsClient(_) | Connection::TlsServer(_))
    }
}

impl ConnectionManager {
    /// Create a new connection manager
    pub fn new(tls_config: TlsConfig, connect_timeout: Duration) -> Self {
        Self {
            tls_config,
            connect_timeout,
        }
    }
    
    /// Connect to a remote address
    pub async fn connect(&self, addr: SocketAddr) -> Result<Connection> {
        debug!("Connecting to {}", addr);
        
        // Establish TCP connection with timeout
        let tcp_stream = timeout(self.connect_timeout, TcpStream::connect(addr))
            .await
            .map_err(|_| NetworkError::Timeout(format!("Connection timeout to {}", addr)))?
            .map_err(|e| NetworkError::Connection(format!("Failed to connect to {}: {}", addr, e)))?;
        
        // Apply TLS if configured
        if let Some(connector) = self.tls_config.create_connector()? {
            debug!("Establishing TLS connection to {}", addr);
            
            // Extract hostname from address (simplified)
            let hostname = addr.ip().to_string();
            let server_name = tokio_rustls::rustls::pki_types::ServerName::try_from(hostname.clone())
                .map_err(|e| NetworkError::Certificate(format!("Invalid server name {}: {}", hostname, e)))?;
            
            let tls_stream = connector.connect(server_name, tcp_stream)
                .await
                .map_err(|e| NetworkError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
            
            debug!("TLS connection established to {}", addr);
            Ok(Connection::TlsClient(tls_stream))
        } else {
            debug!("Plain TCP connection established to {}", addr);
            Ok(Connection::Tcp(tcp_stream))
        }
    }
    
    /// Connect to a remote address with hostname for TLS SNI
    pub async fn connect_with_hostname(&self, addr: SocketAddr, hostname: &str) -> Result<Connection> {
        debug!("Connecting to {} with hostname {}", addr, hostname);
        
        // Establish TCP connection with timeout
        let tcp_stream = timeout(self.connect_timeout, TcpStream::connect(addr))
            .await
            .map_err(|_| NetworkError::Timeout(format!("Connection timeout to {}", addr)))?
            .map_err(|e| NetworkError::Connection(format!("Failed to connect to {}: {}", addr, e)))?;
        
        // Apply TLS if configured
        if let Some(connector) = self.tls_config.create_connector()? {
            debug!("Establishing TLS connection to {} with hostname {}", addr, hostname);
            
            let server_name = tokio_rustls::rustls::pki_types::ServerName::try_from(hostname.to_string())
                .map_err(|e| NetworkError::Certificate(format!("Invalid server name {}: {}", hostname, e)))?;
            
            let tls_stream = connector.connect(server_name, tcp_stream)
                .await
                .map_err(|e| NetworkError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
            
            debug!("TLS connection established to {} with hostname {}", addr, hostname);
            Ok(Connection::TlsClient(tls_stream))
        } else {
            debug!("Plain TCP connection established to {}", addr);
            Ok(Connection::Tcp(tcp_stream))
        }
    }
    
    /// Test connectivity to a remote address
    pub async fn test_connectivity(&self, addr: SocketAddr) -> Result<bool> {
        match self.connect(addr).await {
            Ok(_) => {
                debug!("Connectivity test to {} succeeded", addr);
                Ok(true)
            }
            Err(e) => {
                error!("Connectivity test to {} failed: {}", addr, e);
                Ok(false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_connection_manager_creation() {
        let tls_config = TlsConfig::insecure();
        let manager = ConnectionManager::new(tls_config, Duration::from_secs(10));
        assert_eq!(manager.connect_timeout, Duration::from_secs(10));
    }

    #[tokio::test]
    async fn test_connection_to_invalid_address() {
        let tls_config = TlsConfig::insecure();
        let manager = ConnectionManager::new(tls_config, Duration::from_millis(100));
        
        // Try to connect to a non-routable address
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1)), 12345);
        let result = manager.connect(addr).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_connectivity_test() {
        let tls_config = TlsConfig::insecure();
        let manager = ConnectionManager::new(tls_config, Duration::from_millis(100));
        
        // Test connectivity to a non-routable address
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1)), 12345);
        let result = manager.test_connectivity(addr).await.unwrap();
        assert!(!result);
    }
}
