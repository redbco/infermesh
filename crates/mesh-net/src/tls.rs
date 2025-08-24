//! TLS/mTLS configuration and setup

use crate::{NetworkError, Result};
use rustls_pemfile::{certs, pkcs8_private_keys};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use tokio_rustls::{TlsAcceptor, TlsConnector};
use tokio_rustls::rustls::{ClientConfig, ServerConfig};
use tracing::{debug, info};

/// TLS configuration for both client and server
#[derive(Debug, Clone)]
pub struct TlsConfig {
    /// Client configuration
    pub client_config: Option<Arc<ClientConfig>>,
    
    /// Server configuration
    pub server_config: Option<Arc<ServerConfig>>,
    
    /// Whether to verify peer certificates
    pub verify_peer: bool,
    
    /// Whether to require client certificates
    pub require_client_cert: bool,
}

/// Builder for TLS configuration
#[derive(Debug, Default)]
pub struct TlsConfigBuilder {
    pub(crate) cert_file: Option<String>,
    pub(crate) key_file: Option<String>,
    pub(crate) ca_file: Option<String>,
    pub(crate) verify_peer: bool,
    pub(crate) require_client_cert: bool,
}

impl TlsConfig {
    /// Create a new TLS configuration builder
    pub fn builder() -> TlsConfigBuilder {
        TlsConfigBuilder::default()
    }
    
    /// Create an insecure TLS configuration (for testing)
    pub fn insecure() -> Self {
        Self {
            client_config: None,
            server_config: None,
            verify_peer: false,
            require_client_cert: false,
        }
    }
    
    /// Create a TLS connector for client connections
    pub fn create_connector(&self) -> Result<Option<TlsConnector>> {
        match &self.client_config {
            Some(config) => Ok(Some(TlsConnector::from(config.clone()))),
            None => Ok(None),
        }
    }
    
    /// Create a TLS acceptor for server connections
    pub fn create_acceptor(&self) -> Result<Option<TlsAcceptor>> {
        match &self.server_config {
            Some(config) => Ok(Some(TlsAcceptor::from(config.clone()))),
            None => Ok(None),
        }
    }
}

impl TlsConfigBuilder {
    /// Set the certificate file path
    pub fn with_cert_file<P: AsRef<Path>>(mut self, path: P) -> Result<Self> {
        self.cert_file = Some(path.as_ref().to_string_lossy().to_string());
        Ok(self)
    }
    
    /// Set the private key file path
    pub fn with_key_file<P: AsRef<Path>>(mut self, path: P) -> Result<Self> {
        self.key_file = Some(path.as_ref().to_string_lossy().to_string());
        Ok(self)
    }
    
    /// Set the CA certificate file path
    pub fn with_ca_file<P: AsRef<Path>>(mut self, path: P) -> Result<Self> {
        self.ca_file = Some(path.as_ref().to_string_lossy().to_string());
        Ok(self)
    }
    
    /// Enable peer certificate verification
    pub fn with_peer_verification(mut self) -> Self {
        self.verify_peer = true;
        self
    }
    
    /// Require client certificates
    pub fn with_client_cert_required(mut self) -> Self {
        self.require_client_cert = true;
        self
    }
    
    /// Build the TLS configuration
    pub fn build(self) -> Result<TlsConfig> {
        let mut client_config = None;
        let mut server_config = None;
        
        // Build client configuration if we have CA certificates
        if let Some(ca_file) = &self.ca_file {
            debug!("Building client TLS configuration with CA file: {}", ca_file);
            client_config = Some(Arc::new(self.build_client_config(ca_file)?));
        }
        
        // Build server configuration if we have cert and key files
        if let (Some(cert_file), Some(key_file)) = (&self.cert_file, &self.key_file) {
            debug!("Building server TLS configuration with cert: {}, key: {}", cert_file, key_file);
            server_config = Some(Arc::new(self.build_server_config(cert_file, key_file)?));
        }
        
        Ok(TlsConfig {
            client_config,
            server_config,
            verify_peer: self.verify_peer,
            require_client_cert: self.require_client_cert,
        })
    }
    
    /// Build client configuration
    fn build_client_config(&self, ca_file: &str) -> Result<ClientConfig> {
        let mut root_cert_store = tokio_rustls::rustls::RootCertStore::empty();
        
        // Load CA certificates
        let ca_file = File::open(ca_file)
            .map_err(|e| NetworkError::Certificate(format!("Failed to open CA file {}: {}", ca_file, e)))?;
        let mut ca_reader = BufReader::new(ca_file);
        
        let ca_certs: std::result::Result<Vec<_>, _> = certs(&mut ca_reader).collect();
        let ca_certs = ca_certs
            .map_err(|e| NetworkError::Certificate(format!("Failed to parse CA certificates: {}", e)))?;
        
        for cert in ca_certs {
            root_cert_store.add(cert)
                .map_err(|e| NetworkError::Certificate(format!("Failed to add CA certificate: {}", e)))?;
        }
        
        let config = ClientConfig::builder()
            .with_root_certificates(root_cert_store)
            .with_no_client_auth();
        
        info!("Built client TLS configuration");
        Ok(config)
    }
    
    /// Build server configuration
    fn build_server_config(&self, cert_file: &str, key_file: &str) -> Result<ServerConfig> {
        // Load certificate chain
        let cert_file = File::open(cert_file)
            .map_err(|e| NetworkError::Certificate(format!("Failed to open cert file {}: {}", cert_file, e)))?;
        let mut cert_reader = BufReader::new(cert_file);
        
        let cert_chain: std::result::Result<Vec<_>, _> = certs(&mut cert_reader).collect();
        let cert_chain = cert_chain
            .map_err(|e| NetworkError::Certificate(format!("Failed to parse certificates: {}", e)))?;
        
        // Load private key
        let key_file = File::open(key_file)
            .map_err(|e| NetworkError::Certificate(format!("Failed to open key file {}: {}", key_file, e)))?;
        let mut key_reader = BufReader::new(key_file);
        
        let keys: std::result::Result<Vec<_>, _> = pkcs8_private_keys(&mut key_reader).collect();
        let mut keys = keys
            .map_err(|e| NetworkError::Certificate(format!("Failed to parse private key: {}", e)))?;
        
        if keys.is_empty() {
            return Err(NetworkError::Certificate("No private keys found".to_string()));
        }
        
        let private_key = tokio_rustls::rustls::pki_types::PrivateKeyDer::Pkcs8(keys.remove(0));
        
        let config = if self.require_client_cert {
            // Build with client certificate verification
            let mut root_cert_store = tokio_rustls::rustls::RootCertStore::empty();
            
            if let Some(ca_file) = &self.ca_file {
                let ca_file = File::open(ca_file)
                    .map_err(|e| NetworkError::Certificate(format!("Failed to open CA file {}: {}", ca_file, e)))?;
                let mut ca_reader = BufReader::new(ca_file);
                
                let ca_certs: std::result::Result<Vec<_>, _> = certs(&mut ca_reader).collect();
                let ca_certs = ca_certs
                    .map_err(|e| NetworkError::Certificate(format!("Failed to parse CA certificates: {}", e)))?;
                
                for cert in ca_certs {
                    root_cert_store.add(cert)
                        .map_err(|e| NetworkError::Certificate(format!("Failed to add CA certificate: {}", e)))?;
                }
            }
            
            ServerConfig::builder()
                .with_client_cert_verifier(
                    tokio_rustls::rustls::server::WebPkiClientVerifier::builder(Arc::new(root_cert_store))
                        .build()
                        .map_err(|e| NetworkError::Certificate(format!("Failed to build client verifier: {}", e)))?
                )
                .with_single_cert(cert_chain, private_key)
                .map_err(|e| NetworkError::Certificate(format!("Failed to build server config: {}", e)))?
        } else {
            ServerConfig::builder()
                .with_no_client_auth()
                .with_single_cert(cert_chain, private_key)
                .map_err(|e| NetworkError::Certificate(format!("Failed to build server config: {}", e)))?
        };
        
        info!("Built server TLS configuration");
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_insecure_tls_config() {
        let config = TlsConfig::insecure();
        assert!(config.client_config.is_none());
        assert!(config.server_config.is_none());
        assert!(!config.verify_peer);
        assert!(!config.require_client_cert);
    }

    #[test]
    fn test_tls_config_builder() {
        let builder = TlsConfig::builder()
            .with_peer_verification()
            .with_client_cert_required();
        
        assert!(builder.verify_peer);
        assert!(builder.require_client_cert);
    }

    #[test]
    fn test_tls_config_builder_with_files() {
        let mut cert_file = NamedTempFile::new().unwrap();
        let mut key_file = NamedTempFile::new().unwrap();
        let mut ca_file = NamedTempFile::new().unwrap();
        
        // Write dummy content (not valid certificates, just for path testing)
        writeln!(cert_file, "dummy cert").unwrap();
        writeln!(key_file, "dummy key").unwrap();
        writeln!(ca_file, "dummy ca").unwrap();
        
        let builder = TlsConfig::builder()
            .with_cert_file(cert_file.path()).unwrap()
            .with_key_file(key_file.path()).unwrap()
            .with_ca_file(ca_file.path()).unwrap();
        
        assert!(builder.cert_file.is_some());
        assert!(builder.key_file.is_some());
        assert!(builder.ca_file.is_some());
    }

    #[tokio::test]
    async fn test_insecure_connector_creation() {
        let config = TlsConfig::insecure();
        let connector = config.create_connector().unwrap();
        assert!(connector.is_none());
        
        let acceptor = config.create_acceptor().unwrap();
        assert!(acceptor.is_none());
    }
}
