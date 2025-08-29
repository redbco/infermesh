//! Main router implementation

use crate::config::{RouterConfig, validate_config};
use crate::handler::RequestHandler;
use crate::server::{HttpServer, GrpcServer};
use crate::{Result, RouterError};

use mesh_net::{ConnectionPool, TlsConfig};
use mesh_state::{StateStore, QueryEngine, ScoringEngine};

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::{info, warn};

/// Main router that coordinates HTTP and gRPC servers
pub struct Router {
    /// Router configuration
    config: RouterConfig,
    
    /// HTTP server
    http_server: HttpServer,
    
    /// gRPC server
    grpc_server: GrpcServer,
    
    /// Request handler
    handler: Arc<RequestHandler>,
    
    /// Router statistics
    stats: Arc<RouterStats>,
    
    /// Running server handles
    server_handles: RwLock<Vec<JoinHandle<Result<()>>>>,
}

/// Router statistics
#[derive(Debug)]
pub struct RouterStats {
    /// Total requests received
    pub requests_total: AtomicU64,
    
    /// Total responses sent
    pub responses_total: AtomicU64,
    
    /// Total errors
    pub errors_total: AtomicU64,
    
    /// HTTP requests
    pub http_requests: AtomicU64,
    
    /// gRPC requests
    pub grpc_requests: AtomicU64,
    
    /// WebSocket connections
    pub websocket_connections: AtomicU64,
    
    /// Active connections
    pub active_connections: AtomicU64,
    
    /// Router start time
    pub start_time: Instant,
}

impl Default for RouterStats {
    fn default() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            responses_total: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
            http_requests: AtomicU64::new(0),
            grpc_requests: AtomicU64::new(0),
            websocket_connections: AtomicU64::new(0),
            active_connections: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }
}

impl Router {
    /// Create a new router with the given configuration
    pub async fn new(config: RouterConfig) -> Result<Self> {
        // Validate configuration
        validate_config(&config)
            .map_err(|e| RouterError::Configuration(e))?;

        info!("Creating router with config: {:?}", config);

        // Create state store and engines
        let state_store = StateStore::new();
        let query_engine = QueryEngine::new(state_store.clone());
        let scoring_engine = ScoringEngine::new();

        // Create connection pool for upstream services
        let tls_config = TlsConfig::insecure(); // TODO: Configure proper TLS
        let connection_pool = ConnectionPool::new(tls_config);

        // Create request handler
        let handler = Arc::new(RequestHandler::new(
            config.clone(),
            state_store,
            query_engine,
            scoring_engine,
            connection_pool,
        ).await?);

        // Create servers
        let stats = Arc::new(RouterStats {
            start_time: Instant::now(),
            ..Default::default()
        });

        let http_server = HttpServer::new(
            config.clone(),
            handler.clone(),
            stats.clone(),
        )?;

        let grpc_server = GrpcServer::new(
            config.clone(),
            handler.clone(),
            stats.clone(),
        )?;

        Ok(Self {
            config,
            http_server,
            grpc_server,
            handler,
            stats,
            server_handles: RwLock::new(Vec::new()),
        })
    }

    /// Start the router servers
    pub async fn start(&self) -> Result<()> {
        info!("Starting router on HTTP:{} gRPC:{}", 
              self.config.http_port, self.config.grpc_port);

        let mut handles = self.server_handles.write().await;

        // Start HTTP server
        let http_handle = {
            let server = self.http_server.clone();
            let bind_addr = format!("{}:{}", self.config.bind_address, self.config.http_port);
            tokio::spawn(async move {
                server.serve(&bind_addr).await
            })
        };
        handles.push(http_handle);

        // Start gRPC server
        let grpc_handle = {
            let server = self.grpc_server.clone();
            let bind_addr = format!("{}:{}", self.config.bind_address, self.config.grpc_port);
            tokio::spawn(async move {
                server.serve(&bind_addr).await
            })
        };
        handles.push(grpc_handle);

        info!("Router started successfully");
        Ok(())
    }

    /// Serve and block until shutdown
    pub async fn serve(&self, _bind_addr: &str) -> Result<()> {
        // Start both servers
        self.start().await?;

        // Wait for shutdown signal
        self.wait_for_shutdown().await;

        // Graceful shutdown
        self.shutdown().await?;

        Ok(())
    }

    /// Wait for shutdown signal
    async fn wait_for_shutdown(&self) {
        let ctrl_c = async {
            tokio::signal::ctrl_c()
                .await
                .expect("failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("failed to install signal handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {
                info!("Received Ctrl+C, shutting down");
            }
            _ = terminate => {
                info!("Received SIGTERM, shutting down");
            }
        }
    }

    /// Graceful shutdown
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down router");

        let mut handles = self.server_handles.write().await;
        
        // Cancel all server tasks
        for handle in handles.iter() {
            handle.abort();
        }

        // Wait for tasks to complete (with timeout)
        let timeout = tokio::time::Duration::from_secs(30);
        for handle in handles.drain(..) {
            if let Err(e) = tokio::time::timeout(timeout, handle).await {
                warn!("Server task did not shutdown gracefully: {:?}", e);
            }
        }

        info!("Router shutdown complete");
        Ok(())
    }

    /// Get router statistics
    pub fn stats(&self) -> &RouterStats {
        &self.stats
    }

    /// Get router configuration
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Check if router is healthy
    pub async fn health_check(&self) -> Result<bool> {
        // Check if servers are running
        let handles = self.server_handles.read().await;
        if handles.is_empty() {
            return Ok(false);
        }

        // Check if any server has failed
        for handle in handles.iter() {
            if handle.is_finished() {
                return Ok(false);
            }
        }

        // Check handler health
        self.handler.health_check().await
    }

    /// Get uptime in seconds
    pub fn uptime(&self) -> u64 {
        self.stats.start_time.elapsed().as_secs()
    }
}

impl RouterStats {
    /// Increment request counter
    pub fn increment_requests(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment response counter
    pub fn increment_responses(&self) {
        self.responses_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment error counter
    pub fn increment_errors(&self) {
        self.errors_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment HTTP request counter
    pub fn increment_http_requests(&self) {
        self.http_requests.fetch_add(1, Ordering::Relaxed);
        self.increment_requests();
    }

    /// Increment gRPC request counter
    pub fn increment_grpc_requests(&self) {
        self.grpc_requests.fetch_add(1, Ordering::Relaxed);
        self.increment_requests();
    }

    /// Increment WebSocket connection counter
    pub fn increment_websocket_connections(&self) {
        self.websocket_connections.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment active connections
    pub fn increment_active_connections(&self) {
        self.active_connections.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement active connections
    pub fn decrement_active_connections(&self) {
        self.active_connections.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get total requests
    pub fn total_requests(&self) -> u64 {
        self.requests_total.load(Ordering::Relaxed)
    }

    /// Get total responses
    pub fn total_responses(&self) -> u64 {
        self.responses_total.load(Ordering::Relaxed)
    }

    /// Get total errors
    pub fn total_errors(&self) -> u64 {
        self.errors_total.load(Ordering::Relaxed)
    }

    /// Get HTTP requests
    pub fn http_requests(&self) -> u64 {
        self.http_requests.load(Ordering::Relaxed)
    }

    /// Get gRPC requests
    pub fn grpc_requests(&self) -> u64 {
        self.grpc_requests.load(Ordering::Relaxed)
    }

    /// Get WebSocket connections
    pub fn websocket_connections(&self) -> u64 {
        self.websocket_connections.load(Ordering::Relaxed)
    }

    /// Get active connections
    pub fn active_connections(&self) -> u64 {
        self.active_connections.load(Ordering::Relaxed)
    }

    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RouterConfigBuilder;

    #[tokio::test]
    async fn test_router_creation() {
        let config = RouterConfigBuilder::new()
            .http_port(0) // Use random port for testing
            .grpc_port(0) // Use random port for testing
            .build();

        // This should fail due to port validation
        assert!(Router::new(config).await.is_err());
    }

    #[tokio::test]
    async fn test_router_stats() {
        let stats = RouterStats::default();
        
        assert_eq!(stats.total_requests(), 0);
        assert_eq!(stats.total_responses(), 0);
        assert_eq!(stats.total_errors(), 0);

        stats.increment_http_requests();
        assert_eq!(stats.total_requests(), 1);
        assert_eq!(stats.http_requests(), 1);

        stats.increment_grpc_requests();
        assert_eq!(stats.total_requests(), 2);
        assert_eq!(stats.grpc_requests(), 1);

        stats.increment_responses();
        assert_eq!(stats.total_responses(), 1);

        stats.increment_errors();
        assert_eq!(stats.total_errors(), 1);
    }

    #[tokio::test]
    async fn test_router_with_valid_config() {
        let config = RouterConfigBuilder::new()
            .http_port(8080)
            .grpc_port(9090)
            .build();

        let router = Router::new(config).await;
        assert!(router.is_ok());

        let router = router.unwrap();
        assert_eq!(router.config().http_port, 8080);
        assert_eq!(router.config().grpc_port, 9090);
        assert_eq!(router.stats().total_requests(), 0);
    }
}
