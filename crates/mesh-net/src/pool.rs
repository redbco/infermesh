//! Connection pooling for efficient connection reuse

use crate::{Connection, ConnectionManager, NetworkError, Result, TlsConfig};
use dashmap::DashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::time::sleep;
use tracing::debug;

/// Connection pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum connections per host
    pub max_connections_per_host: usize,
    
    /// Maximum idle connections
    pub max_idle_connections: usize,
    
    /// Connection idle timeout
    pub idle_timeout: Duration,
    
    /// Connection timeout
    pub connect_timeout: Duration,
    
    /// Pool cleanup interval
    pub cleanup_interval: Duration,
}

/// A pooled connection with metadata
#[derive(Debug)]
struct PooledConnection {
    connection: Connection,
    #[allow(dead_code)]
    created_at: Instant,
    last_used: Instant,
    use_count: usize,
}

/// Connection pool for managing and reusing connections
#[derive(Debug)]
pub struct ConnectionPool {
    /// Connection manager
    manager: ConnectionManager,
    
    /// Pool configuration
    config: PoolConfig,
    
    /// Active connections per host
    connections: Arc<DashMap<String, Vec<PooledConnection>>>,
    
    /// Connection semaphores per host
    semaphores: DashMap<String, Arc<Semaphore>>,
    
    /// Active connection count
    active_connections: Arc<AtomicUsize>,
    
    /// Pool statistics
    stats: PoolStats,
}

/// Pool statistics
#[derive(Debug)]
pub struct PoolStats {
    pub connections_created: Arc<AtomicUsize>,
    pub connections_reused: Arc<AtomicUsize>,
    pub connections_closed: Arc<AtomicUsize>,
    pub pool_hits: Arc<AtomicUsize>,
    pub pool_misses: Arc<AtomicUsize>,
}

impl Default for PoolStats {
    fn default() -> Self {
        Self {
            connections_created: Arc::new(AtomicUsize::new(0)),
            connections_reused: Arc::new(AtomicUsize::new(0)),
            connections_closed: Arc::new(AtomicUsize::new(0)),
            pool_hits: Arc::new(AtomicUsize::new(0)),
            pool_misses: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections_per_host: 10,
            max_idle_connections: 100,
            idle_timeout: Duration::from_secs(300),
            connect_timeout: Duration::from_secs(10),
            cleanup_interval: Duration::from_secs(60),
        }
    }
}

impl ConnectionPool {
    /// Create a new connection pool
    pub fn new(tls_config: TlsConfig) -> Self {
        Self::with_config(tls_config, PoolConfig::default())
    }
    
    /// Create a new connection pool with custom configuration
    pub fn with_config(tls_config: TlsConfig, config: PoolConfig) -> Self {
        let manager = ConnectionManager::new(tls_config, config.connect_timeout);
        
        let pool = Self {
            manager,
            config,
            connections: Arc::new(DashMap::new()),
            semaphores: DashMap::new(),
            active_connections: Arc::new(AtomicUsize::new(0)),
            stats: PoolStats::default(),
        };
        
        // Start cleanup task
        pool.start_cleanup_task();
        
        pool
    }
    
    /// Get a connection to the specified address
    pub async fn get_connection(&self, addr: SocketAddr) -> Result<Connection> {
        let host_key = format!("{}:{}", addr.ip(), addr.port());
        
        // Try to get a connection from the pool first
        if let Some(connection) = self.get_pooled_connection(&host_key).await {
            self.stats.pool_hits.fetch_add(1, Ordering::Relaxed);
            self.stats.connections_reused.fetch_add(1, Ordering::Relaxed);
            debug!("Reusing pooled connection to {}", addr);
            return Ok(connection);
        }
        
        self.stats.pool_misses.fetch_add(1, Ordering::Relaxed);
        
        // Get or create semaphore for this host
        let semaphore = self.semaphores
            .entry(host_key.clone())
            .or_insert_with(|| Arc::new(Semaphore::new(self.config.max_connections_per_host)))
            .clone();
        
        // Acquire permit to create new connection
        let _permit = semaphore.acquire().await
            .map_err(|_| NetworkError::Pool("Failed to acquire connection permit".to_string()))?;
        
        // Create new connection
        debug!("Creating new connection to {}", addr);
        let connection = self.manager.connect(addr).await?;
        
        self.active_connections.fetch_add(1, Ordering::Relaxed);
        self.stats.connections_created.fetch_add(1, Ordering::Relaxed);
        
        Ok(connection)
    }
    
    /// Get a connection with hostname for TLS SNI
    pub async fn get_connection_with_hostname(&self, addr: SocketAddr, hostname: &str) -> Result<Connection> {
        let host_key = format!("{}:{}", addr.ip(), addr.port());
        
        // Try to get a connection from the pool first
        if let Some(connection) = self.get_pooled_connection(&host_key).await {
            self.stats.pool_hits.fetch_add(1, Ordering::Relaxed);
            self.stats.connections_reused.fetch_add(1, Ordering::Relaxed);
            debug!("Reusing pooled connection to {} with hostname {}", addr, hostname);
            return Ok(connection);
        }
        
        self.stats.pool_misses.fetch_add(1, Ordering::Relaxed);
        
        // Get or create semaphore for this host
        let semaphore = self.semaphores
            .entry(host_key.clone())
            .or_insert_with(|| Arc::new(Semaphore::new(self.config.max_connections_per_host)))
            .clone();
        
        // Acquire permit to create new connection
        let _permit = semaphore.acquire().await
            .map_err(|_| NetworkError::Pool("Failed to acquire connection permit".to_string()))?;
        
        // Create new connection
        debug!("Creating new connection to {} with hostname {}", addr, hostname);
        let connection = self.manager.connect_with_hostname(addr, hostname).await?;
        
        self.active_connections.fetch_add(1, Ordering::Relaxed);
        self.stats.connections_created.fetch_add(1, Ordering::Relaxed);
        
        Ok(connection)
    }
    
    /// Return a connection to the pool
    pub async fn return_connection(&self, addr: SocketAddr, connection: Connection) {
        let host_key = format!("{}:{}", addr.ip(), addr.port());
        
        // Check if we should pool this connection
        if self.should_pool_connection(&host_key) {
            let pooled = PooledConnection {
                connection,
                created_at: Instant::now(),
                last_used: Instant::now(),
                use_count: 1,
            };
            
            self.connections.entry(host_key).or_insert_with(Vec::new).push(pooled);
            debug!("Returned connection to pool for {}", addr);
        } else {
            // Don't pool the connection, just drop it
            self.active_connections.fetch_sub(1, Ordering::Relaxed);
            self.stats.connections_closed.fetch_add(1, Ordering::Relaxed);
            debug!("Closed connection to {} (pool full)", addr);
        }
    }
    
    /// Get the number of active connections
    pub fn active_connections(&self) -> usize {
        self.active_connections.load(Ordering::Relaxed)
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }
    
    /// Clear all pooled connections
    pub async fn clear(&self) {
        let mut total_closed = 0;
        
        for mut entry in self.connections.iter_mut() {
            let connections = entry.value_mut();
            total_closed += connections.len();
            connections.clear();
        }
        
        self.connections.clear();
        self.active_connections.store(0, Ordering::Relaxed);
        self.stats.connections_closed.fetch_add(total_closed, Ordering::Relaxed);
        
        debug!("Cleared {} connections from pool", total_closed);
    }
    
    /// Get a pooled connection if available
    async fn get_pooled_connection(&self, host_key: &str) -> Option<Connection> {
        if let Some(mut entry) = self.connections.get_mut(host_key) {
            let connections = entry.value_mut();
            
            // Find a usable connection
            while let Some(mut pooled) = connections.pop() {
                // Check if connection is still valid (simplified check)
                if pooled.last_used.elapsed() < self.config.idle_timeout {
                    pooled.last_used = Instant::now();
                    pooled.use_count += 1;
                    return Some(pooled.connection);
                } else {
                    // Connection is too old, close it
                    self.active_connections.fetch_sub(1, Ordering::Relaxed);
                    self.stats.connections_closed.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        
        None
    }
    
    /// Check if we should pool a connection
    fn should_pool_connection(&self, host_key: &str) -> bool {
        let current_idle = self.connections.get(host_key)
            .map(|entry| entry.len())
            .unwrap_or(0);
        
        current_idle < self.config.max_idle_connections
    }
    
    /// Start the cleanup task
    fn start_cleanup_task(&self) {
        let connections = Arc::clone(&self.connections);
        let config = self.config.clone();
        let active_connections = Arc::clone(&self.active_connections);
        // Clone the stats to avoid raw pointer issues
        let stats_connections_closed = Arc::clone(&self.stats.connections_closed);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.cleanup_interval);
            
            loop {
                interval.tick().await;
                
                let mut total_cleaned = 0;
                let now = Instant::now();
                
                // Clean up expired connections
                for mut entry in connections.iter_mut() {
                    let connections_list = entry.value_mut();
                    let initial_len = connections_list.len();
                    
                    connections_list.retain(|conn| {
                        now.duration_since(conn.last_used) < config.idle_timeout
                    });
                    
                    let cleaned = initial_len - connections_list.len();
                    total_cleaned += cleaned;
                }
                
                if total_cleaned > 0 {
                    active_connections.fetch_sub(total_cleaned, Ordering::Relaxed);
                    stats_connections_closed.fetch_add(total_cleaned, Ordering::Relaxed);
                    debug!("Cleaned up {} expired connections", total_cleaned);
                }
                
                // Sleep a bit to avoid busy waiting
                sleep(Duration::from_millis(100)).await;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_config_default() {
        let config = PoolConfig::default();
        assert!(config.max_connections_per_host > 0);
        assert!(config.max_idle_connections > 0);
        assert!(!config.idle_timeout.is_zero());
        assert!(!config.connect_timeout.is_zero());
    }

    #[tokio::test]
    async fn test_connection_pool_creation() {
        let tls_config = TlsConfig::insecure();
        let pool = ConnectionPool::new(tls_config);
        assert_eq!(pool.active_connections(), 0);
    }

    #[tokio::test]
    async fn test_pool_stats() {
        let tls_config = TlsConfig::insecure();
        let pool = ConnectionPool::new(tls_config);
        let stats = pool.stats();
        
        assert_eq!(stats.connections_created.load(Ordering::Relaxed), 0);
        assert_eq!(stats.connections_reused.load(Ordering::Relaxed), 0);
        assert_eq!(stats.pool_hits.load(Ordering::Relaxed), 0);
        assert_eq!(stats.pool_misses.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_pool_clear() {
        let tls_config = TlsConfig::insecure();
        let pool = ConnectionPool::new(tls_config);
        
        // Clear empty pool
        pool.clear().await;
        assert_eq!(pool.active_connections(), 0);
    }
}
