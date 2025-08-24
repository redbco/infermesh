//! Integration test harness for multi-component testing

use crate::{DevError, Result};
use mesh_core::Config;
use std::collections::HashMap;
use std::process::{Child, Command};
use tempfile::TempDir;
use tokio::time::{sleep, Duration};
use tracing::{debug, info, warn};

/// A test cluster for integration testing
pub struct TestCluster {
    nodes: Vec<TestNode>,
    temp_dir: TempDir,
    config: TestClusterConfig,
}

/// Configuration for a test cluster
#[derive(Debug, Clone)]
pub struct TestClusterConfig {
    /// Number of nodes in the cluster
    pub node_count: usize,
    
    /// Base port for services
    pub base_port: u16,
    
    /// Whether to enable metrics
    pub enable_metrics: bool,
    
    /// Whether to enable tracing
    pub enable_tracing: bool,
    
    /// Additional environment variables
    pub env_vars: HashMap<String, String>,
    
    /// Startup timeout (seconds)
    pub startup_timeout_seconds: u64,
}

impl Default for TestClusterConfig {
    fn default() -> Self {
        Self {
            node_count: 3,
            base_port: 50051,
            enable_metrics: true,
            enable_tracing: true,
            env_vars: HashMap::new(),
            startup_timeout_seconds: 30,
        }
    }
}

/// A single test node in the cluster
pub struct TestNode {
    /// Node ID
    pub id: String,
    
    /// Node configuration
    pub config: Config,
    
    /// Process handle (if running as separate process)
    pub process: Option<Child>,
    
    /// gRPC endpoint
    pub grpc_endpoint: String,
    
    /// Metrics endpoint
    pub metrics_endpoint: String,
    
    /// Working directory
    pub work_dir: std::path::PathBuf,
}

impl TestCluster {
    /// Create a new test cluster
    pub async fn new(config: TestClusterConfig) -> Result<Self> {
        let temp_dir = TempDir::new()
            .map_err(|e| DevError::TestHarness(format!("Failed to create temp dir: {}", e)))?;
        
        info!("Creating test cluster with {} nodes", config.node_count);
        
        let mut nodes = Vec::new();
        for i in 0..config.node_count {
            let node = TestNode::new(i, &config, temp_dir.path()).await?;
            nodes.push(node);
        }
        
        Ok(Self {
            nodes,
            temp_dir,
            config,
        })
    }

    /// Start all nodes in the cluster
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting test cluster");
        
        for node in &mut self.nodes {
            node.start().await?;
        }
        
        // Wait for all nodes to be ready
        let timeout = Duration::from_secs(self.config.startup_timeout_seconds);
        let start_time = std::time::Instant::now();
        
        while start_time.elapsed() < timeout {
            let mut all_ready = true;
            
            for node in &self.nodes {
                if !node.is_ready().await {
                    all_ready = false;
                    break;
                }
            }
            
            if all_ready {
                info!("All nodes are ready");
                return Ok(());
            }
            
            sleep(Duration::from_millis(500)).await;
        }
        
        Err(DevError::TestHarness("Cluster startup timeout".to_string()))
    }

    /// Stop all nodes in the cluster
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping test cluster");
        
        for node in &mut self.nodes {
            node.stop().await?;
        }
        
        Ok(())
    }

    /// Get a node by index
    pub fn get_node(&self, index: usize) -> Option<&TestNode> {
        self.nodes.get(index)
    }

    /// Get all nodes
    pub fn nodes(&self) -> &[TestNode] {
        &self.nodes
    }

    /// Get cluster configuration
    pub fn config(&self) -> &TestClusterConfig {
        &self.config
    }

    /// Wait for cluster to be healthy
    pub async fn wait_for_healthy(&self, timeout: Duration) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        while start_time.elapsed() < timeout {
            let mut all_healthy = true;
            
            for node in &self.nodes {
                if !node.is_healthy().await {
                    all_healthy = false;
                    break;
                }
            }
            
            if all_healthy {
                info!("All nodes are healthy");
                return Ok(());
            }
            
            sleep(Duration::from_millis(1000)).await;
        }
        
        Err(DevError::TestHarness("Health check timeout".to_string()))
    }

    /// Get cluster metrics
    pub async fn get_cluster_metrics(&self) -> Result<HashMap<String, String>> {
        let mut metrics = HashMap::new();
        
        for node in &self.nodes {
            if let Ok(node_metrics) = node.get_metrics().await {
                metrics.insert(node.id.clone(), node_metrics);
            }
        }
        
        Ok(metrics)
    }
}

impl Drop for TestCluster {
    fn drop(&mut self) {
        // Best effort cleanup
        for node in &mut self.nodes {
            if let Err(e) = futures::executor::block_on(node.stop()) {
                warn!("Failed to stop node {}: {}", node.id, e);
            }
        }
    }
}

impl TestNode {
    /// Create a new test node
    pub async fn new(
        index: usize,
        cluster_config: &TestClusterConfig,
        base_dir: &std::path::Path,
    ) -> Result<Self> {
        let id = format!("test-node-{}", index);
        let grpc_port = cluster_config.base_port + index as u16;
        let metrics_port = 9090 + index as u16;
        
        // Create node configuration
        let mut config = crate::create_test_config();
        config.node.id = mesh_core::NodeId::new(id.clone());
        config.network.grpc_port = grpc_port;
        config.network.metrics_port = metrics_port;
        config.gossip.port = 7946 + index as u16;
        
        // Create working directory
        let work_dir = base_dir.join(&id);
        std::fs::create_dir_all(&work_dir)
            .map_err(|e| DevError::TestHarness(format!("Failed to create work dir: {}", e)))?;
        
        // Write config file
        let config_path = work_dir.join("config.yaml");
        let config_yaml = serde_yaml::to_string(&config)
            .map_err(|e| DevError::TestHarness(format!("Failed to serialize config: {}", e)))?;
        std::fs::write(&config_path, config_yaml)
            .map_err(|e| DevError::TestHarness(format!("Failed to write config: {}", e)))?;
        
        let grpc_endpoint = format!("http://127.0.0.1:{}", grpc_port);
        let metrics_endpoint = format!("http://127.0.0.1:{}", metrics_port);
        
        Ok(Self {
            id,
            config,
            process: None,
            grpc_endpoint,
            metrics_endpoint,
            work_dir,
        })
    }

    /// Start the node
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting test node: {}", self.id);
        
        // For now, we'll simulate starting a node
        // In a real implementation, this would start the actual mesh-agent process
        debug!("Node {} would be started with config at {:?}", self.id, self.work_dir);
        
        // Simulate startup time
        sleep(Duration::from_millis(100)).await;
        
        Ok(())
    }

    /// Stop the node
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping test node: {}", self.id);
        
        if let Some(mut process) = self.process.take() {
            if let Err(e) = process.kill() {
                warn!("Failed to kill process for node {}: {}", self.id, e);
            }
            
            // Wait for process to exit
            if let Err(e) = process.wait() {
                warn!("Failed to wait for process exit for node {}: {}", self.id, e);
            }
        }
        
        Ok(())
    }

    /// Check if the node is ready
    pub async fn is_ready(&self) -> bool {
        // For mock implementation, always return true after a short delay
        // In a real implementation, this would check the gRPC health endpoint
        true
    }

    /// Check if the node is healthy
    pub async fn is_healthy(&self) -> bool {
        // For mock implementation, always return true
        // In a real implementation, this would check metrics and health endpoints
        true
    }

    /// Get node metrics
    pub async fn get_metrics(&self) -> Result<String> {
        // For mock implementation, return fake metrics
        // In a real implementation, this would fetch from the metrics endpoint
        Ok(format!(
            "# Mock metrics for node {}\nnode_uptime_seconds 100\nnode_health_status 1\n",
            self.id
        ))
    }

    /// Execute a command on the node
    pub async fn execute_command(&self, command: &str, args: &[&str]) -> Result<String> {
        debug!("Executing command on node {}: {} {:?}", self.id, command, args);
        
        let output = Command::new(command)
            .args(args)
            .current_dir(&self.work_dir)
            .output()
            .map_err(|e| DevError::TestHarness(format!("Command execution failed: {}", e)))?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(DevError::TestHarness(format!("Command failed: {}", stderr)));
        }
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout.to_string())
    }

    /// Get the node's gRPC client
    pub async fn get_grpc_client<T>(&self) -> Result<T>
    where
        T: From<tonic::transport::Channel>,
    {
        let channel = tonic::transport::Channel::from_shared(self.grpc_endpoint.clone())
            .map_err(|e| DevError::TestHarness(format!("Invalid gRPC endpoint: {}", e)))?
            .connect()
            .await
            .map_err(|e| DevError::TestHarness(format!("gRPC connection failed: {}", e)))?;
        
        Ok(T::from(channel))
    }
}

/// Builder for creating test clusters
pub struct TestClusterBuilder {
    config: TestClusterConfig,
}

impl TestClusterBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: TestClusterConfig::default(),
        }
    }

    /// Set the number of nodes
    pub fn with_node_count(mut self, count: usize) -> Self {
        self.config.node_count = count;
        self
    }

    /// Set the base port
    pub fn with_base_port(mut self, port: u16) -> Self {
        self.config.base_port = port;
        self
    }

    /// Enable or disable metrics
    pub fn with_metrics(mut self, enabled: bool) -> Self {
        self.config.enable_metrics = enabled;
        self
    }

    /// Enable or disable tracing
    pub fn with_tracing(mut self, enabled: bool) -> Self {
        self.config.enable_tracing = enabled;
        self
    }

    /// Add an environment variable
    pub fn with_env_var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.env_vars.insert(key.into(), value.into());
        self
    }

    /// Set startup timeout
    pub fn with_startup_timeout(mut self, seconds: u64) -> Self {
        self.config.startup_timeout_seconds = seconds;
        self
    }

    /// Build the test cluster
    pub async fn build(self) -> Result<TestCluster> {
        TestCluster::new(self.config).await
    }
}

impl Default for TestClusterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_test_cluster_creation() {
        let cluster = TestClusterBuilder::new()
            .with_node_count(2)
            .with_base_port(60000)
            .build()
            .await
            .unwrap();

        assert_eq!(cluster.nodes().len(), 2);
        assert_eq!(cluster.config().node_count, 2);
        assert_eq!(cluster.config().base_port, 60000);
    }

    #[tokio::test]
    async fn test_test_node_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = TestClusterConfig::default();
        
        let node = TestNode::new(0, &config, temp_dir.path()).await.unwrap();
        
        assert_eq!(node.id, "test-node-0");
        assert!(node.grpc_endpoint.contains("50051"));
        assert!(node.work_dir.exists());
    }

    #[tokio::test]
    async fn test_cluster_lifecycle() {
        let mut cluster = TestClusterBuilder::new()
            .with_node_count(1)
            .with_startup_timeout(5)
            .build()
            .await
            .unwrap();

        // Test start
        cluster.start().await.unwrap();
        
        // Test health check
        let health_result = cluster.wait_for_healthy(Duration::from_secs(1)).await;
        assert!(health_result.is_ok());
        
        // Test stop
        cluster.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_node_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config = TestClusterConfig::default();
        
        let mut node = TestNode::new(0, &config, temp_dir.path()).await.unwrap();
        
        // Test start/stop
        node.start().await.unwrap();
        assert!(node.is_ready().await);
        assert!(node.is_healthy().await);
        
        // Test metrics
        let metrics = node.get_metrics().await.unwrap();
        assert!(metrics.contains("node_uptime_seconds"));
        
        node.stop().await.unwrap();
    }
}
