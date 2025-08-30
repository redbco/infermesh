//! Core agent implementation

use crate::{config::AgentConfig, services, AgentError, Result};
use mesh_metrics::{MetricsRegistry, MetricsRegistryBuilder, PrometheusExporter};
use std::sync::Arc;
use tokio::signal;
use tokio::sync::RwLock;
use tonic::transport::Server;
use tracing::{error, info, warn};

/// The main mesh agent
pub struct Agent {
    config: AgentConfig,
    metrics_registry: MetricsRegistry,
    services: Arc<RwLock<Vec<ServiceHandle>>>,
    adapter_manager: Option<Arc<services::AdapterManager>>,
    state_plane_service: Option<Arc<services::StatePlaneService>>,
    state_sync_service: Option<Arc<services::StateSyncService>>,
    router_service: Option<services::RouterService>,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

/// Handle to a running service
struct ServiceHandle {
    name: String,
    handle: tokio::task::JoinHandle<Result<()>>,
}

impl Agent {
    /// Create a new agent with the given configuration
    pub(crate) fn new(config: AgentConfig, metrics_registry: MetricsRegistry) -> Self {
        Self {
            config,
            metrics_registry,
            services: Arc::new(RwLock::new(Vec::new())),
            adapter_manager: None,
            state_plane_service: None,
            state_sync_service: None,
            router_service: None,
            shutdown_tx: None,
        }
    }

    /// Start the agent and all its services
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting mesh agent: {}", self.config.agent.name);

        // Validate configuration
        self.config.validate()?;

        // Ensure data directory exists
        self.config.ensure_data_dir()?;

        // Write PID file if configured
        if let Some(pid_file) = self.config.pid_file_path() {
            self.write_pid_file(pid_file)?;
        }

        // Start metrics registry
        self.metrics_registry.start_exporters().await?;

        // Start services first to create state plane
        self.start_services().await?;

        // Initialize and start adapter manager (after services)
        self.start_adapter_manager().await?;

        // Initialize and start state synchronization (after adapters)
        self.start_state_sync().await?;

        // Initialize and start router service (after state sync)
        self.start_router_service().await?;

        info!("Mesh agent started successfully");
        Ok(())
    }

    /// Stop the agent and all its services
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping mesh agent");

        // Stop router service
        if let Some(router_service) = &mut self.router_service {
            router_service.stop().await?;
        }

        // Stop state sync service
        if let Some(state_sync) = &self.state_sync_service {
            state_sync.stop().await?;
        }

        // Stop adapter manager
        if let Some(adapter_manager) = &self.adapter_manager {
            adapter_manager.shutdown().await?;
        }

        // Send shutdown signal
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }

        // Stop all services
        self.stop_services().await?;

        // Stop metrics registry
        self.metrics_registry.stop_exporters().await;

        // Clean up PID file
        if let Some(pid_file) = self.config.pid_file_path() {
            if pid_file.exists() {
                if let Err(e) = std::fs::remove_file(pid_file) {
                    warn!("Failed to remove PID file: {}", e);
                }
            }
        }

        info!("Mesh agent stopped");
        Ok(())
    }

    /// Run the agent until shutdown signal is received
    pub async fn run(&mut self) -> Result<()> {
        // Start the agent
        self.start().await?;

        // Set up shutdown signal handling
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        self.shutdown_tx = Some(shutdown_tx);

        // Wait for shutdown signal
        tokio::select! {
            _ = shutdown_rx => {
                info!("Received shutdown signal");
            }
            _ = signal::ctrl_c() => {
                info!("Received Ctrl+C signal");
            }
            _ = self.wait_for_termination() => {
                info!("Received termination signal");
            }
        }

        // Stop the agent
        self.stop().await?;

        Ok(())
    }

    /// Get the agent configuration
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Get the metrics registry
    pub fn metrics_registry(&self) -> &MetricsRegistry {
        &self.metrics_registry
    }

    /// Start all configured services
    async fn start_services(&mut self) -> Result<()> {
        let mut service_handles = Vec::new();

        // Start Control Plane service
        if self.config.services.control_plane.enabled {
            let handle = self.start_control_plane_service().await?;
            service_handles.push(ServiceHandle {
                name: "control-plane".to_string(),
                handle,
            });
        }

        // Start State Plane service
        if self.config.services.state_plane.enabled {
            let handle = self.start_state_plane_service().await?;
            service_handles.push(ServiceHandle {
                name: "state-plane".to_string(),
                handle,
            });
        }

        // Start Scoring service
        if self.config.services.scoring.enabled {
            let handle = self.start_scoring_service().await?;
            service_handles.push(ServiceHandle {
                name: "scoring".to_string(),
                handle,
            });
        }

        // Store all service handles
        {
            let mut services = self.services.write().await;
            services.extend(service_handles);
            info!("Started {} services", services.len());
        }

        Ok(())
    }

    /// Stop all running services
    async fn stop_services(&self) -> Result<()> {
        let mut services = self.services.write().await;
        
        info!("Stopping {} services", services.len());

        for service in services.drain(..) {
            info!("Stopping service: {}", service.name);
            service.handle.abort();
            
            match service.handle.await {
                Ok(Ok(())) => {
                    info!("Service {} stopped successfully", service.name);
                }
                Ok(Err(e)) => {
                    error!("Service {} stopped with error: {}", service.name, e);
                }
                Err(e) if e.is_cancelled() => {
                    info!("Service {} was cancelled", service.name);
                }
                Err(e) => {
                    error!("Failed to stop service {}: {}", service.name, e);
                }
            }
        }

        Ok(())
    }

    /// Start the Control Plane service
    async fn start_control_plane_service(&self) -> Result<tokio::task::JoinHandle<Result<()>>> {
        let bind_addr = self.config.services.control_plane.bind_addr;
        let service = services::ControlPlaneService::new(
            self.config.clone(),
            self.metrics_registry.clone(),
        );

        let handle = tokio::spawn(async move {
            info!("Starting Control Plane service on {}", bind_addr);
            
            Server::builder()
                .add_service(mesh_proto::control::v1::control_plane_server::ControlPlaneServer::new(service))
                .serve(bind_addr)
                .await
                .map_err(|e| AgentError::Transport(e))?;
            
            Ok(())
        });

        Ok(handle)
    }

    /// Start the State Plane service
    async fn start_state_plane_service(&mut self) -> Result<tokio::task::JoinHandle<Result<()>>> {
        let bind_addr = self.config.services.state_plane.bind_addr;
        let service = Arc::new(services::StatePlaneService::new(
            self.config.clone(),
            self.metrics_registry.clone(),
        ));

        // Store reference for adapter manager
        self.state_plane_service = Some(service.clone());

        let handle = tokio::spawn(async move {
            info!("Starting State Plane service on {}", bind_addr);
            
            Server::builder()
                .add_service(mesh_proto::state::v1::state_plane_server::StatePlaneServer::new(service.as_ref().clone()))
                .serve(bind_addr)
                .await
                .map_err(|e| AgentError::Transport(e))?;
            
            Ok(())
        });

        Ok(handle)
    }

    /// Start the Scoring service
    async fn start_scoring_service(&self) -> Result<tokio::task::JoinHandle<Result<()>>> {
        let bind_addr = self.config.services.scoring.bind_addr;
        let service = services::ScoringService::new(
            self.config.clone(),
            self.metrics_registry.clone(),
        );

        let handle = tokio::spawn(async move {
            info!("Starting Scoring service on {}", bind_addr);
            
            Server::builder()
                .add_service(mesh_proto::scoring::v1::scoring_server::ScoringServer::new(service))
                .serve(bind_addr)
                .await
                .map_err(|e| AgentError::Transport(e))?;
            
            Ok(())
        });

        Ok(handle)
    }

    /// Start the adapter manager
    async fn start_adapter_manager(&mut self) -> Result<()> {
        info!("Starting adapter manager");
        
        let mut adapter_manager = services::AdapterManager::new(
            self.config.services.adapters.clone(),
            self.metrics_registry.clone(),
        );

        // Connect with state plane if available
        if let Some(ref state_plane) = self.state_plane_service {
            adapter_manager.set_state_plane(state_plane.clone());
        }

        // Initialize adapters
        adapter_manager.initialize().await?;

        // Start adapter monitoring and telemetry
        adapter_manager.start().await?;

        self.adapter_manager = Some(Arc::new(adapter_manager));
        info!("Adapter manager started successfully");
        Ok(())
    }

    /// Get the adapter manager
    pub fn adapter_manager(&self) -> Option<&Arc<services::AdapterManager>> {
        self.adapter_manager.as_ref()
    }

    /// Start the state synchronization service
    async fn start_state_sync(&mut self) -> Result<()> {
        info!("Starting state synchronization service");
        
        // Create state store and synchronizer
        let state_store = Arc::new(mesh_state::StateStore::new());
        let state_synchronizer = Arc::new(mesh_state::StateSynchronizer::new((*state_store).clone()));

        // Create state sync service
        let node_id = std::env::var("NODE_ID").unwrap_or_else(|_| {
            format!("node-{}", uuid::Uuid::new_v4().to_string()[..8].to_string())
        });
        let sync_interval = std::time::Duration::from_secs(30);
        
        let mut state_sync = services::StateSyncService::new(
            state_store,
            state_synchronizer,
            node_id.into(), // Convert String to NodeId
            sync_interval,
        );

        // Connect with adapter manager
        if let Some(ref adapter_manager) = self.adapter_manager {
            state_sync.set_adapter_manager(adapter_manager.clone());
        }

        // Connect with state plane
        if let Some(ref state_plane) = self.state_plane_service {
            state_sync.set_state_plane(state_plane.clone());
        }

        // Start the service
        state_sync.start().await?;

        self.state_sync_service = Some(Arc::new(state_sync));
        info!("State synchronization service started successfully");
        Ok(())
    }

    /// Start the router service for inter-node request routing
    async fn start_router_service(&mut self) -> Result<()> {
        info!("Starting router service");
        
        // Create router service
        let mut router_service = services::RouterService::new(&self.config, &self.metrics_registry).await?;
        
        // Start the router service
        router_service.start().await?;
        
        self.router_service = Some(router_service);
        info!("Router service started successfully");
        Ok(())
    }

    /// Wait for termination signals (SIGTERM, SIGINT)
    #[cfg(unix)]
    async fn wait_for_termination(&self) {
        use tokio::signal::unix::{signal, SignalKind};
        
        let mut sigterm = signal(SignalKind::terminate()).unwrap();
        let mut sigint = signal(SignalKind::interrupt()).unwrap();
        
        tokio::select! {
            _ = sigterm.recv() => {
                info!("Received SIGTERM");
            }
            _ = sigint.recv() => {
                info!("Received SIGINT");
            }
        }
    }

    /// Wait for termination signals (Windows)
    #[cfg(not(unix))]
    async fn wait_for_termination(&self) {
        // On Windows, we only handle Ctrl+C
        let _ = signal::ctrl_c().await;
    }

    /// Write the PID file
    fn write_pid_file(&self, pid_file: &std::path::Path) -> Result<()> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = pid_file.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| AgentError::Config(format!("Failed to create PID file directory: {}", e)))?;
        }

        let pid = std::process::id();
        std::fs::write(pid_file, pid.to_string())
            .map_err(|e| AgentError::Config(format!("Failed to write PID file: {}", e)))?;

        info!("Wrote PID {} to {}", pid, pid_file.display());
        Ok(())
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        // Best effort cleanup
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

/// Builder for creating agents
pub struct AgentBuilder {
    config: Option<AgentConfig>,
}

impl AgentBuilder {
    /// Create a new agent builder
    pub fn new() -> Self {
        Self { config: None }
    }

    /// Set the agent configuration
    pub fn with_config(mut self, config: AgentConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Build the agent
    pub async fn build(self) -> Result<Agent> {
        let config = self.config.unwrap_or_default();

        // Create metrics registry
        let mut metrics_builder = MetricsRegistryBuilder::new();

        // Add global labels
        for (key, value) in &config.services.metrics.prometheus.global_labels {
            metrics_builder = metrics_builder.with_global_label(key.clone(), value.clone());
        }

        // Add Prometheus exporter if enabled
        if config.services.metrics.prometheus.enabled {
            let prometheus_exporter = PrometheusExporter::new(config.services.metrics.bind_addr)?;
            metrics_builder = metrics_builder.with_prometheus_exporter(prometheus_exporter);
        }

        // Add OpenTelemetry exporter if enabled
        //#[cfg(feature = "opentelemetry")]
        //if config.services.metrics.opentelemetry.enabled {
        //    if let Some(ref endpoint) = config.services.metrics.opentelemetry.otlp_endpoint {
        //        let otel_exporter = mesh_metrics::opentelemetry_metrics::OpenTelemetryExporter::new(endpoint)?;
        //        metrics_builder = metrics_builder.with_opentelemetry_exporter(otel_exporter);
        //    }
        //}

        let metrics_registry = metrics_builder.build()?;

        Ok(Agent::new(config, metrics_registry))
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_agent_builder() {
        let config = AgentConfig::default();
        let agent = AgentBuilder::new()
            .with_config(config)
            .build()
            .await
            .unwrap();

        assert_eq!(agent.config().agent.name, "mesh-agent");
    }

    #[tokio::test]
    async fn test_agent_lifecycle() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = AgentConfig::default();
        config.agent.data_dir = temp_dir.path().to_path_buf();
        config.agent.pid_file = Some(temp_dir.path().join("test.pid"));

        // Disable services to avoid port conflicts in tests
        config.services.control_plane.enabled = false;
        config.services.state_plane.enabled = false;
        config.services.scoring.enabled = false;
        config.services.metrics.enabled = false;

        let mut agent = AgentBuilder::new()
            .with_config(config)
            .build()
            .await
            .unwrap();

        // Test start
        agent.start().await.unwrap();

        // Test that data directory was created
        assert!(agent.config().agent.data_dir.exists());

        // Test that PID file was created
        if let Some(pid_file) = agent.config().pid_file_path() {
            assert!(pid_file.exists());
        }

        // Test stop
        agent.stop().await.unwrap();

        // Test that PID file was removed
        if let Some(pid_file) = agent.config().pid_file_path() {
            assert!(!pid_file.exists());
        }
    }

    #[test]
    fn test_agent_config_access() {
        let config = AgentConfig::default();
        let metrics_registry = MetricsRegistryBuilder::new().build().unwrap();
        let agent = Agent::new(config.clone(), metrics_registry);

        assert_eq!(agent.config().agent.name, config.agent.name);
    }
}
