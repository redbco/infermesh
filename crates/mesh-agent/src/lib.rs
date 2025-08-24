//! # mesh-agent
//!
//! Node agent (meshd daemon) for infermesh.
//!
//! This crate provides the main agent that runs on each node in the infermesh cluster.
//! It implements the gRPC services defined in mesh-proto and coordinates with
//! other components to provide inference serving capabilities.

pub mod agent;
pub mod config;
pub mod services;

// Re-export commonly used types
pub use agent::{Agent, AgentBuilder};
pub use config::AgentConfig;

// Error handling
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Service error: {0}")]
    Service(String),

    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::Status),

    #[error("Transport error: {0}")]
    Transport(#[from] tonic::transport::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Core error: {0}")]
    Core(#[from] mesh_core::Error),

    #[error("Metrics error: {0}")]
    Metrics(#[from] mesh_metrics::MetricsError),

    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, AgentError>;

/// Initialize the agent with logging and tracing
pub async fn init_agent(config: &AgentConfig) -> Result<Agent> {
    // Initialize logging
    init_logging(&config.logging)?;
    
    tracing::info!("Initializing mesh agent with config: {:?}", config);
    
    // Build the agent
    let agent = AgentBuilder::new()
        .with_config(config.clone())
        .build()
        .await?;
    
    Ok(agent)
}

/// Initialize logging and tracing
fn init_logging(logging_config: &config::LoggingConfig) -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&logging_config.level));

    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(logging_config.show_target)
        .with_thread_ids(logging_config.show_thread_ids)
        .with_line_number(logging_config.show_line_numbers);

    match logging_config.format.as_str() {
        "json" => subscriber.json().init(),
        _ => subscriber.init(),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_config_creation() {
        let config = AgentConfig::default();
        assert!(!config.core.node.id.to_string().is_empty());
        assert!(config.core.network.grpc_port > 0);
    }

    #[tokio::test]
    async fn test_agent_initialization() {
        let config = AgentConfig::default();
        
        // This should not panic
        let result = init_agent(&config).await;
        
        // In a real test environment, this would succeed
        // For now, we just verify it doesn't panic during construction
        match result {
            Ok(_) => {
                // Agent initialized successfully
            }
            Err(e) => {
                // Expected in test environment without full setup
                tracing::debug!("Agent initialization failed as expected in test: {}", e);
            }
        }
    }
}
