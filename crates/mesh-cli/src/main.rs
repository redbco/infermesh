//! mesh-cli - Command-line interface for infermesh control plane

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{debug, info};

mod client;
mod commands;
mod config;
mod output;

use client::MeshClient;
use config::CliConfig;
use output::OutputFormat;

/// Command-line interface for infermesh control plane
#[derive(Debug, Parser)]
#[command(name = "mesh")]
#[command(about = "Command-line interface for infermesh control plane")]
#[command(version)]
pub struct Cli {
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Control plane endpoint
    #[arg(short, long, default_value = "http://127.0.0.1:50051")]
    endpoint: String,

    /// Output format
    #[arg(short, long, value_enum, default_value = "table")]
    output: OutputFormat,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Enable JSON output (overrides --output)
    #[arg(long)]
    json: bool,

    /// Timeout for requests in seconds
    #[arg(long, default_value = "30")]
    timeout: u64,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// List nodes in the mesh
    #[command(name = "list-nodes")]
    ListNodes {
        /// Filter by node role
        #[arg(short, long)]
        role: Option<String>,
        
        /// Filter by node status
        #[arg(short, long)]
        status: Option<String>,
        
        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },

    /// Pin a model to specific nodes
    #[command(name = "pin-model")]
    PinModel {
        /// Model identifier
        model: String,
        
        /// Target nodes (comma-separated)
        #[arg(short, long, value_delimiter = ',')]
        nodes: Vec<String>,
        
        /// Model version
        #[arg(short, long)]
        version: Option<String>,
        
        /// Minimum replicas
        #[arg(long, default_value = "1")]
        min_replicas: u32,
        
        /// Maximum replicas
        #[arg(long)]
        max_replicas: Option<u32>,
        
        /// Priority (low, normal, high, critical)
        #[arg(short, long, default_value = "normal")]
        priority: String,
    },

    /// Unpin a model from nodes
    #[command(name = "unpin-model")]
    UnpinModel {
        /// Model identifier
        model: String,
        
        /// Target nodes (comma-separated, optional - unpins from all if not specified)
        #[arg(short, long, value_delimiter = ',')]
        nodes: Option<Vec<String>>,
    },

    /// List model pinning policies
    #[command(name = "list-pins")]
    ListPins {
        /// Filter by model
        #[arg(short, long)]
        model: Option<String>,
        
        /// Filter by node
        #[arg(short, long)]
        node: Option<String>,
    },

    /// Subscribe to mesh events
    #[command(name = "subscribe-events")]
    SubscribeEvents {
        /// Event types to subscribe to (comma-separated)
        #[arg(short, long, value_delimiter = ',')]
        types: Option<Vec<String>>,
        
        /// Follow events (like tail -f)
        #[arg(short, long)]
        follow: bool,
        
        /// Number of historical events to show
        #[arg(long, default_value = "10")]
        history: u32,
    },

    /// Show node details
    #[command(name = "describe-node")]
    DescribeNode {
        /// Node identifier
        node_id: String,
    },

    /// Show model details
    #[command(name = "describe-model")]
    DescribeModel {
        /// Model identifier
        model: String,
        
        /// Model version
        #[arg(short, long)]
        version: Option<String>,
    },

    /// Get mesh statistics
    #[command(name = "stats")]
    Stats {
        /// Show detailed statistics
        #[arg(short, long)]
        detailed: bool,
        
        /// Refresh interval in seconds (for continuous monitoring)
        #[arg(short, long)]
        refresh: Option<u64>,
    },

    /// Manage resource quotas
    #[command(name = "quota")]
    Quota {
        #[command(subcommand)]
        action: QuotaCommands,
    },

    /// Manage access control policies
    #[command(name = "acl")]
    Acl {
        #[command(subcommand)]
        action: AclCommands,
    },

    /// Health check commands
    #[command(name = "health")]
    Health {
        #[command(subcommand)]
        action: HealthCommands,
    },

    /// Configuration management
    #[command(name = "config")]
    Config {
        #[command(subcommand)]
        action: ConfigCommands,
    },
}

#[derive(Debug, Subcommand)]
pub enum QuotaCommands {
    /// List resource quotas
    List {
        /// Filter by scope
        #[arg(short, long)]
        scope: Option<String>,
    },
    
    /// Set a resource quota
    Set {
        /// Quota scope (global, node:<id>, user:<name>)
        scope: String,
        
        /// Maximum CPU cores
        #[arg(long)]
        max_cpu: Option<f64>,
        
        /// Maximum memory in GB
        #[arg(long)]
        max_memory: Option<f64>,
        
        /// Maximum GPU count
        #[arg(long)]
        max_gpu: Option<u32>,
        
        /// Maximum requests per second
        #[arg(long)]
        max_rps: Option<f64>,
    },
    
    /// Remove a resource quota
    Remove {
        /// Quota scope
        scope: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum AclCommands {
    /// List access control policies
    List {
        /// Filter by subject
        #[arg(short, long)]
        subject: Option<String>,
        
        /// Filter by resource
        #[arg(short, long)]
        resource: Option<String>,
    },
    
    /// Grant access
    Grant {
        /// Subject (user:<name>, service:<name>, role:<name>)
        subject: String,
        
        /// Resource (model:<name>, node:<id>, endpoint:<path>)
        resource: String,
        
        /// Actions (comma-separated: read, write, execute, delete, admin)
        #[arg(short, long, value_delimiter = ',')]
        actions: Vec<String>,
    },
    
    /// Revoke access
    Revoke {
        /// Subject
        subject: String,
        
        /// Resource
        resource: String,
        
        /// Actions to revoke (comma-separated, optional - revokes all if not specified)
        #[arg(short, long, value_delimiter = ',')]
        actions: Option<Vec<String>>,
    },
}

#[derive(Debug, Subcommand)]
pub enum HealthCommands {
    /// Check overall mesh health
    Check,
    
    /// Check specific node health
    Node {
        /// Node identifier
        node_id: String,
    },
    
    /// Check model health
    Model {
        /// Model identifier
        model: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum ConfigCommands {
    /// Show current configuration
    Show,
    
    /// Set configuration value
    Set {
        /// Configuration key
        key: String,
        
        /// Configuration value
        value: String,
    },
    
    /// Get configuration value
    Get {
        /// Configuration key
        key: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(format!("mesh_cli={},mesh_core={}", log_level, log_level))
        .with_target(false)
        .init();

    debug!("Starting mesh CLI with config: {:?}", cli);

    // Load configuration
    let config = CliConfig::load(cli.config.as_deref())?;
    info!("Loaded configuration from {:?}", config.source());

    // Determine output format
    let output_format = if cli.json {
        OutputFormat::Json
    } else {
        cli.output
    };

    // Create client
    let client = MeshClient::new(&cli.endpoint, std::time::Duration::from_secs(cli.timeout)).await?;
    info!("Connected to mesh control plane at {}", cli.endpoint);

    // Execute command
    match cli.command {
        Commands::ListNodes { role, status, detailed } => {
            commands::nodes::list_nodes(&client, role, status, detailed, output_format).await?;
        }
        
        Commands::PinModel { model, nodes, version, min_replicas, max_replicas, priority } => {
            commands::models::pin_model(&client, model, nodes, version, min_replicas, max_replicas, priority, output_format).await?;
        }
        
        Commands::UnpinModel { model, nodes } => {
            commands::models::unpin_model(&client, model, nodes, output_format).await?;
        }
        
        Commands::ListPins { model, node } => {
            commands::models::list_pins(&client, model, node, output_format).await?;
        }
        
        Commands::SubscribeEvents { types, follow, history } => {
            commands::events::subscribe_events(&client, types, follow, history, output_format).await?;
        }
        
        Commands::DescribeNode { node_id } => {
            commands::nodes::describe_node(&client, node_id, output_format).await?;
        }
        
        Commands::DescribeModel { model, version } => {
            commands::models::describe_model(&client, model, version, output_format).await?;
        }
        
        Commands::Stats { detailed, refresh } => {
            commands::stats::show_stats(&client, detailed, refresh, output_format).await?;
        }
        
        Commands::Quota { action } => {
            commands::quota::handle_quota_command(&client, action, output_format).await?;
        }
        
        Commands::Acl { action } => {
            commands::acl::handle_acl_command(&client, action, output_format).await?;
        }
        
        Commands::Health { action } => {
            commands::health::handle_health_command(&client, action, output_format).await?;
        }
        
        Commands::Config { action } => {
            commands::config::handle_config_command(&client, action, output_format).await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn verify_cli() {
        Cli::command().debug_assert()
    }

    #[test]
    fn test_cli_parsing() {
        let cli = Cli::try_parse_from(&["mesh", "list-nodes"]).unwrap();
        assert!(matches!(cli.command, Commands::ListNodes { .. }));
        
        let cli = Cli::try_parse_from(&["mesh", "pin-model", "gpt-7b", "--nodes", "node1,node2"]).unwrap();
        assert!(matches!(cli.command, Commands::PinModel { .. }));
    }

    #[test]
    fn test_output_format() {
        let cli = Cli::try_parse_from(&["mesh", "--json", "list-nodes"]).unwrap();
        assert!(cli.json);
        
        let cli = Cli::try_parse_from(&["mesh", "--output", "yaml", "list-nodes"]).unwrap();
        assert_eq!(cli.output, OutputFormat::Yaml);
    }
}
