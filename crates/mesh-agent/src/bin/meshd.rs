//! Main binary for the mesh agent daemon (meshd)

use clap::{Parser, Subcommand};
use mesh_agent::{init_agent, AgentConfig, Result};
use std::path::PathBuf;
use tracing::{error, info};

#[derive(Parser)]
#[command(name = "meshd")]
#[command(about = "Mesh agent daemon for infermesh")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Cli {
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Log level
    #[arg(long, value_name = "LEVEL", default_value = "info")]
    log_level: String,

    /// Run in daemon mode
    #[arg(short, long)]
    daemon: bool,

    /// Data directory
    #[arg(long, value_name = "DIR")]
    data_dir: Option<PathBuf>,

    /// PID file path
    #[arg(long, value_name = "FILE")]
    pid_file: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the agent
    Start {
        /// Override configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,
    },
    /// Stop the agent
    Stop {
        /// PID file path
        #[arg(long)]
        pid_file: Option<PathBuf>,
    },
    /// Check agent status
    Status {
        /// PID file path
        #[arg(long)]
        pid_file: Option<PathBuf>,
    },
    /// Generate default configuration
    Config {
        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Validate configuration
    Validate {
        /// Configuration file to validate
        #[arg(short, long)]
        config: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Start { ref config }) => {
            let config_path = config.clone().or(cli.config.clone());
            start_agent(config_path, &cli).await
        }
        Some(Commands::Stop { pid_file }) => {
            let pid_file = pid_file.or(cli.pid_file);
            stop_agent(pid_file).await
        }
        Some(Commands::Status { pid_file }) => {
            let pid_file = pid_file.or(cli.pid_file);
            check_status(pid_file).await
        }
        Some(Commands::Config { output }) => {
            generate_config(output).await
        }
        Some(Commands::Validate { config }) => {
            validate_config(config).await
        }
        None => {
            // Default behavior: start the agent
            let config_path = cli.config.clone();
            start_agent(config_path, &cli).await
        }
    }
}

async fn start_agent(config_path: Option<PathBuf>, cli: &Cli) -> Result<()> {
    // Load configuration
    let mut config = if let Some(config_path) = config_path {
        info!("Loading configuration from: {}", config_path.display());
        AgentConfig::from_file(config_path)?
    } else {
        info!("Using default configuration");
        AgentConfig::default()
    };

    // Apply CLI overrides
    if !cli.log_level.is_empty() {
        config.logging.level = cli.log_level.clone();
    }

    if cli.daemon {
        config.agent.daemon = true;
    }

    if let Some(ref data_dir) = cli.data_dir {
        config.agent.data_dir = data_dir.clone();
    }

    if let Some(ref pid_file) = cli.pid_file {
        config.agent.pid_file = Some(pid_file.clone());
    }

    // Initialize and run the agent
    let mut agent = init_agent(&config).await?;
    
    info!("Starting mesh agent: {}", config.agent.name);
    
    if let Err(e) = agent.run().await {
        error!("Agent failed: {}", e);
        std::process::exit(1);
    }

    Ok(())
}

async fn stop_agent(pid_file: Option<PathBuf>) -> Result<()> {
    let pid_file = pid_file.unwrap_or_else(|| PathBuf::from("/var/run/infermesh/meshd.pid"));
    
    if !pid_file.exists() {
        println!("PID file not found: {}", pid_file.display());
        println!("Agent may not be running");
        return Ok(());
    }

    let pid_str = std::fs::read_to_string(&pid_file)
        .map_err(|e| mesh_agent::AgentError::Io(e))?;
    
    let pid: u32 = pid_str.trim().parse()
        .map_err(|e| mesh_agent::AgentError::Config(format!("Invalid PID in file: {}", e)))?;

    println!("Stopping agent with PID: {}", pid);

    // Send SIGTERM to the process
    #[cfg(unix)]
    {
        use nix::sys::signal::{self};
        use nix::unistd::Pid;
        
        match signal::kill(Pid::from_raw(pid as i32), nix::sys::signal::Signal::SIGTERM) {
            Ok(()) => {
                println!("Sent SIGTERM to process {}", pid);
                
                // Wait a bit for graceful shutdown
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                
                // Check if process is still running
                match signal::kill(Pid::from_raw(pid as i32), None) {
                    Ok(()) => {
                        println!("Process {} is still running, sending SIGKILL", pid);
                        let _ = signal::kill(Pid::from_raw(pid as i32), nix::sys::signal::Signal::SIGKILL);
                    }
                    Err(_) => {
                        println!("Process {} stopped successfully", pid);
                    }
                }
            }
            Err(e) => {
                println!("Failed to send signal to process {}: {}", pid, e);
            }
        }
    }

    #[cfg(not(unix))]
    {
        println!("Process termination not supported on this platform");
        println!("Please stop the agent manually");
    }

    Ok(())
}

async fn check_status(pid_file: Option<PathBuf>) -> Result<()> {
    let pid_file = pid_file.unwrap_or_else(|| PathBuf::from("/var/run/infermesh/meshd.pid"));
    
    if !pid_file.exists() {
        println!("Status: NOT RUNNING");
        println!("PID file not found: {}", pid_file.display());
        return Ok(());
    }

    let pid_str = std::fs::read_to_string(&pid_file)
        .map_err(|e| mesh_agent::AgentError::Io(e))?;
    
    let pid: u32 = pid_str.trim().parse()
        .map_err(|e| mesh_agent::AgentError::Config(format!("Invalid PID in file: {}", e)))?;

    // Check if process is running
    #[cfg(unix)]
    {
        use nix::sys::signal::{self};
        use nix::unistd::Pid;
        
        match signal::kill(Pid::from_raw(pid as i32), None) {
            Ok(()) => {
                println!("Status: RUNNING");
                println!("PID: {}", pid);
                println!("PID file: {}", pid_file.display());
            }
            Err(_) => {
                println!("Status: NOT RUNNING");
                println!("Stale PID file found: {}", pid_file.display());
                println!("Process {} is not running", pid);
            }
        }
    }

    #[cfg(not(unix))]
    {
        println!("Status: UNKNOWN");
        println!("Process status checking not supported on this platform");
        println!("PID: {}", pid);
        println!("PID file: {}", pid_file.display());
    }

    Ok(())
}

async fn generate_config(output: Option<PathBuf>) -> Result<()> {
    let config = AgentConfig::default();
    
    if let Some(output_path) = output {
        config.to_file(&output_path)?;
        println!("Generated configuration file: {}", output_path.display());
    } else {
        let yaml = serde_yaml::to_string(&config)
            .map_err(|e| mesh_agent::AgentError::Config(format!("Failed to serialize config: {}", e)))?;
        println!("{}", yaml);
    }

    Ok(())
}

async fn validate_config(config_path: PathBuf) -> Result<()> {
    println!("Validating configuration: {}", config_path.display());
    
    let config = AgentConfig::from_file(&config_path)?;
    config.validate()?;
    
    println!("Configuration is valid");
    println!("Agent name: {}", config.agent.name);
    println!("Data directory: {}", config.agent.data_dir.display());
    println!("Services enabled:");
    
    if config.services.control_plane.enabled {
        println!("  - Control Plane: {}", config.services.control_plane.bind_addr);
    }
    if config.services.state_plane.enabled {
        println!("  - State Plane: {}", config.services.state_plane.bind_addr);
    }
    if config.services.scoring.enabled {
        println!("  - Scoring: {}", config.services.scoring.bind_addr);
    }
    if config.services.metrics.enabled {
        println!("  - Metrics: {}", config.services.metrics.bind_addr);
    }

    Ok(())
}
