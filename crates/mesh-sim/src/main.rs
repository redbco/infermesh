use clap::{Parser, Subcommand};
use std::path::PathBuf;
use anyhow::Result;
use tracing::{info, error};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use mesh_sim::{SimulationConfig, SimulationRunner, workload, world, MigProfile};

/// Discrete-event simulator for InferMesh
#[derive(Parser)]
#[command(name = "mesh-sim")]
#[command(about = "A discrete-event simulator for InferMesh")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
    
    /// Log level (error, warn, info, debug, trace)
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a simulation from a configuration file
    Run {
        /// Path to the configuration YAML file
        #[arg(short, long)]
        config: PathBuf,
        
        /// Output directory for results
        #[arg(short, long, default_value = "results")]
        output: PathBuf,
        
        /// Specific strategy to run (if not specified, runs all configured strategies)
        #[arg(short, long)]
        strategy: Option<String>,
        
        /// Export format (json, csv, both)
        #[arg(long, default_value = "both")]
        format: String,
    },
    
    /// Generate example configuration files
    Generate {
        /// Type of example to generate (small, medium, large)
        #[arg(short, long, default_value = "small")]
        example_type: String,
        
        /// Output file path
        #[arg(short, long, default_value = "example.yaml")]
        output: PathBuf,
    },
    
    /// Validate a configuration file
    Validate {
        /// Path to the configuration YAML file
        #[arg(short, long)]
        config: PathBuf,
    },
    
    /// Compare results from multiple simulation runs
    Compare {
        /// Directory containing simulation results
        #[arg(short, long)]
        results_dir: PathBuf,
        
        /// Output file for comparison
        #[arg(short, long, default_value = "comparison.csv")]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize tracing
    init_tracing(&cli.log_level, cli.verbose)?;
    
    match cli.command {
        Commands::Run { config, output, strategy, format } => {
            run_simulation(config, output, strategy, format)
        }
        Commands::Generate { example_type, output } => {
            generate_example(example_type, output)
        }
        Commands::Validate { config } => {
            validate_config(config)
        }
        Commands::Compare { results_dir, output } => {
            compare_results(results_dir, output)
        }
    }
}

fn init_tracing(log_level: &str, verbose: bool) -> Result<()> {
    let level = if verbose {
        tracing::Level::DEBUG
    } else {
        match log_level.to_lowercase().as_str() {
            "error" => tracing::Level::ERROR,
            "warn" => tracing::Level::WARN,
            "info" => tracing::Level::INFO,
            "debug" => tracing::Level::DEBUG,
            "trace" => tracing::Level::TRACE,
            _ => tracing::Level::INFO,
        }
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("mesh_sim={}", level).into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    Ok(())
}

fn run_simulation(
    config_path: PathBuf,
    output_dir: PathBuf,
    strategy: Option<String>,
    format: String,
) -> Result<()> {
    info!("Loading configuration from {:?}", config_path);
    let config = SimulationConfig::from_yaml_file(config_path.to_str().unwrap())?;
    
    info!("Validating configuration");
    config.validate()?;
    
    info!("Creating simulation runner");
    let mut runner = SimulationRunner::new(config.clone())?;
    
    // Create output directory
    std::fs::create_dir_all(&output_dir)?;
    
    // Save configuration to output directory
    let config_output = output_dir.join("config.yaml");
    config.to_yaml_file(config_output.to_str().unwrap())?;
    
    let results = if let Some(strategy_name) = strategy {
        info!("Running simulation with strategy: {}", strategy_name);
        let summary = runner.run_single_strategy(&strategy_name)?;
        let mut results = std::collections::HashMap::new();
        results.insert(strategy_name, summary);
        results
    } else {
        info!("Running simulation with all configured strategies");
        runner.run_all_strategies()?
    };
    
    // Export results
    for (strategy_name, summary) in &results {
        info!("Exporting results for strategy: {}", strategy_name);
        
        let strategy_dir = output_dir.join(strategy_name);
        std::fs::create_dir_all(&strategy_dir)?;
        
        match format.as_str() {
            "json" => {
                let json_path = strategy_dir.join("metrics.json");
                let json_content = summary.to_json()?;
                std::fs::write(json_path, json_content)?;
            }
            "csv" => {
                let csv_path = strategy_dir.join("metrics.csv");
                let csv_content = summary.to_csv()?;
                std::fs::write(csv_path, csv_content)?;
            }
            "both" => {
                let json_path = strategy_dir.join("metrics.json");
                let json_content = summary.to_json()?;
                std::fs::write(json_path, json_content)?;
                
                let csv_path = strategy_dir.join("metrics.csv");
                let csv_content = summary.to_csv()?;
                std::fs::write(csv_path, csv_content)?;
            }
            _ => {
                error!("Unknown format: {}. Use 'json', 'csv', or 'both'", format);
                return Err(anyhow::anyhow!("Unknown format: {}", format));
            }
        }
    }
    
    // Generate comparison if multiple strategies
    if results.len() > 1 {
        info!("Generating strategy comparison");
        let comparison = mesh_sim::analysis::compare_strategies(&results);
        let comparison_path = output_dir.join("comparison.csv");
        std::fs::write(comparison_path, comparison.to_csv())?;
    }
    
    info!("Simulation completed successfully. Results saved to {:?}", output_dir);
    Ok(())
}

fn generate_example(example_type: String, output_path: PathBuf) -> Result<()> {
    info!("Generating {} example configuration", example_type);
    
    let config = match example_type.as_str() {
        "small" => create_small_example(),
        "medium" => create_medium_example(),
        "large" => create_large_example(),
        _ => {
            error!("Unknown example type: {}. Use 'small', 'medium', or 'large'", example_type);
            return Err(anyhow::anyhow!("Unknown example type: {}", example_type));
        }
    };
    
    config.to_yaml_file(output_path.to_str().unwrap())?;
    info!("Example configuration saved to {:?}", output_path);
    Ok(())
}

fn validate_config(config_path: PathBuf) -> Result<()> {
    info!("Validating configuration file: {:?}", config_path);
    
    let config = SimulationConfig::from_yaml_file(config_path.to_str().unwrap())?;
    config.validate()?;
    
    info!("Configuration is valid!");
    info!("  - Duration: {} seconds", config.duration_s);
    info!("  - Cells: {}", config.topology.cells);
    info!("  - Nodes per cell: {}", config.topology.nodes_per_cell);
    info!("  - Total nodes: {}", config.topology.cells * config.topology.nodes_per_cell);
    info!("  - Strategies: {:?}", config.strategies);
    
    Ok(())
}

fn compare_results(_results_dir: PathBuf, _output_path: PathBuf) -> Result<()> {
    info!("Comparing results from directory: {:?}", _results_dir);
    
    // TODO: Implement result comparison from directory
    // This would load multiple result files and compare them
    
    error!("Result comparison not yet implemented");
    Err(anyhow::anyhow!("Result comparison not yet implemented"))
}

fn create_small_example() -> SimulationConfig {
    use mesh_sim::*;
    
    SimulationConfig {
        seed: 42,
        duration_s: 300.0,
        workload: workload::WorkloadConfig {
            duration_s: 300.0,
            arrival: workload::ArrivalConfig::Poisson { rps: 800.0 },
            mix: workload::RequestMix {
                llm: 1.0,
                vision: 0.0,
                asr: 0.0,
            },
            llm: workload::LlmConfig {
                in_tokens: workload::TokenDistribution::LogNormal { mu: 3.8, sigma: 0.6 },
                out_tokens: workload::TokenDistribution::LogNormal { mu: 4.6, sigma: 0.7 },
            },
            vision: None,
            asr: None,
            tenants: None,
        },
        topology: TopologyConfig {
            cells: 4,
            nodes_per_cell: 128,
            gpu_profiles: vec![GpuProfile {
                name: "H100-80G".to_string(),
                tokens_per_s: 240000,
                concurrency: 16,
                vram_total_gb: 80.0,
                batch_window_ms: 8.0,
                kv_cache_gb_per_req: 1.0,
            }],
            mig: None,
        },
        network: NetworkConfig {
            intra_cell_rtt_ms: net::LatencyDistribution::Normal { mean: 0.5, std: 0.1 },
            inter_cell_coords: net::VivalidoConfig {
                dim: 3,
                base_rtt_ms: 25.0,
                noise: 0.1,
            },
            bw_mbps: net::BandwidthConfig {
                intra_cell: 50000,
                inter_region: 5000,
            },
        },
        signals: SignalConfig {
            queue_depth_ms: signals::UpdateFrequency { min: 50.0, max: 100.0 },
            vram_ms: signals::UpdateFrequency { min: 200.0, max: 400.0 },
            p95_ms: signals::UpdateFrequency { min: 1000.0, max: 1500.0 },
            transport_ms: signals::TransportDelayConfig {
                intra_cell: [5.0, 25.0],
                inter_cell: [50.0, 200.0],
            },
        },
        strategies: vec![
            "baseline_rr".to_string(),
            "heuristic".to_string(),
            "mesh".to_string(),
            "mesh_hedge".to_string(),
            "adaptive_mesh".to_string(),
            "predictive_mesh".to_string(),
            "hybrid_mesh".to_string(),
            "ml_enhanced_mesh".to_string(),
        ],
    }
}

fn create_medium_example() -> SimulationConfig {
    let mut config = create_small_example();
    
    // Scale up for medium example
    config.duration_s = 600.0;
    config.workload.duration_s = 600.0;
    config.topology.cells = 16;
    config.topology.nodes_per_cell = 256;
    
    // Add mixed workload
    config.workload.mix = workload::RequestMix {
        llm: 0.7,
        vision: 0.2,
        asr: 0.1,
    };
    
    // Add vision and ASR configs
    config.workload.vision = Some(workload::VisionConfig {
        image_size: workload::ImageSizeDistribution::Uniform {
            min_pixels: 224 * 224,
            max_pixels: 1024 * 1024,
        },
        processing_complexity: workload::ComplexityDistribution::Constant { value: 1.0 },
    });
    
    config.workload.asr = Some(workload::AsrConfig {
        audio_duration_s: workload::DurationDistribution::Uniform { min_s: 1.0, max_s: 30.0 },
        sample_rate: 16000,
    });
    
    // Add tenant skew
    config.workload.tenants = Some(workload::TenantConfig {
        skew: workload::TenantSkew::Zipf { s: 1.1 },
        count: 100,
    });
    
    config
}

fn create_large_example() -> SimulationConfig {
    let mut config = create_medium_example();
    
    // Scale up for large example
    config.duration_s = 1800.0; // 30 minutes
    config.workload.duration_s = 1800.0;
    config.topology.cells = 128;
    config.topology.nodes_per_cell = 1024;
    
    // Use MMPP for burstiness
    config.workload.arrival = workload::ArrivalConfig::MMPP {
        states: 3,
        rates_rps: vec![200.0, 800.0, 1500.0],
        dwell_s: vec![30.0, 30.0, 10.0],
    };
    
    // Add MIG configuration
    config.topology.mig = Some(world::MigConfig {
        enable: true,
        profiles: vec![
            MigProfile {
                name: "1g.10gb".to_string(),
                fraction: 0.125,
                tokens_per_s: 30000,
                concurrency: 2,
            },
            MigProfile {
                name: "2g.20gb".to_string(),
                fraction: 0.25,
                tokens_per_s: 60000,
                concurrency: 4,
            },
        ],
    });
    
    // Add stale strategy
    config.strategies.push("mesh_stale".to_string());
    
    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_example_generation() {
        let temp_dir = tempdir().unwrap();
        let output_path = temp_dir.path().join("test.yaml");
        
        let result = generate_example("small".to_string(), output_path.clone());
        assert!(result.is_ok());
        assert!(output_path.exists());
        
        // Validate the generated config
        let config = SimulationConfig::from_yaml_file(output_path.to_str().unwrap());
        assert!(config.is_ok());
        assert!(config.unwrap().validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let temp_dir = tempdir().unwrap();
        let config_path = temp_dir.path().join("config.yaml");
        
        // Generate a valid config
        let config = create_small_example();
        config.to_yaml_file(config_path.to_str().unwrap()).unwrap();
        
        // Validate it
        let result = validate_config(config_path);
        assert!(result.is_ok());
    }
}
