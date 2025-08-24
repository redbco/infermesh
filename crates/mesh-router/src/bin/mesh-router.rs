//! mesh-router binary

use clap::{Arg, Command};
use mesh_router::{Router, RouterConfig, RouterConfigBuilder};
use std::process;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "mesh_router=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Parse command line arguments
    let matches = Command::new("mesh-router")
        .version(env!("CARGO_PKG_VERSION"))
        .about("HTTP/gRPC ingress and request routing for infermesh")
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Configuration file path")
        )
        .arg(
            Arg::new("http-port")
                .long("http-port")
                .value_name("PORT")
                .help("HTTP server port")
                .value_parser(clap::value_parser!(u16))
        )
        .arg(
            Arg::new("grpc-port")
                .long("grpc-port")
                .value_name("PORT")
                .help("gRPC server port")
                .value_parser(clap::value_parser!(u16))
        )
        .arg(
            Arg::new("bind")
                .short('b')
                .long("bind")
                .value_name("ADDRESS")
                .help("Bind address")
                .default_value("0.0.0.0")
        )
        .arg(
            Arg::new("agent")
                .short('a')
                .long("agent")
                .value_name("ADDRESS")
                .help("mesh-agent address")
                .default_value("127.0.0.1:50051")
        )
        .arg(
            Arg::new("max-requests")
                .long("max-requests")
                .value_name("COUNT")
                .help("Maximum concurrent requests")
                .value_parser(clap::value_parser!(usize))
        )
        .arg(
            Arg::new("request-timeout")
                .long("request-timeout")
                .value_name("SECONDS")
                .help("Request timeout in seconds")
                .value_parser(clap::value_parser!(u64))
        )
        .arg(
            Arg::new("upstream-timeout")
                .long("upstream-timeout")
                .value_name("SECONDS")
                .help("Upstream connection timeout in seconds")
                .value_parser(clap::value_parser!(u64))
        )
        .arg(
            Arg::new("disable-cors")
                .long("disable-cors")
                .help("Disable CORS support")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("disable-websockets")
                .long("disable-websockets")
                .help("Disable WebSocket support")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("disable-grpc-reflection")
                .long("disable-grpc-reflection")
                .help("Disable gRPC reflection")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("load-balancing")
                .long("load-balancing")
                .value_name("STRATEGY")
                .help("Load balancing strategy")
                .value_parser(["round-robin", "random", "least-connections", "weighted-round-robin", "consistent-hashing", "score-based"])
                .default_value("score-based")
        )
        .get_matches();

    // Build configuration
    let config = match build_config(&matches) {
        Ok(config) => config,
        Err(e) => {
            error!("Configuration error: {}", e);
            process::exit(1);
        }
    };

    info!("Starting mesh-router with configuration:");
    info!("  HTTP port: {}", config.http_port);
    info!("  gRPC port: {}", config.grpc_port);
    info!("  Bind address: {}", config.bind_address);
    info!("  Agent address: {}", config.agent_address);
    info!("  Max concurrent requests: {}", config.max_concurrent_requests);
    info!("  Request timeout: {:?}", config.request_timeout);
    info!("  CORS enabled: {}", config.enable_cors);
    info!("  WebSockets enabled: {}", config.enable_websockets);
    info!("  gRPC reflection enabled: {}", config.enable_grpc_reflection);

    // Create and start router
    let router = match Router::new(config).await {
        Ok(router) => router,
        Err(e) => {
            error!("Failed to create router: {}", e);
            process::exit(1);
        }
    };

    // Start the router (this will block until shutdown)
    if let Err(e) = router.serve("").await {
        error!("Router error: {}", e);
        process::exit(1);
    }

    info!("mesh-router shutdown complete");
}

/// Build configuration from command line arguments and config file
fn build_config(matches: &clap::ArgMatches) -> Result<RouterConfig, String> {
    let mut builder = RouterConfigBuilder::new();

    // Load from config file if specified
    if let Some(config_path) = matches.get_one::<String>("config") {
        warn!("Config file loading not yet implemented: {}", config_path);
        // TODO: Load configuration from file
        // let file_config = load_config_file(config_path)?;
        // builder = builder.merge_from_file(file_config);
    }

    // Override with command line arguments
    if let Some(&http_port) = matches.get_one::<u16>("http-port") {
        builder = builder.http_port(http_port);
    }

    if let Some(&grpc_port) = matches.get_one::<u16>("grpc-port") {
        builder = builder.grpc_port(grpc_port);
    }

    if let Some(bind_addr) = matches.get_one::<String>("bind") {
        builder = builder.bind_address(bind_addr);
    }

    if let Some(agent_addr) = matches.get_one::<String>("agent") {
        let addr = agent_addr.parse()
            .map_err(|e| format!("Invalid agent address '{}': {}", agent_addr, e))?;
        builder = builder.agent_address(addr);
    }

    if let Some(&max_requests) = matches.get_one::<usize>("max-requests") {
        builder = builder.max_concurrent_requests(max_requests);
    }

    if let Some(&timeout_secs) = matches.get_one::<u64>("request-timeout") {
        builder = builder.request_timeout(std::time::Duration::from_secs(timeout_secs));
    }

    if let Some(&timeout_secs) = matches.get_one::<u64>("upstream-timeout") {
        builder = builder.upstream_timeout(std::time::Duration::from_secs(timeout_secs));
    }

    if matches.get_flag("disable-cors") {
        builder = builder.enable_cors(false);
    }

    if matches.get_flag("disable-websockets") {
        builder = builder.enable_websockets(false);
    }

    if matches.get_flag("disable-grpc-reflection") {
        builder = builder.enable_grpc_reflection(false);
    }

    if let Some(strategy) = matches.get_one::<String>("load-balancing") {
        let lb_strategy = match strategy.as_str() {
            "round-robin" => mesh_router::config::LoadBalancingStrategy::RoundRobin,
            "random" => mesh_router::config::LoadBalancingStrategy::Random,
            "least-connections" => mesh_router::config::LoadBalancingStrategy::LeastConnections,
            "weighted-round-robin" => mesh_router::config::LoadBalancingStrategy::WeightedRoundRobin,
            "consistent-hashing" => mesh_router::config::LoadBalancingStrategy::ConsistentHashing,
            "score-based" => mesh_router::config::LoadBalancingStrategy::ScoreBased,
            _ => return Err(format!("Invalid load balancing strategy: {}", strategy)),
        };
        builder = builder.load_balancing(lb_strategy);
    }

    let config = builder.build();

    // Validate configuration
    mesh_router::config::validate_config(&config)
        .map_err(|e| format!("Configuration validation failed: {}", e))?;

    Ok(config)
}

/// Load configuration from file (placeholder)
#[allow(dead_code)]
fn load_config_file(_path: &str) -> Result<RouterConfig, String> {
    // TODO: Implement configuration file loading
    // This would typically load from YAML, TOML, or JSON
    Err("Config file loading not implemented".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a test command with all the required arguments defined
    fn create_test_command() -> Command {
        Command::new("test")
            .arg(
                Arg::new("config")
                    .short('c')
                    .long("config")
                    .value_name("FILE")
                    .help("Configuration file path")
            )
            .arg(
                Arg::new("http-port")
                    .long("http-port")
                    .value_name("PORT")
                    .help("HTTP server port")
                    .value_parser(clap::value_parser!(u16))
            )
            .arg(
                Arg::new("grpc-port")
                    .long("grpc-port")
                    .value_name("PORT")
                    .help("gRPC server port")
                    .value_parser(clap::value_parser!(u16))
            )
            .arg(
                Arg::new("bind")
                    .short('b')
                    .long("bind")
                    .value_name("ADDRESS")
                    .help("Bind address")
                    .default_value("0.0.0.0")
            )
            .arg(
                Arg::new("agent")
                    .short('a')
                    .long("agent")
                    .value_name("ADDRESS")
                    .help("mesh-agent address")
                    .default_value("127.0.0.1:50051")
            )
            .arg(
                Arg::new("max-requests")
                    .long("max-requests")
                    .value_name("COUNT")
                    .help("Maximum concurrent requests")
                    .value_parser(clap::value_parser!(usize))
            )
            .arg(
                Arg::new("request-timeout")
                    .long("request-timeout")
                    .value_name("SECONDS")
                    .help("Request timeout in seconds")
                    .value_parser(clap::value_parser!(u64))
            )
            .arg(
                Arg::new("upstream-timeout")
                    .long("upstream-timeout")
                    .value_name("SECONDS")
                    .help("Upstream connection timeout in seconds")
                    .value_parser(clap::value_parser!(u64))
            )
            .arg(
                Arg::new("disable-cors")
                    .long("disable-cors")
                    .help("Disable CORS support")
                    .action(clap::ArgAction::SetTrue)
            )
            .arg(
                Arg::new("disable-websockets")
                    .long("disable-websockets")
                    .help("Disable WebSocket support")
                    .action(clap::ArgAction::SetTrue)
            )
            .arg(
                Arg::new("disable-grpc-reflection")
                    .long("disable-grpc-reflection")
                    .help("Disable gRPC reflection")
                    .action(clap::ArgAction::SetTrue)
            )
            .arg(
                Arg::new("load-balancing")
                    .long("load-balancing")
                    .value_name("STRATEGY")
                    .help("Load balancing strategy")
                    .value_parser(["round-robin", "random", "least-connections", "weighted-round-robin", "consistent-hashing", "score-based"])
                    .default_value("score-based")
            )
    }

    #[test]
    fn test_default_config_build() {
        let matches = create_test_command()
            .get_matches_from(vec!["test"]);
        
        let config = build_config(&matches);
        assert!(config.is_ok());
        
        let config = config.unwrap();
        assert_eq!(config.http_port, 8080);
        assert_eq!(config.grpc_port, 9090);
        assert_eq!(config.bind_address, "0.0.0.0");
    }

    #[test]
    fn test_custom_ports() {
        let matches = create_test_command()
            .get_matches_from(vec!["test", "--http-port", "3000", "--grpc-port", "3001"]);
        
        let config = build_config(&matches);
        assert!(config.is_ok());
        
        let config = config.unwrap();
        assert_eq!(config.http_port, 3000);
        assert_eq!(config.grpc_port, 3001);
    }

    #[test]
    fn test_invalid_agent_address() {
        let matches = create_test_command()
            .get_matches_from(vec!["test", "--agent", "invalid-address"]);
        
        let config = build_config(&matches);
        assert!(config.is_err());
    }

    #[test]
    fn test_load_balancing_strategies() {
        let strategies = vec![
            "round-robin",
            "random", 
            "least-connections",
            "weighted-round-robin",
            "consistent-hashing",
            "score-based",
        ];

        for strategy in strategies {
            let matches = create_test_command()
                .get_matches_from(vec!["test", "--load-balancing", strategy]);
            
            let config = build_config(&matches);
            assert!(config.is_ok(), "Strategy {} should be valid", strategy);
        }
    }

    #[test]
    fn test_invalid_load_balancing_strategy() {
        // Test that clap rejects invalid load balancing strategies at parse time
        let result = create_test_command()
            .try_get_matches_from(vec!["test", "--load-balancing", "invalid-strategy"]);
        
        assert!(result.is_err(), "Should reject invalid load balancing strategy");
    }
}
