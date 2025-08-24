//! Utility functions for development and testing

use crate::{DevError, Result};
use crate::load_generator::TokenDistribution;
use mesh_core::{GpuState, Labels, ModelState};
use rand::Rng;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Setup test logging with appropriate levels
pub fn setup_test_logging() {
    let _ = tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "mesh_dev=debug,mesh_core=debug,mesh_proto=debug,mesh_metrics=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer().with_test_writer())
        .try_init();
}

/// Generate test data for various infermesh components
pub fn generate_test_data() -> TestDataGenerator {
    TestDataGenerator::new()
}

/// Test data generator for creating realistic test data
pub struct TestDataGenerator {
    rng: rand::rngs::ThreadRng,
}

impl TestDataGenerator {
    /// Create a new test data generator
    pub fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }

    /// Generate a random model name
    pub fn random_model_name(&mut self) -> String {
        let models = [
            "gpt-3.5-turbo",
            "gpt-4",
            "claude-3-opus",
            "claude-3-sonnet",
            "llama-2-7b",
            "llama-2-13b",
            "llama-2-70b",
            "mistral-7b",
            "mixtral-8x7b",
            "codellama-7b",
            "codellama-13b",
            "vicuna-7b",
            "vicuna-13b",
            "alpaca-7b",
            "falcon-7b",
            "falcon-40b",
        ];
        
        models[self.rng.gen_range(0..models.len())].to_string()
    }

    /// Generate a random model revision
    pub fn random_model_revision(&mut self) -> String {
        let revisions = ["v1.0", "v1.1", "v1.2", "v2.0", "v2.1", "latest", "main"];
        revisions[self.rng.gen_range(0..revisions.len())].to_string()
    }

    /// Generate a random runtime
    pub fn random_runtime(&mut self) -> String {
        let runtimes = ["triton", "vllm", "tgi", "tensorrt-llm", "onnx", "torch"];
        runtimes[self.rng.gen_range(0..runtimes.len())].to_string()
    }

    /// Generate a random node ID
    pub fn random_node_id(&mut self) -> String {
        format!("node-{:04}", self.rng.gen_range(1000..9999))
    }

    /// Generate a random GPU UUID
    pub fn random_gpu_uuid(&mut self) -> String {
        format!("GPU-{:08X}-{:04X}-{:04X}-{:04X}-{:012X}",
                self.rng.gen::<u32>(),
                self.rng.gen::<u16>(),
                self.rng.gen::<u16>(),
                self.rng.gen::<u16>(),
                self.rng.gen::<u64>() & 0xFFFFFFFFFFFF)
    }

    /// Generate a random zone
    pub fn random_zone(&mut self) -> String {
        let zones = [
            "us-west-1", "us-west-2", "us-east-1", "us-east-2",
            "eu-west-1", "eu-west-2", "eu-central-1",
            "ap-southeast-1", "ap-southeast-2", "ap-northeast-1",
        ];
        zones[self.rng.gen_range(0..zones.len())].to_string()
    }

    /// Generate random labels
    pub fn random_labels(&mut self) -> Labels {
        let mut labels = Labels::new(
            &self.random_model_name(),
            &self.random_model_revision(),
            &self.random_runtime(),
            &self.random_node_id(),
        );

        // Add optional fields
        if self.rng.gen_bool(0.7) {
            labels = labels.with_gpu_uuid(&self.random_gpu_uuid());
        }

        if self.rng.gen_bool(0.3) {
            let quants = ["fp16", "int8", "int4", "fp32"];
            labels = labels.with_quant(quants[self.rng.gen_range(0..quants.len())]);
        }

        if self.rng.gen_bool(0.5) {
            labels = labels.with_zone(&self.random_zone());
        }

        if self.rng.gen_bool(0.4) {
            let tenants = ["customer-a", "customer-b", "internal", "research"];
            labels = labels.with_tenant(tenants[self.rng.gen_range(0..tenants.len())]);
        }

        labels
    }

    /// Generate a realistic model state
    pub fn random_model_state(&mut self) -> ModelState {
        let labels = self.random_labels();
        let mut state = ModelState::new(labels);

        // Generate realistic metrics
        let queue_depth = self.rng.gen_range(0..50);
        let service_rate = self.rng.gen_range(1.0..100.0);
        let p95_latency_ms = self.rng.gen_range(50..2000);
        let batch_fullness = self.rng.gen_range(0.0..1.0);

        state.update(queue_depth, service_rate, p95_latency_ms, batch_fullness);

        // Randomly set loaded/warming status
        if self.rng.gen_bool(0.8) {
            state.mark_loaded();
        } else if self.rng.gen_bool(0.3) {
            state.mark_warming();
        }

        state
    }

    /// Generate a realistic GPU state
    pub fn random_gpu_state(&mut self) -> GpuState {
        let gpu_uuid = self.random_gpu_uuid();
        let node_id = self.random_node_id();
        let mut state = GpuState::new(&gpu_uuid, &node_id);

        // Generate realistic GPU metrics
        let sm_utilization = self.rng.gen_range(0.0..1.0);
        let memory_utilization = self.rng.gen_range(0.0..1.0);
        let vram_total_gb = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 80.0][self.rng.gen_range(0..7)];
        let vram_used_gb = vram_total_gb * memory_utilization;

        state.update_metrics(sm_utilization, memory_utilization, vram_used_gb, vram_total_gb);

        // Add thermal data
        let temperature_c = self.rng.gen_range(40.0..90.0);
        let power_watts = self.rng.gen_range(150.0..400.0);
        state.update_thermal(Some(temperature_c), Some(power_watts));

        // Rarely add errors
        let ecc_errors = self.rng.gen_bool(0.01);
        let throttled = temperature_c > 85.0 || self.rng.gen_bool(0.05);
        state.update_status(ecc_errors, throttled);

        // Add MIG profile occasionally
        if self.rng.gen_bool(0.2) {
            let mig_profiles = ["1g.5gb", "2g.10gb", "3g.20gb", "4g.20gb", "7g.40gb"];
            state.mig_profile = Some(mig_profiles[self.rng.gen_range(0..mig_profiles.len())].to_string());
        }

        state
    }

    /// Generate multiple model states
    pub fn random_model_states(&mut self, count: usize) -> Vec<ModelState> {
        (0..count).map(|_| self.random_model_state()).collect()
    }

    /// Generate multiple GPU states
    pub fn random_gpu_states(&mut self, count: usize) -> Vec<GpuState> {
        (0..count).map(|_| self.random_gpu_state()).collect()
    }

    /// Generate a realistic request pattern
    pub fn random_request_pattern(&mut self) -> crate::RequestPattern {
        let names = ["short", "medium", "long", "batch", "interactive"];
        let name = names[self.rng.gen_range(0..names.len())].to_string();
        
        let weight = self.rng.gen_range(0.1..5.0);
        let think_time_ms = match name.as_str() {
            "interactive" => self.rng.gen_range(100..1000),
            "batch" => 0,
            _ => self.rng.gen_range(0..500),
        };

        let token_distribution = match name.as_str() {
            "short" => TokenDistribution::Uniform { min: 10, max: 100 },
            "medium" => TokenDistribution::Uniform { min: 100, max: 500 },
            "long" => TokenDistribution::Uniform { min: 500, max: 2000 },
            "batch" => TokenDistribution::Fixed(1000),
            "interactive" => TokenDistribution::Normal { mean: 150.0, std_dev: 50.0 },
            _ => TokenDistribution::Fixed(128),
        };

        crate::RequestPattern {
            name,
            weight,
            token_distribution,
            think_time_ms,
        }
    }

    /// Generate a load generator configuration
    pub fn random_load_config(&mut self) -> crate::LoadGeneratorConfig {
        let mut config = crate::LoadGeneratorConfig::default();
        
        config.target_rps = self.rng.gen_range(1.0..100.0);
        config.duration_seconds = self.rng.gen_range(10..300);
        config.model = self.random_model_name();
        config.revision = self.random_model_revision();
        config.worker_count = self.rng.gen_range(1..10);
        
        // Generate 1-3 request patterns
        let pattern_count = self.rng.gen_range(1..4);
        config.patterns = (0..pattern_count)
            .map(|_| self.random_request_pattern())
            .collect();
        
        config
    }

    /// Generate test configuration files
    pub fn generate_config_files(&mut self, output_dir: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
        std::fs::create_dir_all(output_dir)
            .map_err(|e| DevError::Config(format!("Failed to create output directory: {}", e)))?;

        let mut files = Vec::new();

        // Generate node configurations
        for i in 0..3 {
            let config = crate::create_test_config();
            let config_path = output_dir.join(format!("node-{}.yaml", i));
            
            let yaml = serde_yaml::to_string(&config)
                .map_err(|e| DevError::Config(format!("Failed to serialize config: {}", e)))?;
            
            std::fs::write(&config_path, yaml)
                .map_err(|e| DevError::Config(format!("Failed to write config file: {}", e)))?;
            
            files.push(config_path);
        }

        // Generate load test configurations
        for i in 0..2 {
            let load_config = self.random_load_config();
            let config_path = output_dir.join(format!("load-test-{}.yaml", i));
            
            let yaml = serde_yaml::to_string(&load_config)
                .map_err(|e| DevError::Config(format!("Failed to serialize load config: {}", e)))?;
            
            std::fs::write(&config_path, yaml)
                .map_err(|e| DevError::Config(format!("Failed to write load config file: {}", e)))?;
            
            files.push(config_path);
        }

        Ok(files)
    }
}

impl Default for TestDataGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for working with ports
pub mod ports {
    use std::net::{SocketAddr, TcpListener};

    /// Find an available port starting from the given port
    pub fn find_available_port(start_port: u16) -> Option<u16> {
        for port in start_port..start_port + 1000 {
            if is_port_available(port) {
                return Some(port);
            }
        }
        None
    }

    /// Check if a port is available
    pub fn is_port_available(port: u16) -> bool {
        let addr: SocketAddr = format!("127.0.0.1:{}", port).parse().unwrap();
        TcpListener::bind(addr).is_ok()
    }

    /// Get multiple available ports
    pub fn get_available_ports(count: usize, start_port: u16) -> Vec<u16> {
        let mut ports = Vec::new();
        let mut current_port = start_port;
        
        while ports.len() < count {
            if let Some(port) = find_available_port(current_port) {
                ports.push(port);
                current_port = port + 1;
            } else {
                break;
            }
        }
        
        ports
    }
}

/// Utility functions for working with Docker
pub mod docker {
    use crate::{DevError, Result};
    use std::process::Command;

    /// Check if Docker is available
    pub fn is_docker_available() -> bool {
        Command::new("docker")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    /// Generate a docker-compose.yml for testing
    pub fn generate_docker_compose(output_path: &std::path::Path) -> Result<()> {
        let compose_content = r#"version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_OTLP_ENABLED=true

volumes:
  grafana-storage:
"#;

        std::fs::write(output_path, compose_content)
            .map_err(|e| DevError::Config(format!("Failed to write docker-compose.yml: {}", e)))?;

        // Also generate prometheus config
        let prometheus_config = r#"global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'infermesh'
    static_configs:
      - targets: ['host.docker.internal:9090', 'host.docker.internal:9091', 'host.docker.internal:9092']
"#;

        let prometheus_path = output_path.parent().unwrap().join("prometheus.yml");
        std::fs::write(prometheus_path, prometheus_config)
            .map_err(|e| DevError::Config(format!("Failed to write prometheus.yml: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_setup_test_logging() {
        // Should not panic
        setup_test_logging();
    }

    #[test]
    fn test_test_data_generator() {
        let mut generator = TestDataGenerator::new();
        
        let model_name = generator.random_model_name();
        assert!(!model_name.is_empty());
        
        let labels = generator.random_labels();
        assert!(!labels.model.is_empty());
        assert!(!labels.node.is_empty());
        
        let model_state = generator.random_model_state();
        assert!(!model_state.labels.model.is_empty());
        
        let gpu_state = generator.random_gpu_state();
        assert!(!gpu_state.gpu_uuid.is_empty());
        assert!(gpu_state.vram_total_gb > 0.0);
    }

    #[test]
    fn test_generate_multiple_states() {
        let mut generator = TestDataGenerator::new();
        
        let model_states = generator.random_model_states(5);
        assert_eq!(model_states.len(), 5);
        
        let gpu_states = generator.random_gpu_states(3);
        assert_eq!(gpu_states.len(), 3);
    }

    #[test]
    fn test_generate_config_files() {
        let temp_dir = TempDir::new().unwrap();
        let mut generator = TestDataGenerator::new();
        
        let files = generator.generate_config_files(temp_dir.path()).unwrap();
        assert_eq!(files.len(), 5); // 3 node configs + 2 load configs
        
        for file in files {
            assert!(file.exists());
        }
    }

    #[test]
    fn test_port_utilities() {
        let port = ports::find_available_port(50000).unwrap();
        assert!(port >= 50000);
        assert!(ports::is_port_available(port));
        
        let ports_list = ports::get_available_ports(3, 50100);
        assert_eq!(ports_list.len(), 3);
    }

    #[test]
    fn test_docker_utilities() {
        // Test docker availability check (may fail in CI without Docker)
        let _ = docker::is_docker_available();
        
        // Test docker-compose generation
        let temp_dir = TempDir::new().unwrap();
        let compose_path = temp_dir.path().join("docker-compose.yml");
        
        docker::generate_docker_compose(&compose_path).unwrap();
        assert!(compose_path.exists());
        
        let content = std::fs::read_to_string(&compose_path).unwrap();
        assert!(content.contains("prometheus"));
        assert!(content.contains("grafana"));
        assert!(content.contains("jaeger"));
    }
}
