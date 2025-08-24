# Getting Started with infermesh

This guide helps you build, run, and experiment with **infermesh** locally. It covers both mock deployments for testing and real GPU deployments.

---

## 1. Prerequisites

- **Rust** (latest stable via [rustup](https://rustup.rs/))
- **Protobuf compiler** (`protoc`)
- **Docker** (optional, for container builds)
- **NVIDIA GPU drivers + CUDA toolkit** (for real GPU nodes)
- **DCGM** (for GPU telemetry) – optional in mock mode

---

## 2. Clone the Repository

```bash
git clone https://github.com/redbco/infermesh.git
cd infermesh
```

---

## 3. Build

```bash
# Build all crates
cargo build --release

# Or build just agent and router
cargo build --release -p mesh-agent -p mesh-router
```

The binaries will be in `target/release/`.

---

## 4. Run in Development Mode

Start a **single node** with default configuration:

```bash
# Start the mesh agent daemon
cargo run -p mesh-agent -- start

# The agent will start with mock adapters by default
# - Mock GPU telemetry (simulates NVIDIA GPUs)
# - Mock runtime adapters (simulates vLLM, Triton, TGI)
# - Control plane API on port 50051
# - Metrics endpoint on port 9090
```

Interact with the mesh:

```bash
# List nodes in the mesh
cargo run -p mesh-cli -- list-nodes

# Get mesh statistics  
cargo run -p mesh-cli -- stats

# Check health status
cargo run -p mesh-cli -- health

# Check metrics
curl http://127.0.0.1:9090/metrics
```

---

## 5. Multi-Node Setup (local)

You can run multiple agents locally using configuration files:

```bash
# Terminal 1: First node
cargo run -p mesh-agent -- start --config examples/node1.yaml

# Terminal 2: Second node  
cargo run -p mesh-agent -- start --config examples/node2.yaml

# Terminal 3: Third node (router-only)
cargo run -p mesh-agent -- start --config examples/router.yaml
```

Check cluster membership:

```bash
# List all nodes in the mesh
cargo run -p mesh-cli -- list-nodes

# Get detailed node information
cargo run -p mesh-cli -- describe-node node1

# Monitor cluster statistics
cargo run -p mesh-cli -- stats
```

---

## 6. Using Docker

Build container images:

```bash
docker build -t infermesh-agent -f Dockerfile.agent .
docker build -t infermesh-router -f Dockerfile.router .
```

Run with Docker Compose (see `examples/compose.yml`):

```bash
docker-compose up
```

---

## 7. Production Deployment with Real GPUs

### GPU Node Setup

On GPU-capable hosts:

1. **Install NVIDIA drivers + CUDA toolkit**
2. **Install your inference runtime** (Triton, vLLM, or TGI)
3. **Optional**: Install DCGM for enterprise GPU monitoring

Create a configuration file (`gpu-node.yaml`):

```yaml
node:
  id: "gpu-node-1"
  roles: ["gpu"]
  
adapters:
  runtime:
    - type: "triton"
      endpoint: "http://localhost:8000"
      health_check_interval: 30
    - type: "vllm"  
      endpoint: "http://localhost:8001"
      health_check_interval: 30
      
  gpu:
    - type: "nvml"
      collection_interval: 10
    # OR for enterprise setups:
    # - type: "dcgm"
    #   dcgm_socket: "/var/run/dcgm.sock"
```

Start the agent:

```bash
cargo run -p mesh-agent -- start --config gpu-node.yaml
```

### Router Node Setup

On router hosts, create `router-node.yaml`:

```yaml
node:
  id: "router-1"
  roles: ["router"]
  
router:
  listen_address: "0.0.0.0:8080"
  health_check_interval: 15
  connection_pool_size: 100
```

Start the router:

```bash
cargo run -p mesh-agent -- start --config router-node.yaml
```

Now inference requests to the router will be intelligently routed to the best available GPU node based on real-time telemetry.

---

## 8. Observability

### Prometheus
All components expose metrics at `/metrics`:
- Router: request/latency metrics
- Agent: gossip, Raft, control-plane metrics
- Adapters: runtime queue depth, GPU utilization

### Grafana
Use the example dashboards in [DASHBOARDS.md](DASHBOARDS.md).

### Tracing
Enable OpenTelemetry export:

```bash
RUST_LOG=info OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317 cargo run -p mesh-agent -- --otel
```

---

## 9. Control Plane API

The CLI provides comprehensive mesh management capabilities:

```bash
# Node management
cargo run -p mesh-cli -- list-nodes
cargo run -p mesh-cli -- describe-node <node-id>

# Model management  
cargo run -p mesh-cli -- pin-model <model-name> --nodes node1,node2
cargo run -p mesh-cli -- unpin-model <model-name>
cargo run -p mesh-cli -- list-pins

# Monitoring
cargo run -p mesh-cli -- stats
cargo run -p mesh-cli -- health

# Event streaming
cargo run -p mesh-cli -- subscribe-events

# Configuration
cargo run -p mesh-cli -- config --help
```

All commands support JSON output for automation:

```bash
cargo run -p mesh-cli -- list-nodes --output json
cargo run -p mesh-cli -- stats --json
```

---

## 10. Next Steps

- Explore the [ARCHITECTURE.md](ARCHITECTURE.md) for a deeper dive.  
- Try running with a real runtime like Triton or vLLM.  
- Deploy in Kubernetes using a DaemonSet for `mesh-agent` and a Deployment for `mesh-router`.  
- Build custom Grafana dashboards using PromQL queries from [DASHBOARDS.md](DASHBOARDS.md).

---

## 11. Troubleshooting

- **Metrics endpoint not found**: check `--metrics` flag when starting agents.  
- **GPU telemetry missing**: ensure DCGM is installed and running.  
- **Gossip membership issues**: open UDP/TCP ports used by agents (default 7946/7947).  
- **Control-plane writes blocked**: ensure Raft quorum is available (3+ agents recommended).  

---

Happy hacking with **infermesh** 🚀
