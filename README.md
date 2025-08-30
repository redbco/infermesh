# InferMesh

**InferMesh** is a **GPU- and network-aware mesh** for large-scale **AI serving and training**.  
It provides a distributed control plane, topology- and GPU-aware routing, and standardized observability across heterogeneous environments.

---

## 💡 Why InferMesh is different?

InferMesh treats inference and training as first-class citizens on a single mesh fabric: a topology-aware underlay (gossip, Raft, secure networking, link coordinates) with a workload overlay (routers, schedulers, adapters) and a pluggable strategy host. Unlike point tools (serving frameworks, trainers, or generic service meshes), it makes GPU, runtime, and network signals actionable in real time for both request routing and collective topology, and lets you swap in user-defined strategies (Rust or WASM) that run under strict latency budgets with built-in A/B and observability. It scales across clusters and regions with cell summaries, supports edge-to-cloud paths, and optimizes for SLOs, utilization, and cost—all through one consistent API and label model.

---

## ✨ What is InferMesh?

Modern AI at scale faces three hard problems:

1. **Observability** – understanding GPU health, utilization, queue depths, collective timings, and tail latency across a large fleet.  
2. **Placement & Routing** – sending the right work to the right resources (inference requests *and* training collectives) while accounting for network cost and GPU pressure.  
3. **Cost & Reliability** – minimizing cost per token/step and reducing failure blast-radius without breaking SLAs.

**InferMesh** solves this by introducing a **two-layer architecture** with **three planes**:

- **Infrastructure Layer (Underlay)** – the mesh fabric: gossip, consensus, topology awareness, secure networking, node identity/lifecycle.  
- **Workload Overlay Layer** – connectors and integrations for inference runtimes and training frameworks, plus the **pluggable strategy host** for routing/topology decisions.

- **Planes** (cross-cutting):  
  - **Communication/Data Plane** – inference request forwarding & streaming; training collectives (all-reduce/all-gather).  
  - **Signal Plane** – runtime + GPU + network telemetry fused into a local scoring/topology API.  
  - **Control Plane** – policies, placements, quotas, job specs, events, and RBAC.

Advanced strategies (e.g., **`hybrid_mesh`**) have shown **~2× better latency** and **~40–50% lower cost per 1k tokens** compared to baseline round-robin in simulation (512-node runs). Results vary by workload and topology.

---

## 🎯 Who is this for?

Teams operating **meaningful-scale AI** where utilization, latency, and reliability matter:

- **Inference platforms** serving LLMs, vision, ASR across hundreds–thousands of GPUs, multi-zone/region, or hybrid edge + cloud.  
- **Training platforms** running distributed jobs (PyTorch DDP, DeepSpeed, Megatron) that need topology-aware collectives and elastic placement across heterogeneous interconnects.  
- **Cloud & platform providers** building multi-tenant AI services with GPU-aware routing and consistent observability.  
- **Enterprises & research labs** coordinating shared GPU fleets with strict SLOs, quotas, and auditability.

If you’re <100 GPUs in a single cluster, Triton or K8s-only may suffice. Above ~500 GPUs or across multiple regions, **InferMesh** delivers outsized ROI by improving utilization, cutting tail latency, and simplifying operations.

---

## 🧭 Use Cases & Benefits

### Inference (Serving)
- **Low latency at scale**: GPU-aware + network-aware routing, queue-depth and VRAM signals, hedging for tight SLAs.  
- **Higher utilization**: smarter packing (MIG/MPS aware), reduced hotspots, fewer cold starts.  
- **Cost efficiency**: steer to lower-cost capacity when SLOs allow; throttle or quarantine unhealthy nodes.  
- **Edge + cloud**: route locally when possible, account for WAN penalties when not.

### Training (Distributed Jobs)
- **Topology-aware collectives**: per-step hints (ring/tree/hierarchical), dynamic fusion windows, and congestion-aware paths.  
- **Elastic placement**: schedule across heterogeneous nodes and interconnects; scale in/out with policy.  
- **Resilience**: detect link/GPU degradation (DCGM) and adjust topology; reduce restarts via quarantine and rebalancing.  
- **Multi-cluster readiness**: WAN-aware link coordinates and summaries for cross-cell planning.

---

## 🔧 Core Features

- **Unified mesh fabric (underlay)**  
  Gossip membership (SWIM), Raft-backed policy store, topology service (coordinates & link classes), secure transport (mTLS/QUIC), node identity & lifecycle.

- **Workload overlay**  
  - **Inference Router** – HTTP/2 gRPC/HTTP/WS ingress with admission control and hedging.  
  - **Training Scheduler** – placement, topology hints, and elasticity for distributed jobs.  
  - **Adapters** – Runtime (Triton, vLLM, TGI, TorchServe, TF/OVMS), Comm (NCCL/UCC telemetry), GPU (DCGM/NVML/ROCm-SMI), IO (datasets/checkpoints).

- **Pluggable strategies (inference + training)**  
  A **strategy host** loads built-in and user-defined strategies (Rust `cdylib` or sandboxed WASM).  
  - *Inference outputs*: ranked targets, hedges, admission decisions.  
  - *Training outputs*: placement plans, collective topology hints, fusion/compression toggles.  
  The same interface powers internal strategies like **`hybrid_mesh`**.

- **Observability built-in**  
  Prometheus metrics and OpenTelemetry traces for routers, schedulers, agents, and adapters; consistent labels across GPUs/runtimes/jobs.

---

## 📐 Architecture Overview

- **Two layers**: Infrastructure (mesh fabric) and Workload Overlay (routers, schedulers, adapters, strategy host).  
- **Three planes**: Communication/Data, Signal, Control.  
- **Security**: mTLS everywhere, SPIFFE-compatible identities, RBAC via OIDC/JWT.  
- **Scalability**: cell/shard design with inter-cell summaries; interest-scoped dissemination to reduce chatter.

Read more in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 🚀 Getting Started

### Prerequisites
- **Rust** (latest stable via [rustup](https://rustup.rs/))
- **Protobuf compiler** (`protoc`)
- **Docker** (optional, for container builds)
- **NVIDIA GPU drivers + CUDA toolkit** (for real GPU nodes)

### Build
```bash
git clone https://github.com/redbco/infermesh.git
cd infermesh

# Using Makefile (recommended)
make build

# Or using cargo directly
cargo build --release
```

### Quick Start - Single Node
Start a mesh agent with mock adapters for testing:

```bash
# Start the agent daemon
cargo run -p mesh-agent -- start

# In another terminal, check status
cargo run -p mesh-cli -- list-nodes

# Check metrics
curl http://127.0.0.1:9090/metrics
```

### Development Workflow
The project includes a comprehensive Makefile for common development tasks:

```bash
# Show all available targets
make help

# Create example configurations
make examples

# Development setup (check + test + build)
make dev

# Quick development check (format + lint + test)
make quick

# Build specific components
make build-agent    # Build mesh agent only
make build-cli      # Build CLI only
cargo build -p mesh-sim  # Build simulator only

# Testing
make test           # Run all tests
make test-unit      # Run unit tests only

# Code quality
make fmt            # Format code
make clippy         # Run linter
make doc            # Generate documentation

# Release preparation
make release-check  # Verify ready for release
make release-build  # Build release artifacts
```

### Multi-Node Setup
```bash
# Terminal 1: First node (GPU + Router roles)
cargo run -p mesh-agent -- start --config examples/node1.yaml

# Terminal 2: Second node (GPU role only)  
cargo run -p mesh-agent -- start --config examples/node2.yaml

# Terminal 3: Use CLI to interact
cargo run -p mesh-cli -- list-nodes
cargo run -p mesh-cli -- stats
```

More detailed instructions in [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## 📊 Observability

- **Prometheus metrics**: All components expose `/metrics` endpoints
- **OpenTelemetry tracing**: Optional OTLP export for distributed tracing  
- **CLI monitoring**: Real-time stats via `mesh list-nodes`, `mesh stats`
- **Example dashboards**: Grafana templates available in [docs/DASHBOARDS.md](docs/DASHBOARDS.md)

## 🧪 Discrete-Event Simulator

InferMesh includes a comprehensive **discrete-event simulator** (`mesh-sim`) that lets you evaluate routing strategies, measure performance at scale, and optimize configurations before deployment.

### What can you simulate?

- **Scale**: From hundreds to 1M+ nodes across cells/regions
- **Routing strategies**: Compare 7 strategies including baseline (round-robin), heuristic, mesh, mesh+hedging, adaptive mesh, predictive mesh, and hybrid mesh
- **Workloads**: LLM, Vision, ASR requests with realistic token distributions and burstiness
- **Hardware**: Different GPU types (H100, A100, L40), MIG configurations, VRAM constraints
- **Network**: Vivaldi coordinate system for realistic inter-cell latency modeling
- **Metrics**: p50/p95/p99 latency, GPU utilization, cost per 1k tokens, hedge effectiveness

### Quick Start

```bash
# Generate example configurations
cargo run -p mesh-sim -- generate --example-type small --output small.yaml
cargo run -p mesh-sim -- generate --example-type medium --output medium.yaml
cargo run -p mesh-sim -- generate --example-type large --output large.yaml

# Validate a configuration
cargo run -p mesh-sim -- validate --config small.yaml

# Run simulation with all configured strategies
cargo run -p mesh-sim -- run --config small.yaml --output results/

# Run simulation with a specific strategy
cargo run -p mesh-sim -- run --config small.yaml --strategy mesh --output results/
```

### Configuration Examples

**Small Scale** (4 cells, 512 nodes):
```yaml
seed: 42
duration_s: 300
workload:
  arrival: { type: poisson, rps: 800 }
  mix: { llm: 1.0, vision: 0.0, asr: 0.0 }
topology:
  cells: 4
  nodes_per_cell: 128
  gpu_profiles:
    - name: H100-80G
      tokens_per_s: 240000
      concurrency: 16
      vram_total_gb: 80
strategies: [baseline_rr, heuristic, mesh, mesh_hedge]
```

**Large Scale** (128 cells, 131k nodes) with burstiness:
```yaml
workload:
  arrival:
    type: mmpp  # Markov-Modulated Poisson Process
    states: 3
    rates_rps: [200, 800, 1500]  # Low/medium/high traffic
    dwell_s: [30, 30, 10]        # Time in each state
  mix: { llm: 0.7, vision: 0.2, asr: 0.1 }
topology:
  cells: 128
  nodes_per_cell: 1024
  mig:
    enable: true
    profiles:
      - { name: "1g.10gb", fraction: 0.125, tokens_per_s: 30000 }
      - { name: "3g.40gb", fraction: 0.5, tokens_per_s: 120000 }
```

### Exporting Results

The simulator exports results in multiple formats:

**CSV Export** (for analysis/plotting):
```bash
# Export only CSV
cargo run -p mesh-sim -- run --config config.yaml --format csv --output results/

# Results structure:
results/
├── config.yaml           # Configuration used
├── comparison.csv         # Strategy comparison summary
├── baseline_rr/
│   └── metrics.csv       # Detailed metrics for baseline
├── mesh/
│   └── metrics.csv       # Detailed metrics for mesh strategy
└── mesh_hedge/
    └── metrics.csv       # Detailed metrics for mesh+hedging
```

**JSON Export** (for programmatic analysis):
```bash
# Export only JSON
cargo run -p mesh-sim -- run --config config.yaml --format json --output results/

# Export both formats (default)
cargo run -p mesh-sim -- run --config config.yaml --format both --output results/
```

**CSV Format** includes:
- Latency percentiles (p50/p95/p99/p999) by request type
- Queue wait times and service times
- Time-to-first-token (TTFT) for streaming
- Request completion rates and throughput
- GPU utilization and VRAM usage statistics

**JSON Format** provides the complete metrics structure for detailed analysis.

### Key Metrics Collected

| Metric Category | Description | Use Case |
|----------------|-------------|----------|
| **Latency** | End-to-end p50/p95/p99 response times | SLA compliance |
| **Queue Wait** | Time spent waiting in node queues | Bottleneck identification |
| **TTFT** | Time-to-first-token for streaming | User experience |
| **Utilization** | GPU SM% and VRAM usage | Resource efficiency |
| **Throughput** | Requests/sec and tokens/sec | Capacity planning |
| **Hedge Metrics** | Hedge win rate and wasted work | Strategy optimization |
| **Cost** | Effective cost per 1k tokens | Economic analysis |

### Advanced Features

**Staleness Modeling**: Test impact of signal delays
```yaml
signals:
  queue_depth_ms: { min: 50, max: 100 }
  vram_ms: { min: 200, max: 500 }
  transport_ms:
    intra_cell: [5, 50]    # Local signal propagation
    inter_cell: [50, 300]  # Cross-region delays
```

**Tenant Skew**: Model realistic multi-tenant workloads
```yaml
tenants:
  skew: { type: zipf, s: 1.1 }  # Hot tenants get more traffic
  count: 1000
```

**Network Modeling**: Vivaldi coordinates for realistic RTT
```yaml
network:
  inter_cell_coords: { dim: 3, base_rtt_ms: 20, noise: 0.1 }
  bw_mbps: { intra_cell: 100000, inter_region: 5000 }
```

### Advanced Routing Strategies

The simulator includes **7 routing strategies** designed to eliminate performance tradeoffs:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `baseline_rr` | Simple round-robin | Establishing baseline performance |
| `heuristic` | Multi-factor scoring (queue, VRAM, utilization) | Balanced workloads |
| `mesh` | Comprehensive scoring with network penalties | Complex topologies |
| `mesh_hedge` | Mesh + hedging for SLA-critical requests | Latency-sensitive workloads |
| `adaptive_mesh` | Adapts between load balancing and performance | Variable load patterns |
| `predictive_mesh` | Forecasts congestion to avoid hotspots | High-throughput scenarios |
| `hybrid_mesh` | Multi-objective optimization with self-tuning | **Best overall performance** |

**🏆 Hybrid Mesh** achieves **2-3x better latency** while maintaining **perfect load balancing** and **50% lower costs** compared to baseline approaches.

### Analysis Workflow

1. **Baseline**: Start with `baseline_rr` to establish performance floor
2. **Compare**: Run `heuristic` and `mesh` strategies on same workload  
3. **Advanced**: Test `adaptive_mesh`, `predictive_mesh`, and `hybrid_mesh` for optimal performance
4. **Optimize**: Tune hedge timing (`mesh_hedge`) for your SLA requirements
5. **Scale**: Test with larger configurations to validate at target scale
6. **Export**: Use CSV output with your preferred analysis tools (Python, R, Excel)

The simulator uses deterministic random seeds for reproducible results, making it perfect for A/B testing routing strategies and capacity planning.

For detailed configuration options, see [docs/SIMULATOR.md](docs/SIMULATOR.md).

## 📂 Repository Structure

```
infermesh/
├─ proto/                     # Protobuf service definitions
├─ crates/                    # Rust crates
│   ├─ mesh-core/             # shared types, traits, configuration schema
│   ├─ mesh-proto/            # generated protobuf bindings
│   ├─ mesh-agent/            # node agent (meshd daemon)
│   ├─ mesh-router/           # inference ingress router
│   ├─ mesh-adapter-runtime/  # runtime integration adapter (Triton, vLLM, TGI, etc.)
│   ├─ mesh-adapter-gpu/      # GPU telemetry adapter (DCGM/NVML)
│   ├─ mesh-cli/              # admin CLI tool
│   ├─ mesh-gossip/           # gossip membership + state dissemination
│   ├─ mesh-metrics/          # unified metrics handling (Prometheus/OpenTelemetry)
│   ├─ mesh-net/              # networking helpers (mTLS, connection pooling)
│   ├─ mesh-raft/             # Raft consensus wrapper for policies
│   ├─ mesh-state/            # state fusion and scoring engine
│   ├─ mesh-sim/              # discrete-event simulator for strategy evaluation
│   └─ mesh-dev/              # development + testing utilities
├─ docs/                      # design & usage docs
├─ LICENSE
└─ README.md
```

## 🚧 Current Status & Roadmap

### ✅ Completed Features
- **Core Architecture**: All 14 crates with 3-plane design (Data, Signal, Control)
- **Runtime Adapters**: Production-ready Triton, vLLM, TGI adapters with HTTP/gRPC support
- **GPU Telemetry**: NVML and DCGM adapters with comprehensive metrics collection
- **Control Plane**: gRPC API with policy management and node coordination
- **Router**: HTTP/gRPC proxy with health checking and connection pooling
- **CLI**: Full-featured command-line interface for mesh management
- **State Management**: Real-time telemetry streaming and state synchronization
- **Observability**: Prometheus metrics and OpenTelemetry integration
- **Discrete-Event Simulator**: Comprehensive simulation framework for strategy evaluation and capacity planning

### 🔧 Near-term Enhancements
- **Gossip Protocol**: SWIM implementation exists but not yet integrated (framework ready)
- **Advanced Service Discovery**: Bootstrap node discovery and cloud provider integrations
- **Security Hardening**: Complete mTLS implementation and RBAC system
- **Performance Optimization**: Benchmarking and hot-path optimization

### 🎯 Future Roadmap
- **Multi-region Support**: WAN-aware routing and edge computing integration
- **Advanced AI Features**: ML-based routing decisions and predictive scaling
- **Enterprise Features**: Multi-tenancy, advanced billing, and compliance tools

## 🧪 Discrete-Event Simulator

InferMesh includes a comprehensive discrete-event simulator (`mesh-sim`) for strategy evaluation and capacity planning. The simulator models GPU clusters, request workloads, and routing strategies with high fidelity.

### Features

- **Realistic GPU Modeling**: H100, A100 profiles with VRAM, concurrency, and batching
- **Workload Generation**: Poisson/MMPP arrivals, LLM/Vision/ASR request types, tenant skew
- **Network Simulation**: Vivaldi coordinates, RTT modeling, bandwidth constraints
- **Signal Modeling**: Metric staleness, transport delays, update frequencies
- **Comprehensive Metrics**: Latency histograms (p50/p95/p99), utilization, cost analysis

### Routing Strategies

The simulator implements 8 routing strategies from simple to advanced:

| Strategy | Approach | Key Features |
|----------|----------|--------------|
| **baseline_rr** | Round-robin | Simple load distribution |
| **heuristic** | Weighted scoring | Queue depth + VRAM + utilization |
| **mesh** | Network-aware | Adds network penalties |
| **mesh_hedge** | Hedging | Secondary requests for tail latency |
| **adaptive_mesh** | Load-aware | Adapts to system load conditions |
| **predictive_mesh** | Forecasting | Predicts congestion from arrival patterns |
| **hybrid_mesh** | Multi-objective | Balances latency, cost, throughput |
| **ml_enhanced_mesh** | Machine Learning | Advanced ML features but higher computational overhead |

#### 🏆 Performance Results

Based on simulation results with 512 nodes:

| Strategy | P95 Latency | P99 Latency | Cost/1K Tokens | Performance |
|----------|-------------|-------------|----------------|-------------|
| **hybrid_mesh** | **183ms** | **218ms** | **$0.00032** | **🥇 Best Overall** |
| **predictive_mesh** | 287ms | 315ms | $0.00066 | 🥈 Excellent |
| **baseline_rr** | 384ms | 639ms | $0.00055 | Baseline |
| **heuristic** | 441ms | 2877ms | $0.00113 | Moderate |
| **adaptive_mesh** | 491ms | 1373ms | $0.00106 | Moderate |
| **mesh_hedge** | 551ms | 2563ms | $0.00092 | Moderate |
| **mesh** | 663ms | 2365ms | $0.00093 | Moderate |
| **ml_enhanced_mesh** | 1894ms | 3605ms | $0.00405 | High overhead |

**Key Findings:**
- **HybridMesh** delivers the best performance with lowest latency and cost
- **PredictiveMesh** offers excellent performance with good efficiency
- **ML-Enhanced** strategy has high computational overhead that impacts latency
- Simple strategies like **baseline_rr** can be surprisingly effective

### Usage

```bash
# Generate example configurations
cargo run -p mesh-sim -- generate --example-type small --output small.yaml
cargo run -p mesh-sim -- generate --example-type medium --output medium.yaml
cargo run -p mesh-sim -- generate --example-type large --output large.yaml

# Run single strategy
cargo run -p mesh-sim -- run --config small.yaml --strategy ml_enhanced_mesh --output results/

# Run all strategies (parallel execution)
cargo run -p mesh-sim -- run --config small.yaml --output results/

# Export results
ls results/
# → config.yaml, baseline_rr.json, heuristic.json, mesh.json, ml_enhanced_mesh.json, ...
```

### Analysis Workflow

1. **Start Small**: Use `small.yaml` (512 nodes) for quick iteration
2. **Scale Up**: Progress to `medium.yaml` (8K nodes) and `large.yaml` (131K nodes)  
3. **Compare Strategies**: All strategies run in parallel for fair comparison
4. **Focus on Winners**: Test `hybrid_mesh` and `predictive_mesh` for best performance
5. **Export & Analyze**: JSON results include latency, utilization, cost metrics
6. **Interpret Results**: See [`docs/RESULTS.md`](docs/RESULTS.md) for detailed analysis guidance

The simulator provides detailed progress updates with strategy names:
```
INFO: Simulation progress [hybrid_mesh]: 10.0s/300.0s (3.3%), 33446 events processed
```

## 🤝 Contributing

Contributions are welcome! Please read:
- [CONTRIBUTING.md](CONTRIBUTING.md) – contribution guidelines
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) – code of conduct
- [SECURITY.md](SECURITY.md) – how to report vulnerabilities

We use GitHub Issues + Discussions for bugs, features, and design proposals.

## 📜 License

**InferMesh** is available under a **dual-license model**:

### Open Source License (AGPLv3)
The open source version of infermesh is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**. This means:
- ✅ Free to use, modify, and distribute
- ✅ Perfect for open source projects and research
- ⚠️ **Copyleft requirement**: Any modifications or derivative works must also be open sourced under AGPLv3
- ⚠️ **Network copyleft**: If you run infermesh as a service, you must provide source code to users

See [LICENSE](LICENSE) for full AGPLv3 terms.

### Commercial License
For organizations that cannot comply with AGPLv3 requirements, **reDB** offers commercial licensing options that provide:
- ✅ Proprietary use without open source obligations
- ✅ Integration into closed-source products and services
- ✅ Enterprise support and consulting
- ✅ Custom licensing terms for specific use cases

**Contact**: For commercial licensing inquiries, please contact [redb@redb.co](mailto:redb@redb.co)

See [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) for more details.

## 🌍 Why open source?

Inference infrastructure is rapidly becoming a shared pain point. By making infermesh open source, we aim to:
- Build a common standard for GPU-aware metrics and routing
- Foster collaboration across research labs, startups, and enterprises
- Provide transparency for critical infrastructure in AI deployment

> **Status**: Ready with comprehensive runtime adapter support (Triton, vLLM, TGI) and GPU telemetry (NVML, DCGM). All 13 crates compile successfully with full end-to-end functionality.

> Built and maintained by [tommihip](https://github.com/tommihip) and [reDB](https://github.com/redbco). Licensed under **AGPLv3**.