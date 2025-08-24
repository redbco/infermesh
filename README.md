# InferMesh

**InferMesh** is a **GPU-aware inference mesh** designed for large-scale AI serving.  
It provides a distributed control plane, GPU- and network-aware routing, and standardized observability across heterogeneous environments.

---

## ✨ What is InferMesh?

Modern AI inference at scale faces two hard problems:

1. **Observability** – understanding GPU health, utilization, queue depths, and tail latency across a large fleet.
2. **Utilization** – squeezing maximum throughput out of expensive GPUs without breaking SLAs.

**InferMesh** solves this by introducing a *mesh abstraction layer* above Kubernetes, Slurm, or bare metal.  
It coordinates nodes, routes requests based on live GPU + network state, and exposes a distributed API for control and monitoring.

---

## 🎯 Who is this for?

infermesh is targeted at organizations and teams that run **AI inference at meaningful scale** and face challenges in GPU efficiency, reliability, and observability. Typical users include:

- **AI infrastructure teams** running fleets of hundreds or thousands of GPUs who need better utilization and reduced cost per inference.  
- **Cloud providers and platform teams** offering inference services and wanting to integrate multi-tenant, GPU-aware routing.  
- **Enterprises** deploying private or hybrid AI inference clusters with strict SLAs and compliance requirements.  
- **Research labs** coordinating shared GPU clusters across regions or institutions, needing fair scheduling and transparency.  
- **Startups scaling inference-heavy products** (chatbots, copilots, speech/image services) who need production-grade observability and routing without reinventing infra.  

If you are running **<100 GPUs in a single cluster**, Kubernetes or Triton alone may be sufficient.  
If you are scaling **>500 GPUs across multiple clusters/regions**, infermesh delivers outsized ROI by improving utilization, simplifying management, and reducing operational overhead.

---

## 🔧 Core Features

- **GPU-aware routing**  
  Routes requests using live signals: batch fullness, queue depth, VRAM headroom, MIG/MPS profile, and network cost.

- **Distributed control plane**  
  Each node runs a `meshd` agent for membership, state gossip, and Raft-based consensus for policies/placements.

- **Flexible node roles**  
  - **Router Nodes** – accept inference traffic, make routing decisions.  
  - **GPU Nodes** – run runtimes (Triton, vLLM, TGI, TorchServe, …) + GPU telemetry.  
  - **Edge Nodes** – optional ingress points colocated with users.  

- **Observability built-in**  
  - Prometheus metrics for queue depths, latency, throughput, GPU stats.  
  - OpenTelemetry tracing for end-to-end request visibility.  
  - Consistent labels across runtimes and GPUs.

- **Pluggable runtimes**  
  **Production-ready adapters**: Triton, vLLM, TGI with comprehensive HTTP/gRPC integration.  
  **Framework ready**: TorchServe, TF Serving, OVMS adapters with extensible architecture.  
  Standardizes runtime metrics + model control into a uniform contract.

- **GPU telemetry**  
  **NVML adapter**: Complete GPU monitoring with utilization, memory, temperature, and power metrics.  
  **DCGM adapter**: Enterprise-grade GPU monitoring with comprehensive field group support.  
  **ROCm support**: Framework ready for AMD GPU integration.

- **Network-aware scaling**  
  Accounts for WAN latency/bandwidth when routing across regions or edge sites.

---

## 📐 Architecture Overview

The mesh consists of three cooperating layers:

- **Data Plane** – routers forward inference requests to the best target GPU node.  
- **Signal Plane** – agents collect runtime and GPU metrics, gossip them, and provide a local scoring API.  
- **Control Plane** – strongly consistent policies (model pinning, SLO classes, quotas) managed via Raft and exposed as a gRPC/HTTP API.

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
│   └─ mesh-dev/              # development + testing utilities
├─ docs/                      # design & usage docs
├─ LICENSE
└─ README.md
```

## 🚧 Current Status & Roadmap

### ✅ Completed Features
- **Core Architecture**: All 13 crates with 3-plane design (Data, Signal, Control)
- **Runtime Adapters**: Production-ready Triton, vLLM, TGI adapters with HTTP/gRPC support
- **GPU Telemetry**: NVML and DCGM adapters with comprehensive metrics collection
- **Control Plane**: gRPC API with policy management and node coordination
- **Router**: HTTP/gRPC proxy with health checking and connection pooling
- **CLI**: Full-featured command-line interface for mesh management
- **State Management**: Real-time telemetry streaming and state synchronization
- **Observability**: Prometheus metrics and OpenTelemetry integration

### 🔧 Near-term Enhancements
- **Gossip Protocol**: SWIM implementation exists but not yet integrated (framework ready)
- **Advanced Service Discovery**: Bootstrap node discovery and cloud provider integrations
- **Security Hardening**: Complete mTLS implementation and RBAC system
- **Performance Optimization**: Benchmarking and hot-path optimization

### 🎯 Future Roadmap
- **Multi-region Support**: WAN-aware routing and edge computing integration
- **Advanced AI Features**: ML-based routing decisions and predictive scaling
- **Enterprise Features**: Multi-tenancy, advanced billing, and compliance tools

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