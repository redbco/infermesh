# infermesh Architecture

This document provides a detailed description of the **infermesh** architecture, its components, and the interfaces used across the system.

---

## High-Level Overview

infermesh is a **GPU-aware inference mesh** that provides routing, observability, and control for large-scale AI inference workloads.  
It is designed for heterogeneous and distributed environments where nodes may differ in GPU capability, network latency, and bandwidth.

The mesh is composed of three cooperating planes:

1. **Data Plane** – Handles inference traffic routing and request forwarding.  
2. **Signal Plane** – Collects runtime and GPU telemetry, fuses state, and provides scoring APIs.  
3. **Control Plane** – Maintains policies, placements, and exposes a distributed management API.

![Architecture Diagram](assets/architecture.png) <!-- placeholder for your diagram -->

---

## Node Roles

Every node runs a **mesh agent (`meshd`)**. Nodes can be configured with specific roles:

- **Router Nodes**
  - Accept inference requests from clients (HTTP/gRPC/streaming).
  - Query local `meshd` for GPU/network-aware scoring.
  - Apply admission control and hedged requests to meet SLOs.

- **GPU Nodes**
  - Run inference runtimes (e.g., Triton, vLLM, TGI, TorchServe, TF Serving, OVMS).
  - Include a **Runtime Adapter** for control (load/unload) and metric collection.
  - Include a **GPU Telemetry Adapter** for DCGM/NVIDIA metrics.

- **Edge Nodes**
  - Optional ingress close to users.
  - Can run Router + Agent roles to reduce WAN latency.

---

## Planes and Responsibilities

### 1. Data Plane
- **Components**: Routers, GPU Nodes.  
- **Function**: Forward requests, apply routing decisions, cancel hedged requests, stream responses.  
- **Interfaces**:  
  - **Client ↔ Router**: HTTP/1.1, HTTP/2 (gRPC), WebSockets (streaming).  
  - **Router ↔ GPU Node**: gRPC/HTTP; mTLS-secured.  
  - **Standards**: OpenAPI/protobuf schemas, W3C TraceContext for propagation.

### 2. Signal Plane
- **Components**: Runtime Adapters, GPU Adapters, meshd.  
- **Function**: Gather runtime metrics (QPS, latency, queue depth, tokens/sec) and GPU telemetry (SM util, VRAM usage, ECC, MIG profile).  
- **Interfaces**:  
  - **Runtime ↔ Adapter**: Runtime APIs (Triton gRPC, TorchServe mgmt, etc.).  
  - **Adapter ↔ meshd**: gRPC streaming with `ModelStateDelta` and `GpuStateDelta`.  
  - **Metrics Export**: Prometheus `/metrics` endpoints; OpenTelemetry optional.

### 3. Control Plane
- **Components**: meshd (with Raft consensus).  
- **Function**: Manage policies, model placement, quotas, tenant isolation, eventing.  
- **Interfaces**:  
  - **Admin ↔ meshd**: gRPC + JSON/HTTP gateway.  
  - **meshd ↔ meshd**: Gossip (SWIM) for membership; Raft for strongly-consistent writes.  
  - **Events**: Server-streaming gRPC (`SubscribeEvents`).

---

## Interfaces and Standards

| Interface | Protocol | Standards |
|-----------|----------|-----------|
| Inference (client ↔ router) | HTTP/2 gRPC, HTTP/1.1, WebSocket | OpenAPI, protobuf, W3C TraceContext |
| Runtime metrics | HTTP `/metrics` | Prometheus, OpenMetrics |
| GPU metrics | HTTP `/metrics` | Prometheus, DCGM schema |
| Tracing | OTLP/gRPC, OTLP/HTTP | OpenTelemetry, W3C TraceContext |
| Control plane API | gRPC + JSON | protobuf, OpenAPI |
| Membership | Gossip (UDP/TCP) | SWIM-style protocol |
| Consensus | gRPC over TCP | Raft (tikv/raft) |

---

## Data Flow (Example: LLM Request)

1. Client sends a request (e.g., generate tokens) → Router.  
2. Router queries local meshd → `ScoreTargets` with model, SLA, tokens.  
3. meshd returns ranked GPU nodes, factoring:  
   - queue depth / service rate  
   - VRAM pressure  
   - MIG compatibility  
   - recent p95 latency  
   - network RTT/bandwidth cost  
4. Router forwards to best candidate; schedules a hedge if latency budget is exceeded.  
5. GPU node executes via runtime (Triton/vLLM/TGI) and streams results back.  
6. Router reports outcome to meshd → metrics updated → gossip distributes state.  
7. Prometheus/Grafana/OTel collectors can scrape/export metrics externally.

---

## Security Model

- **mTLS everywhere** (Router ↔ meshd ↔ GPU nodes).  
- **Node identity**: X.509 certificates issued by the mesh’s internal CA (SPIFFE-compatible).  
- **RBAC**: enforced in the control-plane API (JWT/OIDC claims).  
- **Isolation**: MIG profiles or MPS enforced at GPU level.  

---

## Observability

- **Prometheus Metrics**: Exposed at `/metrics` by Router, Agent, Adapters.  
- **Tracing**: Spans for `router.choose`, `queue_wait`, `compute`, exported via OpenTelemetry.  
- **Logs**: Structured JSON logs with tracing correlation IDs.  
- **Dashboards**: Example Grafana dashboards provided in [DASHBOARDS.md](DASHBOARDS.md).

---

## Failure Handling

- **Backpressure**: Admission control returns 429 when queues exceed thresholds.  
- **Quarantine**: Automatic node quarantine on ECC/thermal errors.  
- **Partition tolerance**: Routers degrade to “local-only” candidates when gossip fails.  
- **Consensus failure**: Existing configs persist; new writes blocked until quorum returns.

---

## Roadmap (high level)

- v0.1: Mock runtimes, local-only routing, Prometheus metrics.  
- v0.2: Gossip membership, scoring API, Prometheus/OTel integration.  
- v0.3: Raft policies, control-plane API, router hedging.  
- v0.4: Runtime control adapters (Triton, vLLM, TGI).  
- v1.0: Multi-region WAN probes, MIG awareness, security hardening.

---
