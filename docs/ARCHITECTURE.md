# InferMesh Architecture

This document provides a detailed description of the **InferMesh** architecture, its components, and the interfaces used across the system.

---

## High-Level Overview

InferMesh is a **GPU- and network-aware mesh** that provides routing, observability, and control for both **inference** and **training** at scale.  
It is designed for heterogeneous and distributed environments where nodes may differ in GPU capability, interconnect type, network latency, and bandwidth.

The architecture is organized into **two hierarchical layers** with **three cross-cutting planes**:

- **Infrastructure Layer (Underlay)**  
  Provides the distributed mesh fabric: gossip, consensus, topology service, secure networking, node identity, and lifecycle management.

- **Workload Overlay Layer**  
  Provides connectors and integrations for inference runtimes, training frameworks, and GPUs. Hosts routers, schedulers, adapters, and the open strategy interface for routing and topology decisions.

- **Planes**  
  - **Communication/Data Plane**: inference traffic and training collective operations.  
  - **Signal Plane**: runtime, GPU, and network telemetry flowing into fused state.  
  - **Control Plane**: policies, job specs, placements, quotas, events, and administration.

---

## System Architecture (Mermaid)

```mermaid
flowchart TB
  classDef underlay fill:#eef6ff,stroke:#84a9ff,stroke-width:1px,color:#0b2e66;
  classDef overlay fill:#eefcf4,stroke:#8fd19e,stroke-width:1px,color:#145c2e;
  classDef edge fill:#f7ecff,stroke:#c69cf2,stroke-width:1px,color:#442266;
  classDef ext fill:#fff7e6,stroke:#f0b429,stroke-width:1px,color:#6a3d00;

  %% Overlay Layer
  subgraph Overlay["Workload Overlay Layer"]
    direction TB
    R[Inference Router<br/>HTTP/2 gRPC, HTTP, WS/SSE]:::overlay
    SCHED[Training Scheduler<br/>Job API (gRPC/HTTP)]:::overlay
    STRAT[Strategy Host (Inference + Training)<br/>plugins: cdylib/wasm]:::overlay

    subgraph Adapters["Adapters & Connectors"]
      direction LR
      RTAD[Runtime Adapter<br/>Triton/vLLM/TGI/TorchServe/TF/OVMS]:::overlay
      COMMAD[Comm Adapter<br/>NCCL/UCC telemetry]:::overlay
      GPUAD[GPU Adapter<br/>DCGM/NVML/ROCm-SMI]:::overlay
      IOAD[IO Adapter<br/>Datasets/Checkpoint stores]:::overlay
    end
  end

  %% Infrastructure Layer
  subgraph Underlay["Infrastructure Layer (Mesh Fabric)"]
    AG[Mesh Agent (meshd)<br/>state fusion + scoring API]:::underlay
    G[Membership & Gossip (SWIM)]:::underlay
    RF[Consensus (Raft groups)]:::underlay
    TOPO[Topology Service<br/>link coords, RTT, WAN-awareness]:::underlay
    NET[Secure Networking<br/>mTLS, QUIC, pools]:::underlay
    NM[Node Manager<br/>identity, certs, drain/cordon]:::underlay
    POL[Policy Store<br/>placements, quotas, SLOs]:::underlay
  end

  %% Execution / External
  subgraph Exec["Execution & External Systems"]
    TRT[Inference Runtimes<br/>Triton / vLLM / TGI / TorchServe / TF Serving]:::edge
    TRN[Training Runtimes<br/>PyTorch DDP / DeepSpeed / Megatron]:::edge
    NCCL[(Collectives: NCCL/UCC/UCP)]:::edge
    GPU[GPUs + MIG/MPS]:::edge
    DS[(Datasets / Feature Store)]:::ext
    CKPT[(Checkpoint Store: S3/GCS/NFS)]:::ext
    OBS[(Prometheus / Grafana)]:::ext
    OTel[(OpenTelemetry Collector)]:::ext
  end

  %% Overlay <-> Underlay
  R -->|ScoreTargets/Admit| AG
  SCHED -->|Plan/Place/Elastic| AG
  STRAT --> AG

  RTAD -->|ModelStateDelta| AG
  COMMAD -->|CommStatsDelta| AG
  GPUAD -->|GpuStateDelta| AG
  IOAD -->|Dataset/Checkpoint metrics| AG

  AG --> G
  AG --> RF
  AG --> TOPO
  AG --> POL
  AG --> NM
  AG --> NET

  RTAD --> TRT
  COMMAD --> NCCL
  TRN --> NCCL
  GPUAD --> GPU
  IOAD --> DS
  TRN --> CKPT

  R -->|/metrics| OBS
  SCHED -->|/metrics| OBS
  AG -->|/metrics| OBS
  RTAD -->|/metrics| OBS
  COMMAD -->|/metrics| OBS
  GPUAD -->|/metrics| OBS
  R -->|OTLP traces| OTel
  AG -->|OTLP traces| OTel
```

---

## Node Roles

Every node runs a **mesh agent (`meshd`)**. Nodes can be configured with specific roles:

- **Router Nodes**  
  - Accept inference requests from clients.  
  - Invoke routing strategies through the Strategy Host.  
  - Apply admission control and hedged requests.

- **Trainer Nodes**  
  - Run training runtimes (e.g., PyTorch DDP, DeepSpeed, Megatron).  
  - Surface collective timings and communication stats through the Comm Adapter.  
  - Use strategies to adjust topology (ring/tree/hierarchical) or elastic scale.

- **Coordinator Nodes**  
  - Host the Training Scheduler or Router roles.  
  - Participate in Raft for consistent job and policy state.

- **GPU Nodes**  
  - Run inference runtimes or participate in training.  
  - Always include Runtime + GPU Adapters for telemetry.

- **Edge Nodes**  
  - Optional ingress close to users.  
  - Typically run Router + Agent roles to reduce WAN latency.

---

## Planes and Responsibilities

### 1. Communication/Data Plane
- **Inference**: Routers forward requests, apply routing decisions, cancel hedges, stream responses.  
- **Training**: Trainers execute collective ops (all-reduce, all-gather); topology hints (ring/tree/hierarchical) may change mid-flight.  
- **Interfaces**:  
  - **Client ↔ Router**: HTTP/1.1, HTTP/2 (gRPC), WebSockets.  
  - **Router ↔ Runtime**: gRPC/HTTP with mTLS.  
  - **Scheduler ↔ Trainers**: Job API with placement/topology hints.  
  - **Standards**: protobuf/OpenAPI schemas, W3C TraceContext.

### 2. Signal Plane
- **Components**: Adapters (runtime, GPU, comm), mesh agent.  
- **Function**: Gather metrics (queue depth, tokens/sec, gradient timings, GPU VRAM/SM%, ECC, link RTT). Fuse into local scoring APIs.  
- **Interfaces**:  
  - **Adapters ↔ Agent**: gRPC streaming (`ModelStateDelta`, `GpuStateDelta`, `CommStatsDelta`).  
  - **Metrics Export**: Prometheus `/metrics`, optional OTLP traces.

### 3. Control Plane
- **Components**: Agent, Scheduler, Router, Raft group.  
- **Function**: Manage policies, job specs, quotas, tenancy, model pinning, elastic scaling.  
- **Interfaces**:  
  - **Admin ↔ Agent**: gRPC/HTTP API with RBAC.  
  - **Agent ↔ Agent**: Gossip (SWIM) for membership; Raft for strongly consistent writes.  
  - **Events**: Server-streaming gRPC (`SubscribeEvents`).

---

## Open Strategy Interface

InferMesh provides a **pluggable strategy interface** for both inference routing and training topology decisions.  

- **Inputs**: runtime stats, GPU telemetry, network topology, policies, SLOs, historical performance.  
- **Outputs**: ranked targets (inference) or placement/topology plans (training).  
- **Built-ins**: `baseline_rr`, `least_queue`, `hybrid-mesh` (inference); `ring`, `tree`, `hierarchical` (training).  
- **Custom strategies**: users can provide their own via Rust plugins (`cdylib`) or sandboxed WASM (`wasm32-wasi`).  

### Characteristics
- **Unified interface**: built-ins and user-defined strategies follow the same trait definitions.  
- **Safety**: host enforces decision latency budgets (µs for inference, ms for training).  
- **Observability**: decision latency, timeouts, win-rate vs baseline, hedge waste % all exported as metrics.  
- **Experimentation**: A/B testing and shadow mode supported for new strategies.  

---

## Interfaces and Standards

| Interface | Protocol | Standards |
|-----------|----------|-----------|
| Inference APIs | HTTP/2 gRPC, HTTP/1.1, WebSocket | protobuf, OpenAPI, W3C TraceContext |
| Training job control | gRPC + JSON | protobuf, OpenAPI |
| Collectives | In-proc + RDMA fabric | NCCL, UCC/UCP, SHARP, RoCE/IB, TCP |
| Runtime metrics | HTTP `/metrics` | Prometheus, OpenMetrics |
| GPU metrics | HTTP `/metrics` | Prometheus, DCGM schema |
| Tracing | OTLP/gRPC, OTLP/HTTP | OpenTelemetry, W3C TraceContext |
| Membership | Gossip (UDP/TCP) | SWIM-style protocol |
| Consensus | gRPC over TCP | Raft |

> **Label schema** (for all metrics/traces):  
> `model, revision, quant, runtime, node, gpu_uuid, mig_profile, job_id, step, bucket_id, tenant, zone`

---

## Security Model

- **mTLS everywhere** (Router ↔ Agent ↔ GPU/Trainer nodes).  
- **Identity**: X.509 certs (SPIFFE-compatible CA).  
- **RBAC**: enforced on APIs using JWT/OIDC claims.  
- **Isolation**: MIG/MPS enforced at GPU level; quotas and tenancy enforced in control plane.  

---

## Observability

- **Prometheus**: metrics from routers, schedulers, agents, adapters.  
- **OpenTelemetry**: traces for inference requests and training steps/buckets.  
- **Logs**: structured JSON with correlation IDs.  
- **Dashboards**: Grafana dashboards covering inference latency/utilization and training comm efficiency.  

---

## Failure Handling

- **Backpressure**: 429 or admission rejections when queues exceed thresholds.  
- **Quarantine**: automatic node quarantine on ECC/thermal/network degradation.  
- **Partition tolerance**: routers and schedulers degrade to local-only strategies when gossip fails.  
- **Consensus loss**: existing configs persist; new writes blocked until quorum restored.  

---

## Roadmap (high level)

- v0.1: Local-only inference routing, mock training telemetry, Prometheus metrics.  
- v0.2: Gossip membership, scoring API, strategy plugin host, Prometheus/OTel integration.  
- v0.3: Raft policies, control-plane API, router hedging, training topology hints.  
- v0.4: Runtime adapters (Triton, vLLM, TGI) and training comm adapters (NCCL/UCC).  
- v1.0: Multi-region WAN-aware routing, MIG-aware scheduling, full strategy SDK (Rust + WASM).  

---
