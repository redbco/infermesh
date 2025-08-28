# Discrete-Event Simulator for InferMesh

A discrete-event simulator for InferMesh that lets you vary scale (nodes/GPUs), request mixes, and mesh algorithms, then measure the impact on p50/p95/p99 latency, GPU utilization, and cost.

## Goals & scope

- Compare routing strategies: 8 strategies from simple round-robin to advanced ML-based routing with real-time learning and optimization.
- Scale from hundreds → 1M nodes via cells/shards.
- Vary workloads: token length distributions (LLM), image/ASR mix, burstiness, tenant skew.
- Vary nodes: GPU types/MIG, runtime throughput, batching windows, network topology.
- Account for decision cost: per-request routing compute + signal fusion delay.

## Architecture (Rust crate: mesh-sim)

Single toolchain, minimal deps, deterministic runs.

```
crates/mesh-sim/
├─ src/
│  ├─ engine.rs          # discrete-event engine (time-ordered queue)
│  ├─ world.rs           # topology: cells, nodes, links
│  ├─ gpu.rs             # GPU/MIG, VRAM, service model
│  ├─ runtime.rs         # Triton-like batching & concurrency model
│  ├─ workload.rs        # request generators (LLM, vision, ASR)
│  ├─ router.rs          # strategies (baseline, heuristic, mesh, mesh+hedge)
│  ├─ signals.rs         # metric streams + staleness model
│  ├─ net.rs             # latency/bw model, Vivaldi coords
│  ├─ metrics.rs         # histograms, counters, percentiles
│  └─ main.rs            # CLI: run experiments from YAML
└─ examples/
   ├─ small.yaml
   ├─ medium.yaml
   └─ million.yaml
```

### Discrete-event engine

- State is advanced by events: Arrival, Dispatch, BatchClose, ServiceDone, SignalUpdate, HedgeFire, Cancel.
- Min-heap priority queue keyed by simulated time (f64 ms).
- RNG with fixed seed (SmallRng) for reproducibility.

```rust
enum Event {
    Arrival(Request),
    Dispatch(RequestId, Target),
    BatchClose(NodeId, ModelId),
    ServiceDone(RequestId, NodeId),
    HedgeFire(RequestId),
    Cancel(RequestId, Target),
    SignalUpdate(NodeId),
}
```

## Core models

### Workload (configurable)

- Request types: LLM, Vision, ASR (each with service curve).
- LLM: input tokens ~ lognormal; output tokens ~ lognormal/Poisson; throughput in tokens/s.
- Burstiness: arrivals via Markov-Modulated Poisson Process (MMPP) or simple Poisson; support tenant skew (Zipf).
- Mix: percentages per type; per-tenant SLAs (latency vs throughput).

Example YAML:

```yaml
workload:
  duration_s: 600
  arrival: { type: mmpp, states: 3, rates_rps: [200, 800, 1500], dwell_s: [30, 30, 10] }
  mix: { llm: 0.7, vision: 0.2, asr: 0.1 }
  llm:
    in_tokens:  { dist: lognormal, mu: 4.0, sigma: 0.7 }
    out_tokens: { dist: lognormal, mu: 5.0, sigma: 0.8 }
  tenants:
    skew: { type: zipf, s: 1.1 } # hot tenants
```

### Nodes/GPUs
- Types: A100/H100/L40 etc; each with tokens_per_s, concurrency, vram_total, batch_window_ms, kv_cache_gb_per_req.
- MIG: slice profiles with capacity fractions.
- Runtime: batching closes after batch_window_ms or max_batch_size.
- Queue: per-model queue; service ≈ G/G/k (we’ll simulate rather than use closed-form).

Example YAML:

```yaml
topology:
  cells: 32
  nodes_per_cell: 1024
  gpu_profiles:
    - name: H100-80G
      tokens_per_s: 240000
      concurrency: 16
      vram_total_gb: 80
      batch_window_ms: 8
      kv_cache_gb_per_req: 1.2
  mig:
    enable: true
    profiles:
      - name: 1g.10gb
        fraction: 0.125
        tokens_per_s: 30000
        concurrency: 2
```

### Network

- Coordinates: Vivaldi-style 3D to synthesize RTT between cells; add jitter.
- Intra-cell RTT distribution (rack/AZ).
- Bandwidth limits for x-region cross-traffic (affects streaming start time if needed).

```yaml
network:
  intra_cell_rtt_ms: { dist: normal, mean: 0.3, std: 0.1 }
  inter_cell_coords: { dim: 3, base_rtt_ms: 20, noise: 0.1 }
  bw_mbps: { intra_cell: 100000, inter_region: 5000 }
```

### Signals & staleness
- Update cadence per metric with jitter (e.g., queue depth every 50–100ms, VRAM every 200–500ms, p95 every 1–2s).
- Transport delay: piggyback on gossip; configurable 5–50ms intra-cell, 50–500ms inter-cell.
- Downsampling: router reads latest snapshot; can simulate stale reads.

```yaml
signals:
  queue_depth_ms: { min: 50, max: 100 }
  vram_ms: { min: 200, max: 500 }
  p95_ms: { min: 1000, max: 2000 }
  transport_ms: { intra_cell: [5, 50], inter_cell: [50, 300] }
```

## Routing strategies (plug-ins)

Implement as trait RouterStrategy.
- baseline_rr: round-robin among in-cell nodes hosting model.
- least_queue: choose min(queue_depth) locally.
- heuristic: score = α·work_left + β·vram_pressure + γ·recent_p95.
- mesh: full score (adds MIG penalty + net_penalty + cold_penalty).
- mesh_hedge: schedule secondary after α·latency_budget if no first-byte.
- mesh_stale: same as mesh, but restricts to last N ms signals → quantify harm of staleness.

Each strategy includes decision_cost_us (compute overhead) to model “thinking vs sending”.

```rust
pub trait RouterStrategy {
    fn choose(&mut self, ctx: &RequestCtx, view: &StateView) -> Target;
    fn decision_cost_us(&self) -> u64 { 50 } // e.g., mesh: 50–150 µs
}
```

## Routing Strategies

The simulator implements 8 routing strategies with varying complexity and performance characteristics:

### Basic Strategies
- **baseline_rr**: Simple round-robin distribution across available nodes
- **heuristic**: Weighted scoring based on queue depth, VRAM usage, and utilization
- **mesh**: Network-aware routing with inter-cell penalties

### Advanced Strategies  
- **mesh_hedge**: Hedging strategy that sends secondary requests for tail latency reduction
- **adaptive_mesh**: Load-aware adaptation that switches between performance and load balancing
- **predictive_mesh**: Uses arrival history to predict and avoid future congestion
- **hybrid_mesh**: Multi-objective optimization balancing latency, cost, and throughput
- **ml_enhanced_mesh**: Machine learning approach with real-time weight optimization

### Performance Results (512 nodes)

| Strategy | P95 Latency | P99 Latency | Cost/1K Tokens | Recommendation |
|----------|-------------|-------------|----------------|----------------|
| **hybrid_mesh** | **183ms** | **218ms** | **$0.00032** | 🥇 **Best choice** |
| **predictive_mesh** | 287ms | 315ms | $0.00066 | 🥈 **Excellent** |
| **baseline_rr** | 384ms | 639ms | $0.00055 | 🥉 **Good baseline** |
| **heuristic** | 441ms | 2877ms | $0.00113 | Moderate |
| **adaptive_mesh** | 491ms | 1373ms | $0.00106 | Moderate |
| **mesh_hedge** | 551ms | 2563ms | $0.00092 | Moderate |
| **mesh** | 663ms | 2365ms | $0.00093 | Moderate |
| **ml_enhanced_mesh** | 1894ms | 3605ms | $0.00405 | High overhead |

**Key Insights:**
- **HybridMesh** delivers optimal performance through balanced multi-objective optimization
- **PredictiveMesh** excels with proactive congestion avoidance
- **ML-Enhanced** strategy has significant computational overhead that impacts latency
- Simple strategies can be surprisingly effective for many workloads

## Metrics collected

- Latency histograms (end-to-end, queue wait, service, time-to-first-token).
- p50/p95/p99, time-to-first-token for streaming LLM.
- GPU utilization (SM%), VRAM headroom percentiles.
- Throughput (req/s, tokens/s) per model and global.
- Abort/hedge: hedge rate, wasted work %, cancel effectiveness.
- Staleness impact: delta in p95 vs fresh signals.
- Cost metrics: GPUs × $/mo, effective cost per 1k tokens.

Emit CSV/Parquet plus a summary table.

## Experiments to run (matrix)

1. Scale-up: cells ∈ {1, 8, 32, 128}, nodes/cell ∈ {128, 1024}, GPUs ∈ {H100, mix}.
2. Burstiness: steady Poisson vs MMPP (Black Friday spikes).
3. Staleness: transport delays from 10ms → 300ms; cadence stretched.
4. Hedging α: 0.2, 0.35, 0.5 of SLA; measure p99 + wasted work.
5. MIG: 0%, 50% sliced; measure packing efficiency & VRAM OOMs.
6. Inter-cell routing: local-only vs summaries vs summaries+DHT directory (for non-resident models).
7. Decision cost: add 25–250µs per decision; find break-even where “thinking” hurts.

## Usage Examples

### Generate Configurations

```bash
# Generate example configurations
cargo run -p mesh-sim -- generate --example-type small --output small.yaml
cargo run -p mesh-sim -- generate --example-type medium --output medium.yaml  
cargo run -p mesh-sim -- generate --example-type large --output large.yaml
```

### Run Simulations

```bash
# Run single strategy
cargo run -p mesh-sim -- run --config small.yaml --strategy hybrid_mesh --output results/

# Run all strategies in parallel (recommended)
cargo run -p mesh-sim -- run --config small.yaml --output results/

# Run specific strategies
cargo run -p mesh-sim -- run --config small.yaml --strategy predictive_mesh --output results/
```

### Example Configurations

- **small.yaml**: 512 nodes (1 cell × 512 nodes) - Quick testing
- **medium.yaml**: 8,192 nodes (8 cells × 1,024 nodes) - Moderate scale
- **large.yaml**: 131,072 nodes (128 cells × 1,024 nodes) - Large scale

### Progress Monitoring

The simulator provides real-time progress updates:
```
INFO: Starting simulation [hybrid_mesh]: 512 nodes, 300.0s duration
INFO: Simulation progress [hybrid_mesh]: 10.0s/300.0s (3.3%), 33446 events processed
INFO: Simulation progress [hybrid_mesh]: 20.0s/300.0s (6.7%), 67892 events processed
```

## Output & analysis

- CLI flags to run sweeps and write a directory per run with config.yaml, metrics.csv, summary.json, RNG seed.
- Results include per-strategy JSON files with detailed metrics
- CSV comparison files for easy analysis
- See `docs/RESULTS.md` for detailed interpretation guidance

## Implementation hints (Rust)

- Use binary-heap-plus or std::collections::BinaryHeap with reverse ordering for the event queue.
- hdrhistogram crate for latency distributions.
- rand + rand_distr for MMPP/LogNormal/Zipf.
- Keep structs POD-like; pre-allocate per-node vectors to avoid allocation churn.
- Record per-event counters to validate O(1) / O(log n) paths.

Example skeleton for the engine:

```rust
pub struct Sim {
    now_ms: f64,
    events: BinaryHeap<SimEvent>,
    world: World,
    metrics: Metrics,
    router: Box<dyn RouterStrategy>,
}

impl Sim {
    pub fn run(&mut self, until_ms: f64) {
        while let Some(mut ev) = self.events.pop() {
            if ev.at > until_ms { break; }
            self.now_ms = ev.at;
            self.handle(ev.kind);
        }
    }
}
```

Batching model (very simplified):

```rust
fn on_arrival(node: &mut Node, req: Request) {
    node.queue.push(req);
    if node.batch_open_since.is_none() {
        node.batch_open_since = Some(now_ms);
        schedule(BatchClose(node.id), now_ms + node.batch_window_ms);
    }
}

fn on_batch_close(node: &mut Node) {
    let batch = node.queue.pop_up_to(node.max_batch());
    let service_time_ms = batch.service_time_ms(); // tokens / tokens_per_s * 1000
    schedule(ServiceDone(batch.req_ids), now_ms + service_time_ms);
    node.batch_open_since = None;
}
```

## Validating the simulator

- Sanity checks against queueing theory: under light load, latency ≈ network + service; under heavy load, queue wait increases roughly as expected for G/G/1.
- Compare batching throughput to runtime-published limits (tokens/s).
- Calibrate network RTTs to your real measurements if available.

## What “measurements” to publish

- Baseline vs mesh p95/p99 on the same workload & topology.
- Utilization gains (SM% and tokens/s).
- Cost per 1k tokens reduction at given SLA.
- Sensitivity to staleness and decision cost—this addresses the “is the extra computation worth it?” concern directly.
- Hedging waste vs benefit curves to show the sweet spot.

⸻

Example minimal examples/small.yaml

```yaml
seed: 42
duration_s: 300

workload:
  arrival: { type: poisson, rps: 800 }
  mix: { llm: 1.0 }
  llm:
    in_tokens:  { dist: lognormal, mu: 3.8, sigma: 0.6 }
    out_tokens: { dist: lognormal, mu: 4.6, sigma: 0.7 }

topology:
  cells: 4
  nodes_per_cell: 128
  gpu_profiles:
    - name: H100-80G
      tokens_per_s: 240000
      concurrency: 16
      vram_total_gb: 80
      batch_window_ms: 8
      kv_cache_gb_per_req: 1.0

network:
  intra_cell_rtt_ms: { dist: normal, mean: 0.5, std: 0.1 }
  inter_cell_coords: { dim: 3, base_rtt_ms: 25, noise: 0.1 }
  bw_mbps: { intra_cell: 50000, inter_region: 5000 }

signals:
  queue_depth_ms: { min: 50, max: 100 }
  vram_ms: { min: 200, max: 400 }
  p95_ms: { min: 1000, max: 1500 }
  transport_ms: { intra_cell: [5, 25], inter_cell: [50, 200] }

strategies:
  - baseline_rr
  - heuristic
  - mesh
  - mesh_hedge
```
