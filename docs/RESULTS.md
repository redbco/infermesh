# Simulation Results Analysis Guide

This guide explains how to interpret the results from the InferMesh discrete-event simulator and understand the performance characteristics of different routing strategies.

## Output Files Structure

When you run a simulation, the results are organized as follows:

```
results/
├── config.yaml              # Copy of simulation configuration
├── comparison.csv            # Summary comparison of all strategies
├── baseline_rr.json         # Detailed metrics for baseline round-robin
├── heuristic.json           # Detailed metrics for heuristic strategy
├── mesh.json                # Detailed metrics for mesh strategy
├── mesh_hedge.json          # Detailed metrics for mesh with hedging
├── adaptive_mesh.json       # Detailed metrics for adaptive mesh
├── predictive_mesh.json     # Detailed metrics for predictive mesh
├── hybrid_mesh.json         # Detailed metrics for hybrid mesh
└── ml_enhanced_mesh.json    # Detailed metrics for ML-enhanced mesh
```

## Key Metrics Explained

### Latency Metrics

- **p50_latency**: 50th percentile (median) latency in milliseconds
- **p95_latency**: 95th percentile latency - most requests complete within this time
- **p99_latency**: 99th percentile latency - captures tail latency behavior
- **avg_latency**: Average latency across all requests

**Interpretation:**
- Lower values are better
- P95/P99 are more important than average for user experience
- Large gaps between P95 and P99 indicate inconsistent performance

### Utilization Metrics

- **utilization**: Standard deviation of utilization across nodes (lower = better load balancing)
- **avg_utilization**: Average GPU utilization across all nodes
- **max_utilization**: Peak utilization observed on any node

**Interpretation:**
- Lower utilization variance indicates better load balancing
- Higher average utilization means better resource efficiency
- Max utilization shows if any nodes are overloaded

### Cost Metrics

- **cost_per_1k_tokens**: Cost per 1,000 tokens processed (in dollars)
- **total_cost**: Total simulation cost
- **total_gpu_hours**: Total GPU compute hours consumed

**Interpretation:**
- Lower cost per 1k tokens is better for efficiency
- Reflects both latency and resource utilization
- Important for production cost planning

### Hedging Metrics

- **hedge_win_rate**: Percentage of hedge requests that completed first
- **total_hedges**: Total number of hedge requests sent
- **hedge_wins**: Number of successful hedge completions
- **hedge_timeouts**: Hedge requests that timed out
- **hedge_cancellations**: Hedge requests that were cancelled

**Interpretation:**
- Higher win rate indicates effective hedging
- Balance between latency improvement and resource waste
- Only applies to `mesh_hedge` strategy

### Throughput Metrics

- **total_requests**: Total number of requests processed
- **requests_per_second**: Average request processing rate
- **tokens_per_second**: Average token processing rate

## Strategy Performance Analysis

### Current Results (512 nodes, 300s simulation)

| Strategy | P95 Latency | P99 Latency | Cost/1K Tokens | Utilization Dev | Performance Grade |
|----------|-------------|-------------|----------------|-----------------|-------------------|
| **hybrid_mesh** | **183ms** | **218ms** | **$0.00032** | 0.0044 | **A+** |
| **predictive_mesh** | 287ms | 315ms | $0.00066 | 0.0004 | **A** |
| **baseline_rr** | 384ms | 639ms | $0.00055 | 0.0079 | **B+** |
| **heuristic** | 441ms | 2877ms | $0.00113 | 0.0259 | **B** |
| **adaptive_mesh** | 491ms | 1373ms | $0.00106 | 0.0222 | **B** |
| **mesh_hedge** | 551ms | 2563ms | $0.00092 | 0.0256 | **B-** |
| **mesh** | 663ms | 2365ms | $0.00093 | 0.0251 | **C+** |
| **ml_enhanced_mesh** | 1894ms | 3605ms | $0.00405 | 0.0849 | **D** |

### Strategy Recommendations

#### 🥇 **hybrid_mesh** - Best Overall Choice
- **Strengths**: Lowest latency, lowest cost, excellent consistency
- **Use Case**: Production deployments requiring optimal performance
- **Trade-offs**: Moderate complexity, good balance of all factors

#### 🥈 **predictive_mesh** - Excellent Alternative  
- **Strengths**: Very low latency, excellent consistency, low utilization variance
- **Use Case**: Workloads with predictable patterns, cost-sensitive deployments
- **Trade-offs**: Slightly higher cost than hybrid_mesh

#### 🥉 **baseline_rr** - Reliable Baseline
- **Strengths**: Simple, predictable, good cost efficiency
- **Use Case**: Simple deployments, when complexity is a concern
- **Trade-offs**: Higher latency than advanced strategies

#### ⚠️ **ml_enhanced_mesh** - High Overhead
- **Strengths**: Advanced ML features, continuous learning
- **Use Case**: Research, long-running deployments where learning pays off
- **Trade-offs**: Significant computational overhead impacts latency

## Interpreting JSON Results

Each strategy's JSON file contains detailed metrics:

```json
{
  "latency": {
    "p50": 150.5,
    "p95": 183.2,
    "p99": 218.7,
    "avg": 145.8
  },
  "utilization": {
    "avg": 0.004360958440161376,
    "std_dev": 0.004360958440161376,
    "max": 0.15
  },
  "cost": {
    "total_cost": 0.12345,
    "cost_per_1k_tokens": 0.0003208122371509796,
    "total_gpu_hours": 42.67
  },
  "throughput": {
    "total_requests": 15000,
    "requests_per_second": 50.0,
    "tokens_per_second": 125000.0
  },
  "hedge_metrics": {
    "total_hedges": 1500,
    "hedge_wins": 267,
    "hedge_win_rate": 0.17767634140039618,
    "hedge_timeouts": 45,
    "hedge_cancellations": 1188
  }
}
```

## Analysis Workflow

### 1. Quick Performance Check
```bash
# Look at the comparison CSV for overview
cat results/comparison.csv | column -t -s,
```

### 2. Detailed Strategy Analysis
```bash
# Examine top performers in detail
cat results/hybrid_mesh.json | jq '.latency'
cat results/predictive_mesh.json | jq '.cost'
```

### 3. Scaling Analysis
```bash
# Compare across different scales
cargo run -p mesh-sim -- run --config small.yaml --output small_results/
cargo run -p mesh-sim -- run --config medium.yaml --output medium_results/
cargo run -p mesh-sim -- run --config large.yaml --output large_results/

# Compare results
diff small_results/comparison.csv medium_results/comparison.csv
```

### 4. Strategy Selection Guide

**For Production Use:**
1. Start with `hybrid_mesh` - best overall performance
2. Test `predictive_mesh` if you have predictable workloads
3. Use `baseline_rr` for simple, reliable deployments

**For Research/Development:**
1. Compare all strategies to understand trade-offs
2. Focus on `ml_enhanced_mesh` for long-term learning scenarios
3. Experiment with different scales and workload patterns

**For Cost Optimization:**
1. Compare `cost_per_1k_tokens` across strategies
2. Consider utilization efficiency vs. latency trade-offs
3. Factor in operational complexity costs

## Common Patterns

### High Latency Causes
- **Queue buildup**: Check utilization variance
- **Poor load balancing**: Look for high std_dev in utilization
- **Computational overhead**: ML strategies may have high decision costs
- **Network penalties**: Inter-cell routing overhead

### Cost Optimization
- **Lower utilization variance** → Better resource efficiency
- **Faster routing decisions** → Lower computational overhead
- **Fewer hedge requests** → Less wasted work
- **Better load prediction** → More efficient resource allocation

### Scaling Behavior
- **Small scale (512 nodes)**: Simple strategies often competitive
- **Medium scale (8K nodes)**: Advanced strategies show benefits
- **Large scale (131K nodes)**: Network-aware routing becomes critical

## Troubleshooting

### Unexpected Results
1. **Check configuration**: Verify workload parameters match expectations
2. **Examine utilization**: High variance indicates load balancing issues
3. **Review hedge metrics**: Excessive hedging may indicate routing problems
4. **Compare scales**: Some strategies perform differently at different scales

### Performance Issues
1. **High P99 latency**: Look for queue buildup or poor load balancing
2. **High cost**: Check for inefficient resource utilization
3. **Low throughput**: Examine bottlenecks in routing or processing
4. **Inconsistent results**: Verify RNG seed for reproducibility

## Advanced Analysis

### Custom Metrics Extraction
```bash
# Extract specific metrics across all strategies
for file in results/*.json; do
  strategy=$(basename "$file" .json)
  p95=$(jq -r '.latency.p95' "$file")
  cost=$(jq -r '.cost.cost_per_1k_tokens' "$file")
  echo "$strategy,$p95,$cost"
done
```

### Comparative Analysis
```python
import json
import pandas as pd

# Load all strategy results
strategies = {}
for strategy in ['hybrid_mesh', 'predictive_mesh', 'baseline_rr']:
    with open(f'results/{strategy}.json') as f:
        strategies[strategy] = json.load(f)

# Create comparison DataFrame
df = pd.DataFrame({
    strategy: {
        'p95_latency': data['latency']['p95'],
        'cost_per_1k': data['cost']['cost_per_1k_tokens'],
        'utilization': data['utilization']['avg']
    }
    for strategy, data in strategies.items()
}).T

print(df)
```

This analysis framework helps you understand routing strategy performance and make informed decisions for your InferMesh deployment.
