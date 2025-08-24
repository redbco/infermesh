# infermesh Dashboards

This document provides example PromQL queries and Grafana panel suggestions for monitoring an **infermesh** deployment.  
The focus is on GPU utilization, runtime metrics, request latency, and overall fleet health.

---

## Core Metrics

infermesh exposes metrics via Prometheus at `/metrics` on Routers, Agents, and Adapters.  
GPU metrics follow the NVIDIA DCGM schema; runtime metrics are normalized across supported runtimes.

**Key Labels (normalized across all metrics):**
```
model, revision, quant, runtime, node, gpu_uuid, mig_profile, tenant, zone
```

---

## Example Dashboards

### 1. Inference Throughput
**PromQL**
```promql
sum(rate(infermesh_inferences_total[1m])) by (model, revision)
```

**Description**: Total inferences per second per model version.  
**Panel Type**: Time series line chart.

---

### 2. Token Generation Rate (LLMs)
**PromQL**
```promql
sum(rate(infermesh_tokens_generated_total[30s])) by (model, revision)
```

**Description**: Tokens generated per second for LLMs.  
**Panel Type**: Time series line chart.

---

### 3. Queue Depth per Model
**PromQL**
```promql
avg(infermesh_model_queue_depth) by (model, node)
```

**Description**: Average request queue depth per model per node.  
**Panel Type**: Heatmap or table.

---

### 4. GPU Utilization
**PromQL**
```promql
avg(DCGM_FI_DEV_GPU_UTIL{gpu_uuid!=""}) by (gpu_uuid, node)
```

**Description**: Streaming multiprocessor (SM) utilization per GPU.  
**Panel Type**: Gauge or line chart.

---

### 5. VRAM Headroom
**PromQL**
```promql
(avg(DCGM_FI_DEV_FB_USED{gpu_uuid!=""}) by (gpu_uuid))
/
(avg(DCGM_FI_DEV_FB_TOTAL{gpu_uuid!=""}) by (gpu_uuid))
```

**Description**: VRAM usage ratio per GPU.  
**Panel Type**: Gauge.

---

### 6. Latency Distribution
**PromQL**
```promql
histogram_quantile(0.95, sum(rate(infermesh_inference_latency_bucket[5m])) by (le, model))
```

**Description**: 95th percentile inference latency per model.  
**Panel Type**: Time series.

---

### 7. Router Target Choices
**PromQL**
```promql
count by (target_node) (rate(infermesh_router_target_chosen_total[1m]))
```

**Description**: How often each node is chosen as the inference target.  
**Panel Type**: Bar chart.

---

### 8. Admission Control (429s)
**PromQL**
```promql
sum(rate(infermesh_requests_rejected_total[5m])) by (reason, model)
```

**Description**: Requests rejected due to backpressure or policy.  
**Panel Type**: Stacked bar chart.

---

### 9. Node Health & Membership
**PromQL**
```promql
max(infermesh_node_healthy) by (node)
```

**Description**: Shows 1 if node is healthy, 0 if unhealthy (as seen by gossip).  
**Panel Type**: Single stat or table.

---

### 10. ECC Errors / Thermal Throttling
**PromQL**
```promql
sum(rate(DCGM_FI_DEV_ECC_DBE_AGG_TOTAL[5m])) by (gpu_uuid)
```

**Description**: ECC double-bit errors per GPU over time.  
**Panel Type**: Alert table or single stat with thresholds.

---

## Suggested Dashboards Layout

1. **Overview**
   - Fleet-wide GPU utilization
   - Total throughput (inferences/sec)
   - Average latency (p50, p95, p99)

2. **Per-Model Drilldown**
   - QPS, tokens/sec
   - Queue depth
   - Latency histogram
   - Error rates

3. **GPU Health**
   - Utilization & VRAM headroom
   - ECC error alerts
   - Thermal throttling

4. **Router Behavior**
   - Target choice distribution
   - Admission control stats
   - Hedge/cancel effectiveness

---

## Alerts (examples)

- **High queue depth**  
  ```promql
  avg(infermesh_model_queue_depth{model="gpt-7b"}) > 50
  ```

- **GPU near OOM**  
  ```promql
  (DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL) > 0.9
  ```

- **p95 latency breach**  
  ```promql
  histogram_quantile(0.95, rate(infermesh_inference_latency_bucket[5m])) > 2000
  ```

- **Node down**  
  ```promql
  max(infermesh_node_healthy) by (node) == 0
  ```

---

## References

- [Prometheus Docs](https://prometheus.io/docs/introduction/overview/)  
- [Grafana Dashboards](https://grafana.com/docs/)  
- [NVIDIA DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter)

---
