//! Conversions between mesh-core types and protobuf types
//!
//! This module provides bidirectional conversions between the rich types
//! defined in mesh-core and the protobuf message types used for network
//! communication.

use crate::{state::v1 as proto_state, scoring::v1 as proto_scoring, timestamp};
use mesh_core::{GpuState, GpuStateDelta, Labels, ModelState, ModelStateDelta, SloClass};

// Conversions for Labels
impl From<Labels> for proto_state::Labels {
    fn from(labels: Labels) -> Self {
        Self {
            model: labels.model,
            revision: labels.revision,
            quant: labels.quant.unwrap_or_default(),
            runtime: labels.runtime,
            node: labels.node,
            gpu_uuid: labels.gpu_uuid.unwrap_or_default(),
            mig_profile: labels.mig_profile.unwrap_or_default(),
            tenant: labels.tenant.unwrap_or_default(),
            zone: labels.zone.unwrap_or_default(),
            custom: labels.custom,
        }
    }
}

impl From<proto_state::Labels> for Labels {
    fn from(proto: proto_state::Labels) -> Self {
        Self {
            model: proto.model,
            revision: proto.revision,
            quant: if proto.quant.is_empty() { None } else { Some(proto.quant) },
            runtime: proto.runtime,
            node: proto.node,
            gpu_uuid: if proto.gpu_uuid.is_empty() { None } else { Some(proto.gpu_uuid) },
            mig_profile: if proto.mig_profile.is_empty() { None } else { Some(proto.mig_profile) },
            tenant: if proto.tenant.is_empty() { None } else { Some(proto.tenant) },
            zone: if proto.zone.is_empty() { None } else { Some(proto.zone) },
            custom: proto.custom,
        }
    }
}

// Conversions for ModelState
impl From<ModelState> for proto_state::ModelState {
    fn from(state: ModelState) -> Self {
        Self {
            labels: Some(state.labels.into()),
            queue_depth: state.queue_depth,
            service_rate: state.service_rate,
            p95_latency_ms: state.p95_latency_ms,
            batch_fullness: state.batch_fullness,
            loaded: state.loaded,
            warming: state.warming,
            work_left_seconds: state.work_left_seconds,
            last_updated: Some(timestamp::from_system_time(
                state.last_updated.into(),
            )),
        }
    }
}

impl TryFrom<proto_state::ModelState> for ModelState {
    type Error = crate::ProtoError;

    fn try_from(proto: proto_state::ModelState) -> Result<Self, Self::Error> {
        let labels = proto.labels
            .ok_or_else(|| crate::ProtoError::InvalidData("Missing labels".to_string()))?
            .into();

        let last_updated = proto.last_updated
            .map(|ts| timestamp::to_system_time(&ts).into())
            .unwrap_or_else(|| chrono::Utc::now());

        Ok(Self {
            labels,
            queue_depth: proto.queue_depth,
            service_rate: proto.service_rate,
            p95_latency_ms: proto.p95_latency_ms,
            batch_fullness: proto.batch_fullness,
            loaded: proto.loaded,
            warming: proto.warming,
            work_left_seconds: proto.work_left_seconds,
            last_updated,
        })
    }
}

// Conversions for ModelStateDelta
impl From<ModelStateDelta> for proto_state::ModelStateDelta {
    fn from(delta: ModelStateDelta) -> Self {
        Self {
            labels: Some(delta.labels.into()),
            queue_depth: delta.queue_depth,
            service_rate: delta.service_rate,
            p95_latency_ms: delta.p95_latency_ms,
            batch_fullness: delta.batch_fullness,
            loaded: delta.loaded,
            warming: delta.warming,
            timestamp: Some(timestamp::from_system_time(delta.timestamp.into())),
        }
    }
}

impl TryFrom<proto_state::ModelStateDelta> for ModelStateDelta {
    type Error = crate::ProtoError;

    fn try_from(proto: proto_state::ModelStateDelta) -> Result<Self, Self::Error> {
        let labels = proto.labels
            .ok_or_else(|| crate::ProtoError::InvalidData("Missing labels".to_string()))?
            .into();

        let timestamp = proto.timestamp
            .map(|ts| timestamp::to_system_time(&ts).into())
            .unwrap_or_else(|| chrono::Utc::now());

        Ok(Self {
            labels,
            queue_depth: proto.queue_depth,
            service_rate: proto.service_rate,
            p95_latency_ms: proto.p95_latency_ms,
            batch_fullness: proto.batch_fullness,
            loaded: proto.loaded,
            warming: proto.warming,
            timestamp,
        })
    }
}

// Conversions for GpuState
impl From<GpuState> for proto_state::GpuState {
    fn from(state: GpuState) -> Self {
        Self {
            gpu_uuid: state.gpu_uuid,
            node: state.node,
            mig_profile: state.mig_profile.unwrap_or_default(),
            sm_utilization: state.sm_utilization,
            memory_utilization: state.memory_utilization,
            vram_used_gb: state.vram_used_gb,
            vram_total_gb: state.vram_total_gb,
            temperature_c: state.temperature_c.unwrap_or(0.0),
            power_watts: state.power_watts.unwrap_or(0.0),
            ecc_errors: state.ecc_errors,
            throttled: state.throttled,
            last_updated: Some(timestamp::from_system_time(state.last_updated.into())),
        }
    }
}

impl From<proto_state::GpuState> for GpuState {
    fn from(proto: proto_state::GpuState) -> Self {
        let last_updated = proto.last_updated
            .map(|ts| timestamp::to_system_time(&ts).into())
            .unwrap_or_else(|| chrono::Utc::now());

        Self {
            gpu_uuid: proto.gpu_uuid,
            node: proto.node,
            mig_profile: if proto.mig_profile.is_empty() { None } else { Some(proto.mig_profile) },
            sm_utilization: proto.sm_utilization,
            memory_utilization: proto.memory_utilization,
            vram_used_gb: proto.vram_used_gb,
            vram_total_gb: proto.vram_total_gb,
            temperature_c: if proto.temperature_c == 0.0 { None } else { Some(proto.temperature_c) },
            power_watts: if proto.power_watts == 0.0 { None } else { Some(proto.power_watts) },
            ecc_errors: proto.ecc_errors,
            throttled: proto.throttled,
            last_updated,
        }
    }
}

// Conversions for GpuStateDelta
impl From<GpuStateDelta> for proto_state::GpuStateDelta {
    fn from(delta: GpuStateDelta) -> Self {
        Self {
            gpu_uuid: delta.gpu_uuid,
            node: delta.node,
            sm_utilization: delta.sm_utilization,
            memory_utilization: delta.memory_utilization,
            vram_used_gb: delta.vram_used_gb,
            vram_total_gb: delta.vram_total_gb,
            temperature_c: delta.temperature_c,
            power_watts: delta.power_watts,
            ecc_errors: delta.ecc_errors,
            throttled: delta.throttled,
            timestamp: Some(timestamp::from_system_time(delta.timestamp.into())),
        }
    }
}

impl From<proto_state::GpuStateDelta> for GpuStateDelta {
    fn from(proto: proto_state::GpuStateDelta) -> Self {
        let timestamp = proto.timestamp
            .map(|ts| timestamp::to_system_time(&ts).into())
            .unwrap_or_else(|| chrono::Utc::now());

        Self {
            gpu_uuid: proto.gpu_uuid,
            node: proto.node,
            sm_utilization: proto.sm_utilization,
            memory_utilization: proto.memory_utilization,
            vram_used_gb: proto.vram_used_gb,
            vram_total_gb: proto.vram_total_gb,
            temperature_c: proto.temperature_c,
            power_watts: proto.power_watts,
            ecc_errors: proto.ecc_errors,
            throttled: proto.throttled,
            timestamp,
        }
    }
}

// Conversions for SloClass
impl From<SloClass> for proto_scoring::SloClass {
    fn from(slo: SloClass) -> Self {
        match slo {
            SloClass::Latency => proto_scoring::SloClass::Latency,
            SloClass::Throughput => proto_scoring::SloClass::Throughput,
        }
    }
}

impl From<proto_scoring::SloClass> for SloClass {
    fn from(proto: proto_scoring::SloClass) -> Self {
        match proto {
            proto_scoring::SloClass::Latency => SloClass::Latency,
            proto_scoring::SloClass::Throughput => SloClass::Throughput,
            _ => SloClass::Latency, // Default fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_core::Labels;

    #[test]
    fn test_labels_conversion() {
        let original = Labels::new("gpt-4", "v1.0", "triton", "node1")
            .with_quant("fp16")
            .with_gpu_uuid("GPU-12345")
            .with_tenant("customer-a");

        let proto: proto_state::Labels = original.clone().into();
        let converted_back: Labels = proto.into();

        assert_eq!(original.model, converted_back.model);
        assert_eq!(original.revision, converted_back.revision);
        assert_eq!(original.runtime, converted_back.runtime);
        assert_eq!(original.node, converted_back.node);
        assert_eq!(original.quant, converted_back.quant);
        assert_eq!(original.gpu_uuid, converted_back.gpu_uuid);
        assert_eq!(original.tenant, converted_back.tenant);
    }

    #[test]
    fn test_model_state_conversion() {
        let labels = Labels::new("test-model", "v1.0", "mock", "test-node");
        let original = ModelState::new(labels);

        let proto: proto_state::ModelState = original.clone().into();
        let converted_back: ModelState = proto.try_into().unwrap();

        assert_eq!(original.labels.model, converted_back.labels.model);
        assert_eq!(original.queue_depth, converted_back.queue_depth);
        assert_eq!(original.loaded, converted_back.loaded);
    }

    #[test]
    fn test_gpu_state_conversion() {
        let mut original = GpuState::new("GPU-12345", "test-node");
        original.update_metrics(0.8, 0.6, 8.0, 16.0);
        original.update_thermal(Some(75.0), Some(250.0));

        let proto: proto_state::GpuState = original.clone().into();
        let converted_back: GpuState = proto.into();

        assert_eq!(original.gpu_uuid, converted_back.gpu_uuid);
        assert_eq!(original.node, converted_back.node);
        assert_eq!(original.sm_utilization, converted_back.sm_utilization);
        assert_eq!(original.vram_used_gb, converted_back.vram_used_gb);
        assert_eq!(original.temperature_c, converted_back.temperature_c);
    }

    #[test]
    fn test_slo_class_conversion() {
        let latency = SloClass::Latency;
        let proto_latency: proto_scoring::SloClass = latency.into();
        let converted_back: SloClass = proto_latency.into();
        assert_eq!(latency, converted_back);

        let throughput = SloClass::Throughput;
        let proto_throughput: proto_scoring::SloClass = throughput.into();
        let converted_back: SloClass = proto_throughput.into();
        assert_eq!(throughput, converted_back);
    }
}
