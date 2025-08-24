//! Label schema for infermesh components
//!
//! Labels provide a consistent way to identify and categorize models, nodes,
//! GPUs, and other resources across the mesh. They are used for routing
//! decisions, metrics collection, and policy enforcement.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Standard labels used throughout infermesh for consistent identification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Labels {
    /// Model name (e.g., "gpt-4", "llama-2-7b")
    pub model: String,
    
    /// Model revision/version (e.g., "v1.0", "20231201", git hash)
    pub revision: String,
    
    /// Quantization scheme (e.g., "fp16", "int8", "int4")
    pub quant: Option<String>,
    
    /// Runtime type (e.g., "triton", "vllm", "tgi", "torchserve")
    pub runtime: String,
    
    /// Node identifier where this resource is located
    pub node: String,
    
    /// GPU UUID (NVIDIA format) if applicable
    pub gpu_uuid: Option<String>,
    
    /// MIG profile identifier (e.g., "1g.5gb", "3g.20gb")
    pub mig_profile: Option<String>,
    
    /// Tenant identifier for multi-tenancy
    pub tenant: Option<String>,
    
    /// Zone/region identifier (e.g., "us-west-2a", "edge-seattle")
    pub zone: Option<String>,
    
    /// Additional custom labels
    #[serde(flatten)]
    pub custom: HashMap<String, String>,
}

impl Labels {
    /// Create a new Labels instance with required fields
    pub fn new(
        model: impl Into<String>,
        revision: impl Into<String>,
        runtime: impl Into<String>,
        node: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            revision: revision.into(),
            quant: None,
            runtime: runtime.into(),
            node: node.into(),
            gpu_uuid: None,
            mig_profile: None,
            tenant: None,
            zone: None,
            custom: HashMap::new(),
        }
    }

    /// Builder pattern for optional fields
    pub fn with_quant(mut self, quant: impl Into<String>) -> Self {
        self.quant = Some(quant.into());
        self
    }

    pub fn with_gpu_uuid(mut self, gpu_uuid: impl Into<String>) -> Self {
        self.gpu_uuid = Some(gpu_uuid.into());
        self
    }

    pub fn with_mig_profile(mut self, mig_profile: impl Into<String>) -> Self {
        self.mig_profile = Some(mig_profile.into());
        self
    }

    pub fn with_tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = Some(tenant.into());
        self
    }

    pub fn with_zone(mut self, zone: impl Into<String>) -> Self {
        self.zone = Some(zone.into());
        self
    }

    pub fn with_custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.insert(key.into(), value.into());
        self
    }

    /// Generate a unique key for this label combination
    /// Used for indexing and deduplication
    pub fn key(&self) -> String {
        let mut parts = vec![
            self.model.as_str(),
            self.revision.as_str(),
            self.runtime.as_str(),
            self.node.as_str(),
        ];

        if let Some(ref quant) = self.quant {
            parts.push(quant.as_str());
        }
        if let Some(ref gpu_uuid) = self.gpu_uuid {
            parts.push(gpu_uuid.as_str());
        }
        if let Some(ref mig_profile) = self.mig_profile {
            parts.push(mig_profile.as_str());
        }

        parts.join(":")
    }

    /// Check if these labels match a given model and optional filters
    pub fn matches(&self, model: &str, filters: &HashMap<String, String>) -> bool {
        if self.model != model {
            return false;
        }

        for (key, value) in filters {
            let label_value = match key.as_str() {
                "revision" => Some(&self.revision),
                "quant" => self.quant.as_ref(),
                "runtime" => Some(&self.runtime),
                "node" => Some(&self.node),
                "gpu_uuid" => self.gpu_uuid.as_ref(),
                "mig_profile" => self.mig_profile.as_ref(),
                "tenant" => self.tenant.as_ref(),
                "zone" => self.zone.as_ref(),
                _ => self.custom.get(key),
            };

            match label_value {
                Some(v) if v == value => continue,
                _ => return false,
            }
        }

        true
    }

    /// Convert to Prometheus-style labels (sanitized for metric names)
    pub fn to_prometheus_labels(&self) -> HashMap<String, String> {
        let mut labels = HashMap::new();
        
        labels.insert("model".to_string(), sanitize_label_value(&self.model));
        labels.insert("revision".to_string(), sanitize_label_value(&self.revision));
        labels.insert("runtime".to_string(), sanitize_label_value(&self.runtime));
        labels.insert("node".to_string(), sanitize_label_value(&self.node));

        if let Some(ref quant) = self.quant {
            labels.insert("quant".to_string(), sanitize_label_value(quant));
        }
        if let Some(ref gpu_uuid) = self.gpu_uuid {
            labels.insert("gpu_uuid".to_string(), sanitize_label_value(gpu_uuid));
        }
        if let Some(ref mig_profile) = self.mig_profile {
            labels.insert("mig_profile".to_string(), sanitize_label_value(mig_profile));
        }
        if let Some(ref tenant) = self.tenant {
            labels.insert("tenant".to_string(), sanitize_label_value(tenant));
        }
        if let Some(ref zone) = self.zone {
            labels.insert("zone".to_string(), sanitize_label_value(zone));
        }

        // Add custom labels with prefix to avoid conflicts
        for (key, value) in &self.custom {
            labels.insert(
                format!("custom_{}", sanitize_label_name(key)),
                sanitize_label_value(value),
            );
        }

        labels
    }
}

/// Sanitize a label name for Prometheus (alphanumeric + underscore only)
fn sanitize_label_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

/// Sanitize a label value for Prometheus (printable characters only)
fn sanitize_label_value(value: &str) -> String {
    value.chars()
        .map(|c| if c.is_ascii_graphic() || c == ' ' { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_labels_creation() {
        let labels = Labels::new("gpt-4", "v1.0", "triton", "node1");
        
        assert_eq!(labels.model, "gpt-4");
        assert_eq!(labels.revision, "v1.0");
        assert_eq!(labels.runtime, "triton");
        assert_eq!(labels.node, "node1");
        assert!(labels.quant.is_none());
    }

    #[test]
    fn test_labels_builder() {
        let labels = Labels::new("llama-2-7b", "main", "vllm", "gpu-node-1")
            .with_quant("fp16")
            .with_gpu_uuid("GPU-12345678")
            .with_tenant("customer-a")
            .with_zone("us-west-2")
            .with_custom("experiment", "baseline");

        assert_eq!(labels.quant, Some("fp16".to_string()));
        assert_eq!(labels.gpu_uuid, Some("GPU-12345678".to_string()));
        assert_eq!(labels.tenant, Some("customer-a".to_string()));
        assert_eq!(labels.zone, Some("us-west-2".to_string()));
        assert_eq!(labels.custom.get("experiment"), Some(&"baseline".to_string()));
    }

    #[test]
    fn test_labels_key() {
        let labels1 = Labels::new("gpt-4", "v1.0", "triton", "node1");
        let labels2 = Labels::new("gpt-4", "v1.0", "triton", "node1");
        let labels3 = Labels::new("gpt-4", "v1.0", "triton", "node2");

        assert_eq!(labels1.key(), labels2.key());
        assert_ne!(labels1.key(), labels3.key());
    }

    #[test]
    fn test_labels_matching() {
        let labels = Labels::new("gpt-4", "v1.0", "triton", "node1")
            .with_quant("fp16")
            .with_tenant("customer-a");

        // Exact model match
        assert!(labels.matches("gpt-4", &HashMap::new()));
        
        // Model mismatch
        assert!(!labels.matches("gpt-3", &HashMap::new()));

        // Filter match
        let mut filters = HashMap::new();
        filters.insert("runtime".to_string(), "triton".to_string());
        filters.insert("quant".to_string(), "fp16".to_string());
        assert!(labels.matches("gpt-4", &filters));

        // Filter mismatch
        filters.insert("quant".to_string(), "int8".to_string());
        assert!(!labels.matches("gpt-4", &filters));
    }

    #[test]
    fn test_prometheus_labels() {
        let labels = Labels::new("gpt-4", "v1.0", "triton", "node-1")
            .with_custom("test-key", "test value");

        let prom_labels = labels.to_prometheus_labels();
        
        assert_eq!(prom_labels.get("model"), Some(&"gpt-4".to_string()));
        assert_eq!(prom_labels.get("node"), Some(&"node-1".to_string()));
        assert_eq!(prom_labels.get("custom_test_key"), Some(&"test value".to_string()));
    }

    #[test]
    fn test_label_sanitization() {
        assert_eq!(sanitize_label_name("test-key"), "test_key");
        assert_eq!(sanitize_label_name("test.key"), "test_key");
        assert_eq!(sanitize_label_value("test\nvalue"), "test_value");
    }
}
