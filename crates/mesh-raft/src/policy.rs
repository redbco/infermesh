//! Policy definitions and management

use mesh_core::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Policy represents a distributed configuration or rule
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Policy {
    /// Unique policy ID
    pub id: String,
    
    /// Policy type and data
    pub policy_type: PolicyType,
    
    /// Policy metadata
    pub metadata: PolicyMetadata,
}

/// Policy metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PolicyMetadata {
    /// Human-readable name
    pub name: String,
    
    /// Policy description
    pub description: Option<String>,
    
    /// Policy version
    pub version: u64,
    
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    
    /// Last updated timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    
    /// Policy creator/owner
    pub owner: Option<String>,
    
    /// Policy tags for organization
    pub tags: HashMap<String, String>,
    
    /// Whether the policy is enabled
    pub enabled: bool,
}

/// Policy types supported by the system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PolicyType {
    /// Model pinning policy
    ModelPin(ModelPinPolicy),
    
    /// Resource quota policy
    Quota(QuotaPolicy),
    
    /// Access control policy
    Acl(AclPolicy),
    
    /// Load balancing policy
    LoadBalancing(LoadBalancingPolicy),
    
    /// Scaling policy
    Scaling(ScalingPolicy),
}

/// Model pinning policy - ensures specific models are loaded on specific nodes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelPinPolicy {
    /// Model identifier
    pub model_id: String,
    
    /// Model version (optional)
    pub model_version: Option<String>,
    
    /// Target nodes where the model should be pinned
    pub target_nodes: Vec<NodeId>,
    
    /// Minimum number of replicas
    pub min_replicas: u32,
    
    /// Maximum number of replicas
    pub max_replicas: Option<u32>,
    
    /// Priority for resource allocation
    pub priority: PolicyPriority,
    
    /// Constraints for node selection
    pub constraints: Vec<NodeConstraint>,
}

/// Resource quota policy - limits resource usage
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuotaPolicy {
    /// Scope of the quota (global, per-node, per-user, etc.)
    pub scope: QuotaScope,
    
    /// Resource limits
    pub limits: ResourceLimits,
    
    /// Enforcement mode
    pub enforcement: EnforcementMode,
    
    /// Grace period before enforcement
    pub grace_period: Option<std::time::Duration>,
}

/// Access control policy - controls who can access what
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AclPolicy {
    /// Subject (user, service, etc.)
    pub subject: AclSubject,
    
    /// Resource being accessed
    pub resource: AclResource,
    
    /// Actions allowed
    pub actions: Vec<AclAction>,
    
    /// Conditions for access
    pub conditions: Vec<AclCondition>,
    
    /// Effect (allow or deny)
    pub effect: AclEffect,
}

/// Load balancing policy - controls request distribution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoadBalancingPolicy {
    /// Target model or service
    pub target: String,
    
    /// Load balancing algorithm
    pub algorithm: LoadBalancingAlgorithm,
    
    /// Weights for weighted algorithms
    pub weights: HashMap<NodeId, f64>,
    
    /// Health check configuration
    pub health_check: HealthCheckConfig,
    
    /// Failover configuration
    pub failover: FailoverConfig,
}

/// Scaling policy - controls automatic scaling
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScalingPolicy {
    /// Target model or service
    pub target: String,
    
    /// Scaling triggers
    pub triggers: Vec<ScalingTrigger>,
    
    /// Scaling limits
    pub limits: ScalingLimits,
    
    /// Scaling behavior
    pub behavior: ScalingBehavior,
}

/// Policy priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum PolicyPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// Node constraints for policy targeting
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeConstraint {
    /// Node must have specific labels
    HasLabels(HashMap<String, String>),
    
    /// Node must have minimum resources
    MinResources(ResourceRequirements),
    
    /// Node must be in specific availability zone
    AvailabilityZone(String),
    
    /// Node must have specific GPU type
    GpuType(String),
    
    /// Custom constraint expression
    Expression(String),
}

/// Resource requirements
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResourceRequirements {
    /// CPU cores
    pub cpu_cores: Option<f64>,
    
    /// Memory in bytes
    pub memory_bytes: Option<u64>,
    
    /// GPU memory in bytes
    pub gpu_memory_bytes: Option<u64>,
    
    /// Number of GPUs
    pub gpu_count: Option<u32>,
}

/// Quota scope
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuotaScope {
    Global,
    PerNode(NodeId),
    PerUser(String),
    PerNamespace(String),
    PerModel(String),
}

/// Resource limits
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResourceLimits {
    /// Maximum CPU usage
    pub max_cpu: Option<f64>,
    
    /// Maximum memory usage in bytes
    pub max_memory: Option<u64>,
    
    /// Maximum GPU usage
    pub max_gpu: Option<u32>,
    
    /// Maximum requests per second
    pub max_rps: Option<f64>,
    
    /// Maximum concurrent requests
    pub max_concurrent_requests: Option<u32>,
}

/// Enforcement mode for quotas
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EnforcementMode {
    /// Block requests that exceed quota
    Block,
    
    /// Throttle requests that exceed quota
    Throttle,
    
    /// Log violations but allow requests
    Log,
    
    /// Monitor only (no enforcement)
    Monitor,
}

/// ACL subject
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AclSubject {
    User(String),
    Service(String),
    Role(String),
    Group(String),
    Anonymous,
    All,
}

/// ACL resource
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AclResource {
    Model(String),
    Node(NodeId),
    Endpoint(String),
    Namespace(String),
    All,
}

/// ACL action
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AclAction {
    Read,
    Write,
    Execute,
    Delete,
    Admin,
    All,
}

/// ACL condition
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AclCondition {
    TimeRange { start: chrono::DateTime<chrono::Utc>, end: chrono::DateTime<chrono::Utc> },
    IpRange(String),
    UserAgent(String),
    Custom(String),
}

/// ACL effect
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AclEffect {
    Allow,
    Deny,
}

/// Load balancing algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LoadBalancingAlgorithm {
    RoundRobin,
    Random,
    LeastConnections,
    WeightedRoundRobin,
    ConsistentHash,
    ScoreBased,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HealthCheckConfig {
    /// Health check interval
    pub interval: std::time::Duration,
    
    /// Health check timeout
    pub timeout: std::time::Duration,
    
    /// Number of consecutive failures to mark unhealthy
    pub failure_threshold: u32,
    
    /// Number of consecutive successes to mark healthy
    pub success_threshold: u32,
}

/// Failover configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FailoverConfig {
    /// Enable automatic failover
    pub enabled: bool,
    
    /// Maximum failover attempts
    pub max_attempts: u32,
    
    /// Failover timeout
    pub timeout: std::time::Duration,
    
    /// Fallback nodes
    pub fallback_nodes: Vec<NodeId>,
}

/// Scaling trigger
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ScalingTrigger {
    /// CPU utilization threshold
    CpuUtilization { threshold: f64, duration: std::time::Duration },
    
    /// Memory utilization threshold
    MemoryUtilization { threshold: f64, duration: std::time::Duration },
    
    /// Request rate threshold
    RequestRate { threshold: f64, duration: std::time::Duration },
    
    /// Queue depth threshold
    QueueDepth { threshold: u32, duration: std::time::Duration },
    
    /// Custom metric threshold
    CustomMetric { name: String, threshold: f64, duration: std::time::Duration },
}

/// Scaling limits
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScalingLimits {
    /// Minimum replicas
    pub min_replicas: u32,
    
    /// Maximum replicas
    pub max_replicas: u32,
    
    /// Maximum scale up per step
    pub max_scale_up: u32,
    
    /// Maximum scale down per step
    pub max_scale_down: u32,
}

/// Scaling behavior
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScalingBehavior {
    /// Scale up cooldown period
    pub scale_up_cooldown: std::time::Duration,
    
    /// Scale down cooldown period
    pub scale_down_cooldown: std::time::Duration,
    
    /// Stabilization window
    pub stabilization_window: std::time::Duration,
}

impl Policy {
    /// Create a new model pin policy
    pub fn pin_model(model_id: String, target_nodes: Vec<NodeId>) -> Self {
        let id = Uuid::new_v4().to_string();
        let now = chrono::Utc::now();
        
        Self {
            id: id.clone(),
            policy_type: PolicyType::ModelPin(ModelPinPolicy {
                model_id: model_id.clone(),
                model_version: None,
                target_nodes,
                min_replicas: 1,
                max_replicas: None,
                priority: PolicyPriority::Normal,
                constraints: Vec::new(),
            }),
            metadata: PolicyMetadata {
                name: format!("pin-{}", model_id),
                description: Some(format!("Pin model {} to specific nodes", model_id)),
                version: 1,
                created_at: now,
                updated_at: now,
                owner: None,
                tags: HashMap::new(),
                enabled: true,
            },
        }
    }

    /// Create a new quota policy
    pub fn quota(scope: QuotaScope, limits: ResourceLimits) -> Self {
        let id = Uuid::new_v4().to_string();
        let now = chrono::Utc::now();
        
        Self {
            id: id.clone(),
            policy_type: PolicyType::Quota(QuotaPolicy {
                scope: scope.clone(),
                limits,
                enforcement: EnforcementMode::Block,
                grace_period: None,
            }),
            metadata: PolicyMetadata {
                name: format!("quota-{:?}", scope),
                description: Some("Resource quota policy".to_string()),
                version: 1,
                created_at: now,
                updated_at: now,
                owner: None,
                tags: HashMap::new(),
                enabled: true,
            },
        }
    }

    /// Create a new ACL policy
    pub fn acl(subject: AclSubject, resource: AclResource, actions: Vec<AclAction>, effect: AclEffect) -> Self {
        let id = Uuid::new_v4().to_string();
        let now = chrono::Utc::now();
        
        Self {
            id: id.clone(),
            policy_type: PolicyType::Acl(AclPolicy {
                subject: subject.clone(),
                resource: resource.clone(),
                actions,
                conditions: Vec::new(),
                effect,
            }),
            metadata: PolicyMetadata {
                name: format!("acl-{:?}-{:?}", subject, resource),
                description: Some("Access control policy".to_string()),
                version: 1,
                created_at: now,
                updated_at: now,
                owner: None,
                tags: HashMap::new(),
                enabled: true,
            },
        }
    }

    /// Get policy ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get policy name
    pub fn name(&self) -> &str {
        &self.metadata.name
    }

    /// Check if policy is enabled
    pub fn is_enabled(&self) -> bool {
        self.metadata.enabled
    }

    /// Update policy metadata
    pub fn update_metadata(&mut self, name: Option<String>, description: Option<String>, tags: Option<HashMap<String, String>>) {
        if let Some(name) = name {
            self.metadata.name = name;
        }
        if let Some(description) = description {
            self.metadata.description = Some(description);
        }
        if let Some(tags) = tags {
            self.metadata.tags = tags;
        }
        self.metadata.updated_at = chrono::Utc::now();
        self.metadata.version += 1;
    }

    /// Enable or disable policy
    pub fn set_enabled(&mut self, enabled: bool) {
        self.metadata.enabled = enabled;
        self.metadata.updated_at = chrono::Utc::now();
        self.metadata.version += 1;
    }
}

impl Default for PolicyPriority {
    fn default() -> Self {
        PolicyPriority::Normal
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval: std::time::Duration::from_secs(30),
            timeout: std::time::Duration::from_secs(5),
            failure_threshold: 3,
            success_threshold: 2,
        }
    }
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_attempts: 3,
            timeout: std::time::Duration::from_secs(30),
            fallback_nodes: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_pin_policy() {
        let nodes = vec![NodeId::new("node1"), NodeId::new("node2")];
        let policy = Policy::pin_model("gpt-7b".to_string(), nodes.clone());
        
        assert!(!policy.id.is_empty());
        assert_eq!(policy.name(), "pin-gpt-7b");
        assert!(policy.is_enabled());
        
        if let PolicyType::ModelPin(pin_policy) = &policy.policy_type {
            assert_eq!(pin_policy.model_id, "gpt-7b");
            assert_eq!(pin_policy.target_nodes, nodes);
            assert_eq!(pin_policy.min_replicas, 1);
            assert_eq!(pin_policy.priority, PolicyPriority::Normal);
        } else {
            panic!("Expected ModelPin policy type");
        }
    }

    #[test]
    fn test_quota_policy() {
        let limits = ResourceLimits {
            max_cpu: Some(4.0),
            max_memory: Some(8 * 1024 * 1024 * 1024), // 8GB
            max_gpu: Some(2),
            max_rps: Some(100.0),
            max_concurrent_requests: Some(50),
        };
        
        let policy = Policy::quota(QuotaScope::Global, limits.clone());
        
        assert!(!policy.id.is_empty());
        assert!(policy.name().starts_with("quota-"));
        assert!(policy.is_enabled());
        
        if let PolicyType::Quota(quota_policy) = &policy.policy_type {
            assert_eq!(quota_policy.scope, QuotaScope::Global);
            assert_eq!(quota_policy.limits, limits);
            assert_eq!(quota_policy.enforcement, EnforcementMode::Block);
        } else {
            panic!("Expected Quota policy type");
        }
    }

    #[test]
    fn test_acl_policy() {
        let policy = Policy::acl(
            AclSubject::User("alice".to_string()),
            AclResource::Model("gpt-7b".to_string()),
            vec![AclAction::Read, AclAction::Execute],
            AclEffect::Allow,
        );
        
        assert!(!policy.id.is_empty());
        assert!(policy.name().contains("acl"));
        assert!(policy.is_enabled());
        
        if let PolicyType::Acl(acl_policy) = &policy.policy_type {
            assert_eq!(acl_policy.subject, AclSubject::User("alice".to_string()));
            assert_eq!(acl_policy.resource, AclResource::Model("gpt-7b".to_string()));
            assert_eq!(acl_policy.actions, vec![AclAction::Read, AclAction::Execute]);
            assert_eq!(acl_policy.effect, AclEffect::Allow);
        } else {
            panic!("Expected Acl policy type");
        }
    }

    #[test]
    fn test_policy_metadata_update() {
        let mut policy = Policy::pin_model("test".to_string(), vec![]);
        let original_version = policy.metadata.version;
        let original_updated_at = policy.metadata.updated_at;
        
        // Wait a bit to ensure timestamp changes
        std::thread::sleep(std::time::Duration::from_millis(1));
        
        policy.update_metadata(
            Some("new-name".to_string()),
            Some("new description".to_string()),
            Some(HashMap::from([("env".to_string(), "test".to_string())])),
        );
        
        assert_eq!(policy.metadata.name, "new-name");
        assert_eq!(policy.metadata.description, Some("new description".to_string()));
        assert_eq!(policy.metadata.tags.get("env"), Some(&"test".to_string()));
        assert_eq!(policy.metadata.version, original_version + 1);
        assert!(policy.metadata.updated_at > original_updated_at);
    }

    #[test]
    fn test_policy_enable_disable() {
        let mut policy = Policy::pin_model("test".to_string(), vec![]);
        assert!(policy.is_enabled());
        
        policy.set_enabled(false);
        assert!(!policy.is_enabled());
        
        policy.set_enabled(true);
        assert!(policy.is_enabled());
    }

    #[test]
    fn test_policy_priority_ordering() {
        assert!(PolicyPriority::Critical > PolicyPriority::High);
        assert!(PolicyPriority::High > PolicyPriority::Normal);
        assert!(PolicyPriority::Normal > PolicyPriority::Low);
    }

    #[test]
    fn test_resource_requirements() {
        let req = ResourceRequirements {
            cpu_cores: Some(2.0),
            memory_bytes: Some(4 * 1024 * 1024 * 1024), // 4GB
            gpu_memory_bytes: Some(8 * 1024 * 1024 * 1024), // 8GB
            gpu_count: Some(1),
        };
        
        assert_eq!(req.cpu_cores, Some(2.0));
        assert_eq!(req.memory_bytes, Some(4 * 1024 * 1024 * 1024));
        assert_eq!(req.gpu_memory_bytes, Some(8 * 1024 * 1024 * 1024));
        assert_eq!(req.gpu_count, Some(1));
    }
}
