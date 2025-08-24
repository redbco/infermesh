//! Raft state machine for policy management

use crate::policy::{Policy, PolicyType};
use crate::{Result, RaftError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use raft::eraftpb::Entry;

/// Policy state machine that applies policy operations
pub struct PolicyStateMachine {
    /// Current policies indexed by ID
    policies: HashMap<String, Policy>,
    
    /// Applied index
    applied_index: u64,
    
    /// State machine statistics
    stats: StateMachineStats,
}

/// State machine statistics
#[derive(Debug, Clone, Default)]
pub struct StateMachineStats {
    /// Number of policies
    pub policy_count: u64,
    
    /// Number of operations applied
    pub operations_applied: u64,
    
    /// Last applied index
    pub applied_index: u64,
    
    /// Policy type counts
    pub policy_type_counts: HashMap<String, u64>,
}

/// State machine snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMachineSnapshot {
    /// Snapshot of all policies
    pub policies: HashMap<String, Policy>,
    
    /// Applied index at snapshot time
    pub applied_index: u64,
    
    /// Snapshot metadata
    pub metadata: SnapshotMetadata,
}

/// Snapshot metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    /// Snapshot creation time
    pub created_at: chrono::DateTime<chrono::Utc>,
    
    /// Snapshot version
    pub version: u64,
    
    /// Checksum of snapshot data
    pub checksum: String,
}

/// Policy operations that can be applied to the state machine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyOperation {
    /// Create a new policy
    Create(Policy),
    
    /// Update an existing policy
    Update { id: String, policy: Policy },
    
    /// Delete a policy
    Delete { id: String },
    
    /// Enable/disable a policy
    SetEnabled { id: String, enabled: bool },
    
    /// Batch operations
    Batch(Vec<PolicyOperation>),
}

/// Result of applying a policy operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationResult {
    /// Whether the operation succeeded
    pub success: bool,
    
    /// Error message if operation failed
    pub error: Option<String>,
    
    /// Affected policy ID
    pub policy_id: Option<String>,
    
    /// Operation metadata
    pub metadata: OperationMetadata,
}

/// Operation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMetadata {
    /// Operation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Applied index
    pub applied_index: u64,
    
    /// Operation type
    pub operation_type: String,
}

impl PolicyStateMachine {
    /// Create a new policy state machine
    pub fn new() -> Self {
        Self {
            policies: HashMap::new(),
            applied_index: 0,
            stats: StateMachineStats::default(),
        }
    }

    /// Apply a raft log entry to the state machine
    pub fn apply_entry(&mut self, entry: &Entry) -> Result<OperationResult> {
        if entry.index <= self.applied_index {
            return Err(RaftError::InvalidProposal);
        }

        // Deserialize the operation
        let operation: PolicyOperation = bincode::deserialize(&entry.data)
            .map_err(|e| RaftError::Serialization(e))?;

        // Apply the operation
        let result = self.apply_operation(operation, entry.index)?;
        
        // Update applied index
        self.applied_index = entry.index;
        self.stats.applied_index = entry.index;
        self.stats.operations_applied += 1;

        Ok(result)
    }

    /// Apply a policy operation
    fn apply_operation(&mut self, operation: PolicyOperation, index: u64) -> Result<OperationResult> {
        let timestamp = chrono::Utc::now();
        
        match operation {
            PolicyOperation::Create(policy) => {
                let policy_id = policy.id.clone();
                
                if self.policies.contains_key(&policy_id) {
                    return Ok(OperationResult {
                        success: false,
                        error: Some(format!("Policy {} already exists", policy_id)),
                        policy_id: Some(policy_id),
                        metadata: OperationMetadata {
                            timestamp,
                            applied_index: index,
                            operation_type: "create".to_string(),
                        },
                    });
                }
                
                // Update statistics
                self.update_policy_type_stats(&policy.policy_type, 1);
                self.stats.policy_count += 1;
                
                self.policies.insert(policy_id.clone(), policy);
                
                Ok(OperationResult {
                    success: true,
                    error: None,
                    policy_id: Some(policy_id),
                    metadata: OperationMetadata {
                        timestamp,
                        applied_index: index,
                        operation_type: "create".to_string(),
                    },
                })
            }
            
            PolicyOperation::Update { id, policy } => {
                if !self.policies.contains_key(&id) {
                    return Ok(OperationResult {
                        success: false,
                        error: Some(format!("Policy {} not found", id)),
                        policy_id: Some(id),
                        metadata: OperationMetadata {
                            timestamp,
                            applied_index: index,
                            operation_type: "update".to_string(),
                        },
                    });
                }
                
                // Update the policy
                let old_policy_type = self.policies.get(&id).map(|p| p.policy_type.clone());
                if let Some(old_type) = old_policy_type {
                    // Update statistics if policy type changed
                    if std::mem::discriminant(&old_type) != std::mem::discriminant(&policy.policy_type) {
                        self.update_policy_type_stats(&old_type, -1);
                        self.update_policy_type_stats(&policy.policy_type, 1);
                    }
                }
                
                self.policies.insert(id.clone(), policy);
                
                Ok(OperationResult {
                    success: true,
                    error: None,
                    policy_id: Some(id),
                    metadata: OperationMetadata {
                        timestamp,
                        applied_index: index,
                        operation_type: "update".to_string(),
                    },
                })
            }
            
            PolicyOperation::Delete { id } => {
                if let Some(policy) = self.policies.remove(&id) {
                    // Update statistics
                    self.update_policy_type_stats(&policy.policy_type, -1);
                    self.stats.policy_count -= 1;
                    
                    Ok(OperationResult {
                        success: true,
                        error: None,
                        policy_id: Some(id),
                        metadata: OperationMetadata {
                            timestamp,
                            applied_index: index,
                            operation_type: "delete".to_string(),
                        },
                    })
                } else {
                    Ok(OperationResult {
                        success: false,
                        error: Some(format!("Policy {} not found", id)),
                        policy_id: Some(id),
                        metadata: OperationMetadata {
                            timestamp,
                            applied_index: index,
                            operation_type: "delete".to_string(),
                        },
                    })
                }
            }
            
            PolicyOperation::SetEnabled { id, enabled } => {
                if let Some(policy) = self.policies.get_mut(&id) {
                    policy.set_enabled(enabled);
                    
                    Ok(OperationResult {
                        success: true,
                        error: None,
                        policy_id: Some(id),
                        metadata: OperationMetadata {
                            timestamp,
                            applied_index: index,
                            operation_type: "set_enabled".to_string(),
                        },
                    })
                } else {
                    Ok(OperationResult {
                        success: false,
                        error: Some(format!("Policy {} not found", id)),
                        policy_id: Some(id),
                        metadata: OperationMetadata {
                            timestamp,
                            applied_index: index,
                            operation_type: "set_enabled".to_string(),
                        },
                    })
                }
            }
            
            PolicyOperation::Batch(operations) => {
                let mut results = Vec::new();
                let mut all_success = true;
                
                for op in operations {
                    let result = self.apply_operation(op, index)?;
                    if !result.success {
                        all_success = false;
                    }
                    results.push(result);
                }
                
                Ok(OperationResult {
                    success: all_success,
                    error: if all_success { None } else { Some("Some batch operations failed".to_string()) },
                    policy_id: None,
                    metadata: OperationMetadata {
                        timestamp,
                        applied_index: index,
                        operation_type: "batch".to_string(),
                    },
                })
            }
        }
    }

    /// Update policy type statistics
    fn update_policy_type_stats(&mut self, policy_type: &PolicyType, delta: i64) {
        let type_name = match policy_type {
            PolicyType::ModelPin(_) => "model_pin",
            PolicyType::Quota(_) => "quota",
            PolicyType::Acl(_) => "acl",
            PolicyType::LoadBalancing(_) => "load_balancing",
            PolicyType::Scaling(_) => "scaling",
        };
        
        let count = self.stats.policy_type_counts.entry(type_name.to_string()).or_insert(0);
        if delta > 0 {
            *count += delta as u64;
        } else {
            *count = count.saturating_sub((-delta) as u64);
        }
    }

    /// Get a policy by ID
    pub fn get_policy(&self, id: &str) -> Option<&Policy> {
        self.policies.get(id)
    }

    /// Get all policies
    pub fn get_all_policies(&self) -> &HashMap<String, Policy> {
        &self.policies
    }

    /// Get policies by type
    pub fn get_policies_by_type(&self, policy_type: &str) -> Vec<&Policy> {
        self.policies
            .values()
            .filter(|policy| {
                match (&policy.policy_type, policy_type) {
                    (PolicyType::ModelPin(_), "model_pin") => true,
                    (PolicyType::Quota(_), "quota") => true,
                    (PolicyType::Acl(_), "acl") => true,
                    (PolicyType::LoadBalancing(_), "load_balancing") => true,
                    (PolicyType::Scaling(_), "scaling") => true,
                    _ => false,
                }
            })
            .collect()
    }

    /// Get enabled policies
    pub fn get_enabled_policies(&self) -> Vec<&Policy> {
        self.policies
            .values()
            .filter(|policy| policy.is_enabled())
            .collect()
    }

    /// Get applied index
    pub fn applied_index(&self) -> u64 {
        self.applied_index
    }

    /// Get statistics
    pub fn stats(&self) -> &StateMachineStats {
        &self.stats
    }

    /// Create a snapshot of the current state
    pub fn create_snapshot(&self) -> Result<StateMachineSnapshot> {
        let policies_json = serde_json::to_string(&self.policies)?;
        let checksum = format!("{:x}", md5::compute(policies_json.as_bytes()));
        
        Ok(StateMachineSnapshot {
            policies: self.policies.clone(),
            applied_index: self.applied_index,
            metadata: SnapshotMetadata {
                created_at: chrono::Utc::now(),
                version: 1,
                checksum,
            },
        })
    }

    /// Restore from a snapshot
    pub fn restore_from_snapshot(&mut self, snapshot: StateMachineSnapshot) -> Result<()> {
        // Verify checksum
        let policies_json = serde_json::to_string(&snapshot.policies)?;
        let checksum = format!("{:x}", md5::compute(policies_json.as_bytes()));
        
        if checksum != snapshot.metadata.checksum {
            return Err(RaftError::Storage("Snapshot checksum mismatch".to_string()));
        }
        
        // Restore state
        self.policies = snapshot.policies;
        self.applied_index = snapshot.applied_index;
        
        // Recalculate statistics
        self.recalculate_stats();
        
        Ok(())
    }

    /// Recalculate statistics from current state
    fn recalculate_stats(&mut self) {
        self.stats = StateMachineStats {
            policy_count: self.policies.len() as u64,
            operations_applied: self.stats.operations_applied, // Keep existing count
            applied_index: self.applied_index,
            policy_type_counts: HashMap::new(),
        };
        
        // Count policy types
        let policy_types: Vec<_> = self.policies.values().map(|p| p.policy_type.clone()).collect();
        for policy_type in policy_types {
            self.update_policy_type_stats(&policy_type, 1);
        }
    }

    /// Serialize an operation for raft log entry
    pub fn serialize_operation(operation: PolicyOperation) -> Result<Vec<u8>> {
        bincode::serialize(&operation).map_err(RaftError::Serialization)
    }
}

impl Default for PolicyStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{Policy, QuotaScope, ResourceLimits};
    use raft::eraftpb::Entry;

    #[test]
    fn test_state_machine_creation() {
        let sm = PolicyStateMachine::new();
        assert_eq!(sm.applied_index(), 0);
        assert_eq!(sm.stats().policy_count, 0);
        assert!(sm.get_all_policies().is_empty());
    }

    #[test]
    fn test_policy_creation() {
        let mut sm = PolicyStateMachine::new();
        let policy = Policy::pin_model("gpt-7b".to_string(), vec![]);
        let operation = PolicyOperation::Create(policy.clone());
        
        let mut entry = Entry::default();
        entry.index = 1;
        entry.term = 1;
        entry.data = bytes::Bytes::from(PolicyStateMachine::serialize_operation(operation).unwrap());
        
        let result = sm.apply_entry(&entry).unwrap();
        assert!(result.success);
        assert_eq!(result.policy_id, Some(policy.id.clone()));
        
        assert_eq!(sm.applied_index(), 1);
        assert_eq!(sm.stats().policy_count, 1);
        assert!(sm.get_policy(&policy.id).is_some());
    }

    #[test]
    fn test_policy_update() {
        let mut sm = PolicyStateMachine::new();
        let mut policy = Policy::pin_model("gpt-7b".to_string(), vec![]);
        let policy_id = policy.id.clone();
        
        // Create policy
        let create_op = PolicyOperation::Create(policy.clone());
        let mut create_entry = Entry::default();
        create_entry.index = 1;
        create_entry.term = 1;
        create_entry.data = bytes::Bytes::from(PolicyStateMachine::serialize_operation(create_op).unwrap());
        sm.apply_entry(&create_entry).unwrap();
        
        // Update policy
        policy.update_metadata(Some("new-name".to_string()), None, None);
        let update_op = PolicyOperation::Update { id: policy_id.clone(), policy: policy.clone() };
        let mut update_entry = Entry::default();
        update_entry.index = 2;
        update_entry.term = 1;
        update_entry.data = bytes::Bytes::from(PolicyStateMachine::serialize_operation(update_op).unwrap());
        
        let result = sm.apply_entry(&update_entry).unwrap();
        assert!(result.success);
        
        let stored_policy = sm.get_policy(&policy_id).unwrap();
        assert_eq!(stored_policy.name(), "new-name");
    }

    #[test]
    fn test_policy_deletion() {
        let mut sm = PolicyStateMachine::new();
        let policy = Policy::pin_model("gpt-7b".to_string(), vec![]);
        let policy_id = policy.id.clone();
        
        // Create policy
        let create_op = PolicyOperation::Create(policy);
        let mut create_entry = Entry::default();
        create_entry.index = 1;
        create_entry.term = 1;
        create_entry.data = bytes::Bytes::from(PolicyStateMachine::serialize_operation(create_op).unwrap());
        sm.apply_entry(&create_entry).unwrap();
        
        // Delete policy
        let delete_op = PolicyOperation::Delete { id: policy_id.clone() };
        let mut delete_entry = Entry::default();
        delete_entry.index = 2;
        delete_entry.term = 1;
        delete_entry.data = bytes::Bytes::from(PolicyStateMachine::serialize_operation(delete_op).unwrap());
        
        let result = sm.apply_entry(&delete_entry).unwrap();
        assert!(result.success);
        
        assert_eq!(sm.stats().policy_count, 0);
        assert!(sm.get_policy(&policy_id).is_none());
    }

    #[test]
    fn test_policy_enable_disable() {
        let mut sm = PolicyStateMachine::new();
        let policy = Policy::pin_model("gpt-7b".to_string(), vec![]);
        let policy_id = policy.id.clone();
        
        // Create policy
        let create_op = PolicyOperation::Create(policy);
        let mut create_entry = Entry::default();
        create_entry.index = 1;
        create_entry.term = 1;
        create_entry.data = bytes::Bytes::from(PolicyStateMachine::serialize_operation(create_op).unwrap());
        sm.apply_entry(&create_entry).unwrap();
        
        // Disable policy
        let disable_op = PolicyOperation::SetEnabled { id: policy_id.clone(), enabled: false };
        let mut disable_entry = Entry::default();
        disable_entry.index = 2;
        disable_entry.term = 1;
        disable_entry.data = bytes::Bytes::from(PolicyStateMachine::serialize_operation(disable_op).unwrap());
        
        let result = sm.apply_entry(&disable_entry).unwrap();
        assert!(result.success);
        
        let stored_policy = sm.get_policy(&policy_id).unwrap();
        assert!(!stored_policy.is_enabled());
    }

    #[test]
    fn test_batch_operations() {
        let mut sm = PolicyStateMachine::new();
        
        let policy1 = Policy::pin_model("gpt-7b".to_string(), vec![]);
        let policy2 = Policy::quota(QuotaScope::Global, ResourceLimits {
            max_cpu: Some(4.0),
            max_memory: None,
            max_gpu: None,
            max_rps: None,
            max_concurrent_requests: None,
        });
        
        let batch_op = PolicyOperation::Batch(vec![
            PolicyOperation::Create(policy1),
            PolicyOperation::Create(policy2),
        ]);
        
        let mut entry = Entry::default();
        entry.index = 1;
        entry.term = 1;
        entry.data = bytes::Bytes::from(PolicyStateMachine::serialize_operation(batch_op).unwrap());
        
        let result = sm.apply_entry(&entry).unwrap();
        assert!(result.success);
        assert_eq!(sm.stats().policy_count, 2);
    }

    #[test]
    fn test_snapshot_creation_and_restore() {
        let mut sm = PolicyStateMachine::new();
        
        // Add some policies
        let policy1 = Policy::pin_model("gpt-7b".to_string(), vec![]);
        let policy2 = Policy::quota(QuotaScope::Global, ResourceLimits {
            max_cpu: Some(4.0),
            max_memory: None,
            max_gpu: None,
            max_rps: None,
            max_concurrent_requests: None,
        });
        
        let create_op1 = PolicyOperation::Create(policy1.clone());
        let create_op2 = PolicyOperation::Create(policy2.clone());
        
        let mut entry1 = Entry::default();
        entry1.index = 1;
        entry1.term = 1;
        entry1.data = bytes::Bytes::from(PolicyStateMachine::serialize_operation(create_op1).unwrap());
        
        let mut entry2 = Entry::default();
        entry2.index = 2;
        entry2.term = 1;
        entry2.data = bytes::Bytes::from(PolicyStateMachine::serialize_operation(create_op2).unwrap());
        
        sm.apply_entry(&entry1).unwrap();
        sm.apply_entry(&entry2).unwrap();
        
        // Create snapshot
        let snapshot = sm.create_snapshot().unwrap();
        assert_eq!(snapshot.policies.len(), 2);
        assert_eq!(snapshot.applied_index, 2);
        
        // Create new state machine and restore
        let mut sm2 = PolicyStateMachine::new();
        sm2.restore_from_snapshot(snapshot).unwrap();
        
        assert_eq!(sm2.stats().policy_count, 2);
        assert_eq!(sm2.applied_index(), 2);
        assert!(sm2.get_policy(&policy1.id).is_some());
        assert!(sm2.get_policy(&policy2.id).is_some());
    }

    #[test]
    fn test_get_policies_by_type() {
        let mut sm = PolicyStateMachine::new();
        
        let pin_policy = Policy::pin_model("gpt-7b".to_string(), vec![]);
        let quota_policy = Policy::quota(QuotaScope::Global, ResourceLimits {
            max_cpu: Some(4.0),
            max_memory: None,
            max_gpu: None,
            max_rps: None,
            max_concurrent_requests: None,
        });
        
        let create_op1 = PolicyOperation::Create(pin_policy);
        let create_op2 = PolicyOperation::Create(quota_policy);
        
        let mut entry1 = Entry::default();
        entry1.index = 1;
        entry1.term = 1;
        entry1.data = bytes::Bytes::from(PolicyStateMachine::serialize_operation(create_op1).unwrap());
        
        let mut entry2 = Entry::default();
        entry2.index = 2;
        entry2.term = 1;
        entry2.data = bytes::Bytes::from(PolicyStateMachine::serialize_operation(create_op2).unwrap());
        
        sm.apply_entry(&entry1).unwrap();
        sm.apply_entry(&entry2).unwrap();
        
        let pin_policies = sm.get_policies_by_type("model_pin");
        let quota_policies = sm.get_policies_by_type("quota");
        
        assert_eq!(pin_policies.len(), 1);
        assert_eq!(quota_policies.len(), 1);
    }
}
