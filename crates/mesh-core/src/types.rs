//! Core type definitions for infermesh

use crate::Labels;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::net::SocketAddr;
use uuid::Uuid;

/// Unique identifier for a node in the mesh
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(String);

impl NodeId {
    /// Create a new NodeId from a string
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generate a random NodeId
    pub fn generate() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Get the string representation of the NodeId
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for NodeId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl From<&str> for NodeId {
    fn from(id: &str) -> Self {
        Self(id.to_string())
    }
}

/// Roles that a node can fulfill in the mesh
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    /// Router nodes accept inference traffic and make routing decisions
    Router,
    /// GPU nodes run runtimes and provide GPU telemetry
    Gpu,
    /// Edge nodes provide optional ingress points close to users
    Edge,
}

impl NodeRole {
    /// Parse a comma-separated string of roles
    pub fn parse_roles(roles_str: &str) -> crate::Result<Vec<NodeRole>> {
        roles_str
            .split(',')
            .map(|s| s.trim().parse())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| crate::Error::InvalidConfiguration(format!("Invalid node role: {}", e)))
    }
}

impl std::str::FromStr for NodeRole {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "router" => Ok(NodeRole::Router),
            "gpu" => Ok(NodeRole::Gpu),
            "edge" => Ok(NodeRole::Edge),
            _ => Err(format!("Unknown node role: {}", s)),
        }
    }
}

impl fmt::Display for NodeRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeRole::Router => write!(f, "router"),
            NodeRole::Gpu => write!(f, "gpu"),
            NodeRole::Edge => write!(f, "edge"),
        }
    }
}

/// Service Level Objective classes for request prioritization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SloClass {
    /// Optimize for low latency (interactive workloads)
    Latency,
    /// Optimize for high throughput (batch workloads)
    Throughput,
}

impl SloClass {
    /// Get the priority weight for this SLO class (higher = more priority)
    pub fn priority_weight(&self) -> f32 {
        match self {
            SloClass::Latency => 1.0,
            SloClass::Throughput => 0.5,
        }
    }
}

impl std::str::FromStr for SloClass {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "latency" => Ok(SloClass::Latency),
            "throughput" => Ok(SloClass::Throughput),
            _ => Err(format!("Unknown SLO class: {}", s)),
        }
    }
}

impl fmt::Display for SloClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SloClass::Latency => write!(f, "latency"),
            SloClass::Throughput => write!(f, "throughput"),
        }
    }
}

/// Service endpoint information for discovery and routing
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    /// Name of the service
    pub service_name: String,
    
    /// Network address where the service is available
    pub address: SocketAddr,
    
    /// Labels associated with this service endpoint
    pub labels: Labels,
    
    /// Optional health check path
    pub health_check_path: Option<String>,
}

impl ServiceEndpoint {
    /// Create a new service endpoint
    pub fn new(
        service_name: impl Into<String>,
        address: SocketAddr,
        labels: Labels,
    ) -> Self {
        Self {
            service_name: service_name.into(),
            address,
            labels,
            health_check_path: None,
        }
    }
    
    /// Create a service endpoint with a health check path
    pub fn with_health_check(
        service_name: impl Into<String>,
        address: SocketAddr,
        labels: Labels,
        health_check_path: impl Into<String>,
    ) -> Self {
        Self {
            service_name: service_name.into(),
            address,
            labels,
            health_check_path: Some(health_check_path.into()),
        }
    }
    
    /// Get the base URL for this endpoint
    pub fn base_url(&self) -> String {
        format!("http://{}", self.address)
    }
    
    /// Get the health check URL if available
    pub fn health_check_url(&self) -> Option<String> {
        self.health_check_path.as_ref()
            .map(|path| format!("http://{}{}", self.address, path))
    }
}

impl fmt::Display for ServiceEndpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.service_name, self.address)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_creation() {
        let id1 = NodeId::new("test-node");
        assert_eq!(id1.as_str(), "test-node");

        let id2 = NodeId::generate();
        assert!(!id2.as_str().is_empty());
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_node_role_parsing() {
        assert_eq!("router".parse::<NodeRole>().unwrap(), NodeRole::Router);
        assert_eq!("gpu".parse::<NodeRole>().unwrap(), NodeRole::Gpu);
        assert_eq!("edge".parse::<NodeRole>().unwrap(), NodeRole::Edge);
        
        assert!("invalid".parse::<NodeRole>().is_err());
    }

    #[test]
    fn test_node_role_parse_multiple() {
        let roles = NodeRole::parse_roles("router,gpu").unwrap();
        assert_eq!(roles, vec![NodeRole::Router, NodeRole::Gpu]);

        let roles = NodeRole::parse_roles("gpu, edge").unwrap();
        assert_eq!(roles, vec![NodeRole::Gpu, NodeRole::Edge]);

        assert!(NodeRole::parse_roles("router,invalid").is_err());
    }

    #[test]
    fn test_slo_class_parsing() {
        assert_eq!("latency".parse::<SloClass>().unwrap(), SloClass::Latency);
        assert_eq!("throughput".parse::<SloClass>().unwrap(), SloClass::Throughput);
        
        assert!("invalid".parse::<SloClass>().is_err());
    }

    #[test]
    fn test_slo_class_priority() {
        assert!(SloClass::Latency.priority_weight() > SloClass::Throughput.priority_weight());
    }
}
