//! Node management commands

use crate::client::MeshClient;
use crate::output::{OutputFormat, OutputFormatter, Formattable, format_duration, format_bytes};
use anyhow::Result;
use serde::Serialize;

/// Node information for display
#[derive(Debug, Serialize)]
pub struct NodeInfo {
    pub id: String,
    pub role: String,
    pub status: String,
    pub address: String,
    pub uptime: String,
    pub cpu_usage: String,
    pub memory_usage: String,
    pub gpu_count: u32,
    pub model_count: u32,
}

impl Formattable for NodeInfo {
    fn table_headers() -> Vec<String> {
        vec![
            "ID".to_string(),
            "Role".to_string(),
            "Status".to_string(),
            "Address".to_string(),
            "Uptime".to_string(),
            "CPU".to_string(),
            "Memory".to_string(),
            "GPUs".to_string(),
            "Models".to_string(),
        ]
    }

    fn table_row(&self) -> Vec<String> {
        vec![
            self.id.clone(),
            self.role.clone(),
            self.status.clone(),
            self.address.clone(),
            self.uptime.clone(),
            self.cpu_usage.clone(),
            self.memory_usage.clone(),
            self.gpu_count.to_string(),
            self.model_count.to_string(),
        ]
    }

    fn key_value_pairs(&self) -> Vec<(String, String)> {
        vec![
            ("ID".to_string(), self.id.clone()),
            ("Role".to_string(), self.role.clone()),
            ("Status".to_string(), self.status.clone()),
            ("Address".to_string(), self.address.clone()),
            ("Uptime".to_string(), self.uptime.clone()),
            ("CPU Usage".to_string(), self.cpu_usage.clone()),
            ("Memory Usage".to_string(), self.memory_usage.clone()),
            ("GPU Count".to_string(), self.gpu_count.to_string()),
            ("Model Count".to_string(), self.model_count.to_string()),
        ]
    }
}

/// List nodes in the mesh
pub async fn list_nodes(
    _client: &MeshClient,
    role_filter: Option<String>,
    status_filter: Option<String>,
    detailed: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let formatter = OutputFormatter::new(output_format);
    
    formatter.print_progress("Fetching nodes");
    
    // For now, create mock data since we don't have a real server
    let nodes = create_mock_nodes(role_filter, status_filter);
    
    formatter.clear_progress();
    
    if detailed {
        for node in &nodes {
            formatter.print_item(node)?;
            if matches!(output_format, OutputFormat::Table | OutputFormat::Text) {
                println!();
            }
        }
    } else {
        formatter.print_list(&nodes)?;
    }
    
    Ok(())
}

/// Describe a specific node
pub async fn describe_node(
    _client: &MeshClient,
    node_id: String,
    output_format: OutputFormat,
) -> Result<()> {
    let formatter = OutputFormatter::new(output_format);
    
    formatter.print_progress(&format!("Fetching details for node {}", node_id));
    
    // For now, create mock data
    let node = create_mock_node(&node_id);
    
    formatter.clear_progress();
    formatter.print_item(&node)?;
    
    Ok(())
}

/// Create mock nodes for demonstration
fn create_mock_nodes(role_filter: Option<String>, status_filter: Option<String>) -> Vec<NodeInfo> {
    let mut nodes = vec![
        NodeInfo {
            id: "node-1".to_string(),
            role: "gpu".to_string(),
            status: "healthy".to_string(),
            address: "192.168.1.10:50051".to_string(),
            uptime: format_duration(3600 * 24 + 1800), // 1 day, 30 minutes
            cpu_usage: "45.2%".to_string(),
            memory_usage: format!("{} / {}", format_bytes(8 * 1024 * 1024 * 1024), format_bytes(16 * 1024 * 1024 * 1024)),
            gpu_count: 2,
            model_count: 3,
        },
        NodeInfo {
            id: "node-2".to_string(),
            role: "router".to_string(),
            status: "healthy".to_string(),
            address: "192.168.1.11:50051".to_string(),
            uptime: format_duration(3600 * 12 + 900), // 12 hours, 15 minutes
            cpu_usage: "12.8%".to_string(),
            memory_usage: format!("{} / {}", format_bytes(2 * 1024 * 1024 * 1024), format_bytes(8 * 1024 * 1024 * 1024)),
            gpu_count: 0,
            model_count: 0,
        },
        NodeInfo {
            id: "node-3".to_string(),
            role: "gpu".to_string(),
            status: "warning".to_string(),
            address: "192.168.1.12:50051".to_string(),
            uptime: format_duration(3600 * 6), // 6 hours
            cpu_usage: "78.5%".to_string(),
            memory_usage: format!("{} / {}", format_bytes(14 * 1024 * 1024 * 1024), format_bytes(16 * 1024 * 1024 * 1024)),
            gpu_count: 1,
            model_count: 2,
        },
    ];

    // Apply filters
    if let Some(role) = role_filter {
        nodes.retain(|n| n.role == role);
    }
    
    if let Some(status) = status_filter {
        nodes.retain(|n| n.status == status);
    }

    nodes
}

/// Create a mock node for demonstration
fn create_mock_node(node_id: &str) -> NodeInfo {
    NodeInfo {
        id: node_id.to_string(),
        role: "gpu".to_string(),
        status: "healthy".to_string(),
        address: "192.168.1.10:50051".to_string(),
        uptime: format_duration(3600 * 24 + 1800),
        cpu_usage: "45.2%".to_string(),
        memory_usage: format!("{} / {}", format_bytes(8 * 1024 * 1024 * 1024), format_bytes(16 * 1024 * 1024 * 1024)),
        gpu_count: 2,
        model_count: 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_info_formattable() {
        let node = NodeInfo {
            id: "test-node".to_string(),
            role: "gpu".to_string(),
            status: "healthy".to_string(),
            address: "127.0.0.1:50051".to_string(),
            uptime: "1h 30m".to_string(),
            cpu_usage: "50.0%".to_string(),
            memory_usage: "8.0 GB / 16.0 GB".to_string(),
            gpu_count: 2,
            model_count: 3,
        };

        let headers = NodeInfo::table_headers();
        assert_eq!(headers.len(), 9);
        assert!(headers.contains(&"ID".to_string()));

        let row = node.table_row();
        assert_eq!(row.len(), 9);
        assert_eq!(row[0], "test-node");

        let pairs = node.key_value_pairs();
        assert_eq!(pairs.len(), 9);
        assert_eq!(pairs[0].0, "ID");
        assert_eq!(pairs[0].1, "test-node");
    }

    #[test]
    fn test_mock_nodes_creation() {
        let nodes = create_mock_nodes(None, None);
        assert_eq!(nodes.len(), 3);

        let gpu_nodes = create_mock_nodes(Some("gpu".to_string()), None);
        assert_eq!(gpu_nodes.len(), 2);

        let healthy_nodes = create_mock_nodes(None, Some("healthy".to_string()));
        assert_eq!(healthy_nodes.len(), 2);

        let gpu_healthy_nodes = create_mock_nodes(Some("gpu".to_string()), Some("healthy".to_string()));
        assert_eq!(gpu_healthy_nodes.len(), 1);
    }

    #[test]
    fn test_mock_node_creation() {
        let node = create_mock_node("test-123");
        assert_eq!(node.id, "test-123");
        assert_eq!(node.role, "gpu");
        assert_eq!(node.status, "healthy");
    }
}
