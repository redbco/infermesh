//! Output formatting for mesh CLI

use anyhow::Result;
use clap::ValueEnum;
use colored::*;
use comfy_table::{Table, Cell, Color, Attribute, ContentArrangement, presets::UTF8_FULL};
use serde::Serialize;
use serde_json;
use std::collections::HashMap;

/// Output format options
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable table format
    Table,
    /// JSON format
    Json,
    /// YAML format
    Yaml,
    /// Compact text format
    Text,
}

impl Default for OutputFormat {
    fn default() -> Self {
        OutputFormat::Table
    }
}

/// Trait for types that can be formatted for output
pub trait Formattable {
    /// Format as a table row
    fn table_headers() -> Vec<String>;
    fn table_row(&self) -> Vec<String>;
    
    /// Format as key-value pairs for detailed view
    fn key_value_pairs(&self) -> Vec<(String, String)>;
}

/// Output formatter
pub struct OutputFormatter {
    format: OutputFormat,
}

impl OutputFormatter {
    /// Create a new output formatter
    pub fn new(format: OutputFormat) -> Self {
        Self { format }
    }

    /// Format and print a single item
    pub fn print_item<T>(&self, item: &T) -> Result<()>
    where
        T: Serialize + Formattable,
    {
        match self.format {
            OutputFormat::Json => {
                let json = serde_json::to_string_pretty(item)?;
                println!("{}", json);
            }
            OutputFormat::Yaml => {
                let yaml = serde_yaml::to_string(item)?;
                println!("{}", yaml);
            }
            OutputFormat::Table | OutputFormat::Text => {
                // For single items, show as key-value pairs
                let pairs = item.key_value_pairs();
                for (key, value) in pairs {
                    match self.format {
                        OutputFormat::Table => {
                            println!("{}: {}", key.bold().cyan(), value);
                        }
                        OutputFormat::Text => {
                            println!("{}: {}", key, value);
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }
        Ok(())
    }

    /// Format and print a list of items
    pub fn print_list<T>(&self, items: &[T]) -> Result<()>
    where
        T: Serialize + Formattable,
    {
        if items.is_empty() {
            match self.format {
                OutputFormat::Json => println!("[]"),
                OutputFormat::Yaml => println!("[]"),
                OutputFormat::Table | OutputFormat::Text => {
                    println!("{}", "No items found".dimmed());
                }
            }
            return Ok(());
        }

        match self.format {
            OutputFormat::Json => {
                let json = serde_json::to_string_pretty(items)?;
                println!("{}", json);
            }
            OutputFormat::Yaml => {
                let yaml = serde_yaml::to_string(items)?;
                println!("{}", yaml);
            }
            OutputFormat::Table => {
                self.print_table(items)?;
            }
            OutputFormat::Text => {
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        println!();
                    }
                    let pairs = item.key_value_pairs();
                    for (key, value) in pairs {
                        println!("{}: {}", key, value);
                    }
                }
            }
        }
        Ok(())
    }

    /// Print items as a table
    fn print_table<T>(&self, items: &[T]) -> Result<()>
    where
        T: Formattable,
    {
        if items.is_empty() {
            return Ok(());
        }

        let mut table = Table::new();
        table.load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic);

        // Add headers
        let headers = T::table_headers();
        let header_cells: Vec<Cell> = headers.iter()
            .map(|h| Cell::new(h).add_attribute(Attribute::Bold).fg(Color::Cyan))
            .collect();
        table.set_header(header_cells);

        // Add rows
        for item in items {
            let row = item.table_row();
            table.add_row(row);
        }

        println!("{}", table);
        Ok(())
    }

    /// Print a success message
    pub fn print_success(&self, message: &str) -> Result<()> {
        match self.format {
            OutputFormat::Json => {
                let result = serde_json::json!({
                    "status": "success",
                    "message": message
                });
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            OutputFormat::Yaml => {
                println!("status: success");
                println!("message: {}", message);
            }
            OutputFormat::Table | OutputFormat::Text => {
                println!("{} {}", "✓".green().bold(), message.green());
            }
        }
        Ok(())
    }

    /// Print an error message
    #[allow(dead_code)]
    pub fn print_error(&self, message: &str) -> Result<()> {
        match self.format {
            OutputFormat::Json => {
                let result = serde_json::json!({
                    "status": "error",
                    "message": message
                });
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            OutputFormat::Yaml => {
                println!("status: error");
                println!("message: {}", message);
            }
            OutputFormat::Table | OutputFormat::Text => {
                eprintln!("{} {}", "✗".red().bold(), message.red());
            }
        }
        Ok(())
    }

    /// Print a warning message
    #[allow(dead_code)]
    pub fn print_warning(&self, message: &str) -> Result<()> {
        match self.format {
            OutputFormat::Json => {
                let result = serde_json::json!({
                    "status": "warning",
                    "message": message
                });
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            OutputFormat::Yaml => {
                println!("status: warning");
                println!("message: {}", message);
            }
            OutputFormat::Table | OutputFormat::Text => {
                eprintln!("{} {}", "⚠".yellow().bold(), message.yellow());
            }
        }
        Ok(())
    }

    /// Print an info message
    pub fn print_info(&self, message: &str) -> Result<()> {
        match self.format {
            OutputFormat::Json => {
                let result = serde_json::json!({
                    "status": "info",
                    "message": message
                });
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            OutputFormat::Yaml => {
                println!("status: info");
                println!("message: {}", message);
            }
            OutputFormat::Table | OutputFormat::Text => {
                println!("{} {}", "ℹ".blue().bold(), message.blue());
            }
        }
        Ok(())
    }

    /// Print statistics
    pub fn print_stats(&self, stats: &HashMap<String, String>) -> Result<()> {
        match self.format {
            OutputFormat::Json => {
                let json = serde_json::to_string_pretty(stats)?;
                println!("{}", json);
            }
            OutputFormat::Yaml => {
                let yaml = serde_yaml::to_string(stats)?;
                println!("{}", yaml);
            }
            OutputFormat::Table => {
                let mut table = Table::new();
                table.load_preset(UTF8_FULL)
                    .set_content_arrangement(ContentArrangement::Dynamic);

                table.set_header(vec![
                    Cell::new("Metric").add_attribute(Attribute::Bold).fg(Color::Cyan),
                    Cell::new("Value").add_attribute(Attribute::Bold).fg(Color::Cyan),
                ]);

                for (key, value) in stats {
                    table.add_row(vec![key, value]);
                }

                println!("{}", table);
            }
            OutputFormat::Text => {
                for (key, value) in stats {
                    println!("{}: {}", key, value);
                }
            }
        }
        Ok(())
    }

    /// Print a progress message (only for interactive formats)
    pub fn print_progress(&self, message: &str) {
        match self.format {
            OutputFormat::Table | OutputFormat::Text => {
                eprint!("{} {}...\r", "⏳".yellow(), message);
            }
            _ => {
                // Don't print progress for structured formats
            }
        }
    }

    /// Clear progress message (only for interactive formats)
    pub fn clear_progress(&self) {
        match self.format {
            OutputFormat::Table | OutputFormat::Text => {
                eprint!("\r{}\r", " ".repeat(80));
            }
            _ => {
                // Don't clear progress for structured formats
            }
        }
    }
}

/// Helper function to format duration
pub fn format_duration(seconds: u64) -> String {
    if seconds < 60 {
        format!("{}s", seconds)
    } else if seconds < 3600 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else if seconds < 86400 {
        let hours = seconds / 3600;
        let minutes = (seconds % 3600) / 60;
        format!("{}h {}m", hours, minutes)
    } else {
        let days = seconds / 86400;
        let hours = (seconds % 86400) / 3600;
        format!("{}d {}h", days, hours)
    }
}

/// Helper function to format bytes
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

/// Helper function to format percentage
#[allow(dead_code)]
pub fn format_percentage(value: f64) -> String {
    format!("{:.1}%", value * 100.0)
}

/// Helper function to colorize status
#[allow(dead_code)]
pub fn colorize_status(status: &str) -> ColoredString {
    match status.to_lowercase().as_str() {
        "healthy" | "ready" | "active" | "running" | "online" | "up" => status.green(),
        "unhealthy" | "failed" | "error" | "down" | "offline" => status.red(),
        "warning" | "degraded" | "loading" | "starting" => status.yellow(),
        "unknown" | "pending" | "inactive" => status.dimmed(),
        _ => status.normal(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[derive(Serialize)]
    struct TestItem {
        name: String,
        value: i32,
        status: String,
    }

    impl Formattable for TestItem {
        fn table_headers() -> Vec<String> {
            vec!["Name".to_string(), "Value".to_string(), "Status".to_string()]
        }

        fn table_row(&self) -> Vec<String> {
            vec![self.name.clone(), self.value.to_string(), self.status.clone()]
        }

        fn key_value_pairs(&self) -> Vec<(String, String)> {
            vec![
                ("Name".to_string(), self.name.clone()),
                ("Value".to_string(), self.value.to_string()),
                ("Status".to_string(), self.status.clone()),
            ]
        }
    }

    #[test]
    fn test_output_format_enum() {
        assert_eq!(OutputFormat::default(), OutputFormat::Table);
        
        // Test that all variants can be created
        let _table = OutputFormat::Table;
        let _json = OutputFormat::Json;
        let _yaml = OutputFormat::Yaml;
        let _text = OutputFormat::Text;
    }

    #[test]
    fn test_formatter_creation() {
        let formatter = OutputFormatter::new(OutputFormat::Json);
        assert_eq!(formatter.format, OutputFormat::Json);
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30), "30s");
        assert_eq!(format_duration(90), "1m 30s");
        assert_eq!(format_duration(3661), "1h 1m");
        assert_eq!(format_duration(90061), "1d 1h");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1048576), "1.0 MB");
        assert_eq!(format_bytes(1073741824), "1.0 GB");
    }

    #[test]
    fn test_format_percentage() {
        assert_eq!(format_percentage(0.0), "0.0%");
        assert_eq!(format_percentage(0.5), "50.0%");
        assert_eq!(format_percentage(1.0), "100.0%");
        assert_eq!(format_percentage(0.123), "12.3%");
    }

    #[test]
    fn test_colorize_status() {
        // Test that the function doesn't panic and returns a ColoredString
        let healthy = colorize_status("healthy");
        let failed = colorize_status("failed");
        let warning = colorize_status("warning");
        let unknown = colorize_status("unknown");
        
        // We can't easily test the actual colors, but we can test that the function works
        assert!(!healthy.to_string().is_empty());
        assert!(!failed.to_string().is_empty());
        assert!(!warning.to_string().is_empty());
        assert!(!unknown.to_string().is_empty());
    }

    #[test]
    fn test_formattable_trait() {
        let item = TestItem {
            name: "test".to_string(),
            value: 42,
            status: "healthy".to_string(),
        };

        let headers = TestItem::table_headers();
        assert_eq!(headers, vec!["Name", "Value", "Status"]);

        let row = item.table_row();
        assert_eq!(row, vec!["test", "42", "healthy"]);

        let pairs = item.key_value_pairs();
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0], ("Name".to_string(), "test".to_string()));
    }
}
