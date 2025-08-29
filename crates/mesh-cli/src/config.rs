//! Configuration management for mesh CLI

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// CLI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    /// Default control plane endpoint
    pub endpoint: String,
    
    /// Default timeout in seconds
    pub timeout: u64,
    
    /// Default output format
    pub output_format: String,
    
    /// Authentication configuration
    pub auth: AuthConfig,
    
    /// Named profiles
    pub profiles: HashMap<String, ProfileConfig>,
    
    /// Current active profile
    pub current_profile: Option<String>,
    
    /// Configuration source path
    #[serde(skip)]
    source: Option<PathBuf>,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Authentication method (none, token, mtls)
    pub method: String,
    
    /// API token (for token auth)
    pub token: Option<String>,
    
    /// Client certificate path (for mTLS)
    pub cert_path: Option<PathBuf>,
    
    /// Client private key path (for mTLS)
    pub key_path: Option<PathBuf>,
    
    /// CA certificate path (for mTLS)
    pub ca_path: Option<PathBuf>,
}

/// Profile configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileConfig {
    /// Profile name
    pub name: String,
    
    /// Control plane endpoint
    pub endpoint: String,
    
    /// Timeout in seconds
    pub timeout: Option<u64>,
    
    /// Output format
    pub output_format: Option<String>,
    
    /// Authentication configuration
    pub auth: Option<AuthConfig>,
    
    /// Profile-specific settings
    pub settings: HashMap<String, String>,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:50051".to_string(),
            timeout: 30,
            output_format: "table".to_string(),
            auth: AuthConfig::default(),
            profiles: HashMap::new(),
            current_profile: None,
            source: None,
        }
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            method: "none".to_string(),
            token: None,
            cert_path: None,
            key_path: None,
            ca_path: None,
        }
    }
}

impl CliConfig {
    /// Load configuration from file or create default
    pub fn load(config_path: Option<&Path>) -> Result<Self> {
        let config_path = match config_path {
            Some(path) => path.to_path_buf(),
            None => Self::default_config_path()?,
        };

        if config_path.exists() {
            Self::load_from_file(&config_path)
        } else {
            let mut config = Self::default();
            config.source = Some(config_path);
            Ok(config)
        }
    }

    /// Load configuration from a specific file
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        let mut config: Self = if path.extension().and_then(|s| s.to_str()) == Some("json") {
            serde_json::from_str(&content)
                .with_context(|| format!("Failed to parse JSON config: {}", path.display()))?
        } else {
            serde_yaml::from_str(&content)
                .with_context(|| format!("Failed to parse YAML config: {}", path.display()))?
        };

        config.source = Some(path.to_path_buf());
        Ok(config)
    }

    /// Save configuration to file
    #[allow(dead_code)]
    pub fn save(&self) -> Result<()> {
        let path = self.source.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No config file path specified"))?;

        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create config directory: {}", parent.display()))?;
        }

        let content = if path.extension().and_then(|s| s.to_str()) == Some("json") {
            serde_json::to_string_pretty(self)?
        } else {
            serde_yaml::to_string(self)?
        };

        std::fs::write(path, content)
            .with_context(|| format!("Failed to write config file: {}", path.display()))?;

        Ok(())
    }

    /// Get the default configuration file path
    pub fn default_config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine config directory"))?;
        
        Ok(config_dir.join("mesh").join("config.yaml"))
    }

    /// Get the configuration source path
    pub fn source(&self) -> Option<&Path> {
        self.source.as_deref()
    }

    /// Get the active profile configuration
    #[allow(dead_code)]
    pub fn active_profile(&self) -> Option<&ProfileConfig> {
        self.current_profile.as_ref()
            .and_then(|name| self.profiles.get(name))
    }

    /// Get effective endpoint (from active profile or default)
    #[allow(dead_code)]
    pub fn effective_endpoint(&self) -> &str {
        self.active_profile()
            .map(|p| p.endpoint.as_str())
            .unwrap_or(&self.endpoint)
    }

    /// Get effective timeout (from active profile or default)
    #[allow(dead_code)]
    pub fn effective_timeout(&self) -> u64 {
        self.active_profile()
            .and_then(|p| p.timeout)
            .unwrap_or(self.timeout)
    }

    /// Get effective output format (from active profile or default)
    #[allow(dead_code)]
    pub fn effective_output_format(&self) -> &str {
        self.active_profile()
            .and_then(|p| p.output_format.as_ref())
            .map(|s| s.as_str())
            .unwrap_or(&self.output_format)
    }

    /// Get effective auth config (from active profile or default)
    #[allow(dead_code)]
    pub fn effective_auth(&self) -> &AuthConfig {
        self.active_profile()
            .and_then(|p| p.auth.as_ref())
            .unwrap_or(&self.auth)
    }

    /// Add or update a profile
    #[allow(dead_code)]
    pub fn set_profile(&mut self, profile: ProfileConfig) {
        self.profiles.insert(profile.name.clone(), profile);
    }

    /// Remove a profile
    #[allow(dead_code)]
    pub fn remove_profile(&mut self, name: &str) -> Option<ProfileConfig> {
        // If removing the current profile, clear it
        if self.current_profile.as_ref() == Some(&name.to_string()) {
            self.current_profile = None;
        }
        self.profiles.remove(name)
    }

    /// Set the current active profile
    #[allow(dead_code)]
    pub fn set_current_profile(&mut self, name: Option<String>) -> Result<()> {
        if let Some(ref name) = name {
            if !self.profiles.contains_key(name) {
                return Err(anyhow::anyhow!("Profile '{}' does not exist", name));
            }
        }
        self.current_profile = name;
        Ok(())
    }

    /// List all profile names
    #[allow(dead_code)]
    pub fn profile_names(&self) -> Vec<&String> {
        self.profiles.keys().collect()
    }

    /// Get a configuration value by key
    #[allow(dead_code)]
    pub fn get(&self, key: &str) -> Option<String> {
        // Check active profile settings first
        if let Some(profile) = self.active_profile() {
            if let Some(value) = profile.settings.get(key) {
                return Some(value.clone());
            }
        }

        // Check global settings
        match key {
            "endpoint" => Some(self.endpoint.clone()),
            "timeout" => Some(self.timeout.to_string()),
            "output_format" => Some(self.output_format.clone()),
            "auth.method" => Some(self.auth.method.clone()),
            "auth.token" => self.auth.token.clone(),
            "current_profile" => self.current_profile.clone(),
            _ => None,
        }
    }

    /// Set a configuration value by key
    #[allow(dead_code)]
    pub fn set(&mut self, key: &str, value: String) -> Result<()> {
        match key {
            "endpoint" => self.endpoint = value,
            "timeout" => {
                self.timeout = value.parse()
                    .with_context(|| format!("Invalid timeout value: {}", value))?;
            }
            "output_format" => self.output_format = value,
            "auth.method" => self.auth.method = value,
            "auth.token" => self.auth.token = Some(value),
            "current_profile" => {
                let profile_name = if value.is_empty() { None } else { Some(value) };
                self.set_current_profile(profile_name)?;
            }
            _ => {
                // For unknown keys, store in current profile settings if available
                if let Some(profile_name) = &self.current_profile {
                    if let Some(profile) = self.profiles.get_mut(profile_name) {
                        profile.settings.insert(key.to_string(), value);
                    } else {
                        return Err(anyhow::anyhow!("Current profile '{}' not found", profile_name));
                    }
                } else {
                    return Err(anyhow::anyhow!("Unknown configuration key: {}", key));
                }
            }
        }
        Ok(())
    }
}

impl ProfileConfig {
    /// Create a new profile
    #[allow(dead_code)]
    pub fn new(name: String, endpoint: String) -> Self {
        Self {
            name,
            endpoint,
            timeout: None,
            output_format: None,
            auth: None,
            settings: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_default_config() {
        let config = CliConfig::default();
        assert_eq!(config.endpoint, "http://127.0.0.1:50051");
        assert_eq!(config.timeout, 30);
        assert_eq!(config.output_format, "table");
        assert_eq!(config.auth.method, "none");
    }

    #[test]
    fn test_config_serialization() {
        let config = CliConfig::default();
        
        // Test YAML serialization
        let yaml = serde_yaml::to_string(&config).unwrap();
        let deserialized: CliConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(config.endpoint, deserialized.endpoint);
        
        // Test JSON serialization
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: CliConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.endpoint, deserialized.endpoint);
    }

    #[test]
    fn test_profile_management() {
        let mut config = CliConfig::default();
        
        // Add a profile
        let profile = ProfileConfig::new("test".to_string(), "http://test:8080".to_string());
        config.set_profile(profile);
        
        assert!(config.profiles.contains_key("test"));
        assert_eq!(config.profile_names().len(), 1);
        
        // Set current profile
        config.set_current_profile(Some("test".to_string())).unwrap();
        assert_eq!(config.current_profile, Some("test".to_string()));
        assert_eq!(config.effective_endpoint(), "http://test:8080");
        
        // Remove profile
        let removed = config.remove_profile("test");
        assert!(removed.is_some());
        assert!(config.current_profile.is_none());
    }

    #[test]
    fn test_config_get_set() {
        let mut config = CliConfig::default();
        
        // Test getting default values
        assert_eq!(config.get("endpoint"), Some("http://127.0.0.1:50051".to_string()));
        assert_eq!(config.get("timeout"), Some("30".to_string()));
        
        // Test setting values
        config.set("endpoint", "http://new:9090".to_string()).unwrap();
        assert_eq!(config.endpoint, "http://new:9090");
        
        config.set("timeout", "60".to_string()).unwrap();
        assert_eq!(config.timeout, 60);
        
        // Test invalid timeout
        assert!(config.set("timeout", "invalid".to_string()).is_err());
    }

    #[test]
    fn test_config_file_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test_config.yaml");
        
        let mut config = CliConfig::default();
        config.endpoint = "http://test:8080".to_string();
        config.source = Some(config_path.clone());
        
        // Save config
        config.save().unwrap();
        assert!(config_path.exists());
        
        // Load config
        let loaded_config = CliConfig::load_from_file(&config_path).unwrap();
        assert_eq!(loaded_config.endpoint, "http://test:8080");
    }

    #[test]
    fn test_effective_values() {
        let mut config = CliConfig::default();
        
        // Test default effective values
        assert_eq!(config.effective_endpoint(), "http://127.0.0.1:50051");
        assert_eq!(config.effective_timeout(), 30);
        
        // Add profile with overrides
        let mut profile = ProfileConfig::new("test".to_string(), "http://profile:8080".to_string());
        profile.timeout = Some(60);
        profile.output_format = Some("json".to_string());
        config.set_profile(profile);
        
        // Set as current profile
        config.set_current_profile(Some("test".to_string())).unwrap();
        
        // Test profile effective values
        assert_eq!(config.effective_endpoint(), "http://profile:8080");
        assert_eq!(config.effective_timeout(), 60);
        assert_eq!(config.effective_output_format(), "json");
    }
}
