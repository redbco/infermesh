//! OpenTelemetry metrics integration (optional feature)

#[cfg(feature = "opentelemetry")]
use crate::{MetricsError, Result};
#[cfg(feature = "opentelemetry")]
use opentelemetry::{
    metrics::{MeterProvider, Unit},
    KeyValue,
};
#[cfg(feature = "opentelemetry")]
use opentelemetry_otlp::WithExportConfig;
#[cfg(feature = "opentelemetry")]
use std::time::Duration;
#[cfg(feature = "opentelemetry")]
use tracing::{info, warn};

/// OpenTelemetry metrics exporter
#[cfg(feature = "opentelemetry")]
#[derive(Debug)]
pub struct OpenTelemetryExporter {
    endpoint: String,
    meter_provider: Option<Box<dyn MeterProvider>>,
    started: bool,
}

#[cfg(feature = "opentelemetry")]
impl OpenTelemetryExporter {
    /// Create a new OpenTelemetry exporter
    pub fn new(endpoint: &str) -> Result<Self> {
        Ok(Self {
            endpoint: endpoint.to_string(),
            meter_provider: None,
            started: false,
        })
    }

    /// Start the OpenTelemetry exporter
    pub async fn start(&mut self) -> Result<()> {
        if self.started {
            return Err(MetricsError::Config("Exporter already started".to_string()));
        }

        info!("Starting OpenTelemetry metrics exporter to {}", self.endpoint);

        // Initialize the OTLP metrics exporter
        let exporter = opentelemetry_otlp::new_exporter()
            .http()
            .with_endpoint(&self.endpoint)
            .with_timeout(Duration::from_secs(10));

        // Create meter provider
        let meter_provider = opentelemetry_otlp::new_pipeline()
            .metrics(opentelemetry::runtime::Tokio)
            .with_exporter(exporter)
            .with_period(Duration::from_secs(5))
            .with_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| MetricsError::OpenTelemetry(e))?;

        self.meter_provider = Some(Box::new(meter_provider));
        self.started = true;

        info!("OpenTelemetry metrics exporter started");
        Ok(())
    }

    /// Stop the OpenTelemetry exporter
    pub async fn stop(&mut self) {
        if !self.started {
            return;
        }

        info!("Stopping OpenTelemetry metrics exporter");

        if let Some(provider) = self.meter_provider.take() {
            // Shutdown the meter provider
            if let Err(e) = provider.shutdown() {
                warn!("Error shutting down OpenTelemetry meter provider: {:?}", e);
            }
        }

        self.started = false;
        info!("OpenTelemetry metrics exporter stopped");
    }

    /// Export metrics to OpenTelemetry
    pub async fn export_metrics(&self) -> Result<()> {
        if !self.started {
            return Err(MetricsError::Config("Exporter not started".to_string()));
        }

        // In a real implementation, we would collect metrics from the registry
        // and export them to OpenTelemetry. For now, this is a placeholder.
        
        Ok(())
    }

    /// Check if the exporter is started
    pub fn is_started(&self) -> bool {
        self.started
    }

    /// Get the endpoint URL
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

#[cfg(feature = "opentelemetry")]
impl Drop for OpenTelemetryExporter {
    fn drop(&mut self) {
        if self.started {
            // Note: We can't call async stop() in Drop, so we just clean up what we can
            if let Some(provider) = self.meter_provider.take() {
                let _ = provider.shutdown();
            }
        }
    }
}

// Stub implementation when OpenTelemetry feature is not enabled
#[cfg(not(feature = "opentelemetry"))]
pub struct OpenTelemetryExporter;

#[cfg(not(feature = "opentelemetry"))]
impl OpenTelemetryExporter {
    pub fn new(_endpoint: &str) -> Result<Self> {
        Err(MetricsError::Config(
            "OpenTelemetry feature not enabled".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "opentelemetry")]
    #[tokio::test]
    async fn test_opentelemetry_exporter_creation() {
        let exporter = OpenTelemetryExporter::new("http://localhost:4317").unwrap();
        assert!(!exporter.is_started());
        assert_eq!(exporter.endpoint(), "http://localhost:4317");
    }

    #[cfg(feature = "opentelemetry")]
    #[tokio::test]
    async fn test_opentelemetry_lifecycle() {
        let mut exporter = OpenTelemetryExporter::new("http://localhost:4317").unwrap();
        
        // Note: This test will fail if there's no OTLP endpoint running
        // In a real test environment, we'd use a mock OTLP server
        // For now, we just test that the methods don't panic
        
        assert!(!exporter.is_started());
        
        // We can't actually start without a real OTLP endpoint
        // let result = exporter.start().await;
        // This would likely fail in CI, so we skip the actual start
        
        exporter.stop().await;
        assert!(!exporter.is_started());
    }

    #[cfg(not(feature = "opentelemetry"))]
    #[test]
    fn test_opentelemetry_disabled() {
        let result = OpenTelemetryExporter::new("http://localhost:4317");
        assert!(result.is_err());
    }
}
